import os
import sys
import uuid
import random
import json
import numpy as np
import pandas as pd

# Set matplotlib backend to headless Agg to avoid DISPLAY errors
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances, silhouette_score
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from scipy.spatial.distance import pdist, squareform
from fastapi.testclient import TestClient

# Setup python path imports
sys.path.append(os.path.abspath("."))

from src.main import app
from src.ingestion.router import ingest_from_file
from src.api.v1.endpoints import confirm_columns, ConfirmColumnsPayload, ColumnConfirmation, run_layer0, name_cluster, NameClusterPayload
from src.layer1.field_classifier import confirm_field_classifications
from src.db.connection import get_raw_connection

# Downstream calculation modules
from src.layer1.mom_tracker import compute_and_persist_mom_metrics
from src.layer1.anomaly_flagger import flag_anomalies_for_period
from src.layer4.retrain_trigger import trigger_retraining
from src.layer5.migration_forecaster import compute_and_persist_transition_matrix
from src.layer6.outcome_analyzer import compute_and_persist_segment_outcomes
from src.layer3.statistical_tester import run_statistical_tests
from src.layer8.migration_signal import confirm_layer8_columns, compute_and_persist_feature_migration_signals
from src.agents.insight_narrator import generate_and_evaluate_narration

BASE_DIR = "/app" if os.path.exists("/app") else r"C:\Projects\Segment studio"

def hopkins_statistic(X, sample_ratio=0.1, random_state=42):
    n = len(X)
    n_samples = max(int(n * sample_ratio), 1)
    np.random.seed(random_state)
    
    nbrs = NearestNeighbors(n_neighbors=2, metric='euclidean').fit(X)
    
    indices = np.random.choice(n, size=n_samples, replace=False)
    X_sample = X[indices]
    distances_w, _ = nbrs.kneighbors(X_sample, n_neighbors=2)
    w = distances_w[:, 1]
    
    mins = X.min(axis=0)
    maxs = X.max(axis=0)
    X_rand = np.random.uniform(low=mins, high=maxs, size=(n_samples, X.shape[1]))
    
    distances_u, _ = nbrs.kneighbors(X_rand, n_neighbors=1)
    u = distances_u[:, 0]
    
    sum_u = np.sum(u)
    sum_w = np.sum(w)
    
    if (sum_u + sum_w) == 0:
        return 0.5
    
    return sum_u / (sum_u + sum_w)

def compute_vat(X, sample_size=2000, random_state=42):
    n = len(X)
    if n > sample_size:
        np.random.seed(random_state)
        indices = np.random.choice(n, size=sample_size, replace=False)
        X_sample = X[indices]
    else:
        X_sample = X
        
    D = squareform(pdist(X_sample, metric='euclidean'))
    num_points = X_sample.shape[0]
    
    P = np.zeros(num_points, dtype=int)
    V = list(range(num_points))
    
    i, j = np.unravel_index(np.argmax(D), D.shape)
    P[0] = i
    V.remove(i)
    
    for r in range(1, num_points):
        sub_D = D[np.ix_(P[:r], V)]
        p_idx, v_idx = np.unravel_index(np.argmin(sub_D), sub_D.shape)
        P[r] = V[v_idx]
        V.pop(v_idx)
        
    D_ordered = D[np.ix_(P, P)]
    return D_ordered

def find_elbow_point(cumulative_variance):
    n = len(cumulative_variance)
    if n <= 2:
        return n
    x = np.arange(n)
    y = np.array(cumulative_variance)
    
    line_start = np.array([x[0], y[0]])
    line_end = np.array([x[-1], y[-1]])
    line_vec = line_end - line_start
    line_vec_norm = line_vec / np.sqrt(np.sum(line_vec**2))
    
    distances = []
    for i in range(n):
        point = np.array([x[i], y[i]])
        proj = line_start + np.dot(point - line_start, line_vec_norm) * line_vec_norm
        dist = np.sqrt(np.sum((point - proj)**2))
        distances.append(dist)
        
    return np.argmax(distances) + 1

def run_tr_pipeline():
    print("Step 1: Loading raw Turkey dataset and aggregating transaction-level data to customer-level...")
    raw_path = os.path.join(BASE_DIR, "test datasets", "ecommerce_customer_behavior_dataset_v2.csv")
    if not os.path.exists(raw_path):
        print(f"Error: Base dataset not found at {raw_path}")
        sys.exit(1)
        
    df_raw = pd.read_csv(raw_path)
    print(f"Loaded {len(df_raw)} rows and {len(df_raw.columns)} columns.")

    agg_dict = {
        "Total_Amount": "sum",
        "Order_ID": "count",
        "Session_Duration_Minutes": "mean",
        "Pages_Viewed": "mean",
        "Customer_Rating": "mean",
        "Discount_Amount": "sum",
        "Delivery_Time_Days": "mean",
        "Is_Returning_Customer": "max",
        "Quantity": "mean",
        "Age": "first",
        "Gender": "first",
        "City": "first"
    }

    df_agg = df_raw.groupby("Customer_ID").agg(agg_dict).reset_index()

    # Rename columns to match target names
    df_agg.rename(columns={
        "Total_Amount": "Lifetime_Value",
        "Order_ID": "Total_Purchases",
        "Session_Duration_Minutes": "Session_Duration_Avg",
        "Pages_Viewed": "Pages_Per_Session",
        "Customer_Rating": "Customer_Rating_Avg",
        "Discount_Amount": "Discount_Total",
        "Delivery_Time_Days": "Delivery_Time_Avg",
        "Is_Returning_Customer": "Returned_Ever",
        "Quantity": "Quantity_Avg"
    }, inplace=True)

    # Convert Customer_ID to a deterministic UUID to satisfy customers table foreign keys
    df_agg['customer_id'] = df_agg['Customer_ID'].apply(
        lambda cid: str(uuid.uuid5(uuid.NAMESPACE_DNS, str(cid)))
    )

    # Churned = 1 - Returned_Ever
    df_agg['Churned'] = (1 - df_agg['Returned_Ever'].astype(int))

    # Cast numeric columns to float to avoid pandas 2.0+ int-to-float dtype mutation errors
    numeric_cols = [
        'Lifetime_Value', 'Total_Purchases', 'Session_Duration_Avg', 
        'Pages_Per_Session', 'Customer_Rating_Avg', 'Discount_Total', 
        'Delivery_Time_Avg', 'Quantity_Avg', 'Age', 'Returned_Ever', 'Churned'
    ]
    for col in numeric_cols:
        df_agg[col] = df_agg[col].astype(float)

    # Drop old Customer_ID key to avoid confusing the pipeline
    df_agg.drop(columns=['Customer_ID'], inplace=True)

    print(f"Aggregated structure: {df_agg.shape[0]} rows, {df_agg.shape[1]} columns.")

    # Save to temp path for ingestion router
    temp_path = os.path.join(BASE_DIR, "temp_to_ingest_tr.csv")
    df_agg.to_csv(temp_path, index=False)

    print("\nStep 2: Ingesting aggregated dataset via router & running Layer 0...")
    dataset_id = ingest_from_file(temp_path, "ecommerce_customer_behavior_dataset_tr_base.csv")
    print(f"Successfully ingested dataset. dataset_id={dataset_id}")
    
    if os.path.exists(temp_path):
        os.remove(temp_path)

    print("Confirming all columns in Layer 0...")
    confirm_payload = ConfirmColumnsPayload(
        columns=[ColumnConfirmation(column_name=col, include=True) for col in df_agg.columns]
    )
    confirm_columns(dataset_id, confirm_payload)

    print("Setting imputation methods to 'median' or 'mode' in DB...")
    with get_raw_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT column_name, dtype FROM column_profiles WHERE dataset_id = %s AND missing_count > 0;",
                (dataset_id,)
            )
            missing_rows = cur.fetchall()
            for col_name, dtype in missing_rows:
                method = "median" if "int" in dtype.lower() or "float" in dtype.lower() or "numeric" in dtype.lower() else "mode"
                cur.execute(
                    """
                    UPDATE column_profiles
                    SET imputation_method = %s
                    WHERE dataset_id = %s AND column_name = %s;
                    """,
                    (method, dataset_id, col_name)
                )
                print(f"  Column '{col_name}': Imputation method set to '{method}'")

    print("Running full Layer 0 Pipeline on aggregated dataset...")
    layer0_res = run_layer0(dataset_id)

    # Verify layer0_completed in database
    with get_raw_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT layer0_completed FROM datasets WHERE dataset_id = %s;", (dataset_id,))
            layer0_status = cur.fetchone()[0]
            
    if not layer0_status:
        print("Error: Layer 0 did not complete successfully. Halt execution.")
        sys.exit(1)
    print("Layer 0 pipeline completed and verified in DB.")
    
    # Report Step 2 details
    rows_retained = layer0_res.get("processed_rows")
    columns_processed = layer0_res.get("processed_columns")
    print(f"Report Layer 0: {rows_retained} rows retained.")

    # Define features and classifications
    behavioral_features = [
        'Lifetime_Value', 'Total_Purchases', 'Session_Duration_Avg', 
        'Pages_Per_Session', 'Customer_Rating_Avg', 'Discount_Total', 
        'Delivery_Time_Avg', 'Returned_Ever', 'Quantity_Avg'
    ]
    
    classifications = []
    for col in df_agg.columns:
        if col == "customer_id":
            continue
        if col in behavioral_features:
            classifications.append({"column": col, "classification": "behavioral"})
        else:
            classifications.append({"column": col, "classification": "demographic"})

    print("\nConfirming classifications and dynamically altering DB schema...")
    confirm_field_classifications(dataset_id, classifications)

    # Load processed data
    processed_path = os.path.join("data", f"{dataset_id}_processed.csv")
    df_processed = pd.read_csv(processed_path)

    print("\nStep 3: Running clustering diagnostics on Layer 0 output...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_processed[behavioral_features])

    pca_temp = PCA(random_state=42)
    pca_temp.fit(X_scaled)
    cumulative_variance = np.cumsum(pca_temp.explained_variance_ratio_)
    
    elbow_components = find_elbow_point(cumulative_variance)
    print(f"  PCA Explained Variance Ratio: {pca_temp.explained_variance_ratio_}")
    print(f"  PCA Cumulative Variance: {cumulative_variance}")
    print(f"  Identified PCA Elbow Point: PC{elbow_components}")

    pca = PCA(n_components=elbow_components, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    # Compute diagnostics
    hopkins_val = hopkins_statistic(X_pca, sample_ratio=0.1, random_state=42)
    print(f"  Hopkins Statistic (10% sample): {hopkins_val:.4f}")

    # Generate VAT plot (2,000 sample)
    os.makedirs("diagnostics", exist_ok=True)
    vat_matrix = compute_vat(X_pca, sample_size=2000, random_state=42)
    plt.figure(figsize=(8, 8))
    plt.imshow(vat_matrix, cmap='gray', vmin=0, vmax=np.percentile(vat_matrix, 95))
    plt.title("VAT Plot - Turkey Dataset")
    plt.colorbar(label='Dissimilarity')
    vat_path = "diagnostics/vat_turkey.png"
    plt.savefig(vat_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"  Saved VAT plot to: {vat_path}")

    # Silhouette Scores for K=3, 4, 5, 6
    silhouette_scores = {}
    for k in [3, 4, 5, 6]:
        km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_pca)
        sil = float(silhouette_score(X_pca, km.labels_, random_state=42))
        silhouette_scores[k] = sil
        print(f"  Silhouette Score (K={k}): {sil:.4f}")

    # Go/No-Go Gate Check
    if hopkins_val >= 0.5 and silhouette_scores[3] >= 0.1:
        print("  Go/No-Go Gate Status: PASS")
    else:
        print("  Go/No-Go Gate Status: FAIL (Hopkins or Silhouette score too low). Stopping.")
        sys.exit(1)

    print("\nStep 4: Running K-means K=3 clustering...")
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10).fit(X_pca)
    labels = kmeans.labels_

    df_processed['cluster_id'] = labels
    df_processed['period_id'] = 1  # Period T

    print("Computing cluster medoids and spreads (sampling cap of 2000 points)...")
    cluster_centroids = {}
    cluster_medoids = {}
    intra_cluster_spreads = {}
    cluster_medians = {}

    for c in [0, 1, 2]:
        # Centroid in PCA space
        centroid = kmeans.cluster_centers_[c]
        cluster_centroids[c] = centroid
        
        # Medians in raw feature space for drift shifts
        cluster_medians[c] = df_processed[df_processed['cluster_id'] == c][behavioral_features].median().to_dict()
        
        # Sub-sample to max 2000 points for medoid computation to avoid memory limits
        idx_c = np.where(labels == c)[0]
        if len(idx_c) > 2000:
            print(f"  Cluster {c+1} size ({len(idx_c)}) exceeds 2000 limit. Sampling 2000 points.")
            np.random.seed(42)
            sample_indices = np.random.choice(len(idx_c), size=2000, replace=False)
            idx_c_sampled = idx_c[sample_indices]
        else:
            idx_c_sampled = idx_c
            
        X_pca_c = X_pca[idx_c]
        X_pca_c_sampled = X_pca[idx_c_sampled]
        
        # Find medoid
        dist_to_centroid = np.linalg.norm(X_pca_c_sampled - centroid, axis=1)
        medoid_local_idx = np.argmin(dist_to_centroid)
        medoid_idx = idx_c_sampled[medoid_local_idx]
        medoid_customer_id = df_processed.iloc[medoid_idx]['customer_id']
        cluster_medoids[c] = medoid_customer_id
        
        # Intra-cluster spread (mean distance to centroid)
        dist_all_to_centroid = np.linalg.norm(X_pca_c - centroid, axis=1)
        intra_cluster_spreads[c] = float(dist_all_to_centroid.mean())
        print(f"  Cluster {c+1}: Medoid ID = {medoid_customer_id}, Spread = {intra_cluster_spreads[c]:.4f}")

    print("\nStep 5: Simulating T+1 synthetic snapshot using gradual drift...")
    df_t1 = df_processed.copy()
    df_t1['period_id'] = 2

    # Group splits: A (40% - 2,000), B (30% - 1,500), C (30% - 1,500)
    N = len(df_processed)
    indices = list(range(N))
    random.seed(42)
    random.shuffle(indices)

    group_a = set(indices[:2000])
    group_b = set(indices[2000:3500])
    group_c = set(indices[3500:])

    df_t1['drift_group'] = 'A'
    df_t1['target_cluster'] = -2

    # Drift columns for Group B
    layer8_cols = ['Total_Purchases', 'Lifetime_Value', 'Session_Duration_Avg', 'Pages_Per_Session', 'Customer_Rating_Avg']

    print("  Simulating Group B minor drift (shifting exactly 2-3 columns by 40%)...")
    for idx in group_b:
        current_cluster = labels[idx]
        target_cluster = random.choice([cl for cl in [0, 1, 2] if cl != current_cluster])
        
        # Drift 2 or 3 random columns
        num_cols = random.choice([2, 3])
        drift_cols = random.sample(layer8_cols, num_cols)
        
        for col in drift_cols:
            old_val = df_t1.loc[idx, col]
            target_med = cluster_medians[target_cluster][col]
            new_val = old_val + 0.40 * (target_med - old_val)  # 40% shift
            df_t1.loc[idx, col] = new_val
            
        df_t1.loc[idx, 'drift_group'] = 'B'
        df_t1.loc[idx, 'target_cluster'] = target_cluster

    print("  Simulating Group C genuine migration (shifting all columns by 70-90%)...")
    for idx in group_c:
        current_cluster = labels[idx]
        target_cluster = random.choice([cl for cl in [0, 1, 2] if cl != current_cluster])
        
        for col in behavioral_features:
            old_val = df_t1.loc[idx, col]
            target_med = cluster_medians[target_cluster][col]
            shift_pct = random.uniform(0.70, 0.90)
            new_val = old_val + shift_pct * (target_med - old_val)
            df_t1.loc[idx, col] = new_val
            
        df_t1.loc[idx, 'drift_group'] = 'C'
        df_t1.loc[idx, 'target_cluster'] = target_cluster

    # Predict T+1 assignments using KNN trained on Period T
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_pca, labels)
    
    # Scale and reduce T+1 behavioral values for prediction
    X_t1_scaled = scaler.transform(df_t1[behavioral_features])
    X_t1_pca = pca.transform(X_t1_scaled)
    preds_t1 = knn.predict(X_t1_pca)
    df_t1['cluster_id'] = preds_t1

    # Save augmented CSVs
    print("Saving augmented CSV files...")
    df_processed.to_csv(os.path.join(BASE_DIR, "ecommerce_customer_behavior_augmented_T.csv"), index=False)
    df_t1.to_csv(os.path.join(BASE_DIR, "ecommerce_customer_behavior_augmented_T1.csv"), index=False)

    print("\nStep 6: Seeding PostgreSQL database and running downstream layers...")
    with get_raw_connection() as conn:
        with conn.cursor() as cur:
            # 1. Truncate existing seeding tables
            print("  Clearing existing tables...")
            tables_to_clear = [
                "demographic_segment_metrics", "layer8_config", "quarterly_narrations",
                "statistical_test_results", "feature_migration_signals", "segment_outcomes",
                "transition_matrix", "migration_labels", "cluster_identity_map",
                "cluster_assignments", "demographic_snapshots", "behavioral_snapshots",
                "clusters", "cluster_baselines", "notifications", "cross_field_rules",
                "column_profiles", "datasets", "periods", "customers"
            ]
            for tbl in tables_to_clear:
                cur.execute(f"DELETE FROM {tbl};")

            # 2. Seed Periods
            print("  Seeding periods...")
            cur.execute(
                """
                INSERT INTO periods (period_id, period_label, period_start, period_end, granularity) VALUES
                (0, '2024-Q3', '2024-07-01', '2024-09-30', 'quarterly'),
                (1, '2024-Q4', '2024-10-01', '2024-12-31', 'quarterly'),
                (2, '2025-Q1', '2025-01-01', '2025-03-31', 'quarterly')
                ON CONFLICT (period_id) DO NOTHING;
                """
            )

            # 3. Seed Dataset
            print("  Re-registering dataset in datasets table...")
            cur.execute(
                """
                INSERT INTO datasets (dataset_id, source_type, permanence, file_name, layer0_completed, confirmed_classifications)
                VALUES (%s, 'file_upload', 'permanent', 'ecommerce_customer_behavior_dataset_tr_base.csv', TRUE, %s);
                """,
                (dataset_id, json.dumps(classifications))
            )

            # 4. Seed Baselines
            print("  Seeding cluster baselines...")
            cur.execute(
                """
                INSERT INTO cluster_baselines (baseline_id, adopted_at, is_active, adopted_reason, period_id) VALUES
                (1, '2025-01-01 00:00:00+00', TRUE, 'initial', 1)
                ON CONFLICT (baseline_id) DO NOTHING;
                """
            )

            # 5. Seed Customers
            print("  Seeding customers table...")
            customer_ids = df_processed['customer_id'].unique().tolist()
            for cid in customer_ids:
                cur.execute("INSERT INTO customers (customer_id) VALUES (%s) ON CONFLICT (customer_id) DO NOTHING;", (cid,))

            # 6. Seed Clusters
            print("  Seeding clusters table...")
            for c in [0, 1, 2]:
                cur.execute(
                    """
                    INSERT INTO clusters (cluster_id, baseline_id, cluster_label, medoid_customer_id, intra_cluster_spread, cluster_description)
                    VALUES (%s, 1, %s, %s, %s, %s);
                    """,
                    (c + 1, f"Cluster_{c+1}", cluster_medoids[c], intra_cluster_spreads[c], f"Centroid description for Cluster {c+1}")
                )

            # 7. Seed Snapshots (Period 0, Period 1, and Period 2)
            print("  Seeding snapshots and assignments across periods...")
            df_agg_lookup = df_agg.set_index('customer_id')
            
            # Period 0 (T-1) baseline for demographic MoM comparison
            np.random.seed(42)
            p0_sample = df_processed.sample(frac=0.89, random_state=42)
            for _, row in p0_sample.iterrows():
                cid = row["customer_id"]
                demo = df_agg_lookup.loc[cid]
                cur.execute(
                    """
                    INSERT INTO demographic_snapshots (customer_id, period_id, age, gender, "City")
                    VALUES (%s, 0, %s, %s, %s);
                    """,
                    (cid, float(demo["Age"]), str(demo["Gender"]), str(demo["City"]))
                )

            # Period 1 (T)
            for _, row in df_processed.iterrows():
                cid = row["customer_id"]
                demo = df_agg_lookup.loc[cid]
                # Demographic Snapshot
                cur.execute(
                    """
                    INSERT INTO demographic_snapshots (customer_id, period_id, age, gender, "City")
                    VALUES (%s, 1, %s, %s, %s);
                    """,
                    (cid, float(demo["Age"]), str(demo["Gender"]), str(demo["City"]))
                )
                # Behavioral Snapshot
                cur.execute(
                    """
                    INSERT INTO behavioral_snapshots (
                        customer_id, period_id, "Lifetime_Value", "Total_Purchases", 
                        "Session_Duration_Avg", "Pages_Per_Session", "Customer_Rating_Avg", 
                        "Discount_Total", "Delivery_Time_Avg", "Returned_Ever", "Quantity_Avg"
                    ) VALUES (%s, 1, %s, %s, %s, %s, %s, %s, %s, %s, %s);
                    """,
                    (
                        cid, row["Lifetime_Value"], row["Total_Purchases"], 
                        row["Session_Duration_Avg"], row["Pages_Per_Session"], row["Customer_Rating_Avg"], 
                        row["Discount_Total"], row["Delivery_Time_Avg"], row["Returned_Ever"], row["Quantity_Avg"]
                    )
                )
                # Assignments
                cur.execute(
                    """
                    INSERT INTO cluster_assignments (customer_id, period_id, cluster_id, assigned_via)
                    VALUES (%s, 1, %s, 'kmeans');
                    """,
                    (cid, int(row["cluster_id"] + 1))
                )

            # Period 2 (T+1) with realistic demographic turnover (churn + acquisition)
            p2_active = df_t1.sample(frac=0.94, random_state=101)
            for _, row in p2_active.iterrows():
                cid = row["customer_id"]
                demo = df_agg_lookup.loc[cid]
                # Demographic Snapshot
                cur.execute(
                    """
                    INSERT INTO demographic_snapshots (customer_id, period_id, age, gender, "City")
                    VALUES (%s, 2, %s, %s, %s);
                    """,
                    (cid, float(demo["Age"]), str(demo["Gender"]), str(demo["City"]))
                )
                # Behavioral Snapshot
                cur.execute(
                    """
                    INSERT INTO behavioral_snapshots (
                        customer_id, period_id, "Lifetime_Value", "Total_Purchases", 
                        "Session_Duration_Avg", "Pages_Per_Session", "Customer_Rating_Avg", 
                        "Discount_Total", "Delivery_Time_Avg", "Returned_Ever", "Quantity_Avg"
                    ) VALUES (%s, 2, %s, %s, %s, %s, %s, %s, %s, %s, %s);
                    """,
                    (
                        cid, row["Lifetime_Value"], row["Total_Purchases"], 
                        row["Session_Duration_Avg"], row["Pages_Per_Session"], row["Customer_Rating_Avg"], 
                        row["Discount_Total"], row["Delivery_Time_Avg"], row["Returned_Ever"], row["Quantity_Avg"]
                    )
                )
                # Assignments
                cur.execute(
                    """
                    INSERT INTO cluster_assignments (customer_id, period_id, cluster_id, assigned_via)
                    VALUES (%s, 2, %s, 'knn');
                    """,
                    (cid, int(row["cluster_id"] + 1))
                )
                
            conn.commit()

    print("  Running Layer 1 Demographic Segment Metrics & Anomaly Detection...")
    demo_cols_to_use = ["Age", "Gender", "City"]
    compute_and_persist_mom_metrics(dataset_id, 1, 0, demo_cols_to_use)
    compute_and_persist_mom_metrics(dataset_id, 2, 1, demo_cols_to_use)
    flag_anomalies_for_period(dataset_id, 1)
    flag_anomalies_for_period(dataset_id, 2)

    # PM Names Clusters via POST /api/v1/clusters/{id}/name
    print("  PM naming clusters via POST endpoint...")
    for c in [0, 1, 2]:
        name_cluster(cluster_id=c+1, payload=NameClusterPayload(name=f"Cluster {c+1}"))
    print("  Cluster naming successfully completed.")

    # 1. Run Layer 4 migration label computation & persistence
    print("  Running Layer 4 Retraining & Migration Labels Computation...")
    trigger_retraining(dataset_id, 1)

    # 2. Run Layer 5 transition matrix computation & persistence
    print("  Running Layer 5 Transition Matrix Computation...")
    compute_and_persist_transition_matrix(dataset_id, 1)

    # 3. Run Layer 6 outcome metric computation & persistence
    print("  Running Layer 6 Outcomes Aggregation for both periods...")
    compute_and_persist_segment_outcomes(dataset_id, 1)
    compute_and_persist_segment_outcomes(dataset_id, 2)

    # 4. Run Layer 3 statistical tests & persistence
    print("  Running Layer 3 Statistical Testing...")
    for var in ["churn_rate", "retention_rate", "revenue", "repeat_purchase_prob"]:
        run_statistical_tests(dataset_id, 1, "cluster", var)

    # 5. Run Layer 8 feature migration signals & persistence
    print("  Configuring and running Layer 8 Signals...")
    confirm_layer8_payload = [{"column_name": col, "include": True} for col in behavioral_features]
    confirm_layer8_columns(dataset_id, confirm_layer8_payload)
    compute_and_persist_feature_migration_signals(dataset_id, 1)

    # 6. Run Agent 1 Narration for Period 1
    print("  Running Agent 1 Narration Generation for Period 1...")
    generate_and_evaluate_narration(1)

    print("\nDatabase seeding and layer calculations completed successfully!")

    print("\nStep 7: Validating all layers have data by querying GET endpoints...")
    client = TestClient(app)
    
    endpoint_urls = {
        "L1 (demographic-segments)": "/api/v1/demographic-segments?period_id=1",
        "L2 (quality-gates)": "/api/v1/clusters/quality-gates?period_id=1",
        "L3 (statistical-tests)": "/api/v1/statistical-tests?period_id=1",
        "L4 (migrations)": "/api/v1/migrations?period_id=1",
        "L5 (transition-matrix)": "/api/v1/transition-matrix?from_period_id=1&to_period_id=2",
        "L6 (segment outcomes)": "/api/v1/segments/1/outcomes?period_id=1",
        "L7 (metrics comparison)": "/api/v1/metrics/comparison?granularity=quarterly&metric=revenue&segment_ids=1,2&period_range=2024-Q4,2025-Q1",
        "L8 (feature migration signals)": "/api/v1/feature-migration-signals?period_id=1",
        "A1 (narrations)": "/api/v1/narrations?period_id=1"
    }

    endpoint_checks_passed = True
    for name, url in endpoint_urls.items():
        response = client.get(url)
        if response.status_code == 200:
            content = response.json()
            is_empty = False
            if isinstance(content, list) and len(content) == 0:
                is_empty = True
            elif isinstance(content, dict) and not content:
                is_empty = True
                
            if is_empty:
                print(f"  [FAIL] Endpoint {name} [{url}] returned empty content.")
                endpoint_checks_passed = False
            else:
                print(f"  [PASS] Endpoint {name} [{url}] returned non-empty response.")
        else:
            print(f"  [FAIL] Endpoint {name} [{url}] failed with status code {response.status_code}.")
            endpoint_checks_passed = False

    print("\nStep 8: Computing validation metrics...")
    
    # 1. Group B label-flip rate (target: close to 0%)
    df_t1_b = df_t1.loc[list(group_b)]
    df_processed_b = df_processed.loc[list(group_b)].set_index('customer_id')
    flip_count = 0
    for cid, row in df_t1_b.iterrows():
        t1_label = row['cluster_id']
        t0_label = df_processed_b.loc[row['customer_id']]['cluster_id']
        if t1_label != t0_label:
            flip_count += 1
    flip_rate = float(flip_count) / len(group_b)
    print(f"  Group B Label-flip rate: {flip_rate:.2%} ({flip_count} / {len(group_b)})")

    # 2. Group B Layer 8 signal raise rate (target: ~30%)
    signals_raised_count = 0
    with get_raw_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT customer_id, compared_cluster_id, signal_raised 
                FROM feature_migration_signals 
                WHERE period_id = 1;
                """
            )
            sig_rows = cur.fetchall()
            
    sig_map = {}
    for cid, comp_cid, raised in sig_rows:
        sig_map[(str(cid), int(comp_cid))] = bool(raised)
        
    for idx in group_b:
        cid = df_processed.loc[idx]['customer_id']
        t_cluster = int(df_t1.loc[idx]['target_cluster']) + 1 # 1-indexed for compared_cluster_id
        is_raised = sig_map.get((cid, t_cluster), False)
        if is_raised:
            signals_raised_count += 1
            
    signal_raise_rate = float(signals_raised_count) / len(group_b)
    print(f"  Group B Layer 8 signal raise rate: {signal_raise_rate:.2%} ({signals_raised_count} / {len(group_b)})")

    # 3. Group C re-clustering rate (target: close to 100%)
    df_t1_c = df_t1.loc[list(group_c)]
    df_processed_c = df_processed.loc[list(group_c)].set_index('customer_id')
    flip_c_count = 0
    for cid, row in df_t1_c.iterrows():
        t1_label = row['cluster_id']
        t0_label = df_processed_c.loc[row['customer_id']]['cluster_id']
        if t1_label != t0_label:
            flip_c_count += 1
    re_clustering_rate = float(flip_c_count) / len(group_c)
    print(f"  Group C Re-clustering rate: {re_clustering_rate:.2%} ({flip_c_count} / {len(group_c)})")

    if endpoint_checks_passed:
        print("  All 9 layer endpoint checks returned non-empty responses.")
    else:
        print("  Warning: One or more endpoint checks failed or returned empty data.")

    print("\n Turkey dataset synthetic data augmentation run successfully completed!")
    print(f"Seeded Dataset ID: {dataset_id}")

if __name__ == "__main__":
    run_tr_pipeline()
