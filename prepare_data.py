import os
import json
import uuid
import random
import datetime
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import HDBSCAN
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import pairwise_distances

# Setup django-like or python path imports
import sys
sys.path.append(os.path.abspath("."))

from src.ingestion.router import ingest_from_file
from src.api.v1.endpoints import confirm_columns, ConfirmColumnsPayload, ColumnConfirmation, run_layer0
from src.layer2.preprocessor import load_and_preprocess_behavioral_data
from src.layer2.clusterer import run_hdbscan_clustering
from src.db.connection import get_raw_connection

# Path resolution
BASE_DIR = "/app" if os.path.exists("/app") else r"C:\Projects\Segment studio"

def run_augmentation_pipeline():
    # Configure outlier and skewness thresholds
    import src.layer0.outliers
    src.layer0.outliers.MAHALANOBIS_DISTANCE_THRESHOLD = 4.5
    src.layer0.outliers.MEDIAN_SIMILARITY_THRESHOLD = 0.10
    src.layer0.outliers.SKEWNESS_TRANSFORM_THRESHOLD = 1.0
    
    os.environ["MAHALANOBIS_DISTANCE_THRESHOLD"] = "4.5"
    os.environ["MEDIAN_SIMILARITY_THRESHOLD"] = "0.10"
    os.environ["SKEWNESS_TRANSFORM_THRESHOLD"] = "1.0"
    if "PCA_VARIANCE_THRESHOLD" not in os.environ:
        os.environ["PCA_VARIANCE_THRESHOLD"] = "0.85"

    print("Step 1: Loading raw base dataset...")
    base_path = os.path.join(BASE_DIR, "ecommerce_customer_churn_dataset.csv")
    if not os.path.exists(base_path):
        print(f"Error: Base dataset not found at {base_path}")
        sys.exit(1)
        
    df_raw = pd.read_csv(base_path)
    print(f"Loaded {len(df_raw)} rows and {len(df_raw.columns)} columns.")

    print("\nGenerating unique customer IDs and signup dates (seed=42)...")
    random.seed(42)
    np.random.seed(42)
    df_raw['customer_id'] = [str(uuid.UUID(int=random.getrandbits(128))) for _ in range(len(df_raw))]
    
    quarters = {
        'Q1': (1, 1, 3, 31),
        'Q2': (4, 1, 6, 30),
        'Q3': (7, 1, 9, 30),
        'Q4': (10, 1, 12, 31)
    }

    def row_signup_date(row):
        quarter = row['Signup_Quarter']
        years = int(float(row['Membership_Years']))
        year = 2026 - years
        if quarter not in quarters:
            quarter = 'Q1'
        
        start_month, start_day, end_month, end_day = quarters[quarter]
        is_leap = (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)
        if end_month == 2 and end_day == 28 and is_leap:
            end_day = 29
            
        start_date = datetime.date(year, start_month, start_day)
        end_date = datetime.date(year, end_month, end_day)
        
        days_between = (end_date - start_date).days
        random_days = random.randint(0, days_between)
        return (start_date + datetime.timedelta(days=random_days)).isoformat()

    df_raw['signup_date'] = df_raw.apply(row_signup_date, axis=1)

    # Save to temp path for ingestion router
    temp_path = os.path.join(BASE_DIR, "temp_to_ingest.csv")
    df_raw.to_csv(temp_path, index=False)

    print("\nIngesting base dataset via router...")
    dataset_id = ingest_from_file(temp_path, "ecommerce_customer_churn_dataset_base.csv")
    print(f"Successfully ingested dataset. dataset_id={dataset_id}")
    
    if os.path.exists(temp_path):
        os.remove(temp_path)

    print("\nConfirming all columns in Layer 0...")
    confirm_payload = ConfirmColumnsPayload(
        columns=[ColumnConfirmation(column_name=col, include=True) for col in df_raw.columns]
    )
    confirm_columns(dataset_id, confirm_payload)

    print("\nSetting imputation methods to 'median' for missing columns in DB...")
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

    print("\nRunning full Layer 0 Pipeline on base dataset...")
    run_layer0(dataset_id)

    # Verify layer0_completed in database
    with get_raw_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT layer0_completed FROM datasets WHERE dataset_id = %s;", (dataset_id,))
            layer0_status = cur.fetchone()[0]
            
    if not layer0_status:
        print("Error: Layer 0 did not complete successfully. Halt execution.")
        sys.exit(1)
    print("Layer 0 pipeline completed and verified in DB.")

    # 8 behavioral features used for clustering
    features = [
        'Login_Frequency', 'Session_Duration_Avg', 'Pages_Per_Session', 
        'Cart_Abandonment_Rate', 'Total_Purchases', 'Average_Order_Value', 
        'Discount_Usage_Rate', 'Returns_Rate'
    ]

    print("\nLoading processed dataset and classifying columns...")
    processed_path = os.path.join("data", f"{dataset_id}_processed.csv")
    df_processed = pd.read_csv(processed_path)
    
    classifications = []
    for col in df_processed.columns:
        if col == "customer_id":
            continue
        if col in features:
            classifications.append({"column": col, "classification": "behavioral"})
        else:
            classifications.append({"column": col, "classification": "demographic"})

    with get_raw_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE datasets
                SET confirmed_classifications = %s
                WHERE dataset_id = %s;
                """,
                (json.dumps(classifications), dataset_id)
            )

    print("\nRunning Layer 2 preprocessing and verifying checks...")
    X_df, behavioral_cols = load_and_preprocess_behavioral_data(dataset_id)
    assert not X_df[behavioral_cols].isna().any().any(), "Feature matrix contains NaN values!"
    print("Feature matrix preprocessed and verified (no NaN values, numeric only).")

    if "HDBSCAN_MIN_CLUSTER_SIZE" not in os.environ:
        os.environ["HDBSCAN_MIN_CLUSTER_SIZE"] = "30"
    if "HDBSCAN_MIN_SAMPLES" not in os.environ:
        os.environ["HDBSCAN_MIN_SAMPLES"] = "2"
    if "HDBSCAN_METRIC" not in os.environ:
        os.environ["HDBSCAN_METRIC"] = "euclidean"
    
    random.seed(42)
    np.random.seed(42)
    labels = run_hdbscan_clustering(X_df, behavioral_cols)
    
    noise_ratio = np.mean(labels == -1)
    active_clusters = sorted([c for c in set(labels) if c != -1])
    print(f"Found {len(active_clusters)} active clusters. Noise ratio: {noise_ratio:.2%}")

    df_processed['cluster_id'] = labels
    df_processed['period_id'] = 1  # T snapshot is period_id = 1

    # Standardize using RobustScaler for distance calculations
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_df[behavioral_cols])

    # Compute per-cluster medians and medoids with sub-sampling cap
    print("\nComputing per-cluster medians and medoids (sub-sampling cap = 2000)...")
    cluster_medians = {}
    cluster_medoids = {}
    intra_cluster_spreads = {}

    for c in active_clusters:
        # Compute medians on raw values for drift calculations
        cluster_medians[c] = df_processed[df_processed['cluster_id'] == c][behavioral_cols].median().to_dict()
        
        idx_c = np.where(labels == c)[0]
        # Medoid sub-sampling cap of 2000 points
        if len(idx_c) > 2000:
            print(f"  Cluster {c} size ({len(idx_c)}) exceeds 2000 limit. Sub-sampling to 2000 points.")
            random.seed(42)
            np.random.seed(42)
            sample_indices = np.random.choice(len(idx_c), size=2000, replace=False)
            idx_c_sampled = idx_c[sample_indices]
            X_c_sampled = X_scaled[idx_c_sampled]
            
            dist_matrix = pairwise_distances(X_c_sampled, metric='euclidean')
            medoid_local_idx = dist_matrix.sum(axis=1).argmin()
            medoid_idx = idx_c_sampled[medoid_local_idx]
        else:
            X_c = X_scaled[idx_c]
            dist_matrix = pairwise_distances(X_c, metric='euclidean')
            medoid_local_idx = dist_matrix.sum(axis=1).argmin()
            medoid_idx = idx_c[medoid_local_idx]
            
        medoid_customer_id = X_df.iloc[medoid_idx]['customer_id']
        cluster_medoids[c] = medoid_customer_id
        
        # Intra-cluster spread (mean distance to medoid on full cluster)
        medoid_coords = X_scaled[medoid_idx].reshape(1, -1)
        X_c_full = X_scaled[idx_c]
        distances_to_medoid = pairwise_distances(X_c_full, medoid_coords, metric='euclidean').flatten()
        intra_cluster_spreads[c] = float(distances_to_medoid.mean())
        print(f"  Cluster {c}: Medoid ID = {medoid_customer_id}, Intra-Cluster Spread = {intra_cluster_spreads[c]:.4f}")

    print("\nStep 3: Generating T+1 snapshot...")
    df_t1 = df_processed.copy()
    df_t1['period_id'] = 2  # T+1 snapshot is period_id = 2

    # Group splits: A (40,000), B (5,000), C (5,000)
    indices = list(range(len(df_processed)))
    random.seed(42)
    random.shuffle(indices)
    
    N = len(df_processed)
    size_b = int(round(N * 0.10))
    size_c = int(round(N * 0.10))
    size_a = N - size_b - size_c
    
    group_a_indices = set(indices[:size_a])
    group_b_indices = set(indices[size_a : size_a + size_b])
    group_c_indices = set(indices[size_a + size_b :])
    
    layer8_cols = ['Login_Frequency', 'Session_Duration_Avg', 'Pages_Per_Session', 'Total_Purchases', 'Average_Order_Value']
    
    df_t1['drift_group'] = 'A'
    df_t1['target_cluster'] = -2

    print("  Processing Group B (Minor Drift - 5,000 customers)...")
    for idx in group_b_indices:
        current_cluster = labels[idx]
        if current_cluster == -1:
            current_cluster = random.choice(active_clusters)
            
        target_cluster = random.choice([ac for ac in active_clusters if ac != current_cluster])
        
        # 2 columns (70% prob) or 3 columns (30% prob)
        num_cols_to_drift = np.random.choice([2, 3], p=[0.70, 0.30])
        drifted_cols = random.sample(layer8_cols, int(num_cols_to_drift))
        
        for col in drifted_cols:
            old_val = df_t1.loc[idx, col]
            target_med = cluster_medians[target_cluster][col]
            new_val = old_val + 0.60 * (target_med - old_val)
            if df_processed[col].dtype in [np.int64, np.int32, int]:
                new_val = int(round(new_val))
            df_t1.loc[idx, col] = new_val
            
        df_t1.loc[idx, 'drift_group'] = 'B'
        df_t1.loc[idx, 'target_cluster'] = target_cluster

    print("  Processing Group C (Genuine Migration using gradual drift - 5,000 customers)...")
    for idx in group_c_indices:
        current_cluster = labels[idx]
        if current_cluster == -1:
            current_cluster = random.choice(active_clusters)
            
        target_cluster = random.choice([ac for ac in active_clusters if ac != current_cluster])
        
        # Shift all 8 behavioral columns by 70-90% toward target cluster's median
        for col in behavioral_cols:
            old_val = df_t1.loc[idx, col]
            target_med = cluster_medians[target_cluster][col]
            shift_pct = random.uniform(0.70, 0.90)
            new_val = old_val + shift_pct * (target_med - old_val)
            if df_processed[col].dtype in [np.int64, np.int32, int]:
                new_val = int(round(new_val))
            df_t1.loc[idx, col] = new_val
            
        df_t1.loc[idx, 'drift_group'] = 'C'
        df_t1.loc[idx, 'target_cluster'] = target_cluster

    print("\nPipeline Validation Step:")
    # Print cluster sizes
    print("  Cluster sizes:")
    cluster_counts = pd.Series(labels).value_counts()
    for cl, count in cluster_counts.items():
        name = f"Cluster {cl}" if cl != -1 else "Noise (-1)"
        print(f"    {name}: {count} points")

    # 1. Standard KNN (trained on all points, including noise)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_scaled, labels)
    
    X_t1 = scaler.transform(df_t1[behavioral_cols])
    preds_t1 = knn.predict(X_t1)
    
    b_migration_rate = (preds_t1[list(group_b_indices)] != df_t1.loc[list(group_b_indices), 'cluster_id']).mean()
    c_migration_rate = (preds_t1[list(group_c_indices)] == df_t1.loc[list(group_c_indices), 'target_cluster']).mean()
    
    print("\n  [Standard KNN Validation (including Noise & Class Imbalance)]:")
    print(f"    Group B (Drift) label-flip rate (should be close to 0%): {b_migration_rate:.2%}")
    print(f"    Group C (Migration) correct re-clustering rate (should be close to 100%): {c_migration_rate:.2%}")

    # 2. Balanced KNN (trained on a class-balanced subset, excluding noise)
    X_bal = []
    y_bal = []
    for cl in active_clusters:
        idx_cl = np.where(labels == cl)[0]
        np.random.seed(42)
        sampled_indices = np.random.choice(idx_cl, size=min(len(idx_cl), 30), replace=False)
        X_bal.append(X_scaled[sampled_indices])
        y_bal.append(np.full(len(sampled_indices), cl))
        
    X_bal = np.vstack(X_bal)
    y_bal = np.concatenate(y_bal)
    
    knn_bal = KNeighborsClassifier(n_neighbors=5)
    knn_bal.fit(X_bal, y_bal)
    preds_bal = knn_bal.predict(X_t1)
    
    # Filter Group B and Group C for points that were not noise originally
    gb_valid = [idx for idx in group_b_indices if labels[idx] != -1]
    gc_valid = [idx for idx in group_c_indices if labels[idx] != -1]
    
    b_bal_flip = (preds_bal[gb_valid] != df_t1.loc[gb_valid, 'cluster_id']).mean() if gb_valid else 0.0
    c_bal_match = (preds_bal[gc_valid] == df_t1.loc[gc_valid, 'target_cluster']).mean() if gc_valid else 0.0
    
    print("\n  [Class-Balanced KNN Validation (Active Clusters only, no Noise skew)]:")
    print(f"    Group B (Drift) label-flip rate (clean): {b_bal_flip:.2%}")
    print(f"    Group C (Migration) correct re-clustering rate (clean): {c_bal_match:.2%}")

    # Validate Layer 8 per-column medians rollup rate for Group B
    layer8_signals_raised = 0
    for idx in group_b_indices:
        current_cluster = labels[idx]
        target_cluster = df_t1.loc[idx, 'target_cluster']
        if current_cluster == -1 or target_cluster < 0:
            continue
            
        columns_drifted_closer = 0
        for col in layer8_cols:
            val_t1 = df_t1.loc[idx, col]
            own_med = cluster_medians[current_cluster][col]
            target_med = cluster_medians[target_cluster][col]
            
            d_own = abs(val_t1 - own_med)
            d_target = abs(val_t1 - target_med)
            if d_target < d_own:
                columns_drifted_closer += 1
                
        if columns_drifted_closer / len(layer8_cols) > 0.50:
            layer8_signals_raised += 1
            
    signal_raise_rate = layer8_signals_raised / len(group_b_indices)
    print(f"\n  Group B Layer 8 signal raise rate (expected ~30%): {signal_raise_rate:.2%}")

    print("\nStep 4: Persisting snapshots...")
    df_processed.to_csv(os.path.join(BASE_DIR, "ecommerce_customer_churn_augmented_T.csv"), index=False)
    df_t1.to_csv(os.path.join(BASE_DIR, "ecommerce_customer_churn_augmented_T1.csv"), index=False)
    print("CSV files saved successfully.")

    # Write seeder SQL script
    print("Generating Postgres seed SQL script...")
    sql_path = os.path.join(BASE_DIR, "seed_augmented_data.sql")
    with open(sql_path, "w") as f:
        f.write("-- ==========================================================\n")
        f.write("-- Customer Segmentation Studio - Augmented Seeder Script\n")
        f.write("-- Note: T+1 layer is synthetically generated for testing/dev\n")
        f.write("-- ==========================================================\n\n")
        
        # Seed Periods
        f.write("INSERT INTO periods (period_id, period_label, period_start, period_end, granularity) VALUES\n")
        f.write("  (1, '2024-Q4', '2024-10-01', '2024-12-31', 'quarterly'),\n")
        f.write("  (2, '2025-Q1', '2025-01-01', '2025-03-31', 'quarterly')\n")
        f.write("ON CONFLICT (period_id) DO NOTHING;\n\n")
        
        # Seed Baselines
        f.write("INSERT INTO cluster_baselines (baseline_id, adopted_at, is_active, adopted_reason, period_id) VALUES\n")
        f.write("  (1, '2025-01-01 00:00:00+00', TRUE, 'initial', 1)\n")
        f.write("ON CONFLICT (baseline_id) DO NOTHING;\n\n")
        
        # Seed Clusters
        f.write("INSERT INTO clusters (cluster_id, baseline_id, cluster_label, medoid_customer_id, intra_cluster_spread) VALUES\n")
        for c in active_clusters:
            medoid_id = cluster_medoids[c]
            spread = intra_cluster_spreads[c]
            f.write(f"  ({c + 1}, 1, 'Cluster_{c}', '{medoid_id}', {spread}),\n")
        # Remove trailing comma and replace with semicolon
        f.seek(f.tell() - 2)
        f.write("\nON CONFLICT (cluster_id) DO NOTHING;\n\n")
        
    print(f"SQL file saved to {sql_path}")
    print("\nData Augmentation Pipeline finished.")

if __name__ == "__main__":
    run_augmentation_pipeline()
