import pandas as pd
import numpy as np
import uuid
import random
import datetime
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import HDBSCAN
from sklearn.metrics import pairwise_distances

def run_pipeline():
    print("Step 1: Loading base dataset...")
    base_path = r"C:\Projects\Segment studio\ecommerce_customer_churn_dataset.csv"
    df = pd.read_csv(base_path)
    print(f"Loaded {len(df)} rows and {len(df.columns)} columns.")

    print("\nStep 1: Generating unique customer IDs and time anchors...")
    # Generate unique UUIDs for customer_id
    random.seed(42)
    np.random.seed(42)
    df['customer_id'] = [str(uuid.UUID(int=random.getrandbits(128))) for _ in range(len(df))]
    
    # Generate random signup dates falling within the row's Signup_Quarter
    quarters = {
        'Q1': (1, 1, 3, 31),
        'Q2': (4, 1, 6, 30),
        'Q3': (7, 1, 9, 30),
        'Q4': (10, 1, 12, 31)
    }

    def row_signup_date(row):
        quarter = row['Signup_Quarter']
        # Compute year realistically from Membership_Years (base current year: 2026)
        years = int(float(row['Membership_Years']))
        year = 2026 - years
        if quarter not in quarters:
            quarter = 'Q1'
        
        start_month, start_day, end_month, end_day = quarters[quarter]
        # Handle leap years
        is_leap = (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)
        if end_month == 2 and end_day == 28 and is_leap:
            end_day = 29
            
        start_date = datetime.date(year, start_month, start_day)
        end_date = datetime.date(year, end_month, end_day)
        
        days_between = (end_date - start_date).days
        random_days = random.randint(0, days_between)
        return (start_date + datetime.timedelta(days=random_days)).isoformat()

    df['signup_date'] = df.apply(row_signup_date, axis=1)
    df['period_id'] = 1  # T snapshot is period_id = 1
    
    print("Time anchors and customer IDs generated successfully.")

    print("\nStep 2: Running Layer 2 HDBSCAN clustering on T snapshot...")
    # Behavioral features used for clustering
    features = [
        'Login_Frequency', 'Session_Duration_Avg', 'Pages_Per_Session', 
        'Cart_Abandonment_Rate', 'Total_Purchases', 'Average_Order_Value', 
        'Discount_Usage_Rate', 'Returns_Rate'
    ]
    
    # Impute missing values with median for behavioral columns
    for col in features:
        df[col] = df[col].fillna(df[col].median())
        
    # Standardize using RobustScaler as specified in Layer 0 / Layer 2
    scaler = RobustScaler()
    X = scaler.fit_transform(df[features])
    
    # Run HDBSCAN (tuning params: min_cluster_size=200, min_samples=10)
    # This aligns with Layer 2.2 Quality Gates (keeping noise < 30%)
    clusterer = HDBSCAN(min_cluster_size=200, min_samples=10)
    labels = clusterer.fit_predict(X)
    
    active_clusters = sorted([c for c in set(labels) if c != -1])
    noise_ratio = np.mean(labels == -1) if len(labels) > 0 else 1.0
    
    if len(active_clusters) == 0:
        print("HDBSCAN found 0 active clusters on this uniform dataset. Falling back to KMeans for synthetic baseline partitioning...")
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=3, random_state=42)
        labels = kmeans.fit_predict(X)
        active_clusters = sorted(list(set(labels)))
        noise_ratio = 0.0
        print(f"KMeans clustering finished. Found {len(active_clusters)} active clusters.")
    else:
        print(f"HDBSCAN clustering finished. Found {len(active_clusters)} active clusters (excluding noise label -1).")
        print(f"Noise bucket ratio: {noise_ratio:.2%}")
        
    df['cluster_id'] = labels

    # Compute raw column medians per cluster
    print("Computing per-cluster column medians and medoids...")
    cluster_medians = {}
    cluster_medoids = {}
    intra_cluster_spreads = {}

    for c in active_clusters:
        cluster_medians[c] = df[df['cluster_id'] == c][features].median().to_dict()
        
        # Medoid computation (scaled coordinates)
        idx_c = np.where(labels == c)[0]
        X_c = X[idx_c]
        dist_matrix = pairwise_distances(X_c, metric='euclidean')
        medoid_local_idx = dist_matrix.sum(axis=1).argmin()
        medoid_idx = idx_c[medoid_local_idx]
        medoid_customer_id = df.iloc[medoid_idx]['customer_id']
        cluster_medoids[c] = medoid_customer_id
        
        # Compute intra-cluster spread (mean distance to medoid)
        medoid_coords = X_c[medoid_local_idx].reshape(1, -1)
        distances_to_medoid = pairwise_distances(X_c, medoid_coords, metric='euclidean').flatten()
        intra_cluster_spreads[c] = float(distances_to_medoid.mean())
        print(f"  Cluster {c}: Medoid ID = {medoid_customer_id}, Intra-Cluster Spread = {intra_cluster_spreads[c]:.4f}")

    print("\nStep 3: Generating T+1 snapshot...")
    # Create the T+1 DataFrame skeleton
    df_t1 = df.copy()
    df_t1['period_id'] = 2  # T+1 snapshot is period_id = 2
    
    # Split customers into three groups:
    # Group A (40,000): Unchanged
    # Group B (5,000): Minor drift (Layer 8)
    # Group C (5,000): Genuine migration (Layer 4/5)
    indices = list(range(len(df)))
    random.shuffle(indices)
    
    group_a_indices = set(indices[:40000])
    group_b_indices = set(indices[40000:45000])
    group_c_indices = set(indices[45000:])
    
    # Gated columns used for Layer 8 drift comparisons
    layer8_cols = ['Login_Frequency', 'Session_Duration_Avg', 'Pages_Per_Session', 'Total_Purchases', 'Average_Order_Value']
    
    df_t1['drift_group'] = 'A'
    df_t1['target_cluster'] = -2  # Default sentinel value (no target)

    print("  Processing Group B (Minor Drift - 5,000 customers)...")
    for idx in group_b_indices:
        current_cluster = labels[idx]
        if current_cluster == -1:
            current_cluster = random.choice(active_clusters)
        
        # Select target cluster (different from current)
        target_cluster = random.choice([ac for ac in active_clusters if ac != current_cluster])
        
        # Decide number of columns to drift: 2 columns (70% probability) or 3 columns (30% probability)
        # This will test the 50% rollup threshold for Layer 8: 3/5 is >50%, 2/5 is <=50%
        num_cols_to_drift = np.random.choice([2, 3], p=[0.70, 0.30])
        drifted_cols = random.sample(layer8_cols, int(num_cols_to_drift))
        
        for col in drifted_cols:
            old_val = df_t1.loc[idx, col]
            target_med = cluster_medians[target_cluster][col]
            # Nudge value 40% closer to comparison cluster median
            new_val = old_val + 0.40 * (target_med - old_val)
            
            # Preserve column types
            if df[col].dtype in [np.int64, np.int32, int]:
                new_val = int(round(new_val))
            df_t1.loc[idx, col] = new_val
            
        df_t1.loc[idx, 'drift_group'] = 'B'
        df_t1.loc[idx, 'target_cluster'] = target_cluster

    print("  Processing Group C (Genuine Migration - 5,000 customers)...")
    # For genuine migration, we pull the full behavioral profile of a target customer in c_target
    cluster_customer_indices = {c: np.where(labels == c)[0] for c in active_clusters}
    
    for idx in group_c_indices:
        current_cluster = labels[idx]
        if current_cluster == -1:
            current_cluster = random.choice(active_clusters)
            
        target_cluster = random.choice([ac for ac in active_clusters if ac != current_cluster])
        
        # Sample a random target customer from the target cluster
        target_customer_idx = random.choice(cluster_customer_indices[target_cluster])
        
        # Copy all 8 behavioral features to the T+1 snapshot
        for col in features:
            df_t1.loc[idx, col] = df.loc[target_customer_idx, col]
            
        df_t1.loc[idx, 'drift_group'] = 'C'
        df_t1.loc[idx, 'target_cluster'] = target_cluster

    print("\nPipeline Validation Step:")
    # Validate that Group C genuinely migrates and Group B does not.
    # Train a KNN classifier on the T snapshot to predict cluster labels on T+1
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X, labels)
    
    # Predict on T+1 snapshot
    X_t1 = scaler.transform(df_t1[features])
    preds_t1 = knn.predict(X_t1)
    df_t1['predicted_cluster_id'] = preds_t1
    
    group_b_df = df_t1.loc[list(group_b_indices)]
    group_c_df = df_t1.loc[list(group_c_indices)]
    
    b_migration_rate = (group_b_df['predicted_cluster_id'] != group_b_df['cluster_id']).mean()
    c_migration_rate = (group_c_df['predicted_cluster_id'] == group_c_df['target_cluster']).mean()
    
    print(f"  Group B (Drift) label-flip rate (should be close to 0%): {b_migration_rate:.2%}")
    print(f"  Group C (Migration) correct re-clustering rate (should be close to 100%): {c_migration_rate:.2%}")

    # Validate Layer 8 per-column medians rollup rate
    # Verify what percentage of Group B actually triggers the rollup (>50% columns drifted)
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
            
    print(f"  Group B Layer 8 signal raise rate (expected ~30% based on our 2/3 columns split): {layer8_signals_raised / len(group_b_indices):.2%}")

    print("\nStep 4: Persisting snapshots...")
    df.to_csv(r"C:\Projects\Segment studio\ecommerce_customer_churn_augmented_T.csv", index=False)
    df_t1.to_csv(r"C:\Projects\Segment studio\ecommerce_customer_churn_augmented_T1.csv", index=False)
    print("CSV files saved to C:\\Projects\\Segment studio\\")
    
    # Write seeder SQL script
    print("Generating Postgres seed SQL script...")
    sql_path = r"C:\Projects\Segment studio\seed_augmented_data.sql"
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

if __name__ == "__main__":
    run_pipeline()
