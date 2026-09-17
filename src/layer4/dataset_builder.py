import logging
import pandas as pd
import numpy as np
from src.db.connection import get_raw_connection

logger = logging.getLogger("layer4_dataset_builder")

def build_training_data(dataset_id: str, T: int) -> tuple[pd.DataFrame, pd.Series]:
    """
    Constructs the feature matrix X (from behavioral_snapshots at period T) and 
    the target vector y (binary migration flags comparing T assignment vs T+1 assignment).
    
    Customers without a cluster assignment at T (e.g. new signups at T+1) are strictly
    excluded via SQL INNER JOIN.
    """
    logger.info(f"Building predictive segmentation training dataset for period T={T}...")
    
    # 1. Fetch matched cluster assignments at T and T+1 (INNER JOIN automatically excludes customers absent in T or T+1)
    with get_raw_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT ca1.customer_id, ca1.cluster_id as old_cluster_id, ca2.cluster_id as new_cluster_id
                FROM cluster_assignments ca1
                JOIN cluster_assignments ca2 ON ca1.customer_id = ca2.customer_id
                WHERE ca1.period_id = %s AND ca2.period_id = %s;
                """,
                (T, T + 1)
            )
            rows = cur.fetchall()
            
    if not rows:
        logger.warning(f"No matching customer assignments found between period {T} and {T+1}.")
        return pd.DataFrame(), pd.Series()
        
    df_labels = pd.DataFrame(rows, columns=["customer_id", "old_cluster_id", "new_cluster_id"])
    df_labels["customer_id"] = df_labels["customer_id"].astype(str)
    
    # 2. Fetch identity mappings to verify structural transitions
    with get_raw_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT old_cluster_id, new_cluster_id, match_type FROM cluster_identity_map;")
            mappings_rows = cur.fetchall()
            
    valid_mappings = {}
    disappeared_clusters = set()
    new_clusters = set()
    
    for old_cid, new_cid, m_type in mappings_rows:
        if m_type == "disappeared" and old_cid is not None:
            disappeared_clusters.add(old_cid)
        elif m_type == "new" and new_cid is not None:
            new_clusters.add(new_cid)
        elif m_type in ("matched", "split", "merged") and old_cid is not None and new_cid is not None:
            if old_cid not in valid_mappings:
                valid_mappings[old_cid] = set()
            valid_mappings[old_cid].add(new_cid)
            
    def compute_migration_label(row):
        old_cid = row["old_cluster_id"]
        new_cid = row["new_cluster_id"]
        
        # Scenario A: Assignments are identical
        if old_cid == new_cid:
            return False
            
        # Scenario B: The old cluster disappeared entirely
        if old_cid in disappeared_clusters:
            return True
            
        # Scenario C: The customer landed in a brand-new cluster at T+1
        if new_cid in new_clusters:
            return True
            
        # Scenario D: Check if there is an identity mapping old_cid -> new_cid
        if old_cid in valid_mappings and new_cid in valid_mappings[old_cid]:
            return False
            
        # Scenario E: Any other mismatch is a genuine migration
        return True
        
    df_labels["migrated"] = df_labels.apply(compute_migration_label, axis=1)
    
    # Persist computed customer-level migration labels to the database
    with get_raw_connection() as conn:
        with conn.cursor() as cur:
            for _, r in df_labels.iterrows():
                cur.execute(
                    """
                    INSERT INTO migration_labels (
                        customer_id, from_period_id, to_period_id,
                        from_cluster_id, to_cluster_id, migrated
                    )
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (customer_id, from_period_id, to_period_id)
                    DO UPDATE SET
                        from_cluster_id = EXCLUDED.from_cluster_id,
                        to_cluster_id = EXCLUDED.to_cluster_id,
                        migrated = EXCLUDED.migrated;
                    """,
                    (
                        r["customer_id"],
                        T,
                        T + 1,
                        int(r["old_cluster_id"]) if pd.notna(r["old_cluster_id"]) else None,
                        int(r["new_cluster_id"]) if pd.notna(r["new_cluster_id"]) else None,
                        bool(r["migrated"])
                    )
                )
    
    # 3. Pull features strictly from period T (temporal leakage guardrail)
    with get_raw_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = 'behavioral_snapshots';
                """
            )
            cols = [r[0] for r in cur.fetchall()]
            
    non_feature_cols = {"snapshot_id", "customer_id", "period_id", "captured_at"}
    feature_cols = [c for c in cols if c not in non_feature_cols]
    
    with get_raw_connection() as conn:
        with conn.cursor() as cur:
            cols_str = ", ".join([f'"{c}"' for c in ["customer_id"] + feature_cols])
            cur.execute(
                f"SELECT {cols_str} FROM behavioral_snapshots WHERE period_id = %s;",
                (T,)
            )
            snapshot_rows = cur.fetchall()
            
    if not snapshot_rows:
        logger.warning(f"No behavioral snapshots found for period {T}.")
        return pd.DataFrame(), pd.Series()
        
    df_features = pd.DataFrame(snapshot_rows, columns=["customer_id"] + feature_cols)
    df_features["customer_id"] = df_features["customer_id"].astype(str)
    
    # Fill any missing values in behavioral snapshots with zero (or average) to prevent training issues
    df_features = df_features.fillna(0.0)
    
    # 4. Merge labels and features
    merged = pd.merge(df_features, df_labels, on="customer_id", how="inner")
    if merged.empty:
        logger.warning("Merged training dataset is empty. Check mapping keys alignment.")
        return pd.DataFrame(), pd.Series()
        
    # Sort by customer_id to guarantee deterministic ordering
    merged = merged.sort_values("customer_id").reset_index(drop=True)
        
    # Exclude non-feature columns from training matrix X
    X = merged[feature_cols].copy()
    y = merged["migrated"].copy()
    
    logger.info(f"Dataset constructed: X shape = {X.shape}, class distribution = {y.value_counts().to_dict()}")
    return X, y
