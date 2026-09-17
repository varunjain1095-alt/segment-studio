import json
import logging
import numpy as np
import pandas as pd
from src.db.connection import get_raw_connection
from src.notifications.center import emit

logger = logging.getLogger("layer8_migration_signal")

def confirm_layer8_columns(dataset_id: str, columns: list[dict], confirmed_by: str = None) -> list[str]:
    """
    PM confirms which behavioral columns are used for feature-level migration comparison.
    Validates that the dataset exists and the columns are classified as behavioral.
    """
    logger.info(f"Confirming Layer 8 columns for dataset {dataset_id}...")
    
    # 1. Verify dataset exists and retrieve confirmed field classifications
    with get_raw_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT confirmed_classifications 
                FROM datasets 
                WHERE dataset_id = %s;
                """,
                (dataset_id,)
            )
            row = cur.fetchone()
            
    if not row:
        raise ValueError(f"Dataset {dataset_id} not found in database.")
        
    confirmed_classifications_json = row[0]
    if not confirmed_classifications_json:
        raise ValueError(f"Dataset {dataset_id} has no confirmed column classifications. Run Layer 1 classification first.")
        
    if isinstance(confirmed_classifications_json, str):
        classifications = json.loads(confirmed_classifications_json)
    else:
        classifications = confirmed_classifications_json
        
    # Extract behavioral column names
    behavioral_cols = {
        item["column"] for item in classifications
        if item.get("classification") == "behavioral"
    }
    
    # 2. Validate columns from input payload
    confirmed_cols_list = []
    for col_info in columns:
        col_name = col_info.get("column_name")
        include = col_info.get("include", False)
        if include:
            if col_name not in behavioral_cols:
                raise ValueError(
                    f"Column '{col_name}' is not confirmed as a behavioral feature for dataset {dataset_id}."
                )
            confirmed_cols_list.append(col_name)
            
    if not confirmed_cols_list:
        raise ValueError("At least one behavioral column must be selected/included for Layer 8 configuration.")
        
    # 3. Persist confirmed column set to layer8_config table
    confirmed_columns_json = json.dumps(confirmed_cols_list)
    with get_raw_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO layer8_config (dataset_id, confirmed_columns, confirmed_by)
                VALUES (%s, %s, %s)
                ON CONFLICT (dataset_id) DO UPDATE
                SET confirmed_columns = EXCLUDED.confirmed_columns,
                    confirmed_at = now(),
                    confirmed_by = EXCLUDED.confirmed_by;
                """,
                (dataset_id, confirmed_columns_json, confirmed_by)
            )
            
    logger.info(f"Layer 8 columns successfully confirmed: {confirmed_cols_list}")
    return confirmed_cols_list

def compute_and_persist_feature_migration_signals(dataset_id: str, period_id: int) -> int:
    """
    Computes early-warning drift signal based on absolute median comparison 
    across the PM-confirmed behavioral column set for each customer.
    
    If no PM column confirmation exists (bypassed gate), raises PipelineBlockedError
    via notification center.
    """
    logger.info(f"Computing Layer 8 migration signals for dataset {dataset_id}, period {period_id}...")
    
    # 1. Retrieve confirmed columns from layer8_config (P8-T5 Gate Check)
    with get_raw_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT confirmed_columns 
                FROM layer8_config 
                WHERE dataset_id = %s;
                """,
                (dataset_id,)
            )
            row = cur.fetchone()
            
    if not row or not row[0]:
        msg = f"Layer 8 columns have not been confirmed for dataset {dataset_id}. Signal computation blocked."
        logger.error(msg)
        emit(
            source="layer8_migration_signal",
            condition="layer8_columns_not_confirmed",
            severity="hard_fail",
            dataset_type="file_upload",
            details=msg
        )
        # emit raises PipelineBlockedError if severity is hard_fail, but fallback:
        from src.notifications.center import PipelineBlockedError
        raise PipelineBlockedError(msg)
        
    confirmed_cols = json.loads(row[0]) if isinstance(row[0], str) else row[0]
    if not confirmed_cols:
        raise ValueError("Confirmed column list for Layer 8 configuration is empty.")
        
    # 2. Retrieve customer assignments and behavioral snapshot values for period_id
    with get_raw_connection() as conn:
        with conn.cursor() as cur:
            cols_str = ", ".join([f'bs."{col}"' for col in confirmed_cols])
            cur.execute(
                f"""
                SELECT ca.customer_id, ca.cluster_id, {cols_str}
                FROM cluster_assignments ca
                JOIN behavioral_snapshots bs ON ca.customer_id = bs.customer_id AND ca.period_id = bs.period_id
                WHERE ca.period_id = %s;
                """,
                (period_id,)
            )
            rows = cur.fetchall()
            if not rows:
                logger.info(f"No assignments found for period {period_id}. Skipping signal generation.")
                return 0
            col_names = [desc[0] for desc in cur.description]
            
    df = pd.DataFrame(rows, columns=col_names)
    
    # Ensure all confirmed columns are present in df
    missing_cols = [c for c in confirmed_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing behavioral snapshot columns for Layer 8 computation: {missing_cols}")
        
    # 3. Compute medians for each cluster
    # Group by cluster_id and compute median (P8-T4: median, not mean)
    medians = df.groupby("cluster_id")[confirmed_cols].median()
    
    unique_clusters = df["cluster_id"].dropna().unique().tolist()
    if len(unique_clusters) < 2:
        logger.info("Fewer than 2 clusters found. Skipping drift comparison since no comparison cluster exists.")
        return 0
        
    signals_to_persist = []
    
    # 4. For each customer S, check against all other clusters S'
    for _, c_row in df.iterrows():
        customer_id = c_row["customer_id"]
        own_cluster_id = int(c_row["cluster_id"])
        
        for compared_cluster_id in unique_clusters:
            compared_cluster_id = int(compared_cluster_id)
            if compared_cluster_id == own_cluster_id:
                continue
                
            columns_drifted = 0
            for col in confirmed_cols:
                val = float(c_row[col])
                med_own = float(medians.loc[own_cluster_id, col])
                med_compare = float(medians.loc[compared_cluster_id, col])
                
                d_own = abs(val - med_own)
                d_compare = abs(val - med_compare)
                
                # drift is True strictly when d_compare < d_own (closer to comparison cluster)
                if d_compare < d_own:
                    columns_drifted += 1
                    
            drift_fraction = float(columns_drifted) / len(confirmed_cols)
            # signal_raised is True strictly if drift_fraction > 0.50 (P8-T3 boundary: exactly 50% is False)
            signal_raised = drift_fraction > 0.50
            
            signals_to_persist.append((
                customer_id,
                period_id,
                own_cluster_id,
                compared_cluster_id,
                len(confirmed_cols),
                columns_drifted,
                drift_fraction,
                signal_raised
            ))
            
    # 5. Persist signals to feature_migration_signals table
    with get_raw_connection() as conn:
        with conn.cursor() as cur:
            # Delete old signals for this period to guarantee idempotency
            cur.execute("DELETE FROM feature_migration_signals WHERE period_id = %s;", (period_id,))
            
            if signals_to_persist:
                from psycopg2.extras import execute_values
                execute_values(
                    cur,
                    """
                    INSERT INTO feature_migration_signals (
                        customer_id, period_id, own_cluster_id, compared_cluster_id,
                        columns_checked, columns_drifted, drift_fraction, signal_raised
                    )
                    VALUES %s;
                    """,
                    signals_to_persist,
                    page_size=2000
                )
                
    logger.info(f"Layer 8 signal computation completed. Persisted {len(signals_to_persist)} signals.")
    return len(signals_to_persist)
