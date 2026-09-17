import os
import json
import logging
import pandas as pd
import numpy as np
from src.db.connection import get_raw_connection
from src.notifications.center import emit
from src.ingestion.router import DATA_DIR

logger = logging.getLogger("layer2_preprocessor")

def load_and_preprocess_behavioral_data(dataset_id: str) -> tuple[pd.DataFrame, list[str]]:
    """
    Loads the Layer 0 processed dataset, selects columns classified as 'behavioral'
    by the PM, and verifies the following constraints:
      1. layer0_completed = TRUE in datasets table.
      2. No NaN values in the behavioral features.
      3. All behavioral features are fully numeric.
    Returns:
      X_df (pd.DataFrame): DataFrame containing 'customer_id' and behavioral columns.
      behavioral_cols (list[str]): List of behavioral column names.
    """
    logger.info(f"Layer 2.0: Starting pre-clustering preprocessing for dataset {dataset_id}...")
    
    # 1. State Database Verification
    with get_raw_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT layer0_completed, confirmed_classifications
                FROM datasets
                WHERE dataset_id = %s;
                """,
                (dataset_id,)
            )
            row = cur.fetchone()
            
    if not row:
        raise ValueError(f"Dataset {dataset_id} not found in database.")
        
    layer0_completed, confirmed_classifications_json = row
    
    if not layer0_completed:
        msg = f"Dataset {dataset_id} has not successfully completed Layer 0. Processing blocked."
        logger.error(msg)
        emit(
            source="layer2_preprocessor",
            condition="layer0_not_completed",
            severity="hard_fail",
            dataset_type="file_upload",
            details=msg
        )
        raise ValueError(msg)
        
    # 2. Extract PM-confirmed behavioral columns
    if not confirmed_classifications_json:
        msg = f"Dataset {dataset_id} has no confirmed column classifications. Run Layer 1 classification first."
        logger.error(msg)
        emit(
            source="layer2_preprocessor",
            condition="no_confirmed_classifications",
            severity="hard_fail",
            dataset_type="file_upload",
            details=msg
        )
        raise ValueError(msg)
        
    if isinstance(confirmed_classifications_json, str):
        classifications = json.loads(confirmed_classifications_json)
    else:
        classifications = confirmed_classifications_json
        
    behavioral_cols = [
        item["column"] for item in classifications
        if item.get("classification") == "behavioral"
    ]
    
    if not behavioral_cols:
        msg = f"Dataset {dataset_id} has zero columns confirmed as behavioral features."
        logger.error(msg)
        emit(
            source="layer2_preprocessor",
            condition="no_behavioral_features",
            severity="hard_fail",
            dataset_type="file_upload",
            details=msg
        )
        raise ValueError(msg)
        
    # 3. Load processed dataset file
    processed_path = os.path.join(DATA_DIR, f"{dataset_id}_processed.csv")
    if not os.path.exists(processed_path):
        msg = f"Processed dataset file not found at path: {processed_path}"
        logger.error(msg)
        emit(
            source="layer2_preprocessor",
            condition="processed_file_missing",
            severity="hard_fail",
            dataset_type="file_upload",
            details=msg
        )
        raise FileNotFoundError(msg)
        
    df = pd.read_csv(processed_path)
    
    if "customer_id" not in df.columns:
        msg = "Processed dataset is missing required 'customer_id' column."
        logger.error(msg)
        emit(
            source="layer2_preprocessor",
            condition="customer_id_missing",
            severity="hard_fail",
            dataset_type="file_upload",
            details=msg
        )
        raise ValueError(msg)
        
    missing_cols = [col for col in behavioral_cols if col not in df.columns]
    if missing_cols:
        msg = f"Processed dataset is missing confirmed behavioral columns: {missing_cols}"
        logger.error(msg)
        emit(
            source="layer2_preprocessor",
            condition="missing_behavioral_columns",
            severity="hard_fail",
            dataset_type="file_upload",
            details=msg
        )
        raise ValueError(msg)
        
    # Create feature matrix containing customer_id and behavioral columns
    X_df = df[["customer_id"] + behavioral_cols].copy()
    
    # 4. Strict NaN and Type Verification
    for col in behavioral_cols:
        if X_df[col].isna().any():
            msg = f"Behavioral column '{col}' contains missing (NaN) values. Complete imputation is required."
            logger.error(msg)
            emit(
                source="layer2_preprocessor",
                condition="nan_values_detected",
                severity="hard_fail",
                dataset_type="file_upload",
                details=msg
            )
            raise ValueError(msg)
            
        if not pd.api.types.is_numeric_dtype(X_df[col]):
            msg = f"Behavioral column '{col}' is not fully numeric."
            logger.error(msg)
            emit(
                source="layer2_preprocessor",
                condition="non_numeric_behavioral_column",
                severity="hard_fail",
                dataset_type="file_upload",
                details=msg
            )
            raise ValueError(msg)
            
    logger.info(f"Layer 2.0: Successfully preprocessed dataset {dataset_id} with {len(behavioral_cols)} features and {len(X_df)} rows.")
    return X_df, behavioral_cols

def populate_behavioral_snapshots(df: pd.DataFrame, period_id: int, behavioral_cols: list[str]):
    """
    Inserts customer behavioral records into behavioral_snapshots table.
    Ensures customer IDs exist in the customers table first.
    """
    logger.info(f"Layer 2.0: Populating behavioral snapshots for period {period_id}...")
    
    with get_raw_connection() as conn:
        with conn.cursor() as cur:
            # 1. Ensure all customer IDs exist in customers table
            customer_ids = df["customer_id"].unique().tolist()
            for cid in customer_ids:
                cur.execute(
                    """
                    INSERT INTO customers (customer_id)
                    VALUES (%s)
                    ON CONFLICT (customer_id) DO NOTHING;
                    """,
                    (cid,)
                )
                
            # 2. Insert snapshots
            for _, row in df.iterrows():
                cid = row["customer_id"]
                
                cols = ["customer_id", "period_id"]
                vals = [cid, period_id]
                placeholders = ["%s", "%s"]
                
                for col in behavioral_cols:
                    if col in df.columns:
                        cols.append(col)
                        val = row[col]
                        if pd.isna(val):
                            vals.append(None)
                        else:
                            if isinstance(val, (np.integer, np.floating)):
                                vals.append(val.item())
                            else:
                                vals.append(val)
                        placeholders.append("%s")
                        
                insert_query = f"""
                    INSERT INTO behavioral_snapshots ({", ".join([f'"{c}"' for c in cols])})
                    VALUES ({", ".join(placeholders)})
                    ON CONFLICT (customer_id, period_id) DO UPDATE
                    SET {", ".join([f'"{c}" = EXCLUDED."{c}"' for c in cols if c not in ["customer_id", "period_id"]])};
                """
                cur.execute(insert_query, tuple(vals))
                
    logger.info("Layer 2.0: Behavioral snapshots successfully populated.")
