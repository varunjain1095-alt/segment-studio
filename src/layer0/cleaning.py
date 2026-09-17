import logging
import pandas as pd
import numpy as np
from src.layer0.profiler import data_profile, save_profile_to_db
from src.notifications.center import emit
from src.utils.logging_config import setup_logging

setup_logging()
logger = logging.getLogger("cleaning")

def clean_dataset(df: pd.DataFrame, dataset_id: str) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """
    Performs type recovery, scans for negative values, detects duplicate rows.
    Returns:
        cleaned_df (pd.DataFrame): The modified working DataFrame.
        profile_df (pd.DataFrame): The final post-cleaning data profile DataFrame.
        duplicate_count (int): Absolute count of exact duplicate rows.
    """
    logger.info("Starting dataset cleaning...")
    
    # 1. First profiling pass
    profile_df = data_profile(df)
    
    # 2. Type Recovery
    suspect_cols = profile_df[profile_df["suspect_numeric_object"] == True]["column_name"].tolist()
    if suspect_cols:
        logger.info(f"Suspect numeric object columns found: {suspect_cols}. Performing type recovery...")
        for col_name in suspect_cols:
            df[col_name] = pd.to_numeric(df[col_name], errors='coerce')
        # Re-run profiling after type coercion
        profile_df = data_profile(df)
        
    # Save/update the final profile in the DB
    save_profile_to_db(dataset_id, profile_df)
    
    # 3. Negative Value Flagging
    # Unconditionally flags any numeric column containing any negative values
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col_name in numeric_cols:
        # Check if any value is negative (skipping NaNs)
        has_negative = (df[col_name] < 0).any()
        if has_negative:
            logger.warning(f"Unconditional negative value check failed for column: {col_name}")
            emit(
                source="layer0_cleaning",
                condition="negative_value_flagged",
                severity="soft_warning",
                dataset_type="file_upload",
                details=f"Column '{col_name}' contains negative values. No per-column exceptions allowed."
            )
            
    # 4. Duplicate Row Detection (100% exact full-row match only)
    duplicate_mask = df.duplicated(keep='first')
    duplicate_count = int(duplicate_mask.sum())
    logger.info(f"Duplicate row detection: found {duplicate_count} exact duplicate rows.")
    
    return df, profile_df, duplicate_count
