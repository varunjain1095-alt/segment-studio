import logging
import numpy as np
import pandas as pd
from src.db.connection import get_raw_connection
from src.utils.logging_config import setup_logging

setup_logging()
logger = logging.getLogger("profiler")

def data_profile(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes profiling metrics for each column of the input DataFrame.
    Returns a DataFrame with one row per column, sorted descending by missing_pct.
    """
    total_rows = len(df)
    profile_rows = []
    
    for col_name in df.columns:
        col = df[col_name]
        
        missing_count = int(col.isna().sum())
        missing_pct = round((missing_count / total_rows * 100), 2) if total_rows > 0 else 0.0
        
        dtype_str = str(col.dtype)
        
        # Check suspect_numeric_object
        suspect_numeric_object = False
        if col.dtype == object or dtype_str in ['object', 'O', 'str'] or dtype_str.startswith('string') or dtype_str.startswith('str'):
            non_null_vals = col.dropna()
            if len(non_null_vals) > 0:
                coerced = pd.to_numeric(non_null_vals, errors='coerce')
                pct_numeric = coerced.notna().mean()
                if pct_numeric > 0.90:
                    suspect_numeric_object = True
                    
        unique_count = int(col.nunique(dropna=True))
        non_unique_count = int(total_rows - unique_count)
        non_unique_pct = round((non_unique_count / total_rows * 100), 2) if total_rows > 0 else 0.0
        
        is_unique_key = bool(unique_count == total_rows)
        has_duplicates = bool(unique_count < total_rows)
        
        # Assign severity based on missing_pct
        if missing_pct == 0.0:
            severity = "None"
        elif missing_pct <= 25.0:
            severity = "Low"
        elif missing_pct <= 50.0:
            severity = "Moderate"
        elif missing_pct <= 75.0:
            severity = "High"
        else:
            severity = "Critical"
            
        profile_rows.append({
            "column_name": col_name,
            "missing_count": missing_count,
            "missing_pct": missing_pct,
            "dtype": dtype_str,
            "suspect_numeric_object": suspect_numeric_object,
            "unique_count": unique_count,
            "non_unique_count": non_unique_count,
            "non_unique_pct": non_unique_pct,
            "is_unique_key": is_unique_key,
            "has_duplicates": has_duplicates,
            "severity": severity
        })
        
    profile_df = pd.DataFrame(profile_rows)
    if not profile_df.empty:
        profile_df = profile_df.sort_values(by="missing_pct", ascending=False).reset_index(drop=True)
    return profile_df

def save_profile_to_db(dataset_id: str, profile_df: pd.DataFrame):
    """
    Saves a profile DataFrame to the column_profiles table in Postgres.
    """
    logger.info(f"Saving column profiles for dataset {dataset_id} to database...")
    with get_raw_connection() as conn:
        with conn.cursor() as cur:
            for _, row in profile_df.iterrows():
                cur.execute(
                    """
                    INSERT INTO column_profiles (
                        dataset_id, column_name, missing_count, missing_pct, dtype,
                        suspect_numeric_object, unique_count, non_unique_count,
                        non_unique_pct, is_unique_key, has_duplicates, severity
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (dataset_id, column_name) DO UPDATE
                    SET missing_count = EXCLUDED.missing_count,
                        missing_pct = EXCLUDED.missing_pct,
                        dtype = EXCLUDED.dtype,
                        suspect_numeric_object = EXCLUDED.suspect_numeric_object,
                        unique_count = EXCLUDED.unique_count,
                        non_unique_count = EXCLUDED.non_unique_count,
                        non_unique_pct = EXCLUDED.non_unique_pct,
                        is_unique_key = EXCLUDED.is_unique_key,
                        has_duplicates = EXCLUDED.has_duplicates,
                        severity = EXCLUDED.severity,
                        profiled_at = now();
                    """,
                    (
                        dataset_id, row["column_name"], row["missing_count"], row["missing_pct"],
                        row["dtype"], row["suspect_numeric_object"], row["unique_count"],
                        row["non_unique_count"], row["non_unique_pct"], row["is_unique_key"],
                        row["has_duplicates"], row["severity"]
                    )
                )
    logger.info("Profiles successfully saved.")
