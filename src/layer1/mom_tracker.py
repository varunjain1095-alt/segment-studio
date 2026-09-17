import logging
import numpy as np
import pandas as pd
from src.db.connection import get_raw_connection
from src.utils.logging_config import setup_logging

setup_logging()
logger = logging.getLogger("mom_tracker")

def populate_demographic_snapshots(df: pd.DataFrame, period_id: int, demographic_cols: list[str]):
    """
    Inserts raw customer demographic records into demographic_snapshots.
    Ensures customers exist in the customers table first.
    """
    logger.info(f"Populating demographic snapshots for period {period_id}...")
    
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
                
                # Build dynamic insert columns and values
                cols = ["customer_id", "period_id"]
                vals = [cid, period_id]
                placeholders = ["%s", "%s"]
                
                for col in demographic_cols:
                    if col in df.columns:
                        cols.append(col)
                        val = row[col]
                        # Handle NaNs
                        if pd.isna(val):
                            vals.append(None)
                        else:
                            vals.append(val)
                        placeholders.append("%s")
                        
                insert_query = f"""
                    INSERT INTO demographic_snapshots ({", ".join(cols)})
                    VALUES ({", ".join(placeholders)})
                    ON CONFLICT (customer_id, period_id) DO UPDATE
                    SET {", ".join([f"{c} = EXCLUDED.{c}" for c in cols if c not in ["customer_id", "period_id"]])};
                """
                cur.execute(insert_query, tuple(vals))
                
    logger.info("Demographic snapshots successfully populated.")

def compute_and_persist_mom_metrics(dataset_id: str, period_id: int, prev_period_id: int, demographic_cols: list[str]):
    """
    Computes segment sizes for the current period, MoM percentage changes relative to the
    previous period, and updates demographic_segment_metrics.
    Supports case-insensitive column discovery and automatic age cohort binning.
    """
    logger.info(f"Computing MoM metrics for period {period_id} vs {prev_period_id}...")
    
    with get_raw_connection() as conn:
        with conn.cursor() as cur:
            # Query existing column names in demographic_snapshots to avoid casing errors
            cur.execute("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = 'demographic_snapshots';
            """)
            db_cols = {row[0].lower(): row[0] for row in cur.fetchall()}

        # Helper to query segment value counts for a period
        def query_counts(pid: int):
            counts = {}
            if pid is None:
                return counts
            with conn.cursor() as cur:
                for col in demographic_cols:
                    col_lower = col.lower()
                    try:
                        if col_lower in ("age", "age_bracket"):
                            field_key = "Age Bracket"
                            # Coalesce whichever column has data: Age or age
                            cur.execute(f'''
                                SELECT 
                                    CASE 
                                        WHEN COALESCE("Age", age::numeric) < 26 THEN '18-25'
                                        WHEN COALESCE("Age", age::numeric) < 36 THEN '26-35'
                                        WHEN COALESCE("Age", age::numeric) < 51 THEN '36-50'
                                        ELSE '51+'
                                    END AS age_bracket,
                                    COUNT(*)
                                FROM demographic_snapshots
                                WHERE period_id = %s AND (("Age" IS NOT NULL) OR (age IS NOT NULL))
                                GROUP BY 1;
                            ''', (pid,))
                            counts[field_key] = {str(row[0]): int(row[1]) for row in cur.fetchall() if row[0] is not None}
                        elif col_lower == "gender":
                            field_key = "Gender"
                            cur.execute(f'''
                                SELECT COALESCE("Gender", gender) AS gender_val, COUNT(*)
                                FROM demographic_snapshots
                                WHERE period_id = %s AND (("Gender" IS NOT NULL) OR (gender IS NOT NULL))
                                GROUP BY 1;
                            ''', (pid,))
                            counts[field_key] = {str(row[0]): int(row[1]) for row in cur.fetchall() if row[0] is not None}
                        elif col_lower in ("city", "location", "region"):
                            field_key = "City"
                            cur.execute(f'''
                                SELECT "City", COUNT(*)
                                FROM demographic_snapshots
                                WHERE period_id = %s AND "City" IS NOT NULL
                                GROUP BY 1;
                            ''', (pid,))
                            counts[field_key] = {str(row[0]): int(row[1]) for row in cur.fetchall() if row[0] is not None}
                        else:
                            field_key = col
                            cur.execute(f'SELECT "{col}", COUNT(*) FROM demographic_snapshots WHERE period_id = %s AND "{col}" IS NOT NULL GROUP BY "{col}";', (pid,))
                            counts[field_key] = {str(row[0]): int(row[1]) for row in cur.fetchall() if row[0] is not None}
                    except Exception as e:
                        logger.warning(f"Failed to query counts for column {col} in period {pid}: {str(e)}")
            return counts

        current_counts = query_counts(period_id)
        prev_counts = query_counts(prev_period_id)
                        
        with conn.cursor() as cur:
            for field_name, val_dict in current_counts.items():
                for val, size_T in val_dict.items():
                    val_str = str(val)
                    size_T_minus_1 = prev_counts.get(field_name, {}).get(val, 0) if prev_counts else 0
                    
                    mom_change = None
                    # Minimum segment size gate (N >= 30) to prevent tiny denominator volatility flags.
                    if size_T >= 30 and size_T_minus_1 >= 30:
                        mom_change = ((size_T - size_T_minus_1) / size_T_minus_1) * 100.0
                        
                    # 1. Compute historical std dev of MoM changes for this segment
                    cur.execute(
                        """
                        SELECT mom_pct_change 
                        FROM demographic_segment_metrics
                        WHERE dataset_id = %s AND segment_field = %s AND segment_value = %s AND period_id != %s
                        AND mom_pct_change IS NOT NULL;
                        """,
                        (dataset_id, field_name, val_str, period_id)
                    )
                    past_changes = [float(row[0]) for row in cur.fetchall()]
                    
                    # Add current change if not null to calculate updated standard deviation
                    all_changes = list(past_changes)
                    if mom_change is not None:
                        all_changes.append(mom_change)
                        
                    std_dev = None
                    if len(all_changes) >= 2:
                        std_dev = float(np.std(all_changes, ddof=1))
                    elif len(all_changes) == 1:
                        std_dev = 0.0
                        
                    # 2. Persist to DB
                    cur.execute(
                        """
                        INSERT INTO demographic_segment_metrics (
                            dataset_id, period_id, segment_field, segment_value, segment_size,
                            prev_segment_size, mom_pct_change, historical_std_dev
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (dataset_id, period_id, segment_field, segment_value) DO UPDATE
                        SET segment_size = EXCLUDED.segment_size,
                            prev_segment_size = EXCLUDED.prev_segment_size,
                            mom_pct_change = EXCLUDED.mom_pct_change,
                            historical_std_dev = EXCLUDED.historical_std_dev,
                            computed_at = now();
                        """,
                        (
                            dataset_id, period_id, field_name, val_str, size_T,
                            size_T_minus_1 if size_T_minus_1 > 0 else None,
                            mom_change, std_dev
                        )
                    )
                    
    logger.info("MoM metrics successfully persisted.")
