import logging
from src.db.connection import get_raw_connection
from src.notifications.center import emit
from src.utils.logging_config import setup_logging

setup_logging()
logger = logging.getLogger("anomaly_flagger")

def flag_anomalies_for_period(dataset_id: str, period_id: int):
    """
    Computes volatility-aware anomalies for demographic segments in a given period.
    Flags segment changes exceeding their historical 2-sigma noise band and emits soft warnings.
    """
    logger.info(f"Checking for demographic segment size anomalies in period {period_id}...")
    
    with get_raw_connection() as conn:
        with conn.cursor() as cur:
            # 1. Fetch metrics for current period
            cur.execute(
                """
                SELECT segment_field, segment_value, mom_pct_change, historical_std_dev
                FROM demographic_segment_metrics
                WHERE dataset_id = %s AND period_id = %s;
                """,
                (dataset_id, period_id)
            )
            records = cur.fetchall()
            
            for field, val, mom_change, std_dev in records:
                if mom_change is None:
                    continue
                    
                # To establish a 2-sigma noise band, historical_std_dev must be present and > 0.
                if std_dev is not None and float(std_dev) > 0.0:
                    limit = 2.0 * float(std_dev)
                    if abs(float(mom_change)) > limit:
                        msg = f"Demographic segment '{field}'='{val}' in period {period_id} showed anomalous MoM change of {float(mom_change):.2f}% (exceeds 2-sigma noise band of +/- {limit:.2f}%)."
                        logger.warning(msg)
                        
                        # Update DB
                        cur.execute(
                            """
                            UPDATE demographic_segment_metrics
                            SET anomaly_flagged = TRUE
                            WHERE dataset_id = %s AND period_id = %s AND segment_field = %s AND segment_value = %s;
                            """,
                            (dataset_id, period_id, field, val)
                        )
                        
                        # Emit soft warning
                        emit(
                            source="layer1_anomaly_flagger",
                            condition="demographic_segment_anomaly_detected",
                            severity="soft_warning",
                            dataset_type="file_upload",
                            details=msg
                        )
                        
    logger.info("Anomaly checks completed.")
