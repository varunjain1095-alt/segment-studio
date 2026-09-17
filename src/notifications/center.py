import logging
from src.db.connection import get_raw_connection
from src.utils.logging_config import setup_logging

setup_logging()
logger = logging.getLogger("notifications")

class PipelineBlockedError(Exception):
    """Custom exception raised when a hard fail is emitted in the pipeline."""
    pass

def emit(source: str, condition: str, severity: str, dataset_type: str = None, details: str = None):
    """
    Central entry point to log and persist notifications.
    
    If severity is 'hard_fail', raises PipelineBlockedError to stop execution.
    If severity is 'soft_warning', logs the warning and proceeds.
    """
    if severity not in ["hard_fail", "soft_warning"]:
        raise ValueError("Invalid severity. Allowed: 'hard_fail', 'soft_warning'")

    logger.info(
        f"Emitting notification: source={source}, condition={condition}, severity={severity}",
        extra={"source": source, "condition": condition, "severity": severity, "dataset_type": dataset_type, "details": details}
    )

    try:
        with get_raw_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO notifications (source, condition, severity, dataset_type, details, resolved)
                    VALUES (%s, %s, %s, %s, %s, FALSE)
                    RETURNING notification_id;
                    """,
                    (source, condition, severity, dataset_type, details)
                )
                notification_id = cur.fetchone()[0]
                logger.info(f"Notification persisted with ID: {notification_id}")
    except Exception as e:
        logger.error(f"Failed to persist notification to DB: {str(e)}", exc_info=True)
        # We don't want DB failure to mask a hard fail, so we proceed to raising if it is a hard fail.

    if severity == "hard_fail":
        raise PipelineBlockedError(f"Pipeline execution blocked by hard fail from '{source}': {details}")
