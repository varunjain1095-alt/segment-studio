import logging
from src.db.connection import get_raw_connection

logger = logging.getLogger("fallback_state_manager")

AVAILABLE_FALLBACK = [
    "L1 Demographic", "L3 Statistical Testing", "L6 Outcome Analysis",
    "L7 Metric Integration", "A1 Narration", "Ask Your Data",
    "Create Custom Segments", "L8 Feature Signal"
]

DISABLED_FALLBACK = [
    "L2 Behavioral", "L4 Predictive", "L5 Migration"
]

ALL_LAYERS = [
    "L1 Demographic", "L2 Behavioral", "L3 Statistical Testing",
    "L4 Predictive", "L5 Migration", "L6 Outcome Analysis",
    "L7 Metric Integration", "A1 Narration", "Ask Your Data",
    "Create Custom Segments", "L8 Feature Signal"
]

def get_fallback_state(dataset_id: str) -> dict:
    """
    Retrieves the fallback status for a given dataset, listing the available
    and disabled layers depending on whether fallback mode is active.
    """
    logger.info(f"Retrieving fallback state for dataset {dataset_id}")
    
    with get_raw_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT fallback_active FROM datasets WHERE dataset_id = %s;", (dataset_id,))
            row = cur.fetchone()
            
    if not row:
        raise ValueError(f"Dataset {dataset_id} not found.")
        
    fallback_active = bool(row[0])
    
    if fallback_active:
        return {
            "fallback_active": True,
            "available_layers": AVAILABLE_FALLBACK,
            "disabled_layers": DISABLED_FALLBACK
        }
    else:
        return {
            "fallback_active": False,
            "available_layers": ALL_LAYERS,
            "disabled_layers": []
        }
