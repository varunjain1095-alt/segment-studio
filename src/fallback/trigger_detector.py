import logging
import json
import numpy as np
import pandas as pd
from src.db.connection import get_raw_connection
from src.layer2.preprocessor import load_and_preprocess_behavioral_data
from src.layer2.clusterer import run_kmeans_clustering, run_gmm_clustering
from src.notifications.center import emit

logger = logging.getLogger("fallback_trigger_detector")

def _evaluate_labels(labels: np.ndarray) -> dict:
    """
    Evaluates cluster labels to determine:
    1. Number of active clusters (clusters with >= 1 samples).
    2. Details of counts and percentages.
    3. Whether it meets Condition 1 (fewer than 3 active clusters).
    4. Whether it meets Condition 2 (exactly 2 active clusters where one >= 85% and the other <= 15%).
    """
    unique_labels, counts = np.unique(labels, return_counts=True)
    active_clusters = len(unique_labels)
    total = len(labels)
    
    percentages = {int(l): float(counts[i] / total) for i, l in enumerate(unique_labels)}
    counts_dict = {int(l): int(counts[i]) for i, l in enumerate(unique_labels)}
    
    cond1 = active_clusters < 3
    cond2 = False
    if active_clusters == 2:
        pcts = list(percentages.values())
        if (pcts[0] >= 0.85 and pcts[1] <= 0.15) or (pcts[1] >= 0.85 and pcts[0] <= 0.15):
            cond2 = True
            
    return {
        "active_clusters": active_clusters,
        "counts": counts_dict,
        "percentages": percentages,
        "cond1": cond1,
        "cond2": cond2
    }

def evaluate(dataset_id: str, period_id: int = None) -> bool:
    """
    Executes K-means and GMM clustering, checks if they both independently
    confirm poor cluster structure, and sets fallback_active = TRUE if so.
    
    Returns:
        fallback_active (bool): True if fallback was triggered, False otherwise.
    """
    logger.info(f"Evaluating fallback triggers for dataset {dataset_id}...")
    
    # 1. Fetch behavioral columns confirmed classifications
    with get_raw_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT confirmed_classifications FROM datasets WHERE dataset_id = %s;", (dataset_id,))
            res = cur.fetchone()
            
    if not res or not res[0]:
        logger.warning(f"No confirmed classifications for dataset {dataset_id}. Skipping trigger evaluation.")
        return False
        
    classifications = json.loads(res[0]) if isinstance(res[0], str) else res[0]
    behavioral_cols = [c["column"] for c in classifications if c.get("classification") == "behavioral"]
    if not behavioral_cols:
        logger.warning(f"No behavioral columns found for dataset {dataset_id}. Skipping trigger evaluation.")
        return False
        
    # 2. Load and preprocess behavioral data
    try:
        X_df, _ = load_and_preprocess_behavioral_data(dataset_id)
    except Exception as e:
        logger.error(f"Failed to load behavioral data for dataset {dataset_id}: {e}")
        return False
        
    if len(X_df) == 0:
        logger.warning("Empty behavioral dataset. Skipping trigger evaluation.")
        return False
        
    # 3. Run K-means and GMM clustering
    try:
        kmeans_labels = run_kmeans_clustering(X_df, behavioral_cols)
        gmm_labels = run_gmm_clustering(X_df, behavioral_cols)
    except Exception as e:
        logger.error(f"Error running clustering algorithms: {e}")
        return False
        
    # 4. Evaluate both results
    kmeans_eval = _evaluate_labels(kmeans_labels)
    gmm_eval = _evaluate_labels(gmm_labels)
    
    kmeans_fails_cond1 = kmeans_eval["cond1"]
    gmm_fails_cond1 = gmm_eval["cond1"]
    
    kmeans_fails_cond2 = kmeans_eval["cond2"]
    gmm_fails_cond2 = gmm_eval["cond2"]
    
    trigger_cond1 = kmeans_fails_cond1 and gmm_fails_cond1
    trigger_cond2 = kmeans_fails_cond2 and gmm_fails_cond2
    
    if trigger_cond1 or trigger_cond2:
        logger.warning(f"Fallback Mode Triggered for dataset {dataset_id}!")
        
        # Determine trigger condition details
        cond_met = ""
        if trigger_cond2:
            cond_met = "Both K-means and GMM returned exactly 2 clusters with a highly skewed population split (one >=85%, other <=15%)."
        else:
            cond_met = "Both K-means and GMM returned fewer than 3 active clusters."
            
        # Update database
        with get_raw_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE datasets SET fallback_active = TRUE WHERE dataset_id = %s;",
                    (dataset_id,)
                )
                
        # Emit notification
        available = [
            "L1 Demographic", "L3 Statistical Testing", "L6 Outcome Analysis",
            "L7 Metric Integration", "A1 Narration", "Ask Your Data",
            "Create Custom Segments", "L8 Feature Signal"
        ]
        disabled = ["L2 Behavioral", "L4 Predictive", "L5 Migration"]
        
        details = {
            "trigger_condition_met": cond_met,
            "kmeans": {
                "active_clusters": kmeans_eval["active_clusters"],
                "counts": kmeans_eval["counts"],
                "percentages": kmeans_eval["percentages"]
            },
            "gmm": {
                "active_clusters": gmm_eval["active_clusters"],
                "counts": gmm_eval["counts"],
                "percentages": gmm_eval["percentages"]
            },
            "available_layers": available,
            "disabled_layers": disabled,
            "suggestion": "You can use Create Custom Segments as an alternative."
        }
        
        try:
            emit(
                source="fallback_trigger",
                condition="fallback_activated",
                severity="soft_warning",
                dataset_type="file_upload",
                details=json.dumps(details)
            )
        except Exception as e:
            logger.error(f"Failed to emit notification: {e}")
            
        return True
        
    return False
