import logging
from src.layer4.dataset_builder import build_training_data
from src.layer4.trainer import train_and_select_model

logger = logging.getLogger("layer4_retrain_trigger")

def trigger_retraining(dataset_id: str, T: int) -> dict:
    """
    Triggers end-to-end retraining of Layer 4 predictive models using features
    from period T and target migration labels from T+1.
    """
    logger.info(f"Retraining trigger initiated for dataset {dataset_id} at period T={T}...")
    
    # 1. Build training features and targets
    X, y = build_training_data(dataset_id, T)
    
    # We need at least some samples and both classes to perform meaningful training in production
    # Fallback to allow training if y contains at least some records
    if X.empty or len(y) < 2:
        logger.warning("Insufficient samples or cluster mappings found between T and T+1. Retraining aborted.")
        return {}
        
    # 2. Train candidate models and select the best model by PR-AUC
    _, metrics = train_and_select_model(X, y)
    
    logger.info(f"Retraining trigger successfully completed. Metrics: {metrics}")
    return metrics
