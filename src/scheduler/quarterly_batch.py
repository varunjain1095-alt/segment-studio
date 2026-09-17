import logging
from src.db.connection import get_raw_connection
from src.layer2.drift_evaluator import decide_adopt_vs_hold
from src.layer6.outcome_analyzer import compute_and_persist_segment_outcomes
from src.layer3.statistical_tester import run_statistical_tests
from src.layer8.migration_signal import compute_and_persist_feature_migration_signals
from src.agents.insight_narrator import generate_and_evaluate_narration

logger = logging.getLogger("quarterly_batch")

def run_quarterly_batch(dataset_id: str, period_id: int) -> dict:
    """
    Executes the quarterly batch pipeline in the exact sequence:
    1. Collect behavioral + outcome data for the period (assumed pre-ingested or managed)
    2. Layer 0 re-profile / clean (run automatically or assumed completed)
    3. Layer 2.4 -> evaluate drift -> Adopt or Hold
    4a/4b. Run Layer 4 model training or label generation / Layer 5 transition matrix (triggered inside Layer 2.4/scheduler)
    5. Layer 6 -> compute outcome metrics for this period
    6. Layer 3 -> run statistical tests with new outcome data
    7. Layer 8 -> compute feature migration signals for this period
    8. Agent 1 -> generate quarterly narration briefing
    """
    logger.info(f"Starting quarterly batch run for dataset {dataset_id}, period {period_id}...")
    
    # We fetch confirmed classifications to get behavioral columns
    with get_raw_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT confirmed_classifications FROM datasets WHERE dataset_id = %s;", (dataset_id,))
            res = cur.fetchone()
            
    if not res or not res[0]:
        raise ValueError("Confirmed classifications not found for dataset.")
        
    import json
    classifications = json.loads(res[0]) if isinstance(res[0], str) else res[0]
    behavioral_cols = [c["column"] for c in classifications if c.get("classification") == "behavioral"]
    
    # 1 & 2. Assumed collected and preprocessed (preprocessor handles loading)
    from src.layer2.preprocessor import load_and_preprocess_behavioral_data
    X_df, _ = load_and_preprocess_behavioral_data(dataset_id)
    
    from src.layer2.clusterer import run_kmeans_clustering
    labels = run_kmeans_clustering(X_df, behavioral_cols)
    
    # Evaluate fallback trigger
    from src.fallback.trigger_detector import evaluate as evaluate_fallback
    fallback_active = evaluate_fallback(dataset_id, period_id)
    
    if fallback_active:
        logger.warning(f"Fallback mode is active for dataset {dataset_id}. Skipping L2 adopt/hold, L4 retraining, and L5 transition matrix computation.")
        drift_decision = "SKIPPED_FALLBACK_ACTIVE"
    else:
        # 3. Layer 2.4 Drift Evaluation (Adopt vs Hold)
        # Also triggers Step 4a/4b retraining or KNN classification inside decider
        logger.info("Step 3 & 4: Running drift evaluation & KNN / Retraining trigger...")
        drift_decision = decide_adopt_vs_hold(dataset_id, period_id, X_df, labels, behavioral_cols)
    
    # 5. Layer 6 Outcomes
    logger.info("Step 5: Computing outcome metrics...")
    compute_and_persist_segment_outcomes(dataset_id, period_id)
    
    # 6. Layer 3 Statistical Testing
    logger.info("Step 6: Running statistical tests...")
    # Run outcomes testing against segment column (default cluster assignments cluster_id)
    try:
        run_statistical_tests(
            dataset_id=dataset_id,
            period_id=period_id,
            segment_column="cluster_id",
            test_variable="churn_rate",
            parametric=False
        )
    except Exception as e:
        logger.warning(f"Layer 3 test failed/skipped in batch: {e}")
        
    # 7. Layer 8 Signals
    logger.info("Step 7: Computing feature migration signals...")
    try:
        compute_and_persist_feature_migration_signals(dataset_id, period_id)
    except Exception as e:
        logger.warning(f"Layer 8 signals failed/skipped in batch: {e}")
        
    # 8. Agent 1 Narration Briefing
    logger.info("Step 8: Generating Agent 1 insight narration...")
    narration_result = generate_and_evaluate_narration(period_id)
    
    # 9. Custom Segments scheduling run
    logger.info("Step 9: Running scheduled custom segments re-evaluation...")
    try:
        from src.custom_segments.segment_scheduler import run_scheduled_segments
        run_scheduled_segments()
    except Exception as e:
        logger.warning(f"Scheduled custom segments run failed/skipped in batch: {e}")
        
    logger.info("Quarterly batch run completed successfully.")
    
    return {
        "status": "success",
        "drift_decision": drift_decision,
        "narration": narration_result,
        "fallback_active": fallback_active
    }
