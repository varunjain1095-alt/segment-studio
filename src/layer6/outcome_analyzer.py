import logging
import pandas as pd
from src.db.connection import get_raw_connection
from src.layer3.statistical_tester import get_customer_level_outcomes

logger = logging.getLogger("layer6_outcome_analyzer")

def compute_and_persist_segment_outcomes(dataset_id: str, period_id: int) -> list[dict]:
    """
    Computes segment-level outcome metrics for the given period_id by aggregating
    customer-level outcomes, and persists them to the segment_outcomes table.
    
    STATISTICAL INTEGRITY & PROXY METRICS SPECIFICATION:
    - Primary Path (Confirmed Raw CSV):
      When backed by confirmed external dataset fields (e.g. Churned, unscaled Lifetime_Value),
      these outcomes represent true independent ground-truth.
    - Fallback Path (behavioral_snapshots table):
      When falling back to behavioral_snapshots (e.g., in integration smoke tests or demo runs),
      reconstructed revenue (purchase_count * avg_order_value) and repeat_purchase_prob
      are derived from clustering features. Layer 6 computes and persists these metrics
      descriptively for dashboard presentation, but they are DEMO/INTEGRATION PROXIES ONLY
      and must NOT be used as independent validation surfaces in Layer 3 statistical testing.
      Only churn_rate and retention_rate (derived from temporal T-to-T+1 customer presence)
      qualify as independent outcomes under snapshot fallback.
    """
    logger.info(f"Computing segment outcomes for dataset {dataset_id}, period {period_id}...")
    
    # 1. Load customer-level outcomes (established in Phase 4)
    outcomes_df = get_customer_level_outcomes(dataset_id, period_id)
    if outcomes_df.empty:
        logger.warning("Empty customer outcomes. Cannot compute segment-level metrics.")
        return []
        
    outcomes_df["customer_id"] = outcomes_df["customer_id"].astype(str)
    
    # 2. Get cluster assignments for the current period
    with get_raw_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT customer_id, cluster_id 
                FROM cluster_assignments 
                WHERE period_id = %s;
                """,
                (period_id,)
            )
            rows = cur.fetchall()
            
    if not rows:
        logger.warning(f"No cluster assignments found for period {period_id}.")
        return []
        
    assign_df = pd.DataFrame(rows, columns=["customer_id", "cluster_id"])
    assign_df["customer_id"] = assign_df["customer_id"].astype(str)
    
    # 3. Merge assignments with outcomes
    merged_df = pd.merge(assign_df, outcomes_df, on="customer_id", how="inner")
    if merged_df.empty:
        logger.warning("Merged dataset is empty. Verify customer mapping keys.")
        return []
        
    results = []
    
    # 4. Group by cluster and aggregate
    for cluster_id in merged_df["cluster_id"].unique():
        cluster_data = merged_df[merged_df["cluster_id"] == cluster_id]
        n_total = len(cluster_data)
        if n_total == 0:
            continue
            
        # Sum of revenue
        total_revenue = float(cluster_data["revenue"].sum())
        
        # Mean of repeat purchase probability
        repeat_prob = float(cluster_data["repeat_purchase_prob"].mean())
        
        # Segment-level churn & retention strictly computed as average of customer-level values
        # Customer-level churn/retention is 0.0 or 1.0 (mean * 100.0 yields percentage 0-100)
        churn_rate = float(cluster_data["churn_rate"].mean() * 100.0)
        retention_rate = float(cluster_data["retention_rate"].mean() * 100.0)
        
        results.append({
            "cluster_id": int(cluster_id),
            "period_id": period_id,
            "churn_rate": churn_rate,
            "retention_rate": retention_rate,
            "revenue": total_revenue,
            "repeat_purchase_prob": repeat_prob
        })
        
    # 5. Persist to public.segment_outcomes table
    with get_raw_connection() as conn:
        with conn.cursor() as cur:
            for rec in results:
                cur.execute(
                    """
                    INSERT INTO segment_outcomes (
                        cluster_id, period_id, churn_rate, retention_rate,
                        revenue, repeat_purchase_prob
                    )
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (cluster_id, period_id)
                    DO UPDATE SET
                        churn_rate = EXCLUDED.churn_rate,
                        retention_rate = EXCLUDED.retention_rate,
                        revenue = EXCLUDED.revenue,
                        repeat_purchase_prob = EXCLUDED.repeat_purchase_prob;
                    """,
                    (
                        rec["cluster_id"],
                        rec["period_id"],
                        rec["churn_rate"],
                        rec["retention_rate"],
                        rec["revenue"],
                        rec["repeat_purchase_prob"]
                    )
                )
                
    logger.info(f"Successfully persisted outcomes for {len(results)} clusters in period {period_id}.")
    return results
