import json
import logging
from src.db.connection import get_raw_connection
from src.llm.claude_client import claude_client
from src.agents.narration_evaluator import verify_narration

logger = logging.getLogger("insight_narrator")

def collate_narrator_inputs(period_id: int) -> dict:
    """Collates all required pipeline data across layers for a given period."""
    with get_raw_connection() as conn:
        with conn.cursor() as cur:
            # Get current period details
            cur.execute("SELECT period_label, granularity FROM periods WHERE period_id = %s;", (period_id,))
            p_row = cur.fetchone()
            if not p_row:
                raise ValueError(f"Period with ID {period_id} not found.")
            period_label, granularity = p_row

            # Get all periods of same granularity sorted chronologically
            cur.execute(
                """
                SELECT period_id, period_label 
                FROM periods 
                WHERE granularity = %s 
                ORDER BY period_start ASC, period_id ASC;
                """,
                (granularity,)
            )
            all_p = cur.fetchall()
            p_ids = [r[0] for r in all_p]
            
            try:
                curr_idx = p_ids.index(period_id)
            except ValueError:
                curr_idx = -1
                
            prev_period_id = p_ids[curr_idx - 1] if curr_idx > 0 else None
            prev_prev_period_id = p_ids[curr_idx - 2] if curr_idx > 1 else None

            # 1. Layer 2.4: baseline & drift
            cur.execute(
                "SELECT baseline_id, adopted_reason FROM cluster_baselines WHERE period_id = %s;",
                (period_id,)
            )
            baseline_row = cur.fetchone()
            if baseline_row:
                outcome = "ADOPT"
                baseline_id, adopted_reason = baseline_row
            else:
                outcome = "HOLD"
                # If HOLD, the active baseline is the most recent active one before this period
                cur.execute(
                    "SELECT baseline_id, period_id FROM cluster_baselines WHERE period_id < %s AND is_active = TRUE ORDER BY period_id DESC LIMIT 1;",
                    (period_id,)
                )
                b_active = cur.fetchone()
                baseline_id = b_active[0] if b_active else None
                adopted_reason = "hold"
                
            # Fetch clusters of this baseline
            clusters = []
            if baseline_id:
                cur.execute(
                    "SELECT cluster_id, cluster_label, intra_cluster_spread FROM clusters WHERE baseline_id = %s;",
                    (baseline_id,)
                )
                clusters = [{"cluster_id": r[0], "cluster_label": r[1], "spread": float(r[2]) if r[2] is not None else None} for r in cur.fetchall()]
                
            # Fetch identity map for these clusters (splits, merges, etc.)
            identity_maps = []
            if outcome == "ADOPT":
                cluster_ids = [c["cluster_id"] for c in clusters]
                if cluster_ids:
                    cur.execute(
                        """
                        SELECT old_cluster_id, new_cluster_id, match_type 
                        FROM cluster_identity_map 
                        WHERE new_cluster_id = ANY(%s) OR old_cluster_id = ANY(%s);
                        """,
                        (cluster_ids, cluster_ids)
                    )
                    identity_maps = [{"old_cluster_id": r[0], "new_cluster_id": r[1], "match_type": r[2]} for r in cur.fetchall()]

            # 2. Layer 4: migration labels
            migration_data = {}
            if prev_period_id:
                cur.execute(
                    """
                    SELECT COUNT(*), SUM(CASE WHEN migrated THEN 1 ELSE 0 END)
                    FROM migration_labels
                    WHERE from_period_id = %s AND to_period_id = %s;
                    """,
                    (prev_period_id, period_id)
                )
                ml_row = cur.fetchone()
                if ml_row and ml_row[0] > 0:
                    total = ml_row[0]
                    migrated = ml_row[1] or 0
                    migration_data = {
                        "total_customers": total,
                        "migrated_count": migrated,
                        "migration_rate": float(migrated) / total
                    }

            # 3. Layer 5: transition matrix
            transition_data = []
            if prev_period_id:
                cur.execute(
                    """
                    SELECT from_cluster_id, to_cluster_id, transition_prob 
                    FROM transition_matrix 
                    WHERE from_period_id = %s AND to_period_id = %s;
                    """,
                    (prev_period_id, period_id)
                )
                transition_data = [{"from_cluster_id": r[0], "to_cluster_id": r[1], "prob": float(r[2])} for r in cur.fetchall()]

            # 4. Layer 6: segment outcomes
            outcomes_current = []
            cur.execute(
                "SELECT cluster_id, churn_rate, retention_rate, revenue, repeat_purchase_prob FROM segment_outcomes WHERE period_id = %s;",
                (period_id,)
            )
            outcomes_current = [{
                "cluster_id": r[0],
                "churn_rate": float(r[1]) if r[1] is not None else None,
                "retention_rate": float(r[2]) if r[2] is not None else None,
                "revenue": float(r[3]) if r[3] is not None else None,
                "repeat_purchase_prob": float(r[4]) if r[4] is not None else None
            } for r in cur.fetchall()]

            outcomes_prev = []
            if prev_period_id:
                cur.execute(
                    "SELECT cluster_id, churn_rate, retention_rate, revenue, repeat_purchase_prob FROM segment_outcomes WHERE period_id = %s;",
                    (prev_period_id,)
                )
                outcomes_prev = [{
                    "cluster_id": r[0],
                    "churn_rate": float(r[1]) if r[1] is not None else None,
                    "retention_rate": float(r[2]) if r[2] is not None else None,
                    "revenue": float(r[3]) if r[3] is not None else None,
                    "repeat_purchase_prob": float(r[4]) if r[4] is not None else None
                } for r in cur.fetchall()]

            # 5. Layer 3: statistical test results
            stat_tests = []
            cur.execute(
                """
                SELECT test_type, segment_a_id, segment_b_id, test_variable, p_value, test_statistic, effect_size, significant
                FROM statistical_test_results
                WHERE period_id = %s;
                """,
                (period_id,)
            )
            stat_tests = [{
                "test_type": r[0],
                "segment_a_id": r[1],
                "segment_b_id": r[2],
                "test_variable": r[3],
                "p_value": float(r[4]) if r[4] is not None else None,
                "test_statistic": float(r[5]) if r[5] is not None else None,
                "effect_size": float(r[6]) if r[6] is not None else None,
                "significant": r[7]
            } for r in cur.fetchall()]

    return {
        "period_label": period_label,
        "granularity": granularity,
        "outcome": outcome,
        "adopted_reason": adopted_reason,
        "clusters": clusters,
        "identity_maps": identity_maps,
        "migration_data": migration_data,
        "transition_data": transition_data,
        "outcomes_current": outcomes_current,
        "outcomes_prev": outcomes_prev,
        "statistical_tests": stat_tests
    }

SYSTEM_PROMPT = """You are an AI Insight Narration Agent for a customer segmentation platform.
Your task is to generate a concise, objective, plain-language summary narration of the customer segment metrics, structural changes, and outcomes for the current period based strictly on the provided dataset.

You MUST obey the following governing constraints:
1. Only describe what the data explicitly shows. Do not make any guesses, theories, or external inferences.
2. Under no circumstances should you use causal language such as 'because', 'caused by', or 'due to'. Layer 3 statistical significance establishes observational difference or association, NEVER causation. Always frame relationships as associations (e.g., 'Higher churn coincided with lower engagement', NOT 'Churn increased because engagement fell').
3. Do not make forward-looking claims or predictions beyond what is explicitly stated in the predictive segment/matrix outputs.
4. The word 'significantly' is permissible ONLY to describe differences directly validated by Layer 3 statistical tests (e.g. 'The segments differed significantly in churn') or flagged demographic anomalies. Never use 'dramatically' or unverified intensifiers.
5. If Layer 3 statistical tests are inconclusive or show no significance, explicitly state that the outcomes are not statistically significant.
6. Every single numeric citation or metric statement must exactly match or correspond to a value in the provided raw dataset.
7. Always cite the source layer name (e.g. Layer 2.4, Layer 4, Layer 5, Layer 6) when referencing metric facts.
"""

def generate_and_evaluate_narration(period_id: int) -> dict:
    """Collates data, prompts Claude, runs evaluation guardrails, retries if needed, and persists result."""
    logger.info(f"Collating data for period {period_id}...")
    collated_data = collate_narrator_inputs(period_id)
    
    data_json_str = json.dumps(collated_data, indent=2)
    prompt = f"Please generate the quarterly narration based on the following collated pipeline data:\n\n{data_json_str}"
    
    passed = False
    errors = []
    narration_text = ""
    max_attempts = 4
    
    for attempt in range(1, max_attempts + 1):
        logger.info(f"Generating narration: attempt {attempt}/{max_attempts}...")
        
        # Modify prompt slightly on subsequent attempts to append errors
        if attempt > 1:
            adjusted_prompt = (
                f"{prompt}\n\n"
                f"IMPORTANT: The previous narration attempt was rejected due to the following guardrail errors:\n"
                f"{json.dumps(errors, indent=2)}\n"
                f"Please regenerate the narration correcting these violations. Ensure strict alignment with all constraints."
            )
        else:
            adjusted_prompt = prompt
            
        narration_text = claude_client.call(prompt=adjusted_prompt, system_prompt=SYSTEM_PROMPT)
        
        passed, errors = verify_narration(narration_text, collated_data, period_id)
        if passed:
            logger.info("Narration passed post-generation evaluation.")
            break
        else:
            logger.warning(f"Narration failed post-generation evaluation on attempt {attempt}: {errors}")
            
    # Persist narration to DB
    model_used = claude_client.model
    with get_raw_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO quarterly_narrations (period_id, narration_text, model_used, generated_at, passed_guardrails)
                VALUES (%s, %s, %s, now(), %s)
                ON CONFLICT (period_id) DO UPDATE
                SET narration_text = EXCLUDED.narration_text,
                    model_used = EXCLUDED.model_used,
                    generated_at = EXCLUDED.generated_at,
                    passed_guardrails = EXCLUDED.passed_guardrails
                RETURNING narration_id;
                """,
                (period_id, narration_text, model_used, passed)
            )
            narration_id = cur.fetchone()[0]
            
    return {
        "narration_id": str(narration_id),
        "narration_text": narration_text,
        "passed_guardrails": passed,
        "errors": errors
    }
