import json
import logging
from src.llm.claude_client import claude_client
from src.agents.narration_evaluator import verify_narration

logger = logging.getLogger("answer_generator")

SYSTEM_PROMPT = """You are an AI Answer Generation Agent for a customer segmentation dashboard.
Your job is to take an aggregated JSON object containing data from various dashboard layers and generate a clear, concise, objective, plain-language answer to the user's query.

You MUST obey the following governing constraints:
1. Every claim, number, and percentage in your answer must map directly to a value in the provided aggregated JSON data. Do not make any external inferences, assumptions, or theories.
2. Under no circumstances should you use causal language such as 'because', 'caused by', or 'due to'. Layer 3 statistical significance establishes observational difference or association, NEVER causation. Always frame relationships as associations (e.g., 'Higher churn coincided with lower engagement', NOT 'Churn increased because engagement fell').
3. The word 'significantly' is permissible ONLY to describe differences directly validated by Layer 3 statistical tests (e.g. 'The segments differed significantly in churn') or flagged demographic anomalies. Never use 'dramatically' or unverified intensifiers.
4. If any of the layers in the aggregated JSON used historical/cached data (as shown in the historical_flags), you MUST explicitly label those metrics in the answer as coming from that historical period (e.g., "(Historical from period_label)").
5. If Layer 3 statistical tests are inconclusive or show no significance, explicitly state that the outcomes are not statistically significant.
"""

def generate_answer(query: str, aggregated_data: dict, period_id: int) -> dict:
    """
    Generates a plain-language answer to the user's query using Claude,
    validating the output using verify_narration.
    """
    data_json_str = json.dumps(aggregated_data, indent=2)
    prompt = (
        f"User Query: {query}\n\n"
        f"Aggregated Pipeline Data:\n{data_json_str}\n\n"
        f"Please generate a plain language answer to the query based strictly on the data above."
    )
    
    passed = False
    errors = []
    answer_text = ""
    max_attempts = 4
    
    for attempt in range(1, max_attempts + 1):
        if attempt > 1:
            adjusted_prompt = (
                f"{prompt}\n\n"
                f"IMPORTANT: The previous answer attempt was rejected due to the following guardrail errors:\n"
                f"{json.dumps(errors, indent=2)}\n"
                f"Please regenerate the answer, correcting all errors and following the system guidelines strictly."
            )
        else:
            adjusted_prompt = prompt
            
        answer_text = claude_client.call(prompt=adjusted_prompt, system_prompt=SYSTEM_PROMPT)
        
        # Verify answer using existing verification logic
        # verify_narration expects the collated_data format; supporting_data fits this perfectly.
        passed, errors = verify_narration(answer_text, aggregated_data.get("supporting_data", {}), period_id)
        if passed:
            logger.info("Generated answer successfully passed narration evaluation.")
            break
        else:
            logger.warning(f"Answer failed narration evaluation on attempt {attempt}: {errors}")
            
    return {
        "answer": answer_text,
        "passed_guardrails": passed,
        "errors": errors
    }
