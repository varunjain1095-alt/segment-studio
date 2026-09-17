import json
import logging
from src.llm.claude_client import claude_client

logger = logging.getLogger("intent_router")

SYSTEM_PROMPT = """You are an Intent Classifier for a customer segmentation dashboard.
Your job is to classify the user's natural language query into exactly one of these five intents:
- show: when requesting to view, show, or surface a specific metric, segment profile, drift, or layer output. E.g., "Show churn rate", "Show me session time by segment".
- comparison: when asking to compare multiple segments, periods, or metrics. E.g., "Compare Power Users vs Deal Seekers on retention", "How does Cluster_1 compare to Cluster_2".
- root_cause: when asking why something changed or seeking explanations/reasons for an event. E.g., "Why did session time increase for this segment?", "Why did churn increase for Power Users?".
- prediction: when asking forward-looking or predictive questions about transitions, migrations, or future behavior. E.g., "Which customers are likely to migrate next quarter?", "Who will leave next month?".
- out_of_scope: when the query is unrelated to the customer segmentation data/dashboard, or cannot be answered with the available layer metrics.

You must return a raw JSON object with exactly this format:
{
  "intent": "one of: show, comparison, root_cause, prediction, out_of_scope"
}

Do not include any formatting, markdown code fences (like ```json), or conversational text. Output only the JSON.
"""

def classify_intent(query: str) -> dict:
    """Classifies user query intent using Claude."""
    if not query or not isinstance(query, str):
        return {"intent": "out_of_scope", "query": query}
        
    # Predefined template matching to avoid LLM call for routing
    query_lower = query.lower().strip("?. ")
    if "which cluster has the highest" in query_lower:
        return {"intent": "show", "query": query}
        
    try:
        response_text = claude_client.call(prompt=query, system_prompt=SYSTEM_PROMPT)
        cleaned = response_text.strip()
        if cleaned.startswith("```"):
            lines = cleaned.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines[-1].startswith("```"):
                lines = lines[:-1]
            cleaned = "\n".join(lines).strip()
            
        res = json.loads(cleaned)
        intent = res.get("intent", "out_of_scope")
        if intent not in ["show", "comparison", "root_cause", "prediction", "out_of_scope"]:
            intent = "out_of_scope"
        return {"intent": intent, "query": query}
    except Exception as e:
        logger.error(f"Error classifying intent: {e}")
        return {"intent": "out_of_scope", "query": query}
