import json
import re
import logging
from src.db.connection import get_raw_connection
from src.llm.claude_client import claude_client
from src.ask.intent_router import classify_intent
from src.ask.execution_planner import plan_execution
from src.ask.execution_engine import execute_pipeline
from src.ask.results_aggregator import aggregate_results
from src.ask.answer_generator import generate_answer

logger = logging.getLogger("nl_query_agent")

HARD_STOP_MESSAGE = "This query is outside the queryable space."

def resolve_entities_from_query(query: str) -> tuple[int | None, int | None]:
    """
    Attempts to resolve segment_id and period_id from the query text.
    Checks the database for matching cluster labels and period labels.
    """
    resolved_segment_id = None
    resolved_period_id = None
    query_lower = query.lower()
    
    try:
        with get_raw_connection() as conn:
            with conn.cursor() as cur:
                # Retrieve clusters
                cur.execute("SELECT cluster_id, cluster_label FROM clusters;")
                clusters = cur.fetchall()
                for cid, clabel in clusters:
                    if clabel and clabel.lower() in query_lower:
                        resolved_segment_id = cid
                        break
                    if f"segment {cid}" in query_lower or f"cluster {cid}" in query_lower or f"cluster_{cid}" in query_lower:
                        resolved_segment_id = cid
                        break
                
                # Retrieve periods
                cur.execute("SELECT period_id, period_label FROM periods;")
                periods = cur.fetchall()
                for pid, plabel in periods:
                    if plabel and plabel.lower() in query_lower:
                        resolved_period_id = pid
                        break
    except Exception as e:
        logger.warning(f"Entity resolution database lookup failed: {e}")
        
    return resolved_segment_id, resolved_period_id

def process_query(query: str, segment_id: int = None, period_id: int = None) -> dict:
    """
    Natural Language Query Agent (Agent 2):
    - Validates query scope (must be time-bound when referencing segments).
    - Routes to the appropriate layer API.
    - Aggregates and returns existing layer outputs without custom rendering.
    """
    if not query or not query.strip():
        raise ValueError(HARD_STOP_MESSAGE)
        
    # Extract entities if not explicitly supplied
    extracted_segment_id, extracted_period_id = resolve_entities_from_query(query)
    effective_segment_id = segment_id if segment_id is not None else extracted_segment_id
    effective_period_id = period_id if period_id is not None else extracted_period_id
    
    # 1. Intent Classification
    router_res = classify_intent(query)
    intent = router_res.get("intent", "out_of_scope")
    
    # Rule P10-T5: Out of scope returns hard stop
    if intent == "out_of_scope":
        raise ValueError(HARD_STOP_MESSAGE)
        
    # Rule P10-T6: Segment bound but no time period specified returns hard stop
    # Check if a segment was identified or requested, but no period was provided or resolved
    has_segment_mention = (
        effective_segment_id is not None or
        re.search(r"\b(segment|cluster)\s*['\"]?\w+['\"]?\b", query, re.IGNORECASE) is not None
    )
    if has_segment_mention and effective_period_id is None:
        # Check if query contains any temporal indicator (quarter, year, month, P1, etc.)
        has_time_word = re.search(r"\b(q[1-4]|202[0-9]|period|month|quarter|p\d+)\b", query, re.IGNORECASE)
        if not has_time_word:
            logger.info(f"Query '{query}' is segment-bound but lacks time period. Triggering hard stop.")
            raise ValueError(HARD_STOP_MESSAGE)
            
    # 2. Execution Planning
    planner_res = plan_execution(intent, query)
    if planner_res.get("hard_stop"):
        raise ValueError(HARD_STOP_MESSAGE)
        
    # 3. Execution Pipeline
    context_params = {
        "period_id": effective_period_id,
        "cluster_id": effective_segment_id,
        "segment_ids": [effective_segment_id] if effective_segment_id is not None else []
    }
    pipeline_res = execute_pipeline(planner_res["steps"], context_params)
    
    # 4. Results Aggregation
    aggregated = aggregate_results(pipeline_res)
    
    # 5. Answer Generation
    answer_res = generate_answer(query, aggregated, effective_period_id or 1)
    
    return {
        "intent": intent,
        "answer": answer_res["answer"],
        "supporting_data": aggregated["supporting_data"],
        "historical_flags": aggregated["historical_flags"]
    }
