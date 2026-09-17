import pytest
import json
from unittest.mock import patch, ANY
from src.ask.intent_router import classify_intent

def test_intent_router_classification():
    # Test each intent classification on a known mock response
    queries_and_intents = [
        ("Show churn rate for Cluster_1", "show"),
        ("Compare Power Users vs Deal Seekers on retention", "comparison"),
        ("Why did session time increase for Cluster_2?", "root_cause"),
        ("Which customers are likely to migrate next quarter?", "prediction"),
        ("What is the capital of France?", "out_of_scope")
    ]
    
    for query, expected_intent in queries_and_intents:
        mock_response = json.dumps({"intent": expected_intent})
        with patch("src.ask.intent_router.claude_client.call", return_value=mock_response) as mock_call:
            res = classify_intent(query)
            assert res == {"intent": expected_intent, "query": query}
            mock_call.assert_called_once_with(prompt=query, system_prompt=ANY)
            
    # Test fallback on JSON parsing failure
    with patch("src.ask.intent_router.claude_client.call", return_value="invalid json"):
        res = classify_intent("Show churn rate")
        assert res == {"intent": "out_of_scope", "query": "Show churn rate"}

def test_execution_planner():
    from src.ask.execution_planner import plan_execution
    
    # out_of_scope
    p = plan_execution("out_of_scope")
    assert p["hard_stop"] is True
    assert p["steps"] == []
    
    # comparison
    p = plan_execution("comparison")
    assert p["hard_stop"] is False
    assert p["steps"] == [
        {"layer": "Layer7", "action": "retrieve_comparison"},
        {"layer": "Layer6", "action": "retrieve_churn"}
    ]
    
    # root_cause
    p = plan_execution("root_cause")
    assert p["hard_stop"] is False
    assert len(p["steps"]) == 7
    
    # prediction
    p = plan_execution("prediction")
    assert p["hard_stop"] is False
    assert p["steps"] == [
        {"layer": "Layer4", "action": "retrieve_predictions"},
        {"layer": "Layer5", "action": "retrieve_migration"},
        {"layer": "Layer8", "action": "retrieve_signals"}
    ]
    
    # show variants
    p = plan_execution("show", "Show demographics for segment")
    assert p["steps"] == [{"layer": "Layer1", "action": "retrieve_demographics"}]
    
    p = plan_execution("show", "Show drift and medoids")
    assert p["steps"] == [{"layer": "Layer2", "action": "retrieve_drift"}]
    
    p = plan_execution("show", "Is this statistically significant?")
    assert p["steps"] == [{"layer": "Layer3", "action": "retrieve_significance"}]
    
    p = plan_execution("show", "Show me predictions for next quarter")
    assert p["steps"] == [{"layer": "Layer4", "action": "retrieve_predictions"}]
    
    p = plan_execution("show", "What is the transition matrix?")
    assert p["steps"] == [{"layer": "Layer5", "action": "retrieve_migration"}]
    
    p = plan_execution("show", "Show early warning signals")
    assert p["steps"] == [{"layer": "Layer8", "action": "retrieve_signals"}]
    
    p = plan_execution("show", "What is the churn rate?")
    assert p["steps"] == [{"layer": "Layer6", "action": "retrieve_churn"}]

def test_execution_engine():
    from src.ask.execution_engine import dispatch_step, execute_pipeline
    from unittest.mock import MagicMock
    
    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = [{"segment_field": "gender", "segment_value": "Male", "segment_size": 100}]
    mock_client.get.return_value = mock_resp
    
    # Mock get_periods_descending to return a specific period structure
    with patch("src.ask.execution_engine.get_periods_descending", return_value=[{"id": 10, "label": "P10"}]):
        res = dispatch_step({"layer": "Layer1", "action": "retrieve_demographics"}, {"period_id": 10}, client=mock_client)
        assert res["data"] == [{"segment_field": "gender", "segment_value": "Male", "segment_size": 100}]
        assert res.get("meta") == {}
        mock_client.get.assert_called_once_with("/api/v1/demographic-segments?period_id=10")

def test_results_aggregator():
    from src.ask.results_aggregator import aggregate_results
    
    execution_results = [
        {
            "step": {"layer": "Layer6", "action": "retrieve_churn"},
            "result": {"data": {"churn_rate": 0.12}, "meta": {}}
        },
        {
            "step": {"layer": "Layer1", "action": "retrieve_demographics"},
            "result": {"data": [{"segment_field": "gender"}], "meta": {"historical": True, "period": "P9"}}
        }
    ]
    
    aggregated = aggregate_results(execution_results)
    assert aggregated["supporting_data"]["layer6"] == {"churn_rate": 0.12}
    assert aggregated["supporting_data"]["layer1"] == [{"segment_field": "gender"}]
    assert aggregated["historical_flags"]["layer1"] == {"historical": True, "period": "P9"}
    assert "layer6" not in aggregated["historical_flags"]

def test_answer_generator_causal_language_rejection():
    from src.ask.answer_generator import generate_answer
    
    aggregated_data = {
        "supporting_data": {
            "layer6": {"churn_rate": 0.15, "revenue": 1000.0}
        },
        "historical_flags": {}
    }
    
    # Mock database to return 0 significant tests for period 1001 (to trigger failure in verify_narration)
    with patch("src.agents.narration_evaluator.get_raw_connection") as mock_conn:
        mock_cur = mock_conn.return_value.__enter__.return_value.cursor.return_value.__enter__.return_value
        mock_cur.fetchone.return_value = [0] # 0 significant tests
        
        # Claude returns causal language
        mock_causal_response = "Churn increased to 15.0% because revenue was 1000.0."
        with patch("src.ask.answer_generator.claude_client.call", return_value=mock_causal_response):
            res = generate_answer("Why did churn increase?", aggregated_data, 1001)
            # Should fail validation because of causal language with 0 significant tests
            assert res["passed_guardrails"] is False
            assert any("causal language" in err for err in res["errors"])



