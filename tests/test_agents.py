import pytest
import json
import re
from unittest.mock import patch
from fastapi.testclient import TestClient
from src.main import app
from src.db.connection import get_raw_connection
from src.scheduler.quarterly_batch import run_quarterly_batch
from src.agents.insight_narrator import generate_and_evaluate_narration


@pytest.fixture
def client():
    return TestClient(app)

def setup_agent_test_data(seed_statistical_sig: bool = False, seed_anomaly: bool = False):
    """Utility to clear DB and seed database state for Phase 10 testing."""
    with get_raw_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM quarterly_narrations;")
            cur.execute("DELETE FROM statistical_test_results;")
            cur.execute("DELETE FROM feature_migration_signals;")
            cur.execute("DELETE FROM segment_outcomes;")
            cur.execute("DELETE FROM transition_matrix;")
            cur.execute("DELETE FROM migration_labels;")
            cur.execute("DELETE FROM cluster_assignments;")
            cur.execute("DELETE FROM clusters;")
            cur.execute("DELETE FROM cluster_baselines;")
            cur.execute("DELETE FROM demographic_segment_metrics;")
            cur.execute("DELETE FROM periods WHERE period_id IN (1001, 1002);")
            cur.execute("DELETE FROM datasets WHERE dataset_id = '00000000-0000-0000-0000-000000001000';")
            cur.execute("DELETE FROM customers WHERE customer_id = '00000000-0000-0000-0000-000000000001';")

            # Seed customer
            cur.execute("INSERT INTO customers (customer_id) VALUES ('00000000-0000-0000-0000-000000000001');")

            # Seed dataset
            cur.execute(
                """
                INSERT INTO datasets (dataset_id, source_type, permanence, layer0_completed, confirmed_classifications)
                VALUES ('00000000-0000-0000-0000-000000001000', 'file_upload', 'permanent', TRUE, 
                '[{"column": "login_frequency", "classification": "behavioral"}]');
                """
            )

            # Seed periods (granularity = 'quarterly')
            cur.execute(
                """
                INSERT INTO periods (period_id, period_label, period_start, period_end, granularity)
                VALUES 
                    (1001, 'P10', '2026-01-01', '2026-03-31', 'quarterly'),
                    (1002, 'P11', '2026-04-01', '2026-06-30', 'quarterly');
                """
            )
            
            # Seed baseline
            cur.execute(
                """
                INSERT INTO cluster_baselines (baseline_id, is_active, adopted_reason, period_id)
                VALUES (10, TRUE, 'initial', 1001);
                """
            )
            
            # Seed clusters
            cur.execute("INSERT INTO clusters (cluster_id, baseline_id, cluster_label) VALUES (1, 10, 'Cluster_1');")
            cur.execute("INSERT INTO clusters (cluster_id, baseline_id, cluster_label) VALUES (2, 10, 'Cluster_2');")
            
            # Seed assignments
            cur.execute(
                """
                INSERT INTO cluster_assignments (customer_id, period_id, cluster_id, assigned_via)
                VALUES ('00000000-0000-0000-0000-000000000001', 1001, 1, 'hdbscan');
                """
            )

            # Seed segment outcomes for P10 (churn_rate=15.0%, revenue=1200.0)
            cur.execute(
                """
                INSERT INTO segment_outcomes (cluster_id, period_id, churn_rate, retention_rate, revenue, repeat_purchase_prob)
                VALUES 
                    (1, 1001, 15.0, 85.0, 1200.0, 0.80),
                    (2, 1001, 8.0, 92.0, 1000.0, 0.90);
                """
            )

            # Seed migration labels (P10 -> P11)
            cur.execute(
                """
                INSERT INTO migration_labels (customer_id, from_period_id, to_period_id, from_cluster_id, to_cluster_id, migrated)
                VALUES ('00000000-0000-0000-0000-000000000001', 1001, 1002, 1, 1, FALSE);
                """
            )

            # Seed transition matrix
            cur.execute(
                """
                INSERT INTO transition_matrix (from_period_id, to_period_id, from_cluster_id, to_cluster_id, customer_count, transition_prob)
                VALUES (1001, 1002, 1, 1, 1, 1.0);
                """
            )

            # Seed statistical test results
            if seed_statistical_sig:
                cur.execute(
                    """
                    INSERT INTO statistical_test_results (period_id, test_type, segment_a_id, segment_b_id, test_variable, p_value, test_statistic, effect_size, significant)
                    VALUES (1001, 'ANOVA', 1, 1, 'churn_rate', 0.01, 5.2, 0.8, TRUE);
                    """
                )
            else:
                cur.execute(
                    """
                    INSERT INTO statistical_test_results (period_id, test_type, segment_a_id, segment_b_id, test_variable, p_value, test_statistic, effect_size, significant)
                    VALUES (1001, 'ANOVA', 1, 1, 'churn_rate', 0.85, 0.05, 0.01, FALSE);
                    """
                )

            # Seed demographic anomalies
            if seed_anomaly:
                cur.execute(
                    """
                    INSERT INTO demographic_segment_metrics (dataset_id, period_id, segment_field, segment_value, segment_size, anomaly_flagged)
                    VALUES ('00000000-0000-0000-0000-000000001000', 1001, 'age', '25-34', 100, TRUE);
                    """
                )
            else:
                cur.execute(
                    """
                    INSERT INTO demographic_segment_metrics (dataset_id, period_id, segment_field, segment_value, segment_size, anomaly_flagged)
                    VALUES ('00000000-0000-0000-0000-000000001000', 1001, 'age', '25-34', 100, FALSE);
                    """
                )

def test_p10_t1_insight_narrator_quarterly_batch():
    """
    P10-T1: Agent 1 narration is generated and persisted at the end of a quarterly batch run.
    """
    setup_agent_test_data(seed_statistical_sig=False)
    
    mock_narration = "Summary: Cluster_1 has a churn rate of 15.0% and revenue of 1200.0. These results are not statistically significant."
    
    # We patch the Claude calls inside preprocessor, clusterer, and narrator
    with patch("src.layer2.preprocessor.load_and_preprocess_behavioral_data") as mock_prep, \
          patch("src.layer2.clusterer.run_kmeans_clustering") as mock_clust, \
         patch("src.scheduler.quarterly_batch.decide_adopt_vs_hold", return_value="HOLD"), \
         patch("src.agents.insight_narrator.claude_client.call", return_value=mock_narration):
         
        import pandas as pd
        import numpy as np
        mock_prep.return_value = (pd.DataFrame({"customer_id": ["00000000-0000-0000-0000-000000000001"], "login_frequency": [1.0]}), None)
        mock_clust.return_value = np.array([0])
        
        batch_res = run_quarterly_batch("00000000-0000-0000-0000-000000001000", 1001)
        
    assert batch_res["status"] == "success"
    
    # Verify that a row is successfully persisted in quarterly_narrations
    with get_raw_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT narration_text, passed_guardrails FROM quarterly_narrations WHERE period_id = 1001;")
            row = cur.fetchone()
            
    assert row is not None
    assert "15.0%" in row[0]
    assert row[1] is True

def test_p10_t2_evaluator_rejects_causal_language():
    """
    P10-T2: Causal language evaluator detects and rejects unsupported causal claims.
    Regeneration is triggered, and a clean result is stored.
    """
    setup_agent_test_data(seed_statistical_sig=False)
    
    # Attempt 1: contains 'because of' (unsupported) -> Rejected
    # Attempt 2: clean narration with association phrasing -> Accepted
    responses = [
        "Cluster_1 saw churn of 15.0% because of session duration declines.",
        "Cluster_1 outcomes for P10 (Layer 6): Churn rate is 15.0% and revenue is 1200.0. The metrics are not statistically significant."
    ]
    
    with patch("src.agents.insight_narrator.claude_client.call", side_effect=responses):
        res = generate_and_evaluate_narration(1001)
        
    assert res["passed_guardrails"] is True
    assert "because of" not in res["narration_text"]
    assert "15.0%" in res["narration_text"]

def test_p10_t2b_causal_language_rejected_even_with_statistical_significance():
    """
    P10-T2b: Layer 3 statistical significance does NOT establish causation.
    Therefore 'because', 'caused by', and 'due to' are rejected even with p < 0.05.
    Association phrasing ('coincided with') and 'differed significantly' are accepted.
    """
    # Seed statistical significance (p < 0.05) in Layer 3
    setup_agent_test_data(seed_statistical_sig=True)
    
    # Attempt 1: 'because' -> Rejected despite p < 0.05
    # Attempt 2: 'caused by' -> Rejected despite p < 0.05
    # Attempt 3: Valid association phrasing + 'differed significantly' -> Accepted
    responses = [
        "Churn of 15.0% increased because session duration dropped.",
        "Higher churn of 15.0% was caused by reduced purchase frequency.",
        "The segments differed significantly in churn (15.0% vs 8.0%). Higher churn coincided with lower engagement."
    ]
    
    with patch("src.agents.insight_narrator.claude_client.call", side_effect=responses):
        res = generate_and_evaluate_narration(1001)
        
    assert res["passed_guardrails"] is True
    assert "because" not in res["narration_text"]
    assert "caused by" not in res["narration_text"]
    assert "coincided with" in res["narration_text"]
    assert "differed significantly" in res["narration_text"]

def test_p10_t3_evaluator_intensifier_guard():
    """
    P10-T3: Intensifier guard flags 'significantly' or 'dramatically' if not backed.
    Narration is regenerated until clean or warning is persisted.
    """
    setup_agent_test_data(seed_statistical_sig=False, seed_anomaly=False)
    
    # Attempt 1: contains 'significantly' (unsupported) -> Rejected
    # Attempt 2: clean narration -> Accepted
    responses = [
        "Cluster_1 revenue grew significantly to 1200.0.",
        "Cluster_1 outcomes for P10 (Layer 6): Churn rate is 15.0% and revenue is 1200.0."
    ]
    
    with patch("src.agents.insight_narrator.claude_client.call", side_effect=responses):
        res = generate_and_evaluate_narration(1001)
        
    assert res["passed_guardrails"] is True
    assert "significantly" not in res["narration_text"]

def test_p10_t4_agent2_valid_query_routing(client):
    """
    P10-T4: Agent 2 (replaced by /api/v1/ask) valid query is routed correctly.
    """
    setup_agent_test_data(seed_statistical_sig=False)
    
    # Mock global Claude calls. First call is for intent router ("show"), second is for answer generator ("Churn rate is 15.0%.")
    with patch("src.llm.claude_client.claude_client.call", side_effect=['{"intent": "show"}', 'Churn rate is 15.0%.']):
        response = client.post("/api/v1/ask", json={
            "query": "Show outcomes for Cluster_1 in P10",
            "segment_id": 1,
            "period_id": 1001
        })
        
    assert response.status_code == 200
    res = response.json()
    assert res["intent"] == "show"
    # Also verify that /api/v1/query works identically as an alias
    with patch("src.llm.claude_client.claude_client.call", side_effect=['{"intent": "show"}', 'Churn rate is 15.0%.']):
        q_response = client.post("/api/v1/query", json={
            "query": "Show outcomes for Cluster_1 in P10",
            "segment_id": 1,
            "period_id": 1001
        })
    assert q_response.status_code == 200
    assert q_response.json()["intent"] == "show"

def test_p10_t5_agent2_out_of_scope_no_segment(client):
    """
    P10-T5: Out-of-scope query returns the hard stop (status code 400).
    """
    setup_agent_test_data(seed_statistical_sig=False)
    
    with patch("src.llm.claude_client.claude_client.call", return_value='{"intent": "out_of_scope"}'):
        response = client.post("/api/v1/query", json={
            "query": "Which customers are most likely to become VIPs?"
        })
        
    assert response.status_code == 400
    assert response.json()["detail"] == "This query is outside the queryable space."

def test_p10_t6_agent2_no_time_period(client):
    """
    P10-T6: Out-of-scope query lacking a time period/metadata returns the hard stop.
    """
    setup_agent_test_data(seed_statistical_sig=False)
    
    with patch("src.llm.claude_client.claude_client.call", return_value='{"intent": "out_of_scope"}'):
        response = client.post("/api/v1/query", json={
            "query": "Show me data for Cluster_1."
        })
        
    assert response.status_code == 400
    assert response.json()["detail"] == "This query is outside the queryable space."

def test_p10_t7_agent2_no_custom_rendering():
    """
    P10-T7: Ask Your Data and NL Query Agent source code contains no custom HTML, chart, table rendering, or layout config.
    """
    import glob
    files_to_check = glob.glob("src/ask/*.py") + ["src/agents/nl_query_agent.py"]
    for filepath in files_to_check:
        with open(filepath, "r") as f:
            content = f.read()
        forbidden_tokens = ["matplotlib", "plotly", "seaborn", "pandas.plot", "html", "<div>", "<table>"]
        for token in forbidden_tokens:
            assert token not in content, f"Forbidden rendering token '{token}' found in {filepath}."
