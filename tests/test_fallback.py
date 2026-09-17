import pytest
import json
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from src.main import app
from src.db.connection import get_raw_connection
from src.fallback.trigger_detector import evaluate
from src.fallback.state_manager import get_fallback_state
from src.fallback.pattern_discovery import discover_patterns
from src.scheduler.quarterly_batch import run_quarterly_batch

@pytest.fixture
def client():
    return TestClient(app)

def clean_database_tables(cur):
    """Truncates/clears test data only to avoid wiping production data."""
    cur.execute("DELETE FROM column_profiles WHERE dataset_id LIKE '00000000-0000-0000-0000-000000000f%';")
    cur.execute("DELETE FROM datasets WHERE dataset_id LIKE '00000000-0000-0000-0000-000000000f%';")

@pytest.fixture
def setup_dataset():
    dataset_id = "00000000-0000-0000-0000-000000000f01"
    classes = [
        {"column": "login_frequency", "classification": "behavioral"},
        {"column": "session_duration", "classification": "behavioral"}
    ]
    with get_raw_connection() as conn:
        with conn.cursor() as cur:
            clean_database_tables(cur)
            cur.execute(
                """
                INSERT INTO datasets (dataset_id, source_type, permanence, layer0_completed, confirmed_classifications, fallback_active)
                VALUES (%s, 'file_upload', 'permanent', TRUE, %s, FALSE);
                """,
                (dataset_id, json.dumps(classes))
            )
            
            # Insert column profiles for pattern discovery test
            cur.execute(
                """
                INSERT INTO column_profiles (dataset_id, column_name, dtype)
                VALUES (%s, 'purchase_value', 'float64'),
                       (%s, 'gender', 'object');
                """,
                (dataset_id, dataset_id)
            )
    return dataset_id

@patch("src.fallback.trigger_detector.load_and_preprocess_behavioral_data")
@patch("src.fallback.trigger_detector.emit")
def test_fallback_trigger_fewer_than_3_clusters(mock_emit, mock_preprocess, setup_dataset):
    """Condition 1: Both K-means and GMM return fewer than 3 active clusters."""
    dataset_id = setup_dataset
    mock_preprocess.return_value = (pd.DataFrame({"customer_id": [f"cust_{i}" for i in range(100)], "login_frequency": range(100), "session_duration": range(100)}), ["login_frequency", "session_duration"])
    
    # KMeans returns 2 active clusters (0 and 1)
    # GMM returns 2 active clusters (0 and 1)
    kmeans_labels = np.array([0]*50 + [1]*50)
    gmm_labels = np.array([0]*60 + [1]*40)
    
    with patch("src.fallback.trigger_detector.run_kmeans_clustering", return_value=kmeans_labels):
        with patch("src.fallback.trigger_detector.run_gmm_clustering", return_value=gmm_labels):
            triggered = evaluate(dataset_id)
            assert triggered is True
            
            # Check DB is updated
            with get_raw_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT fallback_active FROM datasets WHERE dataset_id = %s;", (dataset_id,))
                    assert cur.fetchone()[0] is True
                    
            # Check notification is emitted
            mock_emit.assert_called_once()
            args, kwargs = mock_emit.call_args
            assert kwargs["severity"] == "soft_warning"
            assert kwargs["source"] == "fallback_trigger"
            details = json.loads(kwargs["details"])
            assert "fewer than 3 active clusters" in details["trigger_condition_met"]

@patch("src.fallback.trigger_detector.load_and_preprocess_behavioral_data")
@patch("src.fallback.trigger_detector.emit")
def test_fallback_trigger_exactly_2_skewed_clusters(mock_emit, mock_preprocess, setup_dataset):
    """Condition 2: Both return exactly 2 active clusters with highly skewed population split (one >=85%, other <=15%)."""
    dataset_id = setup_dataset
    mock_preprocess.return_value = (pd.DataFrame({"customer_id": [f"cust_{i}" for i in range(100)], "login_frequency": range(100), "session_duration": range(100)}), ["login_frequency", "session_duration"])
    
    # Both have 2 active clusters, split is skewed (e.g. 90/10 and 88/12)
    kmeans_labels = np.array([0]*90 + [1]*10)
    gmm_labels = np.array([0]*12 + [1]*88)
    
    with patch("src.fallback.trigger_detector.run_kmeans_clustering", return_value=kmeans_labels):
        with patch("src.fallback.trigger_detector.run_gmm_clustering", return_value=gmm_labels):
            triggered = evaluate(dataset_id)
            assert triggered is True
            
            mock_emit.assert_called_once()
            args, kwargs = mock_emit.call_args
            details = json.loads(kwargs["details"])
            assert "highly skewed population split" in details["trigger_condition_met"]

@patch("src.fallback.trigger_detector.load_and_preprocess_behavioral_data")
@patch("src.fallback.trigger_detector.emit")
def test_fallback_no_trigger_only_one_fails(mock_emit, mock_preprocess, setup_dataset):
    """Fallback does NOT trigger when only one algorithm fails."""
    dataset_id = setup_dataset
    mock_preprocess.return_value = (pd.DataFrame({"customer_id": [f"cust_{i}" for i in range(100)], "login_frequency": range(100), "session_duration": range(100)}), ["login_frequency", "session_duration"])
    
    # KMeans has 3 clusters (passes), GMM has 2 clusters (fails)
    kmeans_labels = np.array([0]*33 + [1]*33 + [2]*34)
    gmm_labels = np.array([0]*50 + [1]*50)
    
    with patch("src.fallback.trigger_detector.run_kmeans_clustering", return_value=kmeans_labels):
        with patch("src.fallback.trigger_detector.run_gmm_clustering", return_value=gmm_labels):
            triggered = evaluate(dataset_id)
            assert triggered is False
            
            # Check DB remains False
            with get_raw_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT fallback_active FROM datasets WHERE dataset_id = %s;", (dataset_id,))
                    assert cur.fetchone()[0] is False
                    
            mock_emit.assert_not_called()

def test_get_fallback_state_normal(setup_dataset):
    """Verify state manager output when fallback is inactive."""
    dataset_id = setup_dataset
    state = get_fallback_state(dataset_id)
    
    assert state["fallback_active"] is False
    assert "L2 Behavioral" in state["available_layers"]
    assert len(state["disabled_layers"]) == 0

def test_get_fallback_state_active(setup_dataset):
    """Verify state manager output when fallback is active."""
    dataset_id = setup_dataset
    
    with get_raw_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("UPDATE datasets SET fallback_active = TRUE WHERE dataset_id = %s;", (dataset_id,))
            
    state = get_fallback_state(dataset_id)
    assert state["fallback_active"] is True
    assert "L2 Behavioral" in state["disabled_layers"]
    assert "L2 Behavioral" not in state["available_layers"]
    assert "L1 Demographic" in state["available_layers"]

@patch("src.fallback.pattern_discovery.pd.read_csv")
@patch("src.fallback.pattern_discovery.os.path.exists", return_value=True)
def test_pattern_discovery_quartile_bucketing_and_ranking(mock_exists, mock_read_csv, setup_dataset):
    """Verify quartile bucketing, top-N ranking, customer counts, and support percentages."""
    dataset_id = setup_dataset
    
    data = {
        "customer_id": [f"cust_{i}" for i in range(100)],
        "purchase_value": [100.0]*20 + [50.0]*40 + [50.0]*10 + [10.0]*20 + [100.0]*5 + [10.0]*5,
        "gender": ['Female']*20 + ['Female']*40 + ['Male']*10 + ['Male']*20 + ['Male']*5 + ['Female']*5
    }
    df = pd.DataFrame(data)
    
    mock_read_csv.side_effect = [df, df]
    
    results = discover_patterns(dataset_id, ["purchase_value", "gender"], top_n=3)
    
    assert len(results) <= 3
    assert results[0]["combination"] == "Medium purchase value + gender: Female"
    assert results[0]["count"] == 40
    assert pytest.approx(results[0]["percentage"], 0.01) == 0.40
    
    combinations = [r["combination"] for r in results]
    assert "High purchase value + gender: Female" in combinations
    assert "Low purchase value + gender: Male" in combinations

def test_api_get_fallback_state(client, setup_dataset):
    """Verify API endpoint returns correct fallback state structure."""
    dataset_id = setup_dataset
    
    # 1. Normal State
    response = client.get(f"/api/v1/datasets/{dataset_id}/fallback-state")
    assert response.status_code == 200
    data = response.json()
    assert data["fallback_active"] is False
    assert "L2 Behavioral" in data["available_layers"]
    
    # 2. Fallback Active State
    with get_raw_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("UPDATE datasets SET fallback_active = TRUE WHERE dataset_id = %s;", (dataset_id,))
            
    response = client.get(f"/api/v1/datasets/{dataset_id}/fallback-state")
    assert response.status_code == 200
    data = response.json()
    assert data["fallback_active"] is True
    assert "L2 Behavioral" in data["disabled_layers"]

@patch("src.fallback.pattern_discovery.pd.read_csv")
@patch("src.fallback.pattern_discovery.os.path.exists", return_value=True)
def test_api_discover_patterns(mock_exists, mock_read_csv, client, setup_dataset):
    """Verify API discover-patterns endpoint processes payload and returns results."""
    dataset_id = setup_dataset
    
    data = {
        "customer_id": [f"cust_{i}" for i in range(10)],
        "purchase_value": [10.0]*5 + [100.0]*5
    }
    df = pd.DataFrame(data)
    mock_read_csv.side_effect = [df, df]
    
    payload = {
        "columns": ["purchase_value"],
        "top_n": 2
    }
    response = client.post(f"/api/v1/datasets/{dataset_id}/fallback/discover-patterns", json=payload)
    assert response.status_code == 200
    res_data = response.json()
    assert "patterns" in res_data
    assert len(res_data["patterns"]) > 0

def test_api_available_columns(client, setup_dataset):
    """Verify API returns the behavioral columns confirmed classifications."""
    dataset_id = setup_dataset
    response = client.get(f"/api/v1/datasets/{dataset_id}/fallback/available-columns")
    assert response.status_code == 200
    data = response.json()
    assert "columns" in data
    assert "login_frequency" in data["columns"]
    assert "session_duration" in data["columns"]

@patch("src.layer2.preprocessor.load_and_preprocess_behavioral_data")
@patch("src.layer2.clusterer.run_kmeans_clustering")
@patch("src.scheduler.quarterly_batch.decide_adopt_vs_hold")
@patch("src.scheduler.quarterly_batch.compute_and_persist_segment_outcomes")
@patch("src.scheduler.quarterly_batch.run_statistical_tests")
@patch("src.scheduler.quarterly_batch.compute_and_persist_feature_migration_signals")
@patch("src.scheduler.quarterly_batch.generate_and_evaluate_narration")
@patch("src.custom_segments.segment_scheduler.run_scheduled_segments")
@patch("src.fallback.trigger_detector.load_and_preprocess_behavioral_data")
@patch("src.fallback.trigger_detector.run_kmeans_clustering")
@patch("src.fallback.trigger_detector.run_gmm_clustering")
def test_quarterly_batch_runner_fallback(
    mock_trigger_gmm, mock_trigger_kmeans, mock_trigger_preproc, mock_run_segments, mock_narration, mock_migrate_signals,
    mock_stat_tests, mock_outcomes, mock_adopt, mock_batch_kmeans, mock_batch_preproc, setup_dataset
):
    """Verify batch runner skips L2/L4/L5 when fallback triggers."""
    dataset_id = setup_dataset
    period_id = 901
    
    # Create the period 901 in DB first to satisfy foreign key / narration lookup constraints
    with get_raw_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO periods (period_id, period_label, period_start, period_end, granularity)
                VALUES (901, 'P901', '2026-04-01', '2026-06-30', 'quarterly');
                """
            )
            
    df = pd.DataFrame({"customer_id": [f"cust_{i}" for i in range(100)], "login_frequency": range(100), "session_duration": range(100)})
    mock_batch_preproc.return_value = (df, ["login_frequency", "session_duration"])
    mock_trigger_preproc.return_value = (df, ["login_frequency", "session_duration"])
    
    # Force fallback triggering: both return fewer than 3 active clusters
    kmeans_labels = np.array([0]*50 + [1]*50)
    gmm_labels = np.array([0]*60 + [1]*40)
    
    mock_batch_kmeans.return_value = kmeans_labels
    mock_trigger_kmeans.return_value = kmeans_labels
    mock_trigger_gmm.return_value = gmm_labels
    
    mock_narration.return_value = "Mock narration briefing"
    
    result = run_quarterly_batch(dataset_id, period_id)
    
    # Verify result
    assert result["status"] == "success"
    assert result["fallback_active"] is True
    assert result["drift_decision"] == "SKIPPED_FALLBACK_ACTIVE"
    
    # L2 adopt/hold must be skipped
    mock_adopt.assert_not_called()
    
    # Standard statistical, narration, outcomes runs should still proceed
    mock_outcomes.assert_called_once()
    mock_stat_tests.assert_called_once()
    mock_narration.assert_called_once()
