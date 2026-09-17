import pytest
import os
import pandas as pd
from src.ingestion.router import DATA_DIR
from src.custom_segments.segment_engine import build_segment

@pytest.fixture
def mock_dataset():
    dataset_id = "00000000-0000-0000-0000-000000000001"
    df = pd.DataFrame({
        "customer_id": ["c1", "c2", "c3", "c4"],
        "age": [20, 30, 40, 50],
        "gender": ["Male", "Female", "Male", "Female"]
    })
    processed_path = os.path.join(DATA_DIR, f"{dataset_id}_processed.csv")
    df.to_csv(processed_path, index=False)
    
    # Insert dataset to satisfy FK constraint
    from src.db.connection import get_raw_connection
    with get_raw_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM datasets WHERE dataset_id = %s;", (dataset_id,))
            cur.execute(
                "INSERT INTO datasets (dataset_id, source_type, permanence) VALUES (%s, 'file_upload', 'permanent');",
                (dataset_id,)
            )
            
    yield dataset_id
    if os.path.exists(processed_path):
        os.remove(processed_path)
    with get_raw_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM datasets WHERE dataset_id = %s;", (dataset_id,))

def test_build_segment_numeric_categorical(mock_dataset):
    # Test numeric range filter
    conds_num = [{"column": "age", "type": "numeric", "range": [25, 45]}]
    res_num = build_segment(mock_dataset, conds_num)
    assert set(res_num) == {"c2", "c3"}
    
    # Test categorical values filter
    conds_cat = [{"column": "gender", "type": "categorical", "values": ["Female"]}]
    res_cat = build_segment(mock_dataset, conds_cat)
    assert set(res_cat) == {"c2", "c4"}
    
    # Test quartile-based filter
    # age values: 20, 30, 40, 50.
    # Q25 = 27.5, Q75 = 42.5.
    # Bottom 25% (<= 27.5) -> c1 (20)
    # Top 25% (>= 42.5) -> c4 (50)
    # Middle 50% (27.5 <= age <= 42.5) -> c2, c3
    conds_q_bottom = [{"column": "age", "type": "numeric", "range": "Bottom 25%"}]
    res_q_bottom = build_segment(mock_dataset, conds_q_bottom)
    assert set(res_q_bottom) == {"c1"}

    conds_q_middle = [{"column": "age", "type": "numeric", "range": "Middle 50%"}]
    res_q_middle = build_segment(mock_dataset, conds_q_middle)
    assert set(res_q_middle) == {"c2", "c3"}

def test_comparison_engine():
    from src.custom_segments.comparison_engine import (
        option1_vs_baseline, option2_aggregate_change, option3_compare_segments,
        option4_statistical_testing, option5_outcome_analysis, option6_metric_integration,
        option7_feature_signal
    )
    from unittest.mock import patch, MagicMock
    import pandas as pd
    
    # 1. Mock get_customer_features
    mock_features = pd.DataFrame({
        "customer_id": ["c1", "c2"],
        "login_frequency": [10.0, 20.0],
        "gender": ["Male", "Female"]
    })
    
    with patch("src.custom_segments.comparison_engine.get_customer_features", return_value=mock_features), \
         patch("src.custom_segments.comparison_engine.get_raw_connection") as mock_conn:
         
        # Mock database cursor
        mock_cur = mock_conn.return_value.__enter__.return_value.cursor.return_value.__enter__.return_value
        # Mock for option 1: baseline cluster assignments
        mock_cur.fetchall.return_value = [
            ("c1", 1, "Cluster_1"),
            ("c2", 2, "Cluster_2")
        ]
        
        # Run Option 1
        res_opt1 = option1_vs_baseline(["c1", "c2"], "ds_id", 10, ["login_frequency"])
        assert "custom_segment" in res_opt1
        assert res_opt1["custom_segment"]["login_frequency"] == 15.0
        assert "Cluster_1" in res_opt1["baseline_clusters"]
        
    # Test Option 3
    with patch("src.custom_segments.comparison_engine.get_customer_features", return_value=mock_features):
        segments = [
            {"name": "Seg A", "customers": ["c1"], "period_id": 10},
            {"name": "Seg B", "customers": ["c2"], "period_id": 10}
        ]
        res_opt3 = option3_compare_segments(segments, ["login_frequency"])
        assert "Seg A" in res_opt3
        assert res_opt3["Seg A"]["login_frequency"] == 15.0

def test_save_and_schedule():
    from src.custom_segments.segment_scheduler import save_segment, schedule_segment, run_scheduled_segments
    from unittest.mock import patch, MagicMock
    from src.db.connection import get_raw_connection
    
    dataset_id = "00000000-0000-0000-0000-000000000000"
    conditions = [{"column": "age", "type": "numeric", "range": [18, 25]}]
    
    # Insert dataset to satisfy FK constraint
    with get_raw_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM datasets WHERE dataset_id = %s;", (dataset_id,))
            cur.execute(
                "INSERT INTO datasets (dataset_id, source_type, permanence) VALUES (%s, 'file_upload', 'permanent');",
                (dataset_id,)
            )
            
    # We run actual save and schedule since we created the tables in migrations!
    segment_id = save_segment(dataset_id, "Test Segment", conditions)
    assert segment_id is not None
    
    schedule_id = schedule_segment(segment_id, "monthly", ["option5"])
    assert schedule_id is not None
    
    # Set next_run_at to the past so it is due
    with get_raw_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE custom_segment_schedules SET next_run_at = NOW() - INTERVAL '1 day' WHERE schedule_id = %s;",
                (schedule_id,)
            )
            
    # Mock build_segment and option5 execution to test run_scheduled_segments
    with patch("src.custom_segments.segment_scheduler.build_segment", return_value=["c1", "c2"]), \
         patch("src.custom_segments.comparison_engine.option5_outcome_analysis", return_value={"churn_rate": 5.0}), \
         patch("src.custom_segments.segment_scheduler.emit") as mock_emit:
         
        run_scheduled_segments()
        
        # Verify notification emission
        mock_emit.assert_called_once()
        assert mock_emit.call_args[1]["condition"] == "scheduled_run_completed"
        assert mock_emit.call_args[1]["severity"] == "info"

def test_custom_segments_api_endpoints(mock_dataset):
    from fastapi.testclient import TestClient
    from src.main import app
    from unittest.mock import patch
    
    client = TestClient(app)
    
    # 1. Build
    resp = client.post("/api/v1/custom-segments/build", json={
        "dataset_id": mock_dataset,
        "conditions": [{"column": "age", "type": "numeric", "range": [25, 45]}]
    })
    assert resp.status_code == 200
    res_data = resp.json()
    assert res_data["status"] == "success"
    assert set(res_data["matching_customer_ids"]) == {"c2", "c3"}
    
    # 2. Run Option (Option 5 outcomes)
    with patch("src.layer3.statistical_tester.get_customer_level_outcomes") as mock_outcomes:
        import pandas as pd
        mock_outcomes.return_value = pd.DataFrame({
            "customer_id": ["c2", "c3"],
            "churn_rate": [0.0, 1.0],
            "retention_rate": [1.0, 0.0],
            "revenue": [100.0, 200.0],
            "repeat_purchase_prob": [0.8, 0.9]
        })
        
        resp_opt = client.post("/api/v1/custom-segments/run-option", json={
            "segment_customers": ["c2", "c3"],
            "dataset_id": mock_dataset,
            "option": "option5",
            "period_id": 1
        })
        assert resp_opt.status_code == 200
        res_opt_data = resp_opt.json()
        assert res_opt_data["status"] == "success"
        assert res_opt_data["result"]["churn_rate"] == 50.0
        assert res_opt_data["result"]["revenue"] == 300.0
        
    # 3. Save
    resp_save = client.post("/api/v1/custom-segments/save", json={
        "dataset_id": mock_dataset,
        "segment_name": "API Saved Segment",
        "conditions": [{"column": "age", "type": "numeric", "range": [20, 30]}]
    })
    assert resp_save.status_code == 200
    segment_id = resp_save.json()["segment_id"]
    
    # 4. Get Saved
    resp_get = client.get(f"/api/v1/custom-segments/saved?dataset_id={mock_dataset}")
    assert resp_get.status_code == 200
    saved_list = resp_get.json()
    assert len(saved_list) > 0
    assert saved_list[0]["segment_name"] == "API Saved Segment"
    
    # 5. Schedule
    resp_sched = client.post("/api/v1/custom-segments/schedule", json={
        "segment_id": segment_id,
        "granularity": "monthly",
        "selected_options": ["option5"]
    })
    assert resp_sched.status_code == 200
    assert resp_sched.json()["status"] == "success"
    
    # 6. Delete
    resp_del = client.delete(f"/api/v1/custom-segments/{segment_id}")
    assert resp_del.status_code == 200
    assert resp_del.json()["deleted_segment_id"] == segment_id
    
    # 7. Exports
    assert client.post("/api/v1/custom-segments/export/pdf", json={}).status_code == 200
    assert client.post("/api/v1/custom-segments/export/email", json={}).status_code == 200



