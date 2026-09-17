import pytest
import json
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
from src.db.connection import get_raw_connection
from src.layer1.field_classifier import classify_fields, confirm_field_classifications, verify_classifications
from src.layer1.mom_tracker import populate_demographic_snapshots, compute_and_persist_mom_metrics
from src.layer1.anomaly_flagger import flag_anomalies_for_period

def test_p2_t1_llm_classifier_format():
    """P2-T1: LLM Field Classifier — Classification Output Format"""
    dataset_id = "00000000-0000-0000-0000-000000000201"
    
    # 10 columns profile
    cols = [f"col_{i}" for i in range(10)]
    profile_df = pd.DataFrame({
        "column_name": cols,
        "dtype": ["object"] * 10
    })
    
    mock_response = json.dumps([
        {"column": f"col_{i}", "classification": "demographic" if i < 3 else ("behavioral" if i < 8 else "unsure")}
        for i in range(10)
    ])
    
    with patch("src.layer1.field_classifier.claude_client.call", return_value=mock_response):
        with patch("src.layer1.field_classifier.get_raw_connection") as mock_conn:
            proposals = classify_fields(dataset_id, profile_df)
            
    assert len(proposals) == 10
    valid_classes = {"demographic", "behavioral", "unsure"}
    for item in proposals:
        assert "column" in item
        assert "classification" in item
        assert item["classification"] in valid_classes

def test_p2_t2_proposal_only():
    """P2-T2: LLM Classification Is Proposal Only"""
    dataset_id = "00000000-0000-0000-0000-000000000202"
    
    # Create dataset record
    with get_raw_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM datasets WHERE dataset_id = %s;", (dataset_id,))
            cur.execute(
                """
                INSERT INTO datasets (dataset_id, source_type, permanence)
                VALUES (%s, 'file_upload', 'permanent');
                """,
                (dataset_id,)
            )
            
    # Mock LLM proposal run
    profile_df = pd.DataFrame({"column_name": ["age"], "dtype": ["int64"]})
    mock_response = json.dumps([{"column": "age", "classification": "demographic"}])
    
    with patch("src.layer1.field_classifier.claude_client.call", return_value=mock_response):
        classify_fields(dataset_id, profile_df)
        
    # Check that confirmed_classifications is still null in DB
    with get_raw_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT confirmed_classifications FROM datasets WHERE dataset_id = %s;", (dataset_id,))
            res = cur.fetchone()
            assert res[0] is None, "Confirmed classifications should be NULL (proposal only)"

def test_p2_t3_unsure_hard_fail():
    """P2-T3: Notification Center — Unsure Column Hard Fail (Dynamic Datasets Only)"""
    dataset_id = "00000000-0000-0000-0000-000000000203"
    classifications = [
        {"column": "age", "classification": "demographic"},
        {"column": "mystery_col", "classification": "unsure"}
    ]
    
    # Scenario A: Dynamic (database_connection) -> Hard fail emitted
    with patch("src.layer1.field_classifier.emit") as mock_emit:
        verify_classifications(dataset_id, "dynamic", classifications)
    mock_emit.assert_called_once()
    assert mock_emit.call_args[1].get("severity") == "hard_fail"
    
    # Scenario B: Permanent (file_upload) -> No hard fail emitted
    with patch("src.layer1.field_classifier.emit") as mock_emit:
        verify_classifications(dataset_id, "permanent", classifications)
    mock_emit.assert_not_called()

def test_p2_t4_mom_percentage_change():
    """P2-T4: Month-over-Month Percentage Change Calculation"""
    dataset_id = "00000000-0000-0000-0000-000000000204"
    period_prev = 101
    period_curr = 102
    
    # Seeding database
    with get_raw_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM demographic_segment_metrics WHERE dataset_id = %s;", (dataset_id,))
            cur.execute("DELETE FROM demographic_snapshots WHERE period_id IN (%s, %s);", (period_prev, period_curr))
            cur.execute("DELETE FROM periods WHERE period_id IN (%s, %s);", (period_prev, period_curr))
            cur.execute("DELETE FROM datasets WHERE dataset_id = %s;", (dataset_id,))
            
            cur.execute("INSERT INTO periods (period_id, period_label, period_start, period_end, granularity) VALUES (%s, 'T-1', '2026-01-01', '2026-01-31', 'monthly');", (period_prev,))
            cur.execute("INSERT INTO periods (period_id, period_label, period_start, period_end, granularity) VALUES (%s, 'T', '2026-02-01', '2026-02-28', 'monthly');", (period_curr,))
            cur.execute("INSERT INTO datasets (dataset_id, source_type, permanence) VALUES (%s, 'file_upload', 'permanent');", (dataset_id,))
            
    # T-1 size = 50, T size = 60 for segment country = India
    df_prev = pd.DataFrame({
        "customer_id": [f"00000000-0000-0000-0000-00000000{i:04d}" for i in range(50)],
        "country": ["India"] * 50
    })
    df_curr = pd.DataFrame({
        "customer_id": [f"00000000-0000-0000-0000-00000000{i:04d}" for i in range(60)],
        "country": ["India"] * 60
    })
    
    populate_demographic_snapshots(df_prev, period_prev, ["country"])
    populate_demographic_snapshots(df_curr, period_curr, ["country"])
    
    compute_and_persist_mom_metrics(dataset_id, period_curr, period_prev, ["country"])
    
    # Query metric record and assert MoM pct change = 20.0
    with get_raw_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT prev_segment_size, segment_size, mom_pct_change
                FROM demographic_segment_metrics
                WHERE dataset_id = %s AND period_id = %s AND segment_field = 'country' AND segment_value = 'India';
                """,
                (dataset_id, period_curr)
            )
            row = cur.fetchone()
            
    assert row is not None
    assert row[0] == 50 # prev_segment_size
    assert row[1] == 60 # segment_size
    assert float(row[2]) == 20.0 # mom_pct_change = ((60 - 50) / 50) * 100

def test_p2_t5_volatility_anomaly_flagging():
    """P2-T5: Volatility-Aware Flagging — Anomaly Verification"""
    dataset_id = "00000000-0000-0000-0000-000000000205"
    period_id = 103
    
    with get_raw_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM demographic_segment_metrics WHERE dataset_id = %s;", (dataset_id,))
            cur.execute("DELETE FROM periods WHERE period_id = %s;", (period_id,))
            cur.execute("DELETE FROM datasets WHERE dataset_id = %s;", (dataset_id,))
            
            cur.execute("INSERT INTO periods (period_id, period_label, period_start, period_end, granularity) VALUES (%s, 'T_test', '2026-03-01', '2026-03-31', 'monthly');", (period_id,))
            cur.execute("INSERT INTO datasets (dataset_id, source_type, permanence) VALUES (%s, 'file_upload', 'permanent');", (dataset_id,))
            
            # Segment A: std_dev = 2.0%, change = 5.0% (> 2*std_dev=4.0%) -> flag True
            cur.execute(
                """
                INSERT INTO demographic_segment_metrics (
                    dataset_id, period_id, segment_field, segment_value, segment_size, mom_pct_change, historical_std_dev, anomaly_flagged
                )
                VALUES (%s, %s, 'age', '25-34', 100, 5.0, 2.0, FALSE);
                """,
                (dataset_id, period_id)
            )
            # Segment B: std_dev = 10.0%, change = 15.0% (<= 2*std_dev=20.0%) -> flag False
            cur.execute(
                """
                INSERT INTO demographic_segment_metrics (
                    dataset_id, period_id, segment_field, segment_value, segment_size, mom_pct_change, historical_std_dev, anomaly_flagged
                )
                VALUES (%s, %s, 'country', 'India', 100, 15.0, 10.0, FALSE);
                """,
                (dataset_id, period_id)
            )
            
    with patch("src.layer1.anomaly_flagger.emit") as mock_emit:
        flag_anomalies_for_period(dataset_id, period_id)
        
    # Check flags in DB
    with get_raw_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT segment_field, segment_value, anomaly_flagged
                FROM demographic_segment_metrics
                WHERE dataset_id = %s AND period_id = %s
                ORDER BY segment_field;
                """,
                (dataset_id, period_id)
            )
            rows = cur.fetchall()
            
    flags = {f"{r[0]}={r[1]}": r[2] for r in rows}
    assert flags["age=25-34"] == True
    assert flags["country=India"] == False
    
    # Assert emission happened only for segment A
    assert mock_emit.call_count == 1
    assert "age" in mock_emit.call_args[1].get("details")

def test_p2_t6_minimum_segment_size_gate():
    """P2-T6: Minimum Segment Size Gate (N < 30)"""
    dataset_id = "00000000-0000-0000-0000-000000000207"
    period_prev = 301
    period_curr = 302
    
    # Seeding database
    with get_raw_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM demographic_segment_metrics WHERE dataset_id = %s;", (dataset_id,))
            cur.execute("DELETE FROM demographic_snapshots WHERE period_id IN (%s, %s);", (period_prev, period_curr))
            cur.execute("DELETE FROM periods WHERE period_id IN (%s, %s);", (period_prev, period_curr))
            cur.execute("DELETE FROM datasets WHERE dataset_id = %s;", (dataset_id,))
            
            cur.execute("INSERT INTO periods (period_id, period_label, period_start, period_end, granularity) VALUES (%s, 'T-1', '2026-01-01', '2026-01-31', 'monthly');", (period_prev,))
            cur.execute("INSERT INTO periods (period_id, period_label, period_start, period_end, granularity) VALUES (%s, 'T', '2026-02-01', '2026-02-28', 'monthly');", (period_curr,))
            cur.execute("INSERT INTO datasets (dataset_id, source_type, permanence) VALUES (%s, 'file_upload', 'permanent');", (dataset_id,))
            
    # T-1 size = 15, T size = 20 for segment location = Iceland (both are < 30)
    df_prev = pd.DataFrame({
        "customer_id": [f"00000000-0000-0000-0000-00000000{i:04d}" for i in range(15)],
        "country": ["Iceland"] * 15
    })
    df_curr = pd.DataFrame({
        "customer_id": [f"00000000-0000-0000-0000-00000000{i:04d}" for i in range(20)],
        "country": ["Iceland"] * 20
    })
    
    populate_demographic_snapshots(df_prev, period_prev, ["country"])
    populate_demographic_snapshots(df_curr, period_curr, ["country"])
    
    compute_and_persist_mom_metrics(dataset_id, period_curr, period_prev, ["country"])
    
    # Run anomaly flagging (should not flag since MoM change is not calculated)
    with patch("src.layer1.anomaly_flagger.emit") as mock_emit:
        flag_anomalies_for_period(dataset_id, period_curr)
        
    mock_emit.assert_not_called()
    
    # Query database to confirm mom_pct_change is NULL and anomaly_flagged is FALSE
    with get_raw_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT prev_segment_size, segment_size, mom_pct_change, anomaly_flagged
                FROM demographic_segment_metrics
                WHERE dataset_id = %s AND period_id = %s AND segment_field = 'country' AND segment_value = 'Iceland';
                """,
                (dataset_id, period_curr)
            )
            row = cur.fetchone()
            
    assert row is not None
    assert row[0] is None or row[0] == 15
    assert row[1] == 20
    assert row[2] is None, "mom_pct_change must be NULL for segments with N < 30"
    assert row[3] == False, "anomaly_flagged must be FALSE for gated segments"

def test_p2_t7_no_eav_structure():
    """P2-T7: Demographic Snapshots Schema Constraint (No EAV)"""
    with get_raw_connection() as conn:
        with conn.cursor() as cur:
            # Introspect demographic_snapshots columns
            cur.execute(
                """
                SELECT column_name
                FROM information_schema.columns
                WHERE table_name = 'demographic_snapshots';
                """
            )
            cols = [r[0] for r in cur.fetchall()]
            
    # Confirm it does not look like an EAV schema (e.g. key/value columns)
    assert "age" in cols
    assert "gender" in cols
    assert "country" in cols
    assert "attribute_name" not in cols
    assert "attribute_value" not in cols

def test_p2_t8_db_schema_alteration():
    """P2-T8: Database Schema Alteration for Demographic Snapshots"""
    dataset_id = "00000000-0000-0000-0000-000000000206"
    
    # Seed DB profile
    with get_raw_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM column_profiles WHERE dataset_id = %s;", (dataset_id,))
            cur.execute("DELETE FROM datasets WHERE dataset_id = %s;", (dataset_id,))
            cur.execute("INSERT INTO datasets (dataset_id, source_type, permanence) VALUES (%s, 'file_upload', 'permanent');", (dataset_id,))
            cur.execute("INSERT INTO column_profiles (dataset_id, column_name, dtype) VALUES (%s, 'income_bracket', 'float64');", (dataset_id,))
            
    classifications = [
        {"column": "income_bracket", "classification": "demographic"}
    ]
    
    confirm_field_classifications(dataset_id, classifications)
    
    # Verify that demographic_snapshots carries income_bracket column
    with get_raw_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT column_name
                FROM information_schema.columns
                WHERE table_name = 'demographic_snapshots' AND column_name = 'income_bracket';
                """
            )
            row = cur.fetchone()
            
    assert row is not None
    assert row[0] == "income_bracket"
