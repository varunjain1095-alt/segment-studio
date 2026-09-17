import pytest
from fastapi.testclient import TestClient
from src.main import app
from src.db.connection import get_raw_connection
from src.layer8.migration_signal import (
    confirm_layer8_columns,
    compute_and_persist_feature_migration_signals
)
from src.notifications.center import PipelineBlockedError

@pytest.fixture
def client():
    return TestClient(app)

def setup_base_data(dataset_id: str, T: int, num_cols: int = 5):
    """Utility helper to clear DB and seed basic dataset structure."""
    with get_raw_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM feature_migration_signals WHERE period_id = %s;", (T,))
            cur.execute("DELETE FROM layer8_config WHERE dataset_id = %s;", (dataset_id,))
            cur.execute("DELETE FROM cluster_assignments WHERE period_id = %s;", (T,))
            cur.execute("DELETE FROM clusters WHERE baseline_id = 8;")
            cur.execute("DELETE FROM cluster_baselines WHERE baseline_id = 8;")
            cur.execute("DELETE FROM demographic_snapshots WHERE period_id = %s;", (T,))
            cur.execute("DELETE FROM behavioral_snapshots WHERE period_id = %s;", (T,))
            cur.execute("DELETE FROM periods WHERE period_id = %s;", (T,))
            cur.execute("DELETE FROM datasets WHERE dataset_id = %s;", (dataset_id,))
            
            # Seed dataset
            import json
            # Confirmed classifications with 8 behavioral columns
            classifications = [
                {"column": "login_frequency", "classification": "behavioral"},
                {"column": "session_duration", "classification": "behavioral"},
                {"column": "purchase_count", "classification": "behavioral"},
                {"column": "avg_order_value", "classification": "behavioral"},
                {"column": "return_rate", "classification": "behavioral"},
                {"column": "discount_sensitivity", "classification": "behavioral"},
                {"column": "email_engagement", "classification": "behavioral"},
                {"column": "app_engagement", "classification": "behavioral"}
            ]
            cur.execute(
                """
                INSERT INTO datasets (dataset_id, source_type, permanence, layer0_completed, confirmed_classifications)
                VALUES (%s, 'file_upload', 'permanent', TRUE, %s);
                """,
                (dataset_id, json.dumps(classifications))
            )
            
            # Seed period
            cur.execute(
                """
                INSERT INTO periods (period_id, period_label, period_start, period_end, granularity)
                VALUES (%s, 'T_800', '2026-04-01', '2026-06-30', 'quarterly');
                """,
                (T,)
            )
            
            # Seed active baseline
            cur.execute(
                """
                INSERT INTO cluster_baselines (baseline_id, is_active, adopted_reason, period_id)
                VALUES (8, TRUE, 'initial', %s);
                """,
                (T,)
            )
            
            # Seed clusters (80: Own, 81: Compared)
            cur.execute("INSERT INTO clusters (cluster_id, baseline_id, cluster_label) VALUES (80, 8, 'Cluster_Own');")
            cur.execute("INSERT INTO clusters (cluster_id, baseline_id, cluster_label) VALUES (81, 8, 'Cluster_Compared');")

def test_p8_t1_t2_signal_thresholds(client):
    """
    P8-T1 & P8-T2: Verifies that:
    - >50% of columns drifting (3/5) raises a signal (signal_raised = True).
    - <=50% of columns drifting (2/5) does NOT raise a signal (signal_raised = False).
    """
    dataset_id = "00000000-0000-0000-0000-000000000801"
    T = 801
    setup_base_data(dataset_id, T)
    
    # 5 columns to be confirmed
    confirmed_cols = ["login_frequency", "session_duration", "purchase_count", "avg_order_value", "return_rate"]
    
    # PM confirms columns via endpoint
    payload = {
        "columns": [{"column_name": col, "include": True} for col in confirmed_cols]
    }
    response = client.post(f"/api/v1/datasets/{dataset_id}/layer8-columns/confirm", json=payload)
    assert response.status_code == 200
    assert response.json()["status"] == "success"
    
    c_ids = {
        "C": "00000000-0000-0000-0000-00000000800c",  # >50% drift (3/5) -> True
        "D": "00000000-0000-0000-0000-00000000800d"   # <=50% drift (2/5) -> False
    }
    
    # 10 background members for Cluster 80 to lock its median to 10.0
    for i in range(10):
        c_ids[f"M80_{i}"] = f"00000000-0000-0000-0000-00000000808{i}"
    # 10 background members for Cluster 81 to lock its median to 20.0
    for i in range(10):
        c_ids[f"M81_{i}"] = f"00000000-0000-0000-0000-00000000818{i}"
    
    with get_raw_connection() as conn:
        with conn.cursor() as cur:
            for name, cid in c_ids.items():
                cur.execute("INSERT INTO customers (customer_id) VALUES (%s) ON CONFLICT (customer_id) DO NOTHING;", (cid,))
                
                # Assign to clusters
                cluster_id = 81 if "M81" in name else 80
                cur.execute(
                    """
                    INSERT INTO cluster_assignments (customer_id, period_id, cluster_id, assigned_via)
                    VALUES (%s, %s, %s, 'hdbscan');
                    """,
                    (cid, T, cluster_id)
                )
                
            # Insert snapshots
            # Cluster 80 medians: all 10.0
            for name, cid in c_ids.items():
                if "M80" in name:
                    cur.execute(
                        """
                        INSERT INTO behavioral_snapshots (customer_id, period_id, login_frequency, session_duration, purchase_count, avg_order_value, return_rate)
                        VALUES (%s, %s, 10.0, 10.0, 10.0, 10.0, 10.0);
                        """,
                        (cid, T)
                    )
                
            # Cluster 81 medians: all 20.0
            for name, cid in c_ids.items():
                if "M81" in name:
                    cur.execute(
                        """
                        INSERT INTO behavioral_snapshots (customer_id, period_id, login_frequency, session_duration, purchase_count, avg_order_value, return_rate)
                        VALUES (%s, %s, 20.0, 20.0, 20.0, 20.0, 20.0);
                        """,
                        (cid, T)
                    )
                
            # Customer C in Cluster 80: closer to 81 on 3 columns
            # Values: [16.0, 16.0, 16.0, 14.0, 14.0]
            # Dist 80: |16-10|=6 (col 0,1,2), |14-10|=4 (col 3,4)
            # Dist 81: |16-20|=4 (col 0,1,2), |14-20|=6 (col 3,4)
            # 4 < 6 (True for col 0,1,2 -> closer to 81)
            # 6 < 4 (False for col 3,4 -> closer to 80)
            # Drifted = 3 columns, fraction = 0.60
            cur.execute(
                """
                INSERT INTO behavioral_snapshots (customer_id, period_id, login_frequency, session_duration, purchase_count, avg_order_value, return_rate)
                VALUES (%s, %s, 16.0, 16.0, 16.0, 14.0, 14.0);
                """,
                (c_ids["C"], T)
            )
            
            # Customer D in Cluster 80: closer to 81 on 2 columns
            # Values: [16.0, 16.0, 14.0, 14.0, 14.0]
            # Dist 80: |16-10|=6 (col 0,1), |14-10|=4 (col 2,3,4)
            # Dist 81: |16-20|=4 (col 0,1), |14-20|=6 (col 2,3,4)
            # 4 < 6 (True for col 0,1 -> closer to 81)
            # 6 < 4 (False for col 2,3,4 -> closer to 80)
            # Drifted = 2 columns, fraction = 0.40
            cur.execute(
                """
                INSERT INTO behavioral_snapshots (customer_id, period_id, login_frequency, session_duration, purchase_count, avg_order_value, return_rate)
                VALUES (%s, %s, 16.0, 16.0, 14.0, 14.0, 14.0);
                """,
                (c_ids["D"], T)
            )
            
    # Compute signals via endpoint
    sig_response = client.post(f"/api/v1/datasets/{dataset_id}/periods/{T}/run-migration-signals")
    assert sig_response.status_code == 200
    
    # Retrieve persisted records for validation
    with get_raw_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT customer_id, drift_fraction, signal_raised 
                FROM feature_migration_signals
                WHERE period_id = %s AND compared_cluster_id = 81
                ORDER BY customer_id;
                """,
                (T,)
            )
            rows = cur.fetchall()
            
    # Should calculate for C and D (excluding medoid builders because they are also evaluated, but C and D are our targets)
    c_res = next(r for r in rows if str(r[0]) == c_ids["C"])
    d_res = next(r for r in rows if str(r[0]) == c_ids["D"])
    
    # Verify P8-T1
    assert abs(float(c_res[1]) - 0.60) < 1e-6
    assert c_res[2] is True
    
    # Verify P8-T2
    assert abs(float(d_res[1]) - 0.40) < 1e-6
    assert d_res[2] is False

def test_p8_t3_boundary_exactly_50_percent(client):
    """
    P8-T3: Boundary case - exactly 50% must NOT raise a signal (strictly > 0.50).
    4 columns confirmed. Customer E has exactly 2 drifting columns.
    """
    dataset_id = "00000000-0000-0000-0000-000000000803"
    T = 803
    setup_base_data(dataset_id, T)
    
    # 4 columns confirmed
    confirmed_cols = ["login_frequency", "session_duration", "purchase_count", "avg_order_value"]
    confirm_layer8_columns(dataset_id, [{"column_name": col, "include": True} for col in confirmed_cols])
    
    c_ids = {
        "E": "00000000-0000-0000-0000-00000000800e"
    }
    for i in range(10):
        c_ids[f"M80_{i}"] = f"00000000-0000-0000-0000-00000000803{i}"
    for i in range(10):
        c_ids[f"M81_{i}"] = f"00000000-0000-0000-0000-00000000813{i}"
    
    with get_raw_connection() as conn:
        with conn.cursor() as cur:
            for name, cid in c_ids.items():
                cur.execute("INSERT INTO customers (customer_id) VALUES (%s) ON CONFLICT (customer_id) DO NOTHING;", (cid,))
                cluster_id = 81 if "M81" in name else 80
                cur.execute(
                    """
                    INSERT INTO cluster_assignments (customer_id, period_id, cluster_id, assigned_via)
                    VALUES (%s, %s, %s, 'hdbscan');
                    """,
                    (cid, T, cluster_id)
                )
                
            # Cluster 80 medians: 10.0
            for name, cid in c_ids.items():
                if "M80" in name:
                    cur.execute(
                        """
                        INSERT INTO behavioral_snapshots (customer_id, period_id, login_frequency, session_duration, purchase_count, avg_order_value)
                        VALUES (%s, %s, 10.0, 10.0, 10.0, 10.0);
                        """,
                        (cid, T)
                    )
            # Cluster 81 medians: 20.0
            for name, cid in c_ids.items():
                if "M81" in name:
                    cur.execute(
                        """
                        INSERT INTO behavioral_snapshots (customer_id, period_id, login_frequency, session_duration, purchase_count, avg_order_value)
                        VALUES (%s, %s, 20.0, 20.0, 20.0, 20.0);
                        """,
                        (cid, T)
                    )
            # Customer E: closer to 81 on exactly 2 of the 4 columns
            # Values: [16.0, 16.0, 14.0, 14.0]
            # Drift fraction: 2/4 = 0.50 exactly.
            cur.execute(
                """
                INSERT INTO behavioral_snapshots (customer_id, period_id, login_frequency, session_duration, purchase_count, avg_order_value)
                VALUES (%s, %s, 16.0, 16.0, 14.0, 14.0);
                """,
                (c_ids["E"], T)
            )
            
    # Compute signals
    compute_and_persist_feature_migration_signals(dataset_id, T)
    
    with get_raw_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT drift_fraction, signal_raised 
                FROM feature_migration_signals
                WHERE customer_id = %s AND compared_cluster_id = 81;
                """,
                (c_ids["E"],)
            )
            row = cur.fetchone()
            
    assert row is not None
    assert abs(float(row[0]) - 0.50) < 1e-6
    # Must be strictly False for 50% boundary check (P8-T3)
    assert row[1] is False

def test_p8_t4_median_not_mean():
    """
    P8-T4: Concrete numeric assertion verifying median, not mean.
    - Cluster column values: [1, 2, 3, 4, 1000] -> Median = 3. Mean = 202.
    - Customer C column value: 4.
    - Expected: Distance to comparison cluster median = |4 - 3| = 1.0 (not |4 - 202| = 198.0).
    """
    dataset_id = "00000000-0000-0000-0000-000000000804"
    T = 804
    setup_base_data(dataset_id, T)
    
    # 1 column confirmed
    confirmed_cols = ["login_frequency"]
    confirm_layer8_columns(dataset_id, [{"column_name": "login_frequency", "include": True}])
    
    c_ids = {
        "C": "00000000-0000-0000-0000-00000000800c"
    }
    for i in range(10):
        c_ids[f"M80_{i}"] = f"00000000-0000-0000-0000-00000000804{i}"
    
    # Cluster 81 members: values [1, 2, 3, 4, 1000]
    for i in range(5):
        c_ids[f"M81_{i}"] = f"00000000-0000-0000-0000-00000000814{i}"
    
    with get_raw_connection() as conn:
        with conn.cursor() as cur:
            for name, cid in c_ids.items():
                cur.execute("INSERT INTO customers (customer_id) VALUES (%s) ON CONFLICT (customer_id) DO NOTHING;", (cid,))
                cluster_id = 81 if "M81" in name else 80
                cur.execute(
                    """
                    INSERT INTO cluster_assignments (customer_id, period_id, cluster_id, assigned_via)
                    VALUES (%s, %s, %s, 'hdbscan');
                    """,
                    (cid, T, cluster_id)
                )
                
            # Cluster 80 medians: 6.0 (own median)
            for name, cid in c_ids.items():
                if "M80" in name:
                    cur.execute(
                        """
                        INSERT INTO behavioral_snapshots (customer_id, period_id, login_frequency)
                        VALUES (%s, %s, 6.0);
                        """,
                        (cid, T)
                    )
            
            # Cluster 81: [1.0, 2.0, 3.0, 4.0, 1000.0] -> Median = 3.0, Mean = 202.0
            vals = [1.0, 2.0, 3.0, 4.0, 1000.0]
            for i in range(5):
                cur.execute(
                    """
                    INSERT INTO behavioral_snapshots (customer_id, period_id, login_frequency)
                    VALUES (%s, %s, %s);
                    """,
                    (c_ids[f"M81_{i}"], T, vals[i])
                )
                
            # Customer C has value 4.0
            cur.execute(
                """
                INSERT INTO behavioral_snapshots (customer_id, period_id, login_frequency)
                VALUES (%s, %s, 4.0);
                """,
                (c_ids["C"], T)
            )
            
    # Compute signals
    compute_and_persist_feature_migration_signals(dataset_id, T)
    
    # Let's assert what the outcomes are:
    # Own Cluster 80 median = 6.0. Customer C value = 4.0.
    # d_own = |4.0 - 6.0| = 2.0.
    # If Median is used:
    #   Cluster 81 median = 3.0.
    #   d_compare = |4.0 - 3.0| = 1.0.
    #   d_compare < d_own (1.0 < 2.0) -> Drifted = 1. Drift fraction = 1.0. signal_raised = True.
    # If Mean is used:
    #   Cluster 81 mean = 202.0.
    #   d_compare = |4.0 - 202.0| = 198.0.
    #   d_compare < d_own (198.0 < 2.0 is False) -> Drifted = 0. Drift fraction = 0.0. signal_raised = False.
    with get_raw_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT columns_drifted, drift_fraction, signal_raised 
                FROM feature_migration_signals
                WHERE customer_id = %s AND compared_cluster_id = 81;
                """,
                (c_ids["C"],)
            )
            row = cur.fetchone()
            
    assert row is not None
    # Verify drift was detected using the median distance (1.0 < 2.0) rather than mean distance (198.0 > 2.0)
    assert row[0] == 1
    assert abs(float(row[1]) - 1.0) < 1e-6
    assert row[2] is True

def test_p8_t5_pm_column_confirmation_gate(client):
    """
    P8-T5: If PM column confirmation gate is bypassed (no layer8_config row),
    compute_and_persist_feature_migration_signals raises PipelineBlockedError
    and endpoint returns HTTP 400.
    """
    dataset_id = "00000000-0000-0000-0000-000000000805"
    T = 805
    setup_base_data(dataset_id, T)
    
    # We do NOT confirm columns.
    
    # 1. Assert calling python function directly raises PipelineBlockedError
    with pytest.raises(PipelineBlockedError) as exc_info:
        compute_and_persist_feature_migration_signals(dataset_id, T)
        
    assert "Signal computation blocked" in str(exc_info.value)
    
    # 2. Assert calling HTTP API endpoint returns HTTP 400
    response = client.post(f"/api/v1/datasets/{dataset_id}/periods/{T}/run-migration-signals")
    assert response.status_code == 400
    assert "Signal computation blocked" in response.json()["detail"]
