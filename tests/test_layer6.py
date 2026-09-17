import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from src.main import app
from src.db.connection import get_raw_connection
from src.layer6.outcome_analyzer import compute_and_persist_segment_outcomes

@pytest.fixture
def client():
    return TestClient(app)

def test_p5_t1_t2_t3_segment_outcomes_execution(client):
    """
    Combined test verifying P5-T1 (Churn + Retention = 100.0%), P5-T2 (all four metrics non-null),
    and P5-T3 (GET endpoints retrieve matching persisted data).
    """
    dataset_id = "00000000-0000-0000-0000-000000000505"
    period_id = 505
    
    # Reset and seed database public tables
    with get_raw_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM segment_outcomes WHERE period_id IN (%s, %s);", (period_id, period_id + 1))
            cur.execute("DELETE FROM cluster_assignments WHERE period_id IN (%s, %s);", (period_id, period_id + 1))
            cur.execute("DELETE FROM clusters WHERE baseline_id = 5;")
            cur.execute("DELETE FROM cluster_baselines WHERE baseline_id = 5;")
            cur.execute("DELETE FROM demographic_snapshots WHERE period_id IN (%s, %s);", (period_id, period_id + 1))
            cur.execute("DELETE FROM behavioral_snapshots WHERE period_id IN (%s, %s);", (period_id, period_id + 1))
            cur.execute("DELETE FROM periods WHERE period_id IN (%s, %s);", (period_id, period_id + 1))
            cur.execute("DELETE FROM datasets WHERE dataset_id = %s;", (dataset_id,))
            
            # Seed periods (FK constraints)
            cur.execute("INSERT INTO periods (period_id, period_label, period_start, period_end, granularity) VALUES (%s, 'T', '2026-04-01', '2026-06-30', 'quarterly');", (period_id,))
            cur.execute("INSERT INTO periods (period_id, period_label, period_start, period_end, granularity) VALUES (%s, 'T+1', '2026-07-01', '2026-09-30', 'quarterly');", (period_id + 1,))
            
            cur.execute(
                """
                INSERT INTO datasets (dataset_id, source_type, permanence, layer0_completed)
                VALUES (%s, 'file_upload', 'permanent', TRUE);
                """,
                (dataset_id,)
            )
            cur.execute(
                """
                INSERT INTO cluster_baselines (baseline_id, is_active, adopted_reason, period_id)
                VALUES (5, TRUE, 'initial', %s);
                """,
                (period_id,)
            )
            # Create two clusters
            cur.execute("INSERT INTO clusters (cluster_id, baseline_id, cluster_label) VALUES (50, 5, 'Cluster_0');")
            cur.execute("INSERT INTO clusters (cluster_id, baseline_id, cluster_label) VALUES (51, 5, 'Cluster_1');")
            
            # Create 10 customers and snapshots, assigning 5 to Cluster 50 and 5 to Cluster 51
            for i in range(10):
                cid = f"00000000-0000-0000-0000-0000000055{i:02d}"
                cur.execute("INSERT INTO customers (customer_id) VALUES (%s);", (cid,))
                cluster_id = 50 if i < 5 else 51
                
                cur.execute(
                    """
                    INSERT INTO cluster_assignments (customer_id, period_id, cluster_id, assigned_via)
                    VALUES (%s, %s, %s, 'hdbscan');
                    """,
                    (cid, period_id, cluster_id)
                )
                
                # Make some active in T+1 (determining retention vs churn)
                # Cluster 50: 3 active at T+1, 2 churned (absent)
                # Cluster 51: 4 active at T+1, 1 churned (absent)
                is_active_next = False
                if i < 5 and i < 3: # cust 0,1,2
                    is_active_next = True
                elif i >= 5 and i < 9: # cust 5,6,7,8
                    is_active_next = True
                    
                cur.execute(
                    """
                    INSERT INTO behavioral_snapshots (customer_id, period_id, purchase_count, avg_order_value)
                    VALUES (%s, %s, %s, 100.0);
                    """,
                    (cid, period_id, 2 if i % 2 == 0 else 1) # alternating repeat purchases
                )
                
                if is_active_next:
                    cur.execute(
                        """
                        INSERT INTO behavioral_snapshots (customer_id, period_id, purchase_count, avg_order_value)
                        VALUES (%s, %s, 1.0, 100.0);
                        """,
                        (cid, period_id + 1)
                    )

    # 1. Run Layer 6 outcome calculations
    results = compute_and_persist_segment_outcomes(dataset_id, period_id)
    assert len(results) == 2
    
    # 2. Check P5-T1 (sum of churn and retention is exactly 100.0%)
    for res in results:
        assert abs((res["churn_rate"] + res["retention_rate"]) - 100.0) < 1e-6
        
    # Find results per cluster
    res_50 = next(r for r in results if r["cluster_id"] == 50)
    res_51 = next(r for r in results if r["cluster_id"] == 51)
    
    # Cluster 50: 3 active at T+1, 2 churned -> 3/5=60% retention, 2/5=40% churn
    assert abs(res_50["retention_rate"] - 60.0) < 1e-6
    assert abs(res_50["churn_rate"] - 40.0) < 1e-6
    
    # Cluster 51: 4 active at T+1, 1 churned -> 4/5=80% retention, 1/5=20% churn
    assert abs(res_51["retention_rate"] - 80.0) < 1e-6
    assert abs(res_51["churn_rate"] - 20.0) < 1e-6
    
    # 3. Check P5-T2 (verify persisted database values are non-null)
    with get_raw_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT churn_rate, retention_rate, revenue, repeat_purchase_prob FROM segment_outcomes;")
            db_rows = cur.fetchall()
            
    assert len(db_rows) == 2
    for row in db_rows:
        assert row[0] is not None # churn
        assert row[1] is not None # retention
        assert row[2] is not None # revenue
        assert row[3] is not None # repeat purchase probability

    # 4. Check P5-T3 (GET endpoints fetch correct data matching DB)
    response = client.get(f"/api/v1/segments/50/outcomes?period_id={period_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["cluster_id"] == 50
    assert data["period_id"] == period_id
    assert abs(data["churn_rate"] - 40.0) < 1e-6
    assert abs(data["retention_rate"] - 60.0) < 1e-6
    assert data["revenue"] is not None
    assert data["repeat_purchase_prob"] is not None
    
    # Fetch for invalid cluster should return 404
    invalid_resp = client.get(f"/api/v1/segments/999/outcomes?period_id={period_id}")
    assert invalid_resp.status_code == 404
