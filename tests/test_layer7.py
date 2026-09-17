import pytest
from fastapi.testclient import TestClient
from src.main import app
from src.db.connection import get_raw_connection

@pytest.fixture
def client():
    return TestClient(app)

def setup_comparison_data():
    """Utility to clear DB and seed segment outcomes for Layer 7 testing."""
    with get_raw_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM segment_outcomes WHERE period_id IN (901, 902, 903);")
            cur.execute("DELETE FROM cluster_assignments WHERE period_id IN (901, 902, 903);")
            cur.execute("DELETE FROM clusters WHERE baseline_id = 9;")
            cur.execute("DELETE FROM cluster_baselines WHERE baseline_id = 9;")
            cur.execute("DELETE FROM periods WHERE period_id IN (901, 902, 903) OR period_label IN ('P1', 'P2', 'P3');")
            
            # Seed periods (granularity = 'monthly')
            cur.execute(
                """
                INSERT INTO periods (period_id, period_label, period_start, period_end, granularity)
                VALUES 
                    (901, 'P1', '2026-04-01', '2026-04-30', 'monthly'),
                    (902, 'P2', '2026-05-01', '2026-05-31', 'monthly'),
                    (903, 'P3', '2026-06-01', '2026-06-30', 'monthly');
                """
            )
            
            # Seed baseline
            cur.execute(
                """
                INSERT INTO cluster_baselines (baseline_id, is_active, adopted_reason, period_id)
                VALUES (9, TRUE, 'comparison_baseline', 901);
                """
            )
            
            # Seed clusters 901, 902, 903
            cur.execute("INSERT INTO clusters (cluster_id, baseline_id, cluster_label) VALUES (901, 9, 'Cluster_1');")
            cur.execute("INSERT INTO clusters (cluster_id, baseline_id, cluster_label) VALUES (902, 9, 'Cluster_2');")
            cur.execute("INSERT INTO clusters (cluster_id, baseline_id, cluster_label) VALUES (903, 9, 'Cluster_3');")
            
            # Seed segment outcomes
            # Cluster 901: P1=10.0, P2=12.0, P3=15.0
            # Cluster 902: P1=8.0, P2=9.0, P3=11.0
            # Cluster 903: P1=5.0, P2=6.0, P3=7.0
            cur.execute(
                """
                INSERT INTO segment_outcomes (cluster_id, period_id, churn_rate, retention_rate, revenue, repeat_purchase_prob)
                VALUES 
                    (901, 901, 10.0, 90.0, 1000.0, 0.8),
                    (901, 902, 12.0, 88.0, 1100.0, 0.75),
                    (901, 903, 15.0, 85.0, 1200.0, 0.7),
                    
                    (902, 901, 8.0, 92.0, 800.0, 0.9),
                    (902, 902, 9.0, 91.0, 850.0, 0.88),
                    (902, 903, 11.0, 89.0, 900.0, 0.85),
                    
                    (903, 901, 5.0, 95.0, 500.0, 0.95),
                    (903, 902, 6.0, 94.0, 550.0, 0.93),
                    (903, 903, 7.0, 93.0, 600.0, 0.91);
                """
            )

def test_p9_t1_t2_comparison_matrix_and_gap_row(client):
    """
    P9-T1: Comparison API returns correct matrix.
    P9-T2: Gap row is present for exactly 2 segments with correct values and signs.
    """
    setup_comparison_data()
    
    url = "/api/v1/metrics/comparison?granularity=monthly&metric=churn_rate&segment_ids=901,902&period_range=P1,P3"
    response = client.get(url)
    assert response.status_code == 200
    
    res_data = response.json()
    assert res_data["periods"] == ["P1", "P2", "P3"]
    assert len(res_data["data"]) == 3  # Segment 901, Segment 902, and Gap Row
    
    row_1 = next(r for r in res_data["data"] if r["segment_id"] == 901)
    row_2 = next(r for r in res_data["data"] if r["segment_id"] == 902)
    row_gap = next(r for r in res_data["data"] if r["segment_id"] == "gap")
    
    # Check segment 901 values (P9-T1 check)
    assert row_1["values"] == [10.0, 12.0, 15.0]
    
    # Check segment 902 values (P9-T1 check)
    assert row_2["values"] == [8.0, 9.0, 11.0]
    
    # Check gap row calculations and signs (P9-T2 check)
    # P1 gap: 10.0 - 8.0 = 2.0 -> "+2.0"
    # P2 gap: 12.0 - 9.0 = 3.0 -> "+3.0"
    # P3 gap: 15.0 - 11.0 = 4.0 -> "+4.0"
    assert row_gap["values"] == ["+2.0", "+3.0", "+4.0"]

def test_p9_t3_gap_row_absent_for_three_plus_segments(client):
    """
    P9-T3: Gap row is absent when 3 or more segments are selected.
    """
    setup_comparison_data()
    
    url = "/api/v1/metrics/comparison?granularity=monthly&metric=churn_rate&segment_ids=901,902,903&period_range=P1,P3"
    response = client.get(url)
    assert response.status_code == 200
    
    res_data = response.json()
    assert res_data["periods"] == ["P1", "P2", "P3"]
    assert len(res_data["data"]) == 3  # Only Segment 901, Segment 902, and Segment 903
    
    # Confirm no gap row
    segment_ids_in_response = [r["segment_id"] for r in res_data["data"]]
    assert "gap" not in segment_ids_in_response
    assert set(segment_ids_in_response) == {901, 902, 903}

def test_p9_t4_single_segment(client):
    """
    P9-T4: Case with exactly 1 segment should return raw row and no gap row.
    """
    setup_comparison_data()
    
    url = "/api/v1/metrics/comparison?granularity=monthly&metric=churn_rate&segment_ids=901&period_range=P1,P3"
    response = client.get(url)
    assert response.status_code == 200
    
    res_data = response.json()
    assert len(res_data["data"]) == 1
    assert res_data["data"][0]["segment_id"] == 901
    assert res_data["data"][0]["values"] == [10.0, 12.0, 15.0]

def test_p9_t5_zero_segments(client):
    """
    P9-T5: Empty segment_ids list returns 400 Bad Request.
    """
    setup_comparison_data()
    
    # Empty segment_ids string
    url = "/api/v1/metrics/comparison?granularity=monthly&metric=churn_rate&segment_ids=&period_range=P1,P3"
    response = client.get(url)
    assert response.status_code == 400
    assert "At least one segment_id is required" in response.json()["detail"]
    
    # Spaces only segment_ids string
    url = "/api/v1/metrics/comparison?granularity=monthly&metric=churn_rate&segment_ids=  &period_range=P1,P3"
    response = client.get(url)
    assert response.status_code == 400
    assert "At least one segment_id is required" in response.json()["detail"]

def test_p9_t6_cluster_identity_lineage_resolution(client):
    """
    P9-T6: When a new baseline is adopted and new cluster IDs are created,
    Layer 7 resolves historical outcomes via cluster_identity_map ('matched').
    """
    setup_comparison_data()
    
    with get_raw_connection() as conn:
        with conn.cursor() as cur:
            # Baseline 10 adopted, new cluster 911 corresponds to old cluster 901
            cur.execute("DELETE FROM cluster_identity_map WHERE old_cluster_id = 901 OR new_cluster_id = 911;")
            cur.execute(
                """
                INSERT INTO cluster_baselines (baseline_id, is_active, adopted_reason, period_id)
                VALUES (10, TRUE, 'q2_rebaseline', 903)
                ON CONFLICT (baseline_id) DO NOTHING;
                """
            )
            cur.execute("INSERT INTO clusters (cluster_id, baseline_id, cluster_label) VALUES (911, 10, 'Cluster_1_New') ON CONFLICT DO NOTHING;")
            cur.execute("INSERT INTO cluster_identity_map (old_cluster_id, new_cluster_id, match_type) VALUES (901, 911, 'matched');")
            
            # Record an outcome for new cluster 911 in period P3, while P1 and P2 outcomes belong to cluster 901
            cur.execute("DELETE FROM segment_outcomes WHERE cluster_id = 901 AND period_id = 903;")
            cur.execute(
                """
                INSERT INTO segment_outcomes (cluster_id, period_id, churn_rate, retention_rate, revenue, repeat_purchase_prob)
                VALUES (911, 903, 15.0, 85.0, 1200.0, 0.7)
                ON CONFLICT (cluster_id, period_id) DO UPDATE SET churn_rate = EXCLUDED.churn_rate;
                """
            )
    
    # Query with new segment ID 911 across P1..P3
    url = "/api/v1/metrics/comparison?granularity=monthly&metric=churn_rate&segment_ids=911&period_range=P1,P3"
    response = client.get(url)
    assert response.status_code == 200
    res_data = response.json()
    assert res_data["periods"] == ["P1", "P2", "P3"]
    assert len(res_data["data"]) == 1
    # Historical P1 (10.0) and P2 (12.0) from cluster 901 are resolved for cluster 911 along with P3 (15.0)
    assert res_data["data"][0]["segment_id"] == 911
    assert res_data["data"][0]["values"] == [10.0, 12.0, 15.0]
