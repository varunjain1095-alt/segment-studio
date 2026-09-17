import pytest
from fastapi.testclient import TestClient
from src.main import app
from src.db.connection import get_raw_connection
from src.layer5.migration_forecaster import compute_and_persist_transition_matrix

@pytest.fixture
def client():
    return TestClient(app)

def test_p7_t1_t2_t3_migration_forecasting(client):
    """
    Combined test verifying Phase 7 transition matrix properties:
    - P7-T1: Transition Matrix Row Sums to 1.0 for populated clusters.
    - P7-T2: Disappeared/empty cluster handling (Option B: row is completely absent).
    - P7-T3: Self-transition probabilities are correctly computed and stored.
    """
    dataset_id = "00000000-0000-0000-0000-000000000707"
    T = 707
    
    with get_raw_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM transition_matrix WHERE from_period_id IN (%s, %s) OR to_period_id IN (%s, %s);", (T, T + 1, T, T + 1))
            cur.execute("DELETE FROM cluster_assignments WHERE period_id IN (%s, %s);", (T, T + 1))
            cur.execute("DELETE FROM clusters WHERE baseline_id = 7;")
            cur.execute("DELETE FROM cluster_baselines WHERE baseline_id = 7;")
            cur.execute("DELETE FROM periods WHERE period_id IN (%s, %s);", (T, T + 1))
            cur.execute("DELETE FROM datasets WHERE dataset_id = %s;", (dataset_id,))
            
            # Seed periods (FK constraints)
            cur.execute("INSERT INTO periods (period_id, period_label, period_start, period_end, granularity) VALUES (%s, 'T', '2026-04-01', '2026-06-30', 'quarterly');", (T,))
            cur.execute("INSERT INTO periods (period_id, period_label, period_start, period_end, granularity) VALUES (%s, 'T+1', '2026-07-01', '2026-09-30', 'quarterly');", (T + 1,))
            
            cur.execute("INSERT INTO datasets (dataset_id, source_type, permanence, layer0_completed) VALUES (%s, 'file_upload', 'permanent', TRUE);", (dataset_id,))
            cur.execute("INSERT INTO cluster_baselines (baseline_id, is_active, adopted_reason, period_id) VALUES (7, TRUE, 'initial', %s);", (T,))
            
            # Create three clusters:
            # Cluster 70 (populated in T)
            # Cluster 71 (populated in T)
            # Cluster 72 (disappeared - exists in database, but has 0 assigned customers at T)
            cur.execute("INSERT INTO clusters (cluster_id, baseline_id, cluster_label) VALUES (70, 7, 'Cluster_A');")
            cur.execute("INSERT INTO clusters (cluster_id, baseline_id, cluster_label) VALUES (71, 7, 'Cluster_B');")
            cur.execute("INSERT INTO clusters (cluster_id, baseline_id, cluster_label) VALUES (72, 7, 'Cluster_Disappeared');")
            
            # Setup assignments at T and T+1
            # Cluster 70 has 5 customers:
            #   4 remain in 70 (80% self-transition)
            #   1 transitions to 71 (20% transition)
            # Cluster 71 has 10 customers:
            #   10 remain in 71 (100% self-transition)
            # Cluster 72 has 0 customers assigned at T
            
            cids = [f"00000000-0000-0000-0000-0000000077{i:02d}" for i in range(15)]
            for cid in cids:
                cur.execute("INSERT INTO customers (customer_id) VALUES (%s);", (cid,))
                
            # Cluster 70 members (cids 0 to 4)
            for i in range(5):
                cid = cids[i]
                cur.execute("INSERT INTO cluster_assignments (customer_id, period_id, cluster_id, assigned_via) VALUES (%s, %s, 70, 'hdbscan');", (cid, T))
                to_cid = 70 if i < 4 else 71
                cur.execute("INSERT INTO cluster_assignments (customer_id, period_id, cluster_id, assigned_via) VALUES (%s, %s, %s, 'hdbscan');", (cid, T + 1, to_cid))
                
            # Cluster 71 members (cids 5 to 14)
            for i in range(5, 15):
                cid = cids[i]
                cur.execute("INSERT INTO cluster_assignments (customer_id, period_id, cluster_id, assigned_via) VALUES (%s, %s, 71, 'hdbscan');", (cid, T))
                cur.execute("INSERT INTO cluster_assignments (customer_id, period_id, cluster_id, assigned_via) VALUES (%s, %s, 71, 'hdbscan');", (cid, T + 1))
                
    # 1. Run Layer 5 transition matrix computation
    results = compute_and_persist_transition_matrix(dataset_id, T)
    
    # 2. Check P7-T1: row probabilities sum to exactly 1.0 (for present source clusters)
    # Check A (70): P(70->70)=0.8, P(70->71)=0.2 -> sum = 1.0
    # Check B (71): P(71->70)=0.0, P(71->71)=1.0 -> sum = 1.0
    from_70 = [r for r in results if r["from_cluster_id"] == 70]
    from_71 = [r for r in results if r["from_cluster_id"] == 71]
    
    assert len(from_70) == 2
    assert len(from_71) == 2
    assert abs(sum(r["transition_prob"] for r in from_70) - 1.0) < 1e-6
    assert abs(sum(r["transition_prob"] for r in from_71) - 1.0) < 1e-6
    
    # 3. Check P7-T2: disappeared/empty cluster 72 is completely absent (Option B)
    from_72 = [r for r in results if r["from_cluster_id"] == 72]
    assert len(from_72) == 0
    
    with get_raw_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM transition_matrix WHERE from_cluster_id = 72;")
            count_72 = cur.fetchone()[0]
    assert count_72 == 0
    
    # 4. Check P7-T3: self-transitions are correct
    cell_70_70 = next(r for r in from_70 if r["to_cluster_id"] == 70)
    cell_71_71 = next(r for r in from_71 if r["to_cluster_id"] == 71)
    
    assert abs(cell_70_70["transition_prob"] - 0.8) < 1e-6
    assert abs(cell_71_71["transition_prob"] - 1.0) < 1e-6
    
    # 5. Check GET endpoint
    response = client.get(f"/api/v1/transition-matrix?from_period_id={T}&to_period_id={T+1}")
    assert response.status_code == 200
    data = response.json()
    
    # Should only return cells for 70 and 71
    assert len(data) == 4
    # No row for 72 present in the response
    assert not any(cell["from_cluster_id"] == 72 for cell in data)

    # 6. Check compute_mobility_metrics
    from src.layer5.migration_forecaster import compute_mobility_metrics, compute_horizon_matrix
    mobility = compute_mobility_metrics(T, T + 1)
    assert "retention_stability_pct" in mobility
    assert mobility["total_transitioned_customers"] == 15
    assert mobility["stable_count"] == 14 # 4 in 70 + 10 in 71

    # 7. Check compute_horizon_matrix
    h_matrix = compute_horizon_matrix(T, T + 1, days=180)
    assert len(h_matrix) > 0
    assert all("transition_prob" in cell for cell in h_matrix)

    # 8. Check GET mobility metrics endpoint
    mob_res = client.get(f"/api/v1/migration-forecasting/mobility-metrics?from_period_id={T}&to_period_id={T+1}")
    assert mob_res.status_code == 200
    assert mob_res.json()["metrics"]["total_transitioned_customers"] == 15

    # 9. Check GET horizon endpoint
    h_res = client.get(f"/api/v1/transition-matrix/horizon?from_period_id={T}&to_period_id={T+1}&days=90")
    assert h_res.status_code == 200
    assert len(h_res.json()["matrix"]) > 0

def test_compute_horizon_matrix_math():
    """Verify matrix exponentiation and mobility math with mock data."""
    from unittest.mock import patch, MagicMock
    from src.layer5.migration_forecaster import compute_mobility_metrics, compute_horizon_matrix
    
    # Mock database rows: Cluster 1 and 2
    # 1 -> 1: 0.8 (80 customers)
    # 1 -> 2: 0.2 (20 customers)
    # 2 -> 1: 0.1 (10 customers)
    # 2 -> 2: 0.9 (90 customers)
    mock_rows = [
        (1, 1, 80, 0.8),
        (1, 2, 20, 0.2),
        (2, 1, 10, 0.1),
        (2, 2, 90, 0.9)
    ]
    
    mock_conn = MagicMock()
    mock_cur = MagicMock()
    mock_cur.fetchall.return_value = mock_rows
    mock_conn.__enter__.return_value = mock_conn
    mock_conn.cursor.return_value.__enter__.return_value = mock_cur
    
    with patch("src.layer5.migration_forecaster.get_raw_connection", return_value=mock_conn):
        mob = compute_mobility_metrics(1, 2)
        assert mob["total_transitioned_customers"] == 200
        assert mob["stable_count"] == 170 # 80 + 90
        assert mob["retention_stability_pct"] == 85.0
        
        # Test 90 days (P^1 = P)
        h_90 = compute_horizon_matrix(1, 2, days=90)
        assert len(h_90) == 4
        p11_90 = next(c["transition_prob"] for c in h_90 if c["from_cluster_id"] == 1 and c["to_cluster_id"] == 1)
        assert abs(p11_90 - 0.8) < 1e-4
        
        # Test 180 days (P^2 = P x P)
        # P = [[0.8, 0.2], [0.1, 0.9]]
        # P^2 = [[0.8*0.8 + 0.2*0.1, 0.8*0.2 + 0.2*0.9]] = [[0.66, 0.34], [0.17, 0.83]]
        h_180 = compute_horizon_matrix(1, 2, days=180)
        p11_180 = next(c["transition_prob"] for c in h_180 if c["from_cluster_id"] == 1 and c["to_cluster_id"] == 1)
        assert abs(p11_180 - 0.66) < 1e-3

def test_layer5_api_endpoints_mocked(client):
    """Verify Layer 5 mobility and horizon API endpoints with mocked backend functions."""
    from unittest.mock import patch
    
    with patch("src.layer5.migration_forecaster.compute_mobility_metrics") as mock_mob:
        mock_mob.return_value = {
            "retention_stability_pct": 78.5,
            "upward_mobility_pct": 12.0,
            "downward_drift_pct": 9.5,
            "total_transitioned_customers": 5000
        }
        res = client.get("/api/v1/migration-forecasting/mobility-metrics?from_period_id=1&to_period_id=2")
        assert res.status_code == 200
        assert res.json()["metrics"]["retention_stability_pct"] == 78.5
        
    with patch("src.layer5.migration_forecaster.compute_horizon_matrix") as mock_hz:
        mock_hz.return_value = [
            {"from_cluster_id": 1, "to_cluster_id": 1, "transition_prob": 0.75, "horizon_days": 60}
        ]
        res = client.get("/api/v1/transition-matrix/horizon?from_period_id=1&to_period_id=2&days=60")
        assert res.status_code == 200
        assert res.json()["matrix"][0]["transition_prob"] == 0.75


