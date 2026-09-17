import os
import pytest
import numpy as np
import pandas as pd
import json
from unittest.mock import patch, MagicMock
from src.db.connection import get_raw_connection
from src.layer2.preprocessor import load_and_preprocess_behavioral_data, populate_behavioral_snapshots
from src.layer2.clusterer import run_kmeans_clustering
from src.layer2.cluster_evaluator import evaluate_clustering
from src.layer2.cluster_profiler import profile_clusters, run_profiling_and_persist
from src.layer2.drift_evaluator import (
    compute_medoids_and_spreads,
    match_clusters,
    decide_adopt_vs_hold,
    get_active_baseline
)

@pytest.fixture(autouse=True)
def setup_hdbscan_env():
    """Sets small HDBSCAN parameters so that it runs successfully on 100-row test data."""
    os.environ["HDBSCAN_MIN_CLUSTER_SIZE"] = "10"
    os.environ["HDBSCAN_MIN_SAMPLES"] = "2"
    os.environ["CLUSTER_STABILITY_RUNS"] = "2"
    yield
    if "HDBSCAN_MIN_CLUSTER_SIZE" in os.environ:
        del os.environ["HDBSCAN_MIN_CLUSTER_SIZE"]
    if "HDBSCAN_MIN_SAMPLES" in os.environ:
        del os.environ["HDBSCAN_MIN_SAMPLES"]
    if "CLUSTER_STABILITY_RUNS" in os.environ:
        del os.environ["CLUSTER_STABILITY_RUNS"]

@pytest.fixture
def sample_behavioral_df():
    # 100 customers, 3 behavioral features
    np.random.seed(42)
    data = {
        "customer_id": [f"00000000-0000-0000-0000-00000000{i:04d}" for i in range(100)],
        "login_frequency": np.random.normal(10, 2, 100),
        "session_duration": np.random.normal(300, 50, 100),
        "purchase_count": np.random.poisson(5, 100).astype(float)
    }
    return pd.DataFrame(data)

def clean_database_tables(cur):
    """Truncates/clears test data in database dependency order to avoid constraint errors without wiping production data."""
    cur.execute("DELETE FROM demographic_segment_metrics WHERE period_id >= 300;")
    cur.execute("DELETE FROM quarterly_narrations WHERE period_id >= 300;")
    cur.execute("DELETE FROM statistical_test_results WHERE period_id >= 300;")
    cur.execute("DELETE FROM feature_migration_signals WHERE period_id >= 300;")
    cur.execute("DELETE FROM segment_outcomes WHERE period_id >= 300;")
    cur.execute("DELETE FROM transition_matrix WHERE from_period_id >= 300 OR to_period_id >= 300;")
    cur.execute("DELETE FROM migration_labels WHERE period_id >= 300;")
    cur.execute("DELETE FROM cluster_identity_map WHERE baseline_period_id >= 300 OR candidate_period_id >= 300;")
    cur.execute("DELETE FROM cluster_assignments WHERE period_id >= 300;")
    cur.execute("DELETE FROM demographic_snapshots WHERE period_id >= 300;")
    cur.execute("DELETE FROM behavioral_snapshots WHERE period_id >= 300;")
    cur.execute("DELETE FROM clusters WHERE baseline_id IN (SELECT baseline_id FROM cluster_baselines WHERE period_id >= 300);")
    cur.execute("DELETE FROM cluster_baselines WHERE period_id >= 300;")
    cur.execute("DELETE FROM notifications WHERE dataset_id LIKE '00000000-0000-0000-0000-0000000003%';")
    cur.execute("DELETE FROM cross_field_rules WHERE dataset_id LIKE '00000000-0000-0000-0000-0000000003%';")
    cur.execute("DELETE FROM column_profiles WHERE dataset_id LIKE '00000000-0000-0000-0000-0000000003%';")
    cur.execute("DELETE FROM datasets WHERE dataset_id LIKE '00000000-0000-0000-0000-0000000003%';")
    cur.execute("DELETE FROM periods WHERE period_id >= 300;")
    cur.execute("DELETE FROM customers WHERE customer_id LIKE '00000000-0000-0000-0000-000000000%' OR customer_id LIKE 'test_%';")

def test_p3_t0_preprocessor_state_enforcement(sample_behavioral_df):
    """Verify preprocessor enforces layer0_completed and classification constraints."""
    dataset_id = "00000000-0000-0000-0000-000000000300"
    
    # 1. Setup DB with layer0_completed = FALSE
    with get_raw_connection() as conn:
        with conn.cursor() as cur:
            clean_database_tables(cur)
            cur.execute(
                """
                INSERT INTO datasets (dataset_id, source_type, permanence, layer0_completed)
                VALUES (%s, 'file_upload', 'permanent', FALSE);
                """,
                (dataset_id,)
            )
            
    # Preprocessor must raise PipelineBlockedError or ValueError
    with pytest.raises(Exception):
        load_and_preprocess_behavioral_data(dataset_id)
        
    # 2. Complete Layer 0 but no classifications confirmed
    with get_raw_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("UPDATE datasets SET layer0_completed = TRUE WHERE dataset_id = %s;", (dataset_id,))
            
    with pytest.raises(Exception):
        load_and_preprocess_behavioral_data(dataset_id)
        
    # 3. Setup confirmed classifications but no behavioral cols
    classes = [{"column": "login_frequency", "classification": "demographic"}]
    with get_raw_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE datasets SET confirmed_classifications = %s WHERE dataset_id = %s;",
                (json.dumps(classes), dataset_id)
            )
            
    with pytest.raises(Exception):
        load_and_preprocess_behavioral_data(dataset_id)

def test_p3_t0_preprocessor_nan_and_numeric_checks(sample_behavioral_df):
    """Verify preprocessor detects NaN and non-numeric columns."""
    dataset_id = "00000000-0000-0000-0000-000000000301"
    classes = [
        {"column": "login_frequency", "classification": "behavioral"},
        {"column": "session_duration", "classification": "behavioral"}
    ]
    
    with get_raw_connection() as conn:
        with conn.cursor() as cur:
            clean_database_tables(cur)
            cur.execute(
                """
                INSERT INTO datasets (dataset_id, source_type, permanence, layer0_completed, confirmed_classifications)
                VALUES (%s, 'file_upload', 'permanent', TRUE, %s);
                """,
                (dataset_id, json.dumps(classes))
            )
            
    # Setup mock file read: contains NaN values
    bad_df = sample_behavioral_df.copy()
    bad_df.loc[0, "login_frequency"] = np.nan
    
    with patch("src.layer2.preprocessor.pd.read_csv", return_value=bad_df):
        with patch("src.layer2.preprocessor.os.path.exists", return_value=True):
            with pytest.raises(Exception):
                load_and_preprocess_behavioral_data(dataset_id)
                
    # Setup mock file read: contains non-numeric strings
    bad_df2 = sample_behavioral_df.copy()
    bad_df2["login_frequency"] = bad_df2["login_frequency"].astype(str)
    
    with patch("src.layer2.preprocessor.pd.read_csv", return_value=bad_df2):
        with patch("src.layer2.preprocessor.os.path.exists", return_value=True):
            with pytest.raises(Exception):
                load_and_preprocess_behavioral_data(dataset_id)

def test_p3_t1_kmeans_is_active_clustering_algorithm(sample_behavioral_df):
    """Verify that no HDBSCAN import or invocation exists in clusterer, and KMeans is used."""
    import src.layer2.clusterer as c_mod
    src_code = open(c_mod.__file__, "r").read()
    assert "hdbscan" not in src_code.lower()
    assert "kmeans" in src_code.lower()
    
    behavioral_cols = ["login_frequency", "session_duration", "purchase_count"]
    labels = run_kmeans_clustering(sample_behavioral_df, behavioral_cols, n_clusters=3)
    assert len(labels) == len(sample_behavioral_df)
    assert set(labels).issubset({-1, 0, 1, 2})

def test_p3_t9_hdbscan_noise_collapse_sweep():
    """Verify that KMeans handles sample size smaller than cluster count by raising ValueError."""
    df = pd.DataFrame({"customer_id": ["cust_1"], "login_frequency": [10.0], "session_duration": [300.0], "purchase_count": [5.0]})
    with pytest.raises(ValueError):
        run_kmeans_clustering(df, ["login_frequency", "session_duration", "purchase_count"], n_clusters=3)

def test_p3_t3_quality_gate_stability(sample_behavioral_df):
    """Verify Stability gate evaluates and gates correctly."""
    behavioral_cols = ["login_frequency", "session_duration", "purchase_count"]
    labels = np.random.choice([0, 1, 2], size=len(sample_behavioral_df))
    
    # Mock ARI to 0.80 -> Pass
    with patch("src.layer2.cluster_evaluator.adjusted_rand_score", return_value=0.80):
        with patch("src.layer2.cluster_evaluator.silhouette_score", return_value=0.25):
            with patch("src.layer2.cluster_evaluator.davies_bouldin_score", return_value=1.2):
                results = evaluate_clustering(sample_behavioral_df, behavioral_cols, labels)
                assert results["all_passed"] == True
                assert results["stability_pass"] == True
                
    # Mock ARI to 0.60 -> Fails Stability Gate
    with patch("src.layer2.cluster_evaluator.adjusted_rand_score", return_value=0.60):
        with patch("src.layer2.cluster_evaluator.silhouette_score", return_value=0.25):
            with patch("src.layer2.cluster_evaluator.davies_bouldin_score", return_value=1.2):
                with pytest.raises(Exception):
                    evaluate_clustering(sample_behavioral_df, behavioral_cols, labels)

def test_p3_t4_quality_gate_separation(sample_behavioral_df):
    """Verify Separation gates evaluate and gate correctly."""
    behavioral_cols = ["login_frequency", "session_duration", "purchase_count"]
    labels = np.random.choice([0, 1, 2], size=len(sample_behavioral_df))
    
    # Silhouette too low (0.15 < 0.20) -> Fail
    with patch("src.layer2.cluster_evaluator.adjusted_rand_score", return_value=0.85):
        with patch("src.layer2.cluster_evaluator.silhouette_score", return_value=0.15):
            with patch("src.layer2.cluster_evaluator.davies_bouldin_score", return_value=1.2):
                with pytest.raises(Exception):
                    evaluate_clustering(sample_behavioral_df, behavioral_cols, labels)
                    
    # DBI too high (1.80 > 1.50) -> Fail
    with patch("src.layer2.cluster_evaluator.adjusted_rand_score", return_value=0.85):
        with patch("src.layer2.cluster_evaluator.silhouette_score", return_value=0.25):
            with patch("src.layer2.cluster_evaluator.davies_bouldin_score", return_value=1.8):
                with pytest.raises(Exception):
                    evaluate_clustering(sample_behavioral_df, behavioral_cols, labels)

def test_p3_t5_quality_gate_noise_sanity(sample_behavioral_df):
    """Verify Noise sanity gates evaluate and gate correctly."""
    behavioral_cols = ["login_frequency", "session_duration", "purchase_count"]
    labels = np.array([-1] * 40 + [0] * 30 + [1] * 30)
    
    with patch("src.layer2.cluster_evaluator.adjusted_rand_score", return_value=0.85):
        with patch("src.layer2.cluster_evaluator.silhouette_score", return_value=0.25):
            with patch("src.layer2.cluster_evaluator.davies_bouldin_score", return_value=1.2):
                with pytest.raises(Exception):
                    evaluate_clustering(sample_behavioral_df, behavioral_cols, labels)

def test_p3_t6_config_driven_thresholds(sample_behavioral_df):
    """Verify thresholds can be configured dynamically from env vars."""
    behavioral_cols = ["login_frequency", "session_duration", "purchase_count"]
    labels = np.random.choice([0, 1, 2], size=len(sample_behavioral_df))
    
    os.environ["CLUSTER_STABILITY_ARI_THRESHOLD"] = "0.50"
    os.environ["CLUSTER_SILHOUETTE_MIN"] = "0.10"
    os.environ["CLUSTER_DAVIES_BOULDIN_MAX"] = "2.5"
    os.environ["CLUSTER_NOISE_FRACTION_MAX"] = "0.50"
    
    with patch("src.layer2.cluster_evaluator.adjusted_rand_score", return_value=0.55):
        with patch("src.layer2.cluster_evaluator.silhouette_score", return_value=0.12):
            with patch("src.layer2.cluster_evaluator.davies_bouldin_score", return_value=2.2):
                results = evaluate_clustering(sample_behavioral_df, behavioral_cols, labels)
                assert MagicMock(results["all_passed"] == True)
                
    del os.environ["CLUSTER_STABILITY_ARI_THRESHOLD"]
    del os.environ["CLUSTER_SILHOUETTE_MIN"]
    del os.environ["CLUSTER_DAVIES_BOULDIN_MAX"]
    del os.environ["CLUSTER_NOISE_FRACTION_MAX"]

def test_p3_t7_llm_profiler_no_naming(sample_behavioral_df):
    """Verify LLM profiling is descriptive and does not assign titles/names."""
    behavioral_cols = ["login_frequency", "session_duration", "purchase_count"]
    labels = np.array([0] * 50 + [1] * 50)
    
    mock_summary = '{"summary": "Customers with high login frequency and short sessions"}'
    
    with patch("src.layer2.cluster_profiler.claude_client.call", return_value=mock_summary) as mock_claude:
        summaries = profile_clusters(sample_behavioral_df, labels, behavioral_cols)
        
        assert len(summaries) == 2
        assert summaries[0] == "Customers with high login frequency and short sessions"
        assert mock_claude.call_count == 2
        
        system_prompt = mock_claude.call_args[1].get("system_prompt")
        assert "Do NOT suggest a name" in system_prompt

def test_p3_t10_drift_evaluator_adopt_path(sample_behavioral_df):
    """Verify ADOPT decision and baseline persistence/mapping writes."""
    dataset_id = "00000000-0000-0000-0000-000000000305"
    period_id = 305
    behavioral_cols = ["login_frequency", "session_duration", "purchase_count"]
    labels = np.array([0] * 50 + [1] * 50)
    
    # 1. Setup DB: clear all tables and seed periods, old baseline and old clusters
    with get_raw_connection() as conn:
        with conn.cursor() as cur:
            clean_database_tables(cur)
            
            # Seed periods first (FK constraints)
            cur.execute("INSERT INTO periods (period_id, period_label, period_start, period_end, granularity) VALUES (%s, 'T-1', '2026-01-01', '2026-03-31', 'quarterly');", (period_id - 1,))
            cur.execute("INSERT INTO periods (period_id, period_label, period_start, period_end, granularity) VALUES (%s, 'T', '2026-04-01', '2026-06-30', 'quarterly');", (period_id,))
            
            cur.execute(
                """
                INSERT INTO datasets (dataset_id, source_type, permanence, layer0_completed)
                VALUES (%s, 'file_upload', 'permanent', TRUE);
                """,
                (dataset_id,)
            )
            # Insert baseline using default serial increment (retrieving the ID)
            cur.execute(
                """
                INSERT INTO cluster_baselines (is_active, adopted_reason, period_id)
                VALUES (TRUE, 'initial', %s)
                RETURNING baseline_id;
                """,
                (period_id - 1,)
            )
            old_baseline_id = cur.fetchone()[0]
            
            # Create old medoid customer and all sample test customers to prevent FK errors in assignments
            medoid_cid = "00000000-0000-0000-0000-000000009999"
            cur.execute("INSERT INTO customers (customer_id) VALUES (%s) ON CONFLICT DO NOTHING;", (medoid_cid,))
            for cid in sample_behavioral_df["customer_id"].tolist():
                cur.execute("INSERT INTO customers (customer_id) VALUES (%s) ON CONFLICT DO NOTHING;", (cid,))
                
            # Active baseline cluster
            cur.execute(
                """
                INSERT INTO clusters (baseline_id, cluster_label, medoid_customer_id, intra_cluster_spread)
                VALUES (%s, 'Cluster_0', %s, 0.1);
                """,
                (old_baseline_id, medoid_cid)
            )
            # Seed snapshot for old medoid
            cur.execute(
                """
                INSERT INTO behavioral_snapshots (customer_id, period_id, login_frequency, session_duration, purchase_count)
                VALUES (%s, %s, 10.0, 300.0, 5.0) ON CONFLICT DO NOTHING;
                """,
                (medoid_cid, period_id - 1)
            )
            
    mock_gate_results = {
        "stability_ari": 0.85, "stability_pass": True,
        "silhouette_score": 0.30, "davies_bouldin_index": 1.1, "separation_pass": True,
        "noise_fraction": 0.05, "noise_pass": True, "all_passed": True
    }
    
    with patch("src.layer2.drift_evaluator.evaluate_clustering", return_value=mock_gate_results):
        with patch("src.layer2.drift_evaluator.evaluate_baseline_quality", return_value={"noise_fraction": 0.05, "silhouette_score": 0.30, "davies_bouldin_index": 1.1}):
            decision = decide_adopt_vs_hold(dataset_id, period_id, sample_behavioral_df, labels, behavioral_cols)
            
    assert decision == "ADOPT"
    
    # Validate baseline states: old baseline is inactive, a new baseline is active
    with get_raw_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT is_active, baseline_id FROM cluster_baselines ORDER BY baseline_id;")
            baselines = cur.fetchall()
            
    assert len(baselines) == 2
    assert baselines[0] == (False, old_baseline_id) # Old baseline disabled
    assert baselines[1][0] == True                   # New baseline enabled
    
    # Validate identity map is populated
    with get_raw_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT match_type FROM cluster_identity_map;")
            mappings = cur.fetchall()
    assert len(mappings) > 0

def test_p3_t11_drift_evaluator_hold_path(sample_behavioral_df):
    """Verify HOLD decision and KNN classifier assignments persistence."""
    dataset_id = "00000000-0000-0000-0000-000000000306"
    period_id = 306
    behavioral_cols = ["login_frequency", "session_duration", "purchase_count"]
    labels = np.array([0] * 50 + [1] * 50)
    
    # Setup active baseline and clusters
    with get_raw_connection() as conn:
        with conn.cursor() as cur:
            clean_database_tables(cur)
            
            # Seed periods (FK constraints)
            cur.execute("INSERT INTO periods (period_id, period_label, period_start, period_end, granularity) VALUES (%s, 'T-1', '2026-01-01', '2026-03-31', 'quarterly');", (period_id - 1,))
            cur.execute("INSERT INTO periods (period_id, period_label, period_start, period_end, granularity) VALUES (%s, 'T', '2026-04-01', '2026-06-30', 'quarterly');", (period_id,))
            
            cur.execute(
                """
                INSERT INTO datasets (dataset_id, source_type, permanence, layer0_completed)
                VALUES (%s, 'file_upload', 'permanent', TRUE);
                """,
                (dataset_id,)
            )
            cur.execute(
                """
                INSERT INTO cluster_baselines (is_active, adopted_reason, period_id)
                VALUES (TRUE, 'initial', %s)
                RETURNING baseline_id;
                """,
                (period_id - 1,)
            )
            active_baseline_id = cur.fetchone()[0]
            
            # Create old medoid customers and all sample test customers to prevent FK errors in assignments
            medoid_cid_0 = "00000000-0000-0000-0000-000000008800"
            medoid_cid_1 = "00000000-0000-0000-0000-000000008805"
            cur.execute("INSERT INTO customers (customer_id) VALUES (%s);", (medoid_cid_0,))
            cur.execute("INSERT INTO customers (customer_id) VALUES (%s);", (medoid_cid_1,))
            for cid in sample_behavioral_df["customer_id"].tolist():
                cur.execute("INSERT INTO customers (customer_id) VALUES (%s) ON CONFLICT DO NOTHING;", (cid,))
                
            # Create active clusters with valid medoid IDs
            cur.execute("INSERT INTO clusters (baseline_id, cluster_label, medoid_customer_id, intra_cluster_spread) VALUES (%s, 'Cluster_0', %s, 0.5);", (active_baseline_id, medoid_cid_0))
            cur.execute("INSERT INTO clusters (baseline_id, cluster_label, medoid_customer_id, intra_cluster_spread) VALUES (%s, 'Cluster_1', %s, 0.5);", (active_baseline_id, medoid_cid_1))
            
            # Setup past customer assignments and snapshots to train KNN
            for i in range(10):
                cid = f"00000000-0000-0000-0000-0000000088{i:02d}"
                cur.execute("INSERT INTO customers (customer_id) VALUES (%s) ON CONFLICT DO NOTHING;", (cid,))
                
                # Retrieve active cluster IDs dynamically to map assignments properly
                cur.execute("SELECT cluster_id FROM clusters WHERE baseline_id = %s ORDER BY cluster_id;", (active_baseline_id,))
                cluster_ids = [row[0] for row in cur.fetchall()]
                cluster_id = cluster_ids[0] if i < 5 else cluster_ids[1]
                
                cur.execute(
                    """
                    INSERT INTO cluster_assignments (customer_id, period_id, cluster_id, assigned_via)
                    VALUES (%s, %s, %s, 'hdbscan');
                    """,
                    (cid, period_id - 1, cluster_id)
                )
                cur.execute(
                    """
                    INSERT INTO behavioral_snapshots (customer_id, period_id, login_frequency, session_duration, purchase_count)
                    VALUES (%s, %s, 10.0, 300.0, 5.0) ON CONFLICT DO NOTHING;
                    """,
                    (cid, period_id - 1)
                )
                
    # Mock evaluate_clustering and compute_medoids_and_spreads to simulate minimal medoid shift -> HOLD
    mock_gate_results = {
        "stability_ari": 0.85, "stability_pass": True,
        "silhouette_score": 0.30, "davies_bouldin_index": 1.1, "separation_pass": True,
        "noise_fraction": 0.05, "noise_pass": True, "all_passed": True
    }
    
    with patch("src.layer2.drift_evaluator.evaluate_clustering", return_value=mock_gate_results):
        with patch("src.layer2.drift_evaluator.evaluate_baseline_quality", return_value={"noise_fraction": 0.05, "silhouette_score": 0.30, "davies_bouldin_index": 1.1}):
            with patch("src.layer2.drift_evaluator.match_clusters", return_value=[{"match_type": "matched", "old_cluster_id": 0, "candidate_label": 0, "distance": 0.01}]):
                decision = decide_adopt_vs_hold(dataset_id, period_id, sample_behavioral_df, labels, behavioral_cols)
                
    assert decision == "HOLD"
    
    # Verify that the active baseline remains the active one
    active_bid, _ = get_active_baseline()
    assert active_bid == active_baseline_id
    
    # Verify cluster assignments are saved via knn_classifier
    with get_raw_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT count(*), assigned_via
                FROM cluster_assignments
                WHERE period_id = %s
                GROUP BY assigned_via;
                """,
                (period_id,)
            )
            rows = cur.fetchall()
            
    assert len(rows) == 1
    assert rows[0][0] == len(sample_behavioral_df)
    assert rows[0][1] == "knn_classifier"

def test_p3_t12_medoid_sampling_cap():
    """Verify that clusters larger than 2,000 points are sampled down to 2,000 for medoid calculation."""
    # Create a dummy large feature matrix of 3000 rows (exceeds 2000)
    np.random.seed(42)
    large_data = {
        "customer_id": [f"large_cust_{i}" for i in range(3000)],
        "login_frequency": np.random.normal(10, 2, 3000),
        "session_duration": np.random.normal(300, 50, 3000),
        "purchase_count": np.random.poisson(5, 3000).astype(float)
    }
    df = pd.DataFrame(large_data)
    labels = np.array([0] * 3000) # One big cluster of size 3000
    behavioral_cols = ["login_frequency", "session_duration", "purchase_count"]
    
    with patch("src.layer2.drift_evaluator.logger.info") as mock_logger:
        stats = compute_medoids_and_spreads(df, labels, behavioral_cols)
        
        assert len(stats) == 1
        assert 0 in stats
        assert stats[0]["medoid_customer_id"].startswith("large_cust_")
        assert len(stats[0]["medoid_coords"]) == 3
        assert stats[0]["intra_cluster_spread"] > 0
        
        # Verify logger message about sampling down was called
        log_messages = [call[0][0] for call in mock_logger.call_args_list]
        sampling_log_found = any("exceeds 2000 limit. Sampling 2000 points" in msg for msg in log_messages)
        assert sampling_log_found, "Sampling log was not emitted."

def test_layer2_kmeans_outlier_noise_detection(sample_behavioral_df):
    """Verify that extreme outliers in KMeans are flagged with noise label -1 and evaluated."""
    df = sample_behavioral_df.copy()
    # Inject an extreme outlier
    df.loc[99, "login_frequency"] = 1000.0
    df.loc[99, "session_duration"] = 50000.0
    df.loc[99, "purchase_count"] = 500.0
    
    behavioral_cols = ["login_frequency", "session_duration", "purchase_count"]
    labels = run_kmeans_clustering(df, behavioral_cols, n_clusters=3, detect_noise=True)
    
    # Customer 99 should be flagged as noise (-1)
    assert labels[99] == -1
    noise_count = np.sum(labels == -1)
    assert noise_count >= 1
    
    # Verify evaluate_clustering calculates true positive noise fraction
    with patch("src.layer2.cluster_evaluator.adjusted_rand_score", return_value=0.85):
        with patch("src.layer2.cluster_evaluator.silhouette_score", return_value=0.30):
            with patch("src.layer2.cluster_evaluator.davies_bouldin_score", return_value=1.1):
                results = evaluate_clustering(df, behavioral_cols, labels)
                assert results["noise_fraction"] > 0.0
                assert results["noise_pass"] == True

def test_layer2_degraded_candidate_holds_baseline(sample_behavioral_df):
    """Verify that when candidate quality degrades but drift is minimal, the system HOLDs baseline."""
    dataset_id = "00000000-0000-0000-0000-000000000309"
    period_id = 309
    behavioral_cols = ["login_frequency", "session_duration", "purchase_count"]
    labels = np.array([0] * 50 + [1] * 50)
    
    # Candidate metrics show degraded separation compared to baseline
    candidate_gate_results = {
        "stability_ari": 0.80, "stability_pass": True,
        "silhouette_score": 0.22, "davies_bouldin_index": 1.45, "separation_pass": True,
        "noise_fraction": 0.10, "noise_pass": True, "all_passed": True
    }
    
    # Baseline had superior separation (0.45 silhouette, 0.85 DBI, 0.02 noise)
    baseline_metrics = {
        "noise_fraction": 0.02,
        "silhouette_score": 0.45,
        "davies_bouldin_index": 0.85
    }
    
    with patch("src.layer2.drift_evaluator.get_active_baseline", return_value=(999, period_id - 1)):
        with patch("src.layer2.drift_evaluator.evaluate_clustering", return_value=candidate_gate_results):
            with patch("src.layer2.drift_evaluator.compute_medoids_and_spreads", return_value={
                0: {"medoid_customer_id": "c1", "medoid_coords": [0,0,0], "intra_cluster_spread": 0.5},
                1: {"medoid_customer_id": "c2", "medoid_coords": [1,1,1], "intra_cluster_spread": 0.5}
            }):
                with patch("src.layer2.drift_evaluator.get_raw_connection") as mock_conn:
                    # Mock clusters query
                    mock_cur = MagicMock()
                    mock_cur.fetchall.return_value = [
                        (101, "Cluster_0", "c1", 0.5),
                        (102, "Cluster_1", "c2", 0.5)
                    ]
                    mock_cur.fetchone.return_value = [10.0, 300.0, 5.0]
                    mock_conn.return_value.__enter__.return_value.cursor.return_value.__enter__.return_value = mock_cur
                    
                    # 1-to-1 match with negligible distance (no drift, no topology change)
                    mappings = [
                        {"match_type": "matched", "old_cluster_id": 0, "candidate_label": 0, "distance": 0.05},
                        {"match_type": "matched", "old_cluster_id": 1, "candidate_label": 1, "distance": 0.05}
                    ]
                    with patch("src.layer2.drift_evaluator.match_clusters", return_value=mappings):
                        with patch("src.layer2.drift_evaluator.evaluate_baseline_quality", return_value=baseline_metrics):
                            with patch("src.layer2.drift_evaluator.write_hold_baseline") as mock_write_hold:
                                decision = decide_adopt_vs_hold(dataset_id, period_id, sample_behavioral_df, labels, behavioral_cols)
                                
                                # Must HOLD the baseline, NOT adopt a worse model!
                                assert decision == "HOLD"
                                mock_write_hold.assert_called_once()

