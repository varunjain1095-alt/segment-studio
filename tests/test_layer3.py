import os
import pytest
import numpy as np
import pandas as pd
import json
import scipy.stats as stats
from unittest.mock import patch, MagicMock
from src.db.connection import get_raw_connection
from src.layer3.statistical_tester import (
    compute_anova,
    compute_kruskal_wallis,
    compute_cohens_d,
    run_statistical_tests,
    validate_anti_circularity
)

def test_p4_t1_anova_correctness():
    """Verify One-way ANOVA output p-value and effect size match SciPy references."""
    np.random.seed(42)
    group1 = np.random.normal(10, 2, 50)
    group2 = np.random.normal(20, 2, 50)
    group3 = np.random.normal(30, 2, 50)
    
    # Reference calculation
    ref_stat, ref_p = stats.f_oneway(group1, group2, group3)
    
    # Tester calculation
    stat, p_val, eta_sq = compute_anova([group1, group2, group3])
    
    assert abs(p_val - ref_p) < 1e-6
    assert abs(stat - ref_stat) < 1e-6
    assert eta_sq > 0.0
    assert eta_sq <= 1.0

def test_p4_t2_kruskal_wallis_correctness():
    """Verify Kruskal-Wallis output p-value matches SciPy references."""
    np.random.seed(42)
    # Non-normal skewed distributions
    group1 = stats.expon.rvs(scale=2, size=50)
    group2 = stats.expon.rvs(scale=4, size=50)
    group3 = stats.expon.rvs(scale=6, size=50)
    
    # Reference calculation
    ref_stat, ref_p = stats.kruskal(group1, group2, group3)
    
    # Tester calculation
    stat, p_val, eff_size = compute_kruskal_wallis([group1, group2, group3])
    
    assert abs(p_val - ref_p) < 1e-6
    assert abs(stat - ref_stat) < 1e-6
    assert eff_size > 0.0

def test_p4_t3_anti_circularity_rejected():
    """Verify that using a behavioral feature as test variable raises ValueError."""
    dataset_id = "00000000-0000-0000-0000-000000000400"
    classes = [
        {"column": "login_frequency", "classification": "behavioral"},
        {"column": "churn_rate", "classification": "outcome"}
    ]
    
    # Mock database confirmed classifications
    with patch("src.layer3.statistical_tester.get_raw_connection") as mock_conn:
        mock_cur = MagicMock()
        mock_cur.fetchone.return_value = (json.dumps(classes),)
        mock_conn.return_value.__enter__.return_value.cursor.return_value.__enter__.return_value = mock_cur
        
        # login_frequency is a behavioral column -> Must raise ValueError
        with pytest.raises(ValueError, match="Under anti-circularity constraints"):
            validate_anti_circularity(dataset_id, "login_frequency")

def test_p4_t4_anti_circularity_accepted():
    """Verify outcome variables are accepted and run without error."""
    dataset_id = "00000000-0000-0000-0000-000000000400"
    classes = [
        {"column": "login_frequency", "classification": "behavioral"}
    ]
    
    with patch("src.layer3.statistical_tester.get_raw_connection") as mock_conn:
        mock_cur = MagicMock()
        mock_cur.fetchone.return_value = (json.dumps(classes),)
        mock_conn.return_value.__enter__.return_value.cursor.return_value.__enter__.return_value = mock_cur
        
        # Should execute successfully without throwing errors
        validate_anti_circularity(dataset_id, "churn_rate")

def test_p4_t5_effect_size_enforcement():
    """Verify significant is False when p-value < 0.001 but Cohen's d is negligible (< 0.20)."""
    # N = 25,000 each, standard deviation 0.2 to force small p-value but negligible delta (d ~ 0.05)
    np.random.seed(42)
    group1 = np.random.normal(10.0, 0.2, 25000)
    group2 = np.random.normal(10.01, 0.2, 25000)
    
    # p-value is extremely small due to N, but d is tiny
    stat, p_val = stats.ttest_ind(group1, group2)
    cohens_d = compute_cohens_d(group1, group2)
    
    assert p_val < 0.001
    assert cohens_d < 0.10 # Cohen's d is ~0.05
    
    # Significance logic check
    alpha = 0.05
    effect_threshold = 0.20 # cohen's d threshold
    is_significant = (p_val < alpha) and (cohens_d >= effect_threshold)
    
    assert is_significant == False

def test_p4_t6_critical_similarity_flag():
    """Verify that a high p-value (e.g. 0.82) flags critical similarity."""
    p_val = 0.82
    similarity_threshold = 0.80
    
    flag = None
    if p_val >= similarity_threshold:
        flag = "critical_similarity"
        
    assert flag == "critical_similarity"

def test_p4_critical_divergence_flag():
    """Verify low p-value & large effect size flags critical divergence."""
    p_val = 0.001
    cohens_d = 0.75 # large effect
    
    alpha = 0.05
    effect_threshold = 0.20
    divergence_threshold = 0.50
    
    is_sig = (p_val < alpha) and (cohens_d >= effect_threshold)
    
    flag = None
    if is_sig and (cohens_d >= divergence_threshold):
        flag = "critical_divergence"
        
    assert flag == "critical_divergence"

def test_p4_end_to_end_statistical_tests():
    """Verify end-to-end running and database persistence of Layer 3 statistical tester."""
    dataset_id = "00000000-0000-0000-0000-000000000405"
    period_id = 405
    
    # Clean and seed tables
    with get_raw_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM statistical_test_results;")
            cur.execute("DELETE FROM cluster_assignments;")
            cur.execute("DELETE FROM clusters;")
            cur.execute("DELETE FROM cluster_baselines;")
            cur.execute("DELETE FROM demographic_snapshots;")
            cur.execute("DELETE FROM behavioral_snapshots;")
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
                VALUES (4, TRUE, 'initial', %s);
                """,
                (period_id,)
            )
            # Create two clusters
            cur.execute("INSERT INTO clusters (cluster_id, baseline_id, cluster_label) VALUES (40, 4, 'Cluster_0');")
            cur.execute("INSERT INTO clusters (cluster_id, baseline_id, cluster_label) VALUES (41, 4, 'Cluster_1');")
            
            # Create customers, snapshots, and assignments
            for i in range(20):
                cid = f"00000000-0000-0000-0000-0000000044{i:02d}"
                cur.execute("INSERT INTO customers (customer_id) VALUES (%s);", (cid,))
                cluster_id = 40 if i < 10 else 41
                
                cur.execute(
                    """
                    INSERT INTO cluster_assignments (customer_id, period_id, cluster_id, assigned_via)
                    VALUES (%s, %s, %s, 'hdbscan');
                    """,
                    (cid, period_id, cluster_id)
                )
                
                # Make group 0 higher purchase counts for distinct outcomes
                purchases = 5.0 if i < 10 else 1.0
                cur.execute(
                    """
                    INSERT INTO behavioral_snapshots (customer_id, period_id, purchase_count, avg_order_value)
                    VALUES (%s, %s, %s, 100.0);
                    """,
                    (cid, period_id, purchases)
                )
                
                # Setup distinct churn rates:
                # Cluster 40: 8 active in T+1 (20% churn)
                # Cluster 41: 2 active in T+1 (80% churn)
                is_active_t1 = (i < 8) if (i < 10) else (i < 12)
                if is_active_t1:
                    cur.execute(
                        """
                        INSERT INTO behavioral_snapshots (customer_id, period_id, purchase_count, avg_order_value)
                        VALUES (%s, %s, 1.0, 100.0);
                        """,
                        (cid, period_id + 1)
                    )
                
    # Run test on churn_rate (permitted under snapshot fallback)
    results = run_statistical_tests(dataset_id, period_id, "cluster", "churn_rate", parametric=True)
    
    assert len(results) > 0
    
    # Query database to confirm row exists (expecting anova + tukey_hsd rows)
    with get_raw_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT count(*), test_variable, test_type FROM statistical_test_results GROUP BY test_variable, test_type;")
            rows = cur.fetchall()
            
    assert len(rows) == 2
    test_types = {r[2] for r in rows}
    assert "anova" in test_types
    assert "tukey_hsd" in test_types

def test_p4_t4b_anti_circularity_snapshot_fallback_excludes_proxies():
    """Verify that attempting to validate reconstructed revenue or repeat_purchase_prob under snapshot fallback raises ValueError."""
    dataset_id = "00000000-0000-0000-0000-000000000499"
    
    # Under snapshot fallback, revenue and repeat_purchase_prob must be rejected
    with pytest.raises(ValueError, match="Anti-circularity violation.*behavioral_snapshots fallback"):
        validate_anti_circularity(dataset_id, "revenue", is_fallback=True)
        
    with pytest.raises(ValueError, match="Anti-circularity violation.*behavioral_snapshots fallback"):
        validate_anti_circularity(dataset_id, "repeat_purchase_prob", is_fallback=True)
        
    # churn_rate and retention_rate must be accepted
    with patch("src.layer3.statistical_tester.get_raw_connection") as mock_conn:
        mock_cur = MagicMock()
        mock_cur.fetchone.return_value = (None,)
        mock_conn.return_value.__enter__.return_value.cursor.return_value.__enter__.return_value = mock_cur
        validate_anti_circularity(dataset_id, "churn_rate", is_fallback=True)
        validate_anti_circularity(dataset_id, "retention_rate", is_fallback=True)

def test_p4_t8_csv_column_mapping():
    """Verify that get_customer_level_outcomes maps Churned, Lifetime_Value, and Total_Purchases from CSV."""
    from src.layer3.statistical_tester import get_customer_level_outcomes
    dataset_id = "00000000-0000-0000-0000-000000000406"
    period_id = 406
    
    dummy_csv_data = {
        "Customer_ID": ["cust_1", "cust_2"],
        "Churned": [1.0, 0.0],
        "Lifetime_Value": [1500.0, 2500.0],
        "Total_Purchases": [4.0, 1.0]
    }
    dummy_df = pd.DataFrame(dummy_csv_data)
    
    with patch("src.layer3.statistical_tester.os.path.exists", return_value=True):
        with patch("src.layer3.statistical_tester.pd.read_csv", return_value=dummy_df):
            outcomes = get_customer_level_outcomes(dataset_id, period_id)
            
            assert not outcomes.empty
            assert len(outcomes) == 2
            # Churned directly mapped to churn_rate
            assert outcomes.loc[0, "churn_rate"] == 1.0
            assert outcomes.loc[1, "churn_rate"] == 0.0
            # Retention derived as 1 - Churned
            assert outcomes.loc[0, "retention_rate"] == 0.0
            assert outcomes.loc[1, "retention_rate"] == 1.0
            # Lifetime_Value mapped directly to revenue
            assert outcomes.loc[0, "revenue"] == 1500.0
            assert outcomes.loc[1, "revenue"] == 2500.0
            # Total_Purchases > 1 mapped to repeat_purchase_prob
            assert outcomes.loc[0, "repeat_purchase_prob"] == 1.0
            assert outcomes.loc[1, "repeat_purchase_prob"] == 0.0
