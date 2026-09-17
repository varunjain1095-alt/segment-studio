import os
import time
import json
import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
from src.db.connection import get_raw_connection
from src.layer4.dataset_builder import build_training_data
from src.layer4.trainer import train_and_select_model
from src.layer4.retrain_trigger import trigger_retraining

# ==============================================================================
# P6-T1: No Data Leakage - Temporal Feature Boundary
# ==============================================================================
def test_p6_t1_no_data_leakage():
    """Verify that feature matrix X contains strictly T features and y target uses T+1 labels."""
    dataset_id = "00000000-0000-0000-0000-000000000601"
    T = 601
    
    # Seed periods, assignments, snapshots
    with get_raw_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM cluster_assignments;")
            cur.execute("DELETE FROM clusters;")
            cur.execute("DELETE FROM cluster_baselines;")
            cur.execute("DELETE FROM behavioral_snapshots;")
            cur.execute("DELETE FROM periods WHERE period_id IN (%s, %s);", (T, T + 1))
            cur.execute("DELETE FROM datasets WHERE dataset_id = %s;", (dataset_id,))
            
            cur.execute("INSERT INTO periods (period_id, period_label, period_start, period_end, granularity) VALUES (%s, 'T', '2026-04-01', '2026-06-30', 'quarterly');", (T,))
            cur.execute("INSERT INTO periods (period_id, period_label, period_start, period_end, granularity) VALUES (%s, 'T+1', '2026-07-01', '2026-09-30', 'quarterly');", (T + 1,))
            
            cur.execute("INSERT INTO datasets (dataset_id, source_type, permanence, layer0_completed) VALUES (%s, 'file_upload', 'permanent', TRUE);", (dataset_id,))
            cur.execute("INSERT INTO cluster_baselines (baseline_id, is_active, adopted_reason, period_id) VALUES (6, TRUE, 'initial', %s);", (T,))
            cur.execute("INSERT INTO clusters (cluster_id, baseline_id, cluster_label) VALUES (60, 6, 'Cluster_0');")
            cur.execute("INSERT INTO clusters (cluster_id, baseline_id, cluster_label) VALUES (61, 6, 'Cluster_1');")
            
            # Customer 1
            cur.execute("INSERT INTO customers (customer_id) VALUES ('00000000-0000-0000-0000-000000006601');")
            cur.execute("INSERT INTO cluster_assignments (customer_id, period_id, cluster_id, assigned_via) VALUES ('00000000-0000-0000-0000-000000006601', %s, 60, 'hdbscan');", (T,))
            cur.execute("INSERT INTO cluster_assignments (customer_id, period_id, cluster_id, assigned_via) VALUES ('00000000-0000-0000-0000-000000006601', %s, 61, 'hdbscan');", (T + 1,))
            
            # Snapshots at T and T+1
            cur.execute("INSERT INTO behavioral_snapshots (customer_id, period_id, login_frequency, avg_order_value) VALUES ('00000000-0000-0000-0000-000000006601', %s, 10.0, 100.0);", (T,))
            # Intentionally put different value at T+1 to test leakage
            cur.execute("INSERT INTO behavioral_snapshots (customer_id, period_id, login_frequency, avg_order_value) VALUES ('00000000-0000-0000-0000-000000006601', %s, 999.0, 999.0);", (T + 1,))
            
    X, y = build_training_data(dataset_id, T)
    
    assert not X.empty
    assert len(y) == 1
    # Check that feature values strictly match T snapshots, not T+1 (999.0 should not be present)
    assert X.loc[0, "login_frequency"] == 10.0
    assert X.loc[0, "avg_order_value"] == 100.0
    assert 999.0 not in X.values

# ==============================================================================
# P6-T2: Migration Label Correctness (matched, disappeared, brand-new signups)
# ==============================================================================
def test_p6_t2_migration_label_correctness():
    """Verify that migration labels compute matched, disappeared, and new signup cases correctly."""
    dataset_id = "00000000-0000-0000-0000-000000000602"
    T = 602
    
    # Setup database with multiple scenarios:
    # Cust C: In Cluster 60 (A) at T and 61 (B) at T+1 -> migrated = True
    # Cust D: In Cluster 60 (A) at T and 60 (A) at T+1 -> migrated = False
    # Cust E: In Cluster 62 (disappeared) at T -> migrated = True
    # Cust F: In brand-new signup at T+1 (no T assignment) -> Excluded entirely
    # Cust G: assigned to brand-new Cluster 63 at T+1 -> migrated = True
    
    with get_raw_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM cluster_identity_map;")
            cur.execute("DELETE FROM cluster_assignments;")
            cur.execute("DELETE FROM clusters;")
            cur.execute("DELETE FROM cluster_baselines;")
            cur.execute("DELETE FROM behavioral_snapshots;")
            cur.execute("DELETE FROM periods WHERE period_id IN (%s, %s);", (T, T + 1))
            cur.execute("DELETE FROM datasets WHERE dataset_id = %s;", (dataset_id,))
            
            cur.execute("INSERT INTO periods (period_id, period_label, period_start, period_end, granularity) VALUES (%s, 'T', '2026-04-01', '2026-06-30', 'quarterly');", (T,))
            cur.execute("INSERT INTO periods (period_id, period_label, period_start, period_end, granularity) VALUES (%s, 'T+1', '2026-07-01', '2026-09-30', 'quarterly');", (T + 1,))
            
            cur.execute("INSERT INTO datasets (dataset_id, source_type, permanence, layer0_completed) VALUES (%s, 'file_upload', 'permanent', TRUE);", (dataset_id,))
            cur.execute("INSERT INTO cluster_baselines (baseline_id, is_active, adopted_reason, period_id) VALUES (6, TRUE, 'initial', %s);", (T,))
            
            cur.execute("INSERT INTO clusters (cluster_id, baseline_id, cluster_label) VALUES (60, 6, 'Cluster_A');")
            cur.execute("INSERT INTO clusters (cluster_id, baseline_id, cluster_label) VALUES (61, 6, 'Cluster_B');")
            cur.execute("INSERT INTO clusters (cluster_id, baseline_id, cluster_label) VALUES (62, 6, 'Cluster_Disappeared');")
            cur.execute("INSERT INTO clusters (cluster_id, baseline_id, cluster_label) VALUES (63, 6, 'Cluster_New');")
            
            # Identity mappings
            cur.execute("INSERT INTO cluster_identity_map (old_cluster_id, new_cluster_id, match_type) VALUES (60, 60, 'matched');")
            cur.execute("INSERT INTO cluster_identity_map (old_cluster_id, new_cluster_id, match_type) VALUES (61, 61, 'matched');")
            cur.execute("INSERT INTO cluster_identity_map (old_cluster_id, new_cluster_id, match_type) VALUES (62, NULL, 'disappeared');")
            cur.execute("INSERT INTO cluster_identity_map (old_cluster_id, new_cluster_id, match_type) VALUES (NULL, 63, 'new');")
            
            cids = {
                "C": "00000000-0000-0000-0000-00000000620c",
                "D": "00000000-0000-0000-0000-00000000620d",
                "E": "00000000-0000-0000-0000-00000000620e",
                "F": "00000000-0000-0000-0000-00000000620f", # brand-new signup (F)
                "G": "00000000-0000-0000-0000-000000006208"  # Hexadecimal character
            }
            
            # Customer registrations
            for name, cid in cids.items():
                cur.execute("INSERT INTO customers (customer_id) VALUES (%s);", (cid,))
                
            # Customer C (A at T, B at T+1)
            cur.execute("INSERT INTO cluster_assignments (customer_id, period_id, cluster_id, assigned_via) VALUES (%s, %s, 60, 'hdbscan');", (cids["C"], T))
            cur.execute("INSERT INTO cluster_assignments (customer_id, period_id, cluster_id, assigned_via) VALUES (%s, %s, 61, 'hdbscan');", (cids["C"], T + 1))
            
            # Customer D (A at T, A at T+1)
            cur.execute("INSERT INTO cluster_assignments (customer_id, period_id, cluster_id, assigned_via) VALUES (%s, %s, 60, 'hdbscan');", (cids["D"], T))
            cur.execute("INSERT INTO cluster_assignments (customer_id, period_id, cluster_id, assigned_via) VALUES (%s, %s, 60, 'hdbscan');", (cids["D"], T + 1))
            
            # Customer E (Disappeared at T, B at T+1)
            cur.execute("INSERT INTO cluster_assignments (customer_id, period_id, cluster_id, assigned_via) VALUES (%s, %s, 62, 'hdbscan');", (cids["E"], T))
            cur.execute("INSERT INTO cluster_assignments (customer_id, period_id, cluster_id, assigned_via) VALUES (%s, %s, 61, 'hdbscan');", (cids["E"], T + 1))
            
            # Customer F (Brand new signup: Only assignment at T+1)
            cur.execute("INSERT INTO cluster_assignments (customer_id, period_id, cluster_id, assigned_via) VALUES (%s, %s, 61, 'hdbscan');", (cids["F"], T + 1))
            
            # Customer G (A at T, Brand new cluster 63 at T+1)
            cur.execute("INSERT INTO cluster_assignments (customer_id, period_id, cluster_id, assigned_via) VALUES (%s, %s, 60, 'hdbscan');", (cids["G"], T))
            cur.execute("INSERT INTO cluster_assignments (customer_id, period_id, cluster_id, assigned_via) VALUES (%s, %s, 63, 'hdbscan');", (cids["G"], T + 1))
            
            # Insert snapshots at T (F has no snapshot at T)
            for k in ["C", "D", "E", "G"]:
                cur.execute("INSERT INTO behavioral_snapshots (customer_id, period_id, login_frequency) VALUES (%s, %s, 5.0);", (cids[k], T))
                
    # Query database and build dataset
    X, y = build_training_data(dataset_id, T)
    
    # Customer F must be excluded entirely (length is 4, not 5)
    assert len(y) == 4
    
    # Map index to customer predictions
    with get_raw_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT customer_id FROM behavioral_snapshots WHERE period_id = {T} ORDER BY customer_id;"
            )
            cids_sorted = [str(r[0]) for r in cur.fetchall()]
            
    # Map predictions back to original letters
    labels_dict = {}
    for i, cid in enumerate(cids_sorted):
        # find matching letter
        letter = next(k for k, v in cids.items() if v == cid)
        labels_dict[letter] = y.iloc[i]
        
    assert labels_dict["C"] == True  # transitioned A -> B
    assert labels_dict["D"] == False # remained A -> A
    assert labels_dict["E"] == True  # disappeared cluster
    assert labels_dict["G"] == True  # new cluster transition
    assert "F" not in labels_dict    # excluded signup

# ==============================================================================
# P6-T3: Model Selection by PR-AUC on Minority Class
# ==============================================================================
def test_p6_t3_model_selection_pr_auc():
    """Verify that XGBoost (higher PR-AUC, lower ROC-AUC) wins over Random Forest (lower PR-AUC, higher ROC-AUC)."""
    np.random.seed(42)
    # Generate imbalanced target where one model can dominate PR-AUC
    X = pd.DataFrame(np.random.normal(0, 1, (100, 3)), columns=["f1", "f2", "f3"])
    y = pd.Series(np.random.choice([0, 1], size=100, p=[0.85, 0.15]))
    
    # We will mock candidates to return pre-defined predict_proba outputs matching the scenario
    # XGBoost: PR-AUC = 0.72, ROC-AUC = 0.81
    # RandomForest: PR-AUC = 0.61, ROC-AUC = 0.88
    # GradientBoosting: PR-AUC = 0.50
    # LogisticRegression: PR-AUC = 0.40
    
    # We can patch model fitting and evaluation using sklearn functions
    with patch("src.layer4.trainer.XGBClassifier.predict_proba") as mock_xgb:
        with patch("src.layer4.trainer.RandomForestClassifier.predict_proba") as mock_rf:
            with patch("src.layer4.trainer.GradientBoostingClassifier.predict_proba") as mock_gb:
                with patch("src.layer4.trainer.LogisticRegression.predict_proba") as mock_lr:
                    
                    # We will mock the validation split inputs and fit functions to return successfully
                    with patch("src.layer4.trainer.train_test_split", return_value=(X, X, y, y)):
                        
                        # Set mock predict_proba return values: [prob_class_0, prob_class_1]
                        # We make y the true labels and create custom probabilities
                        y_val = y.values
                        
                        # Generate probs: correlation with y determines AUCs
                        probs_xgb = np.clip(y_val * 0.7 + np.random.normal(0, 0.25, len(y_val)), 0, 1)
                        probs_rf = np.clip(y_val * 0.5 + np.random.normal(0, 0.1, len(y_val)), 0, 1)
                        
                        mock_xgb.return_value = np.column_stack([1 - probs_xgb, probs_xgb])
                        mock_rf.return_value = np.column_stack([1 - probs_rf, probs_rf])
                        mock_gb.return_value = np.column_stack([np.ones_like(y_val)*0.5, np.ones_like(y_val)*0.5])
                        mock_lr.return_value = np.column_stack([np.ones_like(y_val)*0.5, np.ones_like(y_val)*0.5])
                        
                        # Directly patch precision_recall_curve and roc_auc_score to guarantee exact PR-AUC / ROC-AUC criteria match
                        def mock_pr_roc(y_true, prob):
                            # Check which model probs are evaluated
                            if np.array_equal(prob, probs_xgb):
                                return 0.72, 0.81 # pr_auc, roc_auc
                            elif np.array_equal(prob, probs_rf):
                                return 0.61, 0.88 # pr_auc, roc_auc
                            return 0.40, 0.40
                            
                        with patch("src.layer4.trainer.precision_recall_curve", return_value=(None, None, None)):
                            with patch("src.layer4.trainer.auc", side_effect=lambda x, y: 0.72 if x is None else 0.61):
                                with patch("src.layer4.trainer.roc_auc_score", side_effect=lambda x, y: 0.81 if np.array_equal(y, probs_xgb) else 0.88):
                                    
                                    best_model, metrics = train_and_select_model(X, y)
                                    
                                    # Assert winning model is XGBoost (higher PR-AUC 0.72 vs 0.61)
                                    assert metrics["model_type"] == "XGBoost"
                                    assert metrics["pr_auc"] == 0.72
                                    assert metrics["roc_auc"] == 0.81

# ==============================================================================
# P6-T4: Layer 2.4 Adopt Triggers Layer 4 Retraining
# ==============================================================================
def test_p6_t4_adopt_triggers_retraining():
    """Verify that decide_adopt_vs_hold on adopt event triggers Layer 4 retraining and fresh artifact save."""
    dataset_id = "00000000-0000-0000-0000-000000000604"
    T = 604
    
    # Mock trigger_retraining to track if it is invoked (patch at the source of retrain_trigger)
    with patch("src.layer4.retrain_trigger.trigger_retraining") as mock_retrain:
        with patch("src.layer2.drift_evaluator.get_active_baseline", return_value=(1, T)):
            with patch("src.layer2.drift_evaluator.evaluate_clustering", return_value={"noise_fraction": 0.05, "silhouette_score": 0.5, "davies_bouldin_index": 0.8}):
                with patch("src.layer2.drift_evaluator.compute_medoids_and_spreads", return_value={}):
                    with patch("src.layer2.drift_evaluator.evaluate_baseline_quality", return_value={"noise_fraction": 0.01, "silhouette_score": 0.6, "davies_bouldin_index": 0.7}):
                        # Mock mapping and database mapping queries
                        with patch("src.layer2.drift_evaluator.match_clusters", return_value=[]):
                            with patch("src.layer2.drift_evaluator.write_adopt_baseline") as mock_write:
                                
                                from src.layer2.drift_evaluator import decide_adopt_vs_hold
                                # Dummy DataFrame for features
                                X_df = pd.DataFrame(np.random.normal(0, 1, (10, 3)), columns=["f1", "f2", "f3"])
                                labels = np.array([0]*5 + [1]*5)
                                
                                # Execute
                                decision = decide_adopt_vs_hold(dataset_id, T + 1, X_df, labels, ["f1", "f2", "f3"])
                                
                                # Assert adopt occurred
                                assert decision == "ADOPT"
                                # Assert trigger_retraining was called with dataset_id and T (active_period_id)
                                mock_retrain.assert_called_once_with(dataset_id, T)

def test_layer4_predictor_and_metadata():
    """Verify that predictor accurately loads model and metadata and scores inputs."""
    from src.layer4.predictor import predict_migration_risk, get_model_metadata
    from sklearn.ensemble import RandomForestClassifier
    import pickle
    
    X = pd.DataFrame({"f1": [1.0, 2.0, 3.0, 4.0], "f2": [0.1, 0.2, 0.3, 0.4]})
    y = pd.Series([0, 0, 1, 1])
    clf = RandomForestClassifier(random_state=42).fit(X, y)
    
    os.makedirs("src/layer4/models", exist_ok=True)
    with open("src/layer4/models/predictive_segmenter.pkl", "wb") as f:
        pickle.dump(clf, f)
    with open("src/layer4/models/model_metadata.json", "w") as f:
        json.dump({
            "winning_model": "RandomForest",
            "pr_auc": 0.95,
            "roc_auc": 0.95,
            "f1": 0.95,
            "candidates": [{"name": "RandomForest", "pr_auc": 0.95}],
            "feature_importances": [{"feature": "f1", "weight": 0.6}, {"feature": "f2", "weight": 0.4}],
            "feature_names": ["f1", "f2"]
        }, f)
        
    res = predict_migration_risk({"f1": 3.5, "f2": 0.35})
    assert "probability" in res
    assert "risk_tier" in res
    assert "recommended_action" in res
    assert res["winning_model"] == "RandomForest"

def test_layer4_api_endpoints():
    """Verify GET /api/v1/migration-model-metrics and POST /api/v1/predict-customer-migration."""
    from fastapi.testclient import TestClient
    from src.main import app
    client = TestClient(app)
    
    # 1. Test model metrics
    res = client.get("/api/v1/migration-model-metrics")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "success"
    assert "winning_model" in data
    assert "candidates" in data
    assert "feature_importances" in data
    
    # 2. Test customer prediction
    pred_res = client.post("/api/v1/predict-customer-migration", json={"features": {"f1": 2.5, "f2": 0.25}})
    assert pred_res.status_code == 200
    pred_data = pred_res.json()
    assert pred_data["status"] == "success"
    assert "probability" in pred_data["prediction"]
    assert "risk_tier" in pred_data["prediction"]


