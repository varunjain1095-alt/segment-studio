import os
import json
import logging
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score, f1_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

logger = logging.getLogger("layer4_trainer")

MODEL_DIR = os.path.join("src", "layer4", "models")

def train_and_select_model(X: pd.DataFrame, y: pd.Series) -> tuple[object, dict]:
    """
    Trains XGBoost, Random Forest, Gradient Boosting, and Logistic Regression models.
    Selects the best model based strictly on PR-AUC of the minority class (migrated = True).
    Saves winning model artifact and comprehensive evaluation metadata JSON.
    """
    logger.info("Initializing model training and selection...")
    
    # 1. Train/Validation Split (safely handling small sizes in tests)
    if len(X) < 10:
        logger.info("Dataset too small for train/val split. Using entire set for both training and validation.")
        X_train, X_val = X, X
        y_train, y_val = y, y
    else:
        unique_classes = len(set(y))
        stratify_y = y if unique_classes > 1 else None
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=stratify_y
        )
        
    candidates = {
        "XGBoost": XGBClassifier(random_state=42, eval_metric="logloss"),
        "RandomForest": RandomForestClassifier(random_state=42),
        "GradientBoosting": GradientBoostingClassifier(random_state=42),
        "LogisticRegression": LogisticRegression(random_state=42, max_iter=1000)
    }
    
    best_model_name = None
    best_model = None
    best_pr_auc = -1.0
    best_roc_auc = 0.0
    best_f1 = 0.0
    all_candidates = []
    
    # 2. Train and evaluate each model
    for name, clf in candidates.items():
        try:
            logger.info(f"Training {name}...")
            clf.fit(X_train, y_train)
            
            # Predict probabilities
            if hasattr(clf, "predict_proba"):
                probs = clf.predict_proba(X_val)[:, 1]
            else:
                probs = clf.decision_function(X_val)
                probs = 1.0 / (1.0 + np.exp(-probs))
                
            # Compute evaluation metrics
            if len(set(y_val)) < 2:
                val_pr_auc = 1.0 if (len(y_val) > 0 and y_val.iloc[0] == 1) else 0.0
                val_roc_auc = 1.0 if (len(y_val) > 0 and y_val.iloc[0] == 1) else 0.0
                val_f1 = 1.0 if (len(y_val) > 0 and y_val.iloc[0] == 1) else 0.0
                val_prec = 1.0 if (len(y_val) > 0 and y_val.iloc[0] == 1) else 0.0
                val_rec = 1.0 if (len(y_val) > 0 and y_val.iloc[0] == 1) else 0.0
            else:
                precision, recall, _ = precision_recall_curve(y_val, probs)
                val_pr_auc = float(auc(recall, precision))
                val_roc_auc = float(roc_auc_score(y_val, probs))
                preds = (probs >= 0.5).astype(int)
                val_f1 = float(f1_score(y_val, preds, zero_division=0))
                val_prec = float(precision_score(y_val, preds, zero_division=0))
                val_rec = float(recall_score(y_val, preds, zero_division=0))
                
            candidate_summary = {
                "name": name,
                "pr_auc": round(val_pr_auc, 4),
                "roc_auc": round(val_roc_auc, 4),
                "f1": round(val_f1, 4),
                "precision": round(val_prec, 4),
                "recall": round(val_rec, 4)
            }
            all_candidates.append(candidate_summary)
            logger.info(f"  {name} metrics: PR-AUC={val_pr_auc:.4f}, ROC-AUC={val_roc_auc:.4f}, F1={val_f1:.4f}")
            
            # Select best model strictly by PR-AUC (P6-T3)
            if val_pr_auc > best_pr_auc:
                best_pr_auc = val_pr_auc
                best_roc_auc = val_roc_auc
                best_f1 = val_f1
                best_model_name = name
                best_model = clf
        except Exception as e:
            logger.error(f"Failed to train {name} candidate: {str(e)}", exc_info=True)
            
    if best_model is None:
        raise RuntimeError("Failed to train any Layer 4 candidate models.")
        
    logger.info(f"Model selection complete: basis = PR-AUC. Winner: {best_model_name} with PR-AUC={best_pr_auc:.4f}")
    
    # 3. Extract Feature Importances from winning model
    feature_importances = []
    feature_names = [str(col) for col in X.columns]
    if hasattr(best_model, "feature_importances_"):
        raw_importances = best_model.feature_importances_
        sum_imp = float(np.sum(raw_importances)) if np.sum(raw_importances) > 0 else 1.0
        for feat, imp in zip(feature_names, raw_importances):
            feature_importances.append({
                "feature": feat,
                "weight": round(float(imp / sum_imp), 4)
            })
    elif hasattr(best_model, "coef_"):
        raw_coefs = np.abs(best_model.coef_[0])
        sum_coef = float(np.sum(raw_coefs)) if np.sum(raw_coefs) > 0 else 1.0
        for feat, c in zip(feature_names, raw_coefs):
            feature_importances.append({
                "feature": feat,
                "weight": round(float(c / sum_coef), 4)
            })
    feature_importances.sort(key=lambda x: x["weight"], reverse=True)
    
    # 4. Save winning model artifact and metadata
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, "predictive_segmenter.pkl")
    metadata_path = os.path.join(MODEL_DIR, "model_metadata.json")
    
    with open(model_path, "wb") as f:
        pickle.dump(best_model, f)
        
    best_metrics = {
        "model_type": best_model_name,
        "winning_model": best_model_name,
        "pr_auc": round(best_pr_auc, 4),
        "roc_auc": round(best_roc_auc, 4),
        "f1": round(best_f1, 4),
        "candidates": sorted(all_candidates, key=lambda c: c["pr_auc"], reverse=True),
        "feature_importances": feature_importances,
        "feature_names": feature_names
    }
    
    with open(metadata_path, "w") as f:
        json.dump(best_metrics, f, indent=2)
        
    logger.info(f"Winning model artifact successfully saved to: {model_path}")
    logger.info(f"Model metadata successfully saved to: {metadata_path}")
    
    return best_model, best_metrics
