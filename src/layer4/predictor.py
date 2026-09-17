import os
import json
import pickle
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger("layer4_predictor")

MODEL_DIR = os.path.join("src", "layer4", "models")
MODEL_PATH = os.path.join(MODEL_DIR, "predictive_segmenter.pkl")
METADATA_PATH = os.path.join(MODEL_DIR, "model_metadata.json")

def get_model_metadata() -> dict:
    """Loads model evaluation metadata and feature importances."""
    if os.path.exists(METADATA_PATH):
        try:
            with open(METADATA_PATH, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to read model metadata: {e}")
    return {}

def predict_migration_risk(features: dict) -> dict:
    """
    Runs actual model inference on a dictionary of customer feature values.
    Returns predicted probability, classification, risk tier, and recommended action.
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Trained Layer 4 model artifact not found. Please train model first.")
        
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
        
    metadata = get_model_metadata()
    feature_names = metadata.get("feature_names", [])
    
    # If feature_names is empty, check model.feature_names_in_
    if not feature_names and hasattr(model, "feature_names_in_"):
        feature_names = list(model.feature_names_in_)
        
    if not feature_names:
        raise ValueError("Model does not have feature names recorded.")

    # Map input features to model features
    row_data = {}
    for col in feature_names:
        val = features.get(col)
        if val is None:
            # Check case-insensitive or partial match
            for k, v in features.items():
                if k.lower() == col.lower() or k.lower() in col.lower() or col.lower() in k.lower():
                    val = v
                    break
        try:
            row_data[col] = float(val) if val is not None else 0.0
        except (ValueError, TypeError):
            row_data[col] = 0.0
            
    df_input = pd.DataFrame([row_data], columns=feature_names)
    
    # Predict probability
    if hasattr(model, "predict_proba"):
        prob = float(model.predict_proba(df_input)[0, 1])
    elif hasattr(model, "decision_function"):
        score = float(model.decision_function(df_input)[0])
        prob = float(1.0 / (1.0 + np.exp(-score)))
    else:
        prob = float(model.predict(df_input)[0])
        
    prob_pct = round(prob * 100, 1)
    
    if prob >= 0.70:
        risk_tier = "HIGH"
        risk_label = f"{prob_pct}% High Risk to Migrate"
        action = "Trigger Immediate Priority Support Escalation + 15% Win-Back Offer"
        color = "var(--danger)"
    elif prob >= 0.40:
        risk_tier = "MODERATE"
        risk_label = f"{prob_pct}% Moderate Drift Risk"
        action = "Schedule Automated Re-engagement Email & Retention Nudge"
        color = "var(--warning)"
    else:
        risk_tier = "LOW"
        risk_label = f"{prob_pct}% Low Migration Risk (Healthy Account)"
        action = "Maintain Standard Communication Cadence & Loyalty Upsell"
        color = "var(--success)"
        
    return {
        "probability": round(prob, 4),
        "probability_pct": prob_pct,
        "migrated_prediction": bool(prob >= 0.50),
        "risk_tier": risk_tier,
        "risk_label": risk_label,
        "recommended_action": action,
        "color": color,
        "winning_model": metadata.get("winning_model", type(model).__name__)
    }
