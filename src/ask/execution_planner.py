import re

def plan_execution(intent: str, query: str = "") -> dict:
    """
    Creates a shopping list of execution steps based on the intent and query text.
    Returns:
        dict: {
            "steps": [{"layer": str, "action": str}, ...],
            "hard_stop": bool
        }
    """
    intent = intent.lower()
    
    if intent == "out_of_scope":
        return {
            "steps": [],
            "hard_stop": True
        }
        
    elif intent == "show":
        # Determine single layer based on query content
        q = (query or "").lower()
        if any(w in q for w in ["demographic", "age", "gender", "country", "location"]):
            step = {"layer": "Layer1", "action": "retrieve_demographics"}
        elif any(w in q for w in ["drift", "baseline", "adopt", "hold", "profile", "session time"]):
            if any(w in q for w in ["early warning", "signal", "warning"]):
                step = {"layer": "Layer8", "action": "retrieve_signals"}
            else:
                step = {"layer": "Layer2", "action": "retrieve_drift"}
        elif any(w in q for w in ["statistical", "significance", "p-value", "t-test", "significant"]):
            step = {"layer": "Layer3", "action": "retrieve_significance"}
        elif any(w in q for w in ["predictive", "predict", "prediction", "forecast", "likely to migrate"]):
            step = {"layer": "Layer4", "action": "retrieve_predictions"}
        elif any(w in q for w in ["transition", "flow", "matrix"]):
            step = {"layer": "Layer5", "action": "retrieve_migration"}
        elif any(w in q for w in ["early warning", "signal", "warning"]):
            step = {"layer": "Layer8", "action": "retrieve_signals"}
        elif any(w in q for w in ["comparison", "compare"]):
            step = {"layer": "Layer7", "action": "retrieve_comparison"}
        elif any(w in q for w in ["narration", "briefing", "summary"]):
            step = {"layer": "A1", "action": "retrieve_narration"}
        else:
            # Default to Layer 6 Outcome
            step = {"layer": "Layer6", "action": "retrieve_churn"}
            
        return {
            "steps": [step],
            "hard_stop": False
        }
        
    elif intent == "comparison":
        return {
            "steps": [
                {"layer": "Layer7", "action": "retrieve_comparison"},
                {"layer": "Layer6", "action": "retrieve_churn"}
            ],
            "hard_stop": False
        }
        
    elif intent == "root_cause":
        return {
            "steps": [
                {"layer": "Layer1", "action": "retrieve_demographics"},
                {"layer": "Layer2", "action": "retrieve_drift"},
                {"layer": "Layer3", "action": "retrieve_significance"},
                {"layer": "Layer5", "action": "retrieve_migration"},
                {"layer": "Layer6", "action": "retrieve_churn"},
                {"layer": "Layer8", "action": "retrieve_signals"},
                {"layer": "A1", "action": "retrieve_narration"}
            ],
            "hard_stop": False
        }
        
    elif intent == "prediction":
        return {
            "steps": [
                {"layer": "Layer4", "action": "retrieve_predictions"},
                {"layer": "Layer5", "action": "retrieve_migration"},
                {"layer": "Layer8", "action": "retrieve_signals"}
            ],
            "hard_stop": False
        }
        
    # Default fallback
    return {
        "steps": [],
        "hard_stop": True
    }
