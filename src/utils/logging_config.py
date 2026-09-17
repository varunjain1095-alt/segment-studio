import logging
import json
import sys
import re
import contextvars
from datetime import datetime

# Context variables for log propagation
log_dataset_id = contextvars.ContextVar("log_dataset_id", default="UnknownDataset")

LOGGER_TO_LAYER = {
    "cleaning": "Layer0",
    "outliers": "Layer0",
    "imputer": "Layer0",
    "feature_engineer": "Layer0",
    "cross_field_validator": "Layer0",
    "field_classifier": "Layer1",
    "anomaly_flagger": "Layer1",
    "mom_tracker": "Layer1",
    "clusterer": "Layer2",
    "cluster_evaluator": "Layer2",
    "drift_evaluator": "Layer2",
    "preprocessor": "Layer2",
    "layer2_preprocessor": "Layer2",
    "layer2_clusterer": "Layer2",
    "layer2_cluster_evaluator": "Layer2",
    "layer2_drift_evaluator": "Layer2",
    "statistical_tester": "Layer3",
    "layer3_statistical_tester": "Layer3",
    "trainer": "Layer4",
    "dataset_builder": "Layer4",
    "transition_matrix": "Layer5",
    "outcome_analyzer": "Layer6",
    "layer6_outcome_analyzer": "Layer6",
    "metric_api": "Layer7",
    "layer7_metric_api": "Layer7",
    "migration_signal": "Layer8",
    "layer8_migration_signal": "Layer8",
    "insight_narrator": "Agent1",
    "narration_evaluator": "Agent1",
    "answer_generator": "AskYourData",
    "execution_engine": "AskYourData",
    "intent_router": "AskYourData",
    "quarterly_batch": "QuarterlyBatch",
}

class JsonFormatter(logging.Formatter):
    def format(self, record):
        msg_str = record.getMessage()
        
        # 1. Update/Extract dataset_id from the message if present
        uuid_match = re.search(
            r'\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b',
            msg_str,
            re.IGNORECASE
        )
        if uuid_match:
            ds_id = uuid_match.group(0)
            log_dataset_id.set(ds_id)
        else:
            ds_id = log_dataset_id.get()
            
        # 2. Map logger to layer name
        layer_name = LOGGER_TO_LAYER.get(record.name, "PipelineCore")
        
        # 3. Determine status
        if record.levelno >= logging.ERROR or record.exc_info:
            status = "error"
        else:
            lower_msg = msg_str.lower()
            if any(w in lower_msg for w in ["starting", "running", "initiating", "started"]):
                status = "started"
            else:
                status = "completed"
                
        # 4. Construct log entry
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "layer_name": layer_name,
            "dataset_id": ds_id,
            "status": status,
        }
        
        if status == "error":
            log_data["error_message"] = msg_str
            if record.exc_info:
                log_data["exception"] = self.formatException(record.exc_info)
        else:
            log_data["message"] = msg_str
            
        return json.dumps(log_data)

def setup_logging():
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Remove existing handlers to ensure we don't have unstructured stdout/stderr handlers
    for h in list(root_logger.handlers):
        root_logger.removeHandler(h)
        
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root_logger.addHandler(handler)
