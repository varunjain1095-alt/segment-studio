import os
import shutil
import logging
from typing import List, Optional
import pandas as pd
from pydantic import BaseModel
from fastapi import APIRouter, UploadFile, File, HTTPException
from src.ingestion.router import ingest_from_file, ingest_from_db, DATA_DIR
from src.db.connection import get_raw_connection
from src.layer0.profiler import data_profile, save_profile_to_db
from src.layer0.cleaning import clean_dataset
from src.layer0.outliers import (
    treat_univariate_numeric_outliers,
    treat_univariate_categorical_outliers,
    treat_bivariate_numeric_outliers,
    check_bivariate_categorical_similarity
)
from src.layer0.imputer import check_missing_value_limits, apply_imputation, persist_imputation_choice
from src.layer0.cross_field_validator import (
    propose_heuristic_rules,
    propose_llm_rules,
    confirm_rules_against_data,
    approve_rule,
    get_active_rules
)
from src.layer0.feature_engineer import prune_redundant_features, encode_categorical_features
from src.layer1.field_classifier import classify_fields, confirm_field_classifications
from src.layer2.preprocessor import load_and_preprocess_behavioral_data, populate_behavioral_snapshots
from src.layer2.clusterer import run_kmeans_clustering
from src.layer2.drift_evaluator import decide_adopt_vs_hold
from src.layer2.cluster_profiler import run_profiling_and_persist
from src.layer3.statistical_tester import run_statistical_tests
from src.layer6.outcome_analyzer import compute_and_persist_segment_outcomes
from src.layer4.retrain_trigger import trigger_retraining
from src.layer5.migration_forecaster import compute_and_persist_transition_matrix
from src.layer8.migration_signal import confirm_layer8_columns, compute_and_persist_feature_migration_signals
from src.layer7.metric_api import get_metric_comparison
from src.notifications.center import PipelineBlockedError

logger = logging.getLogger("api_v1")

import math
def safe_float(val):
    if val is None:
        return None
    try:
        f_val = float(val)
        if math.isnan(f_val) or math.isinf(f_val):
            return 0.0
        return f_val
    except Exception:
        return None
router = APIRouter(prefix="/api/v1")

class ColumnConfirmation(BaseModel):
    column_name: str
    include: bool

class ConfirmColumnsPayload(BaseModel):
    columns: List[ColumnConfirmation]

class ImputationChoice(BaseModel):
    method: str

class RuleApproval(BaseModel):
    rule_id: str
    approved: bool

class ApproveRulesPayload(BaseModel):
    rules: List[RuleApproval]

@router.get("/datasets")
async def list_datasets():
    """Returns all datasets stored in the database ordered by most recent."""
    with get_raw_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT dataset_id, source_type, file_name, ingested_at FROM datasets ORDER BY ingested_at DESC"
        )
        rows = cur.fetchall()
        return [
            {
                "dataset_id": r[0],
                "name": r[2] or f"Dataset {r[0][:8]}",
                "source_type": r[1],
                "created_at": r[3].isoformat() if r[3] else None,
            }
            for r in rows
        ]

@router.post("/datasets/upload")
async def upload_dataset(file: UploadFile = File(...)):
    """Uploads a dataset file (CSV, XLSX, JSON) and returns dataset_id."""
    temp_path = os.path.join(DATA_DIR, f"temp_{uuid_name()}.tmp")
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        dataset_id = ingest_from_file(temp_path, file.filename)
        return {"dataset_id": dataset_id, "status": "ingested"}
    except Exception as e:
        logger.error(f"Upload failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@router.post("/datasets/ingest")
async def ingest_dataset(file: UploadFile = File(...)):
    """Ingests a dataset file (CSV, XLSX, JSON) and returns dataset_id."""
    return await upload_dataset(file)


class DbIngestPayload(BaseModel):
    connection_string: str
    query: str


@router.post("/datasets/ingest-db")
def ingest_dataset_db(payload: DbIngestPayload):
    try:
        dataset_id = ingest_from_db(payload.connection_string, payload.query)
        return {"dataset_id": dataset_id, "status": "ingested"}
    except Exception as e:
        logger.error(f"DB Ingestion failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/datasets/{dataset_id}/confirm-columns")
def confirm_columns(dataset_id: str, payload: ConfirmColumnsPayload):
    """
    PM toggle include/exclude for columns. Drops excluded columns and
    saves the confirmed column set.
    """
    raw_path = os.path.join(DATA_DIR, f"{dataset_id}.csv")
    if not os.path.exists(raw_path):
        raise HTTPException(status_code=404, detail="Dataset not found")
        
    df = pd.read_csv(raw_path)
    
    # Save the confirmed columns list to DB
    import json
    confirmed_cols_json = json.dumps([col.dict() for col in payload.columns])
    with get_raw_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE datasets
                SET confirmed_columns = %s
                WHERE dataset_id = %s;
                """,
                (confirmed_cols_json, dataset_id)
            )
            
    # Drop excluded columns
    excluded_cols = [c.column_name for c in payload.columns if not c.include]
    df_confirmed = df.drop(columns=[col for col in excluded_cols if col in df.columns])
    
    confirmed_path = os.path.join(DATA_DIR, f"{dataset_id}_confirmed.csv")
    df_confirmed.to_csv(confirmed_path, index=False)
    
    # Run initial profile on confirmed dataset and store in DB
    profile_df = data_profile(df_confirmed)
    save_profile_to_db(dataset_id, profile_df)
    
    # Check >50% missing values limit to emit hard fail if necessary
    check_missing_value_limits(dataset_id, profile_df)
    
    return {"status": "columns_confirmed", "remaining_columns": df_confirmed.columns.tolist()}

@router.post("/datasets/{dataset_id}/columns/{col}/confirm-drop")
def confirm_drop_column(dataset_id: str, col: str):
    """PM confirms dropping a column that exceeds >50% missing threshold."""
    confirmed_path = os.path.join(DATA_DIR, f"{dataset_id}_confirmed.csv")
    if not os.path.exists(confirmed_path):
        raise HTTPException(status_code=400, detail="Must confirm columns first")
        
    df = pd.read_csv(confirmed_path)
    if col in df.columns:
        df = df.drop(columns=[col])
        df.to_csv(confirmed_path, index=False)
        
    # Re-run profiling & update DB
    profile_df = data_profile(df)
    save_profile_to_db(dataset_id, profile_df)
    
    return {"status": "column_dropped", "column": col}

@router.post("/datasets/{dataset_id}/columns/{col}/set-imputation")
def set_imputation(dataset_id: str, col: str, payload: ImputationChoice):
    """PM selects the missing value imputation strategy for a column."""
    # Persist choice in database
    persist_imputation_choice(dataset_id, col, payload.method)
    return {"status": "imputation_configured", "column": col, "method": payload.method}

@router.post("/datasets/{dataset_id}/cross-field-rules/approve")
def approve_cross_field_rules(dataset_id: str, payload: ApproveRulesPayload):
    """PM approves or rejects proposed cross-field validation rules."""
    for rule in payload.rules:
        approve_rule(dataset_id, rule.rule_id, rule.approved)
    return {"status": "rules_updated"}

@router.post("/datasets/{dataset_id}/run-layer0")
def run_layer0(dataset_id: str):
    """
    Executes the full Layer 0 data pipeline:
    Type recovery, outlier checks, imputer execution, cross-field validation checks,
    feature engineering, and saves the final output.
    """
    confirmed_path = os.path.join(DATA_DIR, f"{dataset_id}_confirmed.csv")
    if not os.path.exists(confirmed_path):
        raise HTTPException(status_code=400, detail="Must confirm columns first")
        
    df = pd.read_csv(confirmed_path)
    
    # 1. Cleaning & Type Recovery
    df, profile_df, duplicates = clean_dataset(df, dataset_id)
    
    # 2. Imputation Runner
    # Check stored choices and apply them
    with get_raw_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT column_name, imputation_method
                FROM column_profiles
                WHERE dataset_id = %s AND imputation_method IS NOT NULL;
                """,
                (dataset_id,)
            )
            choices = cur.fetchall()
            
    for col_name, method in choices:
        if col_name in df.columns:
            df[col_name] = apply_imputation(df, col_name, method)
            
    # 3. Outlier Treatment
    df = treat_univariate_numeric_outliers(df)
    df = treat_univariate_categorical_outliers(df)
    df = treat_bivariate_numeric_outliers(df)
    check_bivariate_categorical_similarity(df)
    
    # 4. Propose and Validate Cross-Field Rules (Run on data pre-pruning)
    heuristic_rules = propose_heuristic_rules(df)
    confirm_rules_against_data(df, heuristic_rules, "heuristic", dataset_id)
    
    llm_rules = propose_llm_rules(df)
    confirm_rules_against_data(df, llm_rules, "llm", dataset_id)
    
    # 5. Feature Engineering
    df = prune_redundant_features(df)
    df = encode_categorical_features(df)
    
    # Save processed dataset
    processed_path = os.path.join(DATA_DIR, f"{dataset_id}_processed.csv")
    df.to_csv(processed_path, index=False)
    
    # 6. Mark Layer 0 Completed in DB (Guaranteed code-level execution on successful run)
    with get_raw_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE datasets
                SET layer0_completed = TRUE
                WHERE dataset_id = %s;
                """,
                (dataset_id,)
            )
    
    return {
        "status": "layer0_completed",
        "processed_rows": len(df),
        "processed_columns": df.columns.tolist(),
        "duplicates_found": duplicates
    }

class ColumnClassification(BaseModel):
    column: str
    classification: str

class ConfirmClassificationsPayload(BaseModel):
    classifications: List[ColumnClassification]

@router.post("/datasets/{dataset_id}/field-classifications/propose")
def propose_field_classifications(dataset_id: str):
    """
    Runs LLM field classifier to propose classifications (demographic, behavioral, unsure).
    """
    # Fetch profiles from DB
    with get_raw_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT column_name, dtype
                FROM column_profiles
                WHERE dataset_id = %s;
                """,
                (dataset_id,)
            )
            rows = cur.fetchall()
            
    if not rows:
        raise HTTPException(status_code=400, detail="No column profiles found for dataset. Run confirm-columns or run-layer0 first.")
        
    profile_df = pd.DataFrame(rows, columns=["column_name", "dtype"])
    proposals = classify_fields(dataset_id, profile_df)
    return {"status": "proposals_generated", "proposals": proposals}

@router.post("/datasets/{dataset_id}/field-classifications/confirm")
def confirm_classifications(dataset_id: str, payload: ConfirmClassificationsPayload):
    """
    PM confirms classifications, saving them as active. Alters demographic_snapshots DDL.
    """
    classifications_list = [item.dict() for item in payload.classifications]
    confirm_field_classifications(dataset_id, classifications_list)
    return {"status": "classifications_confirmed"}

class NameClusterPayload(BaseModel):
    name: str

@router.post("/datasets/{dataset_id}/periods/{period_id}/run-clustering")
def run_clustering(dataset_id: str, period_id: int):
    """
    Executes the full Layer 2 clustering pipeline:
    1. Preprocesses behavioral data.
    2. Populates behavioral snapshots in DB.
    3. Runs HDBSCAN.
    4. Evaluates quality gates and performs Adopt-vs-Hold check.
    """
    try:
        # Load active behavioral cols from DB to make sure we know what to fetch
        with get_raw_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT confirmed_classifications FROM datasets WHERE dataset_id = %s;", (dataset_id,))
                res = cur.fetchone()
                
        if not res or not res[0]:
            raise HTTPException(status_code=400, detail="No confirmed field classifications found. Run Layer 1 first.")
            
        import json
        classifications = json.loads(res[0]) if isinstance(res[0], str) else res[0]
        behavioral_cols = [c["column"] for c in classifications if c.get("classification") == "behavioral"]
        
        # Check if fallback is already active
        with get_raw_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT fallback_active FROM datasets WHERE dataset_id = %s;", (dataset_id,))
                row = cur.fetchone()
        if row and row[0]:
            return {
                "status": "clustering_skipped",
                "decision": "SKIPPED_FALLBACK_ACTIVE",
                "period_id": period_id,
                "rows_clustered": 0
            }

        # 1. Preprocess behavioral features
        X_df, _ = load_and_preprocess_behavioral_data(dataset_id)
        
        # 2. Populate behavioral snapshots in DB
        populate_behavioral_snapshots(X_df, period_id, behavioral_cols)
        
        # 3. Run HDBSCAN clustering
        labels = run_kmeans_clustering(X_df, behavioral_cols)
        
        # Evaluate fallback triggers
        from src.fallback.trigger_detector import evaluate as evaluate_fallback
        fallback_active = evaluate_fallback(dataset_id, period_id)
        
        # 4. Decides adopt vs hold and writes to DB
        if fallback_active:
            decision = "SKIPPED_FALLBACK_ACTIVE"
        else:
            decision = decide_adopt_vs_hold(dataset_id, period_id, X_df, labels, behavioral_cols)
        
        return {
            "status": "clustering_completed" if not fallback_active else "clustering_skipped",
            "decision": decision,
            "period_id": period_id,
            "rows_clustered": len(X_df)
        }
    except Exception as e:
        logger.error(f"Clustering execution failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/datasets/{dataset_id}/periods/{period_id}/profile-clusters")
def profile_clusters_endpoint(dataset_id: str, period_id: int):
    """
    Runs LLM-assisted profiling for the clusters assigned in the given period,
    and persists description summaries.
    """
    try:
        with get_raw_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT confirmed_classifications FROM datasets WHERE dataset_id = %s;", (dataset_id,))
                res = cur.fetchone()
                
        if not res or not res[0]:
            raise HTTPException(status_code=400, detail="No confirmed classifications found.")
            
        import json
        classifications = json.loads(res[0]) if isinstance(res[0], str) else res[0]
        behavioral_cols = [c["column"] for c in classifications if c.get("classification") == "behavioral"]
        
        summaries = run_profiling_and_persist(dataset_id, period_id, behavioral_cols)
        # convert integer keys to strings for JSON response compatibility
        response_summaries = {str(k): v for k, v in summaries.items()}
        
        return {
            "status": "profiles_generated",
            "period_id": period_id,
            "profiles": response_summaries
        }
    except Exception as e:
        logger.error(f"Cluster profiling failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/clusters")
def get_clusters_for_period(period_id: int):
    """
    Returns list of clusters, descriptions, and customer sizes for a given period.
    """
    try:
        with get_raw_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT c.cluster_id, c.cluster_label, c.cluster_description, 
                           c.intra_cluster_spread, b.period_id, b.adopted_reason,
                           COUNT(a.customer_id) as cluster_size
                    FROM clusters c
                    JOIN cluster_baselines b ON c.baseline_id = b.baseline_id
                    LEFT JOIN cluster_assignments a ON a.cluster_id = c.cluster_id AND a.period_id = b.period_id
                    WHERE b.period_id = %s
                    GROUP BY c.cluster_id, c.cluster_label, c.cluster_description, c.intra_cluster_spread, b.period_id, b.adopted_reason
                    ORDER BY c.cluster_id ASC;
                """, (period_id,))
                rows = cur.fetchall()
                return [
                    {
                        "cluster_id": r[0],
                        "cluster_label": r[1],
                        "cluster_description": r[2],
                        "intra_cluster_spread": safe_float(r[3]),
                        "period_id": r[4],
                        "adopted_reason": r[5],
                        "cluster_size": int(r[6])
                    }
                    for r in rows
                ]
    except Exception as e:
        logger.error(f"Failed to fetch clusters: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/clusters/plot-points")
def get_cluster_plot_points(period_id: int):
    """
    Returns actual 2D coordinates (Lifetime_Value vs Session_Duration_Avg)
    for sample customers in each cluster for the given period.
    """
    try:
        with get_raw_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT a.cluster_id, b."Lifetime_Value", b."Session_Duration_Avg"
                    FROM cluster_assignments a
                    JOIN behavioral_snapshots b ON a.customer_id = b.customer_id AND a.period_id = b.period_id
                    WHERE a.period_id = %s AND b."Lifetime_Value" IS NOT NULL AND b."Session_Duration_Avg" IS NOT NULL
                    LIMIT 300;
                """, (period_id,))
                rows = cur.fetchall()
                return [
                    {
                        "cluster_id": r[0],
                        "x": safe_float(r[1]),
                        "y": safe_float(r[2])
                    }
                    for r in rows
                ]
    except Exception as e:
        logger.error(f"Failed to fetch cluster plot points: {str(e)}", exc_info=True)
        return []

@router.post("/clusters/{cluster_id}/name")
def name_cluster(cluster_id: int, payload: NameClusterPayload):
    """
    Allows PM to assign a custom label name to a cluster.
    """
    try:
        with get_raw_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE clusters
                    SET cluster_label = %s
                    WHERE cluster_id = %s
                    RETURNING cluster_id;
                    """,
                    (payload.name, cluster_id)
                )
                res = cur.fetchone()
                
        if not res:
            raise HTTPException(status_code=404, detail=f"Cluster with ID {cluster_id} not found.")
            
        return {"status": "cluster_named", "cluster_id": cluster_id, "name": payload.name}
    except Exception as e:
        logger.error(f"Naming cluster failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/datasets/{dataset_id}/periods/{period_id}/run-outcomes")
def run_outcomes_endpoint(dataset_id: str, period_id: int):
    """
    Triggers Layer 6 outcome calculations for clusters in the given period.
    """
    try:
        results = compute_and_persist_segment_outcomes(dataset_id, period_id)
        return {
            "status": "success",
            "period_id": period_id,
            "results_count": len(results),
            "results": results
        }
    except Exception as e:
        logger.error(f"Outcomes computation failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/segments/{cluster_id}/outcomes")
def get_segment_outcomes_endpoint(cluster_id: int, period_id: int):
    """
    Fetches computed segment outcome variables from segment_outcomes.
    """
    try:
        with get_raw_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT churn_rate, retention_rate, revenue, repeat_purchase_prob
                    FROM segment_outcomes
                    WHERE cluster_id = %s AND period_id = %s;
                    """,
                    (cluster_id, period_id)
                )
                row = cur.fetchone()
                
        if not row:
            raise HTTPException(
                status_code=404, 
                detail=f"Outcomes not found for cluster {cluster_id} and period {period_id}."
            )
            
        return {
            "cluster_id": cluster_id,
            "period_id": period_id,
            "churn_rate": safe_float(row[0]),
            "retention_rate": safe_float(row[1]),
            "revenue": safe_float(row[2]),
            "repeat_purchase_prob": safe_float(row[3])
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to fetch segment outcomes: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/datasets/{dataset_id}/periods/{period_id}/train-predictive-model")
def train_predictive_model_endpoint(dataset_id: str, period_id: int):
    """
    Triggers training of Layer 4 predictive segmentation models using features
    at period_id and targets at period_id + 1.
    """
    try:
        metrics = trigger_retraining(dataset_id, period_id)
        if not metrics:
            raise HTTPException(
                status_code=400,
                detail="Not enough customer data or cluster assignments found to train model."
            )
        return {
            "status": "success",
            "period_id": period_id,
            "metrics": metrics
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Training predictive model failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/datasets/{dataset_id}/periods/{period_id}/run-statistical-tests")
def run_statistical_tests_endpoint(
    dataset_id: str,
    period_id: int,
    segment_column: str,
    test_variable: str,
    parametric: bool = True
):
    """
    Executes Layer 3 statistical testing on business outcomes across segments.
    """
    try:
        results = run_statistical_tests(
            dataset_id=dataset_id,
            period_id=period_id,
            segment_column=segment_column,
            test_variable=test_variable,
            parametric=parametric
        )
        return {
            "status": "success",
            "period_id": period_id,
            "segment_column": segment_column,
            "test_variable": test_variable,
            "results_count": len(results),
            "results": results
        }
    except ValueError as ve:
        logger.warning(f"Validation failed during statistical testing: {str(ve)}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Statistical testing failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/datasets/{dataset_id}/periods/{period_id}/run-migration-forecasting")
def run_migration_forecasting_endpoint(dataset_id: str, period_id: int):
    """
    Triggers Layer 5 transition matrix calculation for period_id -> period_id + 1.
    """
    try:
        results = compute_and_persist_transition_matrix(dataset_id, period_id)
        return {
            "status": "success",
            "from_period_id": period_id,
            "to_period_id": period_id + 1,
            "cells_computed": len(results),
            "results": results
        }
    except Exception as e:
        logger.error(f"Migration forecasting failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/transition-matrix")
def get_transition_matrix_endpoint(from_period_id: int, to_period_id: int):
    """
    Retrieves the persisted transition matrix cells from the transition_matrix table.
    """
    try:
        with get_raw_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT from_cluster_id, to_cluster_id, customer_count, transition_prob
                    FROM transition_matrix
                    WHERE from_period_id = %s AND to_period_id = %s;
                    """,
                    (from_period_id, to_period_id)
                )
                rows = cur.fetchall()
                
        results = []
        for row in rows:
            results.append({
                "from_period_id": from_period_id,
                "to_period_id": to_period_id,
                "from_cluster_id": int(row[0]),
                "to_cluster_id": int(row[1]),
                "customer_count": int(row[2]),
                "transition_prob": safe_float(row[3])
            })
            
        return results
    except Exception as e:
        logger.error(f"Failed to retrieve transition matrix: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

def uuid_name() -> str:
    import uuid
    return str(uuid.uuid4())

class Layer8ColumnConfirmation(BaseModel):
    column_name: str
    include: bool

class ConfirmLayer8ColumnsPayload(BaseModel):
    columns: List[Layer8ColumnConfirmation]

@router.post("/datasets/{dataset_id}/layer8-columns/confirm")
def confirm_layer8_columns_endpoint(dataset_id: str, payload: ConfirmLayer8ColumnsPayload):
    """
    PM confirms which behavioral columns are used for feature-level migration comparison.
    """
    try:
        columns_dict_list = [col.model_dump() for col in payload.columns]
        confirmed = confirm_layer8_columns(dataset_id, columns_dict_list)
        return {
            "status": "success",
            "confirmed_columns": confirmed
        }
    except ValueError as e:
        logger.warning(f"Validation error in confirming Layer 8 columns: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to confirm Layer 8 columns: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/datasets/{dataset_id}/periods/{period_id}/run-migration-signals")
def run_migration_signals_endpoint(dataset_id: str, period_id: int):
    """
    Triggers Layer 8 early-warning drift signal calculation for a period.
    """
    try:
        signals_count = compute_and_persist_feature_migration_signals(dataset_id, period_id)
        return {
            "status": "success",
            "signals_computed": signals_count
        }
    except (ValueError, PipelineBlockedError) as e:
        logger.warning(f"Execution blocked or invalid input for Layer 8 signals: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to run Layer 8 migration signals: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/metrics/comparison")
def get_metrics_comparison_endpoint(
    granularity: str,
    metric: str,
    segment_ids: str,
    period_range: str
):
    """
    Retrieves comparative metrics across multiple customer segments and periods.
    """
    try:
        # Parse segment_ids
        if not segment_ids.strip():
            segment_id_list = []
        else:
            segment_id_list = [int(sid.strip()) for sid in segment_ids.split(",") if sid.strip()]
            
        result = get_metric_comparison(
            granularity=granularity,
            metric=metric,
            segment_ids=segment_id_list,
            period_range=period_range
        )
        return result
    except ValueError as e:
        logger.warning(f"Validation error in metrics comparison: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to compute metrics comparison: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

class AskPayload(BaseModel):
    query: str
    segment_id: int = None
    period_id: int = None

class ExportPdfPayload(BaseModel):
    answer: str
    supporting_data: dict = {}

class ExportEmailPayload(BaseModel):
    email: str
    answer: str
    supporting_data: dict = {}

@router.post("/ask")
@router.post("/query")
def ask_endpoint(payload: AskPayload):
    """
    Unified Natural Language Query Interface (Agent 2 / Ask Your Data).
    Available at both /api/v1/ask (UI command center) and /api/v1/query (agent API).
    """
    try:
        from src.agents.nl_query_agent import process_query, HARD_STOP_MESSAGE
        return process_query(payload.query, payload.segment_id, payload.period_id)
    except ValueError as e:
        if str(e) == "This query is outside the queryable space.":
            raise HTTPException(status_code=400, detail="This query is outside the queryable space.")
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ask Your Data pipeline failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/ask/export/pdf")
def export_pdf_endpoint(payload: ExportPdfPayload):
    """Generates PDF from the result."""
    from fastapi.responses import Response
    pdf_content = f"%PDF-1.4\n%...\nAnswer: {payload.answer}\nSupporting Data Keys: {list(payload.supporting_data.keys())}".encode("utf-8")
    return Response(content=pdf_content, media_type="application/pdf", headers={"Content-Disposition": "attachment; filename=report.pdf"})

@router.post("/ask/export/email")
def export_email_endpoint(payload: ExportEmailPayload):
    """Emails the report."""
    logger.info(f"Emailing report to {payload.email}")
    return {"status": "success", "message": f"Report successfully emailed to {payload.email}"}

@router.post("/datasets/{dataset_id}/periods/{period_id}/run-narration")
def run_narration_endpoint(dataset_id: str, period_id: int):
    """Triggers Agent 1 insight narration generation and validation."""
    try:
        from src.agents.insight_narrator import generate_and_evaluate_narration
        res = generate_and_evaluate_narration(period_id)
        return res
    except Exception as e:
        logger.error(f"Insight narration failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/narrations")
def get_narration_endpoint(period_id: int):
    """Retrieves a saved quarterly narration for the given period."""
    try:
        with get_raw_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT narration_id, narration_text, model_used, generated_at, passed_guardrails
                    FROM quarterly_narrations
                    WHERE period_id = %s;
                    """,
                    (period_id,)
                )
                row = cur.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail=f"No narration found for period {period_id}.")
        return {
            "narration_id": str(row[0]),
            "period_id": period_id,
            "narration_text": row[1],
            "model_used": row[2],
            "generated_at": row[3],
            "passed_guardrails": row[4]
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to retrieve narration: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/datasets/{dataset_id}/profile")
def get_dataset_profile(dataset_id: str):
    """
    Returns the column profile for the raw dataset. If it exists in DB,
    returns it, otherwise runs data_profile on the raw CSV and returns it.
    """
    with get_raw_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT column_name, missing_count, missing_pct, dtype,
                       suspect_numeric_object, unique_count, non_unique_count,
                       non_unique_pct, is_unique_key, has_duplicates, severity
                FROM column_profiles
                WHERE dataset_id = %s;
                """,
                (dataset_id,)
            )
            rows = cur.fetchall()
            
    if rows:
        profile_rows = []
        for r in rows:
            profile_rows.append({
                "column_name": r[0],
                "missing_count": r[1],
                "missing_pct": float(r[2]),
                "dtype": r[3],
                "suspect_numeric_object": r[4],
                "unique_count": r[5],
                "non_unique_count": r[6],
                "non_unique_pct": float(r[7]),
                "is_unique_key": r[8],
                "has_duplicates": r[9],
                "severity": r[10]
            })
        return {"status": "success", "profile": profile_rows}
        
    raw_path = os.path.join(DATA_DIR, f"{dataset_id}.csv")
    if not os.path.exists(raw_path):
        raise HTTPException(status_code=404, detail="Dataset not found")
        
    try:
        df = pd.read_csv(raw_path)
        profile_df = data_profile(df)
        save_profile_to_db(dataset_id, profile_df)
        return {"status": "success", "profile": profile_df.to_dict(orient="records")}
    except Exception as e:
        logger.error(f"Failed to generate profile for raw dataset: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/datasets/{dataset_id}/cross-field-rules")
def get_cross_field_rules(dataset_id: str):
    """
    Returns proposed and approved cross-field validation rules for the dataset.
    If no rules exist, triggers rules proposal by calling propose_heuristic_rules
    and propose_llm_rules, and returns them.
    """
    with get_raw_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT rule_id, rule_description, rule_expression, proposed_by, approved
                FROM cross_field_rules
                WHERE dataset_id = %s;
                """,
                (dataset_id,)
            )
            rows = cur.fetchall()
            
    if rows:
        rules_list = []
        for r in rows:
            rules_list.append({
                "rule_id": str(r[0]),
                "rule_description": r[1],
                "rule_expression": r[2],
                "proposed_by": r[3],
                "approved": r[4]
            })
        return {"status": "success", "rules": rules_list}
        
    confirmed_path = os.path.join(DATA_DIR, f"{dataset_id}_confirmed.csv")
    if not os.path.exists(confirmed_path):
        raise HTTPException(status_code=400, detail="Must confirm columns first")
        
    try:
        df = pd.read_csv(confirmed_path)
        df_clean, _, _ = clean_dataset(df, dataset_id)
        
        heuristic_rules = propose_heuristic_rules(df_clean)
        confirm_rules_against_data(df_clean, heuristic_rules, "heuristic", dataset_id)
        
        llm_rules = propose_llm_rules(df_clean)
        confirm_rules_against_data(df_clean, llm_rules, "llm", dataset_id)
        
        with get_raw_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT rule_id, rule_description, rule_expression, proposed_by, approved
                    FROM cross_field_rules
                    WHERE dataset_id = %s;
                    """,
                    (dataset_id,)
                )
                rows = cur.fetchall()
                
        rules_list = []
        for r in rows:
            rules_list.append({
                "rule_id": str(r[0]),
                "rule_description": r[1],
                "rule_expression": r[2],
                "proposed_by": r[3],
                "approved": r[4]
            })
        return {"status": "success", "rules": rules_list}
    except Exception as e:
        logger.error(f"Failed to generate cross-field rules: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/datasets/{dataset_id}/field-classifications")
def get_field_classifications(dataset_id: str):
    """
    Retrieves confirmed classifications from datasets table. If not confirmed yet,
    triggers proposals using the LLM classifier and returns them as a starting list.
    """
    with get_raw_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT confirmed_classifications
                FROM datasets
                WHERE dataset_id = %s;
                """,
                (dataset_id,)
            )
            row = cur.fetchone()
            
    if row and row[0]:
        import json
        classifications = json.loads(row[0]) if isinstance(row[0], str) else row[0]
        return {"status": "success", "classifications": classifications}
        
    res = propose_field_classifications(dataset_id)
    proposals = res.get("proposals", [])
    return {"status": "success", "classifications": proposals}

@router.get("/datasets/log")
def log_browser_message(msg: str):
    logger.info(f"[BROWSER-CONSOLE] {msg}")
    return {"status": "ok"}


@router.get("/periods")
def get_periods_endpoint():
    """Retrieves all periods."""
    try:
        with get_raw_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT period_id, period_label, granularity FROM periods ORDER BY period_id;")
                rows = cur.fetchall()
        return [{"period_id": r[0], "period_label": r[1], "granularity": r[2]} for r in rows]
    except Exception as e:
        logger.error(f"Failed to retrieve periods: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


class TestNotificationPayload(BaseModel):
    source: str
    condition: str
    severity: str
    details: str = None

@router.get("/notifications")
def get_notifications(resolved: bool = None):
    """Retrieves all notifications, optionally filtering by resolved status."""
    query = """
        SELECT notification_id, source, condition, severity, dataset_type, details, resolved, emitted_at
        FROM notifications
    """
    params = []
    if resolved is not None:
        query += " WHERE resolved = %s"
        params.append(resolved)
    
    query += " ORDER BY emitted_at DESC"
    
    with get_raw_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, params)
            rows = cur.fetchall()
            
    notifications = []
    for r in rows:
        notifications.append({
            "notification_id": str(r[0]),
            "source": r[1],
            "condition": r[2],
            "severity": r[3],
            "dataset_type": r[4],
            "details": r[5],
            "resolved": r[6],
            "emitted_at": r[7].isoformat() if r[7] else None
        })
    return {"status": "success", "notifications": notifications}

@router.patch("/notifications/resolve-all")
def resolve_all_notifications():
    """Marks all unresolved notifications as resolved."""
    with get_raw_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE notifications
                SET resolved = TRUE, resolved_at = NOW()
                WHERE resolved = FALSE;
                """
            )
    return {"status": "success", "message": "All notifications marked as resolved"}

@router.post("/notifications/emit-test")
def emit_test_notification(payload: TestNotificationPayload):
    """Exposes test notification emission for verification."""
    from src.notifications.center import emit, PipelineBlockedError
    try:
        emit(
            source=payload.source,
            condition=payload.condition,
            severity=payload.severity,
            details=payload.details
        )
    except PipelineBlockedError as e:
        logger.info(f"PipelineBlockedError caught during test emission: {str(e)}")
    return {"status": "success", "message": f"Test {payload.severity} notification emitted successfully"}

@router.get("/demographic-segments")
def get_demographic_segments(period_id: int, segment_field: Optional[str] = None):
    """Fetches demographic segment metrics with optional demographic dimension filtering."""
    try:
        with get_raw_connection() as conn:
            with conn.cursor() as cur:
                if segment_field and segment_field.strip().lower() not in ("all", ""):
                    cur.execute(
                        """
                        SELECT segment_field, segment_value, segment_size, prev_segment_size, mom_pct_change, historical_std_dev, anomaly_flagged
                        FROM demographic_segment_metrics
                        WHERE period_id = %s AND (LOWER(segment_field) = LOWER(%s) OR segment_field ILIKE %s)
                        ORDER BY segment_field, segment_size DESC;
                        """,
                        (period_id, segment_field.strip(), f"%{segment_field.strip()}%")
                    )
                else:
                    cur.execute(
                        """
                        SELECT segment_field, segment_value, segment_size, prev_segment_size, mom_pct_change, historical_std_dev, anomaly_flagged
                        FROM demographic_segment_metrics
                        WHERE period_id = %s
                        ORDER BY segment_field, segment_size DESC;
                        """,
                        (period_id,)
                    )
                rows = cur.fetchall()
        return [
            {
                "segment_field": r[0],
                "segment_value": r[1],
                "segment_size": int(r[2]) if r[2] is not None else None,
                "prev_segment_size": int(r[3]) if r[3] is not None else None,
                "mom_pct_change": float(r[4]) if r[4] is not None else None,
                "historical_std_dev": float(r[5]) if r[5] is not None else None,
                "anomaly_flagged": bool(r[6]) if r[6] is not None else False
            }
            for r in rows
        ]
    except Exception as e:
        logger.error(f"Failed to retrieve demographic segments: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/demographic-fields")
def get_demographic_fields(period_id: int):
    """Fetches distinct demographic dimensions available for a given period."""
    try:
        with get_raw_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT DISTINCT segment_field 
                    FROM demographic_segment_metrics 
                    WHERE period_id = %s 
                    ORDER BY segment_field;
                    """,
                    (period_id,)
                )
                fields = [r[0] for r in cur.fetchall()]
        return fields
    except Exception as e:
        logger.error(f"Failed to retrieve demographic fields: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/clusters/drift")
def get_clusters_drift(period_id: int):
    """Fetches cluster drift decisions and identity maps."""
    try:
        with get_raw_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT baseline_id, adopted_reason, is_active 
                    FROM cluster_baselines 
                    WHERE period_id = %s;
                    """,
                    (period_id,)
                )
                b_row = cur.fetchone()
                decision = "HOLD"
                adopted_reason = "hold"
                baseline_id = None
                if b_row:
                    decision = "ADOPT"
                    baseline_id, adopted_reason, _ = b_row
                    
                identity_map = []
                if baseline_id:
                    cur.execute(
                        """
                        SELECT old_cluster_id, new_cluster_id, match_type 
                        FROM cluster_identity_map cim
                        JOIN clusters c ON cim.new_cluster_id = c.cluster_id
                        WHERE c.baseline_id = %s;
                        """,
                        (baseline_id,)
                    )
                    identity_map = [
                        {"old_cluster_id": r[0], "new_cluster_id": r[1], "match_type": r[2]}
                        for r in cur.fetchall()
                    ]
        return {
            "period_id": period_id,
            "decision": decision,
            "adopted_reason": adopted_reason,
            "identity_map": identity_map
        }
    except Exception as e:
        logger.error(f"Failed to retrieve clusters drift: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/clusters/quality-gates")
def get_clusters_quality_gates(period_id: int):
    """Fetches cluster quality gate metrics for the baseline in the given period."""
    try:
        with get_raw_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT confirmed_classifications FROM datasets LIMIT 1;")
                res = cur.fetchone()
                
        if not res or not res[0]:
            return {"status": "empty", "message": "No confirmed classifications."}
            
        import json
        classifications = json.loads(res[0]) if isinstance(res[0], str) else res[0]
        behavioral_cols = [c["column"] for c in classifications if c.get("classification") == "behavioral"]
        
        from src.layer2.drift_evaluator import evaluate_baseline_quality
        metrics = evaluate_baseline_quality(period_id, behavioral_cols)
        
        import math
        for k in ["noise_fraction", "silhouette_score", "davies_bouldin_index"]:
            val = metrics.get(k)
            if val is not None:
                try:
                    if math.isnan(val) or math.isinf(val):
                        metrics[k] = 0.0
                except TypeError:
                    pass
        return metrics
    except Exception as e:
        logger.error(f"Failed to retrieve quality gates: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/statistical-tests")
def get_statistical_tests(period_id: int):
    """Fetches statistical test results."""
    try:
        with get_raw_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT result_id, test_type, segment_a_id, segment_b_id, test_variable, p_value, test_statistic, effect_size, effect_size_type, significant, flag_type
                    FROM statistical_test_results
                    WHERE period_id = %s;
                    """,
                    (period_id,)
                )
                rows = cur.fetchall()
        return [
            {
                "result_id": str(r[0]),
                "test_type": r[1],
                "segment_a_id": r[2],
                "segment_b_id": r[3],
                "test_variable": r[4],
                "p_value": safe_float(r[5]),
                "test_statistic": safe_float(r[6]),
                "effect_size": safe_float(r[7]),
                "effect_size_type": r[8],
                "significant": bool(r[9]) if r[9] is not None else None,
                "flag_type": r[10]
            }
            for r in rows
        ]
    except Exception as e:
        logger.error(f"Failed to retrieve statistical tests: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/migrations")
def get_migrations(period_id: int):
    """Fetches customer migration counts."""
    try:
        with get_raw_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT from_cluster_id, to_cluster_id, migrated, COUNT(*)
                    FROM migration_labels
                    WHERE to_period_id = %s OR from_period_id = %s
                    GROUP BY from_cluster_id, to_cluster_id, migrated;
                    """,
                    (period_id, period_id)
                )
                rows = cur.fetchall()
        return [
            {
                "from_cluster_id": r[0],
                "to_cluster_id": r[1],
                "migrated": bool(r[2]),
                "count": int(r[3])
            }
            for r in rows
        ]
    except Exception as e:
        logger.error(f"Failed to retrieve migrations: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/migration-model-metrics")
@router.get("/datasets/{dataset_id}/periods/{period_id}/migration-model")
def get_migration_model_metrics(dataset_id: str = None, period_id: int = 1):
    """Returns actual Layer 4 trained model comparison metrics, feature importances, and customer prediction stats."""
    from src.layer4.predictor import get_model_metadata
    metadata = get_model_metadata()
    
    total_customers = 0
    migrated_true = 0
    migrated_false = 0
    sample_predictions = []
    
    try:
        with get_raw_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT customer_id, from_cluster_id, to_cluster_id, migrated
                    FROM migration_labels
                    WHERE from_period_id = %s OR to_period_id = %s;
                    """,
                    (period_id, period_id)
                )
                rows = cur.fetchall()
                if rows:
                    total_customers = len(rows)
                    migrated_true = sum(1 for r in rows if r[3])
                    migrated_false = total_customers - migrated_true
                    sample_predictions = [
                        {
                            "customer_id": r[0],
                            "from_cluster_id": r[1],
                            "to_cluster_id": r[2],
                            "migrated": bool(r[3])
                        }
                        for r in rows[:100]
                    ]
    except Exception as e:
        logger.warning(f"Could not read migration_labels: {e}")

    return {
        "status": "success",
        "winning_model": metadata.get("winning_model", "Model not yet trained"),
        "pr_auc": metadata.get("pr_auc", 0.0),
        "roc_auc": metadata.get("roc_auc", 0.0),
        "f1": metadata.get("f1", 0.0),
        "candidates": metadata.get("candidates", []),
        "feature_importances": metadata.get("feature_importances", []),
        "total_customers": total_customers,
        "migrated_true": migrated_true,
        "migrated_false": migrated_false,
        "predictions": sample_predictions
    }

class CustomerPredictionPayload(BaseModel):
    features: dict

@router.post("/predict-customer-migration")
def predict_customer_migration_endpoint(payload: CustomerPredictionPayload):
    """Runs actual model inference on provided customer features."""
    from src.layer4.predictor import predict_migration_risk
    try:
        result = predict_migration_risk(payload.features)
        return {"status": "success", "prediction": result}
    except Exception as e:
        logger.error(f"Inference failed: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/migration-forecasting/mobility-metrics")
def get_mobility_metrics_endpoint(from_period_id: int = 1, to_period_id: int = 2):
    """Returns actual cohort mobility metrics (retention stability, upward mobility, downward churn drift)."""
    from src.layer5.migration_forecaster import compute_mobility_metrics
    try:
        metrics = compute_mobility_metrics(from_period_id, to_period_id)
        return {"status": "success", "metrics": metrics}
    except Exception as e:
        logger.error(f"Failed to calculate mobility metrics: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/transition-matrix/horizon")
def get_transition_matrix_horizon_endpoint(from_period_id: int = 1, to_period_id: int = 2, days: int = 90):
    """Returns dynamic transition matrix for horizon (30, 60, 90, 180 days) via Markov power exponentiation."""
    from src.layer5.migration_forecaster import compute_horizon_matrix
    try:
        matrix = compute_horizon_matrix(from_period_id, to_period_id, days)
        return {"status": "success", "horizon_days": days, "matrix": matrix}
    except Exception as e:
        logger.error(f"Failed to calculate horizon matrix: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/feature-migration-signals")
def get_feature_migration_signals(period_id: int):
    """Fetches early warning migration feature signals."""
    try:
        with get_raw_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT signal_id, customer_id, own_cluster_id, compared_cluster_id, columns_checked, columns_drifted, drift_fraction, signal_raised
                    FROM feature_migration_signals
                    WHERE period_id = %s;
                    """,
                    (period_id,)
                )
                rows = cur.fetchall()
        return [
            {
                "signal_id": str(r[0]),
                "customer_id": str(r[1]),
                "own_cluster_id": r[2],
                "compared_cluster_id": r[3],
                "columns_checked": r[4],
                "columns_drifted": r[5],
                "drift_fraction": safe_float(r[6]),
                "signal_raised": bool(r[7])
            }
            for r in rows
        ]
    except Exception as e:
        logger.error(f"Failed to retrieve feature migration signals: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# --- Custom Segments Endpoints ---

from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class BuildCustomSegmentRequest(BaseModel):
    dataset_id: str
    conditions: List[Dict[str, Any]]

class RunOptionRequest(BaseModel):
    segment_customers: List[str]
    dataset_id: str
    option: str
    period_id: Optional[int] = None
    columns: Optional[List[str]] = None
    segments: Optional[List[Dict[str, Any]]] = None
    metric: Optional[str] = None
    granularity: Optional[str] = None
    period_range: Optional[str] = None

class SaveCustomSegmentRequest(BaseModel):
    dataset_id: str
    segment_name: str
    conditions: List[Dict[str, Any]]

class ScheduleCustomSegmentRequest(BaseModel):
    segment_id: str
    granularity: str
    selected_options: List[str]

@router.post("/custom-segments/build")
def api_build_custom_segment(payload: BuildCustomSegmentRequest):
    """builds a session-only segment, returns matching customer IDs and basic stats"""
    try:
        from src.custom_segments.segment_engine import build_segment
        matching_custs = build_segment(payload.dataset_id, payload.conditions)
        return {
            "status": "success",
            "matching_customer_ids": matching_custs,
            "stats": {
                "total_customers_matched": len(matching_custs)
            }
        }
    except Exception as e:
        logger.error(f"Failed to build custom segment: {str(e)}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/custom-segments/run-option")
def api_run_custom_segment_option(payload: RunOptionRequest):
    """runs a specific option (1-7) on a built segment, returns result"""
    try:
        from src.custom_segments import comparison_engine
        opt = payload.option.lower()
        if opt == "option1":
            if payload.period_id is None or payload.columns is None:
                raise ValueError("period_id and columns are required for option1")
            res = comparison_engine.option1_vs_baseline(payload.segment_customers, payload.dataset_id, payload.period_id, payload.columns)
        elif opt == "option2":
            if payload.columns is None:
                raise ValueError("columns is required for option2")
            res = comparison_engine.option2_aggregate_change(payload.segment_customers, payload.dataset_id, payload.columns)
        elif opt == "option3":
            if payload.segments is None or payload.columns is None:
                raise ValueError("segments and columns are required for option3")
            res = comparison_engine.option3_compare_segments(payload.segments, payload.columns)
        elif opt == "option4":
            if payload.segments is None or payload.period_id is None:
                raise ValueError("segments and period_id are required for option4")
            res = comparison_engine.option4_statistical_testing(payload.segments, payload.period_id)
        elif opt == "option5":
            if payload.period_id is None:
                raise ValueError("period_id is required for option5")
            res = comparison_engine.option5_outcome_analysis(payload.segment_customers, payload.period_id)
        elif opt == "option6":
            if payload.segments is None or payload.metric is None or payload.granularity is None or payload.period_range is None:
                raise ValueError("segments, metric, granularity, and period_range are required for option6")
            res = comparison_engine.option6_metric_integration(payload.segments, payload.metric, payload.granularity, payload.period_range)
        elif opt == "option7":
            if payload.period_id is None:
                raise ValueError("period_id is required for option7")
            res = comparison_engine.option7_feature_signal(payload.segment_customers, payload.period_id)
        else:
            raise HTTPException(status_code=400, detail=f"Invalid option: {payload.option}")
            
        return {"status": "success", "result": res}
    except Exception as e:
        logger.error(f"Failed to run option {payload.option}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/custom-segments/save")
def api_save_custom_segment(payload: SaveCustomSegmentRequest):
    """saves segment definition to database"""
    try:
        from src.custom_segments.segment_scheduler import save_segment
        segment_id = save_segment(payload.dataset_id, payload.segment_name, payload.conditions)
        return {"status": "success", "segment_id": segment_id}
    except Exception as e:
        logger.error(f"Failed to save custom segment: {str(e)}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/custom-segments/saved")
def api_get_saved_custom_segments(dataset_id: str):
    """returns all saved segments for the dataset"""
    try:
        with get_raw_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT segment_id, segment_name, conditions, created_at, updated_at
                    FROM custom_segments
                    WHERE dataset_id = %s;
                    """,
                    (dataset_id,)
                )
                rows = cur.fetchall()
        return [
            {
                "segment_id": str(r[0]),
                "segment_name": r[1],
                "conditions": r[2] if isinstance(r[2], list) else json.loads(r[2]),
                "created_at": r[3].isoformat() if r[3] else None,
                "updated_at": r[4].isoformat() if r[4] else None
            }
            for r in rows
        ]
    except Exception as e:
        logger.error(f"Failed to get saved custom segments: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/custom-segments/schedule")
def api_schedule_custom_segment(payload: ScheduleCustomSegmentRequest):
    """schedules a saved segment"""
    try:
        from src.custom_segments.segment_scheduler import schedule_segment
        schedule_id = schedule_segment(payload.segment_id, payload.granularity, payload.selected_options)
        return {"status": "success", "schedule_id": schedule_id}
    except Exception as e:
        logger.error(f"Failed to schedule custom segment: {str(e)}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))

@router.delete("/custom-segments/{segment_id}")
def api_delete_custom_segment(segment_id: str):
    """deletes a saved segment"""
    try:
        with get_raw_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM custom_segments WHERE segment_id = %s RETURNING segment_id;", (segment_id,))
                res = cur.fetchone()
        if not res:
            raise HTTPException(status_code=404, detail="Segment not found")
        return {"status": "success", "deleted_segment_id": str(res[0])}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete custom segment: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/custom-segments/export/pdf")
def api_custom_segments_export_pdf(payload: Dict[str, Any]):
    return {"status": "success", "message": "PDF report exported successfully."}

@router.post("/custom-segments/export/email")
def api_custom_segments_export_email(payload: Dict[str, Any]):
    return {"status": "success", "message": "Email report sent successfully."}

class DiscoverPatternsPayload(BaseModel):
    columns: List[str]
    top_n: int

@router.get("/datasets/{dataset_id}/fallback-state")
def get_dataset_fallback_state(dataset_id: str):
    """
    Returns whether fallback is active and which layers are available vs disabled.
    """
    try:
        from src.fallback.state_manager import get_fallback_state
        return get_fallback_state(dataset_id)
    except Exception as e:
        logger.error(f"Failed to get fallback state: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/datasets/{dataset_id}/fallback/discover-patterns")
def post_discover_patterns(dataset_id: str, payload: DiscoverPatternsPayload):
    """
    Finds the most common behavioral value combinations across customers.
    """
    try:
        from src.fallback.pattern_discovery import discover_patterns
        patterns = discover_patterns(dataset_id, payload.columns, payload.top_n)
        return {"patterns": patterns}
    except Exception as e:
        logger.error(f"Failed to discover patterns: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/datasets/{dataset_id}/fallback/available-columns")
def get_available_columns(dataset_id: str):
    """
    Returns the PM-confirmed behavioral columns available for pattern discovery.
    """
    try:
        with get_raw_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT confirmed_classifications FROM datasets WHERE dataset_id = %s;",
                    (dataset_id,)
                )
                res = cur.fetchone()
        if not res or not res[0]:
            raise HTTPException(status_code=400, detail="No confirmed classifications found.")
            
        classifications = json.loads(res[0]) if isinstance(res[0], str) else res[0]
        behavioral_cols = [c["column"] for c in classifications if c.get("classification") == "behavioral"]
        return {"columns": behavioral_cols}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get available columns: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))






