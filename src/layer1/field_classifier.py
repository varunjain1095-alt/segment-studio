import logging
import json
import pandas as pd
from src.db.connection import get_raw_connection
from src.llm.claude_client import claude_client
from src.notifications.center import emit
from src.utils.logging_config import setup_logging

setup_logging()
logger = logging.getLogger("field_classifier")

def classify_fields(dataset_id: str, profile_df: pd.DataFrame) -> list[dict]:
    """
    Classifies dataset columns into demographic, behavioral, or unsure using Claude.
    Stores the proposals in the datasets table.
    """
    logger.info(f"Classifying fields for dataset {dataset_id}...")
    
    # Extract columns and dtypes
    cols_and_types = []
    for _, row in profile_df.iterrows():
        cols_and_types.append({
            "column": row["column_name"],
            "dtype": row["dtype"]
        })
        
    prompt = f"""You are a customer segmentation assistant. Given a list of columns and their types in a customer dataset:
{json.dumps(cols_and_types, indent=2)}

Your task is to classify each column into one of three categories:
1. "demographic": Information about who the customer is (e.g. age, gender, location, income, country).
2. "behavioral": Information about what the customer does (e.g. login frequency, order counts, session duration, returns, engagement metrics).
3. "unsure": If the classification is ambiguous or requires human verification.

System Principle: You are an AI assistant who proposes classifications. The human PM is the final decision-maker. Do not assign segment names or make decisions; only propose category tags.

Response format: Return ONLY a valid JSON list of objects (no other text, no markdown wrappers):
[
  {{"column": "column_name", "classification": "demographic|behavioral|unsure"}}
]
"""

    system_prompt = "You are a data classification assistant. Respond only with a JSON array."
    
    try:
        response = claude_client.call(prompt=prompt, system_prompt=system_prompt)
        clean_response = response.strip()
        if clean_response.startswith("```json"):
            clean_response = clean_response.split("```json")[1].split("```")[0].strip()
        elif clean_response.startswith("```"):
            clean_response = clean_response.split("```")[1].split("```")[0].strip()
            
        proposals = json.loads(clean_response)
        
        # Save to DB
        with get_raw_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE datasets
                    SET proposed_classifications = %s
                    WHERE dataset_id = %s;
                    """,
                    (json.dumps(proposals), dataset_id)
                )
                
        # Check for dynamic dataset unsure columns
        logger.info(f"Field classification proposals successfully saved for dataset {dataset_id}.")
        
        # Get permanence flag
        with get_raw_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT permanence FROM datasets WHERE dataset_id = %s;", (dataset_id,))
                res = cur.fetchone()
                permanence = res[0] if res else "permanent"
                
        verify_classifications(dataset_id, permanence, proposals)
        return proposals
    except Exception as e:
        logger.error(f"Failed to classify fields via LLM: {str(e)}", exc_info=True)
        # Fallback to unsure classifications
        fallback = [{"column": row["column_name"], "classification": "unsure"} for _, row in profile_df.iterrows()]
        with get_raw_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE datasets
                    SET proposed_classifications = %s
                    WHERE dataset_id = %s;
                    """,
                    (json.dumps(fallback), dataset_id)
                )
        return fallback

def verify_classifications(dataset_id: str, permanence: str, classifications: list[dict]):
    """
    If dataset is dynamic (database_connection), emits a hard_fail notification
    for any column classified as unsure.
    """
    if permanence == "dynamic":
        for item in classifications:
            if item.get("classification") == "unsure":
                msg = f"Column '{item['column']}' has unsure classification on dynamic dataset {dataset_id}. PM attention required."
                logger.error(msg)
                emit(
                    source="layer1_field_classifier",
                    condition="PM attention required — field classification",
                    severity="hard_fail",
                    dataset_type="database_connection",
                    details=msg
                )

def map_dtype_to_pg(dtype_str: str) -> str:
    """Maps python/pandas dtype string to standard PostgreSQL types."""
    dtype_lower = dtype_str.lower()
    if "int" in dtype_lower:
        return "INTEGER"
    elif "float" in dtype_lower or "double" in dtype_lower or "decimal" in dtype_lower or "numeric" in dtype_lower:
        return "NUMERIC"
    elif "bool" in dtype_lower:
        return "BOOLEAN"
    else:
        return "TEXT"

def confirm_field_classifications(dataset_id: str, classifications: list[dict]):
    """
    Saves PM-confirmed classifications as active field classifications.
    Alters demographic_snapshots table dynamically to include confirmed demographic fields.
    """
    logger.info(f"Confirming field classifications for dataset {dataset_id}...")
    
    with get_raw_connection() as conn:
        with conn.cursor() as cur:
            # 1. Update datasets table
            cur.execute(
                """
                UPDATE datasets
                SET confirmed_classifications = %s
                WHERE dataset_id = %s;
                """,
                (json.dumps(classifications), dataset_id)
            )
            
            # 2. Query column profiles for dtypes to perform ALTER TABLE
            cur.execute(
                """
                SELECT column_name, dtype
                FROM column_profiles
                WHERE dataset_id = %s;
                """,
                (dataset_id,)
            )
            profiles = {row[0]: row[1] for row in cur.fetchall()}
            
            # 3. Dynamically add confirmed demographic & behavioral columns
            for item in classifications:
                col_name = item["column"]
                cls = item["classification"]
                
                if cls == "demographic":
                    # Get postgres data type representation
                    dtype_str = profiles.get(col_name, "object")
                    pg_type = map_dtype_to_pg(dtype_str)
                    
                    alter_query = f'ALTER TABLE demographic_snapshots ADD COLUMN IF NOT EXISTS "{col_name}" {pg_type};'
                    logger.info(f"Running dynamic schema update: {alter_query}")
                    cur.execute(alter_query)
                elif cls == "behavioral":
                    dtype_str = profiles.get(col_name, "object")
                    pg_type = map_dtype_to_pg(dtype_str)
                    
                    alter_query = f'ALTER TABLE behavioral_snapshots ADD COLUMN IF NOT EXISTS "{col_name}" {pg_type};'
                    logger.info(f"Running dynamic schema update: {alter_query}")
                    cur.execute(alter_query)
                    
    logger.info(f"Field classifications confirmed successfully for dataset {dataset_id}.")

