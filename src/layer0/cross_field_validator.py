import logging
import json
import re
import pandas as pd
from src.db.connection import get_raw_connection
from src.llm.claude_client import claude_client
from src.utils.logging_config import setup_logging

setup_logging()
logger = logging.getLogger("cross_field_validator")

def propose_heuristic_rules(df: pd.DataFrame) -> list[dict]:
    """
    Scans column names to propose rules based on naming heuristics.
    """
    logger.info("Running naming heuristics for cross-field rules...")
    cols = df.columns.tolist()
    rules = []
    
    # 1. Start/End date pairs
    # Look for suffix patterns like _start, _end, _start_date, _end_date
    start_cols = [c for c in cols if c.endswith("_start") or c.endswith("_start_date")]
    for sc in start_cols:
        prefix = sc.split("_start")[0]
        ec = next((c for c in cols if c.startswith(prefix) and (c.endswith("_end") or c.endswith("_end_date"))), None)
        if ec:
            rules.append({
                "rule_description": f"End date {ec} must be greater than or equal to start date {sc}",
                "rule_expression": f"df['{ec}'] >= df['{sc}']"
            })
            
    # 2. Total vs individual parts
    # e.g., total_purchases >= average_order_value or similar numeric bounds
    # For general ecommerce columns: Pages_Per_Session >= 0, Session_Duration_Avg >= 0
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    for col in numeric_cols:
        # Most numeric indicators should be non-negative
        if (df[col] >= 0).all():
            rules.append({
                "rule_description": f"Numeric column {col} must be non-negative",
                "rule_expression": f"df['{col}'] >= 0"
            })
            
    return rules

def propose_llm_rules(df: pd.DataFrame) -> list[dict]:
    """
    Calls Claude to propose logical cross-field validation rules based on columns.
    """
    logger.info("Calling Claude to propose cross-field validation rules...")
    cols_and_types = {col: str(df[col].dtype) for col in df.columns}
    
    prompt = f"""You are a data validation expert. Given the list of columns and their types in a dataset:
{json.dumps(cols_and_types, indent=2)}

Propose plausible logical cross-field rules or bounds (e.g. end_date >= start_date, total_purchases >= 0, Session_Duration_Avg >= Pages_Per_Session).
Respond ONLY with a valid JSON array of objects, containing no other text:
[
  {{"rule_description": "Description of rule", "rule_expression": "df['col1'] >= df['col2']"}}
]
Use valid pandas syntax inside 'rule_expression' referencing columns exactly. Keep it simple and logical. Do not use quotes inside the expression other than single quotes for column indexing.
"""
    
    try:
        response = claude_client.call(prompt=prompt, system_prompt="You are a data quality assistant. Return only valid JSON.")
        # Strip code blocks if LLM returns them
        clean_response = response.strip()
        if clean_response.startswith("```json"):
            clean_response = clean_response.split("```json")[1].split("```")[0].strip()
        elif clean_response.startswith("```"):
            clean_response = clean_response.split("```")[1].split("```")[0].strip()
            
        candidates = json.loads(clean_response)
        logger.info(f"Claude proposed {len(candidates)} rules.")
        return candidates
    except Exception as e:
        logger.error(f"Failed to get LLM proposed rules: {str(e)}", exc_info=True)
        return []

def confirm_rules_against_data(df: pd.DataFrame, candidates: list[dict], proposed_by: str, dataset_id: str):
    """
    Evaluates candidate rules against the data and persists successful ones in the DB (approved = False).
    """
    logger.info(f"Confirming {len(candidates)} candidate rules proposed by {proposed_by} against data...")
    confirmed_rules = []
    
    for rule in candidates:
        expr = rule["rule_expression"]
        desc = rule["rule_description"]
        
        try:
            # Safely evaluate expression using pandas eval (restricting environment to df)
            res = eval(expr, {"df": df, "pd": pd})
            
            # Must evaluate to boolean Series or boolean value
            if isinstance(res, pd.Series) and res.dtype == bool:
                pass_rate = res.mean()
                if pass_rate >= 0.95:  # holds on at least 95% of rows
                    confirmed_rules.append(rule)
            elif isinstance(res, bool) and res:
                confirmed_rules.append(rule)
        except Exception as e:
            # If evaluation failed or raised error, ignore candidate
            logger.debug(f"Candidate rule validation failed for '{expr}': {str(e)}")
            continue
            
    # Persist validated candidates (pending PM approval)
    logger.info(f"Persisting {len(confirmed_rules)} confirmed rules to DB...")
    with get_raw_connection() as conn:
        with conn.cursor() as cur:
            for rule in confirmed_rules:
                cur.execute(
                    """
                    INSERT INTO cross_field_rules (dataset_id, rule_description, rule_expression, proposed_by, approved)
                    VALUES (%s, %s, %s, %s, FALSE);
                    """,
                    (dataset_id, rule["rule_description"], rule["rule_expression"], proposed_by)
                )

def approve_rule(dataset_id: str, rule_id: str, approved: bool):
    """
    Approves or rejects a proposed cross-field rule.
    """
    logger.info(f"Setting rule approval status: rule={rule_id}, approved={approved}")
    with get_raw_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE cross_field_rules
                SET approved = %s, approved_at = CASE WHEN %s THEN now() ELSE NULL END
                WHERE dataset_id = %s AND rule_id = %s;
                """,
                (approved, approved, dataset_id, rule_id)
            )

def get_active_rules(dataset_id: str) -> list[str]:
    """
    Retrieves all approved rules for a dataset.
    """
    with get_raw_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT rule_expression
                FROM cross_field_rules
                WHERE dataset_id = %s AND approved = TRUE;
                """,
                (dataset_id,)
            )
            return [row[0] for row in cur.fetchall()]
