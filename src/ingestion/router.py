import os
import uuid
import datetime
import logging
import pandas as pd
from src.db.connection import get_raw_connection
from src.notifications.center import emit
from src.utils.logging_config import setup_logging

setup_logging()
logger = logging.getLogger("ingestion_router")

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

def ingest_from_file(file_path: str, file_name: str) -> str:
    """
    Parses a CSV, Excel, or JSON file, saves it locally as a working CSV,
    and registers it as a permanent dataset in the DB.
    """
    logger.info(f"Ingesting dataset from file: {file_name}")
    
    ext = os.path.splitext(file_name)[1].lower()
    if ext == ".csv":
        df = pd.read_csv(file_path)
    elif ext in [".xlsx", ".xls"]:
        df = pd.read_excel(file_path)
    elif ext == ".json":
        df = pd.read_json(file_path)
    else:
        raise ValueError(f"Unsupported file format: {ext}")
        
    dataset_id = str(uuid.uuid4())
    working_path = os.path.join(DATA_DIR, f"{dataset_id}.csv")
    df.to_csv(working_path, index=False)
    
    with get_raw_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO datasets (dataset_id, source_type, permanence, file_name, ingested_at)
                VALUES (%s, 'file_upload', 'permanent', %s, now());
                """,
                (dataset_id, file_name)
            )
            
    logger.info(f"Successfully registered permanent dataset: id={dataset_id}, rows={len(df)}")
    return dataset_id

def ingest_from_db(connection_string: str, query: str) -> str:
    """
    Connects to external DB, pulls query results into a DataFrame,
    saves it locally, registers as a dynamic dataset, and emits drift monitor check.
    """
    logger.info(f"Ingesting dataset from DB connection using query: {query}")
    
    # In a real environment, we connect to connection_string. For this project,
    # we establish the connection and read the SQL query.
    try:
        import sqlalchemy
        engine = sqlalchemy.create_engine(connection_string)
        with engine.connect() as conn_engine:
            df = pd.read_sql(query, conn_engine)
    except Exception as e:
        logger.error(f"Failed to read from DB connection: {str(e)}", exc_info=True)
        raise e
        
    dataset_id = str(uuid.uuid4())
    working_path = os.path.join(DATA_DIR, f"{dataset_id}.csv")
    df.to_csv(working_path, index=False)
    
    with get_raw_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO datasets (dataset_id, source_type, permanence, connection_string, ingested_at)
                VALUES (%s, 'database_connection', 'dynamic', %s, now());
                """,
                (dataset_id, connection_string)
            )
            
    logger.info(f"Successfully registered dynamic dataset: id={dataset_id}, rows={len(df)}")
    
    # Emit warning/info to notifications about dynamic connection established
    emit(
        source="ingestion_router",
        condition="dynamic_connection_established",
        severity="soft_warning",
        dataset_type="database_connection",
        details=f"Dynamic dataset {dataset_id} registered. Schema drift monitoring active."
    )
    
    return dataset_id
