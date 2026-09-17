import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
import psycopg2
from psycopg2.extras import RealDictCursor
from contextlib import contextmanager
import logging
from src.utils.logging_config import setup_logging

# Load environment variables from .env file
load_dotenv()

setup_logging()
logger = logging.getLogger("db_connection")

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:postgres_password@localhost:5435/segment_studio"
)

# SQLAlchemy setup
engine = create_engine(DATABASE_URL, pool_size=10, max_overflow=20)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

@contextmanager
def get_db_session():
    """SQLAlchemy session context manager."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception as e:
        session.rollback()
        logger.error(f"Database session error: {str(e)}", exc_info=True)
        raise e
    finally:
        session.close()

@contextmanager
def get_raw_connection():
    """Raw psycopg2 connection context manager with RealDictCursor."""
    conn = psycopg2.connect(DATABASE_URL)
    try:
        yield conn
        conn.commit()
    except Exception as e:
        conn.rollback()
        logger.error(f"Raw database connection error: {str(e)}", exc_info=True)
        raise e
    finally:
        conn.close()
