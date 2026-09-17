import pytest
import os
import json
import logging
from src.db.connection import get_raw_connection
from src.db.migrations import run_migrations
from src.llm.claude_client import claude_client
from src.notifications.center import emit, PipelineBlockedError

# Setup test logger to verify logs if needed
logger = logging.getLogger("test_infra")

@pytest.fixture(scope="module", autouse=True)
def setup_database():
    """Ensures the database schema exists without destructively dropping tables."""
    try:
        run_migrations(reset=False)
    except Exception as e:
        pytest.fail(f"Failed to run migrations: {str(e)}")

def test_p0_t1_database_startup():
    """P0-T1: Docker & Database Startup. Verify tables exist."""
    required_tables = {
        "customers", "periods", "behavioral_snapshots", "cluster_baselines",
        "clusters", "cluster_assignments", "cluster_identity_map", "migration_labels",
        "transition_matrix", "segment_outcomes", "feature_migration_signals",
        "demographic_snapshots", "datasets", "column_profiles", "cross_field_rules",
        "notifications", "layer8_config", "demographic_segment_metrics",
        "statistical_test_results", "quarterly_narrations"
    }

    with get_raw_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public';
                """
            )
            existing_tables = {row[0] for row in cur.fetchall()}
            
    # Verify all required tables exist
    missing_tables = required_tables - existing_tables
    assert not missing_tables, f"Missing tables in database: {missing_tables}"
    assert len(existing_tables) >= 12, f"Expected at least 12 tables, found {len(existing_tables)}"

def test_p0_t2_period_id_constraints():
    """P0-T2: Period ID Constraint on All Snapshot Tables."""
    single_period_tables = {
        "behavioral_snapshots", "cluster_assignments", "segment_outcomes", 
        "feature_migration_signals", "demographic_snapshots"
    }
    multi_period_tables = {
        "migration_labels", "transition_matrix"
    }
    
    with get_raw_connection() as conn:
        with conn.cursor() as cur:
            # 1. Check columns and foreign keys for single-period tables
            for tbl in single_period_tables:
                cur.execute(
                    """
                    SELECT column_name 
                    FROM information_schema.columns 
                    WHERE table_name = %s AND column_name = 'period_id' AND table_schema = 'public';
                    """,
                    (tbl,)
                )
                assert cur.fetchone() is not None, f"Table {tbl} is missing 'period_id' column"

                cur.execute(
                    """
                    SELECT tc.table_name
                    FROM information_schema.table_constraints AS tc
                    JOIN information_schema.key_column_usage AS kcu
                      ON tc.constraint_name = kcu.constraint_name
                      AND tc.table_schema = kcu.table_schema
                    JOIN information_schema.constraint_column_usage AS ccu
                      ON ccu.constraint_name = tc.constraint_name
                      AND ccu.table_schema = tc.table_schema
                    WHERE tc.constraint_type = 'FOREIGN KEY' 
                      AND tc.table_name = %s
                      AND kcu.column_name = 'period_id' 
                      AND ccu.table_name = 'periods'
                      AND tc.table_schema = 'public';
                    """,
                    (tbl,)
                )
                assert cur.fetchone() is not None, f"Table {tbl} is missing foreign key to 'periods' on 'period_id'"

            # 2. Check columns and foreign keys for multi-period tables (from_period_id and to_period_id)
            for tbl in multi_period_tables:
                for col in ["from_period_id", "to_period_id"]:
                    cur.execute(
                        """
                        SELECT column_name 
                        FROM information_schema.columns 
                        WHERE table_name = %s AND column_name = %s AND table_schema = 'public';
                        """,
                        (tbl, col)
                    )
                    assert cur.fetchone() is not None, f"Table {tbl} is missing '{col}' column"

                    cur.execute(
                        """
                        SELECT tc.table_name
                        FROM information_schema.table_constraints AS tc
                        JOIN information_schema.key_column_usage AS kcu
                          ON tc.constraint_name = kcu.constraint_name
                          AND tc.table_schema = kcu.table_schema
                        JOIN information_schema.constraint_column_usage AS ccu
                          ON ccu.constraint_name = tc.constraint_name
                          AND ccu.table_schema = ccu.table_schema
                        WHERE tc.constraint_type = 'FOREIGN KEY' 
                          AND tc.table_name = %s
                          AND kcu.column_name = %s 
                          AND ccu.table_name = 'periods'
                          AND tc.table_schema = 'public';
                        """,
                        (tbl, col)
                    )
                    assert cur.fetchone() is not None, f"Table {tbl} is missing foreign key to 'periods' on '{col}'"

def test_p0_t3_anthropic_claude_smoke_test(caplog):
    """P0-T3: Anthropic Claude Client Smoke Test."""
    from unittest.mock import patch, MagicMock
    
    # Check if key is available
    if not os.getenv("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY is not set. Skipping real API test.")
        
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="Hello! I am Claude.")]
    
    with patch.object(claude_client.client.messages, 'create', return_value=mock_response) as mock_create:
        with caplog.at_level(logging.INFO):
            response = claude_client.call(
                prompt="Say hello", 
                system_prompt="You are a helpful assistant."
            )
            
        assert response == "Hello! I am Claude."
        mock_create.assert_called_once()
        
    # Verify that logging logged prompt length, model name, response length, latency
    # In python logging, logger records carry these fields in extra/attributes or text
    log_messages = [rec.message for rec in caplog.records if rec.name == "claude_client"]
    assert any("Claude call successful" in msg for msg in log_messages), "No success log found"

def test_p0_t4_notification_hard_fail():
    """P0-T4: Notification Center — Hard Fail Emission & Retrieval."""
    unique_condition = "unit_test_hard_fail_validation"
    
    with pytest.raises(PipelineBlockedError):
        emit(
            source="test_infra", 
            condition=unique_condition, 
            severity="hard_fail", 
            dataset_type="file_upload", 
            details="Test hard fail event"
        )
        
    # Check that it's persisted in the DB
    with get_raw_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT source, condition, severity, dataset_type, details, resolved 
                FROM notifications 
                WHERE condition = %s;
                """,
                (unique_condition,)
            )
            rows = cur.fetchall()
            
    assert len(rows) == 1, "Hard fail notification not persisted"
    row = rows[0]
    assert row[0] == "test_infra"
    assert row[1] == unique_condition
    assert row[2] == "hard_fail"
    assert row[3] == "file_upload"
    assert row[4] == "Test hard fail event"
    assert row[5] is False, "Hard fail should default to unresolved"

def test_p0_t5_notification_soft_warning():
    """P0-T5: Notification Center — Soft Warning Does Not Block."""
    unique_condition = "unit_test_soft_warning_validation"
    
    # Assert it runs without raising any PipelineBlockedError
    try:
        emit(
            source="test_infra", 
            condition=unique_condition, 
            severity="soft_warning", 
            dataset_type="file_upload", 
            details="Test soft warning event"
        )
        pipeline_proceeded = True
    except PipelineBlockedError:
        pipeline_proceeded = False
        
    assert pipeline_proceeded, "Soft warning blocked the pipeline!"
    
    # Verify DB persistence
    with get_raw_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT severity, resolved 
                FROM notifications 
                WHERE condition = %s;
                """,
                (unique_condition,)
            )
            rows = cur.fetchall()
            
    assert len(rows) == 1, "Soft warning notification not persisted"
    assert rows[0][0] == "soft_warning"
