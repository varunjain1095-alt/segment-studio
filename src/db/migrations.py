import logging
from src.db.connection import get_raw_connection
from src.utils.logging_config import setup_logging

setup_logging()
logger = logging.getLogger("migrations")

MIGRATIONS_DDL = [
    # 1. Customers
    """
    CREATE TABLE IF NOT EXISTS customers (
        customer_id     UUID PRIMARY KEY,
        created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
    );
    """,
    # 2. Periods
    """
    CREATE TABLE IF NOT EXISTS periods (
        period_id       SERIAL PRIMARY KEY,
        period_label    TEXT NOT NULL,
        period_start    DATE NOT NULL,
        period_end      DATE NOT NULL,
        granularity     TEXT NOT NULL
    );
    """,
    # 3. Datasets
    """
    CREATE TABLE IF NOT EXISTS datasets (
        dataset_id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        source_type         TEXT NOT NULL,
        permanence          TEXT NOT NULL,
        ingested_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
        connection_string   TEXT,
        file_name           TEXT,
        confirmed_columns   JSONB,
        layer0_completed    BOOLEAN NOT NULL DEFAULT FALSE
    );
    """,
    # 4. Column Profiles
    """
    CREATE TABLE IF NOT EXISTS column_profiles (
        profile_id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        dataset_id          UUID NOT NULL REFERENCES datasets(dataset_id) ON DELETE CASCADE,
        column_name         TEXT NOT NULL,
        missing_count       INTEGER,
        missing_pct         NUMERIC,
        dtype               TEXT,
        suspect_numeric_object BOOLEAN,
        unique_count        INTEGER,
        non_unique_count    INTEGER,
        non_unique_pct      NUMERIC,
        is_unique_key       BOOLEAN,
        has_duplicates      BOOLEAN,
        severity            TEXT,
        imputation_method   TEXT,
        profiled_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
        UNIQUE (dataset_id, column_name)
    );
    """,
    # 5. Cross Field Rules
    """
    CREATE TABLE IF NOT EXISTS cross_field_rules (
        rule_id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        dataset_id      UUID NOT NULL REFERENCES datasets(dataset_id) ON DELETE CASCADE,
        rule_description TEXT NOT NULL,
        rule_expression TEXT NOT NULL,
        proposed_by     TEXT NOT NULL,
        approved        BOOLEAN NOT NULL DEFAULT false,
        created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
        approved_at     TIMESTAMPTZ
    );
    """,
    # 6. Notifications
    """
    CREATE TABLE IF NOT EXISTS notifications (
        notification_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        source          TEXT NOT NULL,
        condition       TEXT NOT NULL,
        severity        TEXT NOT NULL,
        dataset_type    TEXT,
        details         TEXT,
        resolved        BOOLEAN NOT NULL DEFAULT false,
        emitted_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
        resolved_at     TIMESTAMPTZ
    );
    """,
    # 7. Cluster Baselines
    """
    CREATE TABLE IF NOT EXISTS cluster_baselines (
        baseline_id     SERIAL PRIMARY KEY,
        adopted_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
        is_active       BOOLEAN NOT NULL DEFAULT true,
        adopted_reason  TEXT,
        period_id       INTEGER REFERENCES periods(period_id) ON DELETE SET NULL
    );
    """,
    # 8. Clusters
    """
    CREATE TABLE IF NOT EXISTS clusters (
        cluster_id           SERIAL PRIMARY KEY,
        baseline_id          INTEGER NOT NULL REFERENCES cluster_baselines(baseline_id) ON DELETE CASCADE,
        cluster_label        TEXT,
        cluster_description  TEXT,
        medoid_customer_id   UUID REFERENCES customers(customer_id) ON DELETE SET NULL,
        intra_cluster_spread NUMERIC,
        created_at           TIMESTAMPTZ NOT NULL DEFAULT now()
    );
    """,
    # 9. Behavioral Snapshots
    """
    CREATE TABLE IF NOT EXISTS behavioral_snapshots (
        snapshot_id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        customer_id          UUID NOT NULL REFERENCES customers(customer_id) ON DELETE CASCADE,
        period_id            INTEGER NOT NULL REFERENCES periods(period_id) ON DELETE CASCADE,
        login_frequency      NUMERIC,
        session_duration     NUMERIC,
        purchase_count       INTEGER,
        avg_order_value      NUMERIC,
        return_rate          NUMERIC,
        discount_sensitivity NUMERIC,
        email_engagement     NUMERIC,
        app_engagement       NUMERIC,
        captured_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
        UNIQUE (customer_id, period_id)
    );
    """,
    # 10. Demographic Snapshots
    """
    CREATE TABLE IF NOT EXISTS demographic_snapshots (
        demo_snapshot_id    UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        customer_id         UUID NOT NULL REFERENCES customers(customer_id) ON DELETE CASCADE,
        period_id           INTEGER NOT NULL REFERENCES periods(period_id) ON DELETE CASCADE,
        age                 INTEGER,
        gender              TEXT,
        country             TEXT,
        captured_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
        UNIQUE (customer_id, period_id)
    );
    """,
    # 11. Cluster Assignments
    """
    CREATE TABLE IF NOT EXISTS cluster_assignments (
        assignment_id   UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        customer_id     UUID NOT NULL REFERENCES customers(customer_id) ON DELETE CASCADE,
        period_id       INTEGER NOT NULL REFERENCES periods(period_id) ON DELETE CASCADE,
        cluster_id      INTEGER NOT NULL REFERENCES clusters(cluster_id) ON DELETE CASCADE,
        assigned_via    TEXT NOT NULL,
        assigned_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
        UNIQUE (customer_id, period_id)
    );
    """,
    # 12. Cluster Identity Map
    """
    CREATE TABLE IF NOT EXISTS cluster_identity_map (
        map_id          SERIAL PRIMARY KEY,
        old_cluster_id  INTEGER REFERENCES clusters(cluster_id) ON DELETE CASCADE,
        new_cluster_id  INTEGER REFERENCES clusters(cluster_id) ON DELETE CASCADE,
        match_type      TEXT NOT NULL,
        created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
        CONSTRAINT chk_match_type_nullability CHECK (
            (match_type IN ('matched', 'split', 'merged') AND old_cluster_id IS NOT NULL AND new_cluster_id IS NOT NULL)
            OR (match_type = 'disappeared' AND old_cluster_id IS NOT NULL AND new_cluster_id IS NULL)
            OR (match_type = 'new' AND old_cluster_id IS NULL AND new_cluster_id IS NOT NULL)
        )
    );
    """,
    # 13. Migration Labels
    """
    CREATE TABLE IF NOT EXISTS migration_labels (
        label_id        UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        customer_id     UUID NOT NULL REFERENCES customers(customer_id) ON DELETE CASCADE,
        from_period_id  INTEGER NOT NULL REFERENCES periods(period_id) ON DELETE CASCADE,
        to_period_id    INTEGER NOT NULL REFERENCES periods(period_id) ON DELETE CASCADE,
        from_cluster_id INTEGER REFERENCES clusters(cluster_id) ON DELETE SET NULL,
        to_cluster_id   INTEGER REFERENCES clusters(cluster_id) ON DELETE SET NULL,
        migrated        BOOLEAN NOT NULL,
        labeled_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
        UNIQUE (customer_id, from_period_id, to_period_id)
    );
    """,
    # 14. Transition Matrix
    """
    CREATE TABLE IF NOT EXISTS transition_matrix (
        matrix_id           SERIAL PRIMARY KEY,
        from_period_id      INTEGER NOT NULL REFERENCES periods(period_id) ON DELETE CASCADE,
        to_period_id        INTEGER NOT NULL REFERENCES periods(period_id) ON DELETE CASCADE,
        from_cluster_id     INTEGER NOT NULL REFERENCES clusters(cluster_id) ON DELETE CASCADE,
        to_cluster_id       INTEGER NOT NULL REFERENCES clusters(cluster_id) ON DELETE CASCADE,
        customer_count      INTEGER NOT NULL,
        transition_prob     NUMERIC NOT NULL,
        computed_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
        UNIQUE (from_period_id, to_period_id, from_cluster_id, to_cluster_id)
    );
    """,
    # 15. Segment Outcomes
    """
    CREATE TABLE IF NOT EXISTS segment_outcomes (
        outcome_id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        cluster_id           INTEGER NOT NULL REFERENCES clusters(cluster_id) ON DELETE CASCADE,
        period_id            INTEGER NOT NULL REFERENCES periods(period_id) ON DELETE CASCADE,
        churn_rate           NUMERIC,
        retention_rate       NUMERIC,
        revenue              NUMERIC,
        repeat_purchase_prob NUMERIC,
        computed_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
        UNIQUE (cluster_id, period_id)
    );
    """,
    # 16. Feature Migration Signals
    """
    CREATE TABLE IF NOT EXISTS feature_migration_signals (
        signal_id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        customer_id         UUID NOT NULL REFERENCES customers(customer_id) ON DELETE CASCADE,
        period_id           INTEGER NOT NULL REFERENCES periods(period_id) ON DELETE CASCADE,
        own_cluster_id      INTEGER REFERENCES clusters(cluster_id) ON DELETE SET NULL,
        compared_cluster_id INTEGER REFERENCES clusters(cluster_id) ON DELETE SET NULL,
        columns_checked     INTEGER NOT NULL,
        columns_drifted     INTEGER NOT NULL,
        drift_fraction      NUMERIC NOT NULL,
        signal_raised       BOOLEAN NOT NULL,
        evaluated_at        TIMESTAMPTZ NOT NULL DEFAULT now()
    );
    """,
    # 17. Statistical Test Results
    """
    CREATE TABLE IF NOT EXISTS statistical_test_results (
        result_id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        period_id           INTEGER NOT NULL REFERENCES periods(period_id) ON DELETE CASCADE,
        test_type           TEXT NOT NULL,
        segment_a_id        INTEGER REFERENCES clusters(cluster_id) ON DELETE CASCADE,
        segment_b_id        INTEGER REFERENCES clusters(cluster_id) ON DELETE CASCADE,
        test_variable       TEXT NOT NULL,
        p_value             NUMERIC,
        test_statistic      NUMERIC,
        effect_size         NUMERIC,
        effect_size_type    TEXT,
        significant         BOOLEAN,
        flag_type           TEXT,
        computed_at         TIMESTAMPTZ NOT NULL DEFAULT now()
    );
    """,
    # 18. Quarterly Narrations
    """
    CREATE TABLE IF NOT EXISTS quarterly_narrations (
        narration_id    UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        period_id       INTEGER NOT NULL REFERENCES periods(period_id) ON DELETE CASCADE,
        narration_text  TEXT NOT NULL,
        model_used      TEXT NOT NULL,
        generated_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
        passed_guardrails BOOLEAN NOT NULL DEFAULT false,
        UNIQUE (period_id)
    );
    """,
    # 19. Layer 8 Config
    """
    CREATE TABLE IF NOT EXISTS layer8_config (
        config_id       UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        dataset_id      UUID NOT NULL REFERENCES datasets(dataset_id) ON DELETE CASCADE,
        confirmed_columns JSONB NOT NULL,
        confirmed_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
        confirmed_by    TEXT,
        UNIQUE (dataset_id)
    );
    """,
    # 20. Demographic Segment Metrics
    """
    CREATE TABLE IF NOT EXISTS demographic_segment_metrics (
        metric_id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        dataset_id          UUID NOT NULL REFERENCES datasets(dataset_id) ON DELETE CASCADE,
        period_id           INTEGER NOT NULL REFERENCES periods(period_id) ON DELETE CASCADE,
        segment_field       TEXT NOT NULL,
        segment_value       TEXT NOT NULL,
        segment_size        INTEGER NOT NULL,
        prev_segment_size   INTEGER,
        mom_pct_change      NUMERIC,
        historical_std_dev  NUMERIC,
        anomaly_flagged     BOOLEAN NOT NULL DEFAULT false,
        computed_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
        UNIQUE (dataset_id, period_id, segment_field, segment_value)
    );
    """,
    # 21. Custom Segment Definitions
    """
    CREATE TABLE IF NOT EXISTS custom_segments (
        segment_id      UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        dataset_id      UUID NOT NULL REFERENCES datasets(dataset_id) ON DELETE CASCADE,
        segment_name    TEXT NOT NULL,
        conditions      JSONB NOT NULL,
        created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
        updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
    );
    """,
    # 22. Custom Segment Schedules
    """
    CREATE TABLE IF NOT EXISTS custom_segment_schedules (
        schedule_id     UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        segment_id      UUID NOT NULL REFERENCES custom_segments(segment_id) ON DELETE CASCADE,
        granularity     TEXT NOT NULL,
        selected_options JSONB NOT NULL,
        last_run_at     TIMESTAMPTZ,
        next_run_at     TIMESTAMPTZ,
        created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
    );
    """,
    # 23. Custom Segment Run Results
    """
    CREATE TABLE IF NOT EXISTS custom_segment_results (
        result_id       UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        segment_id      UUID NOT NULL REFERENCES custom_segments(segment_id) ON DELETE CASCADE,
        run_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
        option          TEXT NOT NULL,
        result_data     JSONB NOT NULL
    );
    """
]

def run_migrations(reset=False):
    """Runs schema migrations, with option to drop all tables first."""
    logger.info("Initializing migrations...")
    
    with get_raw_connection() as conn:
        with conn.cursor() as cur:
            # Ensure pgcrypto or uuid-ossp is enabled for gen_random_uuid() support
            cur.execute('CREATE EXTENSION IF NOT EXISTS "uuid-ossp";')
            
            if reset:
                logger.info("Dropping existing tables for reset...")
                tables_to_drop = [
                    "custom_segment_results", "custom_segment_schedules", "custom_segments",
                    "demographic_segment_metrics", "layer8_config", "quarterly_narrations",
                    "statistical_test_results", "feature_migration_signals", "segment_outcomes",
                    "transition_matrix", "migration_labels", "cluster_identity_map",
                    "cluster_assignments", "demographic_snapshots", "behavioral_snapshots",
                    "clusters", "cluster_baselines", "notifications", "cross_field_rules",
                    "column_profiles", "datasets", "periods", "customers"
                ]
                for tbl in tables_to_drop:
                    cur.execute(f"DROP TABLE IF EXISTS {tbl} CASCADE;")
                logger.info("All existing tables dropped.")
            
            for statement in MIGRATIONS_DDL:
                cur.execute(statement)
                
            # Perform Phase 2 column alters if they don't exist
            cur.execute("ALTER TABLE datasets ADD COLUMN IF NOT EXISTS proposed_classifications JSONB;")
            cur.execute("ALTER TABLE datasets ADD COLUMN IF NOT EXISTS confirmed_classifications JSONB;")
            cur.execute("ALTER TABLE datasets ADD COLUMN IF NOT EXISTS layer0_completed BOOLEAN NOT NULL DEFAULT FALSE;")
            cur.execute("ALTER TABLE clusters ADD COLUMN IF NOT EXISTS cluster_description TEXT;")
            cur.execute("ALTER TABLE segment_outcomes DROP COLUMN IF EXISTS profitability;")
            cur.execute("ALTER TABLE datasets ADD COLUMN IF NOT EXISTS fallback_active BOOLEAN NOT NULL DEFAULT FALSE;")
                
    logger.info("Migrations completed successfully.")

if __name__ == "__main__":
    import sys
    reset_db = "--reset" in sys.argv
    run_migrations(reset=reset_db)
