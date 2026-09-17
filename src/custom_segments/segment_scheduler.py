import json
import logging
from datetime import datetime, timedelta
from src.db.connection import get_raw_connection
from src.custom_segments.segment_engine import build_segment
from src.notifications.center import emit

logger = logging.getLogger("segment_scheduler")

def save_segment(dataset_id: str, segment_name: str, conditions: list) -> str:
    """Saves a custom segment definition to the database."""
    with get_raw_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO custom_segments (dataset_id, segment_name, conditions)
                VALUES (%s, %s, %s)
                RETURNING segment_id;
                """,
                (dataset_id, segment_name, json.dumps(conditions))
            )
            segment_id = cur.fetchone()[0]
    return str(segment_id)

def schedule_segment(segment_id: str, granularity: str, selected_options: list) -> str:
    """Schedules a saved custom segment."""
    now = datetime.utcnow()
    next_run_at = now + timedelta(days=30) if granularity == "monthly" else now + timedelta(days=90)
    
    with get_raw_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO custom_segment_schedules (segment_id, granularity, selected_options, next_run_at)
                VALUES (%s, %s, %s, %s)
                RETURNING schedule_id;
                """,
                (segment_id, granularity, json.dumps(selected_options), next_run_at)
            )
            schedule_id = cur.fetchone()[0]
    return str(schedule_id)

def run_scheduled_segments():
    """Runs all due custom segment schedules."""
    from src.custom_segments.comparison_engine import (
        option1_vs_baseline, option2_aggregate_change, option3_compare_segments,
        option4_statistical_testing, option5_outcome_analysis, option6_metric_integration,
        option7_feature_signal
    )
    
    now = datetime.utcnow()
    
    # Query due schedules
    with get_raw_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT css.schedule_id, css.segment_id, css.granularity, css.selected_options,
                       cs.dataset_id, cs.segment_name, cs.conditions
                FROM custom_segment_schedules css
                JOIN custom_segments cs ON css.segment_id = cs.segment_id
                WHERE css.next_run_at IS NULL OR css.next_run_at <= %s;
                """,
                (now,)
            )
            due_schedules = cur.fetchall()
            
    for row in due_schedules:
        schedule_id, segment_id, granularity, selected_options, dataset_id, segment_name, conditions = row
        dataset_id = str(dataset_id)
        segment_id = str(segment_id)
        
        # 1. Re-evaluate customer list
        try:
            options_list = json.loads(selected_options) if isinstance(selected_options, str) else selected_options
            conds_list = json.loads(conditions) if isinstance(conditions, str) else conditions
            
            # Fetch matching customers
            matching_customers = build_segment(dataset_id, conds_list)
            
            # Fetch latest period_id in DB
            with get_raw_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT MAX(period_id) FROM periods;")
                    max_period_row = cur.fetchone()
                    period_id = max_period_row[0] if max_period_row else 1
            
            # 2. Run each selected option
            for opt in options_list:
                result_data = {}
                try:
                    if opt == "option1":
                        # We mock columns list as empty or select all
                        result_data = option1_vs_baseline(matching_customers, dataset_id, period_id, ["login_frequency"])
                    elif opt == "option2":
                        result_data = option2_aggregate_change(matching_customers, dataset_id, ["login_frequency"])
                    elif opt == "option3":
                        result_data = option3_compare_segments([{"name": segment_name, "customers": matching_customers, "period_id": period_id}], ["login_frequency"])
                    elif opt == "option4":
                        result_data = option4_statistical_testing([{"name": segment_name, "customers": matching_customers, "dataset_id": dataset_id}], period_id)
                    elif opt == "option5":
                        result_data = option5_outcome_analysis(matching_customers, period_id)
                    elif opt == "option6":
                        result_data = option6_metric_integration([{"name": segment_name, "customers": matching_customers}], "churn_rate", "quarterly", "P10,P10")
                    elif opt == "option7":
                        result_data = option7_feature_signal(matching_customers, period_id)
                except Exception as ex:
                    logger.error(f"Error executing custom segment comparison option {opt}: {ex}")
                    result_data = {"error": str(ex)}
                    
                # Store result
                with get_raw_connection() as conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            """
                            INSERT INTO custom_segment_results (segment_id, option, result_data)
                            VALUES (%s, %s, %s);
                            """,
                            (segment_id, opt, json.dumps(result_data))
                        )
                        
            # 3. Update next_run_at and last_run_at
            next_run = now + timedelta(days=30) if granularity == "monthly" else now + timedelta(days=90)
            with get_raw_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        UPDATE custom_segment_schedules
                        SET last_run_at = %s, next_run_at = %s
                        WHERE schedule_id = %s;
                        """,
                        (now, next_run, schedule_id)
                    )
                    
            # 4. Emit soft notification to notification center
            emit(
                source="custom_segment_scheduler",
                condition="scheduled_run_completed",
                severity="info",
                details=f"Custom segment scheduled run completed for segment '{segment_name}'."
            )
            
        except Exception as e:
            logger.error(f"Failed scheduled run for segment '{segment_name}': {e}", exc_info=True)
