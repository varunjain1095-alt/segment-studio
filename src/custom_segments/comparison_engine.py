import logging
import pandas as pd
import numpy as np
import os
import json
from src.db.connection import get_raw_connection
from src.ingestion.router import DATA_DIR

logger = logging.getLogger("comparison_engine")

def get_customer_features(customer_ids: list[str], period_id: int, columns: list[str]) -> pd.DataFrame:
    """
    Helper to fetch behavioral and demographic feature values for a list of customers in a given period.
    """
    if not customer_ids:
        return pd.DataFrame(columns=["customer_id"] + columns)
        
    with get_raw_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT column_name FROM information_schema.columns WHERE table_name = 'behavioral_snapshots';")
            behavioral_cols = [r[0] for r in cur.fetchall()]
            
            cur.execute("SELECT column_name FROM information_schema.columns WHERE table_name = 'demographic_snapshots';")
            demo_cols = [r[0] for r in cur.fetchall()]
            
    b_select = [c for c in columns if c in behavioral_cols]
    d_select = [c for c in columns if c in demo_cols and c not in ["demo_snapshot_id", "customer_id", "period_id", "captured_at"]]
    
    select_parts = ["c.customer_id::text"]
    for c in b_select:
        select_parts.append(f"b.{c}")
    for c in d_select:
        select_parts.append(f"d.{c}")
        
    with get_raw_connection() as conn:
        with conn.cursor() as cur:
            # We can't pass an empty tuple or single element tuple formatting without care
            query = f"""
                SELECT {", ".join(select_parts)}
                FROM customers c
                LEFT JOIN behavioral_snapshots b ON c.customer_id = b.customer_id AND b.period_id = %s
                LEFT JOIN demographic_snapshots d ON c.customer_id = d.customer_id AND d.period_id = %s
                WHERE c.customer_id = ANY(%s)
            """
            cur.execute(query, (period_id, period_id, customer_ids))
            rows = cur.fetchall()
            col_names = [desc[0] for desc in cur.description]
            
    df = pd.DataFrame(rows, columns=col_names)
    # Add any columns that might be missing (filled with NaN)
    for col in columns:
        if col not in df.columns:
            df[col] = np.nan
    return df

def aggregate_df(df: pd.DataFrame, columns: list[str]) -> dict:
    """Computes mean for numeric columns and mode for categorical columns."""
    res = {}
    for col in columns:
        if col not in df.columns or df[col].isnull().all():
            res[col] = None
            continue
        # Check if numeric
        if pd.api.types.is_numeric_dtype(df[col]):
            res[col] = float(df[col].mean())
        else:
            # Categorical: get mode
            mode_val = df[col].mode()
            res[col] = str(mode_val.iloc[0]) if not mode_val.empty else None
    return res

# --- Option 1 ---
def option1_vs_baseline(segment_customers: list[str], dataset_id: str, period_id: int, columns: list[str]) -> dict:
    """Compare custom segment vs K-means clusters on selected columns."""
    custom_df = get_customer_features(segment_customers, period_id, columns)
    custom_stats = aggregate_df(custom_df, columns)
    
    # Get baseline assignments
    with get_raw_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT ca.customer_id::text, ca.cluster_id, c.cluster_label
                FROM cluster_assignments ca
                JOIN clusters c ON ca.cluster_id = c.cluster_id
                WHERE ca.period_id = %s;
                """,
                (period_id,)
            )
            rows = cur.fetchall()
            
    if not rows:
        return {"custom_segment": custom_stats, "baseline_clusters": {}}
        
    assign_df = pd.DataFrame(rows, columns=["customer_id", "cluster_id", "cluster_label"])
    # Fetch features for all baseline customers
    all_baseline_custs = assign_df["customer_id"].tolist()
    features_df = get_customer_features(all_baseline_custs, period_id, columns)
    
    merged = pd.merge(assign_df, features_df, on="customer_id", how="inner")
    baseline_stats = {}
    
    for cluster_id, group in merged.groupby("cluster_id"):
        label = group["cluster_label"].iloc[0]
        baseline_stats[label] = aggregate_df(group, columns)
        
    return {
        "custom_segment": custom_stats,
        "baseline_clusters": baseline_stats
    }

# --- Option 2 ---
def option2_aggregate_change(segment_customers: list[str], dataset_id: str, columns: list[str]) -> dict:
    """T vs T+1 change for selected columns."""
    with get_raw_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT period_id, period_label FROM periods ORDER BY period_id DESC LIMIT 2;")
            rows = cur.fetchall()
            
    if len(rows) < 2:
        return {"error": "Insufficient periods in database (minimum 2 periods required)."}
        
    # Order chronologically
    rows = sorted(rows, key=lambda x: x[0])
    p_t, p_t_label = rows[0]
    p_t1, p_t1_label = rows[1]
    
    df_t = get_customer_features(segment_customers, p_t, columns)
    df_t1 = get_customer_features(segment_customers, p_t1, columns)
    
    stats_t = aggregate_df(df_t, columns)
    stats_t1 = aggregate_df(df_t1, columns)
    
    changes = {}
    for col in columns:
        val_t = stats_t.get(col)
        val_t1 = stats_t1.get(col)
        if isinstance(val_t, (int, float)) and isinstance(val_t1, (int, float)):
            changes[col] = val_t1 - val_t
        else:
            changes[col] = "N/A"
            
    return {
        "period_T": {"label": p_t_label, "stats": stats_t},
        "period_T1": {"label": p_t1_label, "stats": stats_t1},
        "changes": changes
    }

# --- Option 3 ---
def option3_compare_segments(segments: list[dict], columns: list[str]) -> dict:
    """Side-by-side comparison of multiple custom segments."""
    comparison = {}
    for seg in segments:
        name = seg.get("name")
        custs = seg.get("customers") or []
        period_id = seg.get("period_id")
        
        df = get_customer_features(custs, period_id, columns)
        comparison[name] = aggregate_df(df, columns)
        
    return comparison

# --- Option 4 ---
def option4_statistical_testing(segments: list[dict], period_id: int) -> list[dict]:
    """Runs ANOVA/Kruskal-Wallis significance tests between custom segments."""
    from src.layer3.statistical_tester import get_customer_level_outcomes, compute_anova, compute_kruskal_wallis
    
    if len(segments) < 2:
        return [{"error": "At least 2 custom segments are required for statistical testing."}]
        
    # Get dataset_id from first segment
    dataset_id = segments[0].get("dataset_id")
    if not dataset_id:
        # Fallback query
        with get_raw_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT dataset_id FROM datasets LIMIT 1;")
                row = cur.fetchone()
                dataset_id = str(row[0]) if row else None
                
    if not dataset_id:
        return [{"error": "No dataset found."}]
        
    outcomes_df = get_customer_level_outcomes(dataset_id, period_id)
    if outcomes_df.empty:
        return []
        
    outcomes_df["customer_id"] = outcomes_df["customer_id"].astype(str)
    
    results = []
    outcome_vars = ["churn_rate", "retention_rate", "revenue", "repeat_purchase_prob"]
    
    # We run statistical tests on each outcome variable across the segment groups
    for var in outcome_vars:
        if var not in outcomes_df.columns:
            continue
            
        groups_data = []
        for seg in segments:
            custs = [str(c) for c in (seg.get("customers") or [])]
            seg_data = outcomes_df[outcomes_df["customer_id"].isin(custs)][var].dropna().to_numpy()
            if len(seg_data) > 0:
                groups_data.append(seg_data)
                
        if len(groups_data) >= 2:
            # We run Kruskal-Wallis as default non-parametric test
            stat, p_val, _ = compute_kruskal_wallis(groups_data)
            results.append({
                "test_variable": var,
                "test_type": "Kruskal-Wallis",
                "test_statistic": float(stat),
                "p_value": float(p_val),
                "significant": bool(p_val < 0.05)
            })
            
    return results

# --- Option 5 ---
def option5_outcome_analysis(segment_customers: list[str], period_id: int) -> dict:
    """Computes Layer 6 outcomes for the custom segment."""
    from src.layer3.statistical_tester import get_customer_level_outcomes
    
    with get_raw_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT dataset_id FROM datasets LIMIT 1;")
            row = cur.fetchone()
            dataset_id = str(row[0]) if row else None
            
    if not dataset_id:
        return {"error": "No dataset found."}
        
    outcomes_df = get_customer_level_outcomes(dataset_id, period_id)
    if outcomes_df.empty:
        return {"churn_rate": None, "retention_rate": None, "revenue": None, "repeat_purchase_prob": None}
        
    outcomes_df["customer_id"] = outcomes_df["customer_id"].astype(str)
    
    # Filter to segment customers
    seg_custs = [str(c) for c in segment_customers]
    filtered = outcomes_df[outcomes_df["customer_id"].isin(seg_custs)]
    
    if filtered.empty:
        return {"churn_rate": None, "retention_rate": None, "revenue": None, "repeat_purchase_prob": None}
        
    return {
        "churn_rate": float(filtered["churn_rate"].mean() * 100.0) if "churn_rate" in filtered.columns else None,
        "retention_rate": float(filtered["retention_rate"].mean() * 100.0) if "retention_rate" in filtered.columns else None,
        "revenue": float(filtered["revenue"].sum()) if "revenue" in filtered.columns else None,
        "repeat_purchase_prob": float(filtered["repeat_purchase_prob"].mean()) if "repeat_purchase_prob" in filtered.columns else None
    }

# --- Option 6 ---
def option6_metric_integration(segments: list[dict], metric: str, granularity: str, period_range: str) -> dict:
    """L7 comparison table for custom segments."""
    if not segments:
        raise ValueError("At least one segment is required")
        
    valid_metrics = {"churn_rate", "retention_rate", "revenue", "repeat_purchase_prob"}
    if metric not in valid_metrics:
        raise ValueError(f"Invalid metric requested: '{metric}'")
        
    if not period_range or "," not in period_range:
        raise ValueError("period_range must be in 'start_label,end_label' format")
    start_label, end_label = [label.strip() for label in period_range.split(",", 1)]
    
    with get_raw_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT period_id, period_label 
                FROM periods 
                WHERE granularity = %s 
                ORDER BY period_start ASC, period_id ASC;
                """,
                (granularity,)
            )
            periods_list = cur.fetchall()
            
    if not periods_list:
        raise ValueError(f"No periods found for granularity '{granularity}'")
        
    period_labels = [row[1] for row in periods_list]
    period_ids = [row[0] for row in periods_list]
    
    if start_label not in period_labels or end_label not in period_labels:
        raise ValueError("Invalid period labels in period_range")
        
    start_idx = period_labels.index(start_label)
    end_idx = period_labels.index(end_label)
    
    selected_period_ids = period_ids[start_idx : end_idx + 1]
    selected_period_labels = period_labels[start_idx : end_idx + 1]
    
    rows_data = []
    for seg in segments:
        custs = seg.get("customers") or []
        values = []
        for pid in selected_period_ids:
            outcomes = option5_outcome_analysis(custs, pid)
            values.append(outcomes.get(metric))
        rows_data.append({
            "segment_name": seg.get("name"),
            "values": values
        })
        
    # Gap row if exactly 2 custom segments are compared
    gap_row = None
    if len(segments) == 2:
        val_A = rows_data[0]["values"]
        val_B = rows_data[1]["values"]
        gap_values = []
        for v_A, v_B in zip(val_A, val_B):
            if v_A is not None and v_B is not None:
                diff = round(v_A - v_B, 4)
                gap_values.append(f"+{diff}" if diff > 0 else f"{diff}")
            else:
                gap_values.append(None)
        gap_row = gap_values
        
    return {
        "periods": selected_period_labels,
        "data": rows_data,
        "gap_row": gap_row
    }

# --- Option 7 ---
def option7_feature_signal(segment_customers: list[str], period_id: int) -> dict:
    """L8 drift signals for custom segment."""
    if not segment_customers:
        return {"total_checked": 0, "signals_raised": 0, "avg_drift_fraction": 0.0}
        
    with get_raw_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT drift_fraction, signal_raised
                FROM feature_migration_signals
                WHERE period_id = %s AND customer_id = ANY(%s);
                """,
                (period_id, segment_customers)
            )
            rows = cur.fetchall()
            
    if not rows:
        return {"total_checked": 0, "signals_raised": 0, "avg_drift_fraction": 0.0}
        
    return {
        "total_checked": len(rows),
        "signals_raised": sum(1 for r in rows if r[1]),
        "avg_drift_fraction": float(sum(float(r[0]) for r in rows) / len(rows))
    }
