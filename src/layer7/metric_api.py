import logging
from src.db.connection import get_raw_connection

logger = logging.getLogger("layer7_metric_api")

def resolve_cluster_lineage(conn, segment_ids: list[int]) -> dict[int, set[int]]:
    """
    For each requested segment_id, resolves all equivalent cluster IDs
    across adopted baselines via cluster_identity_map (match_type = 'matched').
    Returns {requested_id: {requested_id, matched_id1, ...}}
    """
    id_map = {sid: {sid} for sid in segment_ids}
    with conn.cursor() as cur:
        try:
            cur.execute("SELECT old_cluster_id, new_cluster_id FROM cluster_identity_map WHERE match_type = 'matched';")
            pairs = cur.fetchall()
        except Exception:
            pairs = []
        
    for sid in segment_ids:
        changed = True
        while changed:
            changed = False
            for old_id, new_id in pairs:
                if old_id is not None and new_id is not None:
                    old_id_int = int(old_id)
                    new_id_int = int(new_id)
                    if old_id_int in id_map[sid] and new_id_int not in id_map[sid]:
                        id_map[sid].add(new_id_int)
                        changed = True
                    elif new_id_int in id_map[sid] and old_id_int not in id_map[sid]:
                        id_map[sid].add(old_id_int)
                        changed = True
    return id_map


def get_metric_comparison(granularity: str, metric: str, segment_ids: list[int], period_range: str) -> dict:
    """
    Retrieves the comparison matrix for segments and periods based on granularity and metric.
    Computes a gap row if exactly 2 segments are provided.
    """
    logger.info(f"Comparing metric {metric} across segments {segment_ids} with range {period_range} ({granularity})...")

    # 1. Validation check (P9-T5)
    if not segment_ids:
        raise ValueError("At least one segment_id is required")

    valid_metrics = {"churn_rate", "retention_rate", "revenue", "repeat_purchase_prob"}
    if metric not in valid_metrics:
        raise ValueError(f"Invalid metric requested: '{metric}'. Valid options are: {list(valid_metrics)}")

    # 2. Parse period_range
    if not period_range or "," not in period_range:
        raise ValueError("period_range must be in 'start_label,end_label' format")
    start_label, end_label = [label.strip() for label in period_range.split(",", 1)]

    # 3. Retrieve periods of the specified granularity, sorted chronologically
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

    if start_label not in period_labels:
        raise ValueError(f"Start period label '{start_label}' not found")
    if end_label not in period_labels:
        raise ValueError(f"End period label '{end_label}' not found")

    start_idx = period_labels.index(start_label)
    end_idx = period_labels.index(end_label)

    if start_idx > end_idx:
        raise ValueError("Start period must be chronological before or equal to end period")

    # Keep only the range
    selected_period_ids = period_ids[start_idx : end_idx + 1]
    selected_period_labels = period_labels[start_idx : end_idx + 1]

    # 4. Fetch outcomes from segment_outcomes, resolving cluster lineage across adopted baselines
    with get_raw_connection() as conn:
        lineage_map = resolve_cluster_lineage(conn, segment_ids)
        all_cids = list({cid for s in lineage_map.values() for cid in s})

        with conn.cursor() as cur:
            query = f"""
                SELECT cluster_id, period_id, "{metric}"
                FROM segment_outcomes
                WHERE cluster_id = ANY(%s) AND period_id = ANY(%s);
            """
            cur.execute(query, (all_cids, selected_period_ids))
            outcome_rows = cur.fetchall()

    # Map (cluster_id -> requested_id)
    reverse_map = {}
    for req_id, equiv_set in lineage_map.items():
        for eq_id in equiv_set:
            reverse_map[eq_id] = req_id

    # Map (requested_segment_id, period_id) -> value
    outcome_map = {}
    for row in outcome_rows:
        import math
        req_id = reverse_map.get(int(row[0]), int(row[0]))
        pid = int(row[1])
        val = row[2]
        if val is not None:
            f_val = float(val)
            if math.isnan(f_val) or math.isinf(f_val):
                outcome_map[(req_id, pid)] = 0.0
            else:
                outcome_map[(req_id, pid)] = f_val
        else:
            if (req_id, pid) not in outcome_map:
                outcome_map[(req_id, pid)] = None

    # 5. Pivot into segment rows
    rows_data = []
    for cid in segment_ids:
        values = []
        for pid in selected_period_ids:
            values.append(outcome_map.get((cid, pid), None))
        rows_data.append({
            "segment_id": cid,
            "values": values
        })

    # 6. Compute Gap Row if count is exactly 2 (P9-T2 & P9-T3 & P9-T4)
    if len(segment_ids) == 2:
        val_A = rows_data[0]["values"]
        val_B = rows_data[1]["values"]
        gap_values = []
        for v_A, v_B in zip(val_A, val_B):
            if v_A is not None and v_B is not None:
                diff = v_A - v_B
                # Include directional sign (+/-), format to 4 decimals to avoid float precision artifacts
                diff_rounded = round(diff, 4)
                if diff_rounded > 0:
                    gap_values.append(f"+{diff_rounded}")
                elif diff_rounded < 0:
                    gap_values.append(f"{diff_rounded}")
                else:
                    gap_values.append("0")
            else:
                gap_values.append(None)
        rows_data.append({
            "segment_id": "gap",
            "values": gap_values
        })

    return {
        "periods": selected_period_labels,
        "data": rows_data
    }
