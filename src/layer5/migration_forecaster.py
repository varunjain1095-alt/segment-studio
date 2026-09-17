import logging
from src.db.connection import get_raw_connection

logger = logging.getLogger("layer5_migration_forecaster")

def compute_and_persist_transition_matrix(dataset_id: str, T: int) -> list[dict]:
    """
    Aggregates customer-level segment transitions from period T to T+1
    into a population-level transition probability matrix, and persists the
    cells to the transition_matrix table.
    
    Implements Option B: If a cluster has zero customers assigned at period T,
    it is completely skipped and gets no rows in the transition matrix database/output.
    """
    logger.info(f"Computing segment migration transition matrix from period {T} to {T+1}...")
    
    # 1. Retrieve all active/assigned cluster IDs at T and T+1
    with get_raw_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT DISTINCT cluster_id FROM cluster_assignments WHERE period_id = %s;", (T,))
            from_clusters = [row[0] for row in cur.fetchall()]
            
            cur.execute("SELECT DISTINCT cluster_id FROM cluster_assignments WHERE period_id = %s;", (T + 1,))
            to_clusters = [row[0] for row in cur.fetchall()]
            
    if not from_clusters:
        logger.warning(f"No source cluster assignments found at period {T}. Matrix is empty.")
        return []
        
    results = []
    
    # 2. Compute probabilities for each source cluster
    for from_cid in from_clusters:
        # Count customers in from_cid at T who are also present/assigned at T+1 (active/surviving population)
        with get_raw_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT COUNT(*) 
                    FROM cluster_assignments ca1
                    JOIN cluster_assignments ca2 ON ca1.customer_id = ca2.customer_id
                    WHERE ca1.period_id = %s AND ca2.period_id = %s AND ca1.cluster_id = %s;
                    """,
                    (T, T + 1, from_cid)
                )
                n_from = cur.fetchone()[0]
                
        # Option B: If a cluster has zero customers at T (or none survived to T+1), skip it entirely
        if n_from == 0:
            logger.info(f"Source cluster {from_cid} has zero surviving customers at T. Skipping row entirely (Option B).")
            continue
            
        # Get transition counts to each destination cluster
        with get_raw_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT ca2.cluster_id, COUNT(*)
                    FROM cluster_assignments ca1
                    JOIN cluster_assignments ca2 ON ca1.customer_id = ca2.customer_id
                    WHERE ca1.period_id = %s AND ca2.period_id = %s AND ca1.cluster_id = %s
                    GROUP BY ca2.cluster_id;
                    """,
                    (T, T + 1, from_cid)
                )
                counts_rows = cur.fetchall()
                
        to_counts = {row[0]: row[1] for row in counts_rows}
        
        # Calculate transition probabilities for all destination clusters
        for to_cid in to_clusters:
            cnt = to_counts.get(to_cid, 0)
            prob = float(cnt) / n_from
            
            results.append({
                "from_period_id": T,
                "to_period_id": T + 1,
                "from_cluster_id": from_cid,
                "to_cluster_id": to_cid,
                "customer_count": cnt,
                "transition_prob": prob
            })
            
    # 3. Persist transition matrix to transition_matrix database table
    with get_raw_connection() as conn:
        with conn.cursor() as cur:
            for cell in results:
                cur.execute(
                    """
                    INSERT INTO transition_matrix (
                        from_period_id, to_period_id, from_cluster_id, to_cluster_id,
                        customer_count, transition_prob
                    )
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (from_period_id, to_period_id, from_cluster_id, to_cluster_id)
                    DO UPDATE SET
                        customer_count = EXCLUDED.customer_count,
                        transition_prob = EXCLUDED.transition_prob;
                    """,
                    (
                        cell["from_period_id"],
                        cell["to_period_id"],
                        cell["from_cluster_id"],
                        cell["to_cluster_id"],
                        cell["customer_count"],
                        cell["transition_prob"]
                    )
                )
                
    logger.info(f"Transition matrix calculated: persisted {len(results)} cells for period pair {T} -> {T+1}.")
    return results

def compute_mobility_metrics(from_period_id: int, to_period_id: int) -> dict:
    """
    Computes overall cohort mobility rates directly from persisted transition matrix:
    - Retention Stability: percentage of customers staying in the same cluster.
    - Upward Mobility: percentage of customers migrating to higher clusters/revenue.
    - Downward Churn Drift: percentage of customers migrating to lower clusters/at-risk.
    """
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

    if not rows:
        return {
            "retention_stability_pct": 0.0,
            "upward_mobility_pct": 0.0,
            "downward_drift_pct": 0.0,
            "total_transitioned_customers": 0
        }

    total_customers = sum(r[2] for r in rows)
    if total_customers == 0:
        return {
            "retention_stability_pct": 0.0,
            "upward_mobility_pct": 0.0,
            "downward_drift_pct": 0.0,
            "total_transitioned_customers": 0
        }

    stable_customers = sum(r[2] for r in rows if r[0] == r[1])
    # Upward mobility: migrating to lower-numbered / higher-tier cluster
    upward_customers = sum(r[2] for r in rows if r[1] < r[0])
    # Downward drift: migrating to higher-numbered / slipping cluster
    downward_customers = sum(r[2] for r in rows if r[1] > r[0])

    retention_pct = round((stable_customers / total_customers) * 100, 1)
    upward_pct = round((upward_customers / total_customers) * 100, 1)
    downward_pct = round((downward_customers / total_customers) * 100, 1)

    return {
        "retention_stability_pct": retention_pct,
        "upward_mobility_pct": upward_pct,
        "downward_drift_pct": downward_pct,
        "total_transitioned_customers": total_customers,
        "stable_count": stable_customers,
        "upward_count": upward_customers,
        "downward_count": downward_customers
    }

def compute_horizon_matrix(from_period_id: int, to_period_id: int, days: int) -> list[dict]:
    """
    Computes a projected transition probability matrix for a custom horizon (30, 60, 90, 180 days)
    using Markov matrix exponentiation derived from the base 1-quarter (~90 days) transition matrix.
    """
    import numpy as np
    from scipy.linalg import fractional_matrix_power

    with get_raw_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT from_cluster_id, to_cluster_id, transition_prob, customer_count
                FROM transition_matrix
                WHERE from_period_id = %s AND to_period_id = %s
                ORDER BY from_cluster_id, to_cluster_id;
                """,
                (from_period_id, to_period_id)
            )
            rows = cur.fetchall()

    if not rows:
        return []

    clusters = sorted(list(set([r[0] for r in rows] + [r[1] for r in rows])))
    n = len(clusters)
    cluster_to_idx = {cid: idx for idx, cid in enumerate(clusters)}

    # Build square probability matrix
    P = np.zeros((n, n), dtype=float)
    counts = np.zeros((n, n), dtype=int)
    for fc, tc, prob, cnt in rows:
        if fc in cluster_to_idx and tc in cluster_to_idx:
            P[cluster_to_idx[fc], cluster_to_idx[tc]] = float(prob)
            counts[cluster_to_idx[fc], cluster_to_idx[tc]] = int(cnt)

    # Normalize rows to ensure stochastic matrix
    row_sums = P.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    P = P / row_sums

    # Compute fractional or integer power: base is 90 days
    t = float(days) / 90.0
    if abs(t - 1.0) < 1e-6:
        P_horizon = P
    elif abs(t - 2.0) < 1e-6:
        P_horizon = np.linalg.matrix_power(P, 2)
    else:
        try:
            P_frac = fractional_matrix_power(P, t)
            # Take real part and clean numerical artifacts
            P_horizon = np.real(P_frac)
            P_horizon = np.clip(P_horizon, 0.0, 1.0)
            h_sums = P_horizon.sum(axis=1, keepdims=True)
            h_sums[h_sums == 0] = 1.0
            P_horizon = P_horizon / h_sums
        except Exception as e:
            logger.warning(f"Fractional matrix power failed: {e}. Using linear interpolation fallback.")
            I = np.eye(n)
            P_horizon = np.clip(I + t * (P - I), 0.0, 1.0)
            h_sums = P_horizon.sum(axis=1, keepdims=True)
            h_sums[h_sums == 0] = 1.0
            P_horizon = P_horizon / h_sums

    # Format output rows
    horizon_results = []
    for i, fc in enumerate(clusters):
        for j, tc in enumerate(clusters):
            horizon_results.append({
                "from_cluster_id": fc,
                "to_cluster_id": tc,
                "transition_prob": round(float(P_horizon[i, j]), 4),
                "horizon_days": days
            })

    return horizon_results
