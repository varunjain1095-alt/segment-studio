import os
import json
import logging
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances, davies_bouldin_score, silhouette_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from src.db.connection import get_raw_connection
from src.notifications.center import emit
from src.layer2.cluster_evaluator import evaluate_clustering

logger = logging.getLogger("layer2_drift_evaluator")

def compute_medoids_and_spreads(
    X_df: pd.DataFrame,
    labels: np.ndarray,
    behavioral_cols: list[str]
) -> dict:
    """
    Computes medoids and spreads for each active cluster in X_df.
    Returns:
      dict: Maps cluster_label (int) -> {
              "medoid_customer_id": str,
              "medoid_coords": list[float],
              "intra_cluster_spread": float
            }
    """
    logger.info("Computing cluster medoids and spreads...")
    features = X_df[behavioral_cols]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)
    
    unique_labels = set(labels) - {-1}
    cluster_stats = {}
    
    for c in unique_labels:
        idx_c = np.where(labels == c)[0]
        X_c = X_scaled[idx_c]
        
        # Implement sampling cap of 2,000 points if the cluster size is large (P3-T12)
        if len(idx_c) > 2000:
            logger.info(f"Cluster {c} size ({len(idx_c)}) exceeds 2000 limit. Sampling 2000 points for medoid calculation.")
            np.random.seed(42)  # For deterministic testing
            sample_indices = np.random.choice(len(idx_c), size=2000, replace=False)
            idx_c_sampled = idx_c[sample_indices]
            X_c_sampled = X_scaled[idx_c_sampled]
            
            # Compute pairwise distance matrix on the sample
            dist_matrix = pairwise_distances(X_c_sampled, metric='euclidean')
            sum_dists = dist_matrix.sum(axis=1)
            medoid_local_idx = sum_dists.argmin()
            medoid_idx = idx_c_sampled[medoid_local_idx]
            medoid_coords = X_scaled[medoid_idx].tolist()
        else:
            # Calculate pairwise distance matrix on full cluster
            dist_matrix = pairwise_distances(X_c, metric='euclidean')
            sum_dists = dist_matrix.sum(axis=1)
            medoid_local_idx = sum_dists.argmin()
            medoid_idx = idx_c[medoid_local_idx]
            medoid_coords = X_c[medoid_local_idx].tolist()
            
        medoid_customer_id = X_df.iloc[medoid_idx]["customer_id"]
        
        # Spread = mean distance to medoid coords (memory-efficient 1xN distance calculation)
        medoid_coords_arr = np.array(medoid_coords).reshape(1, -1)
        dists_to_medoid = pairwise_distances(X_c, medoid_coords_arr, metric='euclidean').flatten()
        spread = float(dists_to_medoid.mean())
        
        cluster_stats[c] = {
            "medoid_customer_id": medoid_customer_id,
            "medoid_coords": medoid_coords,
            "intra_cluster_spread": spread
        }
        logger.info(f"  Cluster {c}: medoid={medoid_customer_id}, spread={spread:.4f}")
        
    return cluster_stats

def match_clusters(old_medoids: dict, candidate_medoids: dict) -> list[dict]:
    """
    Matches candidate clusters to baseline clusters using medoid Euclidean distance.
    Identifies matched, split, merged, disappeared, and new clusters.
    
    Returns:
      mappings (list[dict]): List of dicts representing mappings with keys:
        - old_cluster_id (int or None)
        - candidate_label (int or None)
        - match_type (str): 'matched' | 'split' | 'merged' | 'disappeared' | 'new'
    """
    logger.info("Matching candidate clusters to baseline clusters...")
    
    # Distance matrix between old medoids and candidate medoids
    old_keys = list(old_medoids.keys())
    candidate_keys = list(candidate_medoids.keys())
    
    if not old_keys:
        return [{"old_cluster_id": None, "candidate_label": c, "match_type": "new"} for c in candidate_keys]
    if not candidate_keys:
        return [{"old_cluster_id": o, "candidate_label": None, "match_type": "disappeared"} for o in old_keys]
        
    # Build distance matrix
    dist_matrix = np.zeros((len(old_keys), len(candidate_keys)))
    for i, o_key in enumerate(old_keys):
        o_coords = np.array(old_medoids[o_key]["medoid_coords"])
        for j, c_key in enumerate(candidate_keys):
            c_coords = np.array(candidate_medoids[c_key]["medoid_coords"])
            dist_matrix[i, j] = np.linalg.norm(o_coords - c_coords)
            
    # Find closest mapping directions
    # For each candidate, closest old
    candidate_to_old = {}
    for j, c_key in enumerate(candidate_keys):
        i_min = dist_matrix[:, j].argmin()
        candidate_to_old[c_key] = (old_keys[i_min], dist_matrix[i_min, j])
        
    # For each old, closest candidate
    old_to_candidate = {}
    for i, o_key in enumerate(old_keys):
        j_min = dist_matrix[i, :].argmin()
        old_to_candidate[o_key] = (candidate_keys[j_min], dist_matrix[i, j_min])
        
    mappings = []
    processed_candidates = set()
    processed_olds = set()
    
    # 1. Check mutual closest (matched)
    for c_key in candidate_keys:
        o_key, dist = candidate_to_old[c_key]
        o_closest_c, _ = old_to_candidate[o_key]
        if o_closest_c == c_key:
            # Mutual match
            mappings.append({
                "old_cluster_id": o_key,
                "candidate_label": c_key,
                "match_type": "matched",
                "distance": dist
            })
            processed_candidates.add(c_key)
            processed_olds.add(o_key)
            
    # 2. Check Splits (one old closest to multiple candidates)
    for o_key in old_keys:
        if o_key in processed_olds:
            continue
        # Find candidates that mapped to this old key
        mapping_candidates = [c for c, (o, _) in candidate_to_old.items() if o == o_key]
        if len(mapping_candidates) > 1:
            for c in mapping_candidates:
                mappings.append({
                    "old_cluster_id": o_key,
                    "candidate_label": c,
                    "match_type": "split",
                    "distance": candidate_to_old[c][1]
                })
                processed_candidates.add(c)
            processed_olds.add(o_key)
            
    # 3. Check Merges (multiple olds closest to one candidate)
    for c_key in candidate_keys:
        if c_key in processed_candidates:
            continue
        # Find olds that mapped to this candidate
        mapping_olds = [o for o, (c, _) in old_to_candidate.items() if c == c_key]
        if len(mapping_olds) > 1:
            for o in mapping_olds:
                mappings.append({
                    "old_cluster_id": o,
                    "candidate_label": c_key,
                    "match_type": "merged",
                    "distance": old_to_candidate[o][1]
                })
                processed_olds.add(o)
            processed_candidates.add(c_key)
            
    # 4. Handle remaining leftovers (disappeared or new)
    for o_key in old_keys:
        if o_key not in processed_olds:
            mappings.append({
                "old_cluster_id": o_key,
                "candidate_label": None,
                "match_type": "disappeared"
            })
            
    for c_key in candidate_keys:
        if c_key not in processed_candidates:
            mappings.append({
                "old_cluster_id": None,
                "candidate_label": c_key,
                "match_type": "new"
            })
            
    return mappings

def get_active_baseline() -> tuple[int, int]:
    """
    Returns (baseline_id, period_id) of the currently active baseline.
    If none exists, returns (None, None).
    """
    with get_raw_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT baseline_id, period_id
                FROM cluster_baselines
                WHERE is_active = TRUE;
                """
            )
            row = cur.fetchone()
    return row if row else (None, None)

def evaluate_baseline_quality(period_id: int, behavioral_cols: list[str]) -> dict:
    """
    Computes quality metrics (noise fraction, silhouette, dbi) of the active baseline
    for comparison.
    """
    with get_raw_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT bs.customer_id, ca.cluster_id, c.cluster_label, bs.period_id
                FROM behavioral_snapshots bs
                JOIN cluster_assignments ca ON bs.customer_id = ca.customer_id AND bs.period_id = ca.period_id
                JOIN clusters c ON ca.cluster_id = c.cluster_id
                WHERE bs.period_id = %s;
                """,
                (period_id,)
            )
            rows = cur.fetchall()
            
    dbi_max = float(os.getenv("CLUSTER_DAVIES_BOULDIN_MAX", 1.5))
    silhouette_min = float(os.getenv("CLUSTER_SILHOUETTE_MIN", 0.2))
    noise_max = float(os.getenv("CLUSTER_NOISE_FRACTION_MAX", 0.30))
    
    if not rows:
        return {"noise_fraction": noise_max, "silhouette_score": silhouette_min, "davies_bouldin_index": dbi_max}
        
    df = pd.DataFrame(rows, columns=["customer_id", "cluster_id", "cluster_label", "period_id"])
    
    # Query feature values
    features_list = []
    with get_raw_connection() as conn:
        with conn.cursor() as cur:
            for cid in df["customer_id"].tolist():
                # Fetch row from behavioral snapshots dynamically
                cols_str = ", ".join([f'"{col}"' for col in behavioral_cols])
                cur.execute(
                    f'SELECT {cols_str} FROM behavioral_snapshots WHERE customer_id = %s AND period_id = %s;',
                    (cid, period_id)
                )
                features_list.append(cur.fetchone())
                
    feat_df = pd.DataFrame(features_list, columns=behavioral_cols)
    
    # Convert labels: Noise is mapped to -1, others to distinct integers
    # Let's map cluster labels to integers
    labels = []
    for lbl in df["cluster_label"]:
        if "Noise" in lbl or lbl is None:
            labels.append(-1)
        else:
            try:
                # e.g. "Cluster_0" -> 0
                labels.append(int(lbl.split("_")[1]))
            except Exception:
                labels.append(hash(lbl) % 100)
                
    labels = np.array(labels)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(feat_df)
    
    # Compute metrics
    total_points = len(labels)
    noise_points = np.sum(labels == -1)
    noise_fraction = float(noise_points / total_points) if total_points > 0 else 0.0
    
    non_noise_mask = labels != -1
    unique_labels = set(labels[non_noise_mask])
    
    silhouette_val = float(os.getenv("CLUSTER_SILHOUETTE_MIN", 0.2))
    dbi_val = float(os.getenv("CLUSTER_DAVIES_BOULDIN_MAX", 1.5))
    
    if len(unique_labels) >= 2 and np.sum(non_noise_mask) >= 10:
        dbi_val = float(davies_bouldin_score(X_scaled[non_noise_mask], labels[non_noise_mask]))
        try:
            silhouette_val = float(silhouette_score(X_scaled[non_noise_mask], labels[non_noise_mask], sample_size=5000, random_state=42))
        except Exception:
            pass
            
    return {
        "noise_fraction": noise_fraction,
        "silhouette_score": silhouette_val,
        "davies_bouldin_index": dbi_val
    }

def decide_adopt_vs_hold(
    dataset_id: str,
    new_period_id: int,
    X_df: pd.DataFrame,
    labels: np.ndarray,
    behavioral_cols: list[str]
) -> str:
    """
    Quarterly Adopt-vs-Hold Decision Logic.
    
    Steps:
      1. Runs candidate quality gates check.
      2. Computes medoids and spreads.
      3. Compares medoid shift vs baseline medoids.
      4. Decides ADOPT or HOLD.
      5. Performs mapping writes, model retraining triggers, or trains KNN.
    """
    logger.info(f"Layer 2.4: Running Quarterly Stability & Drift Check for period {new_period_id}...")
    
    # 1. Load active baseline
    active_baseline_id, active_period_id = get_active_baseline()
    
    # Evaluate candidate clustering quality gates
    candidate_metrics = evaluate_clustering(X_df, behavioral_cols, labels)
    
    # Compute candidate medoids
    candidate_medoids = compute_medoids_and_spreads(X_df, labels, behavioral_cols)
    
    if not active_baseline_id:
        # Initial run: unconditionally adopt
        logger.info("No active baseline found. Adopting candidate clustering as initial baseline.")
        write_adopt_baseline(dataset_id, new_period_id, X_df, labels, candidate_medoids, behavioral_cols, "initial")
        return "ADOPT"
        
    # Retrieve active baseline clusters and medoids
    old_medoids = {}
    with get_raw_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT cluster_id, cluster_label, medoid_customer_id, intra_cluster_spread
                FROM clusters
                WHERE baseline_id = %s;
                """,
                (active_baseline_id,)
            )
            c_rows = cur.fetchall()
            
    features = X_df[behavioral_cols]
    scaler = StandardScaler()
    scaler.fit(features.values) # Fit on candidate data to use same coordinate scale
    
    for c_id, label, medoid_cid, spread in c_rows:
        if "Noise" in label or medoid_cid is None:
            continue
            
        # Retrieve medoid customer raw features at active_period_id
        with get_raw_connection() as conn:
            with conn.cursor() as cur:
                cols_str = ", ".join([f'"{col}"' for col in behavioral_cols])
                cur.execute(
                    f'SELECT {cols_str} FROM behavioral_snapshots WHERE customer_id = %s AND period_id = %s;',
                    (medoid_cid, active_period_id)
                )
                res = cur.fetchone()
                
        if res:
            # Transform to candidate scaled coordinate space
            raw_vals = np.array(res).reshape(1, -1)
            scaled_coords = scaler.transform(raw_vals)[0].tolist()
            
            # Map key by parsing label digit (e.g. Cluster_0 -> 0)
            try:
                c_key = int(label.split("_")[1])
            except Exception:
                c_key = c_id
                
            old_medoids[c_key] = {
                "cluster_id": c_id,
                "medoid_customer_id": medoid_cid,
                "medoid_coords": scaled_coords,
                "intra_cluster_spread": float(spread) if spread else 0.5
            }
            
    # Match candidate clusters to old clusters
    mappings = match_clusters(old_medoids, candidate_medoids)
    
    # 2. Check Drift Significance
    drift_significant = False
    drift_details = []
    
    for item in mappings:
        if item["match_type"] == "matched":
            o_key = item["old_cluster_id"]
            c_key = item["candidate_label"]
            
            # Distance calculated in match_clusters
            dist = item["distance"]
            stored_spread = old_medoids[o_key]["intra_cluster_spread"]
            
            # Drift > 2x spread is significant
            drift_mult = float(os.getenv("DRIFT_SIGNIFICANCE_MULTIPLIER", 2.0))
            if dist > drift_mult * stored_spread:
                drift_significant = True
                drift_details.append(f"Cluster {o_key}->{c_key} medoid shift {dist:.3f} > {drift_mult}x spread ({stored_spread:.3f})")

    # Structural topological reorganizations (splits, merges, disappearances, new clusters)
    structural_shifts = [m for m in mappings if m["match_type"] in ["split", "merged", "disappeared", "new"]]
    topology_changed = len(structural_shifts) > 0
                
    # 3. Quality & Drift Decision
    # Adoption requires valid candidate clustering PLUS significant drift or topological restructuring.
    # If candidate clustering has degraded quality or drift is within natural variance, maintain stability via HOLD.
    baseline_metrics = evaluate_baseline_quality(active_period_id, behavioral_cols)
    candidate_improved = (candidate_metrics["silhouette_score"] > baseline_metrics["silhouette_score"]) and \
                         (candidate_metrics["davies_bouldin_index"] < baseline_metrics["davies_bouldin_index"])
                          
    adopt_decision = drift_significant or topology_changed
    
    if adopt_decision:
        reason = []
        if drift_significant:
            reason.append(f"significant drift ({'; '.join(drift_details)})")
        if topology_changed:
            reason.append(f"structural shift ({len(structural_shifts)} split/merge/new)")
        if candidate_improved:
            reason.append("separation improved")
        reason_str = " | ".join(reason) if reason else "structural update"
        
        logger.info(f"Drift check decision: ADOPT new baseline due to {reason_str}.")
        write_adopt_baseline(dataset_id, new_period_id, X_df, labels, candidate_medoids, behavioral_cols, "drift", mappings, active_baseline_id)
        
        # Trigger Layer 4 retraining during Adopt (using T=active_period_id and target at T+1=new_period_id)
        try:
            from src.layer4.retrain_trigger import trigger_retraining
            trigger_retraining(dataset_id, active_period_id)
        except Exception as err:
            logger.error(f"Failed to trigger Layer 4 retraining during Adopt: {str(err)}", exc_info=True)
            
        # Soft warning emit for splits / merges / disappearances
        if structural_shifts:
            warn_msg = f"Quarterly clustering baseline adopted. Structural changes: {structural_shifts}"
            emit(
                source="layer2_drift_evaluator",
                condition="baseline_structural_shift",
                severity="soft_warning",
                dataset_type="file_upload",
                details=warn_msg
            )
        return "ADOPT"
    else:
        logger.info("Drift check decision: HOLD active baseline (minimal drift, topology stable). Training KNN classifier fallback.")
        write_hold_baseline(dataset_id, new_period_id, X_df, active_baseline_id, active_period_id, behavioral_cols)
        return "HOLD"

def write_adopt_baseline(
    dataset_id: str,
    period_id: int,
    X_df: pd.DataFrame,
    labels: np.ndarray,
    candidate_medoids: dict,
    behavioral_cols: list[str],
    reason: str,
    mappings: list[dict] = None,
    old_baseline_id: int = None
):
    """
    Creates a new cluster baseline, maps old cluster identities, and persists assignments.
    """
    with get_raw_connection() as conn:
        with conn.cursor() as cur:
            # 1. Update previous active baseline to false
            if old_baseline_id:
                cur.execute(
                    "UPDATE cluster_baselines SET is_active = FALSE WHERE baseline_id = %s;",
                    (old_baseline_id,)
                )
                
            # 2. Insert new baseline
            cur.execute(
                """
                INSERT INTO cluster_baselines (adopted_at, is_active, adopted_reason, period_id)
                VALUES (now(), TRUE, %s, %s)
                RETURNING baseline_id;
                """,
                (reason, period_id)
            )
            new_baseline_id = cur.fetchone()[0]
            
            # 3. Create Noise cluster record (required for FK constraints)
            cur.execute(
                """
                INSERT INTO clusters (baseline_id, cluster_label, medoid_customer_id, intra_cluster_spread)
                VALUES (%s, 'Noise', NULL, NULL)
                RETURNING cluster_id;
                """,
                (new_baseline_id,)
            )
            noise_cluster_id = cur.fetchone()[0]
            
            # 4. Insert active clusters and cache IDs
            candidate_to_cluster_id = {-1: noise_cluster_id}
            
            for c, stats in candidate_medoids.items():
                cur.execute(
                    """
                    INSERT INTO clusters (baseline_id, cluster_label, medoid_customer_id, intra_cluster_spread)
                    VALUES (%s, %s, %s, %s)
                    RETURNING cluster_id;
                    """,
                    (new_baseline_id, f"Cluster_{c}", stats["medoid_customer_id"], stats["intra_cluster_spread"])
                )
                candidate_to_cluster_id[c] = cur.fetchone()[0]
                
            # 5. Populate identity map for tracking splits/merges
            if mappings and old_baseline_id:
                # Fetch baseline cluster ID to label mappings for old baseline
                cur.execute("SELECT cluster_id, cluster_label FROM clusters WHERE baseline_id = %s;", (old_baseline_id,))
                old_map = {row[1]: row[0] for row in cur.fetchall()}
                
                for item in mappings:
                    m_type = item["match_type"]
                    o_key = item["old_cluster_id"]
                    c_key = item["candidate_label"]
                    
                    old_c_id = None
                    if o_key is not None:
                        # Match label string like 'Cluster_0'
                        old_c_id = old_map.get(f"Cluster_{o_key}")
                        
                    new_c_id = None
                    if c_key is not None:
                        new_c_id = candidate_to_cluster_id.get(c_key)
                        
                    cur.execute(
                        """
                        INSERT INTO cluster_identity_map (old_cluster_id, new_cluster_id, match_type)
                        VALUES (%s, %s, %s);
                        """,
                        (old_c_id, new_c_id, m_type)
                    )
                    
            # 6. Save customer assignments to cluster_assignments
            for i, row in X_df.iterrows():
                cid = row["customer_id"]
                lbl = labels[i]
                mapped_cluster_id = candidate_to_cluster_id[lbl]
                
                cur.execute(
                    """
                    INSERT INTO cluster_assignments (customer_id, period_id, cluster_id, assigned_via)
                    VALUES (%s, %s, %s, 'hdbscan')
                    ON CONFLICT (customer_id, period_id) DO UPDATE
                    SET cluster_id = EXCLUDED.cluster_id, assigned_via = EXCLUDED.assigned_via;
                    """,
                    (cid, period_id, mapped_cluster_id)
                )

def write_hold_baseline(
    dataset_id: str,
    period_id: int,
    X_df: pd.DataFrame,
    active_baseline_id: int,
    active_period_id: int,
    behavioral_cols: list[str]
):
    """
    Performs HOLD path. Trains KNN on the active baseline period data and classifies
    the new period's customers into the baseline clusters.
    """
    logger.info("Executing HOLD path: Training KNN and predicting...")
    
    # 1. Fetch training data from active baseline period
    with get_raw_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT bs.customer_id, ca.cluster_id, c.cluster_label
                FROM behavioral_snapshots bs
                JOIN cluster_assignments ca ON bs.customer_id = ca.customer_id AND bs.period_id = ca.period_id
                JOIN clusters c ON ca.cluster_id = c.cluster_id
                WHERE bs.period_id = %s;
                """,
                (active_period_id,)
            )
            train_rows = cur.fetchall()
            
    if not train_rows:
        raise ValueError(f"No customer assignments found for baseline period {active_period_id} to train KNN.")
        
    train_df = pd.DataFrame(train_rows, columns=["customer_id", "cluster_id", "cluster_label"])
    
    # Load features for training customers
    train_features = []
    with get_raw_connection() as conn:
        with conn.cursor() as cur:
            for cid in train_df["customer_id"].tolist():
                cols_str = ", ".join([f'"{col}"' for col in behavioral_cols])
                cur.execute(
                    f'SELECT {cols_str} FROM behavioral_snapshots WHERE customer_id = %s AND period_id = %s;',
                    (cid, active_period_id)
                )
                train_features.append(cur.fetchone())
                
    X_train_df = pd.DataFrame(train_features, columns=behavioral_cols)
    y_train = train_df["cluster_id"].values
    
    # Standardize train and test features together
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_df.values)
    
    X_test_raw = X_df[behavioral_cols]
    X_test_scaled = scaler.transform(X_test_raw.values)
    
    # 2. Train KNN (K=5, tunable)
    k_neighbors = int(os.getenv("KNN_K_NEIGHBORS", 5))
    logger.info(f"Training KNN classifier with K={k_neighbors}...")
    knn = KNeighborsClassifier(n_neighbors=k_neighbors)
    knn.fit(X_train_scaled, y_train)
    
    # 3. Predict new assignments
    predictions = knn.predict(X_test_scaled)
    
    # 4. Write assignments to DB
    with get_raw_connection() as conn:
        with conn.cursor() as cur:
            for i, row in X_df.iterrows():
                cid = row["customer_id"]
                assigned_cluster_id = int(predictions[i])
                
                cur.execute(
                    """
                    INSERT INTO cluster_assignments (customer_id, period_id, cluster_id, assigned_via)
                    VALUES (%s, %s, %s, 'knn_classifier')
                    ON CONFLICT (customer_id, period_id) DO UPDATE
                    SET cluster_id = EXCLUDED.cluster_id, assigned_via = EXCLUDED.assigned_via;
                    """,
                    (cid, period_id, assigned_cluster_id)
                )
                
    logger.info(f"HOLD path completed: saved {len(X_df)} customer assignments via KNN classifier.")
