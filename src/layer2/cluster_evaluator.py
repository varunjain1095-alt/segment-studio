import os
import logging
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, silhouette_score, davies_bouldin_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from src.notifications.center import emit

logger = logging.getLogger("layer2_cluster_evaluator")

def evaluate_clustering(
    X_df: pd.DataFrame,
    behavioral_cols: list[str],
    labels: np.ndarray
) -> dict:
    """
    Evaluates the quality of a clustering run against three gates:
      1. Stability: ARI of sub-samples vs full dataset.
      2. Separation: Silhouette Score and Davies-Bouldin Index on non-noise points.
      3. Noise Sanity: Fraction of noise points (-1).
    
    Loads thresholds dynamically from env vars:
      - CLUSTER_STABILITY_ARI_THRESHOLD (default: 0.75)
      - CLUSTER_SILHOUETTE_MIN (default: 0.2)
      - CLUSTER_DAVIES_BOULDIN_MAX (default: 1.5)
      - CLUSTER_NOISE_FRACTION_MAX (default: 0.30)
      - CLUSTER_STABILITY_RUNS (default: 3)
      - CLUSTER_STABILITY_SAMPLE_RATIO (default: 0.8)
    
    Returns a dict with metric values and pass/fail statuses.
    If any check fails, emits a hard_fail notification and raises ValueError.
    """
    logger.info("Layer 2.2: Starting cluster quality gates evaluation...")
    
    # Load threshold configs
    ari_threshold = float(os.getenv("CLUSTER_STABILITY_ARI_THRESHOLD", 0.75))
    silhouette_min = float(os.getenv("CLUSTER_SILHOUETTE_MIN", 0.2))
    dbi_max = float(os.getenv("CLUSTER_DAVIES_BOULDIN_MAX", 1.5))
    noise_max = float(os.getenv("CLUSTER_NOISE_FRACTION_MAX", 0.30))
    
    stability_runs = int(os.getenv("CLUSTER_STABILITY_RUNS", 3))
    sample_ratio = float(os.getenv("CLUSTER_STABILITY_SAMPLE_RATIO", 0.80))
    
    features = X_df[behavioral_cols]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)
    
    # --- Gate 1: Noise Sanity ---
    total_points = len(labels)
    noise_points = np.sum(labels == -1)
    noise_fraction = float(noise_points / total_points) if total_points > 0 else 1.0
    
    noise_pass = noise_fraction <= noise_max
    logger.info(f"Noise Sanity Check: fraction = {noise_fraction:.4f} (max={noise_max}), pass={noise_pass}")
    
    # --- Gate 2: Separation (Silhouette & DBI) ---
    # Exclude noise points (label -1) for separation metrics
    non_noise_mask = labels != -1
    non_noise_count = np.sum(non_noise_mask)
    
    silhouette_val = -1.0
    dbi_val = 999.0
    separation_pass = False
    
    # Check if there are at least 2 active clusters to evaluate separation
    active_labels = labels[non_noise_mask]
    unique_active_clusters = set(active_labels)
    
    if len(unique_active_clusters) >= 2 and non_noise_count >= 10:
        X_non_noise = X_scaled[non_noise_mask]
        
        # Calculate Davies-Bouldin Index (very fast)
        dbi_val = float(davies_bouldin_score(X_non_noise, active_labels))
        
        # Calculate Silhouette Score (can be slow on large datasets, so sample if needed)
        sample_size = int(os.getenv("CLUSTER_SILHOUETTE_SAMPLE_SIZE", 10000))
        if len(X_non_noise) > sample_size:
            silhouette_val = float(silhouette_score(X_non_noise, active_labels, sample_size=sample_size, random_state=42))
        else:
            silhouette_val = float(silhouette_score(X_non_noise, active_labels))
            
        separation_pass = (silhouette_val >= silhouette_min) and (dbi_val <= dbi_max)
    else:
        logger.warning(f"Not enough active clusters or non-noise points ({non_noise_count} points, {len(unique_active_clusters)} active clusters) to evaluate separation.")
        separation_pass = False
        
    logger.info(f"Separation Check: Silhouette = {silhouette_val:.4f} (min={silhouette_min}), DBI = {dbi_val:.4f} (max={dbi_max}), pass={separation_pass}")
    
    # --- Gate 3: Stability (ARI) ---
    stability_pass = False
    ari_scores = []
    
    if len(unique_active_clusters) >= 1:
        n_clusters_eval = len(unique_active_clusters)
        if n_clusters_eval < 1:
            n_clusters_eval = int(os.getenv("KMEANS_N_CLUSTERS", 3))
        
        for i in range(stability_runs):
            # Bootstrap subsampling: sample ratio of the data
            indices = np.random.choice(total_points, size=int(total_points * sample_ratio), replace=False)
            X_sample = X_scaled[indices]
            
            # Run KMeans on sub-sample
            clusterer_sub = KMeans(n_clusters=n_clusters_eval, random_state=42 + i, n_init=10)
            labels_sub = clusterer_sub.fit_predict(X_sample)
            
            # Compare sub-run against standard run labels for matching indices
            labels_full_sub = labels[indices]
            
            # Compute ARI between full labels and sublabels
            ari = float(adjusted_rand_score(labels_full_sub, labels_sub))
            ari_scores.append(ari)
            
        mean_ari = float(np.mean(ari_scores)) if ari_scores else 0.0
        stability_pass = mean_ari >= ari_threshold
    else:
        mean_ari = 0.0
        stability_pass = False
        
    logger.info(f"Stability Check: mean ARI = {mean_ari:.4f} (min={ari_threshold}), pass={stability_pass}")
    
    results = {
        "stability_ari": mean_ari,
        "stability_pass": stability_pass,
        "silhouette_score": silhouette_val,
        "davies_bouldin_index": dbi_val,
        "separation_pass": separation_pass,
        "noise_fraction": noise_fraction,
        "noise_pass": noise_pass,
        "all_passed": bool(stability_pass and separation_pass and noise_pass)
    }
    
    if not results["all_passed"]:
        failed_gates = []
        if not stability_pass:
            failed_gates.append(f"Stability (ARI={mean_ari:.3f} < {ari_threshold})")
        if not separation_pass:
            failed_gates.append(f"Separation (Silhouette={silhouette_val:.3f} < {silhouette_min} OR DBI={dbi_val:.3f} > {dbi_max})")
        if not noise_pass:
            failed_gates.append(f"Noise Sanity (fraction={noise_fraction:.3f} > {noise_max})")
            
        failed_gates_str = ", ".join(failed_gates)
        msg = f"Cluster Quality Gate Failed: {failed_gates_str}. Downstream execution blocked."
        logger.error(msg)
        emit(
            source="layer2_cluster_evaluator",
            condition="quality_gate_failure",
            severity="hard_fail",
            dataset_type="file_upload",
            details=msg
        )
        raise ValueError(msg)
        
    logger.info("Layer 2.2: All cluster quality gates passed successfully.")
    return results
