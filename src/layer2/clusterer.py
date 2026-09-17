import os
import logging
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

logger = logging.getLogger("layer2_clusterer")

PCA_VARIANCE_THRESHOLD = float(os.getenv("PCA_VARIANCE_THRESHOLD", 0.85))
NOISE_DISTANCE_THRESHOLD_SIGMA = float(os.getenv("NOISE_DISTANCE_THRESHOLD_SIGMA", 3.0))

def flag_cluster_noise(
    X_pca: np.ndarray,
    labels: np.ndarray,
    centers: np.ndarray,
    sigma_threshold: float = None,
    min_cluster_size: int = 5
) -> np.ndarray:
    """
    Computes Euclidean distance from each observation to its assigned cluster center in PCA space.
    Observations in tiny isolated clusters (< min_cluster_size) are flagged as noise (-1).
    Observations with distance > mean_distance + sigma_threshold * std_distance for their cluster
    are flagged as statistical outliers (noise label: -1).
    """
    if sigma_threshold is None:
        sigma_threshold = NOISE_DISTANCE_THRESHOLD_SIGMA

    final_labels = labels.copy()
    unique_clusters = set(labels) - {-1}

    for c in unique_clusters:
        idx_c = np.where(labels == c)[0]
        if len(idx_c) < min_cluster_size:
            # Singleton or tiny cluster formed around extreme outlier(s)
            final_labels[idx_c] = -1
            continue

        center_c = centers[c]
        dists = np.linalg.norm(X_pca[idx_c] - center_c, axis=1)
        mean_d = float(np.mean(dists))
        std_d = float(np.std(dists))
        
        if std_d > 1e-6:
            cutoff = mean_d + (sigma_threshold * std_d)
            outlier_mask = dists > cutoff
            outlier_indices = idx_c[outlier_mask]
            final_labels[outlier_indices] = -1

    return final_labels

def run_kmeans_clustering(
    X_df: pd.DataFrame,
    behavioral_cols: list[str],
    n_clusters: int = None,
    detect_noise: bool = True
) -> np.ndarray:
    """
    Runs K-means clustering on the behavioral columns of X_df.
    
    Steps:
      1. Standardizes the feature matrix using StandardScaler.
      2. Applies PCA to reduce dimensionality deterministically.
      3. Fits KMeans and predicts cluster labels.
      4. Detects statistical outliers (distance > 3-sigma from centroid) as noise (-1).
    
    Returns:
      labels (np.ndarray): An array of integer labels for each row (0..k-1, or -1 for noise).
    """
    logger.info("Layer 2.1: Running StandardScaler and KMeans clustering...")
    
    if n_clusters is None:
        n_clusters = int(os.getenv("KMEANS_N_CLUSTERS", 3))
        
    if len(X_df) < n_clusters:
        raise ValueError(f"Number of samples ({len(X_df)}) is less than n_clusters ({n_clusters}).")
    
    # 1. Extract behavioral features and scale
    features = X_df[behavioral_cols]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)
    
    # 2. Apply PCA
    logger.info(f"Applying PCA with explained variance threshold = {PCA_VARIANCE_THRESHOLD}...")
    n_comp = min(X_scaled.shape[0], X_scaled.shape[1])
    n_components_pca = PCA_VARIANCE_THRESHOLD
    if n_comp < 1:
        raise ValueError("No features or samples available for PCA.")
    elif n_comp == 1:
        n_components_pca = 1
        
    pca = PCA(n_components=n_components_pca, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    logger.info(f"PCA reduced dimensions from {X_scaled.shape[1]} to {X_pca.shape[1]} components.")
    
    # 3. Fit KMeans
    logger.info(f"Fitting KMeans with n_clusters={n_clusters}...")
    clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = clusterer.fit_predict(X_pca)
    
    # 4. Statistical Outlier Detection (Noise Gate input)
    if detect_noise:
        labels = flag_cluster_noise(X_pca, labels, clusterer.cluster_centers_)
        noise_cnt = np.sum(labels == -1)
        logger.info(f"Flagged {noise_cnt} statistical outliers as noise (-1).")
    
    return labels

def run_gmm_clustering(
    X_df: pd.DataFrame,
    behavioral_cols: list[str],
    n_clusters: int = None,
    detect_noise: bool = True
) -> np.ndarray:
    """
    Runs Gaussian Mixture Model (GMM) clustering on the behavioral columns of X_df.
    
    Steps:
      1. Standardizes the feature matrix using StandardScaler.
      2. Applies PCA to reduce dimensionality deterministically.
      3. Fits GaussianMixture and predicts cluster labels.
      4. Detects statistical outliers (distance > 3-sigma from component mean) as noise (-1).
    
    Returns:
      labels (np.ndarray): An array of integer labels for each row (0..k-1, or -1 for noise).
    """
    logger.info("Layer 2.1 (GMM): Running StandardScaler and GMM clustering...")
    
    if n_clusters is None:
        n_clusters = int(os.getenv("KMEANS_N_CLUSTERS", 3))
        
    if len(X_df) < n_clusters:
        raise ValueError(f"Number of samples ({len(X_df)}) is less than n_clusters ({n_clusters}).")
    
    # 1. Extract behavioral features and scale
    features = X_df[behavioral_cols]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)
    
    # 2. Apply PCA
    n_comp = min(X_scaled.shape[0], X_scaled.shape[1])
    n_components_pca = PCA_VARIANCE_THRESHOLD
    if n_comp < 1:
        raise ValueError("No features or samples available for PCA.")
    elif n_comp == 1:
        n_components_pca = 1
        
    pca = PCA(n_components=n_components_pca, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    
    # 3. Fit GMM
    logger.info(f"Fitting GMM with n_components={n_clusters}...")
    clusterer = GaussianMixture(n_components=n_clusters, random_state=42)
    labels = clusterer.fit_predict(X_pca)
    
    # 4. Statistical Outlier Detection
    if detect_noise:
        labels = flag_cluster_noise(X_pca, labels, clusterer.means_)
        noise_cnt = np.sum(labels == -1)
        logger.info(f"Flagged {noise_cnt} statistical outliers as noise (-1).")
    
    return labels
