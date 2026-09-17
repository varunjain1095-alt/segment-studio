import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import pdist, squareform

# Ensure diagnostics output directory exists
os.makedirs("diagnostics", exist_ok=True)

def hopkins_statistic(X, sample_ratio=0.1, random_state=42):
    n = len(X)
    n_samples = max(int(n * sample_ratio), 1)
    np.random.seed(random_state)
    
    # Fit Nearest Neighbors on the entire dataset
    nbrs = NearestNeighbors(n_neighbors=2, metric='euclidean').fit(X)
    
    # Sample real points and find distance to their nearest neighbor
    indices = np.random.choice(n, size=n_samples, replace=False)
    X_sample = X[indices]
    distances_w, _ = nbrs.kneighbors(X_sample, n_neighbors=2)
    w = distances_w[:, 1]
    
    # Generate random points inside the bounding box of X
    mins = X.min(axis=0)
    maxs = X.max(axis=0)
    X_rand = np.random.uniform(low=mins, high=maxs, size=(n_samples, X.shape[1]))
    
    # Find distance of random points to their nearest neighbor in X
    distances_u, _ = nbrs.kneighbors(X_rand, n_neighbors=1)
    u = distances_u[:, 0]
    
    sum_u = np.sum(u)
    sum_w = np.sum(w)
    
    if (sum_u + sum_w) == 0:
        return 0.5
    
    return sum_u / (sum_u + sum_w)

def compute_vat(X, sample_size=2000, random_state=42):
    n = len(X)
    if n > sample_size:
        np.random.seed(random_state)
        indices = np.random.choice(n, size=sample_size, replace=False)
        X_sample = X[indices]
    else:
        X_sample = X
        
    D = squareform(pdist(X_sample, metric='euclidean'))
    num_points = X_sample.shape[0]
    
    P = np.zeros(num_points, dtype=int)
    V = list(range(num_points))
    
    # Start with the pair that has maximum distance
    i, j = np.unravel_index(np.argmax(D), D.shape)
    P[0] = i
    V.remove(i)
    
    for r in range(1, num_points):
        sub_D = D[np.ix_(P[:r], V)]
        p_idx, v_idx = np.unravel_index(np.argmin(sub_D), sub_D.shape)
        P[r] = V[v_idx]
        V.pop(v_idx)
        
    D_ordered = D[np.ix_(P, P)]
    return D_ordered

def find_elbow_point(cumulative_variance):
    n = len(cumulative_variance)
    if n <= 2:
        return n
    x = np.arange(n)
    y = np.array(cumulative_variance)
    
    # Line from start to end of curve
    line_start = np.array([x[0], y[0]])
    line_end = np.array([x[-1], y[-1]])
    line_vec = line_end - line_start
    line_vec_norm = line_vec / np.sqrt(np.sum(line_vec**2))
    
    distances = []
    for i in range(n):
        point = np.array([x[i], y[i]])
        proj = line_start + np.dot(point - line_start, line_vec_norm) * line_vec_norm
        dist = np.sqrt(np.sum((point - proj)**2))
        distances.append(dist)
        
    return np.argmax(distances) + 1

# Dataset configurations
datasets_config = {
    "ecommerce_customer_churn": {
        "path": r"C:\Projects\Segment studio\test datasets\E Commerce Dataset.xlsx",
        "sheet_name": "E Comm",
        "is_excel": True,
        "behavioral_cols": [
            "Tenure", "HourSpendOnApp", "NumberOfDeviceRegistered", 
            "SatisfactionScore", "Complain", "OrderAmountHikeFromlastYear", 
            "CouponUsed", "OrderCount", "DaySinceLastOrder", "CashbackAmount"
        ]
    },
    "ecommerce_customer_behavior_tr": {
        "path": r"C:\Projects\Segment studio\test datasets\ecommerce_customer_behavior_dataset_v2.csv",
        "is_excel": False,
        "behavioral_cols": [
            "Unit_Price", "Quantity", "Discount_Amount", "Total_Amount", 
            "Session_Duration_Minutes", "Pages_Viewed", "Is_Returning_Customer", 
            "Customer_Rating"
        ]
    },
    "ecommerce_consumer_behavior": {
        "path": r"C:\Projects\Segment studio\test datasets\Ecommerce_Consumer_Behavior_Analysis_Data.csv",
        "is_excel": False,
        "behavioral_cols": [
            "Purchase_Amount", "Frequency_of_Purchase", "Brand_Loyalty", 
            "Product_Rating", "Time_Spent_on_Product_Research(hours)", 
            "Social_Media_Influence", "Discount_Sensitivity", "Return_Rate", 
            "Customer_Satisfaction", "Engagement_with_Ads", "Discount_Used", 
            "Customer_Loyalty_Program_Member", "Purchase_Intent", "Time_to_Decision"
        ]
    }
}

for name, cfg in datasets_config.items():
    print(f"\n========================================\nPROCESSING DATASET: {name}\n========================================")
    
    # Step 1: Load and report basic structure
    if cfg["is_excel"]:
        df = pd.read_excel(cfg["path"], sheet_name=cfg["sheet_name"])
    else:
        df = pd.read_csv(cfg["path"])
        
    print(f"Row count: {df.shape[0]}")
    print(f"Column count: {df.shape[1]}")
    print("\nColumn List and Dtypes:")
    for col, dtype in zip(df.columns, df.dtypes):
        print(f"  - {col}: {dtype}")
        
    print("\nMissing value counts per column:")
    missing_counts = df.isnull().sum()
    for col, count in missing_counts.items():
        print(f"  - {col}: {count}")
        
    print("\nFile structure: Single file")
    
    # Step 2: Report identified behavioral columns
    print(f"\nIdentified numeric behavioral columns:\n  {cfg['behavioral_cols']}")
    
    # Step 3: Light preprocessing only
    df_proc = df.copy()
    
    # Impute missing values
    for col in df_proc.columns:
        if pd.api.types.is_numeric_dtype(df_proc[col]):
            median_val = df_proc[col].median()
            df_proc[col] = df_proc[col].fillna(median_val)
        else:
            mode_val = df_proc[col].mode().iloc[0] if not df_proc[col].mode().empty else np.nan
            df_proc[col] = df_proc[col].fillna(mode_val)
            
    # Encode categorical columns using frequency encoding
    for col in df_proc.columns:
        if not pd.api.types.is_numeric_dtype(df_proc[col]):
            freqs = df_proc[col].value_counts(normalize=True)
            df_proc[col] = df_proc[col].map(freqs)
            
    # Scale numeric columns only
    # (Since we are building a feature matrix of behavioral columns, let's select those columns and scale them)
    # RobustScaler only to numeric columns
    numeric_cols = [c for c in cfg["behavioral_cols"] if df[c].dtype != 'object' and not isinstance(df[c].dtype, pd.CategoricalDtype)]
    scaler = RobustScaler()
    df_proc[numeric_cols] = scaler.fit_transform(df_proc[numeric_cols])
    
    # Feature matrix consists of the behavioral columns
    X = df_proc[cfg["behavioral_cols"]].values
    
    # Apply PCA
    pca = PCA().fit(X)
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    
    print("\nPCA Cumulative Variance Table:")
    print("Component | Incremental Var | Cumulative Var")
    print("-" * 45)
    for idx, (inc, cum) in enumerate(zip(pca.explained_variance_ratio_, cumulative_variance)):
        print(f"   PC{idx+1:02d}   |     {inc:.6f}    |    {cum:.6f}")
        
    elbow = find_elbow_point(cumulative_variance)
    print(f"\nIdentified PCA Elbow Point: PC{elbow}")
    
    # Reduce to elbow components
    X_pca = PCA(n_components=elbow).fit_transform(X)
    
    # Step 4: Run diagnostic tests on PCA-reduced behavioral feature matrix
    print("\nRunning Diagnostic Tests...")
    
    # Hopkins Statistic (10% sample)
    hopkins_val = hopkins_statistic(X_pca, sample_ratio=0.1, random_state=42)
    print(f"Hopkins Statistic (10% sample): {hopkins_val:.6f}")
    
    # VAT Plot (2,000 sample)
    vat_matrix = compute_vat(X_pca, sample_size=2000, random_state=42)
    plt.figure(figsize=(8, 8))
    plt.imshow(vat_matrix, cmap='gray', vmin=0, vmax=np.percentile(vat_matrix, 95))
    plt.title(f"VAT Plot - {name}")
    plt.colorbar(label='Dissimilarity')
    plot_path = f"diagnostics/vat_{name}.png"
    plt.savefig(plot_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Saved VAT plot to: {plot_path}")
    
    # Silhouette and Calinski-Harabasz Scores
    K_values = [3, 4, 5, 6]
    silhouette_km = []
    silhouette_agg = []
    ch_km = []
    
    # For large datasets, sample 10,000 rows for silhouette scoring to be computationally feasible
    sample_size_score = 10000
    if len(X_pca) > sample_size_score:
        np.random.seed(42)
        score_indices = np.random.choice(len(X_pca), size=sample_size_score, replace=False)
        X_score = X_pca[score_indices]
    else:
        X_score = X_pca
        score_indices = np.arange(len(X_pca))
        
    for k in K_values:
        # K-means
        km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_pca)
        km_labels_all = km.labels_
        km_labels_score = km_labels_all[score_indices]
        
        sil_km = silhouette_score(X_score, km_labels_score, random_state=42)
        silhouette_km.append(sil_km)
        
        ch = calinski_harabasz_score(X_score, km_labels_score)
        ch_km.append(ch)
        
        # Agglomerative
        agg = AgglomerativeClustering(n_clusters=k).fit(X_score)
        sil_agg = silhouette_score(X_score, agg.labels_, random_state=42)
        silhouette_agg.append(sil_agg)
        
    print("\nSilhouette Scores (K-means vs Agglomerative):")
    print(" K | K-means | Agglomerative")
    print("-" * 30)
    for k, skm, sagg in zip(K_values, silhouette_km, silhouette_agg):
        print(f" {k} | {skm:.6f} | {sagg:.6f}")
        
    print("\nCalinski-Harabasz Index (K-means):")
    print(" K | K-means CH Score")
    print("-" * 25)
    for k, ch_val in zip(K_values, ch_km):
        print(f" {k} | {ch_val:.2f}")

print("\nAll diagnostics completed successfully.")
