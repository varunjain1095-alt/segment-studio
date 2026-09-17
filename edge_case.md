# Customer Segmentation Studio — Edge Case Registry & Failure Modes

This document catalogs critical corner scenarios, algorithmic failure modes, memory bottlenecks, and logical constraints identified across all layers of the Segment Studio pipeline. It serves as a checklist for test suite coverage and robust system design.

---

## 1. Layer 0 — Data Preparation Foundation

### 1.1 Outlier Scaling Compression
* **Scenario:** The base dataset contains massive numeric outliers (e.g., `Average_Order_Value` has a maximum of \$9,666.37 but a 75th percentile of \$144.44). 
* **Failure Mode:** Standardization via `RobustScaler` maps normal data points into an extremely compressed range (e.g., $[0, 0.01]$) while the outliers are scaled to massive numbers. KMeans sees the normal data as a single dense point cloud and classifies it entirely as a single cluster or noise.
* **Mitigation:** 
  - Apply pre-scaling transforms: `log1p` for strictly positive columns, and Yeo-Johnson for columns crossing negative boundaries.
  - Implement percentile-based winsorization (clipping values at 1% and 99%) during the outlier detection phase.

### 1.2 Imputation-Induced Density Spikes
* **Scenario:** A column has a high percentage of missing values (e.g., 40%) and is imputed using a single value like the median.
* **Failure Mode:** Imputing thousands of rows with the exact median creates a massive, artificial spike in density at a single coordinate. After scaling, this manifests as a high-density cluster centered exactly at `0.0`. KMeans centroids get heavily pulled toward this density spike, distorting the cluster boundaries.
* **Mitigation:**
  - Enforce a strict warning threshold for columns with missing values $>25\%$.
  - **Proposed Open Question for PM:** Imputing missing values with a single constant (like the median) can create density spikes that degrade KMeans' clustering. The PM must explicitly decide whether to approve advanced imputation models (KNN/regression) or alternate handling (such as dropping rows/columns) rather than introducing unapproved mechanisms like random jitter.

### 1.3 Numeric-Object String Coercion Collision
* **Scenario:** A numeric column is stored as an `object` type because it contains text placeholders like `"N/A"`, `"none"`, or blank strings `" "`.
* **Failure Mode:** If not coerced, it is treated as categorical, leading to high-cardinality frequency encoding or one-hot explosion. If coerced blindly, it may produce massive NaN blocks that trigger the $>50\%$ column-drop rule.
* **Mitigation:** 
  - The `suspect_numeric_object` heuristic must inspect the non-null string patterns. If $>90\%$ of non-null values are numeric, coerce using `pd.to_numeric(col, errors='coerce')` and immediately re-trigger the missing-value validator before validation runs.

### 1.4 Valid Negatives triggering Alerts
* **Scenario:** Columns like `Account_Balance_Delta` or `Net_Revenue` naturally contain negative values.
* **Failure Mode:** The negative-value flagging logic is unconditional, generating constant false-positive warnings that desensitize the PM.
* **Mitigation:** Keep the negative-value flagging logic unconditional across all numeric columns to prevent silent ingestion errors. Do not implement a per-column mute or exception mechanism. If PM alert desensitization becomes a significant concern, this must be raised as a design review question to be resolved at the UI/UX layer rather than altering the backend validation rules.

---

## 2. Layer 1 — Demographic Segmentation

### 2.1 Low Segment Size Volatility (Tiny Denominators)
* **Scenario:** A rules-based demographic segment is very small (e.g., `Location = Iceland` contains 4 customers).
* **Failure Mode:** A single customer signing up or churns causes a $+25\%$ or $-25\%$ month-over-month swing. This triggers false positive anomalies in the volatility-aware flagger.
* **Mitigation:** 
  - Set a minimum segment size gate ($N \ge 30$) before computing percentage change metrics.
  - Small segments should default to absolute change tracking or be pooled into an "Other" category.

### 2.2 Schema Invalidation Lock
* **Scenario:** In dynamic database connections, a routine query pulls a new categorical field (e.g., shipping provider).
* **Failure Mode:** The schema monitor detects a new column, triggers a hard fail, and blocks the ingestion pipeline waiting for PM classification.
* **Mitigation:** Categorize new columns as `Unsure` and log a warning, but only block the pipeline if the new column collides with existing behavioral or demographic keys, or if it is classified as a primary key.

---

## 3. Layer 2 — Behavioral Clustering Pipeline

### 3.1 Poor Separation / Uniform Data Collapse
* **Scenario:** The behavioral metrics are uniformly distributed (e.g., generated synthetically with uniform intervals), meaning there are no natural high-density clusters separated by low-density valleys.
* **Failure Mode:** KMeans partitions the data, but the silhouette score drops below the minimum threshold (0.2) or Davies-Bouldin index exceeds the maximum (1.5). The separation quality gate fails, blocking the pipeline.
* **Mitigation:**
  - **Do NOT implement an automatic KMeans fallback:** Swapping the clustering algorithm in the background violates the structural design and changes downstream segment behavior.
  - If KMeans quality gates fail, flag the dataset as *clustering-unsuitable*, surface a hard warning to the PM, and block downstream steps.
  - Perform hyperparameter tuning via an automated grid search sweep over KMeans parameters (`n_clusters`) to find settings that minimize the noise ratio before declaring the dataset unsuitable.

### 3.2 Medoid Pairwise Distance Memory Exhaustion
* **Scenario:** A cluster contains a large number of rows (e.g., $N = 31,399$).
* **Failure Mode:** Computing pairwise Euclidean distances across the entire cluster requires allocating a square matrix of size $31399 \times 31399$ floats. At 8 bytes per float64, this requires **7.35 GiB** of contiguous memory, crashing the python process with `numpy._core._exceptions._ArrayMemoryError`.
* **Mitigation:**
  - Down-sample the cluster coordinates before calculating the medoid. If a cluster size exceeds $M = 2,000$ points, randomly sample 2,000 points to represent the cluster and find the medoid among them. This limits the maximum distance matrix to $2000 \times 2000$ (30.5 MB of memory).

### 3.3 Medoid-Matching Cluster Collisions (Splits and Merges)
* **Scenario:** During a quarterly re-clustering (Layer 2.4), Cluster A splits into two new clusters, or Cluster A and B merge into one.
* **Failure Mode:** Downstream transition matrices (Layer 5) and predictive models (Layer 4) expect a static 1-to-1 relationship. A split or merge ruins historical continuity.
* **Mitigation:**
  - Implement the `cluster_identity_map` with specific match categories: `split`, `merge`, `disappeared`, and `new`.
  - On a `split` event: Map the old cluster ID to both new IDs, and alert the PM to rename them.
  - On a `merge` event: Map both old IDs to the new merged ID, and trigger retraining for Layer 4 models.

---

## 4. Layer 3 — Statistical Testing Layer

### 4.1 Circular Feature Leakage
* **Scenario:** Testing whether the clusters are statistically different using the same behavioral features (e.g., `Average_Order_Value`) that KMeans used to build the clusters.
* **Failure Mode:** The tests will trivially show extreme significance ($p \approx 0.000$) and large effect sizes, generating false confidence in the segmentation.
* **Mitigation:** 
  - Strictly validate inputs in the testing suite. Only allow columns registered as outcome metrics (Layer 6) to be passed as test targets. Raise a `ValueError` if any feature in the clustering set is passed.

### 4.2 Large-Sample p-Value Bias
* **Scenario:** The dataset is very large ($N = 50,000$ rows).
* **Failure Mode:** For large samples, even a trivial difference in means (e.g., an AOV difference of \$0.05 between segments) will yield a p-value of $<0.01$, indicating statistical significance. This leads to over-segmentation.
* **Mitigation:**
  - Never evaluate significance solely on p-values.
  - Report and enforce minimum threshold checks on **Effect Size** (Cohen's d for means, Cramér's V for Chi-square, or Eta-squared for ANOVA). If effect size is negligible, do not flag the segments as diverging.

---

## 5. Layer 4 — Predictive Segmentation

### 5.1 Minority Class Imbalance (Predictive Death)
* **Scenario:** Only a tiny fraction of customers migrate between period T and T+1 (e.g., 1% migration rate).
* **Failure Mode:** A classifier can achieve $99\%$ accuracy by predicting `migrated = False` for all rows. Overall ROC-AUC will remain high, masking the fact that the model is completely useless at detecting actual migrations.
* **Mitigation:**
  - Evaluate model selection using **Precision-Recall AUC (PR-AUC)** on the minority class (`migrated = True`) instead of ROC-AUC.
  - Monitor minority F1-score and minority recall.
  - Utilize class-weighting (e.g., `scale_pos_weight` in XGBoost) or SMOTE during training to handle the imbalance.

### 5.2 Lookahead Leakage
* **Scenario:** Features for period T contain metrics that incorporate data from period T+1 (e.g., `Days_Since_Last_Purchase` calculated at the end of T+1).
* **Failure Mode:** The predictive model achieves near-perfect validation metrics, but fails in production because it is relying on future values.
* **Mitigation:**
  - Build the training datasets using strict temporal offsets. Features must be derived strictly from the time window $[T_{start}, T_{end}]$. Target labels must be derived strictly from the time window $[T+1_{start}, T+1_{end}]$.

---

## 6. Layer 5 — Segment Migration Forecasting

### 6.1 Empty Cluster Denominator
* **Scenario:** A cluster has 0 members at period T (e.g., a newly defined cluster or a cluster that temporarily emptied).
* **Failure Mode:** Calculating transition probability $P(A \rightarrow B)$ divides by the customer count in A, resulting in a `ZeroDivisionError`.
* **Mitigation:**
  - Check the denominator before division. If count is 0, set all transition probabilities from that cluster to 0.0 and log a warning.

---

## 7. Layer 8 — Feature-Level Migration Signal

### 7.1 Equidistant Distance Tie
* **Scenario:** A customer's column value lies exactly halfway between their own cluster's median and the comparison cluster's median.
* **Failure Mode:** Float precision comparison could result in arbitrary drift assignment.
* **Mitigation:**
  - Enforce a strict inequality: `drift[col] = True if d_compare < d_own else False`. If the distance is equal, it defaults to `False` (conservative assumption).

### 7.2 Missing Cluster Median Referential Integrity
* **Scenario:** Layer 8 runs a comparison against a cluster that has no computed median (e.g., a newly introduced cluster in a draft baseline).
* **Failure Mode:** Crashes the script with a `KeyError`.
* **Mitigation:**
  - Validate that the target cluster exists in the currently active baseline definition before running the column loop. If not found, skip the comparison and flag a database integrity warning.
