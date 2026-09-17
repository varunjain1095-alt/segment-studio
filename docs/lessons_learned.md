# Lessons Learned

## Lesson 1 — Kaggle Synthetic Dataset Elimination

*   **Dataset**: `ecommerce_customer_churn_dataset.csv` (50,000 rows, 24 columns)
*   **Decision**: Dataset eliminated and replaced. Do not use this dataset for any future pipeline runs.
*   **Reason**: Four diagnostic tests confirmed the dataset is structurally unimodal — it has no natural cluster separation in its behavioral feature space. This is a characteristic of synthetically generated Kaggle datasets where each column is generated independently with uniform random distributions, rather than reflecting real correlated customer behavior.

### Diagnostic Evidence

| Test | Result | Interpretation |
| :--- | :---: | :--- |
| **Hopkins Statistic** | 0.793 | Single concentrated density mass, not multiple groups |
| **VAT Plot** | Smooth continuous gradient | No geometric cluster structure exists |
| **Silhouette Score (best)** | 0.204 (K-means, K=3) | Universally weak across all algorithms and K values |
| **DBCV Score** | -0.363 | Density-based separation statistically absent |

*   **Key Finding**: All silhouette scores remained below 0.21 across K-means, Agglomerative Clustering, and HDBSCAN at every parameter combination. This is not an HDBSCAN problem, not a parameter tuning problem, and not an implementation problem — K-means and Agglomerative Clustering failed equally. The dataset itself has no separable behavioral structure.
*   **Portfolio Note**: This diagnostic process is itself valuable evidence of analytical maturity — the dataset was formally tested and eliminated rather than silently swapped. Real-world data pipelines regularly encounter fabricated or poorly structured datasets; knowing how to detect and document this is a production-grade skill.
*   **Replacement**: See `New_Data_Set.csv` (1,000,000 rows, 80+ columns, real transactional + behavioral + demographic structure).
