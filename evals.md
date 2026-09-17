# Customer Segmentation Studio — Evaluation Specification

> **Source documents:** [implementation_plan.md](file:///c:/Projects/Segment%20studio/implementation_plan.md) · [architecture.md](file:///c:/Projects/Segment%20studio/architecture.md) · [context.md](file:///c:/Projects/Segment%20studio/context.md) · [edge_case.md](file:///C:/Users/Varun%20Jain/.gemini/antigravity-ide/brain/fa2330b3-608f-4ee2-ab7f-00a19abbfeef/edge_case.md)

This document defines test scenarios, expected behavior, and pass/fail criteria for every implementation phase. Each section maps to its corresponding phase in `implementation_plan.md`. All tests are to be implemented under `tests/` using `pytest`.

> [!IMPORTANT]
> **UI Gate is active.** Eval scenarios for Phase 11 (UI) are intentionally omitted until the UI gate is lifted. All other phases focus on backend, pipeline, and API correctness only.

---

## Eval Conventions

| Symbol | Meaning |
| :--- | :--- |
| ✅ PASS | Assertion holds; the phase component works as specified |
| ❌ FAIL | Assertion fails; the component does not meet spec and must be fixed before proceeding |
| ⚠️ SOFT WARN | Non-blocking warning emitted; pipeline continues but event is logged |
| 🚫 HARD FAIL | Blocking failure emitted; pipeline halts and PM must resolve |

All hard-fail notifications must be retrievable from the `notifications` table with `severity = 'hard_fail'`.

---

## Phase 0 — Project Foundation & Infrastructure

### P0-T1: Docker & Database Startup

| Field | Detail |
| :--- | :--- |
| **Scenario** | Run `docker-compose up` on a clean environment. |
| **Expected behavior** | Postgres container starts. All 12 tables (`customers`, `periods`, `behavioral_snapshots`, `cluster_baselines`, `clusters`, `cluster_assignments`, `cluster_identity_map`, `migration_labels`, `transition_matrix`, `segment_outcomes`, `feature_migration_signals`, `demographic_snapshots`) are created without migration errors. |
| **Pass** | `docker-compose up` exits with 0 errors. `psql` count of tables equals 12. All FK references resolve. |
| **Fail** | Any migration error, missing table, or unresolvable FK. |

### P0-T2: Period ID Constraint on All Snapshot Tables

| Field | Detail |
| :--- | :--- |
| **Scenario** | Inspect the DDL for every table involved in T vs T+1 comparisons. |
| **Expected behavior** | `behavioral_snapshots`, `cluster_assignments`, `migration_labels`, `transition_matrix`, `segment_outcomes`, `feature_migration_signals`, and `demographic_snapshots` all carry a `period_id` foreign key column referencing the `periods` table. |
| **Pass** | All listed tables contain `period_id NOT NULL REFERENCES periods(id)`. |
| **Fail** | Any of the listed tables lacks `period_id`. |

### P0-T3: Anthropic Claude Client Smoke Test

| Field | Detail |
| :--- | :--- |
| **Scenario** | Call `claude_client.call(prompt="Say hello", system_prompt="You are a helpful assistant.")` from the wrapper. |
| **Expected behavior** | The function returns a non-empty string response. The call is logged (prompt length, model name, response length, latency) to structured JSON log output. |
| **Pass** | Response is a non-empty string. Log entry exists with all four fields populated. |
| **Fail** | Empty response, API error, or missing log entry. |

### P0-T4: Notification Center — Hard Fail Emission & Retrieval

| Field | Detail |
| :--- | :--- |
| **Scenario** | Call `emit(source="test", condition="unit_test", severity="hard_fail", dataset_type="file_upload", details="Test event")`. |
| **Expected behavior** | An entry is inserted to the `notifications` table with `severity = 'hard_fail'`. |
| **Pass** | Row exists in `notifications` where `severity = 'hard_fail'` and `condition = 'unit_test'`. |
| **Fail** | No row inserted, or `severity` is not `hard_fail`. |

### P0-T5: Notification Center — Soft Warning Does Not Block

| Field | Detail |
| :--- | :--- |
| **Scenario** | Call `emit()` with `severity="soft_warning"`. Then immediately call the next pipeline step. |
| **Expected behavior** | The pipeline step proceeds without raising an exception. The warning is logged. |
| **Pass** | No exception raised. Warning row is present in `notifications`. |
| **Fail** | Exception raised after soft warning emission. |

---

## Phase 1 — Data Ingestion & Layer 0: Data Preparation Foundation

### P1-T1: `data_profile()` — Correct Output Shape

| Field | Detail |
| :--- | :--- |
| **Scenario** | Pass a known test DataFrame with 5 columns: 1 all-null numeric, 1 fully unique, 1 with 30% missing, 1 object with mixed numeric strings, 1 all-integer. |
| **Expected behavior** | `data_profile()` returns a DataFrame with one row per column and all fields: `missing_count`, `missing_pct`, `dtype`, `suspect_numeric_object`, `unique_count`, `non_unique_count`, `non_unique_pct`, `is_unique_key`, `has_duplicates`, `severity`. |
| **Pass** | All 10 expected columns present. Rows sorted descending by `missing_pct`. Values are arithmetically correct for each column. |
| **Fail** | Missing output column, incorrect `missing_pct` calculation, or incorrect `severity` bucket assignment. |

### P1-T2: `data_profile()` — Severity Buckets

| Field | Detail |
| :--- | :--- |
| **Scenario** | Construct columns with 0%, 20%, 40%, 60%, and 90% missing values. |
| **Expected behavior** | Severity labels: `None`, `Low`, `Moderate`, `High`, `Critical` respectively. |
| **Pass** | Each column maps to its exact bucket label. |
| **Fail** | Any column maps to an incorrect severity bucket. |

### P1-T3: `suspect_numeric_object` Detection

| Field | Detail |
| :--- | :--- |
| **Scenario** | Create a column with dtype `object` where 95% of values are numeric strings (e.g., `"3.5"`, `"12"`) and 5% are `NaN`. |
| **Expected behavior** | `suspect_numeric_object = True` for that column. |
| **Pass** | Flag is `True`. |
| **Fail** | Flag is `False`. |

### P1-T4: Type Recovery Coercion

| Field | Detail |
| :--- | :--- |
| **Scenario** | A column is `object` dtype with 92% numeric-coercible strings and 8% `NaN`. |
| **Expected behavior** | The cleaning step coerces the column to numeric, re-runs `data_profile()`, and updates the dtype. The missing count reflects the previously non-null NaN strings. |
| **Pass** | Column dtype changes from `object` to `float64`. Re-profiled `missing_pct` is correct. |
| **Fail** | Column remains `object` after cleaning, or `data_profile()` is not re-run. |

### P1-T5: Negative Value Flagging — Unconditional, No Per-Column Exception

| Field | Detail |
| :--- | :--- |
| **Scenario** | A numeric column `account_delta` contains values `[-50, 100, 200]`. |
| **Expected behavior** | A soft warning is emitted to the notification center for `account_delta`. No exception is raised. No column mute or per-column exception mechanism exists — flagging is unconditional. |
| **Pass** | Soft warning row in `notifications` for `account_delta`. Pipeline continues. |
| **Fail** | No warning emitted, hard fail raised, or a per-column exception/mute mechanism is invoked. |

### P1-T6: Rare-Category Detection at the 1.5% Threshold

| Field | Detail |
| :--- | :--- |
| **Scenario** | A categorical column has 10,000 rows. Value `"X"` appears 150 times (1.5%). Value `"Y"` appears 149 times (1.49%). Value `"Z"` appears 151 times (1.51%). |
| **Expected behavior** | `"X"` (exactly 1.5%) and `"Y"` (below 1.5%) are flagged and excluded. `"Z"` (above 1.5%) is not flagged. |
| **Pass** | `"X"` and `"Y"` excluded. `"Z"` retained. Soft warning emitted per excluded value with reason logged. |
| **Fail** | `"X"` is not flagged, `"Z"` is flagged, or no reason is logged. |

### P1-T7: IQR Outlier Treatment — Correct Transform Selection

| Field | Detail |
| :--- | :--- |
| **Scenario A** | A column has all positive values (`min = 1`). |
| **Scenario B** | A column has values crossing into negative bounds (`min = -5`). |
| **Expected behavior A** | `log1p` transform applied, then `RobustScaler`. |
| **Expected behavior B** | `Yeo-Johnson` transform applied, then `RobustScaler`. |
| **Pass** | Each scenario uses the correct transform. Transform is applied column-level, not per-value. |
| **Fail** | Wrong transform applied, transform applied per-value instead of column-level, or `RobustScaler` omitted. |

### P1-T8: Missing Value — >50% Column Drop Recommendation

| Field | Detail |
| :--- | :--- |
| **Scenario** | A column has 55% missing values. |
| **Expected behavior** | A hard-fail notification is emitted with a "drop column" recommendation. PM must confirm via `POST /api/v1/datasets/{id}/columns/{col}/confirm-drop` before the pipeline proceeds. |
| **Pass** | Hard-fail row in `notifications`. Column is blocked from downstream processing until PM confirms. |
| **Fail** | Imputation is attempted on the column, or no hard-fail is emitted. |

### P1-T9: Missing Value — PM Imputation Method Application

| Field | Detail |
| :--- | :--- |
| **Scenario** | A numeric column has 20% missing. PM selects `median` imputation via `POST /api/v1/datasets/{id}/columns/{col}/set-imputation`. |
| **Expected behavior** | All missing values in that column are filled with the column's pre-imputation median. No other imputation method is applied. The PM's choice is logged against the column and dataset. |
| **Pass** | Zero NaN values remain. All filled values equal the pre-imputation median. PM's method is stored. |
| **Fail** | NaN values remain, or mean/mode is applied instead of median. |

### P1-T10: Cross-Field Rule — PM Approval Gate

| Field | Detail |
| :--- | :--- |
| **Scenario** | The cross-field validator proposes a rule: `end_date >= start_date`. PM rejects it via `POST /api/v1/datasets/{id}/cross-field-rules/approve` with `{rule_id: X, approved: false}`. |
| **Expected behavior** | The rule is stored with `approved = false` and is NOT activated in the pipeline. |
| **Pass** | No active rule for rule_id X. |
| **Fail** | Rule is activated despite PM rejection. |

### P1-T11: Correlation Pruning — Threshold at 0.9

| Field | Detail |
| :--- | :--- |
| **Scenario** | Column A and Column B have Pearson correlation `r = 0.91`. Column B has lower variance. |
| **Expected behavior** | Column B is dropped. Column A is retained. A log entry records which column was dropped and why. |
| **Pass** | Column B absent from the engineered feature matrix. Log entry present. |
| **Fail** | Column A dropped instead of B, neither dropped, or no log entry. |

### P1-T12: Categorical Encoding Rules

| Field | Detail |
| :--- | :--- |
| **Scenario A** | A categorical column has 3 unique values. |
| **Scenario B** | A categorical column has 7 unique values. |
| **Expected behavior A** | One-hot encoding applied (3 indicator columns created). |
| **Expected behavior B** | Frequency encoding applied (column replaced with proportion of total rows per value). |
| **Pass** | Correct encoding applied to each scenario. Encoding happens after correlation pruning. |
| **Fail** | Encoding order wrong (encoding before pruning), or wrong encoding method applied. |

---

## Phase 2 — Layer 1: Demographic Segmentation

### P2-T1: LLM Field Classifier — Classification Output Format

| Field | Detail |
| :--- | :--- |
| **Scenario** | Pass a sample schema of 10 columns to `field_classifier.py`. |
| **Expected behavior** | The LLM returns a valid JSON list `[{column, classification}]` where each `classification` is exactly one of `"demographic"`, `"behavioral"`, or `"unsure"`. No column is given a name. No segment is assigned. |
| **Pass** | Response parses as JSON. All 10 columns classified. All values in the approved enum. |
| **Fail** | Parse error, missing column, classification outside the enum, or LLM assigns segment names. |

### P2-T2: LLM Classification Is Proposal Only

| Field | Detail |
| :--- | :--- |
| **Scenario** | LLM classifies column `age` as `"demographic"`. |
| **Expected behavior** | This classification is stored as a *proposal*, not an active classification. The active classification is only set after `POST /api/v1/datasets/{id}/field-classifications/confirm` is called by the PM. |
| **Pass** | Column `age` does not enter downstream pipeline until PM confirmation is recorded. |
| **Fail** | Column enters pipeline based on LLM classification alone without PM confirmation. |

### P2-T3: Notification Center — Unsure Column Hard Fail (Dynamic Datasets Only)

| Field | Detail |
| :--- | :--- |
| **Scenario A** | LLM returns `"unsure"` for a column on a *dynamic* (DB-connected) dataset. |
| **Scenario B** | LLM returns `"unsure"` for a column on a *permanent* (file-upload) dataset. |
| **Expected behavior A** | Hard-fail emitted. Column blocked from downstream pipeline until PM resolves. |
| **Expected behavior B** | No hard-fail. The permanent dataset path does not route through the notification center for schema changes. |
| **Pass** | Hard-fail present for A; none for B. |
| **Fail** | Hard-fail absent for A, or hard-fail incorrectly emitted for B. |

### P2-T4: Month-over-Month Percentage Change Computation

| Field | Detail |
| :--- | :--- |
| **Scenario** | Segment `"Age 25-34"` has size 1000 at period T-1 and 1100 at period T. |
| **Expected behavior** | MoM change = `(1100 - 1000) / 1000 * 100 = 10.0%`. Persisted to `demographic_segment_metrics`. |
| **Pass** | Computed value equals `10.0`. Row exists in `demographic_segment_metrics` for the correct period. |
| **Fail** | Incorrect calculation, missing row, or wrong period_id. |

### P2-T5: Volatility-Aware Flagging — Fires Above Noise Band, Suppressed Below

| Field | Detail |
| :--- | :--- |
| **Scenario A** | A volatile segment has a historical 2-sigma noise band of ±25%. A 20% MoM change occurs. |
| **Scenario B** | A stable segment has a historical 2-sigma noise band of ±3%. A 20% MoM change occurs. |
| **Expected behavior A** | No flag. Change is within the segment's noise band. |
| **Expected behavior B** | Soft warning emitted. Change exceeds the 2-sigma band for this specific segment. |
| **Pass** | No notification for A; soft warning for B. |
| **Fail** | False positive for A, or false negative for B. |

### P2-T6: Minimum Segment Size Gate

| Field | Detail |
| :--- | :--- |
| **Scenario** | A demographic segment has N = 15 customers. A 33% MoM change occurs (5 customers joining). |
| **Expected behavior** | Percentage-change metric is not computed for this segment. The segment is below the minimum size threshold (N ≥ 30). Absolute change tracking or pooling into "Other" is applied instead. |
| **Pass** | No percentage change metric logged for this segment. No false anomaly flag raised. |
| **Fail** | Percentage change metric computed, leading to a false-positive anomaly flag. |

### P2-T7: `demographic_snapshots` — Typed Columnar Schema, Not EAV

| Field | Detail |
| :--- | :--- |
| **Scenario** | PM confirms demographic fields `[age, gender, country]` via `POST /api/v1/datasets/{id}/field-classifications/confirm`. |
| **Expected behavior** | The `demographic_snapshots` table (or an `ALTER TABLE` migration executed immediately after PM confirmation) generates actual typed columns — `age INTEGER`, `gender TEXT`, `country TEXT` — rather than generic `field_name` / `field_value` rows. Querying the table returns one row per customer per period, with each confirmed field as its own typed column. |
| **Pass** | Table schema contains `age`, `gender`, `country` as distinct typed columns. No `field_name` or `field_value` columns present. |
| **Fail** | Table uses a `field_name` / `field_value` EAV pattern, or PM-confirmed fields are not reflected as actual typed columns in the schema. |

---

## Phase 3 — Layer 2: Behavioral Clustering Pipeline

### P3-T1: KMeans — Deterministic PCA Applied

| Field | Detail |
| :--- | :--- |
| **Scenario** | Inspect `clusterer.py` source code and execution trace. |
| **Expected behavior** | StandardScaler is fit, PCA is applied (explained variance >= 0.85) and KMeans is fit. No KMeans import or call exists in the clustering path. |
| **Pass** | No UMAP import or invocation found in the clustering path. |
| **Fail** | KMeans is used, or PCA/KMeans is missing, or UMAP is used. |

### P3-T2: KMeans — Cluster Assignment Persistence

| Field | Detail |
| :--- | :--- |
| **Scenario** | Run the clustering pipeline on the test dataset. |
| **Expected behavior** | Every row in the feature matrix receives a cluster label (assigned to one of the clusters). Each assignment is persisted to `cluster_assignments` with a valid `customer_id` and `period_id`. |
| **Pass** | Row count in `cluster_assignments` equals row count in the feature matrix. No null `cluster_label`. |
| **Fail** | Rows missing from `cluster_assignments`, or `period_id` is null. |

### P3-T3: Quality Gate — Stability (ARI)

| Field | Detail |
| :--- | :--- |
| **Scenario A** | Simulated clustering with ARI = 0.80 (above threshold of 0.75). |
| **Scenario B** | Simulated clustering with ARI = 0.70 (below threshold of 0.75). |
| **Expected behavior A** | Stability gate passes. |
| **Expected behavior B** | Stability gate fails. Hard-fail emitted. Downstream pipeline blocked. |
| **Pass** | A passes, B blocks. Failure message identifies which gate failed. |
| **Fail** | B passes, or A blocks. |

### P3-T4: Quality Gate — Separation (Silhouette & DBI)

| Field | Detail |
| :--- | :--- |
| **Scenario A** | Silhouette = 0.25 (above 0.2), DBI = 1.2 (below 1.5). |
| **Scenario B** | Silhouette = 0.15 (below 0.2). |
| **Scenario C** | DBI = 1.8 (above 1.5). |
| **Expected behavior** | A passes both gates. B and C each fail individually. |
| **Pass** | A passes, B and C each emit a hard-fail identifying the failing metric. |
| **Fail** | B or C passes, or hard-fail message does not identify which metric failed. |

### P3-T5: Quality Gate — Noise Sanity (Always Passes)

| Field | Detail |
| :--- | :--- |
| **Scenario A** | Noise fraction = 0.25 (below threshold of 0.30). |
| **Scenario B** | Noise fraction = 0.35 (above threshold of 0.30). |
| **Expected behavior A** | Noise sanity gate passes. |
| **Expected behavior B** | Hard-fail emitted. Downstream blocked. |
| **Pass** | A passes, B blocks with hard-fail. |
| **Fail** | B passes or A blocks. |

### P3-T6: Quality Gate — Config-Driven Thresholds (Not Hardcoded)

| Field | Detail |
| :--- | :--- |
| **Scenario** | Change `CLUSTER_STABILITY_ARI_THRESHOLD` to `0.60` in the environment. Re-run the evaluator against a clustering with ARI = 0.65. |
| **Expected behavior** | Gate passes under the new threshold without any code change. |
| **Pass** | Gate passes. |
| **Fail** | Gate fails (indicating the threshold was hardcoded), or an exception is raised reading the env var. |

### P3-T7: LLM Profiling — No Cluster Name Assigned by LLM

| Field | Detail |
| :--- | :--- |
| **Scenario** | Call `cluster_profiler.py` for a cluster. Inspect the LLM output. |
| **Expected behavior** | LLM output is a descriptive summary of behavioral feature averages. It does not include a proposed name, label, or title for the cluster. |
| **Pass** | LLM output is parseable and contains no field `name`, `label`, or equivalent. |
| **Fail** | LLM output includes a cluster name. |

### P3-T8: PM Naming Endpoint Persists Label

| Field | Detail |
| :--- | :--- |
| **Scenario** | Call `POST /api/v1/clusters/{id}/name` with body `{name: "High-Value Loyalists"}`. |
| **Expected behavior** | `cluster_label = "High-Value Loyalists"` in the `clusters` table for that cluster ID. |
| **Pass** | Correct label stored. |
| **Fail** | Label not persisted, or LLM-generated name stored instead. |

### P3-T9: Quality Gate Failure — No Silent Algorithm Swapping

| Field | Detail |
| :--- | :--- |
| **Scenario** | Feed the clusterer a dataset that fails separation or stability quality gates. |
| **Expected behavior** | A hard-fail is emitted. The pipeline blocks. The system does not silently swap algorithms. A hyperparameter sweep over `n_clusters` (from 2 to 6) is initiated before any further action. |
| **Pass** | Hard-fail emitted. Hyperparameter sweep logs results. No algorithm swap in execution trace. |
| **Fail** | Pipeline continues with failing gates without blocking, or silently swaps algorithms. |

### P3-T10: Drift Evaluator — Adopt Path

| Field | Detail |
| :--- | :--- |
| **Scenario** | Simulate a candidate re-clustering where medoid shift = 3× the stored `intra_cluster_spread` (above the 2× threshold). |
| **Expected behavior** | Adopt decision made. New `cluster_baselines` record inserted with `is_active = true`. Old baseline set to `is_active = false`. `cluster_identity_map` rows created. Layer 4 retraining triggered. Soft warning listing any splits/merges emitted. |
| **Pass** | All four outcomes verified: new baseline active, old inactive, identity map populated, retraining triggered. |
| **Fail** | Hold decision made instead, or any of the four outcomes is missing. |

### P3-T11: Drift Evaluator — Hold Path & KNN Classifier

| Field | Detail |
| :--- | :--- |
| **Scenario** | Simulate a candidate re-clustering where medoid shift = 1.5× the `intra_cluster_spread` (below the 2× threshold). |
| **Expected behavior** | Hold decision made. Current baseline unchanged. KNN classifier (K=5) trained on current baseline. A new data point can be assigned to the nearest cluster without rerunning KMeans. |
| **Pass** | Current baseline's `is_active` remains `true`. KNN assigns new point to the correct cluster (verified against ground truth). |
| **Fail** | Adopt fires below 2× threshold, or KNN is not trained on hold. |

### P3-T12: Medoid Sampling Cap on Large Clusters

| Field | Detail |
| :--- | :--- |
| **Scenario** | A cluster contains 15,000 rows. |
| **Expected behavior** | Before computing the medoid, the cluster is randomly sampled down to max 2,000 points. The medoid is computed on the sample, not the full cluster. Maximum distance matrix dimensions are 2000 × 2000. |
| **Pass** | No `MemoryError`. Medoid computed successfully. Log confirms sample size ≤ 2,000. |
| **Fail** | `ArrayMemoryError` raised, or the full N×N matrix is allocated. |

---

## Phase 4 — Layer 3: Statistical Testing

### P4-T1: ANOVA Output Correctness

| Field | Detail |
| :--- | :--- |
| **Scenario** | Three groups with known means (10, 20, 30) and equal variance. Reference p-value computed independently from `scipy.stats.f_oneway`. |
| **Expected behavior** | `statistical_tester.py` returns a p-value matching the reference within floating-point tolerance (< 1e-6 difference). Effect size (eta-squared) also returned. |
| **Pass** | Values match within tolerance. Effect size field present. |
| **Fail** | p-value mismatch, or effect size not computed. |

### P4-T2: Kruskal-Wallis Output Correctness

| Field | Detail |
| :--- | :--- |
| **Scenario** | Three non-normal groups with known ranks. Reference p-value from `scipy.stats.kruskal`. |
| **Expected behavior** | Same correctness contract as P4-T1 for the non-parametric path. |
| **Pass** | p-value matches reference within tolerance. |
| **Fail** | Mismatch. |

### P4-T3: Anti-Circularity Check — Behavioral Feature Rejected

| Field | Detail |
| :--- | :--- |
| **Scenario** | Pass `login_frequency` (a behavioral clustering feature) as the test variable. |
| **Expected behavior** | A `ValueError` (or equivalent hard error) is raised before any test runs. The error message identifies the column as a clustering input that cannot be used as a test variable. |
| **Pass** | `ValueError` raised. Test does not execute. |
| **Fail** | Test executes using the clustering input feature (circular result). |

### P4-T4: Anti-Circularity Check — Outcome Variable Accepted

| Field | Detail |
| :--- | :--- |
| **Scenario** | Pass `churn_rate` (from `segment_outcomes`) as the test variable. |
| **Expected behavior** | Test runs without error. Result is persisted to `statistical_test_results`. |
| **Pass** | Test completes and result row exists in `statistical_test_results`. |
| **Fail** | `ValueError` raised for a valid outcome variable. |

### P4-T4b: Anti-Circularity Check — Snapshot Fallback Excludes Reconstructed Proxies

| Field | Detail |
| :--- | :--- |
| **Scenario** | Dataset lacks raw confirmed CSV and falls back to `behavioral_snapshots`. Pass reconstructed `revenue` or `repeat_purchase_prob` as the test variable in Layer 3. |
| **Expected behavior** | `ValueError` raised. Layer 3 statistical testing strictly permits only `churn_rate` and `retention_rate` under snapshot fallback because `revenue` and `repeat_purchase_prob` were derived from variables that may have participated in clustering. |
| **Pass** | `ValueError` raised for `revenue` and `repeat_purchase_prob` under fallback; `churn_rate` and `retention_rate` succeed. |
| **Fail** | Reconstructed proxies accepted for Layer 3 statistical testing under snapshot fallback. |

### P4-T5: Effect Size Enforcement (Large-Sample Bias Guard)

| Field | Detail |
| :--- | :--- |
| **Scenario** | Two groups of N = 25,000 each with a trivially small mean difference (Δ = 0.01 units). The resulting p-value is < 0.001, but Cohen's d = 0.004 (negligible). |
| **Expected behavior** | Even though p < 0.001, the `significant` flag is `False` because effect size is below the minimum meaningful threshold. The result is not surfaced as a diverging segment pair. |
| **Pass** | `significant = False`. Effect size reported alongside p-value. |
| **Fail** | `significant = True` based solely on p-value. |

### P4-T6: Critical Similarity Flag

| Field | Detail |
| :--- | :--- |
| **Scenario** | Segment pair with p-value = 0.82 (failing to reject equality). |
| **Expected behavior** | This pair is flagged as approaching critical similarity in `statistical_test_results`. |
| **Pass** | Flag present in the output row for this pair. |
| **Fail** | Flag absent. |

---

## Phase 5 — Layer 6: Segment vs. Outcome Analysis

### P5-T1: Churn + Retention Sum to 100%

| Field | Detail |
| :--- | :--- |
| **Scenario** | Closed-population scenario: all T customers are accounted for at T+1 (either retained or churned, no new entrants). |
| **Expected behavior** | For each cluster, `churn_rate + retention_rate = 100.0%`. |
| **Pass** | Arithmetic check passes for all clusters. |
| **Fail** | Sum deviates by more than 0.01% for any cluster. |

### P5-T2: All Four Metrics Persisted

| Field | Detail |
| :--- | :--- |
| **Scenario** | Run outcome computation for a cluster/period pair. |
| **Expected behavior** | A row in `segment_outcomes` contains non-null values for `churn_rate`, `retention_rate`, `revenue`, and `repeat_purchase_prob`. |
| **Pass** | All four columns non-null. |
| **Fail** | Any column is null or absent. |

### P5-T3: API Returns Correct Outcome Data

| Field | Detail |
| :--- | :--- |
| **Scenario** | Call `GET /api/v1/segments/{cluster_id}/outcomes?period_id={id}` for a known cluster/period with pre-seeded values. |
| **Expected behavior** | Response body matches the values stored in `segment_outcomes` exactly. |
| **Pass** | All four metrics in response match DB values. HTTP 200. |
| **Fail** | Mismatch, missing field, or non-200 status. |

---

## Phase 6 — Layer 4: Predictive Segmentation

### P6-T1: No Data Leakage — Temporal Feature Boundary

| Field | Detail |
| :--- | :--- |
| **Scenario** | Inspect `dataset_builder.py`: verify the feature extraction query filters by `period_id = T` strictly and the target extraction uses `period_id = T+1` strictly. |
| **Expected behavior** | Feature columns carry no field computed after period T. Target labels carry no field from before period T+1. |
| **Pass** | No T+1 column present in the feature matrix. No T column present in the target. |
| **Fail** | Any temporal leakage detected. |

### P6-T2: Migration Label Correctness

| Field | Detail |
| :--- | :--- |
| **Scenario** | Customer C is in Cluster A at T and Cluster B at T+1 (matched via identity map). Customer D is in Cluster A at both T and T+1. |
| **Expected behavior** | `migrated = True` for C. `migrated = False` for D. |
| **Pass** | Binary labels correct for both customers. |
| **Fail** | Either label incorrect. |

### P6-T3: Model Selection by PR-AUC on Minority Class

| Field | Detail |
| :--- | :--- |
| **Scenario** | Train all four candidate models. XGBoost: PR-AUC = 0.72, ROC-AUC = 0.81. Random Forest: PR-AUC = 0.61, ROC-AUC = 0.88. |
| **Expected behavior** | XGBoost is selected as the winning model because it has the higher PR-AUC on the minority class (`migrated = True`), not the higher overall ROC-AUC. |
| **Pass** | XGBoost artifact persisted. Log confirms model selection basis = PR-AUC. |
| **Fail** | Random Forest selected because of higher overall ROC-AUC. |

### P6-T4: Layer 2.4 Adopt Triggers Layer 4 Retraining

| Field | Detail |
| :--- | :--- |
| **Scenario** | Simulate a Layer 2.4 Adopt event. |
| **Expected behavior** | `retrain_trigger.py` is invoked. A new model training run begins against the new cluster baseline. A new model artifact is saved with a creation timestamp after the adopt event. |
| **Pass** | Retraining job starts. New model artifact timestamp is after the adopt event. |
| **Fail** | Retraining not triggered, or old artifact is reused without retraining. |

---

## Phase 7 — Layer 5: Segment Migration Forecasting

### P7-T1: Transition Matrix Row Sums to 1.0

| Field | Detail |
| :--- | :--- |
| **Scenario** | Compute the transition matrix for a known set of customer migration labels. |
| **Expected behavior** | For every source cluster, the sum of all outgoing probabilities (including self-transition) equals 1.0. |
| **Pass** | All rows sum to 1.0 ± 1e-6. |
| **Fail** | Any row sum deviates by more than 1e-6. |

### P7-T2: Disappeared Cluster Handling

| Field | Detail |
| :--- | :--- |
| **Scenario** | Cluster X exists at T but is absent at T+1 (all members migrated, or the cluster disappeared during drift evaluation). |
| **Expected behavior** | The row for Cluster X either sums to 0 or is absent entirely. No division-by-zero error. |
| **Pass** | No error. Row either absent or sums to 0. |
| **Fail** | Division-by-zero exception, or row is present with non-zero invalid probabilities. |

### P7-T3: Self-Transition Probabilities Included

| Field | Detail |
| :--- | :--- |
| **Scenario** | 80% of Cluster A members remain in Cluster A at T+1. |
| **Expected behavior** | `P(A → A) = 0.80` is stored in the `transition_matrix` table. |
| **Pass** | Cell value = 0.80. |
| **Fail** | Self-transition cell missing or value incorrect. |

---

## Phase 8 — Layer 8: Feature-Level Migration Signal

### P8-T1: Signal Raised When >50% of Columns Drift

| Field | Detail |
| :--- | :--- |
| **Scenario** | 5 PM-confirmed behavioral columns. Customer C's values are closer to the comparison cluster's median on 3 of the 5 columns. |
| **Expected behavior** | `drift_fraction = 3/5 = 0.60`. `signal_raised = True`. |
| **Pass** | `signal_raised = True` in `feature_migration_signals` for customer C. |
| **Fail** | `signal_raised = False` or `drift_fraction` computed incorrectly. |

### P8-T2: Signal NOT Raised When ≤50% of Columns Drift

| Field | Detail |
| :--- | :--- |
| **Scenario** | 5 PM-confirmed behavioral columns. Customer D's values are closer to the comparison cluster's median on exactly 2 of the 5 columns. |
| **Expected behavior** | `drift_fraction = 2/5 = 0.40`. `signal_raised = False`. |
| **Pass** | `signal_raised = False`. |
| **Fail** | `signal_raised = True` (off-by-one or threshold boundary error). |

### P8-T3: Boundary Case — Exactly 50% Is Not a Signal

| Field | Detail |
| :--- | :--- |
| **Scenario** | 4 PM-confirmed behavioral columns. Customer E's values are closer to the comparison cluster on exactly 2 of the 4 columns (`drift_fraction = 0.50`). |
| **Expected behavior** | `signal_raised = False`. The rule is strictly `> 50%`, not `≥ 50%`. |
| **Pass** | `signal_raised = False`. |
| **Fail** | `signal_raised = True` at exactly 50%. |

### P8-T4: Median Used, Not Mean

| Field | Detail |
| :--- | :--- |
| **Scenario** | A cluster column has values `[1, 2, 3, 4, 1000]`. Median = 3. Mean = 202. Customer value = 4. |
| **Expected behavior** | Distance is computed as `\|customer_value - median\| = \|4 - 3\| = 1`, using the **median**, not the mean (`\|4 - 202\| = 198`). This confirms Layer 8's design principle: median is used specifically to avoid outlier sensitivity, since outliers are already handled in Layer 0. |
| **Pass** | Computed distance = 1. |
| **Fail** | Computed distance = 198 (mean used instead of median). |

### P8-T5: PM Column Confirmation Gate

| Field | Detail |
| :--- | :--- |
| **Scenario** | Layer 8 signal computation is attempted before `POST /api/v1/datasets/{id}/layer8-columns/confirm` is called. |
| **Expected behavior** | Hard-fail emitted. Layer 8 signal computation blocked until PM confirms the column set. |
| **Pass** | Hard-fail emitted. No signals computed. |
| **Fail** | Signals computed using a default or unconfirmed column set. |

---

## Phase 9 — Layer 7: Metric Integration (Backend API)

### P9-T1: Comparison API Returns Correct Matrix

| Field | Detail |
| :--- | :--- |
| **Scenario** | Call `GET /api/v1/metrics/comparison?granularity=monthly&metric=churn_rate&segment_ids=1,2&period_range=P1,P3` with pre-seeded values in `segment_outcomes`. |
| **Expected behavior** | Response contains a 2×3 matrix (2 segments × 3 periods) with values matching the seeded data exactly. |
| **Pass** | All 6 cells match seeded values. HTTP 200. |
| **Fail** | Any cell mismatch, missing cell, or wrong granularity applied. |

### P9-T2: Gap Row Present for Exactly 2 Segments

| Field | Detail |
| :--- | :--- |
| **Scenario** | Call the comparison API with `segment_ids=1,2`. |
| **Expected behavior** | Response includes a `gap` row where each cell = `segment_1_value - segment_2_value`. Directional sign (`+` / `-`) is included. Gap row is computed in the backend — not derived by the frontend. |
| **Pass** | Gap row present. Values arithmetically correct. Sign correct. |
| **Fail** | Gap row absent, or values incorrect. |

### P9-T3: Gap Row Absent for 3+ Segments

| Field | Detail |
| :--- | :--- |
| **Scenario** | Call the comparison API with `segment_ids=1,2,3`. |
| **Expected behavior** | Response contains raw segment rows only. No gap row. |
| **Pass** | Response has exactly 3 rows. No gap row. |
| **Fail** | Gap row present for 3 segments. |

### P9-T6: Cluster Lineage Resolution Across Adopted Baselines

| Field | Detail |
| :--- | :--- |
| **Scenario** | Call the comparison API with `segment_ids` when a new baseline has been adopted in Layer 2.4 and cluster IDs have changed across periods. |
| **Expected behavior** | Historical period outcomes are resolved via `cluster_identity_map` (`match_type = 'matched'`). Period metrics are presented as observed period outcomes without 'Forecast' labeling. Un-annualized period churn is labeled as `Churn Rate (%)`. |
| **Pass** | Complete longitudinal time series returned across baseline boundaries. HTTP 200. |
| **Fail** | Missing values for periods prior to adoption or unmapped cluster IDs. |

---

## Phase 10 — AI Agents

### P10-T1: Agent 1 Narration Generated After Quarterly Batch

| Field | Detail |
| :--- | :--- |
| **Scenario** | Run a simulated quarterly batch to completion (all 8 steps). |
| **Expected behavior** | `insight_narrator.py` is invoked at the end of the batch. A narration text is stored in `quarterly_narrations` for the completed period. |
| **Pass** | Row exists in `quarterly_narrations` with non-empty text and correct `period_id`. |
| **Fail** | No narration generated, or narration stored with wrong `period_id`. |

### P10-T2: Agent 1 — Causal Language Evaluator Rejects Unsupported Claims

| Field | Detail |
| :--- | :--- |
| **Scenario** | Inject a mock LLM response containing: `"Churn increased because session duration declined."` |
| **Expected behavior** | `narration_evaluator.py` detects the causal language. The narration is rejected and regeneration is triggered. The final stored narration does not contain causal language, rephrasing as association. |
| **Pass** | Original narration rejected. Regeneration triggered. Final narration stored with association language (e.g. "coincided with"). |
| **Fail** | Original narration with causal language stored as-is. |

### P10-T2b: Layer 3 Significance Does NOT Establish Causation

| Field | Detail |
| :--- | :--- |
| **Scenario** | Layer 3 pairwise statistical test yields $p < 0.05$. Mock LLM response uses causal claims: `"Churn increased because engagement fell."` |
| **Expected behavior** | `narration_evaluator.py` strictly rejects `because`, `caused by`, and `due to` regardless of $p < 0.05$, because observational statistical significance does not establish causality. Narration is rewritten as association: `"Higher churn coincided with lower engagement"` or `"The segments differed significantly in churn"`. |
| **Pass** | Causal terms rejected despite $p < 0.05$. Association phrasing accepted. |
| **Fail** | Causal language accepted on the basis of Layer 3 significance. |

### P10-T3: Agent 1 — Intensifier Guard

| Field | Detail |
| :--- | :--- |
| **Scenario** | Mock LLM response contains: `"Revenue grew significantly."` The revenue change is within the historical volatility band (no Layer 3 backing). |
| **Expected behavior** | Evaluator detects `"significantly"` without statistical backing. Warning or regeneration is triggered. Final narration does not contain the unsupported intensifier. (Note: `"differed significantly"` is acceptable when supported by Layer 3). |
| **Pass** | Intensifier flagged when unsupported. Narration without unsupported intensifier stored. |
| **Fail** | Narration with unsupported intensifier stored unmodified. |

### P10-T4: Agent 2 — Valid Query Routed Correctly

| Field | Detail |
| :--- | :--- |
| **Scenario** | `POST /api/v1/query` with body `{query: "What was the churn rate for Segment 'High-Value Loyalists' in Q3 2024?"}`. The segment and period are resolvable from the database. |
| **Expected behavior** | Agent extracts `segment_id` and `time_period`. Routes to Layer 6 outcome API. Returns the churn rate value for that segment and period. |
| **Pass** | HTTP 200. Response contains `churn_rate` for the correct segment and period. |
| **Fail** | Hard stop returned for a resolvable query, or wrong data returned. |

### P10-T5: Agent 2 — Out-of-Scope Query Returns Hard Stop

| Field | Detail |
| :--- | :--- |
| **Scenario** | `POST /api/v1/query` with body `{query: "Which customers are most likely to become VIPs?"}`. No segment reference or time period present. |
| **Expected behavior** | Agent returns the exact hard-stop response: `"This query is outside the queryable space."` No approximation, no partial result, no custom rendering. |
| **Pass** | Response body is the exact hard-stop message. No data returned. |
| **Fail** | Agent attempts to answer or returns partial data. |

### P10-T6: Agent 2 — Segment Bound But No Time Period Returns Hard Stop

| Field | Detail |
| :--- | :--- |
| **Scenario** | `POST /api/v1/query` with body `{query: "Show me data for Segment 3."}`. No time period specified. |
| **Expected behavior** | Agent returns a hard-stop response because the query is not time-bound. No assumed default period is used. |
| **Pass** | Hard-stop returned. |
| **Fail** | Query routed using a default or assumed time period. |

### P10-T7: Agent 2 — No Custom Rendering

| Field | Detail |
| :--- | :--- |
| **Scenario** | Inspect the `nl_query_agent.py` codebase. |
| **Expected behavior** | The agent does not generate new tables, custom charts, visualizations, or layout configurations. It only returns existing layer outputs filtered to the requested criteria. |
| **Pass** | No rendering logic, chart generation, or layout manipulation code present in `nl_query_agent.py`. |
| **Fail** | Custom rendering code found. |

---

## Cross-Cutting Eval Scenarios

### CC-T1: Config — No Hardcoded Threshold Values

| Field | Detail |
| :--- | :--- |
| **Scenario** | Search the codebase for magic numbers corresponding to the 10 config parameters defined in `implementation_plan.md` (e.g., `0.015`, `0.9`, `5`, `2`, `0.75`, `0.2`, `1.5`, `0.30`). |
| **Expected behavior** | None of these values appear as hardcoded literals in business logic source code. All are loaded from environment variables or a config module. |
| **Pass** | No raw numeric literals found for these parameters in conditional branches or computations. |
| **Fail** | Any threshold value is hardcoded. |

### CC-T2: Quarterly Batch — Execution Order

| Field | Detail |
| :--- | :--- |
| **Scenario** | Run the quarterly batch runner with execution tracing enabled. |
| **Expected behavior** | Steps execute in the exact sequence: (1) collect data → (2) Layer 0 re-profile → (3) Layer 2.4 drift evaluation → (4a/4b) Layer 4 retrain or label generation → (5) Layer 6 outcomes → (6) Layer 3 tests → (7) Layer 8 signals → (8) Agent 1 narration. |
| **Pass** | Execution trace confirms the correct order. No step runs before its dependencies complete. |
| **Fail** | Any step executes out of order. |

### CC-T3: LLM Governing Principle — LLM Never Makes Decisions

| Field | Detail |
| :--- | :--- |
| **Scenario** | Review all LLM call sites: `field_classifier.py`, `cluster_profiler.py`, `cross_field_validator.py`, `insight_narrator.py`, `nl_query_agent.py`. |
| **Expected behavior** | Every LLM output is a *proposal* awaiting PM confirmation, a *description* presented to the PM, or a *retrieval* of existing data. No LLM output is directly committed to the database or used in pipeline logic without a PM gate or code-level validation step. |
| **Pass** | All LLM outputs pass through a PM confirmation endpoint or a validation layer before being stored or acted upon. |
| **Fail** | Any LLM output directly commits to the database or controls pipeline branching without a PM gate. |

### CC-T4: Structured Logging on All Pipeline Steps

| Field | Detail |
| :--- | :--- |
| **Scenario** | Run any pipeline step. Inspect the log output. |
| **Expected behavior** | Log output is valid JSON. Each entry contains `layer_name`, `dataset_id`, `timestamp`, and `status` (one of `started`, `completed`, `error`). Error entries include an error message. |
| **Pass** | All entries parse as JSON. All four fields present per entry. |
| **Fail** | Log output is unstructured, or any required field is missing. |

---

## Eval Status Tracker

| Phase | ID | Scenario | Status |
| :--- | :--- | :--- | :--- |
| 0 | P0-T1 | Docker & DB startup | ⬜ Not run |
| 0 | P0-T2 | Period ID constraint on all snapshot tables | ⬜ Not run |
| 0 | P0-T3 | Anthropic Claude smoke test | ⬜ Not run |
| 0 | P0-T4 | Hard fail emission & retrieval | ⬜ Not run |
| 0 | P0-T5 | Soft warning does not block | ⬜ Not run |
| 1 | P1-T1 | `data_profile()` output shape | ⬜ Not run |
| 1 | P1-T2 | Severity buckets | ⬜ Not run |
| 1 | P1-T3 | `suspect_numeric_object` detection | ⬜ Not run |
| 1 | P1-T4 | Type recovery coercion | ⬜ Not run |
| 1 | P1-T5 | Negative value flagging — unconditional | ⬜ Not run |
| 1 | P1-T6 | Rare-category 1.5% threshold | ⬜ Not run |
| 1 | P1-T7 | IQR transform selection | ⬜ Not run |
| 1 | P1-T8 | >50% missing — hard fail | ⬜ Not run |
| 1 | P1-T9 | PM imputation method application | ⬜ Not run |
| 1 | P1-T10 | Cross-field PM approval gate | ⬜ Not run |
| 1 | P1-T11 | Correlation pruning at 0.9 | ⬜ Not run |
| 1 | P1-T12 | Categorical encoding rules | ⬜ Not run |
| 2 | P2-T1 | LLM classifier output format | ⬜ Not run |
| 2 | P2-T2 | LLM classification is proposal only | ⬜ Not run |
| 2 | P2-T3 | Unsure column hard fail routing (dynamic vs. permanent) | ⬜ Not run |
| 2 | P2-T4 | MoM percentage change computation | ⬜ Not run |
| 2 | P2-T5 | Volatility-aware flagging | ⬜ Not run |
| 2 | P2-T6 | Minimum segment size gate (N ≥ 30) | ⬜ Not run |
| 2 | P2-T7 | `demographic_snapshots` typed-columnar schema — not EAV | ⬜ Not run |
| 3 | P3-T1 | No UMAP applied | ⬜ Not run |
| 3 | P3-T2 | Cluster assignment persistence | ⬜ Not run |
| 3 | P3-T3 | Quality gate: stability (ARI) | ⬜ Not run |
| 3 | P3-T4 | Quality gate: separation (Silhouette & DBI) | ⬜ Not run |
| 3 | P3-T5 | Quality gate: noise sanity | ⬜ Not run |
| 3 | P3-T6 | Config-driven thresholds | ⬜ Not run |
| 3 | P3-T7 | LLM no cluster name assigned | ⬜ Not run |
| 3 | P3-T8 | PM naming endpoint persists label | ⬜ Not run |
| 3 | P3-T9 | Noise collapse — no KMeans fallback | ⬜ Not run |
| 3 | P3-T10 | Drift evaluator — adopt path | ⬜ Not run |
| 3 | P3-T11 | Drift evaluator — hold path & KNN | ⬜ Not run |
| 3 | P3-T12 | Medoid sampling cap on large clusters | ⬜ Not run |
| 4 | P4-T1 | ANOVA output correctness | ⬜ Not run |
| 4 | P4-T2 | Kruskal-Wallis output correctness | ⬜ Not run |
| 4 | P4-T3 | Anti-circularity — behavioral feature rejected | ⬜ Not run |
| 4 | P4-T4 | Anti-circularity — outcome variable accepted | ⬜ Not run |
| 4 | P4-T5 | Effect size enforcement (large-sample bias guard) | ⬜ Not run |
| 4 | P4-T6 | Critical similarity flag | ⬜ Not run |
| 5 | P5-T1 | Churn + retention sum to 100% | ⬜ Not run |
| 5 | P5-T2 | All five outcome metrics persisted | ⬜ Not run |
| 5 | P5-T3 | Outcome API response correctness | ⬜ Not run |
| 6 | P6-T1 | No temporal data leakage | ⬜ Not run |
| 6 | P6-T2 | Migration label correctness | ⬜ Not run |
| 6 | P6-T3 | Model selection by PR-AUC | ⬜ Not run |
| 6 | P6-T4 | Adopt event triggers Layer 4 retraining | ⬜ Not run |
| 7 | P7-T1 | Transition matrix rows sum to 1.0 | ⬜ Not run |
| 7 | P7-T2 | Disappeared cluster handling | ⬜ Not run |
| 7 | P7-T3 | Self-transition probabilities included | ⬜ Not run |
| 8 | P8-T1 | Signal raised when >50% columns drift | ⬜ Not run |
| 8 | P8-T2 | Signal not raised when ≤50% columns drift | ⬜ Not run |
| 8 | P8-T3 | Boundary case — exactly 50% is not a signal | ⬜ Not run |
| 8 | P8-T4 | Median used, not mean | ⬜ Not run |
| 8 | P8-T5 | PM column confirmation gate | ⬜ Not run |
| 9 | P9-T1 | Comparison API returns correct matrix | ⬜ Not run |
| 9 | P9-T2 | Gap row present for exactly 2 segments | ⬜ Not run |
| 9 | P9-T3 | No gap row for 3+ segments | ⬜ Not run |
| 10 | P10-T1 | Agent 1 narration generated after batch | ⬜ Not run |
| 10 | P10-T2 | Causal language evaluator rejects | ⬜ Not run |
| 10 | P10-T3 | Intensifier guard | ⬜ Not run |
| 10 | P10-T4 | Agent 2 valid query routed correctly | ⬜ Not run |
| 10 | P10-T5 | Agent 2 out-of-scope query — hard stop | ⬜ Not run |
| 10 | P10-T6 | Agent 2 no time period — hard stop | ⬜ Not run |
| 10 | P10-T7 | Agent 2 no custom rendering | ⬜ Not run |
| CC | CC-T1 | No hardcoded threshold values | ⬜ Not run |
| CC | CC-T2 | Quarterly batch execution order | ⬜ Not run |
| CC | CC-T3 | LLM never makes decisions | ⬜ Not run |
| CC | CC-T4 | Structured logging on all pipeline steps | ⬜ Not run |
