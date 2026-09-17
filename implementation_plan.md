# Customer Segmentation Studio — Phase-Wise Implementation Plan

> **Source documents:** [architecture.md](file:///c:/Projects/Segment%20studio/architecture.md) · [context.md](file:///c:/Projects/Segment%20studio/context.md)
>
> **LLM Provider:** Anthropic Claude (API key configured in `.env`)

> [!CAUTION]
> **UI Design Gate has been lifted.** The PM wireframes have been completed, signed off, and implemented. The dashboard UI is served directly at the root `/` endpoint of the FastAPI app from `Segmentation Studio Wireframes.html`.

---

## Phase Summary

| Phase | Name | Layers / Components | UI Gate Blocked? | Key Deliverable |
| :--- | :--- | :--- | :--- | :--- |
| **0** | [Project Foundation & Infrastructure](#phase-0--project-foundation--infrastructure) | Infra, DB, LLM client | No | Postgres schema, Docker, Anthropic client |
| **1** | [Data Ingestion & Layer 0](#phase-1--data-ingestion--layer-0-data-preparation-foundation) | ARD-1, Layer 0 | Partial | Profiling, cleaning, validation, feature engineering |
| **2** | [Layer 1 — Demographic Segmentation](#phase-2--layer-1-demographic-segmentation) | Layer 1 | Partial | LLM field classifier, volatility-aware flagging |
| **3** | [Layer 2 — Behavioral Clustering Pipeline](#phase-3--layer-2-behavioral-clustering-pipeline) | Layer 2.0–2.4 | No | KMeans pipeline, drift evaluation, adopt/hold cycle |
| **4** | [Layer 3 — Statistical Testing](#phase-4--layer-3-statistical-testing-layer) | Layer 3 | No | Statistical significance flags |
| **5** | [Layer 6 — Segment vs. Outcome Analysis](#phase-5--layer-6-segment-vs-outcome-analysis) | Layer 6 | No | Outcome metric computation |
| **6** | [Layer 4 — Predictive Segmentation](#phase-6--layer-4-predictive-segmentation) | Layer 4 | No | Binary migration labels, trained classifier |
| **7** | [Layer 5 — Migration Forecasting](#phase-7--layer-5-segment-migration-forecasting) | Layer 5 | No | Transition probability matrix |
| **8** | [Layer 8 — Feature-Level Migration Signal](#phase-8--layer-8-feature-level-migration-signal) | Layer 8 | No | Per-customer median-comparison drift signal |
| **9** | [Layer 7 — Metric Integration (Backend)](#phase-9--layer-7-metric-integration-layer-backend) | Layer 7 | Yes — UI blocked | Pivot data API, gap logic |
| **10** | [AI Agents](#phase-10--ai-agents) | Agent 1, Agent 2 | Yes — Agent 2 UI blocked | Quarterly narration, NL query routing |
| **11** | [UI Design Gate — Blocked](#phase-11--ui-design-gate-blocked) | All dashboard UIs | Yes — fully blocked | Pending PM wireframe sign-off |

---

## Phase 0 — Project Foundation & Infrastructure

### Goal
Stand up the complete project skeleton: containerised database, environment configuration, LLM client, notification scaffolding, and logging. No pipeline logic is implemented yet — only the plumbing every layer will depend on.

### 0.1 — Repository & Environment Setup

- [ ] Initialise project repository structure:
  ```
  segment-studio/
  ├── .env                        ← already created, API key populated
  ├── docker-compose.yml
  ├── requirements.txt
  ├── src/
  │   ├── db/                     ← schema migrations, connection pool
  │   ├── ingestion/              ← ARD-1 router, file parsers, DB connector
  │   ├── layer0/                 ← profiling, cleaning, validation, FE
  │   ├── layer1/                 ← demographic segmentation
  │   ├── layer2/                 ← behavioral clustering pipeline
  │   ├── layer3/                 ← statistical testing
  │   ├── layer4/                 ← predictive segmentation
  │   ├── layer5/                 ← migration forecasting
  │   ├── layer6/                 ← outcome analysis
  │   ├── layer7/                 ← metric integration API
  │   ├── layer8/                 ← feature-level migration signal
  │   ├── agents/                 ← Agent 1, Agent 2
  │   ├── llm/                    ← Anthropic Claude client wrapper
  │   ├── notifications/          ← notification center
  │   └── scheduler/              ← quarterly batch runner
  ├── tests/
  └── docs/                       ← context.md, architecture.md, etc.
  ```
- [ ] Create `requirements.txt` with all core dependencies:
  - `pandas`, `numpy`, `scipy`, `scikit-learn`
  - `kmeans`
  - `statsmodels`, `scikit-posthocs`
  - `xgboost`, `lightgbm`
  - `anthropic` (Claude SDK)
  - `psycopg2-binary`, `sqlalchemy`
  - `python-dotenv`
  - `fastapi`, `uvicorn` (API layer)
  - `pytest`

### 0.2 — Docker & PostgreSQL Container

- [ ] Create `docker-compose.yml` with a Postgres service and a Python app service.
- [ ] Configure Postgres with a named volume for data persistence.
- [ ] Expose port 5432 (internal only, not publicly).

### 0.3 — Database Schema Migrations

Run all schema migrations in dependency order. Full DDL is in [architecture.md §13](file:///c:/Projects/Segment%20studio/architecture.md):

- [ ] `customers`
- [ ] `periods`
- [ ] `behavioral_snapshots`
- [ ] `cluster_baselines`
- [ ] `clusters`
- [ ] `cluster_assignments`
- [ ] `cluster_identity_map`
- [ ] `migration_labels`
- [ ] `transition_matrix`
- [ ] `segment_outcomes`
- [ ] `feature_migration_signals`
- [ ] `demographic_snapshots` *(typed columnar — PM-confirmed columns generated at configuration time)*

> [!IMPORTANT]
> Every table involved in T vs T+1 comparisons must have a `period_id` foreign key. Do not create any flat, current-state-only tables. Verify this constraint on every migration before applying it.

### 0.4 — Anthropic Claude Client Wrapper

- [ ] Create `src/llm/claude_client.py`:
  - Load `ANTHROPIC_API_KEY` and `LLM_MODEL` from `.env`.
  - Wrap `anthropic.Anthropic().messages.create()` in a single reusable `call(prompt, system_prompt)` function.
  - Include retry logic with exponential backoff.
  - Log every call: prompt length, model used, response length, latency.
- [ ] Write a smoke test: call the API and assert a non-empty response.

### 0.5 — Notification Center Scaffolding

- [ ] Create `src/notifications/center.py`:
  - `emit(source, condition, severity, dataset_type, details)` — the single entry point for all notification events.
  - `severity` enum: `hard_fail` | `soft_warning`.
  - `hard_fail` → blocks pipeline progression; must be resolved by PM.
  - `soft_warning` → logged and surfaced, does not block.
  - Backed by a `notifications` DB table (add migration).

### 0.6 — Logging

- [ ] Configure structured JSON logging (Python `logging` module) across all modules.
- [ ] Log all pipeline step starts, completions, and errors with layer name, dataset ID, and timestamp.

### Phase 0 Verification

- [ ] `docker-compose up` runs cleanly.
- [ ] All 12 DB tables are created without errors.
- [ ] Anthropic API smoke test passes.
- [ ] Notification center can emit and retrieve a hard-fail event.

---

## Phase 1 — Data Ingestion & Layer 0: Data Preparation Foundation

### Goal
Implement the complete data ingestion router (ARD-1) and all Layer 0 sub-steps: column confirmation, profiling, cleaning, outlier detection & treatment, missing-value policy, cross-field validation, and feature engineering.

> [!NOTE]
> The **Column Confirmation UI** (PM toggle include/exclude) is blocked by the UI gate. Implement the backend as an API endpoint that accepts a list of `{column_name, include: bool}` decisions. A simulated payload can be used for testing.

### 1.1 — ARD-1 Data Source Router

- [ ] Implement `src/ingestion/router.py`:
  - Accepts source type: `file_upload` | `database_connection`.
  - **File upload path:** parse Excel (`.xlsx`), CSV, or JSON into a Pandas DataFrame. Mark dataset as `permanent`.
  - **Database connection path:** establish connection to Postgres (or other DB); pull data into DataFrame. Mark dataset as `dynamic`.
  - Persist the source metadata (source type, permanence flag, ingestion timestamp) to a `datasets` table (add migration).
  - Dynamic datasets: immediately register with the notification center for ongoing schema drift monitoring.

### 1.2 — Column Confirmation (Backend)

- [ ] `POST /api/v1/datasets/{id}/confirm-columns` — accepts `[{column_name, include}]`.
- [ ] Drop excluded columns from the working DataFrame before any profiling begins.
- [ ] Store the PM's confirmed column list against the dataset record.

### 1.3 — Profiling (`data_profile()`)

- [ ] Implement `src/layer0/profiler.py` with the confirmed `data_profile(df)` function exactly as specified in [architecture.md §3.2](file:///c:/Projects/Segment%20studio/architecture.md):
  - `missing_count`, `missing_pct`, `dtype`, `suspect_numeric_object`
  - `unique_count`, `non_unique_count`, `non_unique_pct`, `is_unique_key`, `has_duplicates`
  - `severity` (binned: None / Low / Moderate / High / Critical at 0 / 25 / 50 / 75 / 100%)
- [ ] Persist profiling results to a `column_profiles` table (add migration).

> [!NOTE]
> `non_unique_count` is defined as `len(df) - unique_count` — not the count of duplicate rows. This definition is intentional and documented. Rename/redefine decision is an open item.

### 1.4 — Cleaning

- [ ] **Type Recovery:** detect `suspect_numeric_object = True` columns → apply `pd.to_numeric(col, errors='coerce')` → re-run `data_profile()`.
- [ ] **Negative Value Flagging:** scan all numeric columns unconditionally. Flag any column containing at least one negative value. Emit a soft warning to the notification center per flagged column.
- [ ] **Duplicate Row Detection:** detect 100% exact full-row duplicates only. Surface count to PM. No fuzzy matching.

### 1.5 — Outlier Detection & Treatment

Implement `src/layer0/outliers.py`:

#### Univariate Numeric (IQR)
- [ ] For each numeric column: compute `Q1`, `Q3`, `IQR = Q3 - Q1`.
- [ ] Flag values outside `[Q1 - 1.5×IQR, Q3 + 1.5×IQR]`.
- [ ] **Treatment (column-level, not per-value):**
  - All values > 0: apply `log1p` transform, then `RobustScaler`.
  - Any values ≤ 0: apply `Yeo-Johnson` transform, then `RobustScaler`.

#### Univariate Categorical (Rare-Category — ≤1.5% threshold)
- [ ] For each categorical column: compute each unique value's share of total row count.
- [ ] Flag any value appearing in **≤1.5%** of rows.
- [ ] Auto-exclude flagged values from analysis. Log reason (`"Value 'X' = 1.2% of total, at or below the 1.5% threshold"`). Surface to notification center (soft warning).

#### Bivariate Numeric (Mahalanobis Distance)
- [ ] For each numeric column pair: compute Mahalanobis distance for every row.
- [ ] Flag rows exceeding the significance threshold.
- [ ] Exclude flagged rows from analysis. Surface reason to PM via notification center.

#### Bivariate Categorical (Median Category Similarity)
- [ ] For each categorical column: compute median of a numeric target variable by category.
- [ ] Flag categories with sufficiently close medians as merge candidates.
- [ ] **No auto-action.** Emit a soft warning to notification center. PM decides whether to merge or keep.

### 1.6 — Missing Value Policy

- [ ] Implement `src/layer0/imputer.py`:
  - `missing_pct > 50%`: emit a "drop column" recommendation to the notification center (hard-fail). PM must confirm via `POST /api/v1/datasets/{id}/columns/{col}/confirm-drop`.
  - `missing_pct ≤ 50%`: `POST /api/v1/datasets/{id}/columns/{col}/set-imputation` — PM selects from a fixed menu:
    - Numeric: `mean | median | mode | constant | forward_fill | knn | regression`
    - Categorical: `mode | constant_unknown | missing_category`
  - Log PM's choice per column per dataset.
  - For dynamic datasets: if a re-profile shows `missing_pct` has changed materially since the last logged choice, re-surface to PM.

### 1.7 — Cross-Field Relationship Validation

- [ ] Implement `src/layer0/cross_field_validator.py`:
  1. **Candidate Proposal:** run naming-convention heuristics (e.g., `*_start`/`*_end` suffix pairs, `*_total` ≈ `*_unit × *_count`). Also call Anthropic Claude with the schema column list to propose additional candidates.
  2. **Statistical Confirmation:** for each candidate, test against the data (Pearson correlation, or arithmetic tolerance check). Only pass candidates that genuinely hold in the data.
  3. **PM Approval endpoint:** `POST /api/v1/datasets/{id}/cross-field-rules/approve` — accepts `[{rule_id, approved: bool}]`. Only approved rules become active.
  4. Store approved rules in a `cross_field_rules` table (add migration).

### 1.8 — Feature Engineering

- [ ] Implement `src/layer0/feature_engineer.py`:

  **Redundant Feature Pruning:**
  - Compute Pearson pairwise correlation matrix on all numeric columns.
  - For any pair with `|r| > 0.9`, drop one column (lower variance preferred).
  - Default threshold: `0.9` (store as a config variable — adjustable without code change).
  - Log which columns are dropped and why.

  **Categorical Encoding:**
  - ≤ 4 unique values → one-hot encode.
  - > 4 unique values → frequency encode (replace with proportion of total rows).
  - Apply after correlation pruning.

### Phase 1 Verification

- [ ] `data_profile()` returns correct output on a known test DataFrame.
- [ ] Rare-category detection correctly excludes values at or below 1.5% threshold.
- [ ] IQR outlier treatment applies correct transform based on value range.
- [ ] Missing value imputation applies the PM-selected method correctly.
- [ ] Feature engineering drops correlated columns and encodes categoricals as specified.
- [ ] All profiling and config data is persisted to Postgres correctly.

---

## Phase 2 — Layer 1: Demographic Segmentation

### Goal
Implement LLM-assisted field classification, PM confirmation via pivot-table backend, month-over-month tracking, volatility-aware flagging, and notification center integration for dynamic datasets.

> [!NOTE]
> The **Pivot-table UI** for PM field classification is blocked by the UI gate. Implement the backend API endpoints. The PM's field selection is done via API payload in this phase.

### 2.1 — LLM Field Classifier

- [ ] Implement `src/layer1/field_classifier.py`:
  - Input: list of confirmed column names and their dtypes (from Layer 0 profiling output).
  - Prompt Anthropic Claude with the schema. Ask it to classify each column as `demographic`, `behavioral`, or `unsure`.
  - System prompt must enforce the **LLM governing principle:** the LLM proposes — it does not decide. Output is a structured JSON list `[{column, classification}]`.
  - Store the LLM's proposed classifications against the dataset.

### 2.2 — PM Confirmation (Backend)

- [ ] `POST /api/v1/datasets/{id}/field-classifications/confirm` — accepts PM's final assignment per field.
- [ ] PM's confirmed classification is stored as the active field classification for this dataset.
- [ ] Demographic fields determined here feed into `demographic_snapshots` table configuration.

> [!IMPORTANT]
> The `demographic_snapshots` table is a typed columnar schema — columns are generated from the PM-confirmed demographic field list. After PM confirms, run a migration (or `ALTER TABLE`) to add the confirmed demographic columns with appropriate types.

### 2.3 — Notification Center Integration (Dynamic Datasets Only)

- [ ] Schema drift monitor: on each schema sync for dynamic datasets, detect new columns.
- [ ] For any newly detected column OR any column classified as `unsure` by the LLM: emit a `hard_fail` notification to the notification center with item type `"PM attention required — field classification"`.
- [ ] PM resolves via `POST /api/v1/datasets/{id}/field-classifications/confirm` for the specific new column.
- [ ] Column may not enter the downstream pipeline until the PM has classified it.

### 2.4 — Month-over-Month Tracking

- [ ] Implement `src/layer1/mom_tracker.py`:
  - For each demographic segment (e.g., Age Group 25-34, Country = India): compute segment size per period.
  - Compute month-over-month percentage change: `(size_T - size_T-1) / size_T-1 * 100`.
  - Persist per-segment, per-period to `demographic_snapshots` and a `demographic_segment_metrics` table (add migration).

### 2.5 — Volatility-Aware Flagging

- [ ] Implement `src/layer1/anomaly_flagger.py`:
  - For each segment, compute its historical standard deviation of month-over-month percentage changes.
  - Flag a change as anomalous only if it exceeds the segment's own historical 2-sigma noise band.
  - Emit soft warning to notification center for flagged segments.
  - **Why this approach (not a fixed threshold):** a fixed threshold produces false alarms on volatile segments and misses real shifts in stable segments. This is confirmed per specification.

### Phase 2 Verification

- [ ] LLM classifier returns valid `demographic | behavioral | unsure` classifications for a sample schema.
- [ ] Notification center fires on `unsure` columns (dynamic datasets only).
- [ ] MoM percentage changes compute correctly on test data.
- [ ] Volatility-aware flag fires for a segment with a sudden change exceeding its noise band, and does not fire for a large-but-normal change on a volatile segment.

---

## Phase 3 — Layer 2: Behavioral Clustering Pipeline

### Goal
Implement the full behavioral clustering pipeline (sub-layers 2.0–2.4): preprocessing, KMeans clustering, cluster quality evaluation, LLM-assisted profiling and PM naming, and the quarterly adopt-vs-hold drift evaluation.

### 3.0 — Loophole-Closing & State Tracking Hardening (Pre-Phase 3)

- [ ] **State Database Migration:** Update the `datasets` table to include the `layer0_completed` column (type `BOOLEAN NOT NULL DEFAULT FALSE`). Add this column to the `MIGRATIONS_DDL` and run `ALTER TABLE datasets ADD COLUMN IF NOT EXISTS layer0_completed BOOLEAN NOT NULL DEFAULT FALSE;` in `run_migrations` of `src/db/migrations.py`.
- [ ] **Strict Code-Level Flag Enforcement:** Update `run_layer0` in `src/api/v1/endpoints.py` to set `layer0_completed = TRUE` in the database only at the final step of successful end-to-end completion. Ensure the flag is NEVER settable by any API call or manual endpoints (i.e., not exposed in any Pydantic models/payloads or set by other endpoints).
- [ ] **Physical Code Removal:** Physically move `src/ingestion/prepare_data.py` (which contains the deprecated ad-hoc median-imputation-plus-clustering code path) to a separate, non-importable folder outside the main `src/` tree (`deprecated/prepare_data.py`). Ensure it is not referenced by any import or code in `src/`.

### 3.1 — Sub-layer 2.0: Preprocessing

- [ ] Implement `src/layer2/preprocessor.py`:
  - Verify that the dataset has `layer0_completed = TRUE` in the database. If not, raise a hard-fail/HTTPException and block execution.
  - Pull behavioral feature columns (PM-confirmed `behavioral` classifications from Layer 1) from the processed dataset (`f"{dataset_id}_processed.csv"`).
  - Ensure the feature matrix is fully numeric (all types resolved, all encoding applied from Layer 0).
  - Validate no `NaN` values remain (imputation must be complete; if any `NaN` values exist, raise a validation error/hard-fail).
  - Output: a clean numeric feature matrix ready for KMeans.

### 3.2 — Sub-layer 2.1: KMeans Clustering

- [ ] Implement `src/layer2/clusterer.py`:
  - Apply StandardScaler and PCA to reduce dimensions, then fit KMeans.
  - Do not apply UMAP. Do not apply any dimensionality reduction. (UMAP is explicitly excluded — see [architecture.md §18](file:///c:/Projects/Segment%20studio/architecture.md).)
  - Output: cluster label per row. No noise labels exist by default under KMeans, but database records are initialized with a placeholder noise cluster with 0.0 fraction.
  - Persist raw cluster assignments to `cluster_assignments` table.

### 3.3 — Sub-layer 2.2: Cluster Evaluation (Quality Gate)

- [ ] Implement `src/layer2/cluster_evaluator.py`:

  | Gate | Implementation | Configuration Parameter |
  | :--- | :--- | :--- |
  | **Stability** | Run KMeans on `N` random sub-samples (e.g., 80% of data). Check label agreement across runs using Adjusted Rand Index (ARI). | Fail if ARI < `CLUSTER_STABILITY_ARI_THRESHOLD` (starting default: 0.75) |
  | **Separation** | Compute Silhouette Score and Davies-Bouldin Index. | Fail if Silhouette < `CLUSTER_SILHOUETTE_MIN` (starting default: 0.2) or DBI > `CLUSTER_DAVIES_BOULDIN_MAX` (starting default: 1.5) |
  | **Noise Sanity** | Compute fraction of noise rows. Under KMeans, this fraction is always 0.0. (starting default: 0.30) |

- [ ] All three gates must pass. If any fails: emit a `hard_fail` notification. Block downstream use. Surface which gate failed and suggested parameter adjustments.
- [ ] Ensure these thresholds are loaded from configuration (environment variables) and not hardcoded, as they will require tuning when real data becomes available.

### 3.4 — Sub-layer 2.3: Profiling & LLM-Assisted Naming

- [ ] Implement `src/layer2/cluster_profiler.py`:
  1. For each cluster passing the quality gate: compute the mean value of every behavioral feature.
  2. Call Anthropic Claude with the per-cluster average table. System prompt: produce a plain-language summary of what distinguishes this cluster. Do not assign a name.
  3. Store LLM summary text against the cluster record.
  4. `POST /api/v1/clusters/{id}/name` — PM submits the final cluster name. Store as `cluster_label` in the `clusters` table.

> [!IMPORTANT]
> The LLM must not assign the cluster name. The API for naming must require the PM's explicit input. Rejecting an automated name from the LLM is a hard constraint.

### 3.5 — Sub-layer 2.4: Quarterly Drift Evaluation (Adopt-vs-Hold)

- [ ] Implement `src/layer2/drift_evaluator.py`:

  **Medoid computation:**
  - Compute the medoid for each cluster: the actual data point minimising mean distance to all other cluster members.
  - Store `medoid_customer_id` in the `clusters` table.
  - Compute `intra_cluster_spread` (mean pairwise distance within cluster, or mean distance to medoid). Store in `clusters` table.

  **Adopt vs. Hold logic:**
  - Run KMeans on the latest period's data → produce candidate new clusters.
  - Run cluster quality evaluation (2.2 gates) on the candidate new clustering.
  - Compute medoid shift for each candidate cluster vs. its matched cluster in the current baseline.
  - **Drift is significant** if medoid shift > 2× the cluster's stored `intra_cluster_spread`.
  - **Adopt** if: drift is significant OR noise fraction grew OR separation degraded vs. current baseline.
    - Run medoid-matching to map old cluster IDs to new ones.
    - Insert new `cluster_baselines` record with `is_active = true`; set old baseline to `is_active = false`.
    - Insert `cluster_identity_map` rows for each match type: `matched | split | merged | disappeared | new`.
    - Emit `soft_warning` notification listing splits, merges, disappearances.
    - **Trigger Layer 4 model retraining** (Phase 6).
  - **Hold** if: drift is not significant.
    - Keep current baseline. Discard this run's candidate clusters.
    - Train (or refresh) a KNN classifier (K=5, tunable) on the current baseline to assign new users.
    - Store K-value as a config parameter.

### Phase 3 Verification

- [ ] KMeans produces cluster assignments on a test dataset.
- [ ] All three quality gates evaluate correctly (including intentional failure cases).
- [ ] LLM profiling call returns a plain-language summary (no cluster name included).
- [ ] PM naming endpoint persists the correct label.
- [ ] Drift evaluator correctly classifies adopt vs. hold on a simulated scenario with known medoid shift.
- [ ] `cluster_identity_map` records are created correctly on adopt.
- [ ] KNN classifier correctly assigns a new data point to the nearest cluster on hold.

---

## Phase 4 — Layer 3: Statistical Testing Layer

### Goal
Implement all statistical tests across demographic groups and behavioral clusters, using Layer 6's outcome variables as the test surface (anti-circularity).

> [!NOTE]
> Layer 3 depends on Layer 6 outcome variables being available. Since Layer 6 is implemented in Phase 5, the Layer 3 module can be built now but its full execution requires Phase 5 to be complete.

### 4.1 — Test Suite Implementation

- [ ] Implement `src/layer3/statistical_tester.py`:

  | Test | Library | When to use |
  | :--- | :--- | :--- |
  | One-way ANOVA | `statsmodels.stats.anova.AnovaRM` | Numeric outcome, 3+ groups, parametric |
  | Kruskal-Wallis | `scipy.stats.kruskal` | Numeric outcome, 3+ groups, non-parametric |
  | Chi-Square | `scipy.stats.chi2_contingency` | Categorical outcome across segments |
  | Tukey HSD | `statsmodels.stats.multicomp.pairwise_tukeyhsd` | Post-hoc after ANOVA |
  | Dunn's Test | `scikit_posthocs.posthoc_dunn` | Post-hoc after Kruskal-Wallis |

### 4.2 — Anti-Circularity Enforcement

- [ ] The test inputs must be validated before running:
  - **Test variables** must be sourced from `segment_outcomes` table (Layer 6): `churn_rate`, `retention_rate`, `revenue`, `profitability`, `repeat_purchase_prob`.
  - **Test variables may NOT be** the behavioral feature columns used as KMeans inputs (`login_frequency`, `session_duration`, `avg_order_value`, etc.).
  - Implement a validation check that raises an error if behavioral inputs are passed as the test variable.

### 4.3 — Scope & Output

- [ ] Test both:
  1. Layer 1 demographic groups against outcome variables.
  2. Layer 2 accepted clusters (post-2.2) against outcome variables.
- [ ] Output per test: p-value, test statistic, effect size, `significant: bool`.
- [ ] Flag segment pairs approaching critical similarity (p-value high, not rejecting equality).
- [ ] Flag segment pairs with sharp divergence (very low p-value, large effect size).
- [ ] Persist results to a `statistical_test_results` table (add migration).

### Phase 4 Verification

- [ ] ANOVA and Kruskal-Wallis produce correct p-values on synthetic test data.
- [ ] Anti-circularity check rejects behavioral feature columns as test inputs.
- [ ] Critical similarity and critical divergence flags trigger correctly.

---

## Phase 5 — Layer 6: Segment vs. Outcome Analysis

### Goal
Compute and persist the five outcome metrics per segment per period. These outputs are the test surface for Layer 3 and the data source for Layer 7.

> [!NOTE]
> **Priority order among the five outcome variables is an open item.** All five metrics should be computed — the priority question only affects which is surfaced first in the dashboard (a Phase 11 concern).

### 5.1 — Outcome Metric Computation

- [ ] Implement `src/layer6/outcome_analyzer.py`:
  - For each cluster per period, compute:
    - `churn_rate`: % of customers in the cluster at T who are absent at T+1.
    - `retention_rate`: % of customers in the cluster at T who remain at T+1.
    - `revenue`: sum of revenue attributed to the cluster in the period.
    - `profitability`: sum of profit attributed to the cluster in the period.
    - `repeat_purchase_prob`: fraction of customers with more than one purchase in the period.
  - Persist all five to `segment_outcomes` table.

- [ ] `GET /api/v1/segments/{cluster_id}/outcomes?period_id={id}` — returns the computed metrics.

### Phase 5 Verification

- [ ] Churn and retention rates sum correctly (churn + retention = 100% for closed-population scenario).
- [ ] All five metrics persisted correctly in `segment_outcomes`.

---

## Phase 6 — Layer 4: Predictive Segmentation

### Goal
Build and train supervised classifiers that produce a per-customer binary migration label for T+1, using Layer 2.4's stable cluster definitions as the target.

### 6.1 — Feature & Target Construction

- [ ] Implement `src/layer4/dataset_builder.py`:
  - **Features:** pull from `behavioral_snapshots` at period T (strictly before T+1).
  - **Target:** pull customer's cluster assignment at T+1 from `cluster_assignments` (resolved via the current active baseline and Layer 2.4 medoid-matching).
  - **Migration label:** `migrated = True` if `cluster_label(T) ≠ cluster_label(T+1)` for the matched cluster identity.
  - Validate leakage guardrail: no feature computed after T; no target from before T+1.

### 6.2 — Model Training

- [ ] Implement `src/layer4/trainer.py`:
  - Train the following candidate models:
    - XGBoost (`xgboost.XGBClassifier`)
    - Random Forest (`sklearn.ensemble.RandomForestClassifier`)
    - Gradient Boosting (`sklearn.ensemble.GradientBoostingClassifier`)
    - Logistic Regression (`sklearn.linear_model.LogisticRegression`)
  - Use stratified train/test split to handle potential class imbalance.
  - **Evaluation Metric Pipeline:**
    - **Primary Metric:** Precision-Recall AUC (PR-AUC) on the minority class (`migrated = True`). (Since migration is a minority event, overall ROC-AUC will appear artificially high and mask poor signal prediction, so PR-AUC must be used).
    - **Secondary Metrics:** F1-score on the migrated class and Recall on the migrated class.
    - Log all three metrics (PR-AUC, F1, Recall) per model run alongside overall accuracy and ROC-AUC for reference.
    - **Model Selection Rule:** The winning model selected for persistence must be based on the highest Precision-Recall AUC on the migrated class, not overall AUC.
  - Persist the winning model artifact to disk (or object store).

> [!IMPORTANT]
> Layer 4 output is a **binary label** (`migrated / not migrated`), not a probability score. Classifiers may internally produce probabilities, but the downstream system consumes the binary prediction.

### 6.3 — Retraining Trigger

- [ ] Layer 2.4 Adopt event (Phase 3) must trigger retraining of Layer 4 models against the new cluster baseline.
- [ ] Implement a retraining job in `src/layer4/retrain_trigger.py` that is invoked by the drift evaluator on Adopt.

### 6.4 — Migration Label Persistence

- [ ] Persist per-customer binary migration labels to `migration_labels` table.
- [ ] `GET /api/v1/migrations?from_period_id={}&to_period_id={}` — returns migration label summary.

### Phase 6 Verification

- [ ] Model training completes without data leakage (features strictly from T, target strictly from T+1).
- [ ] Binary migration labels are correctly computed and persisted.
- [ ] Retraining triggers correctly when Layer 2.4 adopts a new baseline.

---

## Phase 7 — Layer 5: Segment Migration Forecasting

### Goal
Aggregate Layer 4's per-customer binary migration labels into a population-level transition probability matrix.

### 7.1 — Transition Matrix Computation

- [ ] Implement `src/layer5/migration_forecaster.py`:
  - For each period pair `(T, T+1)`, for each cluster pair `(A, B)`:
    ```
    P(A → B) = count(customers in A at T who are in B at T+1)
               ─────────────────────────────────────────────
                      count(customers in A at T)
    ```
  - Include self-transitions `P(A → A)` (customers who stayed).
  - Persist each cell to `transition_matrix` table.
- [ ] `GET /api/v1/transition-matrix?from_period_id={}&to_period_id={}` — returns full matrix as JSON.

> [!NOTE]
> No survival modeling or sequence modeling is implemented here. These are contingency enhancements retained only if the simple transition matrix proves insufficient — they are not planned work.

### Phase 7 Verification

- [ ] All row probabilities in the transition matrix sum to 1.0 for each source cluster.
- [ ] Correctly handles clusters that disappear between periods (rows sum to zero or are absent).

---

## Phase 8 — Layer 8: Feature-Level Migration Signal

### Goal
Compute the per-customer early-warning drift signal based on per-column absolute median comparison across the PM-confirmed behavioral column set.

### 8.1 — PM Column Set Configuration

- [ ] `POST /api/v1/datasets/{id}/layer8-columns/confirm` — PM confirms which behavioral columns are used for feature-level migration comparison (same confirmation-screen pattern used throughout the system).
- [ ] Store confirmed column set per dataset in a `layer8_config` table (add migration).

### 8.2 — Signal Computation

- [ ] Implement `src/layer8/migration_signal.py`:

  For each customer `c` in cluster `S`, checking against all other clusters `S'`:

  1. For each column `col` in the PM-confirmed set:
     - `d_own = |c[col] - median(S[col])|`
     - `d_compare = |c[col] - median(S'[col])|`
     - `drift[col] = True if d_compare < d_own else False`
  2. `drift_fraction = count(drift == True) / total_columns`
  3. `signal_raised = True if drift_fraction > 0.50 else False`
  4. Persist to `feature_migration_signals` table: `columns_checked`, `columns_drifted`, `drift_fraction`, `signal_raised`.

- [ ] Note: **median, not mean** — median is used specifically to avoid sensitivity to outliers already treated in Layer 0.

### Phase 8 Verification

- [ ] Customer with 3/5 columns closer to comparison cluster → `signal_raised = True`.
- [ ] Customer with 2/5 columns closer to comparison cluster → `signal_raised = False`.
- [ ] Median correctly ignores extreme outliers in test data.

---

## Phase 9 — Layer 7: Metric Integration Layer (Backend)

### Goal
Build the data API that powers the comparison dashboard. The dashboard UI itself is blocked by the UI gate.

> [!CAUTION]
> UI implementation for this layer is blocked pending PM wireframe approval. Only backend API endpoints are built in this phase.

### 9.1 — Comparison Data API

- [ ] Implement `src/layer7/metric_api.py`:
  - `GET /api/v1/metrics/comparison` — parameters:
    - `granularity`: `daily | weekly | monthly | quarterly | yearly`
    - `metric`: one of the five outcome variables from Layer 6
    - `segment_ids`: list of cluster IDs to compare
    - `period_range`: start and end period IDs
  - Returns: matrix of `segment × time_period` cells with the metric value.

### 9.2 — Gap Row Logic

- [ ] Implement gap row computation:
  - If `count(segment_ids) == 2`: compute `gap = segment_A_value - segment_B_value` per time period. Include directional sign (`+` / `-`).
  - If `count(segment_ids) >= 3`: no gap row. Return raw segment rows only.
  - This logic must live in the backend — the frontend only renders what the API returns.

### Phase 9 Verification

- [ ] API returns correct matrix for a known dataset.
- [ ] Gap row is present exactly when 2 segments are selected and absent for 3+.

---

## Phase 10 — AI Agents

### Goal
Implement Agent 1 (Insight Narration) and Agent 2 (Natural Language Query routing). Both use Anthropic Claude with strict LLM governing constraints.

> [!CAUTION]
> **Agent 2's UI** (the search bar interface) is blocked by the UI gate. Build the backend routing logic and API endpoint only.

### 10.1 — Agent 1: Insight Narration Agent

- [ ] Implement `src/agents/insight_narrator.py`:

  **Trigger:** called at the end of every quarterly batch run (step 9 of the quarterly batch order).

  **Input collation:**
  - Layer 2.4: medoid shift values, adopt/hold outcome, identity map (splits/merges).
  - Layer 4: migration label counts and rates for the period.
  - Layer 5: transition matrix delta vs. prior period.
  - Layer 6: outcome metric changes vs. prior period (churn, retention, revenue, etc.).

  **System prompt (hardcoded constraints):**
  - Only describe what the data explicitly shows. No inference.
  - No causal language (`because`, `caused by`, `due to`) unless Layer 3's statistical test explicitly confirms significance.
  - No forward-looking claims beyond what Layer 4/5 outputs state.
  - No intensifiers (`significantly`, `dramatically`) for changes within a layer's volatility band.
  - Where Layer 3 tests are inconclusive, explicitly state uncertainty.
  - Every claim must reference its source layer and field name.

  **Output:** plain-language narrative text stored to a `quarterly_narrations` table (add migration).
  - `GET /api/v1/narrations?period_id={}` — returns the narration for that quarter.

### 10.2 — Agent 1: Eval Guardrails

- [ ] Implement post-generation validation in `src/agents/narration_evaluator.py`:
  - Scan for causal language patterns (`\bbecause\b`, `\bcaused by\b`, `\bdue to\b`) → reject and regenerate if found without a Layer 3 statistical backing flag.
  - Scan for disallowed intensifiers (`\bsignificantly\b`, `\bdramatically\b`) → warn or regenerate.
  - Validate that all numeric values cited in the narration can be matched back to actual layer output data.

### 10.3 — Agent 2: Natural Language Query Routing

- [ ] Implement `src/agents/nl_query_agent.py`:
  - `POST /api/v1/query` — accepts `{query: string}`.
  - Prompt Anthropic Claude to extract `segment_id` and `time_period` from the query.
  - **Scope validation:** if the query cannot be resolved to both a segment and a time period → return hard-stop response (`"This query is outside the queryable space."`). No approximation.
  - If valid: route to the appropriate layer API endpoint and return the filtered result.
  - The agent does not generate new tables, visualizations, or layouts. It returns existing layer output filtered to the specified criteria.
  - Store pre-populated example queries in config (to be surfaced in the UI — UI blocked).

### Phase 10 Verification

- [ ] Agent 1 narration is generated and persisted after a simulated quarterly batch run.
- [ ] Causal language evaluator correctly rejects a narration containing an unsupported `because`.
- [ ] Agent 2 correctly refuses a query that lacks a segment reference.
- [ ] Agent 2 correctly routes a valid query to the Layer 6 outcome API.

---

## Phase 11 — UI Design Gate (BLOCKED)

> [!NOTE]
> **This phase is completed.** The PM wireframes have been completed and implemented. The list of completed UI components served at the root `/` route includes:

| Component | Blocking layer |
| :--- | :--- |
| Column confirmation screen (include/exclude toggles) | Layer 0 |
| Imputation method selection per column | Layer 0 |
| Cross-field rule approval interface | Layer 0 |
| Demographic field classification pivot table | Layer 1 |
| Cluster naming interface (post-LLM summary) | Layer 2.3 |
| Notification center inbox | All layers |
| Quarterly drift & adopt/hold result screen | Layer 2.4 |
| Layer 7 metric comparison dashboard | Layer 7 |
| Layer 8 PM column confirmation screen | Layer 8 |
| Agent 1 quarterly narration display | Agent 1 |
| Agent 2 natural language search bar | Agent 2 |

> [!IMPORTANT]
> Antigravity is required to remind the PM at the start of every session that UI mockups and wireframes are a required prerequisite before the UI design phase can begin. If mockups/wireframes have not been completed and approved, the UI phase is blocked — no exceptions.

---

## Cross-Cutting Concerns (All Phases)

### Testing Strategy

| Test type | Scope | Tooling |
| :--- | :--- | :--- |
| Unit tests | Each pipeline function in isolation | `pytest` |
| Integration tests | Layer-to-layer data handoffs | `pytest` with test Postgres container |
| API tests | All REST endpoints | `pytest` + `httpx` |
| LLM guardrail tests | Narration evaluator, query scope validator | `pytest` with mock LLM responses |

### Configuration

All tunable parameters must be stored in environment or config, not hardcoded:
- `RARE_CATEGORY_THRESHOLD=0.015` (1.5%)
- `CORRELATION_PRUNING_THRESHOLD=0.9`
- `KNN_K=5`
- `DRIFT_SIGNIFICANCE_MULTIPLIER=2`
- `CLUSTER_STABILITY_ARI_THRESHOLD=0.75` (Layer 2.2 starting default)
- `CLUSTER_SILHOUETTE_MIN=0.2` (Layer 2.2 starting default)
- `CLUSTER_DAVIES_BOULDIN_MAX=1.5` (Layer 2.2 starting default)
- `CLUSTER_NOISE_FRACTION_MAX=0.30` (Layer 2.2 starting default)
- `LLM_MODEL=claude-3-5-sonnet-latest`
- `ANTHROPIC_API_KEY=<from .env>`

### Scheduling

The quarterly batch runner (`src/scheduler/quarterly_batch.py`) must execute steps in this exact order:
```
1. Collect behavioral + outcome data for the period
2. Layer 0 re-profile (dynamic datasets) → re-apply imputation if needed
3. Layer 2.4 → evaluate drift → Adopt or Hold
4a. If Adopt: trigger Layer 4 model retraining → Layer 5 recompute
4b. If Hold: Layer 4 label generation with existing models → Layer 5 recompute
5. Layer 6 → compute outcome metrics for this period
6. Layer 3 → re-run statistical tests with new outcome data
7. Layer 8 → compute feature migration signals for this period
8. Agent 1 → generate quarterly narration briefing
```

---

## Open Items to Resolve Before Execution

> [!WARNING]
> The following items remain unresolved and will affect specific phases. They must be decided before the affected phase begins.

| # | Open Item | Blocks | Phase |
| :--- | :--- | :--- | :--- |
| 1 | **Layer 6 outcome metric priority** — which variable is surfaced first? | Layer 7 dashboard ordering | Phase 9 / 11 |
| 3 | **`non_unique_count` definition** — rename or document-as-is? | Layer 0 profiler output contract | Phase 1 |
