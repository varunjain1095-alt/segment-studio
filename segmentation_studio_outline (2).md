# Customer Segmentation Studio — project outline

Working notes for the segmentation framework, built layer by layer and substantially revised from the original draft. This is a planning document, not a build.

---

## Overview of the revision process

The original framework had 10 layers (0–9), including separate manual rule-based behavioral segmentation, algorithmic clustering, predictive segmentation, and migration forecasting as independent layers. Through review, the structure was consolidated and renumbered. Key resolved tensions:

- **Manual vs. algorithmic segmentation conflict** (behavioral layer): resolved by dropping manual rule-based behavioral segments entirely. Clustering now decides what behavioral groups exist; rules don't pre-define them.
- **Cluster instability over time** (the issue that nearly killed predictive segmentation and migration forecasting): resolved with a hybrid schedule — quarterly evaluation cadence, but adopt-vs-hold decided by actual drift, not the calendar.
- **Redundant profiling**: a proposed enrichment layer combining predictive + migration output into personas was considered and explicitly cut as unneeded, since core profiling already happens earlier (Layer 2.3).

Final structure: **9 layers (0–8)** — 8 layers (0–7) from the original consolidation down from 10, plus Layer 8 (Feature-Level Migration Signal) added later as a secondary early-warning signal distinct from Layer 4's cluster-reassignment migration.

---

## Architecture Requirement Decisions (ARD)

Cross-cutting system-level decisions that constrain multiple layers — kept separate from layer-by-layer logic since these are infrastructure/routing decisions, not segmentation logic.

**ARD-1 — Data source entry & permanence.** The PM starts by selecting a data source: Excel, CSV, JSON, or database connection (e.g., Postgres). This selection goes through a router. The connection type itself determines dataset permanence — no separate PM flag needed:
- **File upload (Excel/CSV/JSON) → permanent dataset.** A snapshot; schema doesn't change on its own. Proceeds straight to dashboard after Layer 0's confirmation step.
- **Database connection → dynamic dataset.** Live; schema can change underneath. Feeds the notification center (see Layer 1) for ongoing schema drift, in addition to Layer 0's initial confirmation step.

Referenced by: Layer 0 (entry point), Layer 1 (notification center trigger).

---

## Layer 0 — Data Preparation Foundation

**Entry** — see ARD-1 for source selection/router/permanence. Layer 0 begins once a dataset (permanent or dynamic) has entered the system.

**User Confirmation Section — confirmed.** First step of Layer 0. Shows the PM every column detected, each marked include/exclude (default sign shown per column), with the PM able to toggle any column in or out. Excluded columns are dropped here and don't proceed to profiling, cleaning, validation, or feature engineering.

Order confirmed: **profiling → cleaning → validation → feature engineering (if required)**. Treated as a non-negotiable prerequisite — without it, every downstream segmentation layer becomes unstable, especially clustering, which is highly sensitive to unscaled features and outliers.

**Profiling implementation — confirmed.** `data_profile()` function (pandas) generates a per-column table: missing_count, missing_pct, dtype, suspect_numeric_object (flags object columns where >90% of values parse as numeric), unique_count, non_unique_count, non_unique_pct, is_unique_key, has_duplicates, severity (binned on missing_pct).

```python
import pandas as pd

def data_profile(df_analysis):
    return (
        df_analysis.isna()
        .sum()
        .to_frame("missing_count")
        .assign(
            missing_pct=lambda x: ((x["missing_count"] / len(df_analysis)) * 100).round(2),
            dtype=df_analysis.dtypes.astype(str),
            suspect_numeric_object=lambda x: (
                df_analysis.dtypes.astype(str).eq("object")
                & df_analysis.apply(
                    lambda col: (
                        pd.to_numeric(col.dropna(), errors="coerce").notna().mean() > 0.9
                        if col.dropna().shape[0] > 0
                        else False
                    )
                )
            ),
            unique_count=df_analysis.nunique(dropna=True),
            non_unique_count=lambda x: len(df_analysis) - x["unique_count"],
            non_unique_pct=lambda x: ((x["non_unique_count"] / len(df_analysis)) * 100).round(2),
            is_unique_key=lambda x: x["unique_count"] == len(df_analysis),
            has_duplicates=lambda x: x["unique_count"] < len(df_analysis),
            severity=lambda x: pd.cut(
                x["missing_pct"],
                bins=[-0.01, 0, 25, 50, 75, 100],
                labels=["None", "Low", "Moderate", "High", "Critical"],
                include_lowest=True
            )
        )
        .sort_values("missing_pct", ascending=False)
    )
```

**Known caveat on `non_unique_count`/`non_unique_pct`:** these are computed as `len(df) - unique_count`, which is *not* the same as "number of duplicate rows" or "number of distinct values that repeat." It's the count of extra occurrences beyond each value's first appearance. Not yet decided whether to rename/redefine these or leave as-is with documentation.

**Cleaning fix — confirmed for Layer 0 cleaning step.** A numeric column stored as dtype `object` due to blank-string values (e.g. `" "`) instead of true NaNs will show `suspect_numeric_object = True` while `missing_count` under-reports real nulls. Fix: coerce such columns with `pd.to_numeric(col, errors='coerce')`, then re-run `data_profile()` to get accurate missing_count/missing_pct before validation runs.

**Cleaning rules — confirmed.**
- **Negative-value flagging:** every numeric column containing any negative value is flagged, unconditionally — no per-column judgment or PM confirmation gate. Deliberate tradeoff: catches genuine errors (negative age, negative BMI, negative counts) without exception-handling cost; accepted cost is occasional false positives on columns where negatives are legitimate (temperature, profit/loss, deltas).
- **Duplicate detection:** 100% exact full-row match only. Fuzzy/near-duplicate matching (e.g., pairwise 90% column-match between two rows) was considered and dropped — it's an O(n²) pairwise comparison that doesn't scale past thousands of rows without blocking/indexing, and was judged not worth the complexity for now.

**Outlier detection — confirmed, four checks, all automated. Flagging only — treatment is a separate, later step, not yet specified.**
- **Univariate numeric:** IQR-based. Flag values outside `Q1 - 1.5×IQR` / `Q3 + 1.5×IQR`.
- **Univariate categorical:** flag categories below a minimum share of total count (rare-category detection).
- **Bivariate numeric:** Mahalanobis distance — flags points far from the joint distribution of a variable pair, accounting for correlation between them (replaces manual scatterplot review with an automated equivalent).
- **Bivariate categorical:** automated median comparison between categories within a column — categories with sufficiently close medians are flagged as merge *candidates*. Flagging only; no automatic merge happens here. Distinct from Layer 3's near-identical-segment flagging: this operates on raw categorical *values within a column*, pre-clustering; Layer 3 flags similarity between resulting *clusters*, post-clustering. No overlap.

**Treatment of flagged items — outlier treatment confirmed for all four checks.**
- **Univariate numeric (IQR outliers) — column-level transform, not per-value treatment:**
  - Column strictly positive (all values > 0): apply `log1p` transform, then RobustScaler.
  - Column spans negative to positive: apply Yeo-Johnson transform (handles zero/negative values, unlike `log1p` which is undefined below -1), then RobustScaler. Entirely-negative or entirely-zero columns fall under this branch by default.
- **Univariate categorical (rare categories, below minimum % of total):** system auto-excludes the rare category from analysis by default, and surfaces the reason for exclusion (e.g., "X% of total, below threshold") — automatic exclude with a visible reason, not a silent drop.
- **Bivariate numeric (Mahalanobis outliers):** flagged row is excluded from analysis, with the reason surfaced to the PM — same pattern as rare-category exclusion.
- **Bivariate categorical (merge candidates, similar medians):** no automatic action. System notifies PM of the implication (these categories look similar enough to consider merging/treating as one), and PM decides include or exclude.

**Missing-value handling policy — confirmed, fixed threshold.**
- `missing_pct > 50` → system auto-generates a "drop this column" recommendation. PM confirms/overrides in the confirmation screen — not silently auto-executed.
- `missing_pct <= 50` → PM selects an imputation method per column from a fixed menu (no silent default applied):
  - Numeric: mean, median, mode, constant/zero-fill, forward-fill (if time-ordered), or model-based (KNN/regression imputation) for higher-stakes columns.
  - Categorical: mode, constant/"Unknown" category, or a distinct "missing" category if missingness itself may be predictive.
- Threshold is fixed at 50%, not PM-adjustable.
- PM's method choice is logged per column per dataset — a re-profile later (dynamic dataset) can reapply automatically or re-surface if missing_pct has since changed significantly.

**Validation checks — gaps beyond current `data_profile()` output, and enforcement model.**

Current profiler covers missingness, dtype, suspect-numeric detection, column-level uniqueness, and severity. Not yet covered: range checks, categorical domain/whitelist checks, date validity, referential integrity, full-row duplicate detection, cross-field/logical relationship checks, outlier detection, class imbalance, drift (dynamic datasets), expected-column presence, row count sanity.

**Enforcement model, confirmed:**
- **Rule source:** auto-inferred from observed data (ranges, categorical whitelists) where no business context is needed; PM-set for business-specific rules (e.g., age bounds); proposed-and-confirmed for cross-field relationships (see below).
- **Failure action, set per check:** hard fail (blocks progression, routes to notification center) for things that would silently corrupt downstream layers — PK duplicates, referential integrity breaks, expected column missing. Soft warning (logged, doesn't block) for informative-but-not-fatal issues — outliers, class imbalance, drift. Threshold-escalated checks reuse the existing `severity` bins, escalating from warning to hard fail past a severity level.
- **Mechanically:** each check returns a structured result (pass/fail/warning + affected column/rows), not a bare boolean, so the notification center can show the PM specifics.

**Cross-field relationship checks — confirmed approach.** Hand-writing rules for every possible relationship (e.g., `end_date >= start_date`, `TotalCharges ≈ MonthlyCharges × tenure`) doesn't scale — "a million such relationships" isn't realistically enumerable by hand. Resolved as a three-step pipeline:
1. **Candidate proposal** — naming-convention heuristics (e.g., `*_start_date`/`*_end_date` pairs) and/or an LLM reading the schema propose plausible relationships. Same legibility-not-decision principle as Layer 2.3: the LLM surfaces candidates, it does not enforce anything unilaterally.
2. **Statistical confirmation** — candidate relationships are tested against the actual data (correlation, tolerance-band check on arithmetic relationships) to confirm they genuinely hold, rather than trusting a name-based or LLM guess blindly.
3. **PM approval** — PM does final approve/reject per candidate, same pivot-style confirmation pattern used elsewhere. Only approved relationships become active, logged rules.

Explicitly not aiming for exhaustive coverage — the value is catching common/obvious relationships cheaply, not guaranteeing every logical relationship in the dataset is discovered.

**Feature engineering — confirmed.**
- **Redundant-feature pruning:** pairwise correlation check (`df.corr()`) on numeric features; drop one feature from any pair above a fixed threshold (default 0.9, adjustable). No VIF/ANOVA/Chi-square — those were considered and dropped as unnecessary overhead: VIF addresses regression instability, which doesn't apply to distance-based HDBSCAN clustering; ANOVA/Chi-square require a target/label, which doesn't exist at this unsupervised stage.
- **Categorical encoding rule:** categorical field with ≤4 unique values → one-hot encoding. Categorical field with >4 unique values → frequency encoding (replace category with its count/proportion in the data). Chosen over target encoding since no target exists at this stage; chosen over unconditional one-hot to avoid dimensionality blowup now that UMAP (which would've absorbed it) has been dropped from Layer 2.

**Dataset permanence** — architecture requirement decision, see ARD section below.

---

## Layer 1 — Demographic Segmentation

**Purpose:** study basic demographic groups (age, gender, country). Remains rule-based — no clustering needed here, since demographic grouping is just grouping people by attributes they already have, not a pattern to be discovered.

**Field classification — confirmed.** An LLM (prompted/few-shot, not fine-tuned) proposes which columns in the dataset are demographic vs. behavioral vs. unsure. This list feeds the dashboard as a **pivot-table-style interface**: the PM selects which demographic fields to analyze directly in the dashboard, rather than reviewing a separate approval list. The PM's field selection *is* the review step.

**Notification center — confirmed.** For dynamic (database-connected) datasets, any field the LLM classifies as "unsure," or any new field that appears after the initial classification, surfaces as a "PM attention required" item in a notification center. The PM decides how to treat the field from there. Permanent (file-upload) datasets skip this — no schema drift to monitor.

**What's studied:**
- Growth and decline of segments over time
- Month-over-month percentage changes

**Flagging logic — revised.** Originally a fixed percentage threshold. Changed to **volatility-aware flagging**: the system learns each segment's own normal month-to-month variability and only flags changes that exceed that segment's typical noise band. This avoids two failure modes of a fixed threshold: false alarms on small/naturally-volatile segments, and missed real shifts on large/normally-stable segments where even a small percentage move is meaningful.

**Tooling:** Python.

---

## Layer 2 — Behavioral Clustering Pipeline

This is the most substantially revised layer. Originally "Behavioral Segmentation" with rule-based labels (Loyalists, Browsers, Discount hunters, At-risk inactive) defined by hand via threshold logic / scorecards.

**Resolved decision:** manual rule-based behavioral segmentation is dropped entirely. If rule-based and algorithmic (clustering) segmentation would ever conflict, clustering wins — the data dictates what behavioral groups exist, rather than groups being pre-defined and checked against data.

Behavioral metrics themselves are unchanged and still required: login frequency, session duration, purchases, AOV, return rate, discount sensitivity, email/app engagement. They no longer define segments directly — they become the **feature input** to clustering.

### Sub-layers

**2.0 — Preprocessing.** Compute and clean the behavioral feature table for every user. This is the layer's only job now with respect to "segmentation logic" — produce a clean numeric feature set, make no labeling decisions.

**2.1 — Clustering.** HDBSCAN run directly on the 2.0 feature table. UMAP dropped — confirmed unnecessary since dimensionality is expected to stay low (not hundreds of features), removing UMAP-seed instability as a variable in 2.4's drift detection. Produces natural clusters, not manually imposed ones — intended to reveal patterns a human wouldn't have pre-specified.

**2.2 — Cluster evaluation.** Added during review as the single most important addition to the original framework. Clustering will always return *some* clusters, even on pure noise — this step checks whether the clusters are actually good:
- **Stability** — do clusters survive re-running on a data subset?
- **Separation** — silhouette score, Davies-Bouldin index
- **Noise-bucket sanity** — is HDBSCAN dumping an excessive proportion of users into noise (a sign parameters are misconfigured)?

Only clusters that pass this step are considered valid and proceed to naming/use downstream.

**2.3 — Variable averages + LLM-assisted naming.** For each valid cluster, compute average values of every behavioral variable (average AOV, average session duration, average return rate, etc.) — the numeric profile of the cluster.

An LLM reads these averages and produces a plain-language summary (e.g., "high return rate, low AOV, frequent logins"). The PM reads the summary and **decides the actual cluster name** — the LLM does not name clusters unilaterally.

*Design principle established here, intended to apply everywhere an LLM is used in this framework:* the LLM's role is to make cluster output **legible** (summarize numbers into language a person can act on quickly), not to make the **decision** (the name itself, and what it implies about the business). Every future LLM touchpoint in this framework should be checked against this same test — summarizing/teaching vs. deciding.

**2.4 — Quarterly stability check.** Added to resolve the cluster-instability problem that initially threatened to invalidate Layers 4 and 5 (predictive segmentation and migration forecasting both depend on cluster identity meaning the same thing over time).

**Drift metric — confirmed.** Medoid (the actual data point most central to a cluster), not centroid — HDBSCAN clusters are density-based and can be irregularly shaped, so a computed mean (centroid) can be non-representative or even fall outside the cluster; a medoid is always a real member of the cluster. Significance of drift is self-calibrating and fixed at **2x**: a medoid shift only counts as meaningful drift if it exceeds 2x that specific cluster's own typical intra-cluster spread, not a fixed absolute distance — same "learn its own normal variability" philosophy as Layer 1's volatility-aware flagging.

**The rule, precisely:**
- Re-clustering is *evaluated* on a fixed quarterly cadence (the calendar decides *when to check*, not *what to do*).
- If the quarterly re-cluster result is **meaningfully different** from the current baseline (by the same drift diagnostics as 2.2 — noise growth, separation degrading — plus medoid shift exceeding the cluster's own typical spread) → **adopt** the new clustering as the new baseline. Medoid matching runs to map old cluster identities to new ones where they genuinely correspond, and explicitly flags splits, merges, or disappearances as findings worth surfacing to a PM.
- If the quarterly re-cluster result is **not meaningfully different** → **hold** the existing baseline; discard this run's result; continue classifying new users into the existing stable cluster definitions (via a lightweight classifier — e.g., KNN — trained on the current baseline, rather than re-deriving clusters from scratch on every use).

This is the same volatility-aware philosophy as Layer 1's flagging logic, applied to cluster identity instead of metric values.

---

## Layer 3 — Statistical Testing Layer

**Purpose:** validate that observed segment/cluster differences are statistically real, preventing false storytelling.

**Tooling, confirmed:** Python throughout (not R, as originally drafted) — `statsmodels` for ANOVA, Kruskal-Wallis, Chi-square, Tukey HSD; `scikit-posthocs` specifically for Dunn's test (not available in `statsmodels`).

**Scope, revised to match Layer 2's changes.** Validates two things now, not one:
1. **Layer 1's demographic groups** — same as original intent.
2. **Layer 2's accepted clusters (post-2.2)** — confirms clusters are not just mathematically well-formed (2.2 already checks that) but also statistically distinct on outcomes that matter to the business. This is a complementary check to 2.2, not a duplicate: 2.2 asks "are these clusters real," Layer 3 asks "do these real clusters actually differ on things we care about."

**Circularity fix — confirmed.** Testing must use **Layer 6's outcome variables** (churn, retention, revenue, repeat purchase probability) — not the raw behavioral features fed into clustering (AOV, return rate, etc.). Testing clusters on their own clustering inputs is circular: clusters were built to separate on exactly those variables, so they'll trivially test as statistically different regardless of whether the segmentation is actually meaningful. Testing against genuine outcome variables the clusters were never built to optimize for is the valid check.

Also retains the original framing: flag situations where segments approach critical similarity or critical difference, since both are informative (near-identical segments may not be worth treating separately; sharply diverging segments are worth surfacing).

---

## Layer 4 — Predictive Segmentation

**Purpose:** forward-looking prediction — not "what cluster is this customer in now" (Layer 2) but "what cluster will they likely be in next month/quarter."

This layer was nearly cut entirely during review, for two compounding reasons: (1) a leakage risk where predictive features could overlap with however a target segment was defined, and (2) a more serious problem — clusters re-derived from scratch each run don't have stable identities, making "predict future cluster" a moving target. Both concerns are resolved by changes made earlier in the framework, which is why this layer was restored rather than permanently dropped.

**How it works:**
1. Features from time T (the Layer 2.0 feature set).
2. Target: the customer's actual cluster assignment at time T+1, using the **stable** cluster definitions maintained by Layer 2.4 — this is what makes the target meaningful across time, resolving the original instability problem.

**Migration definition — confirmed, customer-level.** Run HDBSCAN at T and T+1, match cluster identities across the two runs via 2.4's medoid-matching, then check per customer: is the matched cluster label the same or different? Different label = migration (binary). Same label = no migration, regardless of how much any individual feature value moved. This is the ground-truth migration signal for Layer 4's target — determined entirely by existing cluster-matching machinery, no new logic required.

Note: a separate **feature-level migration signal** (tracking whether an individual customer's raw feature values are shifting toward another segment's typical profile, even before HDBSCAN formally reassigns them) was discussed and deliberately kept out of Layer 4 — it will be specified as its own later layer (Layer 8) rather than conflated with this binary cluster-reassignment definition.

3. Train a supervised model — XGBoost, Random Forest, Logistic Regression, or Gradient Boosting — to learn the T → T+1 mapping.

**Leakage guardrail:** features must be drawn strictly from before the prediction window; the target strictly from after it. Combined with Layer 2's move away from hand-defined segment formulas, this substantially reduces (though doesn't eliminate by default) the risk of the model trivially re-deriving its own label from an overlapping input feature.

**Dependency on Layer 2.4:** if a quarter ever does adopt a new cluster baseline, any model trained on the old cluster definitions needs retraining against the new ones. Layer 4 inherits whichever baseline is currently active — it is not exempt from 2.4's adopt/hold logic.

---

## Layer 5 — Segment Migration Forecasting

**Purpose:** track population-level flow between clusters over time — distinct from Layer 4, which predicts one customer's individual future cluster. This layer asks "what fraction of cluster A typically moves to cluster B," not "where will this specific person go."

Also depends on Layer 2.4's stable cluster identities — without that foundation, this layer has the same instability problem Layer 4 had.

**Relationship to Layer 4 — confirmed.** The transition matrix is an aggregation of Layer 4's per-customer binary migration signal (matched cluster label via 2.4's medoid-matching) across the whole population, not a separately-defined mechanism. Layer 5 doesn't recompute migration independently — it counts Layer 4's customer-level outcomes into pairwise cluster proportions.

**Starting implementation:** a plain transition probability matrix — for every pair of clusters (A, B), the proportion of customers in A at time T who are in B at time T+1 (e.g., "12% of Browsers become Loyalists each quarter, 8% of Loyalists become At-risk").

**Noted limitation:** a plain transition matrix assumes only the *current* cluster matters — not how long someone has been there, or what path led them there (the Markov assumption). Two more sophisticated methods from the original spec — **survival transition modeling** (captures time-in-state) and **sequence modeling** (captures path history) — are retained as later enhancements if the simple transition matrix proves insufficient, rather than built upfront.

---

## Layer 6 — Segment vs. Outcome Analysis

Unchanged from the original spec. Compares clusters against churn, retention, revenue, and repeat purchase probability. Answers: which segments actually matter to the business. Slots in cleanly on top of the now-stable Layer 2 cluster foundation.

---

## Layer 7 — Metric Integration Layer

Embeds segmentation into operational KPI tracking — churn rate by segment, ARPU by segment, CAC payback by segment, retention by segment. The terminal layer; makes the entire framework operational rather than a one-off analysis, consuming outputs from every layer above it.

**Comparison table UI — confirmed.** PM has three selectable controls:
- **Time granularity:** daily, weekly, monthly, quarterly, yearly.
- **Metric/KPI:** which of the tracked metrics to view.
- **Segments to compare:** any number of segments/clusters, PM's choice.

Table renders as segment rows × time-period columns.

**Gap row — confirmed rule.** A gap row (segment A's value minus segment B's value per time period, directionally colored) renders only when exactly 2 segments are selected. When 3+ segments are selected, no gap row — just the raw value rows per segment.

---

## Layer 8 — Feature-Level Migration Signal

**Purpose:** an early-warning signal distinct from Layer 4's cluster-reassignment migration. Layer 4 asks "did HDBSCAN formally reassign this customer to a different cluster" (binary, based on matched cluster labels). Layer 8 asks "are this customer's raw behavioral values drifting toward another segment's profile," which can surface *before* an actual cluster reassignment happens.

**Mechanics — confirmed:**
1. For each of the customer's 4-5 behavioral columns (session time, page views, discount usage, etc.), compare their current value's distance to their own segment's median for that column vs. its distance to a comparison segment's median for that column. Median chosen specifically over mean to avoid sensitivity to outliers already handled elsewhere in the pipeline.
2. This produces a per-column binary: closer to own segment, or closer to the compared segment.
3. **Rollup rule:** if more than 50% of the 4-5 columns are closer to the compared segment's median, the customer is flagged as showing meaningful feature-level migration toward that segment.

**Explicitly distinct from Layer 4:** this is a secondary, column-level heuristic signal, not the ground-truth migration definition. A customer can be flagged here without ever being formally reassigned by HDBSCAN, and vice versa (a formal reassignment could occur without more than 50% of individual columns having crossed toward the new segment's median, depending on how HDBSCAN's joint distance metric weights the full feature set).

---

## Layers considered and explicitly cut

For traceability — these existed in the original 10-layer draft and were deliberately removed, not forgotten:

- **Original "rule-based behavioral segmentation" (old Layer 2)** — replaced entirely by Layer 2's clustering pipeline. Manual labels (Loyalists, Discount hunters, etc.) no longer exist as hand-defined categories; clustering decides what groups exist.
- **Original "Segment profiling and interpretation" (old Layer 7)** — considered as a possible "persona enrichment" step combining Layer 4 and 5 output into richer personas, but judged not needed. Core profiling/naming already happens at Layer 2.3; the enrichment idea was floated but explicitly rejected as an unnecessary addition.

---

## Advanced AI Agent Capabilities

Two agent capabilities confirmed, both consistent with the governing LLM principle (summarizing/teaching, not deciding):

---

### Agent 1 — Insight Narration Agent

**Purpose:** auto-triggered each quarter (matching Layer 2.4's cadence), reads across all layer outputs for that period and produces a plain-language briefing for the PM — connecting signals across layers (e.g., "Cluster C's medoid drifted, 14% of it migrated to Cluster A, and that migration correlates with the churn spike Layer 6 flagged"). Synthesizes; PM decides what to act on.

**Strict data restriction — confirmed.** Every claim in the narration must be traceable to an actual data output from the pipeline. No inference, extrapolation, or language beyond what the data explicitly shows.

**Evals — confirmed, three failure modes guarded against:**
- **Hallucinated causation:** agent must not use causal language ("because," "caused by," "due to") unless causation is explicitly supported by the data. Correlation and timing are not causation.
- **Extrapolation beyond the window:** agent must not make forward-looking claims beyond what Layer 4/5's actual output states.
- **Invented magnitudes:** agent must not use intensifiers ("significantly," "dramatically") for changes that fall within normal volatility bands already established by each layer's flagging logic.

Where Layer 3's tests were inconclusive, the narration must flag uncertainty explicitly rather than smooth over it.

---

### Agent 2 — Natural Language Query Agent

**Purpose:** a "search" section in the dashboard — "Need to look at something specific? Search here" — allowing PMs to ask questions in plain language and get answers surfaced from the relevant existing layer outputs.

**Pre-populated suggested questions — confirmed.** The interface shows example questions to teach the PM what's queryable, consistent with the legibility principle. Suggestions are scoped to the actual queryable space so PMs don't ask outside what the system can answer.

**Scope boundaries — confirmed, two hard constraints:**
- **Segment-bound:** queries must reference specific segments/clusters in the dataset.
- **Time-bound:** queries must reference a specific time period or range.
Anything outside these two axes returns a hard-stop "this isn't queryable here" — no approximation attempted.

**Response format — confirmed.** The agent does not generate new tables, visualizations, or layouts. It routes to and surfaces the relevant existing layer output(s) that answer the question, filtered to the segments and time period specified — the existing dashboard layer *becomes* the answer format. LLM's role is routing and filtering, not rendering.

---

## Summary of LLM touchpoints

Confirmed uses, all checked against the governing principle:

- **Layer 1** — LLM proposes which columns are demographic/behavioral/unsure from the schema; PM confirms via pivot-table-style dashboard interface.
- **Layer 2.3** — LLM summarizes per-cluster variable averages into plain language; PM decides the final cluster name.
- **Agent 1 (Insight Narration)** — LLM synthesizes cross-layer signals into a plain-language quarterly briefing; PM decides what to act on. Strict data restriction enforced via evals.
- **Agent 2 (Natural Language Query)** — LLM routes PM questions to the relevant existing layer outputs; no new rendering or layout generated.
- **Governing principle:** legitimate use is summarizing/teaching so a human can decide faster; illegitimate use is letting the LLM make the actual decision. Every touchpoint above passes this test.

---

## Data Model

**Infrastructure — confirmed.** Postgres, hosted in a new container. Antigravity is responsible for generating the data model from this spec.

**Critical constraint — time must be a first-class citizen in the schema.** The data model must support multiple snapshots per customer across time periods, not a flat single-snapshot table. A current-state-only schema will be unable to support the migration and drift logic in Layers 4, 5, 7, and 8 without a full redesign. Every table that feeds T vs T+1 comparisons must have a time dimension baked in from the start.

**Antigravity's responsibility:** generate a data model it can reason about and query against, derived from this spec, that satisfies the time-native constraint above. The data model must be reviewed and confirmed before any layer implementation begins.

---

## STRICT GATE — UI Design Phase

**Antigravity must not proceed to any UI implementation, component building, or code generation for the dashboard until the following condition is met:**

The PM has completed and approved UI mockups and wireframes for every layer and agent capability in this framework. These mockups/wireframes must be designed and signed off by the PM before any UI work begins.

**Antigravity's responsibility:** remind the PM at the start of every session that UI mockups and wireframes are a required prerequisite before the UI design phase can begin. If mockups/wireframes have not been completed and approved, the UI phase is blocked — no exceptions.

---

## Open items / not yet decided

- Layer 2.4's adopt-vs-hold drift metric is now fully confirmed: medoid shift exceeding 2x the cluster's own typical intra-cluster spread, plus noise/separation diagnostics from 2.2.
- Which specific outcome variables Layer 6 will prioritize first (churn vs. revenue vs. repeat purchase) — not yet decided.
- Data source/dataset for this track — consistent with the other two tracks, still in early/exploratory planning rather than working with real data yet.
