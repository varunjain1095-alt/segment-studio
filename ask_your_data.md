# Ask Your Data — Architectural Specification

This document captures the confirmed architectural decisions for the Ask Your Data layer. It is a living document — updated as decisions are made during brainstorming sessions.

---

## Overview

Ask Your Data is a unified natural language query and command interface for the Segmentation Studio dashboard. It replaces and absorbs the original Agent 2 (A2) functionality, extending it with intent classification, root cause analysis, comparison, prediction routing, and visualization surfacing — all from a single persistent entry point.

---

## Entry Point

**Location:** Top bar — persistent, always visible regardless of which layer is active in the main content area. Same position as the original "ask about any segment..." bar.

**Behavior on activation:** PM types a question and hits enter. A result overlay opens on top of the current content area. The PM's active layer panel remains underneath — the overlay does not navigate away from it.

---

## Step 1 — User Input

A search/input bar in the top bar. Always accessible. PM types a natural language question or command and submits.

---

## Step 2 — Intent Router

**Method:** LLM classification. Rule-based keyword matching was considered and rejected — too brittle for natural language variation (e.g. "show me churn", "what's happening with churn rate", "which customers are leaving" all express the same intent but look different to a regex). LLM classification handles natural language variation naturally.

**Five intent categories:**

| Intent | Description | Example |
|--------|-------------|---------|
| **Show / Visualisation** | Surface a specific metric or layer output | "Show churn rate", "Show me session time by segment" |
| **Comparison** | Compare segments, periods, or metrics | "Compare Power Users vs Deal Seekers on retention" |
| **Root Cause** | Explain why something changed | "Why did session time increase for this segment?", "Why did churn increase for Power Users?" |
| **Prediction** | Forward-looking questions | "Which customers are likely to migrate next quarter?" |
| **Out of scope** | Not answerable from available layer data | Returns a hard stop — no approximation attempted |

---

## Step 3 — Result Routing per Intent

### Show / Visualisation
Routes to the relevant existing layer output. The result is the actual layer panel embedded in the overlay — no new rendering, no custom views generated.

Examples:
- "Show churn rate" → L6 Outcome table filtered to churn_rate
- "Show me session time by segment" → L2 behavioral cluster profiles
- "Show this quarter's drift" → L2 quarterly drift section

### Comparison
Routes to L7 Metric Integration with the relevant metric and segment parameters pre-populated.

### Root Cause
**Two-path logic — confirmed:**

1. **If a quarterly narration exists** for the relevant period (from Agent 1, stored in `quarterly_narrations` table): LLM reads the narration and extracts the relevant section addressing the PM's specific question. Returns that extracted explanation.

2. **If no quarterly narration exists** for the relevant period: LLM reads live layer outputs directly — L2 drift results, L6 outcome metrics, L8 feature signals, L3 statistical test results — for the relevant segment and period, and synthesizes an explanation on the spot.

**No dead ends.** The PM never receives a redirect message telling them to go generate something first. Root cause always produces an answer.

### Prediction
Routes to L4 Predictive (per-customer migration predictions) or L5 Migration (population-level transition matrix) depending on whether the question is individual-level or population-level.

### Out of scope
Returns a hard stop message. No approximation, no partial answer.

---

## Result Overlay

**Behavior:** Opens on top of the current content area when a query is submitted. The PM's active layer panel remains underneath.

**Contents:**
- The answer/result (embedded existing layer output, or synthesized root cause explanation)
- Two action options as placeholders:
  - **Download as PDF report**
  - **Email this report**

---

## Governing Principles (inherited from existing LLM architecture)

- LLM routes and synthesizes — it does not make decisions or generate new data
- Every answer is traceable to an actual pipeline output or the quarterly narration
- No causal language unless statistically supported by Layer 3 output
- No custom rendering — existing layer outputs are reused, not regenerated
- Out of scope queries return a hard stop — no approximation attempted

---

## Step 3 — Execution Planner

The execution planner sits between the intent router and the actual answer. It does not answer the question — it produces a shopping list of what data needs to be gathered before any answer can be generated.

**Example:**
PM asks "Why did churn increase for Power Users?"
Intent Router → Root Cause
Execution Planner produces:
- L2 — cluster drift results for Power Users cluster
- L3 — statistical significance tests on churn outcome
- L5 — migration matrix showing where Power Users moved
- L6 — churn/outcome history for Power Users
- L8 — feature-level drift signals
- A1 — quarterly narration if available

**Decision 1 — Who builds the shopping list:** Predefined templates per intent type. No LLM call needed at this step — cheaper, faster, and more predictable than dynamic LLM-generated shopping lists.

### Predefined Templates per Intent Type

**Show / Visualisation** — fetches from the single most relevant layer:
- Demographic question → L1
- Cluster/behavioral question → L2
- Statistical significance question → L3
- Migration/churn prediction question → L4
- Transition/flow question → L5
- Outcome metrics (churn, retention, revenue, repeat purchase) → L6
- Metric comparison across time → L7
- Early warning/drift question → L8
- Narration/briefing question → A1

**Comparison** — fetches:
- L7 (metric comparison table with specified metric, segments, granularity)
- L6 (outcome metrics for the segments being compared)

**Root Cause** — always fetches:
- L1 (demographic shifts that may explain the change)
- L2 (cluster drift, adopt/hold decisions, splits/merges)
- L3 (statistical significance of the change)
- L5 (migration flows showing where customers moved)
- L6 (outcome metrics showing the actual impact)
- L8 (feature-level drift signals showing early warning signs)
- A1 (quarterly narration if available for the relevant period)

**Prediction** — fetches:
- L4 (per-customer binary migration predictions)
- L5 (population-level transition matrix)
- L8 (feature-level drift signals as leading indicators)

**Out of scope** — no fetch, immediate hard stop.

**Decision 2 — Parallel vs sequential fetch:** Hybrid approach based on intent type.
- **Parallel:** Show/Visualisation and Comparison — layers are independent of each other, parallel is faster with no benefit to waiting.
- **Sequential:** Root Cause and Prediction — earlier results inform later fetches (e.g. L2 drift identifies which specific cluster drifted → L3 tests that specific cluster → L6 shows outcome impact for that cluster → L8 confirms feature-level signals for that cluster). Running these in parallel would mean later fetches can't be scoped to what earlier fetches revealed.

**Decision 3 — Unavailable data handling:** Use cached/historical data as fallback. If any layer's data is unavailable for the requested period, the system pulls from the most recent period that has data. The PM is explicitly informed in the overlay result which layers used historical data and from which period. No hard stops, no blocked answers — always produces a response with clear labeling of data recency.

## Step 4 — Execution Engine

The execution engine takes the planner's JSON shopping list and runs each step mechanically. No reasoning, no decision making, no LLM — purely a dispatcher that maps each `{layer, action}` pair to the corresponding existing backend endpoint.

**Example input from planner:**
```json
{
    "steps": [
        {"layer": "Layer6", "action": "retrieve_churn"},
        {"layer": "Layer5", "action": "retrieve_migration"},
        {"layer": "Layer3", "action": "retrieve_significance"}
    ]
}
```

**Layer → endpoint mapping:**
- `Layer1 → retrieve_demographics` → `GET /api/v1/demographic-segments`
- `Layer2 → retrieve_drift` → `GET /api/v1/clusters/drift`
- `Layer3 → retrieve_significance` → `GET /api/v1/statistical-tests`
- `Layer4 → retrieve_predictions` → `GET /api/v1/migrations`
- `Layer5 → retrieve_migration` → `GET /api/v1/transition-matrix`
- `Layer6 → retrieve_churn` → `GET /api/v1/segments/{id}/outcomes`
- `Layer7 → retrieve_comparison` → `GET /api/v1/metrics/comparison`
- `Layer8 → retrieve_signals` → `GET /api/v1/feature-migration-signals`
- `A1 → retrieve_narration` → `GET /api/v1/narrations`

All existing endpoints — no new data logic built here. The execution engine is purely a dispatcher.

---

## Step 5 — Results Aggregator

Takes all individual layer outputs from the execution engine and merges them into one structured JSON object. Still no LLM, still no English — purely structural combination.

**Example output:**
```json
{
    "layer6": {"churn": 8.4},
    "layer5": {"migration_rate": 12.3},
    "layer3": {"p_value": 0.01}
}
```

**Key properties:**
- **No transformation** — data from each layer goes in exactly as returned, no modification
- **Layer-keyed** — each layer's output stored under its layer key
- **Handles partial results** — if a layer used historical data (Decision 3 in execution planning), the aggregator includes a metadata flag on that layer's entry indicating which period the data came from
- **No LLM** — purely structural combination

---

## Step 6 — Answer Generator

Only at this step does the LLM come back. It receives the fully aggregated JSON and generates a plain language answer.

**Example input:**
```json
{
    "layer6": {"churn": 8.4},
    "layer5": {"migration_rate": 12.3},
    "layer3": {"p_value": 0.01}
}
```

**Example output:**
"Churn increased to 8.4% last quarter. During the same period, 12.3% of Loyalists migrated to another segment. Statistical testing indicates this change is significant (p = 0.01)."

**Governing constraints (consistent with existing LLM architecture):**
- Every claim must map to a value in the aggregated JSON — no inference beyond what the data shows
- No causal language unless L3's p-value confirms statistical significance
- No intensifiers ("significantly", "dramatically") unless statistically backed
- Historical data explicitly labeled in the answer if any layer was flagged as historical
- The existing `narration_evaluator.py` runs on the output before it reaches the PM — same causal language check, same intensifier guard, same traceability check. No new evaluation logic needed.

---

## Step 7 — Overlay Rendering

The overlay opens on top of the current content area. Structure top to bottom:

**1. Plain language answer** — text card at the top. What the LLM generated. This is what the PM reads first.

**2. Supporting data** — below the answer, the actual layer outputs that backed it up, rendered in their natural format:
- Table → if data came from L6 or L7
- Matrix → if data came from L5
- Stat row → if data came from L3
- Drift list → if data came from L8
- Each panel labeled with its source layer (e.g. "L6 · Outcome", "L5 · Migration")

**3. Historical data warning** — if any layer used cached/historical data, a visible banner stating which layers and from which period.

**4. Action buttons** — Download as PDF, Email this report.

---

## Open Items

- Fallback layer — separate architectural addition, documented separately

---

## Decision Log

| Decision | Outcome | Reason |
|----------|---------|--------|
| Entry point location | Top bar only | Always accessible without interrupting active layer view |
| Result display | Overlay on current content area | Non-disruptive, preserves context |
| Intent routing method | LLM classification | More robust than rule-based keyword matching for natural language variation |
| Root cause data source | Agent 1 narration if available, live layer data if not | No dead ends for PM; always produces an answer |
| Agent 2 relationship | Ask Your Data absorbs and replaces A2 | Eliminates redundancy between two similar interfaces |
