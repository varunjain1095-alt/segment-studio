Please read this file in full before writing any UI code. It contains the complete visual specification for the Segmentation Studio dashboard. Here is a summary of what it covers:

### Data Ingestion Flow (6-step wizard, runs before dashboard loads):

- **Step 1: Data source selection** (file upload vs. database connection, drop zone swaps for connection form on DB path)
- **Step 2: Column confirmation** (include/exclude checkboxes, type, missing % with color coding — columns >50% missing auto-suggested for exclusion)
- **Step 3: Imputation per column** (only columns with gaps appear; per-column method dropdown with options: mean, median, mode, forward-fill, drop — KNN and regression explicitly excluded from UI)
- **Step 4: Cross-field rule approval** (per-rule approve/reject with edit option, "Approve all" shortcut)
- **Step 5: Classify demographic fields** (LLM badge visible, proposed classification shown, PM confirms via OK toggle — no confidence score column)
- **Step 6: Confirm behavioral columns for Layer 8 drift monitoring** (toggle per column, "Finish setup →" completes the wizard)

### Main Dashboard — single page, sidebar navigation:

- **Left sidebar**: `SEG · STUDIO` branding, navigation items `00 Setup`, `L1–L8` (layers), `A1 Narration`, `A2 Ask a question`. No global search in the sidebar.
- **Top bar**: active layer label + title on left; Agent 2 search bar ("ask about any segment...") + notification bell with badge + user avatar on right.
- **Content area**: swaps based on active sidebar item — each layer renders its own panel inside the same content area.

### Layer panels (content area):

- **00 Setup**: post-ingestion audit trail of all Layer 0 actions taken (columns confirmed, imputation methods, outlier treatments, cross-field rules, demographic classifications). Option to add additional files or run a new project.
- **L1 Demographic**: segment performance table (segment name, size, MoM % change with directional color coding, trend sparkline, volatility flag). Stacked segment-share chart below.
- **L2 Behavioral**: three sections — cluster quality gate cards (Stability/Separation/Noise with PASS/WATCH badges and threshold labels), LLM cluster profiles with PM naming fields on the right ("accept LLM name or write your own"), quarterly drift section showing MERGE/SPLIT/medoid-shift events and ADOPT/HOLD decision.
- **L3 Statistical**: significance table per segment pair (PAIR, METRIC, p-value, effect size d, VERDICT badge — DIVERGENT or SIMILAR). Outcome variables only — no behavioral clustering inputs permitted as test metrics.
- **L4 Predictive**: three sections — model selection result (winning model name, PR-AUC score on minority migrated=True class, comparison of all four candidates' PR-AUC scores), migration label distribution (migrated=True vs migrated=False count for current T→T+1 period), per-customer migration predictions table (customer ID, current cluster, predicted destination cluster).
- **L5 Migration**: transition probability matrix (from/to cluster heatmap, diagonal = self-transitions highlighted, rows sum to 1.0, hover cell for count + confidence interval).
- **L6 Outcome**: segment → outcome mapping table (SEGMENT, CHURN, RETAIN, REV/U, REPEAT columns — four metrics only, no profitability). Each cell links to its cohort curve.
- **L7 Metric Integration**: three PM-controlled selectors (GRANULARITY dropdown, METRIC dropdown, SEGMENTS multi-select with removable chips). Comparison table with segment rows × time-period columns. Gap row (labeled "Δ Gap · 2-SEG ONLY") appears only when exactly 2 segments are selected, absent for 3+.
- **L8 Feature Signal**: per-customer early-warning drift list (customer ID, sparkline trajectory, drift fraction score, HIGH/MED/LOW badge). Click → customer timeline detail.
- **A1 Narration**: quarterly briefing card (period label, generated-for and date metadata, Agent 1 badge, narrative text with key callouts highlighted, Regenerate and Export buttons).
- **A2 Ask a question**: search bar ("Ask anything about your segments..."), pre-populated example query chips (e.g. "Which segment is churning fastest?", "Compare Power Users vs Deal Seekers", "Show me this quarter's drift"), query result panel showing routed layer tag (e.g. "→ routed to L6 · Outcome") and embedded existing layer output — never a new custom view.

### Notification Center:
Pinned to top-right bell icon with unread badge. Dropdown shows notifications across all layers with source tag (L1, L2, A1 etc.), timestamp, and "mark all read" option. Hard-fail notifications are blocking; soft warnings are informational.

### Design system (Curiosity Cue Library — apply throughout):
- **Glowing accents**: active layer pulsing halo, unread badge glow, focused input outer glow.
- **Subtle motion**: floating orbs behind glass, bars ease into place, metrics count up.
- **Reveal interactions**: hover flag → why? tooltip, click cell → drill to cohort, expand row → per-customer list.
- **Exploratory copy**: microcopy that asks questions ("Where's your data hiding?", "What's shifting this quarter?", "Follow the drift.").
- **Visual language**: dark purple palette, mono pills (✦) for behavior/interaction notes, dashed blocks for chart placeholders, striped fills for charts to build.

### Implementation notes from the design file:
- These are structural wireframes, not final pixel-perfect designs — use them as the layout and interaction specification.
- Every layer panel renders inside the same content area — no separate pages or routes.
- Real numbers shown are illustrative placeholders.
- Layer numbering follows L1–L8 (no L4 in the original sidebar, but L4 has been confirmed as a required addition per prior PM decision — please add it between L3 and L5 in the sidebar).
