# Create Custom Segments — Architectural Specification

This document captures the confirmed architectural decisions for the Create Custom Segments mode. It is a living document — updated as decisions are made during brainstorming sessions.

---

## Overview

Create Custom Segments is a PM-facing mode that allows building ad-hoc segment definitions on top of existing data, independent of the K-means clustering pipeline. Segments can be session-only or saved permanently to the database with optional scheduling.

---

## Entry Points

**Create Custom Segments** — left sidebar item, opens a new window with the header "Welcome to Build Custom Segments."

**Saved Custom Segments** — separate left sidebar item, shows all previously saved segment definitions. PM can re-run, edit, or delete saved segments from here.

---

## Step 1 — Define Segments

PM selects data points from available columns to define one or more custom segments.

### Segment Naming
- PM can name each custom segment (e.g. "High Value Dormant", "Young Discount Hunters")
- No hard limit on number of segments

### Numeric Columns
- Presented as quartile-based range options:
  - Bottom 25%
  - Middle 50%
  - Top 25%
  - Add Custom Range
- Add Custom Range opens a slider for the PM to set exact boundaries

### Categorical Columns
- Presented as a checklist of all available values in that column
- PM checks/unchecks values to include in the segment definition

### Multiple Segments
- A (+) button allows the PM to add additional custom segment definition panels
- Overlaps are permitted — a customer can belong to more than one custom segment simultaneously
- Each segment panel has its own independent set of column conditions

### Navigation
- Back button available at all times to return to Step 1 and modify segment definitions

---

## Step 2 — Select Comparison Options

PM selects what analysis to run on the defined custom segment(s). Multiple options can be selected simultaneously. PM can select time period per option.

**Export:** Every option's result has its own Download PDF and Email Report action buttons — not a single export for the whole page.

### Option 1 — Compare with Baseline Segments
- PM selects which file/time period to compare against
- Table shows how the custom segment's selected columns sit relative to the existing K-means baseline clusters
- PM selects which columns appear in the comparison table

### Option 2 — How Aggregates Changed
- Uses T snapshot (file 1) vs T+1 snapshot (file 2)
- Shows how the custom segment's selected columns changed between the two periods
- PM selects which columns appear in the comparison table

### Option 3 — Compare Custom Segments (visible only when 2+ custom segments exist)
- Only appears when the PM has created more than one custom segment via the (+) button
- Table shows selected columns side by side across all custom segments the PM built
- Same column-selection approach as Options 1 and 2
- PM selects which columns appear in the comparison table

### Option 4 — L3 Statistical Testing
- Runs significance tests between custom segments (ANOVA, Kruskal-Wallis, Chi-square — same logic as the existing L3 pipeline)
- Shows whether differences between custom segments are statistically real, not just noise
- Requires 2+ custom segments to run
- PM selects time period

### Option 5 — L6 Outcome Analysis
- Computes churn rate, retention rate, revenue, repeat purchase probability for the custom segment
- Shows whether the custom segment performs differently on business outcomes compared to baseline
- PM selects time period

### Option 6 — L7 Metric Integration
- Comparison table with granularity/metric/time selectors — same controls as the existing L7 dashboard panel
- Granularity selector applies to custom segments (daily/weekly/monthly/quarterly/yearly)
- Gap row applies if exactly 2 custom segments are selected (consistent with confirmed L7 gap row rule)

### Option 7 — L8 Feature Signal
- Checks whether customers in the custom segment are showing early-warning drift toward other segments or K-means clusters
- Uses existing Layer 8 median-comparison logic applied to the custom segment population
- PM selects time period

### Column Selection for Comparison Table (Options 1, 2, 3)
- PM selects which columns appear in the comparison table — not a fixed set of outcome metrics
- Flexible, PM-controlled — any column from the dataset can be selected

---

## Saving & Scheduling

- PM can save a custom segment — stored in database with segment name and column conditions
- Saved segments accessible via the "Saved Custom Segments" left sidebar item
- PM can schedule a saved segment to run automatically at a selected granularity (monthly or custom)
- At each scheduled run, the system re-evaluates which customers fall into the saved segment definition against the latest data and runs the PM's previously selected options (1-7)
- When a scheduled run completes, a notification appears in the notification center
- No confirmation prompt when closing without saving — window closes immediately

---

## Key Constraints

- Overlaps allowed — segments are not mutually exclusive
- No Step 3 — no actions beyond PDF/email export per result
- Back button — PM can return to Step 1 at any time to modify definitions
- No hard limit on number of segments per session

---

## Decision Log

| Decision | Outcome |
|----------|---------|
| Entry point | Left sidebar item, opens new window |
| Saved segments entry point | Separate left sidebar item "Saved Custom Segments" |
| Segment naming | PM names each segment, no hard limit on number |
| Numeric column presentation | Quartile-based ranges (Bottom 25%, Middle 50%, Top 25%) + Add Custom Range slider |
| Categorical column presentation | Checklist of available values |
| Multiple segments | (+) button to add more, overlaps permitted |
| Back button | Available to return to Step 1 at any time |
| Comparison options | Multiple can be selected simultaneously |
| Time period selection | PM selects per option |
| Option 3 visibility | Only shown when 2+ custom segments exist |
| Options 4-7 | L3 Statistical Testing, L6 Outcome Analysis, L7 Metric Integration, L8 Feature Signal |
| L7 granularity | Applies to custom segments same as baseline |
| Export per result | Every option has its own Download PDF and Email Report buttons |
| Comparison table columns | PM-selected, not a fixed metric set |
| Segment persistence | Optional — PM can save to database |
| Scheduling | PM can schedule saved segments at monthly or custom granularity |
| Scheduled run notification | Notification center alert on completion |
| Window close without saving | Closes immediately, no confirmation prompt |
| Step after results | None — PDF/email export is the final action |
