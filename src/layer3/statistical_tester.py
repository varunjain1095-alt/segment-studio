import os
import logging
import itertools
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import scikit_posthocs as sp

from src.db.connection import get_raw_connection

logger = logging.getLogger("layer3_statistical_tester")

# ==============================================================================
# USER REVIEW REQUIRED: Provisional Customer Outcome Formulas
# Placeholder business logic — derived from available dataset columns, pending PM confirmation
# ==============================================================================

# Configurable statistical thresholds
ALPHA_THRESHOLD = 0.05
EFFECT_SIZE_COHENS_D_MIN = 0.20
EFFECT_SIZE_CRAMERS_V_MIN = 0.10
EFFECT_SIZE_ETA_SQUARED_MIN = 0.01

SIMILARITY_P_VALUE_THRESHOLD = 0.80
DIVERGENCE_EFFECT_COHENS_D = 0.50
DIVERGENCE_EFFECT_CRAMERS_V = 0.30
DIVERGENCE_EFFECT_ETA_SQUARED = 0.06

ALLOWED_OUTCOMES = {"churn_rate", "retention_rate", "revenue", "repeat_purchase_prob"}
FALLBACK_PERMITTED_OUTCOMES = {"churn_rate", "retention_rate"}

def is_snapshot_fallback_active(dataset_id: str) -> bool:
    """
    Checks if a dataset lacks confirmed/processed raw CSV files and thus relies on
    the behavioral_snapshots database fallback path.
    """
    try:
        from src.ingestion.router import DATA_DIR
    except ImportError:
        DATA_DIR = "data"
    csv_confirmed = os.path.join(DATA_DIR, f"{dataset_id}_confirmed.csv")
    csv_processed = os.path.join(DATA_DIR, f"{dataset_id}_processed.csv")
    return not (os.path.exists(csv_confirmed) or os.path.exists(csv_processed))

def get_customer_level_outcomes(dataset_id: str, period_id: int) -> pd.DataFrame:
    """
    Retrieves customer-level outcomes.
    
    PRIMARY PATH (Confirmed Raw CSV):
      Maps directly to independent external dataset fields:
      - churn_rate: from Churned (0 or 1)
      - retention_rate: 1.0 - churn_rate
      - revenue: unscaled Lifetime_Value (independent monetary outcome)
      - repeat_purchase_prob: indicator for Total_Purchases > 1
      Provides true independent ground-truth for Layer 3 validation.
      
    FALLBACK PATH (behavioral_snapshots table):
      Used only when external CSV is absent (e.g. integration smoke tests and demo mocks).
      Reconstructed revenue (purchase_count * avg_order_value) and repeat_purchase_prob
      are derived from clustering features and are demo/integration proxies only.
      Only churn_rate and retention_rate (derived from T-to-T+1 customer presence)
      are permitted for Layer 3 anti-circular validation under this fallback.
    """
    logger.info(f"Retrieving customer-level outcomes for period {period_id}...")
    
    # 1. Try to read from the confirmed CSV first (which has unscaled/raw values), falling back to processed
    try:
        from src.ingestion.router import DATA_DIR
    except ImportError:
        DATA_DIR = "data"
        
    csv_path = os.path.join(DATA_DIR, f"{dataset_id}_confirmed.csv")
    if not os.path.exists(csv_path):
        csv_path = os.path.join(DATA_DIR, f"{dataset_id}_processed.csv")
    
    if os.path.exists(csv_path):
        logger.info(f"Loading customer outcomes directly from file: {csv_path}")
        try:
            csv_df = pd.read_csv(csv_path)
            col_map = {col.lower(): col for col in csv_df.columns}
            
            # Extract customer_id
            cid_col = col_map.get("customer_id")
            if cid_col is None:
                raise ValueError("customer_id column not found in processed CSV.")
                
            out_df = pd.DataFrame()
            out_df["customer_id"] = csv_df[cid_col].astype(str)
            
            # Map Churned directly to churn_rate (directly using existing Churned column)
            churned_col = col_map.get("churned")
            if churned_col is not None:
                out_df["churn_rate"] = csv_df[churned_col].fillna(0.0).astype(float)
            else:
                out_df["churn_rate"] = 0.0
                
            # Retention derived as inverse of Churned: retention_rate = 1 - churned (using existing Churned column)
            out_df["retention_rate"] = 1.0 - out_df["churn_rate"]
            
            # Map Lifetime_Value directly to revenue (directly using existing Lifetime_Value column)
            ltv_col = col_map.get("lifetime_value")
            if ltv_col is not None:
                out_df["revenue"] = csv_df[ltv_col].fillna(0.0).astype(float)
                out_df["revenue"] = out_df["revenue"].apply(lambda val: max(0.0, val))
            else:
                out_df["revenue"] = 0.0
                
            # Repeat purchase probability: Use Total_Purchases if exists (P4-T9 Total_Purchases mapping)
            purchases_col = col_map.get("total_purchases") or col_map.get("purchase_count") or col_map.get("purchase_frequency")
            if purchases_col is not None:
                out_df["repeat_purchase_prob"] = (csv_df[purchases_col].fillna(0.0).astype(float) > 1.0).astype(float)
            else:
                out_df["repeat_purchase_prob"] = 0.0
                
            out_df.attrs["is_fallback"] = False
            out_df.attrs["source"] = "confirmed_csv"
            return out_df
        except Exception as e:
            logger.error(f"Failed to load outcomes from CSV: {str(e)}. Falling back to database snapshots.")
            
    # 2. Database fallback path (runs if CSV not found, e.g., in unit tests)
    logger.info("Processed CSV not found. Loading outcomes from database snapshots (Demo/Integration Fallback).")
    with get_raw_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = 'behavioral_snapshots';
                """
            )
            avail_cols = [row[0] for row in cur.fetchall()]
            
            cols_to_select = ["customer_id"]
            if "purchase_count" in avail_cols:
                cols_to_select.append("purchase_count")
            elif "purchase_frequency" in avail_cols:
                cols_to_select.append("purchase_frequency")
                
            if "avg_order_value" in avail_cols:
                cols_to_select.append("avg_order_value")
                
            cols_str = ", ".join([f'"{c}"' for c in cols_to_select])
            cur.execute(
                f"SELECT {cols_str} FROM behavioral_snapshots WHERE period_id = %s;",
                (period_id,)
            )
            rows = cur.fetchall()
            
    if not rows:
        logger.warning(f"No behavioral snapshot records found for period {period_id}.")
        return pd.DataFrame()
        
    df = pd.DataFrame(rows, columns=cols_to_select)
    
    p_col = "purchase_count" if "purchase_count" in df.columns else ("purchase_frequency" if "purchase_frequency" in df.columns else None)
    if p_col:
        df["purchase_count_std"] = df[p_col].fillna(0.0).astype(float)
    else:
        df["purchase_count_std"] = 1.0
        
    if "avg_order_value" in df.columns:
        df["avg_order_value_std"] = df["avg_order_value"].fillna(0.0).astype(float)
    else:
        df["avg_order_value_std"] = 50.0
        
    # Reconstructed revenue (proxy derived from purchase frequency * AOV)
    df["revenue"] = df["purchase_count_std"] * df["avg_order_value_std"]
    
    # Reconstructed repeat purchase probability (proxy derived from purchases > 1)
    df["repeat_purchase_prob"] = (df["purchase_count_std"] > 1.0).astype(float)
    
    # Genuine temporal presence churn & retention: derived from T vs T+1 presence in snapshots
    with get_raw_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT DISTINCT customer_id FROM behavioral_snapshots WHERE period_id = %s;",
                (period_id + 1,)
            )
            next_customers = {row[0] for row in cur.fetchall()}
            
    df["churn_rate"] = df["customer_id"].apply(lambda cid: 0.0 if str(cid) in next_customers else 1.0)
    df["retention_rate"] = df["customer_id"].apply(lambda cid: 1.0 if str(cid) in next_customers else 0.0)
    
    df.attrs["is_fallback"] = True
    df.attrs["source"] = "behavioral_snapshots"
    return df

def validate_anti_circularity(dataset_id: str, test_variable: str, is_fallback: bool = None):
    """
    Enforces the anti-circularity guard (P4-T3, P4-T4).
    Validates that test_variable is not a behavioral feature column used in clustering.
    
    When backed by behavioral_snapshots fallback, allows statistical validation ONLY
    for churn_rate and retention_rate (derived from temporal T-to-T+1 presence).
    Reconstructed revenue and repeat_purchase_prob are excluded from Layer 3 validation
    because their source variables (e.g. purchase_count, avg_order_value) may have
    participated in clustering.
    """
    if test_variable not in ALLOWED_OUTCOMES:
        raise ValueError(
            f"Invalid test variable '{test_variable}'. Under anti-circularity constraints, "
            f"test variable must be an outcome from {ALLOWED_OUTCOMES}."
        )
        
    fallback_active = is_fallback if is_fallback is not None else is_snapshot_fallback_active(dataset_id)
    if fallback_active and test_variable not in FALLBACK_PERMITTED_OUTCOMES:
        raise ValueError(
            f"Anti-circularity violation: Under behavioral_snapshots fallback, '{test_variable}' "
            "is reconstructed from variables that may have participated in clustering "
            "(e.g., purchase_count, avg_order_value). Layer 3 statistical validation allows only "
            f"{FALLBACK_PERMITTED_OUTCOMES} derived from temporal T-to-T+1 presence when using "
            "snapshot fallback. Reconstructed revenue and repeat_purchase_prob are demo/integration proxies only."
        )
        
    with get_raw_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT confirmed_classifications FROM datasets WHERE dataset_id = %s;", (dataset_id,))
            row = cur.fetchone()
            
    if row and row[0]:
        classifications = json_loads_helper(row[0])
        behavioral_cols = [c["column"] for c in classifications if c.get("classification") == "behavioral"]
        if test_variable in behavioral_cols:
            raise ValueError(
                f"Anti-circularity violation: '{test_variable}' is a behavioral feature used in clustering. "
                "Testing clusters on their own inputs is circular and strictly prohibited."
            )

def json_loads_helper(val):
    if isinstance(val, str):
        import json
        return json.loads(val)
    return val

# ==============================================================================
# Core Statistical Math Functions
# ==============================================================================

def compute_anova(groups_data: list[np.ndarray]) -> tuple[float, float, float]:
    """Computes One-way ANOVA and returns (F-stat, p-value, eta-squared)."""
    f_stat, p_val = stats.f_oneway(*groups_data)
    
    # Calculate eta-squared (SS_between / SS_total)
    all_data = np.concatenate(groups_data)
    grand_mean = np.mean(all_data)
    ss_total = np.sum((all_data - grand_mean) ** 2)
    ss_within = sum(np.sum((g - np.mean(g)) ** 2) for g in groups_data)
    ss_between = ss_total - ss_within
    eta_squared = ss_between / ss_total if ss_total != 0.0 else 0.0
    
    return float(f_stat), float(p_val), float(eta_squared)

def compute_kruskal_wallis(groups_data: list[np.ndarray]) -> tuple[float, float, float]:
    """Computes Kruskal-Wallis and returns (H-stat, p-value, epsilon-squared)."""
    h_stat, p_val = stats.kruskal(*groups_data)
    
    # Epsilon-squared = H / (N - 1)
    all_data = np.concatenate(groups_data)
    n = len(all_data)
    epsilon_sq = float(h_stat) / (n - 1) if n > 1 else 0.0
    
    return float(h_stat), float(p_val), epsilon_sq

def compute_chi_square(contingency_table: np.ndarray) -> tuple[float, float, float]:
    """Computes Chi-Square contingency test and returns (chi2-stat, p-value, cramers_v)."""
    chi2_stat, p_val, _, _ = stats.chi2_contingency(contingency_table)
    
    n = contingency_table.sum()
    r, c = contingency_table.shape
    min_dim = min(r - 1, c - 1)
    cramers_v = np.sqrt(chi2_stat / (n * min_dim)) if n > 0 and min_dim > 0 else 0.0
    
    return float(chi2_stat), float(p_val), float(cramers_v)

def compute_cohens_d(group_a: np.ndarray, group_b: np.ndarray) -> float:
    """Computes Cohen's d effect size between group_a and group_b."""
    n_a, n_b = len(group_a), len(group_b)
    if n_a <= 1 or n_b <= 1:
        return 0.0
    mean_a, mean_b = np.mean(group_a), np.mean(group_b)
    var_a, var_b = np.var(group_a, ddof=1), np.var(group_b, ddof=1)
    s_pooled = np.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))
    return float(abs(mean_a - mean_b) / s_pooled) if s_pooled != 0.0 else 0.0

# ==============================================================================
# High-Level Statistical Testing Pipeline
# ==============================================================================

def run_statistical_tests(
    dataset_id: str,
    period_id: int,
    segment_column: str,
    test_variable: str,
    alpha: float = ALPHA_THRESHOLD,
    parametric: bool = True
) -> list[dict]:
    """
    Executes statistical testing on outcomes across demographic groups or clusters.
    
    Parameters:
      dataset_id (str): UUID of the dataset.
      period_id (int): Period ID.
      segment_column (str): Name of demographic column (e.g. 'gender') or 'cluster' for Layer 2.
      test_variable (str): Name of outcome column.
      alpha (float): Significance p-value alpha threshold.
      parametric (bool): If True, runs ANOVA + Tukey. If False, runs Kruskal-Wallis + Dunn's.
      
    Returns:
      results (list[dict]): Statistical comparison results.
    """
    # 1. Get customer outcome metrics
    outcomes_df = get_customer_level_outcomes(dataset_id, period_id)
    if outcomes_df.empty:
        logger.warning("Empty customer outcomes. Aborting test execution.")
        return []
        
    is_fallback = bool(outcomes_df.attrs.get("is_fallback", False))
    
    # 2. Enforce anti-circularity checks (strictly forbidding reconstructed proxy variables under snapshot fallback)
    validate_anti_circularity(dataset_id, test_variable, is_fallback=is_fallback)
    
    # Convert outcomes customer_id to string to allow proper merge
    outcomes_df["customer_id"] = outcomes_df["customer_id"].astype(str)
    
    # 3. Load segment group assignments
    if segment_column.lower() == "cluster":
        # Load active cluster assignments from DB
        with get_raw_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT ca.customer_id, ca.cluster_id, c.cluster_label
                    FROM cluster_assignments ca
                    JOIN clusters c ON ca.cluster_id = c.cluster_id
                    WHERE ca.period_id = %s;
                    """,
                    (period_id,)
                )
                rows = cur.fetchall()
        if not rows:
            logger.warning(f"No cluster assignments found for period {period_id}.")
            return []
        segment_df = pd.DataFrame(rows, columns=["customer_id", "group_id", "group_label"])
    else:
        # Load demographic group assignments from DB
        with get_raw_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f'SELECT customer_id, "{segment_column}" FROM demographic_snapshots WHERE period_id = %s;',
                    (period_id,)
                )
                rows = cur.fetchall()
        if not rows:
            logger.warning(f"No demographic snapshot column '{segment_column}' found for period {period_id}.")
            return []
        segment_df = pd.DataFrame(rows, columns=["customer_id", "group_label"])
        segment_df["group_id"] = None
        
    segment_df["customer_id"] = segment_df["customer_id"].astype(str)
    
    # 4. Merge groups and outcomes
    merged_df = pd.merge(segment_df, outcomes_df, on="customer_id", how="inner")
    if merged_df.empty:
        logger.warning("Merged dataset is empty. Check mapping keys.")
        return []
        
    # Drop rows containing missing values in test_variable
    merged_df = merged_df.dropna(subset=[test_variable, "group_label"])
    
    groups = sorted(merged_df["group_label"].unique().tolist())
    if len(groups) < 2:
        logger.warning(f"Fewer than 2 active groups found for '{segment_column}'. Cannot test.")
        return []
        
    # Build list of arrays per group
    groups_data = [merged_df[merged_df["group_label"] == g][test_variable].values for g in groups]
    
    # Filter out empty groups
    groups_data = [g for g in groups_data if len(g) > 0]
    if len(groups_data) < 2:
        return []
        
    # Initialize DB rows to write
    records = []
    
    # 5. Run Omnibus Test (ANOVA or Kruskal-Wallis)
    test_type = "anova" if parametric else "kruskal_wallis"
    
    try:
        if parametric:
            stat, p_val, eff_size = compute_anova(groups_data)
            eff_type = "eta_squared"
            eff_threshold = EFFECT_SIZE_ETA_SQUARED_MIN
        else:
            stat, p_val, eff_size = compute_kruskal_wallis(groups_data)
            eff_type = "eta_squared" # using epsilon-squared mapped as eta-squared
            eff_threshold = EFFECT_SIZE_ETA_SQUARED_MIN
            
        # Large-sample bias guard: significance requires p < alpha AND effect size >= threshold
        is_significant = (p_val < alpha) and (eff_size >= eff_threshold)
        
        # Volatility & similarity flags
        flag = None
        if p_val >= SIMILARITY_P_VALUE_THRESHOLD:
            flag = "critical_similarity"
        elif is_significant and (eff_size >= DIVERGENCE_EFFECT_ETA_SQUARED):
            flag = "critical_divergence"
            
        records.append({
            "test_type": test_type,
            "segment_a_label": f"Omnibus ({segment_column})",
            "segment_b_label": None,
            "p_value": p_val,
            "test_statistic": stat,
            "effect_size": eff_size,
            "effect_size_type": eff_type,
            "significant": is_significant,
            "flag_type": flag
        })
    except Exception as e:
        logger.error(f"Omnibus test failed: {str(e)}", exc_info=True)
        return []
        
    # 6. Run Pairwise Post-Hoc Tests
    # For numeric outcomes, we run pairwise post-hoc tests (Tukey HSD or Dunn's test)
    try:
        val_array = merged_df[test_variable].values
        group_array = merged_df["group_label"].values
        
        # Mapping group label string to cluster ID if cluster comparison
        label_to_id = {}
        if segment_column.lower() == "cluster":
            label_to_id = {row["group_label"]: row["group_id"] for _, row in segment_df.iterrows()}
            
        if parametric:
            # Tukey HSD
            tukey = pairwise_tukeyhsd(endog=val_array, groups=group_array, alpha=alpha)
            pairs = list(itertools.combinations(groups, 2))
            
            for idx, (g1, g2) in enumerate(pairs):
                # statsmodels tukey HSD orders pairs alphabetically
                # find match index
                p_val = float(tukey.pvalues[idx])
                stat_val = float(abs(tukey.meandiffs[idx]))
                
                # Compute Cohen's d
                val1 = merged_df[merged_df["group_label"] == g1][test_variable].values
                val2 = merged_df[merged_df["group_label"] == g2][test_variable].values
                cohens_d = compute_cohens_d(val1, val2)
                
                # Significance: both p-value and Cohen's d threshold met
                is_sig = (p_val < alpha) and (cohens_d >= EFFECT_SIZE_COHENS_D_MIN)
                
                flag = None
                if p_val >= SIMILARITY_P_VALUE_THRESHOLD:
                    flag = "critical_similarity"
                elif is_sig and (cohens_d >= DIVERGENCE_EFFECT_COHENS_D):
                    flag = "critical_divergence"
                    
                records.append({
                    "test_type": "tukey_hsd",
                    "segment_a_label": g1,
                    "segment_b_label": g2,
                    "segment_a_id": label_to_id.get(g1),
                    "segment_b_id": label_to_id.get(g2),
                    "p_value": p_val,
                    "test_statistic": stat_val,
                    "effect_size": cohens_d,
                    "effect_size_type": "cohens_d",
                    "significant": is_sig,
                    "flag_type": flag
                })
        else:
            # Dunn's test
            # Dunn's test using scikit-posthocs
            dunn_df = sp.posthoc_dunn(merged_df, val_col=test_variable, group_col="group_label", p_adjust="holm")
            pairs = list(itertools.combinations(groups, 2))
            
            for g1, g2 in pairs:
                p_val = float(dunn_df.loc[g1, g2])
                
                # Compute Cohen's d for pairwise effect size
                val1 = merged_df[merged_df["group_label"] == g1][test_variable].values
                val2 = merged_df[merged_df["group_label"] == g2][test_variable].values
                cohens_d = compute_cohens_d(val1, val2)
                
                is_sig = (p_val < alpha) and (cohens_d >= EFFECT_SIZE_COHENS_D_MIN)
                
                flag = None
                if p_val >= SIMILARITY_P_VALUE_THRESHOLD:
                    flag = "critical_similarity"
                elif is_sig and (cohens_d >= DIVERGENCE_EFFECT_COHENS_D):
                    flag = "critical_divergence"
                    
                records.append({
                    "test_type": "dunns",
                    "segment_a_label": g1,
                    "segment_b_label": g2,
                    "segment_a_id": label_to_id.get(g1),
                    "segment_b_id": label_to_id.get(g2),
                    "p_value": p_val,
                    "test_statistic": None, # statistic not easily mapped per-pair from omnibus
                    "effect_size": cohens_d,
                    "effect_size_type": "cohens_d",
                    "significant": is_sig,
                    "flag_type": flag
                })
    except Exception as e:
        logger.error(f"Pairwise post-hoc test failed: {str(e)}", exc_info=True)
        
    # 7. Persist Results to DB
    with get_raw_connection() as conn:
        with conn.cursor() as cur:
            for rec in records:
                cur.execute(
                    """
                    INSERT INTO statistical_test_results (
                        period_id, test_type, segment_a_id, segment_b_id,
                        test_variable, p_value, test_statistic, effect_size,
                        effect_size_type, significant, flag_type
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
                    """,
                    (
                        period_id,
                        rec["test_type"],
                        rec.get("segment_a_id"),
                        rec.get("segment_b_id"),
                        test_variable,
                        rec["p_value"],
                        rec["test_statistic"],
                        rec["effect_size"],
                        rec["effect_size_type"],
                        rec["significant"],
                        rec["flag_type"]
                    )
                )
                
    logger.info(f"Persisted {len(records)} test results to database.")
    return records
