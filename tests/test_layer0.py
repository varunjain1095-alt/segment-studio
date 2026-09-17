import pytest
import os
import numpy as np
import pandas as pd
import logging
from unittest.mock import patch, MagicMock
from src.db.connection import get_raw_connection
from src.layer0.profiler import data_profile, save_profile_to_db
from src.layer0.cleaning import clean_dataset
from src.layer0.outliers import (
    treat_univariate_numeric_outliers,
    treat_univariate_categorical_outliers,
    treat_bivariate_numeric_outliers,
    check_bivariate_categorical_similarity
)
from src.layer0.imputer import apply_imputation, check_missing_value_limits
from src.layer0.cross_field_validator import approve_rule, get_active_rules
from src.layer0.feature_engineer import prune_redundant_features, encode_categorical_features

def test_p1_t1_data_profile_shape():
    """P1-T1: data_profile() — Correct Output Shape"""
    # 5 columns: 1 all-null numeric, 1 fully unique, 1 with 30% missing, 1 object with mixed numeric strings, 1 all-integer.
    # We use 20 rows to construct clean integer/float/unique values.
    # col_mixed_numeric: 18 numeric strings, 2 non-numeric strings (90% parse as numeric) -> wait, to test suspect numeric
    # let's make it 19/20 numeric strings (95%) so it triggers suspect_numeric_object = True.
    data = {
        "col_all_null": [np.nan] * 20,
        "col_unique": list(range(20)),
        "col_30_missing": [1.0] * 14 + [np.nan] * 6, # 6/20 is 30.0%
        "col_mixed_numeric": ["1"] * 19 + ["A"], # 19/20 is 95% -> suspect_numeric_object = True
        "col_integer": [5] * 20
    }
    df = pd.DataFrame(data)
    profile = data_profile(df)
    
    # Verify columns in returned profile
    expected_cols = {
        "column_name", "missing_count", "missing_pct", "dtype", "suspect_numeric_object",
        "unique_count", "non_unique_count", "non_unique_pct", "is_unique_key", "has_duplicates", "severity"
    }
    assert expected_cols.issubset(set(profile.columns))
    
    # Assert sorted descending by missing_pct
    assert profile.iloc[0]["column_name"] == "col_all_null"
    assert profile.iloc[0]["missing_pct"] == 100.0
    
    # Assert values are arithmetically correct
    row_unique = profile[profile["column_name"] == "col_unique"].iloc[0]
    assert row_unique["unique_count"] == 20
    assert row_unique["is_unique_key"] == True
    
    row_mixed = profile[profile["column_name"] == "col_mixed_numeric"].iloc[0]
    assert row_mixed["suspect_numeric_object"] == True

def test_p1_t2_severity_buckets():
    """P1-T2: data_profile() — Severity Buckets (0%, 20%, 40%, 60%, 90% missing)"""
    data = {
        "col_0": [1.0] * 10,
        "col_20": [1.0] * 8 + [np.nan] * 2,
        "col_40": [1.0] * 6 + [np.nan] * 4,
        "col_60": [1.0] * 4 + [np.nan] * 6,
        "col_90": [1.0] * 1 + [np.nan] * 9
    }
    df = pd.DataFrame(data)
    profile = data_profile(df)
    
    sev_map = profile.set_index("column_name")["severity"].to_dict()
    assert sev_map["col_0"] == "None"
    assert sev_map["col_20"] == "Low"
    assert sev_map["col_40"] == "Moderate"
    assert sev_map["col_60"] == "High"
    assert sev_map["col_90"] == "Critical"

def test_p1_t3_suspect_numeric_object():
    """P1-T3: suspect_numeric_object Detection (95% numeric strings + 5% NaN)"""
    # 95 numeric strings, 5 NaN out of 100 rows
    vals = ["12.5"] * 95 + [np.nan] * 5
    df = pd.DataFrame({"suspect_col": vals})
    profile = data_profile(df)
    
    row = profile[profile["column_name"] == "suspect_col"].iloc[0]
    assert row["suspect_numeric_object"] == True

def test_p1_t4_type_recovery_coercion():
    """P1-T4: Type Recovery Coercion (92% numeric-coercible, 8% NaN)"""
    # Column has 92% numeric-coercible strings, 8% NaN
    vals = ["42"] * 92 + [np.nan] * 8
    df = pd.DataFrame({"target_col": vals})
    
    # Call clean_dataset (which executes type recovery and profiling)
    dataset_id = "00000000-0000-0000-0000-000000000004"
    with patch("src.layer0.cleaning.save_profile_to_db"):
        df_clean, profile, duplicates = clean_dataset(df, dataset_id)
        
    assert pd.api.types.is_numeric_dtype(df_clean["target_col"])
    assert df_clean["target_col"].dtype == "float64"
    
    # Re-profiled check
    row = profile[profile["column_name"] == "target_col"].iloc[0]
    assert row["missing_count"] == 8
    assert row["missing_pct"] == 8.0

def test_p1_t5_negative_value_flagging():
    """P1-T5: Negative Value Flagging — Unconditional, No Per-Column Exception"""
    df = pd.DataFrame({"account_delta": [-50, 100, 200]})
    
    # Emit checks
    with patch("src.layer0.cleaning.emit") as mock_emit:
        with patch("src.layer0.cleaning.save_profile_to_db"):
            clean_dataset(df, "00000000-0000-0000-0000-000000000005")
        
    mock_emit.assert_called_once()
    args, kwargs = mock_emit.call_args
    assert kwargs.get("condition") == "negative_value_flagged"
    assert kwargs.get("severity") == "soft_warning"
    assert "account_delta" in kwargs.get("details")

def test_p1_t6_rare_category_detection():
    """P1-T6: Rare-Category Detection at the 1.5% Threshold (1.5%, 1.49%, 1.51%)"""
    # Column has 10,000 rows
    # "X": 150 (1.50%) -> Excluded
    # "Y": 149 (1.49%) -> Excluded
    # "Z": 151 (1.51%) -> Retained
    # "Other": 9,550 -> Retained
    rows = []
    rows.extend(["X"] * 150)
    rows.extend(["Y"] * 149)
    rows.extend(["Z"] * 151)
    rows.extend(["Other"] * (10000 - len(rows)))
    
    df = pd.DataFrame({"cat_col": rows})
    
    with patch("src.layer0.outliers.emit") as mock_emit:
        df_clean = treat_univariate_categorical_outliers(df)
        
    unique_vals = df_clean["cat_col"].unique()
    assert "Z" in unique_vals, "Z (1.51%) should be retained"
    assert "Other" in unique_vals, "Other should be retained"
    assert "X" not in unique_vals, "X (1.50% - boundary) should be excluded"
    assert "Y" not in unique_vals, "Y (1.49% - below) should be excluded"
    assert mock_emit.call_count == 2 # for X and Y

def test_p1_t7_iqr_outlier_transform():
    """P1-T7: IQR Outlier Treatment — Correct Transform Selection"""
    # 1. Genuinely skewed positive column: min > 0, skew > 1.0 -> log1p + RobustScaler
    col_skew_pos = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 100.0]
    # 2. Symmetric positive column: min > 0, skew <= 1.0 -> RobustScaler only (no power transform)
    col_sym_pos = [10.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 38.0]
    # 3. Crossing negative skewed column: contains <= 0, skew > 1.0 -> Yeo-Johnson + RobustScaler
    col_skew_neg = [-10.0, -9.9, -9.8, -9.7, -9.6, -9.5, -9.4, -9.3, -9.2, -9.1, 100.0]
    
    df = pd.DataFrame({
        "skew_pos": col_skew_pos,
        "sym_pos": col_sym_pos,
        "skew_neg": col_skew_neg
    })
    
    df_treated = treat_univariate_numeric_outliers(df)
    
    # A. Verify sym_pos: scaled directly without log/power transform
    col_sym = df["sym_pos"]
    expected_sym = (col_sym - col_sym.median()) / (col_sym.quantile(0.75) - col_sym.quantile(0.25))
    np.testing.assert_array_almost_equal(df_treated["sym_pos"].values, expected_sym.values)
    
    # B. Verify skew_pos: log1p applied before robust scaling
    col_pos = np.log1p(df["skew_pos"])
    expected_skew_pos = (col_pos - col_pos.median()) / (col_pos.quantile(0.75) - col_pos.quantile(0.25))
    np.testing.assert_array_almost_equal(df_treated["skew_pos"].values, expected_skew_pos.values)
    
    # C. Verify skew_neg: Yeo-Johnson transform applied
    col_neg = df["skew_neg"]
    expected_raw_scaled = (col_neg - col_neg.median()) / (col_neg.quantile(0.75) - col_neg.quantile(0.25))
    # It should not match raw robust scaled since Yeo-Johnson was applied
    assert not np.allclose(df_treated["skew_neg"].values, expected_raw_scaled.values)

def test_p1_t8_missing_value_column_drop_recommendation():
    """P1-T8: Missing Value — >50% Column Drop Recommendation"""
    profile_df = pd.DataFrame([{
        "column_name": "bad_col",
        "missing_pct": 55.0
    }])
    
    # Calling check_missing_value_limits must emit a hard_fail
    with patch("src.layer0.imputer.emit") as mock_emit:
        check_missing_value_limits("00000000-0000-0000-0000-000000000008", profile_df)
        
    mock_emit.assert_called_once()
    args, kwargs = mock_emit.call_args
    assert kwargs.get("severity") == "hard_fail"
    assert "bad_col" in kwargs.get("details")

def test_p1_t9_missing_value_pm_imputation_application():
    """P1-T9: Missing Value — PM Imputation Method Application"""
    # 20% missing numeric
    df = pd.DataFrame({"num": [10.0, 20.0, np.nan, 40.0, 50.0]})
    # Median is 30.0
    
    imputed = apply_imputation(df, "num", "median")
    assert imputed.isna().sum() == 0
    assert imputed.iloc[2] == 30.0

def test_p1_t10_cross_field_rule_pm_approval_gate():
    """P1-T10: Cross-Field Rule — PM Approval Gate"""
    dataset_id = "00000000-0000-0000-0000-000000000010"
    rule_id = "00000000-0000-0000-0000-000000000001"
    
    # Seed database with dataset and proposed rule
    with get_raw_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM cross_field_rules WHERE dataset_id = %s;", (dataset_id,))
            cur.execute("DELETE FROM datasets WHERE dataset_id = %s;", (dataset_id,))
            cur.execute(
                """
                INSERT INTO datasets (dataset_id, source_type, permanence)
                VALUES (%s, 'file_upload', 'permanent');
                """,
                (dataset_id,)
            )
            cur.execute(
                """
                INSERT INTO cross_field_rules (rule_id, dataset_id, rule_description, rule_expression, proposed_by, approved)
                VALUES (%s, %s, 'End Date Check', 'df["end_date"] >= df["start_date"]', 'heuristic', FALSE);
                """,
                (rule_id, dataset_id)
            )
            
    # Set approval to False (PM rejected)
    approve_rule(dataset_id, rule_id, approved=False)
    
    # Check that rule is NOT active
    active = get_active_rules(dataset_id)
    assert 'df["end_date"] >= df["start_date"]' not in active
    
    # Set approval to True (PM approved)
    approve_rule(dataset_id, rule_id, approved=True)
    active = get_active_rules(dataset_id)
    assert 'df["end_date"] >= df["start_date"]' in active

def test_p1_t11_correlation_pruning():
    """P1-T11: Correlation Pruning — Threshold at 0.9 (Lower-variance column is dropped)"""
    # Pearson correlation r = 0.91
    # Col A: high variance
    # Col B: low variance
    col_a = [1.0, 2.0, 3.0, 4.0, 10.0]
    col_b = [1.1, 1.9, 3.0, 3.9, 9.8] # slightly closer, wait, let's verify variances
    # Let's verify exactly
    df = pd.DataFrame({"col_a": col_a, "col_b": col_b})
    var_a = df["col_a"].var()
    var_b = df["col_b"].var()
    assert var_b < var_a
    assert df["col_a"].corr(df["col_b"]) > 0.90
    
    # Call prune_redundant_features
    # Set logging to catch warning/info
    logger = logging.getLogger("feature_engineer")
    with patch.object(logger, 'info') as mock_info:
        df_pruned = prune_redundant_features(df, threshold=0.90)
        
    assert "col_b" not in df_pruned.columns
    assert "col_a" in df_pruned.columns
    
    # Log assertion
    log_messages = [call[0][0] for call in mock_info.call_args_list]
    assert any("Dropped 'col_b'" in msg for msg in log_messages)

def test_p1_t12_categorical_encoding_rules():
    """P1-T12: Categorical Encoding Rules (<= 4 One-Hot vs > 4 Freq)"""
    df = pd.DataFrame({
        "low_card": ["A", "B", "A", "C", "A"], # 3 unique values -> One-hot
        "high_card": ["A", "B", "C", "D", "E"] # 5 unique values -> Freq
    })
    
    df_encoded = encode_categorical_features(df)
    
    # low_card should be one-hot encoded (creating low_card_A, low_card_B, etc)
    assert "low_card" not in df_encoded.columns
    assert "low_card_A" in df_encoded.columns
    assert "low_card_B" in df_encoded.columns
    assert "low_card_C" in df_encoded.columns
    
    # high_card should be frequency encoded (mapped to float counts)
    assert "high_card" in df_encoded.columns
    assert df_encoded["high_card"].dtype == float
