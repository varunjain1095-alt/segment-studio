import os
import logging
import numpy as np
import pandas as pd
from src.utils.logging_config import setup_logging

setup_logging()
logger = logging.getLogger("feature_engineer")

# Configurable threshold, defaulting to 0.9
CORRELATION_THRESHOLD = float(os.getenv("FE_CORRELATION_THRESHOLD", 0.90))

def prune_redundant_features(df: pd.DataFrame, threshold: float = CORRELATION_THRESHOLD) -> pd.DataFrame:
    """
    Computes Pearson correlation matrix on numeric columns.
    For any pair with |r| > threshold, drops the column with lower variance.
    Logs column drops.
    """
    logger.info(f"Running redundant feature pruning (threshold={threshold})...")
    df_pruned = df.copy()
    numeric_cols = df_pruned.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) < 2:
        return df_pruned
        
    corr_matrix = df_pruned[numeric_cols].corr(method='pearson')
    
    # Track columns to drop
    dropped_cols = set()
    protected_cols = {"Returned_Ever", "Churned", "customer_id"}
    
    for i in range(len(numeric_cols)):
        for j in range(i + 1, len(numeric_cols)):
            col1 = numeric_cols[i]
            col2 = numeric_cols[j]
            
            if col1 in dropped_cols or col2 in dropped_cols:
                continue
                
            r = corr_matrix.loc[col1, col2]
            if abs(r) > threshold:
                if col1 in protected_cols and col2 in protected_cols:
                    continue
                if col1 in protected_cols:
                    col_to_drop = col2
                    var_other = df_pruned[col1].var()
                    var_drop = df_pruned[col2].var()
                    reason = f"Dropped '{col2}' (variance={var_drop:.4f}) due to high correlation with protected '{col1}' (r={r:.4f}, variance={var_other:.4f})"
                elif col2 in protected_cols:
                    col_to_drop = col1
                    var_other = df_pruned[col2].var()
                    var_drop = df_pruned[col1].var()
                    reason = f"Dropped '{col1}' (variance={var_drop:.4f}) due to high correlation with protected '{col2}' (r={r:.4f}, variance={var_other:.4f})"
                else:
                    # Compare variances to determine which to drop (drop lower variance)
                    var1 = df_pruned[col1].var()
                    var2 = df_pruned[col2].var()
                    
                    # Default drop first if equal
                    if var1 < var2:
                        col_to_drop = col1
                        reason = f"Dropped '{col1}' (variance={var1:.4f}) due to high correlation with '{col2}' (r={r:.4f}, variance={var2:.4f})"
                    else:
                        col_to_drop = col2
                        reason = f"Dropped '{col2}' (variance={var2:.4f}) due to high correlation with '{col1}' (r={r:.4f}, variance={var1:.4f})"
                    
                dropped_cols.add(col_to_drop)
                logger.info(reason)
                
    if dropped_cols:
        df_pruned = df_pruned.drop(columns=list(dropped_cols))
        
    return df_pruned

def encode_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Categorical Encoding:
    - <= 4 unique values -> one-hot encode.
    - > 4 unique values -> frequency encode.
    """
    logger.info("Running categorical encoding...")
    df_encoded = df.copy()
    cat_cols = df_encoded.select_dtypes(include=['object', 'category']).columns.tolist()
    cat_cols = [c for c in cat_cols if c not in ["customer_id", "signup_date"]]
    
    for col in cat_cols:
        unique_vals = df_encoded[col].nunique(dropna=True)
        
        if unique_vals <= 4:
            logger.info(f"One-hot encoding column '{col}' ({unique_vals} unique values)...")
            # Apply get_dummies
            dummies = pd.get_dummies(df_encoded[col], prefix=col, dtype=float)
            df_encoded = pd.concat([df_encoded, dummies], axis=1)
            df_encoded = df_encoded.drop(columns=[col])
        else:
            logger.info(f"Frequency encoding column '{col}' ({unique_vals} unique values)...")
            # Apply frequency encoding
            freq = df_encoded[col].value_counts(normalize=True)
            df_encoded[col] = df_encoded[col].map(freq).astype(float)
            
    return df_encoded
