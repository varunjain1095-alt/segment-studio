import logging
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.linear_model import LinearRegression
from src.db.connection import get_raw_connection
from src.notifications.center import emit
from src.utils.logging_config import setup_logging

setup_logging()
logger = logging.getLogger("imputer")

def check_missing_value_limits(dataset_id: str, profile_df: pd.DataFrame):
    """
    Checks if any column has missing_pct > 50%.
    If so, emits a hard_fail notification blocking the pipeline.
    """
    logger.info("Checking missing value limits...")
    for _, row in profile_df.iterrows():
        pct = row["missing_pct"]
        col = row["column_name"]
        
        if pct > 50.0:
            msg = f"Column '{col}' has {pct:.2f}% missing values, which exceeds the 50% limit. Column drop recommended."
            logger.error(msg)
            # Emit hard fail to block pipeline
            emit(
                source="layer0_imputer",
                condition="missing_pct_threshold_exceeded",
                severity="hard_fail",
                dataset_type="file_upload",
                details=msg
            )

def apply_imputation(df: pd.DataFrame, col_name: str, method: str) -> pd.Series:
    """
    Applies the specified imputation method to a single column in the DataFrame.
    """
    logger.info(f"Applying imputation to column '{col_name}' using method '{method}'...")
    col = df[col_name]
    
    if col.isna().sum() == 0:
        return col
        
    is_numeric = pd.api.types.is_numeric_dtype(col.dtype)
    
    if is_numeric:
        if method == "mean":
            return col.fillna(col.mean())
        elif method == "median":
            return col.fillna(col.median())
        elif method == "mode":
            mode_val = col.mode()
            return col.fillna(mode_val[0] if not mode_val.empty else 0.0)
        elif method == "constant":
            return col.fillna(0.0)
        elif method == "forward_fill":
            return col.ffill().bfill()  # ffill and fallback bfill for leading NaNs
        elif method == "knn":
            k_val = int(os.getenv("KNN_K_NEIGHBORS", 5))
            imputer = KNNImputer(n_neighbors=k_val)
            # We need other numeric features for KNN. Select all numeric columns.
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            # If the column has missing values, impute using the subset of numeric columns
            sub_df = df[numeric_cols].copy()
            imputed_array = imputer.fit_transform(sub_df)
            col_idx = numeric_cols.index(col_name)
            return pd.Series(imputed_array[:, col_idx], index=df.index, name=col_name)
        elif method == "regression":
            # Simple linear regression using the first non-null numeric column as feature
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            feature_cols = [c for c in numeric_cols if c != col_name and df[c].isna().sum() == 0]
            
            if not feature_cols:
                logger.warning("No complete numeric features available for regression imputation. Falling back to median.")
                return col.fillna(col.median())
                
            # Fit regression on non-null rows
            train_mask = col.notna()
            X_train = df.loc[train_mask, feature_cols]
            y_train = col[train_mask]
            
            X_test = df.loc[~train_mask, feature_cols]
            
            reg = LinearRegression()
            reg.fit(X_train, y_train)
            preds = reg.predict(X_test)
            
            imputed_col = col.copy()
            imputed_col[~train_mask] = preds
            return imputed_col
        else:
            raise ValueError(f"Unsupported numeric imputation method: {method}")
            
    else:
        # Categorical imputation
        if method == "mode":
            mode_val = col.mode()
            return col.fillna(mode_val[0] if not mode_val.empty else "unknown")
        elif method == "constant_unknown":
            return col.fillna("unknown")
        elif method == "missing_category":
            return col.fillna("missing")
        else:
            raise ValueError(f"Unsupported categorical imputation method: {method}")
            
def persist_imputation_choice(dataset_id: str, col_name: str, method: str):
    """
    Updates the column_profiles database record with the PM-selected imputation method.
    """
    logger.info(f"Persisting imputation choice: dataset={dataset_id}, col={col_name}, method={method}")
    with get_raw_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE column_profiles
                SET imputation_method = %s
                WHERE dataset_id = %s AND column_name = %s;
                """,
                (method, dataset_id, col_name)
            )
