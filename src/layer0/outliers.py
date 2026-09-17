import logging
import numpy as np
import pandas as pd
import os
from scipy.stats import chi2, skew
from sklearn.preprocessing import RobustScaler, PowerTransformer
from src.notifications.center import emit
from src.utils.logging_config import setup_logging

setup_logging()
logger = logging.getLogger("outliers")

# Configurable thresholds (explicitly PM-approved values)
MAHALANOBIS_DISTANCE_THRESHOLD = float(os.getenv("MAHALANOBIS_DISTANCE_THRESHOLD", 3.0))
MEDIAN_SIMILARITY_THRESHOLD = float(os.getenv("MEDIAN_SIMILARITY_THRESHOLD", 0.05))
SKEWNESS_TRANSFORM_THRESHOLD = float(os.getenv("SKEWNESS_TRANSFORM_THRESHOLD", 1.0))
RARE_CATEGORY_THRESHOLD = float(os.getenv("RARE_CATEGORY_THRESHOLD", 0.015))

def treat_univariate_numeric_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies IQR outlier detection on numeric columns and treats them at column-level:
    - If skewness > SKEWNESS_TRANSFORM_THRESHOLD:
      - If all values > 0: log1p + RobustScaler.
      - If any value <= 0: Yeo-Johnson + RobustScaler.
    - Else: RobustScaler only.
    """
    logger.info("Treating univariate numeric outliers...")
    df_treated = df.copy()
    numeric_cols = df_treated.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if df_treated[c].nunique(dropna=True) > 2]
    
    for col_name in numeric_cols:
        col = df_treated[col_name]
        
        # Calculate IQR boundaries
        q1 = col.quantile(0.25)
        q3 = col.quantile(0.75)
        iqr = q3 - q1
        
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        # Check if outliers exist
        outliers_mask = (col < lower_bound) | (col > upper_bound)
        outlier_count = outliers_mask.sum()
        
        if outlier_count > 0:
            # Drop NaNs for fit
            col_clean = col.dropna()
            if len(col_clean) == 0:
                continue
                
            # Check skewness
            col_skew = abs(skew(col_clean))
            
            if col_skew > SKEWNESS_TRANSFORM_THRESHOLD:
                logger.info(f"Column '{col_name}' has {outlier_count} IQR outliers and skewness {col_skew:.2f} > {SKEWNESS_TRANSFORM_THRESHOLD}. Applying power transform...")
                # Check if all values > 0
                if (col_clean > 0).all():
                    # Apply log1p transform
                    logger.info(f"Applying log1p transform to strictly positive column '{col_name}'.")
                    transformed = np.log1p(col)
                else:
                    # Apply Yeo-Johnson transform
                    logger.info(f"Applying Yeo-Johnson transform to zero/negative column '{col_name}'.")
                    pt = PowerTransformer(method='yeo-johnson', standardize=False)
                    # Reshape for fit
                    non_null_indices = col.notna()
                    vals = col[non_null_indices].values.reshape(-1, 1)
                    transformed_vals = pt.fit_transform(vals).flatten()
                    
                    transformed = pd.Series(index=col.index, dtype=float)
                    transformed[non_null_indices] = transformed_vals
            else:
                logger.info(f"Column '{col_name}' has {outlier_count} IQR outliers but skewness {col_skew:.2f} <= {SKEWNESS_TRANSFORM_THRESHOLD}. Applying RobustScaler only without power transform.")
                transformed = col.copy()
                
            # Apply RobustScaler
            scaler = RobustScaler()
            non_null_indices = transformed.notna()
            scaled_vals = scaler.fit_transform(transformed[non_null_indices].values.reshape(-1, 1)).flatten()
            
            df_treated.loc[non_null_indices, col_name] = scaled_vals
            
    return df_treated

def treat_univariate_categorical_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flags categorical values appearing in <= 1.5% of rows.
    Auto-excludes these rows and emits soft warning.
    """
    logger.info("Treating univariate categorical rare categories...")
    df_filtered = df.copy()
    
    # Categorical columns are objects/categories
    cat_cols = df_filtered.select_dtypes(include=['object', 'category']).columns.tolist()
    cat_cols = [c for c in cat_cols if c not in ["customer_id", "signup_date"]]
    total_rows = len(df_filtered)
    
    if total_rows == 0:
        return df_filtered
        
    rows_to_drop = pd.Series(False, index=df_filtered.index)
    
    for col_name in cat_cols:
        col = df_filtered[col_name]
        nunique = col.nunique(dropna=True)
        
        # Skip high-cardinality columns (like dates, codes, IDs, description texts)
        if nunique > 20:
            logger.info(f"Skipping categorical outlier treatment for high-cardinality column '{col_name}' ({nunique} unique values).")
            continue
            
        value_counts = col.value_counts(dropna=True)
        
        # Identify all categories at or below the 1.5% threshold
        shares = value_counts / total_rows
        rare_mask = shares <= RARE_CATEGORY_THRESHOLD
        rare_categories = value_counts.index[rare_mask].tolist()
        
        if rare_categories:
            if len(rare_categories) <= 10:
                # Individual notifications for low-cardinality columns (maintains test expectations)
                for val in rare_categories:
                    count = int(value_counts[val])
                    share = float(shares[val])
                    msg = f"Value '{val}' in column '{col_name}' represents {share:.2%} of total ({count} rows), which is at or below the 1.5% threshold."
                    logger.info(f"Categorical Rare Outlier: {msg}")
                    emit(
                        source="layer0_categorical_outliers",
                        condition="rare_category_excluded",
                        severity="soft_warning",
                        dataset_type="file_upload",
                        details=msg
                    )
            else:
                # Aggregated notification for high-cardinality columns (prevents massive DB overhead)
                num_rare = len(rare_categories)
                total_rare_rows = int(value_counts[rare_categories].sum())
                msg = f"Column '{col_name}' has {num_rare} rare categories (<= 1.5% share) representing {total_rare_rows} total rows ({total_rare_rows / total_rows:.2%})."
                logger.info(f"Categorical Rare Outliers (Aggregated): {msg}")
                emit(
                    source="layer0_categorical_outliers",
                    condition="rare_category_excluded",
                    severity="soft_warning",
                    dataset_type="file_upload",
                    details=msg
                )
            
            # Flag rows in bulk
            rows_to_drop = rows_to_drop | col.isin(rare_categories)
            
    dropped_count = rows_to_drop.sum()
    if dropped_count > 0:
        logger.info(f"Excluding {dropped_count} rows with rare categories.")
        df_filtered = df_filtered[~rows_to_drop].reset_index(drop=True)
        
    return df_filtered

def treat_bivariate_numeric_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates Mahalanobis distance for each numeric column pair.
    Flags rows exceeding critical threshold (distance > 3.0) and excludes them.
    """
    logger.info("Treating bivariate numeric outliers using Mahalanobis distance...")
    df_filtered = df.copy()
    numeric_cols = df_filtered.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if df_filtered[c].nunique(dropna=True) > 2]
    
    if len(numeric_cols) < 2 or len(df_filtered) < 5:
        return df_filtered
        
    rows_to_drop = pd.Series(False, index=df_filtered.index)
    
    # Check all distinct pairs of numeric columns
    for i in range(len(numeric_cols)):
        for j in range(i + 1, len(numeric_cols)):
            col1 = numeric_cols[i]
            col2 = numeric_cols[j]
            
            # Select values and drop rows with NaNs in either column
            sub_df = df_filtered[[col1, col2]].dropna()
            if len(sub_df) < 5:
                continue
                
            X = sub_df.values
            try:
                # Compute Covariance and its inverse
                cov = np.cov(X.T)
                inv_cov = np.linalg.inv(cov)
                mean = X.mean(axis=0)
                
                # Compute Mahalanobis distance for each row vector
                diff = X - mean
                dists = np.sqrt(np.sum(np.dot(diff, inv_cov) * diff, axis=1))
                
                # Flag rows exceeding threshold
                outlier_local_mask = dists > MAHALANOBIS_DISTANCE_THRESHOLD
                outlier_indices = sub_df.index[outlier_local_mask]
                
                if len(outlier_indices) > 0:
                    for idx in outlier_indices:
                        rows_to_drop.at[idx] = True
                        
            except np.linalg.LinAlgError:
                # Singular matrix (columns perfectly correlated)
                continue
                
    dropped_count = rows_to_drop.sum()
    if dropped_count > 0:
        logger.info(f"Mahalanobis outlier check: excluding {dropped_count} rows.")
        emit(
            source="layer0_bivariate_outliers",
            condition="bivariate_outliers_excluded",
            severity="soft_warning",
            dataset_type="file_upload",
            details=f"Excluded {dropped_count} rows exceeding bivariate Mahalanobis distance threshold (> {MAHALANOBIS_DISTANCE_THRESHOLD})."
        )
        df_filtered = df_filtered[~rows_to_drop].reset_index(drop=True)
        
    return df_filtered

def check_bivariate_categorical_similarity(df: pd.DataFrame, default_numeric_col: str = None):
    """
    Computes median of a numeric target variable by category.
    Flags category pairs with close medians (< 5% difference) as merge candidates.
    Does NOT auto-action. Emits soft warning.
    """
    logger.info("Checking bivariate categorical median similarity...")
    cat_cols = df.select_dtypes(include=[object, 'category']).columns.tolist()
    cat_cols = [c for c in cat_cols if c not in ["customer_id", "signup_date"]]
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if not cat_cols or not numeric_cols:
        return
        
    # Use specified numeric column or fall back to first numeric column
    num_col = default_numeric_col if default_numeric_col in numeric_cols else numeric_cols[0]
    
    for col_name in cat_cols:
        # Group by and calculate medians
        medians = df.groupby(col_name)[num_col].median()
        
        cat_list = medians.index.tolist()
        
        # Skip high-cardinality columns (prevents expensive O(N^2) pair-wise search)
        if len(cat_list) > 20:
            logger.info(f"Skipping categorical similarity check for high-cardinality column '{col_name}' ({len(cat_list)} unique categories).")
            continue
            
        for idx1 in range(len(cat_list)):
            for idx2 in range(idx1 + 1, len(cat_list)):
                c1 = cat_list[idx1]
                c2 = cat_list[idx2]
                m1 = medians[c1]
                m2 = medians[c2]
                
                # Check percentage difference relative to average
                avg = (m1 + m2) / 2
                if avg > 0:
                    diff_pct = abs(m1 - m2) / avg
                    if diff_pct <= MEDIAN_SIMILARITY_THRESHOLD:  # Within similarity threshold
                        msg = f"Categories '{c1}' and '{c2}' in column '{col_name}' have very close medians for target '{num_col}' ({m1:.2f} vs {m2:.2f}, diff={diff_pct:.1%}). PM merge recommended."
                        logger.info(f"Bivariate Categorical Similarity Candidate: {msg}")
                        emit(
                            source="layer0_categorical_similarity",
                            condition="median_similarity_merge_candidate",
                            severity="soft_warning",
                            dataset_type="file_upload",
                            details=msg
                        )
