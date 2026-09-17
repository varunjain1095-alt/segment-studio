import os
import logging
import json
import pandas as pd
from mlxtend.frequent_patterns import fpgrowth
from src.db.connection import get_raw_connection
from src.ingestion.router import DATA_DIR

logger = logging.getLogger("fallback_pattern_discovery")

def discover_patterns(dataset_id: str, columns: list[str], top_n: int) -> list[dict]:
    """
    Finds the most common behavioral value combinations across customers at the
    specified columns using FP-Growth frequent pattern mining with a fixed-width constraint.
    """
    logger.info(f"Running pattern discovery for dataset {dataset_id} on columns {columns} (top_n={top_n})")
    
    # 1. Load the processed dataset to get correct customers/rows
    processed_path = os.path.join(DATA_DIR, f"{dataset_id}_processed.csv")
    if not os.path.exists(processed_path):
        raise ValueError(f"Processed dataset not found for dataset {dataset_id}")
    processed_df = pd.read_csv(processed_path)
    
    # 2. Load confirmed dataset containing original categorical and numeric values
    confirmed_path = os.path.join(DATA_DIR, f"{dataset_id}_confirmed.csv")
    if not os.path.exists(confirmed_path):
        raise ValueError(f"Confirmed dataset not found for dataset {dataset_id}")
    confirmed_df = pd.read_csv(confirmed_path)
    
    # Align confirmed_df with processed_df on customer_id
    if "customer_id" in processed_df.columns and "customer_id" in confirmed_df.columns:
        confirmed_df = confirmed_df.set_index("customer_id").loc[processed_df["customer_id"]].reset_index()
    
    # 3. Retrieve column types from database profiles
    with get_raw_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT column_name, dtype FROM column_profiles WHERE dataset_id = %s;",
                (dataset_id,)
            )
            profiles = {row[0]: row[1] for row in cur.fetchall()}
            
    # 4. Perform bucketing
    bucketed_df = pd.DataFrame(index=confirmed_df.index)
    total_customers = len(confirmed_df)
    
    for col in columns:
        if col not in confirmed_df.columns:
            logger.warning(f"Column '{col}' not found in confirmed dataset. Skipping.")
            continue
            
        col_clean = col.replace('_', ' ')
        dtype = profiles.get(col, '')
        is_categorical = dtype in ['object', 'category', 'string'] or confirmed_df[col].dtype == object
        
        if is_categorical:
            # Categorical: treat each unique value as a separate item
            bucketed_df[col] = confirmed_df[col].apply(lambda val: f"{col_clean}: {val}" if pd.notna(val) else None)
        else:
            # Numeric: quartile bucketing
            q25 = confirmed_df[col].quantile(0.25)
            q75 = confirmed_df[col].quantile(0.75)
            
            def get_bucket(val):
                if pd.isna(val):
                    return None
                if val <= q25:
                    return f"Low {col_clean}"
                elif val <= q75:
                    return f"Medium {col_clean}"
                else:
                    return f"High {col_clean}"
            bucketed_df[col] = confirmed_df[col].apply(get_bucket)
            
    # 5. One-hot encode into transaction matrix
    df_trans = pd.get_dummies(bucketed_df, prefix="", prefix_sep="", dtype=bool)
    
    # 6. Run FP-Growth with dynamic support adaptation
    frequent_itemsets = pd.DataFrame()
    for support in [0.01, 0.005, 0.001, 0.0001, 0.0]:
        try:
            frequent_itemsets = fpgrowth(df_trans, min_support=support, use_colnames=True)
        except Exception as e:
            logger.warning(f"FP-Growth failed at support {support}: {e}. Trying Apriori.")
            from mlxtend.frequent_patterns import apriori
            try:
                frequent_itemsets = apriori(df_trans, min_support=support, use_colnames=True)
            except Exception as ap_err:
                logger.error(f"Apriori failed too: {ap_err}")
                break
                
        if not frequent_itemsets.empty:
            frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(len)
            filtered = frequent_itemsets[frequent_itemsets['length'] == len(columns)]
            if not filtered.empty:
                frequent_itemsets = filtered
                break
                
    if frequent_itemsets.empty or 'itemsets' not in frequent_itemsets.columns:
        logger.warning("No frequent itemsets found matching the criteria.")
        return []
        
    # 7. Sort itemsets in columns order and build labels
    results = []
    
    # Sort by support descending
    frequent_itemsets = frequent_itemsets.sort_values(by="support", ascending=False)
    
    for _, row in frequent_itemsets.iterrows():
        itemset = row["itemsets"]
        support = row["support"]
        
        # Format the combination label keeping column order
        sorted_items = []
        for col in columns:
            col_clean = col.replace('_', ' ')
            for item in itemset:
                if item == f"Low {col_clean}" or item == f"Medium {col_clean}" or item == f"High {col_clean}" or item.startswith(f"{col_clean}: "):
                    sorted_items.append(item)
                    break
        
        combination_label = " + ".join(sorted_items)
        customer_count = int(round(support * total_customers))
        
        results.append({
            "combination": combination_label,
            "count": customer_count,
            "percentage": float(support)
        })
        
    return results[:top_n]
