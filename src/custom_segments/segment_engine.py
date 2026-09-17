import os
import logging
import pandas as pd
from src.ingestion.router import DATA_DIR

logger = logging.getLogger("custom_segment_engine")

def build_segment(dataset_id: str, conditions: list[dict]) -> list[str]:
    """
    Takes a list of column conditions and returns the matching customer IDs from the dataset.
    
    Conditions format:
    [
      {
        "column": "column_name",
        "type": "numeric" | "categorical",
        "range": [low, high] | "Bottom 25%" | "Middle 50%" | "Top 25%" | null,
        "values": [val1, val2, ...] | null
      }
    ]
    """
    processed_path = os.path.join(DATA_DIR, f"{dataset_id}_processed.csv")
    if not os.path.exists(processed_path):
        # Try raw fallback if processed is not found (useful for some test setups)
        processed_path = os.path.join(DATA_DIR, f"{dataset_id}.csv")
        if not os.path.exists(processed_path):
            raise FileNotFoundError(f"Dataset file not found for dataset_id: {dataset_id}")
            
    df = pd.read_csv(processed_path)
    
    if "customer_id" not in df.columns:
        raise ValueError("Dataset is missing 'customer_id' column.")
        
    filtered_df = df.copy()
    
    for cond in conditions:
        col = cond.get("column")
        if col not in filtered_df.columns:
            logger.warning(f"Condition column '{col}' not found in dataset. Skipping.")
            continue
            
        c_type = cond.get("type")
        
        if c_type == "numeric":
            rng = cond.get("range")
            if not rng:
                continue
                
            # Handle quartile strings
            if isinstance(rng, str):
                q25 = filtered_df[col].quantile(0.25)
                q75 = filtered_df[col].quantile(0.75)
                if rng == "Bottom 25%":
                    filtered_df = filtered_df[filtered_df[col] <= q25]
                elif rng == "Middle 50%":
                    filtered_df = filtered_df[(filtered_df[col] >= q25) & (filtered_df[col] <= q75)]
                elif rng == "Top 25%":
                    filtered_df = filtered_df[filtered_df[col] >= q75]
            elif isinstance(rng, list) or isinstance(rng, tuple):
                # Check if first element is a quartile string
                if len(rng) > 0 and rng[0] in ["Bottom 25%", "Middle 50%", "Top 25%"]:
                    rng_str = rng[0]
                    q25 = filtered_df[col].quantile(0.25)
                    q75 = filtered_df[col].quantile(0.75)
                    if rng_str == "Bottom 25%":
                        filtered_df = filtered_df[filtered_df[col] <= q25]
                    elif rng_str == "Middle 50%":
                        filtered_df = filtered_df[(filtered_df[col] >= q25) & (filtered_df[col] <= q75)]
                    elif rng_str == "Top 25%":
                        filtered_df = filtered_df[filtered_df[col] >= q75]
                else:
                    low = rng[0] if len(rng) > 0 else None
                    high = rng[1] if len(rng) > 1 else None
                    if low is not None:
                        filtered_df = filtered_df[filtered_df[col] >= float(low)]
                    if high is not None:
                        filtered_df = filtered_df[filtered_df[col] <= float(high)]
                        
        elif c_type == "categorical":
            vals = cond.get("values")
            if vals is None:
                continue
            # Keep matching checklist items
            str_vals = [str(v).strip().lower() for v in vals]
            filtered_df = filtered_df[filtered_df[col].astype(str).str.strip().str.lower().isin(str_vals)]
            
    return filtered_df["customer_id"].astype(str).tolist()
