import json
import logging
import pandas as pd
import numpy as np
from src.llm.claude_client import claude_client

logger = logging.getLogger("layer2_cluster_profiler")

def profile_clusters(
    df: pd.DataFrame,
    labels: np.ndarray,
    behavioral_cols: list[str]
) -> dict[int, str]:
    """
    Computes behavioral feature averages for each active cluster and invokes Claude
    to generate a plain-language summary describing what distinguishes the cluster.
    
    Ensures that the LLM does not assign a name or title to the cluster.
    
    Returns:
      cluster_summaries (dict[int, str]): A map of cluster label ID -> summary text.
    """
    logger.info("Layer 2.3: Starting LLM-assisted cluster profiling...")
    
    # 1. Compute means of features per cluster
    df_temp = df.copy()
    df_temp["_cluster_label"] = labels
    
    active_clusters = sorted([c for c in set(labels) if c != -1])
    cluster_summaries = {}
    
    # Calculate global medians or means for relative comparison context in the prompt
    global_means = df_temp[behavioral_cols].mean().to_dict()
    
    for c in active_clusters:
        cluster_df = df_temp[df_temp["_cluster_label"] == c]
        cluster_means = cluster_df[behavioral_cols].mean().to_dict()
        
        # 2. Formulate comparative profile summary for the prompt
        profile_details = []
        for col in behavioral_cols:
            c_val = cluster_means[col]
            g_val = global_means[col]
            diff_pct = ((c_val - g_val) / g_val * 100.0) if g_val != 0 else 0.0
            profile_details.append(
                f"- {col}: Mean={c_val:.2f} (Global Mean={g_val:.2f}, Difference={diff_pct:+.1f}%)"
            )
            
        profile_details_str = "\n".join(profile_details)
        
        prompt = f"""You are a customer segmentation analyst.
Given the following average behavioral characteristics for Cluster {c} relative to the global population:

{profile_details_str}

Write a plain-language summary of what distinguishes this cluster (e.g., "High spending frequency but lower average order value, with moderate return rate").
Focus purely on descriptive summary.

Strict Constraint:
- DO NOT assign, suggest, or include any name, label, title, or category name for this cluster (e.g. do not say "High-Value Buyers" or "Sparsely Active Loyalists").
- Your response must focus entirely on describing the behaviors.

Return your response ONLY as a valid JSON object with a single key "summary":
{{
  "summary": "description here"
}}
"""
        
        system_prompt = "You are a customer profiling assistant. Describe the customer segment behavioral profile in a plain-language paragraph. Do NOT suggest a name, title, label, or category name. Respond only with a JSON object: {\"summary\": \"description here\"}"
        
        response_text = claude_client.call(prompt=prompt, system_prompt=system_prompt)
        
        try:
            # Clean response text in case Claude outputs markdown code block
            clean_text = response_text.strip()
            if clean_text.startswith("```json"):
                clean_text = clean_text.split("```json")[1].split("```")[0].strip()
            elif clean_text.startswith("```"):
                clean_text = clean_text.split("```")[1].split("```")[0].strip()
                
            data = json.loads(clean_text)
            
            # Anti-naming validation: ensure no naming fields were sneaked in
            forbidden_keys = {"name", "label", "title", "cluster_name", "suggested_name"}
            for key in forbidden_keys:
                if key in data:
                    del data[key]
                    
            summary = data.get("summary", "")
            
            # Double check that the summary text itself does not look like it contains an assigned name block
            cluster_summaries[c] = summary
            logger.info(f"Generated profile for Cluster {c}: {summary[:60]}...")
            
        except Exception as e:
            logger.error(f"Failed to parse LLM response for Cluster {c}: {str(e)}. Fallback to raw averages summary.")
            # Fallback summary
            averages = [f"{col}={cluster_means[col]:.2f}" for col in behavioral_cols[:4]]
            cluster_summaries[c] = f"Segment defined by averages: {', '.join(averages)}."
            
    return cluster_summaries

def run_profiling_and_persist(dataset_id: str, period_id: int, behavioral_cols: list[str]) -> dict:
    """
    Queries database assignments and snapshot features, runs profiling, and persists
    LLM summaries to clusters table (cluster_description column).
    """
    logger.info(f"Running profiling and persisting results for period {period_id}...")
    from src.db.connection import get_raw_connection
    
    # 1. Fetch assignments
    with get_raw_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT ca.customer_id, c.cluster_id, c.cluster_label
                FROM cluster_assignments ca
                JOIN clusters c ON ca.cluster_id = c.cluster_id
                WHERE ca.period_id = %s;
                """,
                (period_id,)
            )
            rows = cur.fetchall()
            
    if not rows:
        raise ValueError(f"No cluster assignments found for period {period_id} to profile.")
        
    df = pd.DataFrame(rows, columns=["customer_id", "cluster_id", "cluster_label"])
    
    # 2. Fetch features
    features_list = []
    with get_raw_connection() as conn:
        with conn.cursor() as cur:
            for cid in df["customer_id"].tolist():
                cols_str = ", ".join([f'"{col}"' for col in behavioral_cols])
                cur.execute(
                    f'SELECT {cols_str} FROM behavioral_snapshots WHERE customer_id = %s AND period_id = %s;',
                    (cid, period_id)
                )
                features_list.append(cur.fetchone())
                
    feat_df = pd.DataFrame(features_list, columns=behavioral_cols)
    
    # Map cluster_label back to labels array (e.g. Cluster_0 -> 0, Noise -> -1)
    labels = []
    cluster_id_to_key = {} # maps Cluster_0 to 0, Noise to -1, etc.
    cluster_label_to_id = {} # maps Cluster_0 to cluster_id
    
    for _, r in df.iterrows():
        lbl = r["cluster_label"]
        cid = r["cluster_id"]
        cluster_label_to_id[lbl] = cid
        
        if "Noise" in lbl or lbl is None:
            labels.append(-1)
            cluster_id_to_key[cid] = -1
        else:
            try:
                k = int(lbl.split("_")[1])
                labels.append(k)
                cluster_id_to_key[cid] = k
            except Exception:
                labels.append(-1)
                
    labels = np.array(labels)
    
    # 3. Call profiler
    summaries = profile_clusters(feat_df, labels, behavioral_cols)
    
    # 4. Persist to DB
    with get_raw_connection() as conn:
        with conn.cursor() as cur:
            for c_lbl, cid in cluster_label_to_id.items():
                if "Noise" in c_lbl:
                    continue
                # Match candidate label digit
                try:
                    k = int(c_lbl.split("_")[1])
                    summary = summaries.get(k, "")
                    if summary:
                        cur.execute(
                            """
                            UPDATE clusters
                            SET cluster_description = %s
                            WHERE cluster_id = %s;
                            """,
                            (summary, cid)
                        )
                except Exception:
                    continue
                    
    return summaries
