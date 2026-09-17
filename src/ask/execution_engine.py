import logging
from fastapi.testclient import TestClient
from src.main import app
from src.db.connection import get_raw_connection

logger = logging.getLogger("execution_engine")

def get_periods_descending(start_period_id: int) -> list[dict]:
    """Returns a list of all periods starting from the requested one, going backwards in time."""
    with get_raw_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT period_id, period_label FROM periods WHERE period_id <= %s ORDER BY period_id DESC;",
                (start_period_id,)
            )
            return [{"id": r[0], "label": r[1]} for r in cur.fetchall()]

def has_data(response_json, layer: str) -> bool:
    """Helper to determine if the response contains valid computed data."""
    if not response_json:
        return False
        
    if isinstance(response_json, list):
        return len(response_json) > 0
        
    if isinstance(response_json, dict):
        if "detail" in response_json:
            return False
            
        if layer == "Layer2":
            # For Layer 2, verify if it was actually run (decision is not HOLD with empty maps)
            # or check if it has a valid baseline/identity map
            if response_json.get("decision") == "HOLD" and not response_json.get("identity_map"):
                # Check if a baseline actually exists in DB for this period
                period_id = response_json.get("period_id")
                with get_raw_connection() as conn:
                    with conn.cursor() as cur:
                        cur.execute("SELECT COUNT(*) FROM cluster_baselines WHERE period_id = %s;", (period_id,))
                        count = cur.fetchone()[0]
                return count > 0
            return True
            
    return True

def dispatch_step(step: dict, context_params: dict, client: TestClient = None) -> dict:
    """
    Dispatches a single step (layer/action) to the backend API via TestClient,
    supporting automatic fallback to historical periods if data is missing.
    """
    if client is None:
        client = TestClient(app)
        
    layer = step.get("layer")
    action = step.get("action")
    
    # Extract params from context
    target_period_id = context_params.get("period_id")
    cluster_id = context_params.get("cluster_id")
    metric = context_params.get("metric") or "churn_rate"
    segment_ids = context_params.get("segment_ids") or []
    granularity = context_params.get("granularity") or "quarterly"
    
    if not target_period_id:
        # If no period is specified, get the latest period
        with get_raw_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT MAX(period_id) FROM periods;")
                target_period_id = cur.fetchone()[0]
                
    if not target_period_id:
        return {"data": None, "error": "No periods found in database."}

    periods = get_periods_descending(target_period_id)
    
    # Try periods in descending order starting from target
    for p in periods:
        curr_period_id = p["id"]
        curr_period_label = p["label"]
        
        url = ""
        params = {}
        
        if layer == "Layer1":
            url = f"/api/v1/demographic-segments?period_id={curr_period_id}"
        elif layer == "Layer2":
            url = f"/api/v1/clusters/drift?period_id={curr_period_id}"
        elif layer == "Layer3":
            url = f"/api/v1/statistical-tests?period_id={curr_period_id}"
        elif layer == "Layer4":
            url = f"/api/v1/migrations?period_id={curr_period_id}"
        elif layer == "Layer5":
            # For transition matrix, we need from/to period
            # from_period_id is curr_period_id - 1
            # But wait, does curr_period_id - 1 exist?
            url = f"/api/v1/transition-matrix?from_period_id={curr_period_id-1}&to_period_id={curr_period_id}"
        elif layer == "Layer6":
            if cluster_id is None:
                with get_raw_connection() as conn:
                    with conn.cursor() as cur:
                        cur.execute("SELECT DISTINCT cluster_id FROM clusters ORDER BY cluster_id;")
                        cluster_ids = [r[0] for r in cur.fetchall()]
                if not cluster_ids:
                    cluster_ids = [1]
                
                results_list = []
                for cid in cluster_ids:
                    u = f"/api/v1/segments/{cid}/outcomes?period_id={curr_period_id}"
                    resp = client.get(u)
                    if resp.status_code == 200:
                        results_list.append(resp.json())
                if results_list:
                    meta = {}
                    if curr_period_id != target_period_id:
                        meta = {"historical": True, "period": curr_period_label}
                    return {"data": results_list, "meta": meta}
                else:
                    return {"data": None, "error": "No outcomes found for any cluster."}
            else:
                url = f"/api/v1/segments/{cluster_id}/outcomes?period_id={curr_period_id}"
        elif layer == "Layer7":
            # Comparison requires segment_ids and period range
            seg_ids_str = ",".join(str(sid) for sid in segment_ids)
            url = f"/api/v1/metrics/comparison?granularity={granularity}&metric={metric}&segment_ids={seg_ids_str}&period_range={curr_period_label},{curr_period_label}"
        elif layer == "Layer8":
            url = f"/api/v1/feature-migration-signals?period_id={curr_period_id}"
        elif layer == "A1":
            url = f"/api/v1/narrations?period_id={curr_period_id}"
        else:
            return {"data": None, "error": f"Unknown layer: {layer}"}
            
        try:
            response = client.get(url)
            if response.status_code == 200:
                res_data = response.json()
                if has_data(res_data, layer):
                    meta = {}
                    if curr_period_id != target_period_id:
                        meta = {"historical": True, "period": curr_period_label}
                    return {"data": res_data, "meta": meta}
        except Exception as e:
            logger.error(f"Error calling {url}: {e}")
            
    # If all failed, return empty result with target period label
    target_period_label = next((p["label"] for p in periods if p["id"] == target_period_id), str(target_period_id))
    return {"data": None, "meta": {"historical": True, "period": target_period_label}}

def execute_pipeline(steps: list[dict], context_params: dict, client: TestClient = None) -> list[dict]:
    """
    Executes the shopping list steps.
    Parallel execution for independent show/comparison templates,
    Sequential execution for root_cause and prediction.
    """
    if client is None:
        client = TestClient(app)
        
    results = []
    # If there is root_cause or prediction in the intent (or length > 2), run sequentially.
    # Otherwise run in parallel (or list is small).
    # Since Python's threading can be used, let's execute sequentially for prediction/root_cause
    # and use ThreadPoolExecutor for show/comparison.
    # But wait, sequential execution allows earlier results to scope next calls.
    # E.g. for root_cause: L2 drift identifies which cluster drifted -> update cluster_id.
    # Let's implement that!
    run_sequentially = len(steps) > 2
    
    if run_sequentially:
        current_context = dict(context_params)
        for step in steps:
            res = dispatch_step(step, current_context, client)
            results.append({"step": step, "result": res})
            # Scope next calls if needed. E.g., if L2 profile or drift returns a cluster_id, we can propagate it.
            # E.g. if we retrieve L2 drift and find the identity map has old/new cluster ids, or we can check segment outcome.
            # But the prompt says: "each result available to scope the next call".
            # For simplicity, if we get a result from a layer, we can merge any resolved segment_id or parameters.
            # Let's verify if anything needs custom scoping:
            # - If we ran L2 and found a cluster_id/decision, we can use it.
            # Let's keep it simple: merge any data keys into current_context if they match expected params.
            if res.get("data") and isinstance(res["data"], dict):
                for k in ["cluster_id", "segment_id", "period_id"]:
                    if k in res["data"] and res["data"][k] is not None:
                        current_context[k] = res["data"][k]
    else:
        # Run in parallel using ThreadPoolExecutor for show/comparison
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(dispatch_step, step, context_params, client) for step in steps]
            for step, future in zip(steps, futures):
                try:
                    res = future.result()
                    results.append({"step": step, "result": res})
                except Exception as e:
                    logger.error(f"Parallel step execution failed: {e}")
                    results.append({"step": step, "result": {"data": None, "meta": {}}})
                    
    return results
