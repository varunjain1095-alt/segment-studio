def aggregate_results(execution_results: list[dict]) -> dict:
    """
    Aggregates the individual layer outputs from the execution engine.
    Returns:
        dict: {
            "supporting_data": {layer_name: exact_data, ...},
            "historical_flags": {layer_name: meta, ...}
        }
    """
    supporting_data = {}
    historical_flags = {}
    
    for item in execution_results:
        step = item.get("step", {})
        res = item.get("result", {})
        
        layer = step.get("layer")
        if not layer:
            continue
            
        layer_key = layer.lower()
        supporting_data[layer_key] = res.get("data")
        
        meta = res.get("meta")
        if meta and meta.get("historical"):
            historical_flags[layer_key] = meta
            
    return {
        "supporting_data": supporting_data,
        "historical_flags": historical_flags
    }
