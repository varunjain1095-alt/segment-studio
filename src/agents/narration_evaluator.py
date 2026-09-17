import re
import logging
from src.db.connection import get_raw_connection

logger = logging.getLogger("narration_evaluator")

def extract_all_numbers(data) -> set[float]:
    """Recursively extracts all numeric values from a nested data structure."""
    nums = set()
    if isinstance(data, dict):
        for v in data.values():
            nums.update(extract_all_numbers(v))
    elif isinstance(data, list):
        for v in data:
            nums.update(extract_all_numbers(v))
    elif isinstance(data, (int, float)):
        # Exclude common default period ids or constants like K=5 if necessary, 
        # but generally keeping them is fine for comparison.
        nums.add(float(data))
    elif isinstance(data, str):
        # Remove commas, dollar signs, percent signs and try parsing
        cleaned = data.replace(",", "").replace("$", "").replace("%", "").strip()
        try:
            nums.add(float(cleaned))
        except ValueError:
            pass
    return nums

def is_num_close(val: float, target_nums: set[float]) -> bool:
    """Checks if a value is close to any number in target_nums (handles direct float and percentage scaling)."""
    for t in target_nums:
        # Check standard equality/closeness
        if abs(val - t) < 0.05 or (t != 0 and abs(val - t) / abs(t) < 0.02):
            return True
        # Check percentage scaling (e.g. 12% in text vs 0.12 in data)
        if abs(val - (t * 100.0)) < 0.05 or (t != 0 and abs(val - (t * 100.0)) / abs(t * 100.0) < 0.02):
            return True
        if abs((val / 100.0) - t) < 0.05 or (t != 0 and abs((val / 100.0) - t) / abs(t) < 0.02):
            return True
    return False

def verify_narration(narration: str, collated_data: dict, period_id: int) -> tuple[bool, list[str]]:
    """
    Evaluates the narration text against strict guardrails:
    1. Causal language checks ('because', 'caused by', 'due to').
    2. Intensifier checks ('significantly', 'dramatically').
    3. Numeric validation (checks that all numbers/percentages cited in the text match database metrics).
    
    Returns:
      (passed: bool, errors: list[str])
    """
    errors = []
    
    # 1. Causal Language check (Strict: Layer 3 statistical significance does not establish causation)
    causal_patterns = [r"\bbecause\b", r"\bcaused\s+by\b", r"\bdue\s+to\b"]
    matched_causal = [pat for pat in causal_patterns if re.search(pat, narration, re.IGNORECASE)]
    if matched_causal:
        errors.append(
            "Narration contains causal language ('because', 'caused by', 'due to'). Layer 3 statistical "
            "significance establishes observational difference/association, NEVER causation. Causal claims "
            "must be rephrased as association (e.g., 'coincided with', 'associated with', 'accompanied by')."
        )
            
    # 2. Intensifier check
    intensifier_patterns = [r"\bsignificantly\b", r"\bdramatically\b"]
    has_intensifiers = any(re.search(pat, narration, re.IGNORECASE) for pat in intensifier_patterns)
    
    if has_intensifiers:
        # Intensifiers are disallowed unless there is an active significant test, or some major metric change
        # Let's check if there are significant test results or large MoM changes
        with get_raw_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT COUNT(*) FROM statistical_test_results WHERE period_id = %s AND significant = TRUE;",
                    (period_id,)
                )
                sig_count = cur.fetchone()[0]
                
        # Also check demographic metrics anomaly flagging
        with get_raw_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT COUNT(*) FROM demographic_segment_metrics WHERE period_id = %s AND anomaly_flagged = TRUE;",
                    (period_id,)
                )
                anomaly_count = cur.fetchone()[0]
                
        if sig_count == 0 and anomaly_count == 0:
            errors.append(
                "Narration contains intensifier words ('significantly', 'dramatically') but no significant "
                "statistical tests or demographic anomalies exist for this period."
            )
            
    # 3. Numeric validation
    target_nums = extract_all_numbers(collated_data)
    # Extract all numbers from narration (e.g. 12.5, 90, 1000)
    # Regex handles numbers like 10.0, 12, 1,200, 15%
    narr_nums = re.findall(r"\b\d+(?:,\d+)*(?:\.\d+)?%?\b", narration)
    
    for num_str in narr_nums:
        # Clean comma and percent
        cleaned = num_str.replace(",", "").replace("%", "").strip()
        try:
            val = float(cleaned)
        except ValueError:
            continue
            
        # Ignore common calendar years or small index integers
        if val in [2024, 2025, 2026, 0, 1, 2, 3, 4, 5]:
            continue
            
        if not is_num_close(val, target_nums):
            errors.append(
                f"Numeric citation '{num_str}' (parsed as {val}) in the narration cannot be matched to "
                f"any raw outcomes or metrics in the database for this period."
            )
            
    passed = len(errors) == 0
    return passed, errors
