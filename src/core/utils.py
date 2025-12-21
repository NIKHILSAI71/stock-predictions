
import math
import numpy as np
import pandas as pd
from typing import Any

def sanitize_for_json(obj: Any) -> Any:
    """
    Recursively sanitize data for JSON serialization.
    Replaces NaN, Inf, -Inf with None.
    Converts pandas Series/DataFrame to lists/dicts.
    """
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(item) for item in obj]
    elif isinstance(obj, pd.Series):
        # Skip Series - they're too large for JSON output
        # Return only the last value or None
        return None
    elif isinstance(obj, pd.DataFrame):
        # Skip DataFrame - too large for JSON output
        return None
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return round(obj, 4)
    elif isinstance(obj, (np.floating, np.integer)):
        val = float(obj)
        if math.isnan(val) or math.isinf(val):
            return None
        return round(val, 4)
    elif isinstance(obj, np.ndarray):
        return sanitize_for_json(obj.tolist())
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj
