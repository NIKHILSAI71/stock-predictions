
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
    # Check for None first
    if obj is None:
        return None

    # Check for bool types EARLY (before numeric types)
    # This catches numpy bool, pandas bool, and Python bool
    if isinstance(obj, (np.bool_, bool)) or type(obj).__name__ in ['bool_', 'bool']:
        return bool(obj)

    # Check for dict and list (recursive structures)
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(item) for item in obj]
    elif isinstance(obj, tuple):
        return [sanitize_for_json(item) for item in obj]

    # Check for pandas types
    elif isinstance(obj, pd.Series):
        # Skip Series - they're too large for JSON output
        return None
    elif isinstance(obj, pd.DataFrame):
        # Skip DataFrame - too large for JSON output
        return None

    # Check for numpy array
    elif isinstance(obj, np.ndarray):
        return sanitize_for_json(obj.tolist())

    # Check for numeric types (int, float)
    elif isinstance(obj, (np.floating, np.integer)):
        val = float(obj)
        if math.isnan(val) or math.isinf(val):
            return None
        return round(val, 4)
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return round(obj, 4)
    elif isinstance(obj, (int, np.int_)):
        return int(obj)

    # Check for string types
    elif isinstance(obj, (str, np.str_)):
        return str(obj)

    return obj
