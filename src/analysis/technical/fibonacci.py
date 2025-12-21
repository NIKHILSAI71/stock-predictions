"""
Fibonacci Retracement and Extension Levels
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple


# Standard Fibonacci retracement levels
FIBONACCI_LEVELS = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]

# Fibonacci extension levels
FIBONACCI_EXTENSIONS = [1.0, 1.272, 1.414, 1.618, 2.0, 2.618]


def fibonacci_retracement(
    data: pd.DataFrame,
    swing_high: Optional[float] = None,
    swing_low: Optional[float] = None,
    levels: List[float] = None
) -> Dict[str, float]:
    """
    Calculate Fibonacci Retracement levels.
    
    Formula:
        For an uptrend (retracement down):
            Level = Swing High - (Range × Ratio)
        For a downtrend (retracement up):
            Level = Swing Low + (Range × Ratio)
    
    Fibonacci ratios: 23.6%, 38.2%, 50%, 61.8%, 78.6%
    
    Args:
        data: DataFrame with High, Low, Close columns
        swing_high: The swing high price (optional, auto-detected if None)
        swing_low: The swing low price (optional, auto-detected if None)
        levels: Custom Fibonacci levels to calculate
    
    Returns:
        Dictionary with Fibonacci level names and their price values
    """
    if levels is None:
        levels = FIBONACCI_LEVELS
    
    # Auto-detect swing points if not provided
    if swing_high is None:
        swing_high = data['High'].max()
    if swing_low is None:
        swing_low = data['Low'].min()
    
    # Calculate price range
    price_range = swing_high - swing_low
    
    # Determine trend direction based on where high/low occurred
    high_idx = data['High'].idxmax()
    low_idx = data['Low'].idxmin()
    
    retracement_levels = {}
    
    if high_idx > low_idx:
        # Uptrend - calculate retracement levels going down from high
        for level in levels:
            level_name = f"{level * 100:.1f}%"
            level_price = swing_high - (price_range * level)
            retracement_levels[level_name] = round(level_price, 2)
        retracement_levels['swing_high'] = swing_high
        retracement_levels['swing_low'] = swing_low
        retracement_levels['trend'] = 'uptrend'
    else:
        # Downtrend - calculate retracement levels going up from low
        for level in levels:
            level_name = f"{level * 100:.1f}%"
            level_price = swing_low + (price_range * level)
            retracement_levels[level_name] = round(level_price, 2)
        retracement_levels['swing_high'] = swing_high
        retracement_levels['swing_low'] = swing_low
        retracement_levels['trend'] = 'downtrend'
    
    return retracement_levels


def fibonacci_extensions(
    swing_high: float,
    swing_low: float,
    retracement_point: float,
    levels: List[float] = None
) -> Dict[str, float]:
    """
    Calculate Fibonacci Extension levels.
    
    Used to project potential price targets after a retracement.
    
    Args:
        swing_high: The swing high price
        swing_low: The swing low price  
        retracement_point: The end of the retracement (wave B)
        levels: Custom extension levels to calculate
    
    Returns:
        Dictionary with extension level names and their price values
    """
    if levels is None:
        levels = FIBONACCI_EXTENSIONS
    
    price_range = swing_high - swing_low
    
    extension_levels = {}
    
    # For an uptrend extension
    if retracement_point < swing_high:
        for level in levels:
            level_name = f"{level * 100:.1f}%"
            level_price = retracement_point + (price_range * level)
            extension_levels[level_name] = round(level_price, 2)
    else:
        # For a downtrend extension
        for level in levels:
            level_name = f"{level * 100:.1f}%"
            level_price = retracement_point - (price_range * level)
            extension_levels[level_name] = round(level_price, 2)
    
    return extension_levels


def detect_swing_points(
    data: pd.DataFrame,
    lookback: int = 5
) -> Tuple[List[Dict], List[Dict]]:
    """
    Detect swing highs and swing lows in price data.
    
    A swing high is a high surrounded by lower highs.
    A swing low is a low surrounded by higher lows.
    
    Args:
        data: DataFrame with High, Low columns
        lookback: Number of bars to look on each side
    
    Returns:
        Tuple of (swing_highs, swing_lows) lists
    """
    highs = data['High'].values
    lows = data['Low'].values
    index = data.index
    
    swing_highs = []
    swing_lows = []
    
    for i in range(lookback, len(data) - lookback):
        # Check for swing high
        is_swing_high = True
        for j in range(1, lookback + 1):
            if highs[i] <= highs[i - j] or highs[i] <= highs[i + j]:
                is_swing_high = False
                break
        
        if is_swing_high:
            swing_highs.append({
                'index': index[i],
                'price': highs[i],
                'position': i
            })
        
        # Check for swing low
        is_swing_low = True
        for j in range(1, lookback + 1):
            if lows[i] >= lows[i - j] or lows[i] >= lows[i + j]:
                is_swing_low = False
                break
        
        if is_swing_low:
            swing_lows.append({
                'index': index[i],
                'price': lows[i],
                'position': i
            })
    
    return swing_highs, swing_lows


def auto_fibonacci(data: pd.DataFrame, lookback: int = 5) -> Dict:
    """
    Automatically calculate Fibonacci levels based on detected swing points.
    
    Args:
        data: DataFrame with High, Low, Close columns
        lookback: Bars to look for swing detection
    
    Returns:
        Dictionary with detected swing points and calculated Fibonacci levels
    """
    swing_highs, swing_lows = detect_swing_points(data, lookback)
    
    if not swing_highs or not swing_lows:
        # Fall back to period high/low
        return fibonacci_retracement(data)
    
    # Use most recent significant swing points
    latest_high = max(swing_highs[-3:], key=lambda x: x['price']) if len(swing_highs) >= 1 else None
    latest_low = min(swing_lows[-3:], key=lambda x: x['price']) if len(swing_lows) >= 1 else None
    
    if latest_high and latest_low:
        return fibonacci_retracement(
            data,
            swing_high=latest_high['price'],
            swing_low=latest_low['price']
        )
    
    return fibonacci_retracement(data)
