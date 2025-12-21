"""
Chart Pattern Recognition Module
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional


def detect_head_and_shoulders(
    data: pd.DataFrame,
    window: int = 5
) -> Dict[str, Any]:
    """
    Detect Head and Shoulders pattern.
    
    Pattern: Left Shoulder - Head (higher) - Right Shoulder
    Indicates bearish reversal.
    
    Args:
        data: DataFrame with OHLC data
        window: Lookback for peak detection
    
    Returns:
        Dictionary with pattern detection results
    """
    highs = data['High'].values
    n = len(highs)
    
    # Find local maxima
    peaks = []
    for i in range(window, n - window):
        if all(highs[i] > highs[i-j] for j in range(1, window+1)) and \
           all(highs[i] > highs[i+j] for j in range(1, window+1)):
            peaks.append({'index': i, 'price': highs[i], 'date': data.index[i]})
    
    # Look for H&S pattern in last 3 peaks
    if len(peaks) >= 3:
        recent_peaks = peaks[-3:]
        left = recent_peaks[0]['price']
        head = recent_peaks[1]['price']
        right = recent_peaks[2]['price']
        
        # Head should be highest, shoulders roughly equal
        if head > left and head > right:
            shoulder_diff = abs(left - right) / ((left + right) / 2)
            if shoulder_diff < 0.15:  # Shoulders within 15% of each other
                # Calculate neckline
                neckline = (left + right) / 2 * 0.95  # Approximate
                
                return {
                    'pattern_detected': True,
                    'pattern_type': 'Head and Shoulders (Bearish)',
                    'left_shoulder': round(left, 2),
                    'head': round(head, 2),
                    'right_shoulder': round(right, 2),
                    'neckline': round(neckline, 2),
                    'target_price': round(neckline - (head - neckline), 2),
                    'confidence': round((1 - shoulder_diff) * 100, 1)
                }
    
    return {'pattern_detected': False}


def detect_double_top_bottom(
    data: pd.DataFrame,
    window: int = 5,
    tolerance: float = 0.03
) -> Dict[str, Any]:
    """
    Detect Double Top or Double Bottom patterns.
    
    Double Top: Two peaks at similar levels (bearish)
    Double Bottom: Two troughs at similar levels (bullish)
    
    Args:
        data: DataFrame with OHLC data
        window: Lookback for peak/trough detection
        tolerance: Price tolerance for pattern matching
    
    Returns:
        Dictionary with pattern detection results
    """
    highs = data['High'].values
    lows = data['Low'].values
    n = len(data)
    
    # Find local maxima (tops)
    tops = []
    for i in range(window, n - window):
        if all(highs[i] >= highs[i-j] for j in range(1, window+1)) and \
           all(highs[i] >= highs[i+j] for j in range(1, window+1)):
            tops.append({'index': i, 'price': highs[i]})
    
    # Find local minima (bottoms)
    bottoms = []
    for i in range(window, n - window):
        if all(lows[i] <= lows[i-j] for j in range(1, window+1)) and \
           all(lows[i] <= lows[i+j] for j in range(1, window+1)):
            bottoms.append({'index': i, 'price': lows[i]})
    
    top_result = None
    bottom_result = None
    
    # Check for Double Top
    if len(tops) >= 2:
        recent_tops = tops[-2:]
        price_diff = abs(recent_tops[0]['price'] - recent_tops[1]['price'])
        avg_price = (recent_tops[0]['price'] + recent_tops[1]['price']) / 2
        
        if price_diff / avg_price <= tolerance:
            top_result = {
                'pattern_detected': True,
                'pattern_type': 'Double Top (Bearish)',
                'first_top': round(recent_tops[0]['price'], 2),
                'second_top': round(recent_tops[1]['price'], 2),
                'signal': 'Sell',
                'confidence': round((1 - price_diff/avg_price) * 100, 1),
                'last_index': recent_tops[1]['index']
            }
    
    # Check for Double Bottom
    if len(bottoms) >= 2:
        recent_bottoms = bottoms[-2:]
        price_diff = abs(recent_bottoms[0]['price'] - recent_bottoms[1]['price'])
        avg_price = (recent_bottoms[0]['price'] + recent_bottoms[1]['price']) / 2
        
        if price_diff / avg_price <= tolerance:
            bottom_result = {
                'pattern_detected': True,
                'pattern_type': 'Double Bottom (Bullish)',
                'first_bottom': round(recent_bottoms[0]['price'], 2),
                'second_bottom': round(recent_bottoms[1]['price'], 2),
                'signal': 'Buy',
                'confidence': round((1 - price_diff/avg_price) * 100, 1),
                'last_index': recent_bottoms[1]['index']
            }
    
    # Return the most recent pattern
    if top_result and bottom_result:
        if top_result['last_index'] > bottom_result['last_index']:
            del top_result['last_index']
            return top_result
        else:
            del bottom_result['last_index']
            return bottom_result
    elif top_result:
        del top_result['last_index']
        return top_result
    elif bottom_result:
        del bottom_result['last_index']
        return bottom_result
    
    return {'pattern_detected': False}


def detect_triangle_pattern(
    data: pd.DataFrame,
    lookback: int = 30
) -> Dict[str, Any]:
    """
    Detect Triangle patterns (Ascending, Descending, Symmetrical).
    
    Args:
        data: DataFrame with OHLC data
        lookback: Number of bars to analyze
    
    Returns:
        Dictionary with pattern detection results
    """
    if len(data) < lookback:
        return {'pattern_detected': False}
    
    recent = data.tail(lookback)
    highs = recent['High'].values
    lows = recent['Low'].values
    
    # Calculate trendlines
    x = np.arange(lookback)
    
    # Upper trendline (highs)
    high_slope = np.polyfit(x, highs, 1)[0]
    
    # Lower trendline (lows)
    low_slope = np.polyfit(x, lows, 1)[0]
    
    # Determine triangle type
    if abs(high_slope) < 0.01 and low_slope > 0.05:
        pattern_type = 'Ascending Triangle (Bullish)'
        signal = 'Buy'
    elif high_slope < -0.05 and abs(low_slope) < 0.01:
        pattern_type = 'Descending Triangle (Bearish)'
        signal = 'Sell'
    elif high_slope < -0.02 and low_slope > 0.02:
        pattern_type = 'Symmetrical Triangle (Neutral)'
        signal = 'Wait for breakout'
    else:
        return {'pattern_detected': False}
    
    return {
        'pattern_detected': True,
        'pattern_type': pattern_type,
        'signal': signal,
        'upper_slope': round(high_slope, 4),
        'lower_slope': round(low_slope, 4),
        'current_range': round(highs[-1] - lows[-1], 2),
        'breakout_expected': 'Soon' if (highs[-1] - lows[-1]) < (highs[0] - lows[0]) * 0.5 else 'Later'
    }


def detect_all_chart_patterns(data: pd.DataFrame) -> Dict[str, Any]:
    """
    Run all chart pattern detection algorithms.
    
    Args:
        data: DataFrame with OHLC data
    
    Returns:
        Dictionary with all detected patterns
    """
    patterns = {
        'head_and_shoulders': detect_head_and_shoulders(data),
        'double_top_bottom': detect_double_top_bottom(data),
        'triangle': detect_triangle_pattern(data)
    }
    
    # Summary
    detected = [k for k, v in patterns.items() if v.get('pattern_detected')]
    
    return {
        'patterns': patterns,
        'detected_count': len(detected),
        'detected_patterns': detected
    }
