"""
Support and Resistance Level Detection
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from scipy.signal import argrelextrema


def find_support_resistance(
    data: pd.DataFrame,
    window: int = 20,
    num_levels: int = 5
) -> Dict[str, List[float]]:
    """
    Identify key support and resistance levels.
    
    Uses local minima/maxima detection and clustering.
    
    Args:
        data: DataFrame with High, Low, Close columns
        window: Lookback window for extrema detection
        num_levels: Maximum number of levels to return
    
    Returns:
        Dictionary with 'support' and 'resistance' level lists
    """
    high = data['High'].values
    low = data['Low'].values
    close = data['Close'].values
    
    # Find local maxima (resistance) and minima (support)
    resistance_idx = argrelextrema(high, np.greater, order=window)[0]
    support_idx = argrelextrema(low, np.less, order=window)[0]
    
    # Get prices at these indices
    resistance_levels = high[resistance_idx].tolist()
    support_levels = low[support_idx].tolist()
    
    # Cluster nearby levels
    current_price = close[-1]
    
    resistance_levels = cluster_levels(resistance_levels, tolerance=0.02)
    support_levels = cluster_levels(support_levels, tolerance=0.02)
    
    # Filter levels relative to current price
    resistance_levels = [r for r in resistance_levels if r > current_price]
    support_levels = [s for s in support_levels if s < current_price]
    
    # Sort and limit
    resistance_levels = sorted(resistance_levels)[:num_levels]
    support_levels = sorted(support_levels, reverse=True)[:num_levels]
    
    return {
        'support': support_levels,
        'resistance': resistance_levels,
        'current_price': current_price
    }


def cluster_levels(levels: List[float], tolerance: float = 0.02) -> List[float]:
    """
    Cluster nearby price levels and return averaged values.
    
    Args:
        levels: List of price levels
        tolerance: Percentage tolerance for clustering (0.02 = 2%)
    
    Returns:
        List of clustered (averaged) price levels
    """
    if not levels:
        return []
    
    levels = sorted(levels)
    clusters = []
    current_cluster = [levels[0]]
    
    for i in range(1, len(levels)):
        # Check if within tolerance of cluster average
        cluster_avg = np.mean(current_cluster)
        if abs(levels[i] - cluster_avg) / cluster_avg <= tolerance:
            current_cluster.append(levels[i])
        else:
            # Save current cluster and start new one
            clusters.append(np.mean(current_cluster))
            current_cluster = [levels[i]]
    
    # Don't forget last cluster
    clusters.append(np.mean(current_cluster))
    
    return [round(c, 2) for c in clusters]


def pivot_points(data: pd.DataFrame, method: str = 'standard') -> Dict[str, float]:
    """
    Calculate pivot points for support/resistance.
    
    Methods:
        - standard: Classic pivot point formula
        - fibonacci: Fibonacci-based pivot points
        - woodie: Woodie's pivot points
        - camarilla: Camarilla pivot points
    
    Args:
        data: DataFrame with High, Low, Close columns
        method: Pivot point calculation method
    
    Returns:
        Dictionary with pivot point levels
    """
    # Use previous day's data
    high = data['High'].iloc[-1]
    low = data['Low'].iloc[-1]
    close = data['Close'].iloc[-1]
    open_price = data['Open'].iloc[-1] if 'Open' in data.columns else close
    
    if method == 'standard':
        pivot = (high + low + close) / 3
        
        return {
            'pivot': round(pivot, 2),
            'r1': round(2 * pivot - low, 2),
            'r2': round(pivot + (high - low), 2),
            'r3': round(high + 2 * (pivot - low), 2),
            's1': round(2 * pivot - high, 2),
            's2': round(pivot - (high - low), 2),
            's3': round(low - 2 * (high - pivot), 2)
        }
    
    elif method == 'fibonacci':
        pivot = (high + low + close) / 3
        range_hl = high - low
        
        return {
            'pivot': round(pivot, 2),
            'r1': round(pivot + 0.382 * range_hl, 2),
            'r2': round(pivot + 0.618 * range_hl, 2),
            'r3': round(pivot + 1.0 * range_hl, 2),
            's1': round(pivot - 0.382 * range_hl, 2),
            's2': round(pivot - 0.618 * range_hl, 2),
            's3': round(pivot - 1.0 * range_hl, 2)
        }
    
    elif method == 'woodie':
        pivot = (high + low + 2 * close) / 4
        
        return {
            'pivot': round(pivot, 2),
            'r1': round(2 * pivot - low, 2),
            'r2': round(pivot + high - low, 2),
            's1': round(2 * pivot - high, 2),
            's2': round(pivot - high + low, 2)
        }
    
    elif method == 'camarilla':
        diff = high - low
        
        return {
            'pivot': round((high + low + close) / 3, 2),
            'r1': round(close + diff * 1.1 / 12, 2),
            'r2': round(close + diff * 1.1 / 6, 2),
            'r3': round(close + diff * 1.1 / 4, 2),
            'r4': round(close + diff * 1.1 / 2, 2),
            's1': round(close - diff * 1.1 / 12, 2),
            's2': round(close - diff * 1.1 / 6, 2),
            's3': round(close - diff * 1.1 / 4, 2),
            's4': round(close - diff * 1.1 / 2, 2)
        }
    
    else:
        raise ValueError(f"Unknown pivot method: {method}")


def volume_profile(
    data: pd.DataFrame,
    bins: int = 20
) -> Dict[str, any]:
    """
    Calculate volume profile to identify high-volume price levels.
    
    Args:
        data: DataFrame with Close and Volume columns
        bins: Number of price bins
    
    Returns:
        Dictionary with volume profile data
    """
    close = data['Close'].values
    volume = data['Volume'].values
    
    # Create price bins
    price_min = close.min()
    price_max = close.max()
    bin_edges = np.linspace(price_min, price_max, bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Aggregate volume by price bin
    volume_by_price = np.zeros(bins)
    for i in range(len(close)):
        bin_idx = np.searchsorted(bin_edges[:-1], close[i]) - 1
        bin_idx = max(0, min(bins - 1, bin_idx))
        volume_by_price[bin_idx] += volume[i]
    
    # Find Point of Control (POC) - highest volume price
    poc_idx = np.argmax(volume_by_price)
    poc_price = bin_centers[poc_idx]
    
    # Calculate Value Area (70% of volume)
    total_volume = volume_by_price.sum()
    target_volume = total_volume * 0.70
    
    # Start from POC and expand outward
    cumulative_volume = volume_by_price[poc_idx]
    lower_idx = poc_idx
    upper_idx = poc_idx
    
    while cumulative_volume < target_volume and (lower_idx > 0 or upper_idx < bins - 1):
        # Add whichever side has more volume
        lower_vol = volume_by_price[lower_idx - 1] if lower_idx > 0 else 0
        upper_vol = volume_by_price[upper_idx + 1] if upper_idx < bins - 1 else 0
        
        if lower_vol >= upper_vol and lower_idx > 0:
            lower_idx -= 1
            cumulative_volume += lower_vol
        elif upper_idx < bins - 1:
            upper_idx += 1
            cumulative_volume += upper_vol
        else:
            break
    
    return {
        'poc': round(poc_price, 2),  # Point of Control
        'vah': round(bin_centers[upper_idx], 2),  # Value Area High
        'val': round(bin_centers[lower_idx], 2),  # Value Area Low
        'profile': list(zip(bin_centers.round(2).tolist(), volume_by_price.tolist()))
    }
