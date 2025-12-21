"""
Market Breadth Indicators Module
Measures the overall health of the market by analyzing advancing vs. declining stocks.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional


def advance_decline_line(
    advances: pd.Series,
    declines: pd.Series
) -> pd.Series:
    """
    Calculate Advance/Decline Line (A/D Line).
    
    Formula:
        A/D Line = Previous A/D + (Advances - Declines)
    
    Args:
        advances: Number of advancing stocks per day
        declines: Number of declining stocks per day
    
    Returns:
        Series with cumulative A/D Line values
    """
    net_advances = advances - declines
    ad_line = net_advances.cumsum()
    return ad_line


def advance_decline_ratio(
    advances: int,
    declines: int
) -> Dict[str, Any]:
    """
    Calculate Advance/Decline Ratio.
    
    Formula:
        A/D Ratio = Advances / Declines
    
    Interpretation:
        > 2.0: Strong bullish breadth
        1.0 - 2.0: Moderate bullish
        0.5 - 1.0: Moderate bearish
        < 0.5: Strong bearish breadth
    
    Args:
        advances: Number of advancing stocks
        declines: Number of declining stocks
    
    Returns:
        Dictionary with ratio and interpretation
    """
    if declines == 0:
        ratio = float('inf') if advances > 0 else 1.0
    else:
        ratio = advances / declines
    
    if ratio > 2.0:
        signal = 'Strong Bullish'
    elif ratio > 1.0:
        signal = 'Moderate Bullish'
    elif ratio > 0.5:
        signal = 'Moderate Bearish'
    else:
        signal = 'Strong Bearish'
    
    return {
        'ratio': round(ratio, 3) if ratio != float('inf') else 'Infinity',
        'advances': advances,
        'declines': declines,
        'net': advances - declines,
        'signal': signal
    }


def trin_arms_index(
    advances: int,
    declines: int,
    advancing_volume: float,
    declining_volume: float
) -> Dict[str, Any]:
    """
    Calculate TRIN (Arms Index).
    
    Formula:
        TRIN = (Advances / Declines) / (Advancing Volume / Declining Volume)
    
    Interpretation:
        < 0.5: Overbought (too bullish, potential reversal)
        0.5 - 1.0: Bullish
        1.0: Neutral
        1.0 - 2.0: Bearish
        > 2.0: Oversold (too bearish, potential bounce)
    
    Args:
        advances: Number of advancing stocks
        declines: Number of declining stocks
        advancing_volume: Total volume of advancing stocks
        declining_volume: Total volume of declining stocks
    
    Returns:
        Dictionary with TRIN value and interpretation
    """
    if declines == 0 or declining_volume == 0:
        return {'error': 'Cannot calculate TRIN with zero declines or volume'}
    
    ad_ratio = advances / declines
    volume_ratio = advancing_volume / declining_volume
    
    if volume_ratio == 0:
        return {'error': 'Volume ratio is zero'}
    
    trin = ad_ratio / volume_ratio
    
    if trin < 0.5:
        interpretation = 'Overbought - Potential Reversal'
        signal = 'bearish'
    elif trin < 1.0:
        interpretation = 'Bullish'
        signal = 'bullish'
    elif trin == 1.0:
        interpretation = 'Neutral'
        signal = 'neutral'
    elif trin < 2.0:
        interpretation = 'Bearish'
        signal = 'bearish'
    else:
        interpretation = 'Oversold - Potential Bounce'
        signal = 'contrarian_bullish'
    
    return {
        'trin': round(trin, 3),
        'ad_ratio': round(ad_ratio, 3),
        'volume_ratio': round(volume_ratio, 3),
        'interpretation': interpretation,
        'signal': signal
    }


def mcclellan_oscillator(
    advances: pd.Series,
    declines: pd.Series,
    fast_period: int = 19,
    slow_period: int = 39
) -> pd.Series:
    """
    Calculate McClellan Oscillator.
    
    Formula:
        Net Advances = Advances - Declines
        McClellan Oscillator = EMA(19) of Net Advances - EMA(39) of Net Advances
    
    Interpretation:
        > 100: Strongly overbought
        50 to 100: Overbought
        -50 to 50: Neutral zone
        -100 to -50: Oversold
        < -100: Strongly oversold
    
    Args:
        advances: Series of daily advancing stocks
        declines: Series of daily declining stocks
        fast_period: Fast EMA period (default 19)
        slow_period: Slow EMA period (default 39)
    
    Returns:
        Series with McClellan Oscillator values
    """
    net_advances = advances - declines
    
    # Calculate EMAs using the ratio method (Appel's original formula)
    # Fast EMA multiplier: 2 / (19 + 1) = 0.10
    # Slow EMA multiplier: 2 / (39 + 1) = 0.05
    fast_ema = net_advances.ewm(span=fast_period, adjust=False).mean()
    slow_ema = net_advances.ewm(span=slow_period, adjust=False).mean()
    
    mcclellan_osc = fast_ema - slow_ema
    
    return mcclellan_osc


def mcclellan_summation_index(
    advances: pd.Series,
    declines: pd.Series
) -> pd.Series:
    """
    Calculate McClellan Summation Index.
    
    Formula:
        Summation Index = Cumulative sum of McClellan Oscillator
        Or: Previous Sum + Current McClellan Oscillator
    
    Interpretation:
        Rising above 0: Bullish market breadth
        Falling below 0: Bearish market breadth
        Extreme high (>1000): Overbought market
        Extreme low (<-1000): Oversold market
    
    Args:
        advances: Series of daily advancing stocks
        declines: Series of daily declining stocks
    
    Returns:
        Series with McClellan Summation Index values
    """
    mcclellan_osc = mcclellan_oscillator(advances, declines)
    summation_index = mcclellan_osc.cumsum()
    
    return summation_index


def breadth_thrust(
    advances: pd.Series,
    total_issues: pd.Series,
    period: int = 10
) -> Dict[str, Any]:
    """
    Calculate Breadth Thrust Indicator.
    
    Formula:
        Breadth Thrust = EMA(10) of (Advances / Total Issues)
    
    Signal:
        A Breadth Thrust occurs when the indicator moves from below 0.40
        to above 0.615 within 10 days - a rare but powerful bullish signal.
    
    Args:
        advances: Series of daily advancing stocks
        total_issues: Series of total issues traded
        period: EMA period (default 10)
    
    Returns:
        Dictionary with current value and thrust detection
    """
    breadth_ratio = advances / total_issues
    thrust_indicator = breadth_ratio.ewm(span=period, adjust=False).mean()
    
    current_value = thrust_indicator.iloc[-1]
    
    # Check for thrust signal (move from <0.40 to >0.615 in 10 days)
    thrust_detected = False
    if len(thrust_indicator) >= 10:
        recent = thrust_indicator.iloc[-10:]
        min_val = recent.min()
        max_val = recent.max()
        if min_val < 0.40 and max_val > 0.615:
            thrust_detected = True
    
    return {
        'current_value': round(current_value, 4),
        'thrust_detected': thrust_detected,
        'signal': 'Strong Bullish Signal' if thrust_detected else 'No Signal',
        'interpretation': {
            'above_0.615': 'Overbought' if current_value > 0.615 else False,
            'below_0.40': 'Oversold' if current_value < 0.40 else False
        }
    }


def high_low_index(
    new_highs: pd.Series,
    new_lows: pd.Series,
    period: int = 10
) -> pd.Series:
    """
    Calculate High-Low Index.
    
    Formula:
        High-Low Index = EMA of (New Highs / (New Highs + New Lows))
    
    Interpretation:
        > 70%: Bullish (more new highs)
        50%: Neutral
        < 30%: Bearish (more new lows)
    
    Args:
        new_highs: Series of stocks making new 52-week highs
        new_lows: Series of stocks making new 52-week lows
        period: EMA smoothing period
    
    Returns:
        Series with High-Low Index values (0-100)
    """
    total = new_highs + new_lows
    # Avoid division by zero
    ratio = new_highs / total.replace(0, 1)
    ratio = ratio.fillna(0.5)  # Default to neutral if no data
    
    hl_index = ratio.ewm(span=period, adjust=False).mean() * 100
    
    return hl_index


def percent_above_ma(
    stock_prices: pd.DataFrame,
    ma_period: int = 200
) -> float:
    """
    Calculate percentage of stocks above their moving average.
    
    This is a market breadth indicator showing broad market health.
    
    Args:
        stock_prices: DataFrame with stock prices (columns = tickers)
        ma_period: Moving average period (50, 100, or 200)
    
    Returns:
        Percentage of stocks above their MA
    """
    above_ma_count = 0
    total_stocks = 0
    
    for col in stock_prices.columns:
        prices = stock_prices[col].dropna()
        if len(prices) >= ma_period:
            ma = prices.rolling(window=ma_period).mean()
            if prices.iloc[-1] > ma.iloc[-1]:
                above_ma_count += 1
            total_stocks += 1
    
    if total_stocks == 0:
        return 0.0
    
    return round((above_ma_count / total_stocks) * 100, 2)


def comprehensive_breadth_analysis(
    advances: int,
    declines: int,
    advancing_volume: float,
    declining_volume: float,
    new_highs: int,
    new_lows: int
) -> Dict[str, Any]:
    """
    Comprehensive market breadth analysis.
    
    Args:
        advances: Number of advancing stocks
        declines: Number of declining stocks
        advancing_volume: Volume of advancing stocks
        declining_volume: Volume of declining stocks
        new_highs: Stocks at 52-week highs
        new_lows: Stocks at 52-week lows
    
    Returns:
        Dictionary with all breadth indicators
    """
    ad_result = advance_decline_ratio(advances, declines)
    trin_result = trin_arms_index(advances, declines, advancing_volume, declining_volume)
    
    # High-Low analysis
    total_hl = new_highs + new_lows
    hl_ratio = (new_highs / total_hl * 100) if total_hl > 0 else 50
    
    # Overall score (0-100)
    scores = []
    
    # A/D Ratio score
    if isinstance(ad_result['ratio'], (int, float)):
        if ad_result['ratio'] > 1.5:
            scores.append(80)
        elif ad_result['ratio'] > 1.0:
            scores.append(60)
        elif ad_result['ratio'] > 0.7:
            scores.append(40)
        else:
            scores.append(20)
    
    # TRIN score (inverted - lower is more bullish)
    if 'trin' in trin_result:
        trin = trin_result['trin']
        if trin < 0.7:
            scores.append(75)
        elif trin < 1.0:
            scores.append(60)
        elif trin < 1.3:
            scores.append(40)
        else:
            scores.append(25)
    
    # High-Low score
    scores.append(hl_ratio)
    
    overall_score = sum(scores) / len(scores) if scores else 50
    
    return {
        'advance_decline': ad_result,
        'trin': trin_result,
        'new_highs': new_highs,
        'new_lows': new_lows,
        'high_low_ratio': round(hl_ratio, 2),
        'overall_breadth_score': round(overall_score, 1),
        'market_health': 'Healthy' if overall_score > 60 else 'Neutral' if overall_score > 40 else 'Weak'
    }
