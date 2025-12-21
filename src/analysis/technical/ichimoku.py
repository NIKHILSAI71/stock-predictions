"""
Ichimoku Cloud Indicator Module
Complete implementation of Ichimoku Kinko Hyo (One Glance Equilibrium Chart)
"""

import pandas as pd
import numpy as np
from typing import Dict, Any


def ichimoku_cloud(
    data: pd.DataFrame,
    tenkan_period: int = 9,
    kijun_period: int = 26,
    senkou_b_period: int = 52,
    displacement: int = 26
) -> Dict[str, pd.Series]:
    """
    Calculate complete Ichimoku Cloud components.
    
    Components:
        1. Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
        2. Kijun-sen (Base Line): (26-period high + 26-period low) / 2
        3. Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2, plotted 26 periods ahead
        4. Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2, plotted 26 periods ahead
        5. Chikou Span (Lagging Span): Close plotted 26 periods back
    
    Args:
        data: DataFrame with High, Low, Close columns
        tenkan_period: Tenkan-sen period (default 9)
        kijun_period: Kijun-sen period (default 26)
        senkou_b_period: Senkou Span B period (default 52)
        displacement: Cloud displacement (default 26)
    
    Returns:
        Dictionary with all Ichimoku components
    """
    high = data['High']
    low = data['Low']
    close = data['Close']
    
    # Tenkan-sen (Conversion Line)
    tenkan_high = high.rolling(window=tenkan_period).max()
    tenkan_low = low.rolling(window=tenkan_period).min()
    tenkan_sen = (tenkan_high + tenkan_low) / 2
    
    # Kijun-sen (Base Line)
    kijun_high = high.rolling(window=kijun_period).max()
    kijun_low = low.rolling(window=kijun_period).min()
    kijun_sen = (kijun_high + kijun_low) / 2
    
    # Senkou Span A (Leading Span A) - shifted forward
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(displacement)
    
    # Senkou Span B (Leading Span B) - shifted forward
    senkou_b_high = high.rolling(window=senkou_b_period).max()
    senkou_b_low = low.rolling(window=senkou_b_period).min()
    senkou_span_b = ((senkou_b_high + senkou_b_low) / 2).shift(displacement)
    
    # Chikou Span (Lagging Span) - shifted backward
    chikou_span = close.shift(-displacement)
    
    return {
        'tenkan_sen': tenkan_sen,
        'kijun_sen': kijun_sen,
        'senkou_span_a': senkou_span_a,
        'senkou_span_b': senkou_span_b,
        'chikou_span': chikou_span
    }


def ichimoku_signals(
    data: pd.DataFrame,
    ichimoku_data: Dict[str, pd.Series] = None
) -> Dict[str, Any]:
    """
    Generate trading signals from Ichimoku Cloud.
    
    Signals:
        - TK Cross: Tenkan crosses Kijun (golden/death cross)
        - Price vs Cloud: Price above/below/inside cloud
        - Cloud Color: Senkou A vs Senkou B (bullish/bearish)
        - Chikou confirmation: Lagging span vs price
    
    Args:
        data: DataFrame with OHLC data
        ichimoku_data: Pre-calculated Ichimoku data (optional)
    
    Returns:
        Dictionary with all signals and interpretations
    """
    if ichimoku_data is None:
        ichimoku_data = ichimoku_cloud(data)
    
    close = data['Close']
    tenkan = ichimoku_data['tenkan_sen']
    kijun = ichimoku_data['kijun_sen']
    span_a = ichimoku_data['senkou_span_a']
    span_b = ichimoku_data['senkou_span_b']
    
    current_idx = -1
    
    # Current values
    current_close = close.iloc[current_idx]
    current_tenkan = tenkan.iloc[current_idx]
    current_kijun = kijun.iloc[current_idx]
    current_span_a = span_a.iloc[current_idx] if not pd.isna(span_a.iloc[current_idx]) else 0
    current_span_b = span_b.iloc[current_idx] if not pd.isna(span_b.iloc[current_idx]) else 0
    
    # Cloud boundaries
    cloud_top = max(current_span_a, current_span_b)
    cloud_bottom = min(current_span_a, current_span_b)
    
    # TK Cross Signal
    prev_tenkan = tenkan.iloc[current_idx - 1] if len(tenkan) > 1 else current_tenkan
    prev_kijun = kijun.iloc[current_idx - 1] if len(kijun) > 1 else current_kijun
    
    if current_tenkan > current_kijun and prev_tenkan <= prev_kijun:
        tk_signal = 'Bullish Cross (Golden)'
    elif current_tenkan < current_kijun and prev_tenkan >= prev_kijun:
        tk_signal = 'Bearish Cross (Death)'
    elif current_tenkan > current_kijun:
        tk_signal = 'Bullish (Tenkan above Kijun)'
    else:
        tk_signal = 'Bearish (Tenkan below Kijun)'
    
    # Price vs Cloud
    if current_close > cloud_top:
        price_cloud = 'Above Cloud (Bullish)'
        cloud_score = 80
    elif current_close < cloud_bottom:
        price_cloud = 'Below Cloud (Bearish)'
        cloud_score = 20
    else:
        price_cloud = 'Inside Cloud (Neutral/Consolidation)'
        cloud_score = 50
    
    # Cloud Color (future trend)
    if current_span_a > current_span_b:
        cloud_color = 'Green (Bullish)'
        color_score = 70
    else:
        cloud_color = 'Red (Bearish)'
        color_score = 30
    
    # Overall trend strength
    signals_bullish = 0
    signals_total = 3
    
    if 'Bullish' in tk_signal:
        signals_bullish += 1
    if current_close > cloud_top:
        signals_bullish += 1
    if current_span_a > current_span_b:
        signals_bullish += 1
    
    overall_score = (cloud_score + color_score) / 2
    
    if signals_bullish == 3:
        trend = 'Strong Bullish'
    elif signals_bullish == 2:
        trend = 'Moderate Bullish'
    elif signals_bullish == 1:
        trend = 'Moderate Bearish'
    else:
        trend = 'Strong Bearish'
    
    return {
        'values': {
            'tenkan_sen': round(current_tenkan, 2),
            'kijun_sen': round(current_kijun, 2),
            'senkou_span_a': round(current_span_a, 2),
            'senkou_span_b': round(current_span_b, 2),
            'cloud_top': round(cloud_top, 2),
            'cloud_bottom': round(cloud_bottom, 2)
        },
        'signals': {
            'tk_cross': tk_signal,
            'price_vs_cloud': price_cloud,
            'cloud_color': cloud_color
        },
        'overall_trend': trend,
        'ichimoku_score': round(overall_score, 1),
        'current_price': round(current_close, 2)
    }


def ichimoku_support_resistance(
    data: pd.DataFrame,
    ichimoku_data: Dict[str, pd.Series] = None
) -> Dict[str, float]:
    """
    Extract support and resistance levels from Ichimoku Cloud.
    
    Key levels:
        - Tenkan-sen: Short-term support/resistance
        - Kijun-sen: Medium-term support/resistance (most important)
        - Cloud boundaries: Major support/resistance zones
    
    Args:
        data: DataFrame with OHLC data
        ichimoku_data: Pre-calculated Ichimoku data (optional)
    
    Returns:
        Dictionary with support and resistance levels
    """
    if ichimoku_data is None:
        ichimoku_data = ichimoku_cloud(data)
    
    current_close = data['Close'].iloc[-1]
    tenkan = ichimoku_data['tenkan_sen'].iloc[-1]
    kijun = ichimoku_data['kijun_sen'].iloc[-1]
    span_a = ichimoku_data['senkou_span_a'].iloc[-1]
    span_b = ichimoku_data['senkou_span_b'].iloc[-1]
    
    # Handle NaN values
    span_a = span_a if not pd.isna(span_a) else current_close
    span_b = span_b if not pd.isna(span_b) else current_close
    
    cloud_top = max(span_a, span_b)
    cloud_bottom = min(span_a, span_b)
    
    # Determine support and resistance based on price position
    all_levels = sorted([tenkan, kijun, cloud_top, cloud_bottom])
    
    support_levels = [l for l in all_levels if l < current_close]
    resistance_levels = [l for l in all_levels if l > current_close]
    
    return {
        'immediate_support': round(support_levels[-1], 2) if support_levels else None,
        'major_support': round(support_levels[0], 2) if len(support_levels) > 1 else None,
        'immediate_resistance': round(resistance_levels[0], 2) if resistance_levels else None,
        'major_resistance': round(resistance_levels[-1], 2) if len(resistance_levels) > 1 else None,
        'kijun_sen_level': round(kijun, 2),
        'cloud_top': round(cloud_top, 2),
        'cloud_bottom': round(cloud_bottom, 2)
    }
