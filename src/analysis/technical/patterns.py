"""
Candlestick Pattern Recognition
"""

import pandas as pd
import numpy as np
from typing import Dict, List


def detect_patterns(data: pd.DataFrame) -> Dict[str, List]:
    """
    Detect candlestick patterns in price data.
    
    Args:
        data: DataFrame with Open, High, Low, Close columns
    
    Returns:
        Dictionary with pattern names and their occurrences
    """
    patterns = {
        'doji': [],
        'hammer': [],
        'inverted_hammer': [],
        'bullish_engulfing': [],
        'bearish_engulfing': [],
        'morning_star': [],
        'evening_star': [],
        'shooting_star': [],
        'hanging_man': []
    }
    
    open_prices = data['Open'].values
    high = data['High'].values
    low = data['Low'].values
    close = data['Close'].values
    
    for i in range(2, len(data)):
        # Calculate body and shadows
        body = abs(close[i] - open_prices[i])
        upper_shadow = high[i] - max(close[i], open_prices[i])
        lower_shadow = min(close[i], open_prices[i]) - low[i]
        full_range = high[i] - low[i]
        
        if full_range == 0:
            continue
        
        # Doji - very small body relative to range
        if body / full_range < 0.1:
            patterns['doji'].append({
                'index': data.index[i],
                'price': close[i]
            })
        
        # Hammer - small body at top, long lower shadow
        if (lower_shadow >= 2 * body and 
            upper_shadow <= body * 0.3 and 
            close[i] > open_prices[i]):
            patterns['hammer'].append({
                'index': data.index[i],
                'price': close[i],
                'signal': 'bullish'
            })
        
        # Inverted Hammer - small body at bottom, long upper shadow
        if (upper_shadow >= 2 * body and 
            lower_shadow <= body * 0.3 and 
            close[i] > open_prices[i]):
            patterns['inverted_hammer'].append({
                'index': data.index[i],
                'price': close[i],
                'signal': 'bullish'
            })
        
        # Shooting Star - small body at bottom, long upper shadow (after uptrend)
        if (upper_shadow >= 2 * body and 
            lower_shadow <= body * 0.3 and 
            close[i] < open_prices[i] and
            close[i-1] > open_prices[i-1]):  # Previous bullish
            patterns['shooting_star'].append({
                'index': data.index[i],
                'price': close[i],
                'signal': 'bearish'
            })
        
        # Hanging Man - like hammer but after uptrend
        if (lower_shadow >= 2 * body and 
            upper_shadow <= body * 0.3 and
            close[i-1] > open_prices[i-1] and close[i-2] > open_prices[i-2]):  # Uptrend
            patterns['hanging_man'].append({
                'index': data.index[i],
                'price': close[i],
                'signal': 'bearish'
            })
        
        # Bullish Engulfing
        prev_body = abs(close[i-1] - open_prices[i-1])
        if (close[i-1] < open_prices[i-1] and  # Previous bearish
            close[i] > open_prices[i] and  # Current bullish
            open_prices[i] < close[i-1] and  # Open below prev close
            close[i] > open_prices[i-1]):  # Close above prev open
            patterns['bullish_engulfing'].append({
                'index': data.index[i],
                'price': close[i],
                'signal': 'bullish'
            })
        
        # Bearish Engulfing
        if (close[i-1] > open_prices[i-1] and  # Previous bullish
            close[i] < open_prices[i] and  # Current bearish
            open_prices[i] > close[i-1] and  # Open above prev close
            close[i] < open_prices[i-1]):  # Close below prev open
            patterns['bearish_engulfing'].append({
                'index': data.index[i],
                'price': close[i],
                'signal': 'bearish'
            })
        
        # Morning Star (3 candle pattern)
        if i >= 2:
            body1 = abs(close[i-2] - open_prices[i-2])
            body2 = abs(close[i-1] - open_prices[i-1])
            body3 = abs(close[i] - open_prices[i])
            
            if (close[i-2] < open_prices[i-2] and  # First: bearish
                body2 < body1 * 0.3 and  # Second: small body
                close[i] > open_prices[i] and  # Third: bullish
                close[i] > (open_prices[i-2] + close[i-2]) / 2):  # Close above mid of first
                patterns['morning_star'].append({
                    'index': data.index[i],
                    'price': close[i],
                    'signal': 'bullish'
                })
        
        # Evening Star (3 candle pattern)
        if i >= 2:
            body1 = abs(close[i-2] - open_prices[i-2])
            body2 = abs(close[i-1] - open_prices[i-1])
            
            if (close[i-2] > open_prices[i-2] and  # First: bullish
                body2 < body1 * 0.3 and  # Second: small body
                close[i] < open_prices[i] and  # Third: bearish
                close[i] < (open_prices[i-2] + close[i-2]) / 2):  # Close below mid of first
                patterns['evening_star'].append({
                    'index': data.index[i],
                    'price': close[i],
                    'signal': 'bearish'
                })
    
    return patterns


def get_pattern_signals(data: pd.DataFrame) -> pd.DataFrame:
    """
    Generate trading signals from candlestick patterns.
    
    Args:
        data: DataFrame with OHLC data
    
    Returns:
        DataFrame with pattern signals
    """
    patterns = detect_patterns(data)
    
    signals = pd.DataFrame(index=data.index)
    signals['bullish_patterns'] = 0
    signals['bearish_patterns'] = 0
    signals['pattern_names'] = ''
    
    bullish_patterns = ['hammer', 'inverted_hammer', 'bullish_engulfing', 'morning_star']
    bearish_patterns = ['shooting_star', 'hanging_man', 'bearish_engulfing', 'evening_star']
    
    for pattern_name, occurrences in patterns.items():
        for occ in occurrences:
            idx = occ['index']
            if idx in signals.index:
                if pattern_name in bullish_patterns:
                    signals.loc[idx, 'bullish_patterns'] += 1
                elif pattern_name in bearish_patterns:
                    signals.loc[idx, 'bearish_patterns'] += 1
                
                current = signals.loc[idx, 'pattern_names']
                signals.loc[idx, 'pattern_names'] = f"{current}, {pattern_name}" if current else pattern_name
    
    return signals
