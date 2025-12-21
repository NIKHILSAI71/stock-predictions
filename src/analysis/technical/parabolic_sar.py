"""
Parabolic SAR (Stop and Reverse) Indicator
"""

import pandas as pd
import numpy as np
from typing import Dict


def parabolic_sar(
    data: pd.DataFrame,
    af_start: float = 0.02,
    af_increment: float = 0.02,
    af_max: float = 0.20
) -> Dict[str, pd.Series]:
    """
    Calculate Parabolic SAR.
    
    Formula:
        Uptrend: SAR(next) = SAR(current) + AF × (EP - SAR(current))
        Downtrend: SAR(next) = SAR(current) - AF × (SAR(current) - EP)
        
        AF starts at 0.02, increments by 0.02 on new EP, max 0.20
        EP = Extreme Point (Highest high in uptrend, Lowest low in downtrend)
        Reversal occurs when price crosses SAR
    
    Args:
        data: DataFrame with High, Low, Close columns
        af_start: Initial acceleration factor (default 0.02)
        af_increment: AF increment on new extreme point (default 0.02)
        af_max: Maximum acceleration factor (default 0.20)
    
    Returns:
        Dictionary with 'sar', 'trend' (1 for up, -1 for down), and 'af' Series
    """
    high = data['High'].values
    low = data['Low'].values
    close = data['Close'].values
    n = len(close)
    
    # Initialize arrays
    sar = np.zeros(n)
    trend = np.zeros(n)  # 1 = uptrend, -1 = downtrend
    af = np.zeros(n)
    ep = np.zeros(n)
    
    # Initialize first values
    # Determine initial trend based on first few bars
    if close[1] > close[0]:
        trend[0] = 1  # Uptrend
        sar[0] = low[0]  # Initial SAR at first low
        ep[0] = high[0]  # Initial EP at first high
    else:
        trend[0] = -1  # Downtrend
        sar[0] = high[0]  # Initial SAR at first high
        ep[0] = low[0]  # Initial EP at first low
    
    af[0] = af_start
    
    for i in range(1, n):
        # Previous values
        prev_sar = sar[i - 1]
        prev_trend = trend[i - 1]
        prev_ep = ep[i - 1]
        prev_af = af[i - 1]
        
        if prev_trend == 1:  # Previous bar was uptrend
            # Calculate new SAR
            new_sar = prev_sar + prev_af * (prev_ep - prev_sar)
            
            # SAR cannot be above the prior two lows
            new_sar = min(new_sar, low[i - 1])
            if i >= 2:
                new_sar = min(new_sar, low[i - 2])
            
            # Check for reversal
            if low[i] < new_sar:
                # Trend reversal to downtrend
                trend[i] = -1
                sar[i] = prev_ep  # New SAR = previous EP
                ep[i] = low[i]  # New EP = current low
                af[i] = af_start  # Reset AF
            else:
                # Continue uptrend
                trend[i] = 1
                sar[i] = new_sar
                
                # Update EP and AF if new high
                if high[i] > prev_ep:
                    ep[i] = high[i]
                    af[i] = min(prev_af + af_increment, af_max)
                else:
                    ep[i] = prev_ep
                    af[i] = prev_af
        
        else:  # Previous bar was downtrend
            # Calculate new SAR
            new_sar = prev_sar - prev_af * (prev_sar - prev_ep)
            
            # SAR cannot be below the prior two highs
            new_sar = max(new_sar, high[i - 1])
            if i >= 2:
                new_sar = max(new_sar, high[i - 2])
            
            # Check for reversal
            if high[i] > new_sar:
                # Trend reversal to uptrend
                trend[i] = 1
                sar[i] = prev_ep  # New SAR = previous EP
                ep[i] = high[i]  # New EP = current high
                af[i] = af_start  # Reset AF
            else:
                # Continue downtrend
                trend[i] = -1
                sar[i] = new_sar
                
                # Update EP and AF if new low
                if low[i] < prev_ep:
                    ep[i] = low[i]
                    af[i] = min(prev_af + af_increment, af_max)
                else:
                    ep[i] = prev_ep
                    af[i] = prev_af
    
    return {
        'sar': pd.Series(sar, index=data.index),
        'trend': pd.Series(trend, index=data.index),
        'af': pd.Series(af, index=data.index),
        'ep': pd.Series(ep, index=data.index)
    }
