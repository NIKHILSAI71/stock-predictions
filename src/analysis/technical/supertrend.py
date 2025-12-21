"""
Supertrend Indicator
"""

import pandas as pd
import numpy as np
from typing import Dict
from .indicators import atr


def supertrend(
    data: pd.DataFrame,
    period: int = 10,
    multiplier: float = 3.0
) -> Dict[str, pd.Series]:
    """
    Calculate Supertrend indicator.
    
    Formula:
        ATR = Average True Range over period
        Basic Upper Band = (High + Low) / 2 + (Multiplier × ATR)
        Basic Lower Band = (High + Low) / 2 - (Multiplier × ATR)
        
        Final Upper Band:
            If Current Basic Upper Band < Previous Final Upper Band OR
            Previous Close > Previous Final Upper Band:
                Final Upper Band = Current Basic Upper Band
            Else:
                Final Upper Band = Previous Final Upper Band
                
        Final Lower Band:
            If Current Basic Lower Band > Previous Final Lower Band OR
            Previous Close < Previous Final Lower Band:
                Final Lower Band = Current Basic Lower Band
            Else:
                Final Lower Band = Previous Final Lower Band
        
        Supertrend = Lower Band (uptrend) or Upper Band (downtrend)
        Trend flips when price closes beyond active band
    
    Args:
        data: DataFrame with High, Low, Close columns
        period: ATR period (default 10)
        multiplier: Band multiplier (default 3)
    
    Returns:
        Dictionary with 'supertrend', 'trend' (1 for up, -1 for down), 
        'upper_band', and 'lower_band' Series
    """
    high = data['High']
    low = data['Low']
    close = data['Close']
    
    # Calculate ATR
    atr_values = atr(data, period)
    
    # Calculate basic bands
    hl2 = (high + low) / 2  # Median price
    basic_upper = hl2 + (multiplier * atr_values)
    basic_lower = hl2 - (multiplier * atr_values)
    
    # Initialize final bands
    final_upper = pd.Series(index=data.index, dtype=float)
    final_lower = pd.Series(index=data.index, dtype=float)
    supertrend_values = pd.Series(index=data.index, dtype=float)
    trend = pd.Series(index=data.index, dtype=float)
    
    # Set initial values
    final_upper.iloc[0] = basic_upper.iloc[0]
    final_lower.iloc[0] = basic_lower.iloc[0]
    
    for i in range(1, len(data)):
        # Calculate Final Upper Band
        if basic_upper.iloc[i] < final_upper.iloc[i - 1] or close.iloc[i - 1] > final_upper.iloc[i - 1]:
            final_upper.iloc[i] = basic_upper.iloc[i]
        else:
            final_upper.iloc[i] = final_upper.iloc[i - 1]
        
        # Calculate Final Lower Band
        if basic_lower.iloc[i] > final_lower.iloc[i - 1] or close.iloc[i - 1] < final_lower.iloc[i - 1]:
            final_lower.iloc[i] = basic_lower.iloc[i]
        else:
            final_lower.iloc[i] = final_lower.iloc[i - 1]
    
    # Determine trend and supertrend value
    for i in range(len(data)):
        if i == 0:
            if close.iloc[0] <= final_upper.iloc[0]:
                trend.iloc[0] = 1  # Uptrend
                supertrend_values.iloc[0] = final_lower.iloc[0]
            else:
                trend.iloc[0] = -1  # Downtrend
                supertrend_values.iloc[0] = final_upper.iloc[0]
        else:
            if trend.iloc[i - 1] == 1:  # Previous uptrend
                if close.iloc[i] < final_lower.iloc[i]:
                    trend.iloc[i] = -1  # Flip to downtrend
                    supertrend_values.iloc[i] = final_upper.iloc[i]
                else:
                    trend.iloc[i] = 1  # Continue uptrend
                    supertrend_values.iloc[i] = final_lower.iloc[i]
            else:  # Previous downtrend
                if close.iloc[i] > final_upper.iloc[i]:
                    trend.iloc[i] = 1  # Flip to uptrend
                    supertrend_values.iloc[i] = final_lower.iloc[i]
                else:
                    trend.iloc[i] = -1  # Continue downtrend
                    supertrend_values.iloc[i] = final_upper.iloc[i]
    
    return {
        'supertrend': supertrend_values,
        'trend': trend,
        'upper_band': final_upper,
        'lower_band': final_lower
    }
