"""
Volume Analysis Module
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Union


def volume_moving_average(
    data: pd.DataFrame,
    period: int = 20
) -> pd.Series:
    """
    Calculate Volume Moving Average.
    
    Args:
        data: DataFrame with Volume column
        period: Moving average period
    
    Returns:
        Series with volume moving average
    """
    return data['Volume'].rolling(window=period).mean()


def on_balance_volume(
    data: pd.DataFrame
) -> pd.Series:
    """
    Calculate On-Balance Volume (OBV).
    
    Formula:
        If Close > Previous Close: OBV = Previous OBV + Volume
        If Close < Previous Close: OBV = Previous OBV - Volume
        If Close = Previous Close: OBV = Previous OBV
    
    Args:
        data: DataFrame with Close and Volume columns
    
    Returns:
        Series with OBV values
    """
    close = data['Close']
    volume = data['Volume']
    
    obv = pd.Series(index=data.index, dtype=float)
    obv.iloc[0] = volume.iloc[0]
    
    for i in range(1, len(data)):
        if close.iloc[i] > close.iloc[i-1]:
            obv.iloc[i] = obv.iloc[i-1] + volume.iloc[i]
        elif close.iloc[i] < close.iloc[i-1]:
            obv.iloc[i] = obv.iloc[i-1] - volume.iloc[i]
        else:
            obv.iloc[i] = obv.iloc[i-1]
    
    return obv


def accumulation_distribution_line(
    data: pd.DataFrame
) -> pd.Series:
    """
    Calculate Accumulation/Distribution Line (A/D Line).
    
    Formula:
        Money Flow Multiplier = ((Close - Low) - (High - Close)) / (High - Low)
        Money Flow Volume = MFM × Volume
        A/D Line = Previous A/D + Current Money Flow Volume
    
    Args:
        data: DataFrame with OHLCV data
    
    Returns:
        Series with A/D Line values
    """
    high = data['High']
    low = data['Low']
    close = data['Close']
    volume = data['Volume']
    
    # Money Flow Multiplier
    mfm = ((close - low) - (high - close)) / (high - low)
    mfm = mfm.fillna(0)  # Handle division by zero
    
    # Money Flow Volume
    mfv = mfm * volume
    
    # Cumulative A/D Line
    ad_line = mfv.cumsum()
    
    return ad_line


def money_flow_index(
    data: pd.DataFrame,
    period: int = 14
) -> pd.Series:
    """
    Calculate Money Flow Index (MFI) - Volume-weighted RSI.
    
    Formula:
        Typical Price = (High + Low + Close) / 3
        Raw Money Flow = Typical Price × Volume
        Money Flow Ratio = Positive Money Flow / Negative Money Flow
        MFI = 100 - (100 / (1 + Money Flow Ratio))
    
    Args:
        data: DataFrame with OHLCV data
        period: Lookback period
    
    Returns:
        Series with MFI values (0-100)
    """
    # Typical Price
    typical_price = (data['High'] + data['Low'] + data['Close']) / 3
    
    # Raw Money Flow
    raw_money_flow = typical_price * data['Volume']
    
    # Positive and Negative Money Flow
    tp_diff = typical_price.diff()
    positive_flow = raw_money_flow.where(tp_diff > 0, 0)
    negative_flow = raw_money_flow.where(tp_diff < 0, 0)
    
    # Sum over period
    positive_sum = positive_flow.rolling(window=period).sum()
    negative_sum = negative_flow.rolling(window=period).sum()
    
    # Money Flow Ratio and MFI
    money_flow_ratio = positive_sum / negative_sum.replace(0, 1)
    mfi = 100 - (100 / (1 + money_flow_ratio))
    
    return mfi


def volume_price_trend(
    data: pd.DataFrame
) -> pd.Series:
    """
    Calculate Volume Price Trend (VPT).
    
    Formula:
        VPT = Previous VPT + Volume × ((Close - Previous Close) / Previous Close)
    
    Args:
        data: DataFrame with Close and Volume columns
    
    Returns:
        Series with VPT values
    """
    close = data['Close']
    volume = data['Volume']
    
    pct_change = close.pct_change()
    vpt = (volume * pct_change).cumsum()
    
    return vpt


def chaikin_money_flow(
    data: pd.DataFrame,
    period: int = 20
) -> pd.Series:
    """
    Calculate Chaikin Money Flow (CMF).
    
    Formula:
        MFM = ((Close - Low) - (High - Close)) / (High - Low)
        CMF = Sum(MFM × Volume, period) / Sum(Volume, period)
    
    Args:
        data: DataFrame with OHLCV data
        period: Lookback period
    
    Returns:
        Series with CMF values (-1 to 1)
    """
    high = data['High']
    low = data['Low']
    close = data['Close']
    volume = data['Volume']
    
    # Money Flow Multiplier
    mfm = ((close - low) - (high - close)) / (high - low)
    mfm = mfm.fillna(0)
    
    # Money Flow Volume
    mfv = mfm * volume
    
    # CMF
    cmf = mfv.rolling(window=period).sum() / volume.rolling(window=period).sum()
    
    return cmf


def volume_analysis(data: pd.DataFrame) -> Dict[str, Any]:
    """
    Comprehensive volume analysis.
    
    Args:
        data: DataFrame with OHLCV data
    
    Returns:
        Dictionary with all volume indicators
    """
    current_volume = data['Volume'].iloc[-1]
    avg_volume = data['Volume'].rolling(20).mean().iloc[-1]
    
    # Volume spike
    volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
    
    # OBV trend
    obv = on_balance_volume(data)
    obv_sma = obv.rolling(20).mean()
    obv_trend = 'Bullish' if obv.iloc[-1] > obv_sma.iloc[-1] else 'Bearish'
    
    # MFI
    mfi = money_flow_index(data)
    mfi_value = mfi.iloc[-1]
    
    if mfi_value > 80:
        mfi_signal = 'Overbought'
    elif mfi_value < 20:
        mfi_signal = 'Oversold'
    else:
        mfi_signal = 'Neutral'
    
    # CMF
    cmf = chaikin_money_flow(data)
    cmf_value = cmf.iloc[-1]
    cmf_signal = 'Buying Pressure' if cmf_value > 0 else 'Selling Pressure'
    
    return {
        'current_volume': int(current_volume),
        'avg_volume_20': int(avg_volume),
        'volume_ratio': round(volume_ratio, 2),
        'volume_signal': 'High Volume' if volume_ratio > 1.5 else 'Normal' if volume_ratio > 0.5 else 'Low Volume',
        'obv': round(obv.iloc[-1], 0),
        'obv_trend': obv_trend,
        'mfi': round(mfi_value, 2),
        'mfi_signal': mfi_signal,
        'cmf': round(cmf_value, 3),
        'cmf_signal': cmf_signal
    }
