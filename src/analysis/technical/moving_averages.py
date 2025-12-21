"""
Moving Averages Module
Implements SMA, EMA, and WMA calculations
"""

import pandas as pd
import numpy as np
from typing import Union


def sma(data: Union[pd.Series, pd.DataFrame], period: int = 20, column: str = "Close") -> pd.Series:
    """
    Calculate Simple Moving Average (SMA).
    
    Formula: SMA = Sum of closing prices over N periods / N
    
    Args:
        data: Price data (Series or DataFrame)
        period: Number of periods for the moving average
        column: Column name if DataFrame is provided
    
    Returns:
        Series with SMA values
    """
    if isinstance(data, pd.DataFrame):
        prices = data[column]
    else:
        prices = data
    
    return prices.rolling(window=period).mean()


def ema(data: Union[pd.Series, pd.DataFrame], period: int = 20, column: str = "Close") -> pd.Series:
    """
    Calculate Exponential Moving Average (EMA).
    
    Formula:
        Multiplier = 2 / (N + 1)
        EMA = (Close - Previous EMA) × Multiplier + Previous EMA
    
    Args:
        data: Price data (Series or DataFrame)
        period: Number of periods for the moving average
        column: Column name if DataFrame is provided
    
    Returns:
        Series with EMA values
    """
    if isinstance(data, pd.DataFrame):
        prices = data[column]
    else:
        prices = data
    
    # pandas ewm uses span which is equivalent to the period
    # The multiplier is calculated as 2 / (span + 1)
    return prices.ewm(span=period, adjust=False).mean()


def wma(data: Union[pd.Series, pd.DataFrame], period: int = 20, column: str = "Close") -> pd.Series:
    """
    Calculate Weighted Moving Average (WMA).
    
    Formula: WMA = (P₁×1 + P₂×2 + ... + Pₙ×n) / (1+2+...+n)
    
    The most recent price has the highest weight.
    
    Args:
        data: Price data (Series or DataFrame)
        period: Number of periods for the moving average
        column: Column name if DataFrame is provided
    
    Returns:
        Series with WMA values
    """
    if isinstance(data, pd.DataFrame):
        prices = data[column]
    else:
        prices = data
    
    weights = np.arange(1, period + 1)
    
    def weighted_avg(x):
        return np.sum(weights * x) / weights.sum()
    
    return prices.rolling(window=period).apply(weighted_avg, raw=True)


def dema(data: Union[pd.Series, pd.DataFrame], period: int = 20, column: str = "Close") -> pd.Series:
    """
    Calculate Double Exponential Moving Average (DEMA).
    
    Formula: DEMA = 2 × EMA - EMA(EMA)
    
    Args:
        data: Price data (Series or DataFrame)
        period: Number of periods
        column: Column name if DataFrame is provided
    
    Returns:
        Series with DEMA values
    """
    if isinstance(data, pd.DataFrame):
        prices = data[column]
    else:
        prices = data
    
    ema1 = ema(prices, period)
    ema2 = ema(ema1, period)
    
    return 2 * ema1 - ema2


def tema(data: Union[pd.Series, pd.DataFrame], period: int = 20, column: str = "Close") -> pd.Series:
    """
    Calculate Triple Exponential Moving Average (TEMA).
    
    Formula: TEMA = 3 × EMA - 3 × EMA(EMA) + EMA(EMA(EMA))
    
    Args:
        data: Price data (Series or DataFrame)
        period: Number of periods
        column: Column name if DataFrame is provided
    
    Returns:
        Series with TEMA values
    """
    if isinstance(data, pd.DataFrame):
        prices = data[column]
    else:
        prices = data
    
    ema1 = ema(prices, period)
    ema2 = ema(ema1, period)
    ema3 = ema(ema2, period)
    
    return 3 * ema1 - 3 * ema2 + ema3


def vwap(data: pd.DataFrame) -> pd.Series:
    """
    Calculate Volume Weighted Average Price (VWAP).
    
    Formula: VWAP = Cumulative(Typical Price × Volume) / Cumulative(Volume)
    Typical Price = (High + Low + Close) / 3
    
    Args:
        data: DataFrame with High, Low, Close, and Volume columns
    
    Returns:
        Series with VWAP values
    """
    typical_price = (data['High'] + data['Low'] + data['Close']) / 3
    
    cumulative_tp_volume = (typical_price * data['Volume']).cumsum()
    cumulative_volume = data['Volume'].cumsum()
    
    return cumulative_tp_volume / cumulative_volume
