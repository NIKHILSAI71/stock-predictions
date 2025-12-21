"""
Technical Indicators Module
Implements RSI, MACD, Bollinger Bands, Stochastic, ADX, CCI
"""

import pandas as pd
import numpy as np
from typing import Union, Tuple, Dict
from .moving_averages import sma, ema


def rsi(data: Union[pd.Series, pd.DataFrame], period: int = 14, column: str = "Close") -> pd.Series:
    """
    Calculate Relative Strength Index (RSI).
    
    Formula:
        Step 1: Calculate price changes
        Step 2: Separate gains and losses
        Step 3: First Average Gain/Loss = Sum over N periods / N
        Step 4: Smoothed Average = (Previous Average × (N-1) + Current) / N
        Step 5: RS = Average Gain / Average Loss
        Step 6: RSI = 100 - (100 / (1 + RS))
    
    Args:
        data: Price data (Series or DataFrame)
        period: RSI period (default 14)
        column: Column name if DataFrame is provided
    
    Returns:
        Series with RSI values (0-100)
    """
    if isinstance(data, pd.DataFrame):
        prices = data[column]
    else:
        prices = data
    
    # Calculate price changes
    delta = prices.diff()
    
    # Separate gains and losses
    gains = delta.where(delta > 0, 0.0)
    losses = (-delta).where(delta < 0, 0.0)
    
    # Calculate first average using simple average
    first_avg_gain = gains.iloc[:period].mean()
    first_avg_loss = losses.iloc[:period].mean()
    
    # Initialize average gains and losses series
    avg_gains = pd.Series(index=prices.index, dtype=float)
    avg_losses = pd.Series(index=prices.index, dtype=float)
    
    avg_gains.iloc[period - 1] = first_avg_gain
    avg_losses.iloc[period - 1] = first_avg_loss
    
    # Calculate subsequent smoothed averages using Wilder's smoothing
    for i in range(period, len(prices)):
        avg_gains.iloc[i] = (avg_gains.iloc[i - 1] * (period - 1) + gains.iloc[i]) / period
        avg_losses.iloc[i] = (avg_losses.iloc[i - 1] * (period - 1) + losses.iloc[i]) / period
    
    # Calculate RS and RSI
    rs = avg_gains / avg_losses
    rsi_values = 100 - (100 / (1 + rs))
    
    # Handle edge case where average loss is 0
    rsi_values = rsi_values.replace([np.inf, -np.inf], 100)
    rsi_values = rsi_values.fillna(50)  # Neutral when undefined
    
    return rsi_values


def macd(
    data: Union[pd.Series, pd.DataFrame],
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
    column: str = "Close"
) -> Dict[str, pd.Series]:
    """
    Calculate Moving Average Convergence Divergence (MACD).
    
    Formula:
        MACD Line = 12-period EMA - 26-period EMA
        Signal Line = 9-period EMA of MACD Line
        Histogram = MACD Line - Signal Line
    
    Args:
        data: Price data (Series or DataFrame)
        fast_period: Fast EMA period (default 12)
        slow_period: Slow EMA period (default 26)
        signal_period: Signal line EMA period (default 9)
        column: Column name if DataFrame is provided
    
    Returns:
        Dictionary with 'macd', 'signal', and 'histogram' Series
    """
    if isinstance(data, pd.DataFrame):
        prices = data[column]
    else:
        prices = data
    
    # Calculate EMAs
    ema_fast = ema(prices, fast_period)
    ema_slow = ema(prices, slow_period)
    
    # Calculate MACD line
    macd_line = ema_fast - ema_slow
    
    # Calculate signal line
    signal_line = ema(macd_line, signal_period)
    
    # Calculate histogram
    histogram = macd_line - signal_line
    
    return {
        'macd': macd_line,
        'signal': signal_line,
        'histogram': histogram
    }


def bollinger_bands(
    data: Union[pd.Series, pd.DataFrame],
    period: int = 20,
    std_dev: float = 2.0,
    column: str = "Close"
) -> Dict[str, pd.Series]:
    """
    Calculate Bollinger Bands.
    
    Formula:
        Middle Band = 20-day SMA
        Upper Band = Middle Band + (2 × 20-day Std Dev)
        Lower Band = Middle Band - (2 × 20-day Std Dev)
    
    Args:
        data: Price data (Series or DataFrame)
        period: SMA period (default 20)
        std_dev: Number of standard deviations (default 2)
        column: Column name if DataFrame is provided
    
    Returns:
        Dictionary with 'upper', 'middle', 'lower', and 'bandwidth' Series
    """
    if isinstance(data, pd.DataFrame):
        prices = data[column]
    else:
        prices = data
    
    # Calculate middle band (SMA)
    middle = sma(prices, period)
    
    # Calculate standard deviation
    rolling_std = prices.rolling(window=period).std()
    
    # Calculate upper and lower bands
    upper = middle + (std_dev * rolling_std)
    lower = middle - (std_dev * rolling_std)
    
    # Calculate bandwidth (volatility indicator)
    bandwidth = (upper - lower) / middle * 100
    
    # Calculate %B (position within bands)
    percent_b = (prices - lower) / (upper - lower)
    
    return {
        'upper': upper,
        'middle': middle,
        'lower': lower,
        'bandwidth': bandwidth,
        'percent_b': percent_b
    }


def stochastic_oscillator(
    data: pd.DataFrame,
    k_period: int = 14,
    d_period: int = 3
) -> Dict[str, pd.Series]:
    """
    Calculate Stochastic Oscillator.
    
    Formula:
        %K = ((Close - Lowest Low) / (Highest High - Lowest Low)) × 100
        %D = 3-period SMA of %K
    
    Args:
        data: DataFrame with High, Low, Close columns
        k_period: %K period (default 14)
        d_period: %D smoothing period (default 3)
    
    Returns:
        Dictionary with 'k' and 'd' Series
    """
    # Calculate lowest low and highest high over the period
    lowest_low = data['Low'].rolling(window=k_period).min()
    highest_high = data['High'].rolling(window=k_period).max()
    
    # Calculate %K
    k = ((data['Close'] - lowest_low) / (highest_high - lowest_low)) * 100
    
    # Calculate %D (SMA of %K)
    d = sma(k, d_period)
    
    return {
        'k': k,
        'd': d
    }


def adx(data: pd.DataFrame, period: int = 14) -> Dict[str, pd.Series]:
    """
    Calculate Average Directional Index (ADX).
    
    Formula:
        Step 1: Calculate +DM and -DM
        Step 2: Calculate True Range (TR)
        Step 3: Smooth +DM, -DM, TR using Wilder's smoothing
        Step 4: +DI = 100 × (Smoothed +DM / Smoothed TR)
        Step 5: -DI = 100 × (Smoothed -DM / Smoothed TR)
        Step 6: DX = 100 × |+DI - -DI| / (+DI + -DI)
        Step 7: ADX = Smoothed average of DX
    
    Args:
        data: DataFrame with High, Low, Close columns
        period: ADX period (default 14)
    
    Returns:
        Dictionary with 'adx', 'plus_di', and 'minus_di' Series
    """
    high = data['High']
    low = data['Low']
    close = data['Close']
    
    # Calculate +DM and -DM
    up_move = high.diff()
    down_move = low.diff().multiply(-1)
    
    plus_dm = pd.Series(0.0, index=data.index)
    minus_dm = pd.Series(0.0, index=data.index)
    
    # +DM = Up Move if Up Move > Down Move and > 0
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    # -DM = Down Move if Down Move > Up Move and > 0
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    plus_dm = pd.Series(plus_dm, index=data.index)
    minus_dm = pd.Series(minus_dm, index=data.index)
    
    # Calculate True Range
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Smooth using Wilder's method (equivalent to EMA with period multiplier)
    atr = tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    smoothed_plus_dm = plus_dm.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    smoothed_minus_dm = minus_dm.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    
    # Calculate +DI and -DI
    plus_di = 100 * (smoothed_plus_dm / atr)
    minus_di = 100 * (smoothed_minus_dm / atr)
    
    # Calculate DX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    dx = dx.replace([np.inf, -np.inf], 0).fillna(0)
    
    # Calculate ADX (smoothed DX)
    adx_values = dx.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    
    return {
        'adx': adx_values,
        'plus_di': plus_di,
        'minus_di': minus_di,
        'atr': atr
    }


def cci(data: pd.DataFrame, period: int = 20) -> pd.Series:
    """
    Calculate Commodity Channel Index (CCI).
    
    Formula:
        Typical Price = (High + Low + Close) / 3
        SMA of TP = Sum of TP over N periods / N
        Mean Deviation = Sum of |TP - SMA| / N
        CCI = (TP - SMA) / (0.015 × Mean Deviation)
    
    Args:
        data: DataFrame with High, Low, Close columns
        period: CCI period (default 20)
    
    Returns:
        Series with CCI values
    """
    # Calculate Typical Price
    typical_price = (data['High'] + data['Low'] + data['Close']) / 3
    
    # Calculate SMA of Typical Price
    tp_sma = sma(typical_price, period)
    
    # Calculate Mean Deviation
    def mean_deviation(x):
        return np.mean(np.abs(x - np.mean(x)))
    
    mad = typical_price.rolling(window=period).apply(mean_deviation, raw=True)
    
    # Calculate CCI
    # The constant 0.015 ensures ~70-80% of values fall between -100 and +100
    cci_values = (typical_price - tp_sma) / (0.015 * mad)
    
    return cci_values


def atr(data: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculate Average True Range (ATR).
    
    Formula:
        TR = max(High-Low, |High-PrevClose|, |Low-PrevClose|)
        ATR = Smoothed average of TR
    
    Args:
        data: DataFrame with High, Low, Close columns
        period: ATR period (default 14)
    
    Returns:
        Series with ATR values
    """
    high = data['High']
    low = data['Low']
    close = data['Close']
    
    # Calculate True Range components
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    
    # True Range is the maximum of the three
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # ATR is the smoothed average using Wilder's method
    atr_values = tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    
    return atr_values


def williams_r(data: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculate Williams %R.
    
    Formula:
        %R = ((Highest High - Close) / (Highest High - Lowest Low)) × -100
    
    Args:
        data: DataFrame with High, Low, Close columns
        period: Lookback period (default 14)
    
    Returns:
        Series with Williams %R values (-100 to 0)
    """
    highest_high = data['High'].rolling(window=period).max()
    lowest_low = data['Low'].rolling(window=period).min()
    
    wr = ((highest_high - data['Close']) / (highest_high - lowest_low)) * -100
    
    return wr


def momentum(data: Union[pd.Series, pd.DataFrame], period: int = 10, column: str = "Close") -> pd.Series:
    """
    Calculate Momentum indicator.
    
    Formula: Momentum = Close - Close(n periods ago)
    
    Args:
        data: Price data (Series or DataFrame)
        period: Lookback period (default 10)
        column: Column name if DataFrame is provided
    
    Returns:
        Series with Momentum values
    """
    if isinstance(data, pd.DataFrame):
        prices = data[column]
    else:
        prices = data
    
    return prices.diff(period)


def roc(data: Union[pd.Series, pd.DataFrame], period: int = 10, column: str = "Close") -> pd.Series:
    """
    Calculate Rate of Change (ROC).
    
    Formula: ROC = ((Close - Close(n periods ago)) / Close(n periods ago)) × 100
    
    Args:
        data: Price data (Series or DataFrame)
        period: Lookback period (default 10)
        column: Column name if DataFrame is provided
    
    Returns:
        Series with ROC values (percentage)
    """
    if isinstance(data, pd.DataFrame):
        prices = data[column]
    else:
        prices = data
    
    return ((prices - prices.shift(period)) / prices.shift(period)) * 100


def true_strength_index(
    data: Union[pd.Series, pd.DataFrame],
    long_period: int = 25,
    short_period: int = 13,
    signal_period: int = 7,
    column: str = "Close"
) -> Dict[str, pd.Series]:
    """
    Calculate True Strength Index (TSI).
    
    Formula:
        Price Change = Close - Previous Close
        Double Smoothed PC = EMA(EMA(PC, long), short)
        Double Smoothed Abs PC = EMA(EMA(|PC|, long), short)
        TSI = 100 × (Double Smoothed PC / Double Smoothed Abs PC)
        Signal = EMA(TSI, signal_period)
    
    Interpretation:
        > 25: Overbought
        < -25: Oversold
        Crossover of TSI and Signal: Buy/Sell signals
    
    Args:
        data: Price data (Series or DataFrame)
        long_period: First EMA period (default 25)
        short_period: Second EMA period (default 13)
        signal_period: Signal line EMA period (default 7)
        column: Column name if DataFrame is provided
    
    Returns:
        Dictionary with 'tsi' and 'signal' Series
    """
    if isinstance(data, pd.DataFrame):
        prices = data[column]
    else:
        prices = data
    
    # Calculate price change
    price_change = prices.diff()
    
    # Double smooth the price change
    ema1_pc = price_change.ewm(span=long_period, adjust=False).mean()
    double_smoothed_pc = ema1_pc.ewm(span=short_period, adjust=False).mean()
    
    # Double smooth the absolute price change
    abs_price_change = price_change.abs()
    ema1_abs = abs_price_change.ewm(span=long_period, adjust=False).mean()
    double_smoothed_abs = ema1_abs.ewm(span=short_period, adjust=False).mean()
    
    # Calculate TSI
    tsi = 100 * (double_smoothed_pc / double_smoothed_abs)
    tsi = tsi.replace([np.inf, -np.inf], 0).fillna(0)
    
    # Calculate signal line
    signal = tsi.ewm(span=signal_period, adjust=False).mean()
    
    return {
        'tsi': tsi,
        'signal': signal
    }

