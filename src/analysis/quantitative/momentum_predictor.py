"""
Enhanced Momentum Predictor for Stock Price Direction
Multi-timeframe momentum analysis with adaptive thresholds and regime detection.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


def calculate_rsi(prices: pd.Series, period: int = 14) -> float:
    """Calculate Relative Strength Index."""
    returns = prices.pct_change()
    gains = returns.copy()
    gains[gains < 0] = 0
    losses = -returns.copy()
    losses[losses < 0] = 0

    avg_gain = gains.tail(period).mean()
    avg_loss = losses.tail(period).mean()

    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 50.0

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return float(rsi)


def check_ma_crossover(prices: pd.Series, short_period: int = 5, long_period: int = 20) -> Dict[str, Any]:
    """Check moving average crossover signals."""
    if len(prices) < long_period:
        return {"signal": 0, "strength": 0}

    sma_short = prices.rolling(short_period).mean()
    sma_long = prices.rolling(long_period).mean()

    current_diff = sma_short.iloc[-1] - sma_long.iloc[-1]
    prev_diff = sma_short.iloc[-2] - sma_long.iloc[-2]

    # Check for crossover
    if current_diff > 0 and prev_diff <= 0:
        # Bullish crossover (golden cross)
        return {"signal": 1, "strength": abs(current_diff / sma_long.iloc[-1]) * 100}
    elif current_diff < 0 and prev_diff >= 0:
        # Bearish crossover (death cross)
        return {"signal": -1, "strength": abs(current_diff / sma_long.iloc[-1]) * 100}
    else:
        # No crossover, but measure current relationship
        signal = 1 if current_diff > 0 else -1
        strength = abs(current_diff / sma_long.iloc[-1]) * 100
        return {"signal": signal * 0.5, "strength": strength}


def volume_confirmation(stock_data: pd.DataFrame) -> float:
    """Check volume trend for confirmation (0-100 score)."""
    if 'Volume' not in stock_data.columns:
        return 50.0  # Neutral if no volume data

    volume = stock_data['Volume']
    if len(volume) < 20:
        return 50.0

    # Compare recent volume to average
    recent_vol = volume.tail(5).mean()
    avg_vol = volume.tail(20).mean()

    if avg_vol == 0:
        return 50.0

    volume_ratio = recent_vol / avg_vol

    # High volume is good (>1.2 = 70+, >1.5 = 85+)
    if volume_ratio > 1.5:
        return 90.0
    elif volume_ratio > 1.2:
        return 75.0
    elif volume_ratio > 0.8:
        return 60.0
    elif volume_ratio > 0.6:
        return 45.0
    else:
        return 30.0  # Very low volume


def trend_consistency(prices: pd.Series, period: int = 10) -> float:
    """Measure trend consistency (0-100 score)."""
    if len(prices) < period:
        return 50.0

    returns = prices.pct_change().tail(period)

    # Count positive vs negative returns
    positive_count = (returns > 0).sum()
    consistency = (positive_count / period) * 100

    return float(consistency)


def detect_market_regime(stock_data: pd.DataFrame) -> Dict[str, Any]:
    """
    Detect market regime: Bullish, Bearish, or Sideways.

    Returns regime with adaptive thresholds for momentum signals.
    """
    close = stock_data['Close']

    if len(close) < 60:
        return {
            "regime": "Unknown",
            "momentum_threshold": 0.02,
            "confidence_multiplier": 0.8
        }

    # Calculate rolling statistics
    returns = close.pct_change()
    rolling_mean = returns.rolling(60).mean()
    rolling_vol = returns.rolling(60).std()

    current_mean = rolling_mean.iloc[-1]
    current_vol = rolling_vol.iloc[-1]

    # Thresholds
    mean_threshold = returns.tail(252).mean()  # 1-year average
    vol_threshold = returns.tail(252).std()  # 1-year volatility

    # Classify regime
    if current_mean > mean_threshold * 1.5:
        regime = "Strong Bullish"
        momentum_threshold = 0.015  # Lower threshold in bull market
        confidence_multiplier = 1.2  # Higher confidence
    elif current_mean > mean_threshold:
        regime = "Bullish"
        momentum_threshold = 0.02
        confidence_multiplier = 1.1
    elif current_mean < mean_threshold * -1.5:
        regime = "Strong Bearish"
        momentum_threshold = 0.025  # Higher threshold in bear market
        confidence_multiplier = 0.7  # Lower confidence
    elif current_mean < mean_threshold:
        regime = "Bearish"
        momentum_threshold = 0.02
        confidence_multiplier = 0.85
    else:
        regime = "Sideways"
        momentum_threshold = 0.03  # Highest threshold in sideways
        confidence_multiplier = 0.6  # Lowest confidence (choppy)

    return {
        "regime": regime,
        "momentum_threshold": momentum_threshold,
        "confidence_multiplier": confidence_multiplier,
        "current_return": float(current_mean * 252),  # Annualized
        "current_volatility": float(current_vol * np.sqrt(252))  # Annualized
    }


def get_momentum_prediction(
    stock_data: pd.DataFrame,
    short_period: int = 5,
    medium_period: int = 20,
    long_period: int = 50
) -> Dict[str, Any]:
    """
    Enhanced momentum prediction with multi-timeframe analysis.

    Combines:
    - Multi-timeframe momentum (5, 20, 50 days)
    - RSI indicator (30% weight)
    - MA crossovers (30% weight)
    - Volume confirmation (20% weight)
    - Trend consistency (20% weight)
    - Adaptive thresholds by market regime

    Args:
        stock_data: DataFrame with OHLCV data
        short_period: Short-term momentum period (default: 5)
        medium_period: Medium-term momentum period (default: 20)
        long_period: Long-term momentum period (default: 50)

    Returns:
        Dictionary with prediction results
    """
    try:
        close = stock_data['Close']

        if len(close) < long_period:
            return {
                "error": f"Insufficient data: need at least {long_period} days",
                "direction": "Neutral",
                "confidence": 50,
                "status": "failed"
            }

        # 1. Multi-timeframe Momentum Signals
        momentum_signals = {}

        for period, name in [(short_period, 'short'), (medium_period, 'medium'), (long_period, 'long')]:
            if len(close) > period:
                momentum = (close.iloc[-1] - close.iloc[-period]) / \
                    close.iloc[-period]
                momentum_signals[name] = {
                    "value": float(momentum),
                    "signal": 1 if momentum > 0 else -1
                }

        # 2. Momentum Strength Score (0-100)

        # RSI Component (30% weight)
        rsi_value = calculate_rsi(close, period=14)
        # Normalize RSI: 70+ = Bullish, 30- = Bearish, 50 = Neutral
        if rsi_value > 70:
            rsi_score = 70 + (rsi_value - 70) / 30 * 30  # 70-100
        elif rsi_value < 30:
            rsi_score = 30 - (30 - rsi_value) / 30 * 30  # 0-30
        else:
            rsi_score = rsi_value  # 30-70

        # MA Crossover Component (30% weight)
        ma_cross = check_ma_crossover(close, short_period, medium_period)
        # Normalize to 0-100: positive signal = 50-100, negative = 0-50
        ma_score = 50 + (ma_cross['signal'] * ma_cross['strength'])
        ma_score = max(0, min(100, ma_score))

        # Volume Confirmation Component (20% weight)
        volume_score = volume_confirmation(stock_data)

        # Trend Consistency Component (20% weight)
        consistency_score = trend_consistency(close, period=10)

        # Combined Momentum Strength
        momentum_strength = (
            rsi_score * 0.3 +
            ma_score * 0.3 +
            volume_score * 0.2 +
            consistency_score * 0.2
        )

        # 3. Detect Market Regime and Apply Adaptive Thresholds
        regime_info = detect_market_regime(stock_data)
        regime = regime_info['regime']
        confidence_multiplier = regime_info['confidence_multiplier']

        # 4. Aggregate Signals
        # Weight: short=0.2, medium=0.5, long=0.3 (medium-term most important)
        weighted_momentum = (
            momentum_signals.get('short', {}).get('value', 0) * 0.2 +
            momentum_signals.get('medium', {}).get('value', 0) * 0.5 +
            momentum_signals.get('long', {}).get('value', 0) * 0.3
        )

        # Determine Direction
        momentum_threshold = regime_info['momentum_threshold']

        if weighted_momentum > momentum_threshold:
            direction = "Bullish"
            base_confidence = momentum_strength
        elif weighted_momentum < -momentum_threshold:
            direction = "Bearish"
            base_confidence = 100 - momentum_strength  # Invert for bearish
        else:
            direction = "Neutral"
            base_confidence = 50

        # Apply regime-based confidence adjustment
        final_confidence = base_confidence * confidence_multiplier

        # Clamp to 30-95 range
        final_confidence = max(30, min(95, final_confidence))

        # Calculate expected change
        recent_volatility = close.pct_change().tail(20).std()
        expected_change_pct = weighted_momentum * 100

        # Get current price
        current_price = close.iloc[-1]

        return {
            "direction": direction,
            "confidence": round(final_confidence, 1),
            "predicted_change_pct": round(expected_change_pct, 2),
            "current_price": float(current_price),
            "momentum_signals": {
                "short_term": momentum_signals.get('short', {}),
                "medium_term": momentum_signals.get('medium', {}),
                "long_term": momentum_signals.get('long', {})
            },
            "momentum_strength": {
                "overall": round(momentum_strength, 1),
                "rsi": round(rsi_score, 1),
                "ma_crossover": round(ma_score, 1),
                "volume": round(volume_score, 1),
                "trend_consistency": round(consistency_score, 1)
            },
            "market_regime": regime_info,
            "weighted_momentum": round(weighted_momentum * 100, 2),
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "method": "enhanced_momentum",
            "status": "success"
        }

    except Exception as e:
        logger.error(f"Error in momentum prediction: {e}", exc_info=True)
        return {
            "error": str(e),
            "direction": "Neutral",
            "confidence": 50,
            "status": "failed"
        }
