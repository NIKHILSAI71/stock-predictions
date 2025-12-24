"""
Statistical Arbitrage / Pair Trading Module

Implements pair trading strategies based on:
- Cointegration testing (Engle-Granger)
- Spread calculation and normalization
- Z-score based entry/exit signals
- Mean reversion trading logic

The core idea: If two assets are cointegrated, their spread will
revert to the mean. We buy the underperformer and sell the outperformer
when they diverge, profiting when they converge.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime
import logging

logger = logging.getLogger("uvicorn.info")


def calculate_spread(
    price_a: pd.Series,
    price_b: pd.Series,
    hedge_ratio: float = None
) -> Tuple[pd.Series, float]:
    """
    Calculate the spread between two price series.
    
    Spread = Price_A - (hedge_ratio * Price_B)
    
    Args:
        price_a: First asset prices
        price_b: Second asset prices  
        hedge_ratio: If None, calculated via OLS regression
        
    Returns:
        Tuple of (spread series, hedge_ratio used)
    """
    # Align series
    aligned = pd.concat([price_a, price_b], axis=1).dropna()
    a = aligned.iloc[:, 0].values
    b = aligned.iloc[:, 1].values
    
    if hedge_ratio is None:
        # Calculate hedge ratio via OLS: A = β * B + ε
        # β = Cov(A, B) / Var(B)
        hedge_ratio = np.cov(a, b)[0, 1] / np.var(b)
    
    spread = a - hedge_ratio * b
    spread_series = pd.Series(spread, index=aligned.index)
    
    return spread_series, hedge_ratio


def test_cointegration(
    price_a: pd.Series,
    price_b: pd.Series,
    significance_level: float = 0.05
) -> Dict[str, Any]:
    """
    Test for cointegration using Engle-Granger method.
    
    Two series are cointegrated if:
    1. Both are non-stationary (have unit root)
    2. Their linear combination (spread) is stationary
    
    Args:
        price_a: First asset prices
        price_b: Second asset prices
        significance_level: p-value threshold (default 0.05)
        
    Returns:
        Dictionary with test results and interpretation
    """
    # Calculate spread
    spread, hedge_ratio = calculate_spread(price_a, price_b)
    
    # ADF test on spread (simplified implementation)
    # H0: Unit root exists (non-stationary)
    # We reject H0 if test statistic < critical value
    
    # Calculate ADF test statistic approximation
    n = len(spread)
    spread_array = spread.values
    
    # First difference
    diff = np.diff(spread_array)
    lagged = spread_array[:-1]
    
    # Regression: Δspread_t = α + β * spread_{t-1} + ε
    # ADF statistic = β / SE(β)
    
    # Simple OLS to get coefficient
    X = np.column_stack([np.ones(len(lagged)), lagged])
    y = diff
    
    try:
        # Solve normal equations: (X'X)^-1 X'y
        XtX_inv = np.linalg.inv(X.T @ X)
        beta = XtX_inv @ X.T @ y
        
        # Residuals and SE
        residuals = y - X @ beta
        mse = np.sum(residuals**2) / (n - 2)
        se = np.sqrt(mse * XtX_inv[1, 1])
        
        adf_stat = beta[1] / se if se > 0 else 0
        
        # Critical values for Engle-Granger (approximation)
        # At 5%: -3.37 (for n=100), becomes more negative for smaller samples
        critical_value_5pct = -3.37 - 6.5 / n
        critical_value_1pct = -3.96 - 10.0 / n
        
        is_cointegrated = adf_stat < critical_value_5pct
        
        # Half-life of mean reversion
        # λ = log(2) / |β|
        half_life = np.log(2) / abs(beta[1]) if beta[1] != 0 else np.inf
        half_life = min(half_life, 252)  # Cap at 1 year
        
    except np.linalg.LinAlgError:
        adf_stat = 0
        critical_value_5pct = -3.37
        critical_value_1pct = -3.96
        is_cointegrated = False
        half_life = np.inf
    
    # Correlation check
    correlation = price_a.corr(price_b)
    
    return {
        "is_cointegrated": is_cointegrated,
        "adf_statistic": round(adf_stat, 3),
        "critical_value_5pct": round(critical_value_5pct, 3),
        "critical_value_1pct": round(critical_value_1pct, 3),
        "hedge_ratio": round(hedge_ratio, 4),
        "correlation": round(correlation, 3),
        "half_life_days": round(half_life, 1),
        "interpretation": _interpret_cointegration(is_cointegrated, correlation, half_life)
    }


def _interpret_cointegration(is_cointegrated: bool, correlation: float, half_life: float) -> str:
    """Generate human-readable interpretation."""
    if not is_cointegrated:
        if correlation > 0.7:
            return "High correlation but not cointegrated - risky for pair trading"
        return "Not suitable for pair trading - spread does not mean-revert"
    
    if half_life < 5:
        speed = "very fast"
    elif half_life < 15:
        speed = "fast"
    elif half_life < 30:
        speed = "moderate"
    else:
        speed = "slow"
    
    return f"Cointegrated pair suitable for trading. Mean reversion is {speed} (~{half_life:.0f} days)"


def calculate_zscore(
    spread: pd.Series,
    lookback: int = 20
) -> pd.Series:
    """
    Calculate rolling Z-score of the spread.
    
    Z-score = (spread - mean) / std
    
    Args:
        spread: Spread between two assets
        lookback: Rolling window for mean/std
        
    Returns:
        Z-score series
    """
    rolling_mean = spread.rolling(window=lookback).mean()
    rolling_std = spread.rolling(window=lookback).std()
    
    zscore = (spread - rolling_mean) / rolling_std
    
    # Replace inf/nan with 0
    zscore = zscore.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    return zscore


def generate_pair_signals(
    price_a: pd.Series,
    price_b: pd.Series,
    entry_zscore: float = 2.0,
    exit_zscore: float = 0.5,
    stop_zscore: float = 3.0,
    lookback: int = 20
) -> Dict[str, Any]:
    """
    Generate pair trading signals based on Z-score.
    
    Strategy:
    - LONG spread (buy A, sell B) when Z < -entry_zscore
    - SHORT spread (sell A, buy B) when Z > entry_zscore
    - EXIT when Z crosses exit_zscore toward 0
    - STOP LOSS when Z exceeds stop_zscore
    
    Args:
        price_a: First asset prices (the "long leg" when spread is low)
        price_b: Second asset prices (the "short leg" when spread is low)
        entry_zscore: Z-score threshold to enter trades
        exit_zscore: Z-score threshold to exit trades
        stop_zscore: Z-score threshold for stop loss
        lookback: Lookback period for Z-score calculation
        
    Returns:
        Dictionary with signals and analysis
    """
    # Calculate spread and Z-score
    spread, hedge_ratio = calculate_spread(price_a, price_b)
    zscore = calculate_zscore(spread, lookback)
    
    # Current values
    current_zscore = zscore.iloc[-1]
    current_spread = spread.iloc[-1]
    
    # Determine signal
    if current_zscore <= -entry_zscore:
        signal = "LONG_SPREAD"
        action_a = "BUY"
        action_b = "SELL"
        reasoning = f"Spread is {abs(current_zscore):.2f} std below mean - expecting reversion up"
    elif current_zscore >= entry_zscore:
        signal = "SHORT_SPREAD"
        action_a = "SELL"
        action_b = "BUY"
        reasoning = f"Spread is {current_zscore:.2f} std above mean - expecting reversion down"
    elif abs(current_zscore) <= exit_zscore:
        signal = "EXIT"
        action_a = "CLOSE"
        action_b = "CLOSE"
        reasoning = "Spread has reverted to mean - take profit"
    else:
        signal = "HOLD"
        action_a = "WAIT"
        action_b = "WAIT"
        reasoning = f"Z-score ({current_zscore:.2f}) in neutral zone"
    
    # Check for stop loss
    if abs(current_zscore) >= stop_zscore:
        signal = "STOP_LOSS"
        action_a = "CLOSE"
        action_b = "CLOSE"
        reasoning = f"Z-score ({current_zscore:.2f}) exceeded stop threshold - cut losses"
    
    # Calculate confidence based on Z-score extremity and mean reversion history
    zscore_magnitude = min(abs(current_zscore), 5) / 5  # Normalize to [0, 1]
    
    # Check historical mean reversion success rate
    historical_reversions = _count_reversions(zscore, entry_zscore)
    reversion_rate = historical_reversions.get("success_rate", 0.5)
    
    confidence = (zscore_magnitude * 0.4 + reversion_rate * 0.6) * 100
    confidence = min(max(confidence, 20), 95)  # Clamp to reasonable range
    
    return {
        "signal": signal,
        "current_zscore": round(current_zscore, 3),
        "current_spread": round(current_spread, 4),
        "hedge_ratio": round(hedge_ratio, 4),
        "actions": {
            "asset_a": action_a,
            "asset_b": action_b,
            "hedge_ratio": round(hedge_ratio, 4)
        },
        "confidence": round(confidence, 1),
        "reasoning": reasoning,
        "thresholds": {
            "entry": entry_zscore,
            "exit": exit_zscore,
            "stop": stop_zscore
        },
        "historical_stats": historical_reversions,
        "timestamp": datetime.now().isoformat()
    }


def _count_reversions(zscore: pd.Series, threshold: float) -> Dict[str, float]:
    """Count how often extreme Z-scores reverted to mean historically."""
    if len(zscore) < 60:
        return {"success_rate": 0.5, "sample_size": 0}
    
    z = zscore.values
    total_extremes = 0
    successful_reversions = 0
    
    for i in range(20, len(z) - 20):
        if abs(z[i]) >= threshold:
            total_extremes += 1
            # Check if reverted within next 20 periods
            future_z = z[i+1:i+21]
            if len(future_z) > 0:
                # Success if Z-score crosses zero or reaches opposite side
                if np.sign(z[i]) != np.sign(future_z).mean() or abs(future_z).min() < threshold/2:
                    successful_reversions += 1
    
    success_rate = successful_reversions / total_extremes if total_extremes > 0 else 0.5
    
    return {
        "success_rate": round(success_rate, 3),
        "sample_size": total_extremes,
        "successful_reversions": successful_reversions
    }


def find_cointegrated_pairs(
    price_data: Dict[str, pd.Series],
    min_correlation: float = 0.7,
    max_pairs: int = 10
) -> List[Dict[str, Any]]:
    """
    Find cointegrated pairs from a universe of assets.
    
    Args:
        price_data: Dictionary mapping symbol to price series
        min_correlation: Minimum correlation to consider
        max_pairs: Maximum number of pairs to return
        
    Returns:
        List of cointegrated pair candidates sorted by quality
    """
    symbols = list(price_data.keys())
    pairs = []
    
    for i, sym_a in enumerate(symbols):
        for sym_b in symbols[i+1:]:
            price_a = price_data[sym_a]
            price_b = price_data[sym_b]
            
            # Quick correlation check first
            correlation = price_a.corr(price_b)
            if abs(correlation) < min_correlation:
                continue
            
            # Full cointegration test
            result = test_cointegration(price_a, price_b)
            
            if result["is_cointegrated"]:
                pairs.append({
                    "symbol_a": sym_a,
                    "symbol_b": sym_b,
                    "correlation": result["correlation"],
                    "hedge_ratio": result["hedge_ratio"],
                    "half_life": result["half_life_days"],
                    "adf_statistic": result["adf_statistic"],
                    "quality_score": _calculate_pair_quality(result)
                })
    
    # Sort by quality score
    pairs.sort(key=lambda x: x["quality_score"], reverse=True)
    
    return pairs[:max_pairs]


def _calculate_pair_quality(coint_result: Dict[str, Any]) -> float:
    """
    Calculate quality score for a cointegrated pair.
    
    Higher score = better pair for trading.
    """
    score = 0
    
    # Strong cointegration (more negative ADF stat is better)
    adf = coint_result["adf_statistic"]
    if adf < -4:
        score += 30
    elif adf < -3.5:
        score += 20
    else:
        score += 10
    
    # High correlation is good
    corr = abs(coint_result["correlation"])
    score += corr * 30
    
    # Moderate half-life is ideal (5-30 days)
    hl = coint_result["half_life_days"]
    if 5 <= hl <= 30:
        score += 30
    elif hl < 5:
        score += 20  # Too fast might be noise
    else:
        score += max(0, 30 - (hl - 30) / 2)  # Penalize slow reversion
    
    return round(score, 2)


def get_pair_trading_analysis(
    symbol_a: str,
    symbol_b: str,
    price_a: pd.Series,
    price_b: pd.Series
) -> Dict[str, Any]:
    """
    Complete pair trading analysis for two assets.
    
    Args:
        symbol_a: First asset symbol
        symbol_b: Second asset symbol
        price_a: First asset prices
        price_b: Second asset prices
        
    Returns:
        Comprehensive pair trading analysis
    """
    # Cointegration test
    coint_result = test_cointegration(price_a, price_b)
    
    # Generate signals
    signals = generate_pair_signals(price_a, price_b)
    
    # Calculate spread statistics
    spread, hedge_ratio = calculate_spread(price_a, price_b)
    zscore = calculate_zscore(spread)
    
    # Recent performance
    recent_spread = spread.tail(20)
    
    result = {
        "pair": f"{symbol_a}/{symbol_b}",
        "cointegration": coint_result,
        "current_signal": signals,
        "spread_stats": {
            "current": round(spread.iloc[-1], 4),
            "mean_20d": round(recent_spread.mean(), 4),
            "std_20d": round(recent_spread.std(), 4),
            "min_20d": round(recent_spread.min(), 4),
            "max_20d": round(recent_spread.max(), 4)
        },
        "zscore_history": {
            "current": round(zscore.iloc[-1], 3),
            "max_1m": round(zscore.tail(20).max(), 3),
            "min_1m": round(zscore.tail(20).min(), 3)
        },
        "tradeable": coint_result["is_cointegrated"] and 5 <= coint_result["half_life_days"] <= 60,
        "quality_score": _calculate_pair_quality(coint_result) if coint_result["is_cointegrated"] else 0,
        "timestamp": datetime.now().isoformat()
    }
    
    # Add recommendation
    if result["tradeable"]:
        if signals["signal"] in ["LONG_SPREAD", "SHORT_SPREAD"]:
            result["recommendation"] = f"Active opportunity: {signals['reasoning']}"
        else:
            result["recommendation"] = "Pair is tradeable. Monitor for entry signals."
    else:
        result["recommendation"] = "Not recommended for pair trading. " + coint_result["interpretation"]
    
    return result
