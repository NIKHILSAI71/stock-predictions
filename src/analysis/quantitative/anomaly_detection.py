"""
Anomaly Detection Module

Detects unusual market activity that may precede significant price moves:
- Volume spikes
- Price gaps
- Volatility clusters
- Momentum divergences
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from datetime import datetime


def detect_volume_spikes(
    volume: pd.Series,
    lookback: int = 20,
    spike_threshold: float = 2.0
) -> Dict[str, Any]:
    """
    Detect unusually high volume days.
    
    Args:
        volume: Series of daily volumes
        lookback: Days for average calculation
        spike_threshold: Multiple of average to consider a spike
    
    Returns:
        Spike detection results
    """
    if len(volume) < lookback + 1:
        return {"detected": False, "reason": "Insufficient data"}
    
    avg_volume = volume.rolling(lookback).mean()
    current_vol = volume.iloc[-1]
    avg_vol = avg_volume.iloc[-1]
    
    if avg_vol == 0:
        return {"detected": False, "reason": "Zero average volume"}
    
    ratio = current_vol / avg_vol
    
    # Check for spike
    is_spike = ratio >= spike_threshold
    
    # Calculate historical spikes
    volume_ratios = volume / avg_volume
    recent_spikes = (volume_ratios.iloc[-5:] >= spike_threshold).sum()
    
    return {
        "detected": is_spike,
        "current_volume": int(current_vol),
        "average_volume": int(avg_vol),
        "volume_ratio": round(ratio, 2),
        "threshold": spike_threshold,
        "severity": "HIGH" if ratio >= 3.0 else ("MEDIUM" if ratio >= 2.0 else "LOW"),
        "recent_spike_count": int(recent_spikes),
        "interpretation": _interpret_volume_spike(ratio, is_spike)
    }


def _interpret_volume_spike(ratio: float, is_spike: bool) -> str:
    """Interpret volume spike significance."""
    if not is_spike:
        return "Normal trading volume"
    elif ratio >= 5.0:
        return "EXTREME volume - major event or institutional activity likely"
    elif ratio >= 3.0:
        return "HIGH volume - significant interest from large players"
    elif ratio >= 2.0:
        return "Elevated volume - increased market attention"
    return "Normal volume range"


def detect_price_gaps(
    data: pd.DataFrame,
    gap_threshold_pct: float = 3.0
) -> Dict[str, Any]:
    """
    Detect gap up/down patterns.
    
    Args:
        data: OHLC DataFrame
        gap_threshold_pct: Minimum gap percentage to flag
    
    Returns:
        Gap detection results
    """
    required_cols = ['Open', 'Close', 'High', 'Low']
    if not all(col in data.columns for col in required_cols):
        return {"detected": False, "reason": "Missing OHLC columns"}
    
    if len(data) < 2:
        return {"detected": False, "reason": "Insufficient data"}
    
    # Calculate gaps
    prev_close = data['Close'].shift(1)
    current_open = data['Open']
    
    gap_pct = ((current_open - prev_close) / prev_close * 100).fillna(0)
    
    # Current gap
    current_gap = gap_pct.iloc[-1]
    
    # Detect gap
    is_gap = abs(current_gap) >= gap_threshold_pct
    gap_direction = "UP" if current_gap > 0 else "DOWN"
    
    # Historical gaps
    historical_gaps = gap_pct[abs(gap_pct) >= gap_threshold_pct]
    
    # Gap fill analysis - was the gap filled during the day?
    if is_gap:
        if current_gap > 0:
            # Gap up - check if price went below previous close
            gap_filled = data['Low'].iloc[-1] <= prev_close.iloc[-1]
        else:
            # Gap down - check if price went above previous close
            gap_filled = data['High'].iloc[-1] >= prev_close.iloc[-1]
    else:
        gap_filled = None
    
    return {
        "detected": is_gap,
        "direction": gap_direction if is_gap else None,
        "gap_pct": round(current_gap, 2),
        "threshold": gap_threshold_pct,
        "gap_filled": gap_filled,
        "severity": _classify_gap_severity(abs(current_gap)),
        "recent_gaps": len(historical_gaps.tail(20)),
        "interpretation": _interpret_gap(current_gap, gap_threshold_pct, gap_filled)
    }


def _classify_gap_severity(gap_pct: float) -> str:
    """Classify gap severity."""
    if gap_pct >= 10:
        return "EXTREME"
    elif gap_pct >= 5:
        return "HIGH"
    elif gap_pct >= 3:
        return "MEDIUM"
    return "LOW"


def _interpret_gap(gap_pct: float, threshold: float, filled: Optional[bool]) -> str:
    """Interpret gap significance."""
    if abs(gap_pct) < threshold:
        return "No significant gap detected"
    
    direction = "Gap UP" if gap_pct > 0 else "Gap DOWN"
    
    if filled:
        return f"{direction} of {abs(gap_pct):.1f}% - GAP FILLED (potential reversal signal)"
    elif filled is False:
        return f"{direction} of {abs(gap_pct):.1f}% - Gap held (momentum continuation likely)"
    else:
        return f"{direction} of {abs(gap_pct):.1f}%"


def detect_volatility_cluster(
    data: pd.DataFrame,
    short_window: int = 5,
    long_window: int = 20,
    cluster_threshold: float = 1.5
) -> Dict[str, Any]:
    """
    Detect volatility clustering (GARCH-like behavior).
    
    Args:
        data: DataFrame with Close prices
        short_window: Short-term volatility window
        long_window: Long-term volatility window
        cluster_threshold: Ratio threshold for cluster detection
    
    Returns:
        Volatility cluster detection results
    """
    if 'Close' not in data.columns:
        return {"detected": False, "reason": "No Close column"}
    
    if len(data) < long_window + 5:
        return {"detected": False, "reason": "Insufficient data"}
    
    # Calculate returns
    returns = data['Close'].pct_change().dropna()
    
    # Calculate rolling volatility
    short_vol = returns.rolling(short_window).std() * np.sqrt(252) * 100
    long_vol = returns.rolling(long_window).std() * np.sqrt(252) * 100
    
    current_short_vol = short_vol.iloc[-1]
    current_long_vol = long_vol.iloc[-1]
    
    if current_long_vol == 0:
        return {"detected": False, "reason": "Zero long-term volatility"}
    
    vol_ratio = current_short_vol / current_long_vol
    
    # Detect cluster
    is_cluster = vol_ratio >= cluster_threshold
    
    # Trend in volatility
    vol_trend = "INCREASING" if short_vol.iloc[-1] > short_vol.iloc[-5] else "DECREASING"
    
    # Historical volatility percentile
    vol_percentile = (returns.rolling(long_window).std().iloc[-1] <= 
                      returns.rolling(long_window).std()).mean() * 100
    
    return {
        "detected": is_cluster,
        "short_term_vol": round(current_short_vol, 2),
        "long_term_vol": round(current_long_vol, 2),
        "vol_ratio": round(vol_ratio, 2),
        "threshold": cluster_threshold,
        "volatility_trend": vol_trend,
        "vol_percentile": round(vol_percentile, 1),
        "regime": _classify_vol_regime(current_short_vol, vol_ratio),
        "interpretation": _interpret_volatility(vol_ratio, is_cluster, vol_trend)
    }


def _classify_vol_regime(vol: float, ratio: float) -> str:
    """Classify volatility regime."""
    if vol > 50:
        return "EXTREME"
    elif vol > 30:
        return "HIGH"
    elif vol > 15:
        return "NORMAL"
    else:
        return "LOW"


def _interpret_volatility(ratio: float, is_cluster: bool, trend: str) -> str:
    """Interpret volatility pattern."""
    if not is_cluster:
        return f"Volatility within normal range ({trend.lower()} trend)"
    
    if ratio >= 2.0:
        return f"HIGH volatility cluster - expect large price swings. Trend: {trend}"
    else:
        return f"Elevated volatility detected. Trend: {trend}"


def detect_momentum_divergence(
    data: pd.DataFrame,
    rsi_period: int = 14
) -> Dict[str, Any]:
    """
    Detect price-RSI divergences.
    
    Args:
        data: DataFrame with Close prices
        rsi_period: RSI calculation period
    
    Returns:
        Divergence detection results
    """
    if 'Close' not in data.columns or len(data) < rsi_period + 10:
        return {"detected": False, "reason": "Insufficient data"}
    
    prices = data['Close']
    
    # Calculate RSI
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(rsi_period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(rsi_period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    # Look for divergence in last 10 days
    price_recent = prices.iloc[-10:]
    rsi_recent = rsi.iloc[-10:]
    
    # Price making higher high but RSI making lower high = bearish divergence
    price_trend = price_recent.iloc[-1] > price_recent.iloc[0]
    rsi_trend = rsi_recent.iloc[-1] > rsi_recent.iloc[0]
    
    if price_trend and not rsi_trend:
        divergence_type = "BEARISH"
        detected = True
    elif not price_trend and rsi_trend:
        divergence_type = "BULLISH"
        detected = True
    else:
        divergence_type = None
        detected = False
    
    current_rsi = rsi.iloc[-1]
    
    return {
        "detected": detected,
        "type": divergence_type,
        "current_rsi": round(current_rsi, 1),
        "rsi_overbought": current_rsi > 70,
        "rsi_oversold": current_rsi < 30,
        "interpretation": _interpret_divergence(divergence_type, current_rsi)
    }


def _interpret_divergence(div_type: Optional[str], rsi: float) -> str:
    """Interpret momentum divergence."""
    if not div_type:
        if rsi > 70:
            return "No divergence, but RSI overbought - caution advised"
        elif rsi < 30:
            return "No divergence, but RSI oversold - potential bounce"
        return "No momentum divergence detected"
    
    if div_type == "BEARISH":
        return "BEARISH DIVERGENCE: Price up but momentum weakening - potential reversal"
    else:
        return "BULLISH DIVERGENCE: Price down but momentum strengthening - potential bottom"


def get_anomaly_alerts(stock_data: pd.DataFrame) -> Dict[str, Any]:
    """
    Run all anomaly detection checks and return aggregated alerts.
    
    Args:
        stock_data: DataFrame with OHLCV data
    
    Returns:
        Aggregated anomaly alerts with severity rankings
    """
    alerts = []
    
    # Volume spike
    if 'Volume' in stock_data.columns:
        volume_result = detect_volume_spikes(stock_data['Volume'])
        if volume_result.get("detected"):
            alerts.append({
                "type": "VOLUME_SPIKE",
                "severity": volume_result["severity"],
                "message": volume_result["interpretation"],
                "data": {
                    "ratio": volume_result["volume_ratio"],
                    "current": volume_result["current_volume"]
                }
            })
    
    # Price gaps
    gap_result = detect_price_gaps(stock_data)
    if gap_result.get("detected"):
        alerts.append({
            "type": "PRICE_GAP",
            "severity": gap_result["severity"],
            "message": gap_result["interpretation"],
            "data": {
                "direction": gap_result["direction"],
                "gap_pct": gap_result["gap_pct"],
                "filled": gap_result["gap_filled"]
            }
        })
    
    # Volatility cluster
    vol_result = detect_volatility_cluster(stock_data)
    if vol_result.get("detected"):
        alerts.append({
            "type": "VOLATILITY_CLUSTER",
            "severity": "HIGH" if vol_result["vol_ratio"] >= 2.0 else "MEDIUM",
            "message": vol_result["interpretation"],
            "data": {
                "short_vol": vol_result["short_term_vol"],
                "long_vol": vol_result["long_term_vol"],
                "trend": vol_result["volatility_trend"]
            }
        })
    
    # Momentum divergence
    div_result = detect_momentum_divergence(stock_data)
    if div_result.get("detected"):
        alerts.append({
            "type": "MOMENTUM_DIVERGENCE",
            "severity": "HIGH" if div_result["type"] else "MEDIUM",
            "message": div_result["interpretation"],
            "data": {
                "type": div_result["type"],
                "rsi": div_result["current_rsi"]
            }
        })
    
    # Sort by severity
    severity_order = {"EXTREME": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
    alerts.sort(key=lambda x: severity_order.get(x["severity"], 4))
    
    return {
        "total_alerts": len(alerts),
        "alerts": alerts,
        "has_critical": any(a["severity"] in ["EXTREME", "HIGH"] for a in alerts),
        "summary": _generate_alert_summary(alerts),
        "timestamp": datetime.now().isoformat()
    }


def _generate_alert_summary(alerts: List[Dict]) -> str:
    """Generate human-readable alert summary."""
    if not alerts:
        return "No anomalies detected - normal market conditions"
    
    critical = [a for a in alerts if a["severity"] in ["EXTREME", "HIGH"]]
    
    if critical:
        types = [a["type"].replace("_", " ").title() for a in critical]
        return f"ALERT: {len(critical)} high-severity anomalies detected: {', '.join(types)}"
    else:
        return f"{len(alerts)} minor anomalies detected - heightened attention advised"
