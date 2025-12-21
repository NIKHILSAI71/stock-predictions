"""
Trading Signals Module
Consolidated signal generation with valuation filters, RSI protection,
MACD divergence detection, and risk-adjusted recommendations.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List

# Default industry averages for comparison
INDUSTRY_PE_AVERAGE = 17.8  # S&P 500 average
OVERVALUATION_THRESHOLD = 1.15  # 15% above fair value = overvalued
UNDERVALUATION_THRESHOLD = 0.85  # 15% below fair value = undervalued


def calculate_volume_signal(volume_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze volume to confirm price action.
    
    Args:
        volume_data: Dictionary with volume metrics (current_volume, avg_volume_20, volume_ratio, etc.)
    
    Returns:
        Dictionary with volume signal and conviction assessment
    """
    result = {
        "volume_status": "NEUTRAL",
        "volume_factor": 1.0,
        "conviction": "NORMAL",
        "warning": None
    }
    
    if not volume_data:
        return result
    
    volume_ratio = volume_data.get("volume_ratio", 1.0)
    current_volume = volume_data.get("current_volume", 0)
    avg_volume = volume_data.get("avg_volume_20", 0)
    
    if volume_ratio >= 2.0:
        result["volume_status"] = "VERY_HIGH"
        result["volume_factor"] = 1.15  # Boost signal
        result["conviction"] = "STRONG"
    elif volume_ratio >= 1.5:
        result["volume_status"] = "HIGH"
        result["volume_factor"] = 1.1
        result["conviction"] = "CONFIRMED"
    elif volume_ratio >= 0.8:
        result["volume_status"] = "NORMAL"
        result["volume_factor"] = 1.0
        result["conviction"] = "NORMAL"
    elif volume_ratio >= 0.5:
        result["volume_status"] = "LOW"
        result["volume_factor"] = 0.9
        result["conviction"] = "WEAK"
        result["warning"] = f"Below-average volume ({volume_ratio:.2f}x avg) - weaker conviction"
    else:
        result["volume_status"] = "VERY_LOW"
        result["volume_factor"] = 0.8
        result["conviction"] = "VERY_WEAK"
        result["warning"] = f"Very low volume ({volume_ratio:.2f}x avg) - price move lacks conviction"
    
    # Cap factor between 0.8 and 1.15
    result["volume_factor"] = max(0.8, min(1.15, result["volume_factor"]))
    
    return result


def calculate_peg_signal(peg_ratio: Optional[float]) -> Dict[str, Any]:
    """
    Analyze PEG ratio for growth-adjusted valuation.
    
    Args:
        peg_ratio: Price/Earnings to Growth ratio
    
    Returns:
        Dictionary with PEG analysis and adjustments
    """
    result = {
        "peg_status": "N/A",
        "peg_factor": 1.0,
        "warning": None
    }
    
    if peg_ratio is None or peg_ratio <= 0:
        return result
    
    if peg_ratio > 3.0:
        result["peg_status"] = "VERY_OVERVALUED"
        result["peg_factor"] = 0.6
        result["warning"] = f"PEG ratio of {peg_ratio:.2f} indicates significantly overvalued growth"
    elif peg_ratio > 2.0:
        result["peg_status"] = "OVERVALUED"
        result["peg_factor"] = 0.8
        result["warning"] = f"PEG ratio of {peg_ratio:.2f} suggests overvalued for growth rate"
    elif peg_ratio > 1.5:
        result["peg_status"] = "FAIRLY_VALUED"
        result["peg_factor"] = 0.95
    elif peg_ratio >= 1.0:
        result["peg_status"] = "REASONABLE"
        result["peg_factor"] = 1.0
    else:
        result["peg_status"] = "UNDERVALUED"
        result["peg_factor"] = 1.1
    
    return result


def calculate_valuation_signal(
    current_price: float,
    fair_value: Optional[float],
    pe_ratio: Optional[float],
    industry_pe: float = INDUSTRY_PE_AVERAGE,
    forward_pe: Optional[float] = None
) -> Dict[str, Any]:
    """
    Calculate valuation-based signal adjustments.
    
    Args:
        current_price: Current stock price
        fair_value: Estimated fair value (from DCF, DDM, etc.)
        pe_ratio: Trailing P/E ratio
        industry_pe: Industry average P/E (default S&P 500 avg)
        forward_pe: Forward P/E ratio (optional)
    
    Returns:
        Dictionary with valuation signal and adjustments
    """
    result = {
        "valuation_status": "NEUTRAL",
        "valuation_factor": 1.0,  # Multiplier for signal strength (0-1)
        "pe_vs_industry": None,
        "price_vs_fair_value_pct": None,
        "warnings": []
    }
    
    # Check PE vs industry average
    if pe_ratio and pe_ratio > 0:
        pe_premium = pe_ratio / industry_pe
        result["pe_vs_industry"] = round(pe_premium, 2)
        
        if pe_ratio > industry_pe * 1.5:  # 50% above industry
            result["warnings"].append(f"P/E ({pe_ratio:.1f}) is significantly above industry average ({industry_pe:.1f})")
            result["valuation_factor"] *= 0.6  # Reduce signal by 40%
        elif pe_ratio > industry_pe * 1.2:  # 20% above industry
            result["warnings"].append(f"P/E ({pe_ratio:.1f}) is above industry average ({industry_pe:.1f})")
            result["valuation_factor"] *= 0.8  # Reduce signal by 20%
        elif pe_ratio < industry_pe * 0.7:  # 30% below industry
            result["valuation_factor"] *= 1.1  # Boost signal by 10% (capped at 1.0 later)
    
    # Check price vs fair value
    if fair_value and fair_value > 0 and current_price:
        price_to_fair = current_price / fair_value
        deviation_pct = (current_price - fair_value) / fair_value * 100
        result["price_vs_fair_value_pct"] = round(deviation_pct, 2)
        
        if price_to_fair > OVERVALUATION_THRESHOLD:
            result["valuation_status"] = "OVERVALUED"
            result["warnings"].append(f"Stock trading {deviation_pct:.1f}% above fair value (${fair_value:.2f})")
            result["valuation_factor"] *= 0.5  # Significant reduction
        elif price_to_fair > 1.05:  # 5-15% overvalued
            result["valuation_status"] = "SLIGHTLY_OVERVALUED"
            result["valuation_factor"] *= 0.75
        elif price_to_fair < UNDERVALUATION_THRESHOLD:
            result["valuation_status"] = "UNDERVALUED"
            result["valuation_factor"] *= 1.2  # Boost (capped later)
        else:
            result["valuation_status"] = "FAIRLY_VALUED"
    
    # Cap factor between 0.3 and 1.0
    result["valuation_factor"] = max(0.3, min(1.0, result["valuation_factor"]))
    
    return result


def calculate_rsi_signal(rsi_value: float) -> Dict[str, Any]:
    """
    Calculate RSI-based signal with overbought protection.
    
    Args:
        rsi_value: Current RSI value (0-100)
    
    Returns:
        Dictionary with RSI signal and adjustments
    """
    result = {
        "rsi_status": "NEUTRAL",
        "rsi_factor": 1.0,
        "action": "NORMAL",
        "warning": None
    }
    
    if rsi_value >= 80:
        result["rsi_status"] = "EXTREMELY_OVERBOUGHT"
        result["rsi_factor"] = 0.3
        result["action"] = "AVOID_ENTRY"
        result["warning"] = f"RSI at {rsi_value:.1f} - extremely overbought, high reversal risk"
    elif rsi_value >= 70:
        result["rsi_status"] = "OVERBOUGHT"
        result["rsi_factor"] = 0.5
        result["action"] = "REDUCE_POSITION_SIZE"
        result["warning"] = f"RSI at {rsi_value:.1f} - overbought, consider smaller position"
    elif rsi_value >= 65:
        result["rsi_status"] = "APPROACHING_OVERBOUGHT"
        result["rsi_factor"] = 0.75
        result["action"] = "CAUTION"
        result["warning"] = f"RSI at {rsi_value:.1f} - approaching overbought territory"
    elif rsi_value <= 20:
        result["rsi_status"] = "EXTREMELY_OVERSOLD"
        result["rsi_factor"] = 1.2  # Boost for potential bounce
        result["action"] = "POTENTIAL_ENTRY"
    elif rsi_value <= 30:
        result["rsi_status"] = "OVERSOLD"
        result["rsi_factor"] = 1.1
        result["action"] = "POTENTIAL_ENTRY"
    
    # Cap factor
    result["rsi_factor"] = max(0.3, min(1.2, result["rsi_factor"]))
    
    return result


def detect_macd_divergence(
    prices: pd.Series,
    macd_histogram: pd.Series,
    lookback: int = 14
) -> Dict[str, Any]:
    """
    Detect MACD divergence (price making new highs/lows while MACD doesn't).
    
    Args:
        prices: Price series (Close prices)
        macd_histogram: MACD histogram series
        lookback: Number of periods to look back for divergence
    
    Returns:
        Dictionary with divergence type and details
    """
    result = {
        "divergence_type": "NONE",
        "divergence_strength": 0,
        "macd_penalty": 0.0,
        "warning": None
    }
    
    if len(prices) < lookback or len(macd_histogram) < lookback:
        return result
    
    # Get recent data
    recent_prices = prices.iloc[-lookback:]
    recent_macd = macd_histogram.iloc[-lookback:]
    
    # Find recent highs and lows
    price_high_idx = recent_prices.idxmax()
    price_low_idx = recent_prices.idxmin()
    
    current_price = prices.iloc[-1]
    current_macd = macd_histogram.iloc[-1]
    
    # Check for bearish divergence: price making higher highs, MACD making lower highs
    if current_price >= recent_prices.max() * 0.98:  # Near recent high
        # Check if MACD is showing weakness (lower than previous peak)
        macd_at_price_high = macd_histogram.loc[price_high_idx] if price_high_idx in macd_histogram.index else current_macd
        if current_macd < macd_at_price_high * 0.8:  # MACD significantly lower
            result["divergence_type"] = "BEARISH"
            result["divergence_strength"] = min(1.0, abs(macd_at_price_high - current_macd) / abs(macd_at_price_high) if macd_at_price_high != 0 else 0.5)
            result["macd_penalty"] = 0.2 + (result["divergence_strength"] * 0.1)
            result["warning"] = "Bearish MACD divergence detected - price making highs while momentum weakens"
    
    # Check for bullish divergence: price making lower lows, MACD making higher lows
    elif current_price <= recent_prices.min() * 1.02:  # Near recent low
        macd_at_price_low = macd_histogram.loc[price_low_idx] if price_low_idx in macd_histogram.index else current_macd
        if current_macd > macd_at_price_low * 1.2:  # MACD significantly higher
            result["divergence_type"] = "BULLISH"
            result["divergence_strength"] = min(1.0, abs(current_macd - macd_at_price_low) / abs(macd_at_price_low) if macd_at_price_low != 0 else 0.5)
            result["macd_penalty"] = -0.1  # Negative penalty = boost
    
    # Check if histogram is turning negative (momentum shift)
    if len(macd_histogram) >= 3:
        if macd_histogram.iloc[-3] > 0 and macd_histogram.iloc[-2] > 0 and macd_histogram.iloc[-1] < 0:
            if result["divergence_type"] == "NONE":
                result["divergence_type"] = "MOMENTUM_SHIFT_BEARISH"
                result["macd_penalty"] = 0.1
                result["warning"] = "MACD histogram turning negative - momentum shifting bearish"
    
    return result


def calculate_entry_recommendation(
    current_price: float,
    support_levels: List[float],
    resistance_levels: List[float],
    atr: float,
    fifty_two_week_high: Optional[float] = None
) -> Dict[str, Any]:
    """
    Calculate optimal entry point recommendations.
    
    Args:
        current_price: Current stock price
        support_levels: List of support price levels
        resistance_levels: List of resistance price levels
        atr: Average True Range for volatility
        fifty_two_week_high: 52-week high price
    
    Returns:
        Entry recommendation with pullback targets
    """
    result = {
        "immediate_entry_ok": True,
        "wait_for_pullback": False,
        "pullback_target": None,
        "pullback_pct": None,
        "near_resistance": False,
        "near_all_time_high": False,
        "rationale": []
    }
    
    # Check if near 52-week high (within 3%)
    if fifty_two_week_high and current_price >= fifty_two_week_high * 0.97:
        result["near_all_time_high"] = True
        result["wait_for_pullback"] = True
        result["rationale"].append("Stock near 52-week high - elevated reversal risk")
    
    # Check proximity to resistance
    if resistance_levels:
        nearest_resistance = min([r for r in resistance_levels if r > current_price], default=None)
        if nearest_resistance and (nearest_resistance - current_price) / current_price < 0.02:
            result["near_resistance"] = True
            result["wait_for_pullback"] = True
            result["rationale"].append(f"Near resistance at ${nearest_resistance:.2f}")
    
    # Calculate pullback target based on support and ATR
    if support_levels:
        valid_supports = [s for s in support_levels if s < current_price]
        if valid_supports:
            nearest_support = max(valid_supports)
            
            # Suggest entry between current price and nearest support
            pullback_target = current_price - (1.5 * atr)  # 1.5x ATR pullback
            
            # Don't go below support
            pullback_target = max(pullback_target, nearest_support * 1.01)
            
            if result["wait_for_pullback"]:
                result["pullback_target"] = round(pullback_target, 2)
                result["pullback_pct"] = round((current_price - pullback_target) / current_price * 100, 1)
    
    # If waiting for pullback, entry not immediately OK
    if result["wait_for_pullback"]:
        result["immediate_entry_ok"] = False
    
    return result


def calculate_staged_entry(
    current_price: float,
    ideal_entry_price: float,
    regime: Optional[Dict[str, Any]] = None,
    valuation_status: str = "NEUTRAL",
    rsi_status: str = "NEUTRAL",
    intrinsic_value: Optional[float] = None  # DCF-based fair value for overvaluation check
) -> Dict[str, Any]:
    """
    Calculate staged entry recommendation with position sizing.
    
    Instead of binary BUY/WAIT, recommends scaled position sizes:
    - 30% position if price >15% above ideal entry
    - 50% position if price 5-15% above ideal entry  
    - 100% position if price at or below ideal entry
    
    CRITICAL: If intrinsic_value provided and stock is >30% overvalued,
    severely reduces position and requires deeper pullback.
    
    Args:
        current_price: Current stock price
        ideal_entry_price: Calculated ideal entry point (from support/fair value)
        regime: Market regime dictionary with 'current_regime' key
        valuation_status: Valuation assessment (OVERVALUED, FAIR, etc.)
        rsi_status: RSI assessment (OVERBOUGHT, NEUTRAL, etc.)
        intrinsic_value: DCF fair value for overvaluation calculation
    
    Returns:
        Dictionary with staged entry recommendation
    """
    result = {
        "recommended_position_pct": 100,
        "price_vs_ideal_pct": 0.0,
        "ideal_entry_price": round(ideal_entry_price, 2),
        "regime_override_applied": False,
        "base_position_pct": 100,
        "rationale": [],
        "entry_tiers": [],
        "dcf_overvaluation_pct": None,  # NEW: Quantified overvaluation
        "valuation_warning": None,  # NEW: Critical warning for severely overvalued
        "max_position_cap": 100  # NEW: Hard cap from overvaluation
    }
    
    # Guard against invalid ideal entry price
    if ideal_entry_price <= 0:
        ideal_entry_price = current_price * 0.95
        result["ideal_entry_price"] = round(ideal_entry_price, 2)
        result["rationale"].append("Using fallback ideal entry (5% below current)")
    
    # ============= NEW: DCF OVERVALUATION CHECK =============
    # This is CRITICAL - if stock is severely overvalued by DCF, cap position
    overvaluation_cap = 100  # Default: no cap
    
    if intrinsic_value and intrinsic_value > 0:
        dcf_overval_pct = ((current_price - intrinsic_value) / intrinsic_value) * 100
        result["dcf_overvaluation_pct"] = round(dcf_overval_pct, 1)
        
        if dcf_overval_pct > 50:
            # SEVERELY overvalued (>50%) - near-avoid territory
            overvaluation_cap = 15
            result["valuation_warning"] = f"SEVERELY OVERVALUED: {dcf_overval_pct:.0f}% above intrinsic value (${intrinsic_value:.2f}). WAIT for 15%+ pullback."
            result["rationale"].append(f"WARNING: DCF shows {dcf_overval_pct:.0f}% overvaluation - maximum 15% position")
        elif dcf_overval_pct > 30:
            # Significantly overvalued (30-50%)
            overvaluation_cap = 25
            result["valuation_warning"] = f"OVERVALUED: {dcf_overval_pct:.0f}% above intrinsic value (${intrinsic_value:.2f}). Consider deeper pullback."
            result["rationale"].append(f"WARNING: DCF shows {dcf_overval_pct:.0f}% overvaluation - maximum 25% position")
        elif dcf_overval_pct > 15:
            # Moderately overvalued (15-30%)
            overvaluation_cap = 40
            result["rationale"].append(f"DCF shows {dcf_overval_pct:.0f}% overvaluation - reduced position sizing")
        elif dcf_overval_pct < -10:
            # Undervalued by >10% - positive signal
            result["rationale"].append(f"DCF shows {abs(dcf_overval_pct):.0f}% undervaluation - favorable entry")
    
    result["max_position_cap"] = overvaluation_cap
    
    # Calculate price stretch percentage
    price_stretch_pct = ((current_price - ideal_entry_price) / ideal_entry_price) * 100
    result["price_vs_ideal_pct"] = round(price_stretch_pct, 2)
    
    # Determine base position size from stretch percentage
    if price_stretch_pct <= 0:
        # At or below ideal entry - full position
        base_position = 100
        result["rationale"].append(f"Price at/below ideal entry (${ideal_entry_price:.2f}) - full position justified")
    elif price_stretch_pct <= 5:
        # Slightly above ideal - 80% position
        base_position = 80
        result["rationale"].append(f"Price {price_stretch_pct:.1f}% above ideal - near-full position")
    elif price_stretch_pct <= 10:
        # Moderately above ideal - 60% position
        base_position = 60
        result["rationale"].append(f"Price {price_stretch_pct:.1f}% above ideal - moderate position")
    elif price_stretch_pct <= 15:
        # Stretched - 50% position
        base_position = 50
        result["rationale"].append(f"Price {price_stretch_pct:.1f}% above ideal - half position, scale in on dips")
    elif price_stretch_pct <= 25:
        # Very stretched - 30% position
        base_position = 30
        result["rationale"].append(f"Price {price_stretch_pct:.1f}% above ideal - small initial position only")
    else:
        # Extremely stretched - 20% position
        base_position = 20
        result["rationale"].append(f"Price {price_stretch_pct:.1f}% above ideal - minimal position, wait for pullback")
    
    result["base_position_pct"] = base_position
    
    # Apply regime override
    current_regime = (regime or {}).get("current_regime", "Unknown")
    
    # Bull Low Volatility regime = can justify higher positions at stretched prices
    if "Bull" in current_regime and "Low" in current_regime:
        if base_position < 50:
            # In strong bull regime, boost minimum to 40%
            regime_adjusted = max(base_position + 20, 40)
            result["regime_override_applied"] = True
            result["rationale"].append(
                f"Bull Low Volatility regime justifies higher entry: {base_position}% → {regime_adjusted}%"
            )
            base_position = regime_adjusted
        elif base_position < 100:
            # Moderate boost
            regime_adjusted = min(base_position + 15, 100)
            if regime_adjusted > base_position:
                result["regime_override_applied"] = True
                result["rationale"].append(
                    f"Favorable regime allows increased position: {base_position}% → {regime_adjusted}%"
                )
                base_position = regime_adjusted
    
    # Bear regime = reduce positions
    elif "Bear" in current_regime:
        regime_adjusted = max(base_position - 20, 20)
        if regime_adjusted < base_position:
            result["regime_override_applied"] = True
            result["rationale"].append(
                f"Bear regime requires caution: {base_position}% → {regime_adjusted}%"
            )
            base_position = regime_adjusted
    
    # Apply additional adjustments for stretched technicals
    if "OVERBOUGHT" in rsi_status.upper() or "EXTREMELY" in rsi_status.upper():
        adjusted = max(base_position - 15, 20)
        if adjusted < base_position:
            result["rationale"].append(f"RSI overbought reduces position: {base_position}% → {adjusted}%")
            base_position = adjusted
    
    if valuation_status in ["OVERVALUED", "EXPENSIVE"]:
        adjusted = max(base_position - 10, 20)
        if adjusted < base_position:
            result["rationale"].append(f"Overvaluation reduces position: {base_position}% → {adjusted}%")
            base_position = adjusted
    
    # ============= NEW: APPLY DCF OVERVALUATION CAP =============
    # This is the hard cap - cannot exceed this regardless of other factors
    if base_position > overvaluation_cap:
        result["rationale"].append(f"DCF overvaluation caps position: {base_position}% → {overvaluation_cap}%")
        base_position = overvaluation_cap
    
    result["recommended_position_pct"] = base_position
    
    # Calculate entry tiers for limit orders (DCA strategy)
    # NEW: When severely overvalued, require DEEPER pullbacks
    remaining_pct = 100 - base_position
    
    if remaining_pct > 0:
        tiers = []
        
        # Determine pullback depth based on overvaluation severity
        dcf_overval = result.get("dcf_overvaluation_pct")
        if dcf_overval and dcf_overval > 30:
            # Severely overvalued: require 10-15% pullbacks
            tier1_pullback = 0.90  # 10% below current
            tier2_pullback = 0.85  # 15% below current
            tier1_desc = "10% pullback entry"
            tier2_desc = "15% deep value entry"
        elif dcf_overval and dcf_overval > 15:
            # Moderately overvalued: require 7-12% pullbacks
            tier1_pullback = 0.93  # 7% below
            tier2_pullback = 0.88  # 12% below
            tier1_desc = "7% pullback entry"
            tier2_desc = "12% pullback entry"
        else:
            # Normal: standard 3-6% pullbacks
            tier1_pullback = 0.97  # 3% below
            tier2_pullback = 0.94  # 6% below
            tier1_desc = "Quick dip entry"
            tier2_desc = "Pullback entry"
        
        # Tier 1
        tier1_price = current_price * tier1_pullback
        tier1_pct = min(remaining_pct * 0.35, 35)
        if tier1_pct >= 5:
            tiers.append({
                "price": round(tier1_price, 2),
                "position_pct": round(tier1_pct),
                "description": tier1_desc
            })
            remaining_pct -= tier1_pct
        
        # Tier 2
        tier2_price = current_price * tier2_pullback
        tier2_pct = min(remaining_pct * 0.5, 35)
        if tier2_pct >= 5:
            tiers.append({
                "price": round(tier2_price, 2),
                "position_pct": round(tier2_pct),
                "description": tier2_desc
            })
            remaining_pct -= tier2_pct
        
        # Tier 3: At intrinsic value if severely overvalued
        if remaining_pct >= 5:
            if intrinsic_value and intrinsic_value > 0 and intrinsic_value < tier2_price * 0.98:
                tiers.append({
                    "price": round(intrinsic_value, 2),
                    "position_pct": round(remaining_pct),
                    "description": "At intrinsic value"
                })
            elif ideal_entry_price < tier2_price * 0.98:
                tiers.append({
                    "price": round(ideal_entry_price, 2),
                    "position_pct": round(remaining_pct),
                    "description": "Ideal entry point"
                })
        
        result["entry_tiers"] = tiers
    
    return result


def calculate_risk_management(
    current_price: float,
    atr: float,
    support_levels: List[float],
    volatility_annualized: Optional[float] = None,
    risk_tolerance: str = "medium"
) -> Dict[str, Any]:
    """
    Calculate stop-loss and position sizing recommendations.
    
    Args:
        current_price: Current stock price
        atr: Average True Range
        support_levels: Support price levels
        volatility_annualized: Annualized volatility (optional)
        risk_tolerance: "low", "medium", or "high"
    
    Returns:
        Risk management recommendations
    """
    # Risk multipliers based on tolerance
    risk_multipliers = {
        "low": {"stop_atr": 1.5, "position_base": 2.0},
        "medium": {"stop_atr": 2.0, "position_base": 5.0},
        "high": {"stop_atr": 3.0, "position_base": 10.0}
    }
    
    multipliers = risk_multipliers.get(risk_tolerance, risk_multipliers["medium"])
    
    # Calculate stop-loss based on ATR
    atr_stop = current_price - (multipliers["stop_atr"] * atr)
    
    # Also consider support levels
    valid_supports = [s for s in support_levels if s < current_price] if support_levels else []
    if valid_supports:
        support_stop = max(valid_supports) * 0.98  # Just below support
        stop_loss = max(atr_stop, support_stop)  # Use the higher (tighter) stop
    else:
        stop_loss = atr_stop
    
    stop_loss_pct = (current_price - stop_loss) / current_price * 100
    
    # Calculate position size based on volatility and stop distance
    # Rule: Risk max 2% of portfolio per trade
    risk_per_trade = 0.02
    max_loss_per_share = current_price - stop_loss
    
    if max_loss_per_share > 0:
        # Position size = (Portfolio * Risk%) / (Price * Stop%)  
        # Simplified to suggested percentage of portfolio
        position_pct = min(multipliers["position_base"], (risk_per_trade / (stop_loss_pct / 100)) * 100)
    else:
        position_pct = multipliers["position_base"]
    
    # Adjust for high volatility
    if volatility_annualized and volatility_annualized > 0.5:  # > 50% annualized vol
        position_pct *= 0.7  # Reduce by 30%
    
    return {
        "stop_loss_price": round(stop_loss, 2),
        "stop_loss_pct": round(stop_loss_pct, 2),
        "position_size_pct": round(min(position_pct, 15), 2),  # Cap at 15%
        "trailing_stop_recommended": volatility_annualized and volatility_annualized > 0.3,
        "trailing_stop_pct": round(stop_loss_pct * 0.8, 2) if volatility_annualized and volatility_annualized > 0.3 else None,
        "risk_reward_notes": f"Stop at ${stop_loss:.2f} ({stop_loss_pct:.1f}% below current)"
    }


def generate_entry_signal(
    current_price: float,
    technical_data: Dict[str, Any],
    fundamental_data: Dict[str, Any],
    risk_tolerance: str = "medium",
    volume_data: Dict[str, Any] = None,
    market_regime: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Generate comprehensive entry signal with all adjustments.
    
    Args:
        current_price: Current stock price
        technical_data: Dictionary with RSI, MACD, ATR, support/resistance, etc.
        fundamental_data: Dictionary with PE, fair value, etc.
        risk_tolerance: "low", "medium", or "high"
        volume_data: Dictionary with volume metrics (optional)
        market_regime: Market regime data with 'current_regime' key (optional)
    
    Returns:
        Comprehensive signal with recommendations including staged entry"""
    warnings = []
    
    # 1. Base signal from technical trend
    base_signal_strength = 50  # Start neutral
    
    # MACD trend
    macd_trend = technical_data.get("macd_trend", "Neutral")
    if macd_trend == "Bullish":
        base_signal_strength += 15
    elif macd_trend == "Bearish":
        base_signal_strength -= 15
    
    # Moving average alignment
    sma_50 = technical_data.get("sma_50")
    sma_200 = technical_data.get("sma_200")
    if sma_50 and sma_200:
        if sma_50 > sma_200:  # Golden cross territory
            base_signal_strength += 10
        else:  # Death cross territory
            base_signal_strength -= 10
    
    # Price vs SMA
    if sma_50 and current_price > sma_50:
        base_signal_strength += 5
    elif sma_50 and current_price < sma_50:
        base_signal_strength -= 5
    
    # 2. Valuation adjustment
    valuation_signal = calculate_valuation_signal(
        current_price=current_price,
        fair_value=fundamental_data.get("fair_value"),
        pe_ratio=fundamental_data.get("pe_ratio"),
        industry_pe=fundamental_data.get("industry_pe", INDUSTRY_PE_AVERAGE),
        forward_pe=fundamental_data.get("forward_pe")
    )
    warnings.extend(valuation_signal.get("warnings", []))
    
    # 2b. PEG Ratio adjustment (growth-adjusted valuation)
    peg_signal = calculate_peg_signal(fundamental_data.get("peg_ratio"))
    if peg_signal.get("warning"):
        warnings.append(peg_signal["warning"])
    
    # 2c. Volume confirmation
    volume_signal = calculate_volume_signal(volume_data)
    if volume_signal.get("warning"):
        warnings.append(volume_signal["warning"])
    
    # 3. RSI adjustment
    rsi_value = technical_data.get("rsi", 50)
    rsi_signal = calculate_rsi_signal(rsi_value)
    if rsi_signal.get("warning"):
        warnings.append(rsi_signal["warning"])
    
    # 4. MACD divergence check
    prices = technical_data.get("price_series")
    macd_hist = technical_data.get("macd_histogram_series")
    divergence = {"divergence_type": "NONE", "macd_penalty": 0}
    
    if prices is not None and macd_hist is not None:
        divergence = detect_macd_divergence(prices, macd_hist)
        if divergence.get("warning"):
            warnings.append(divergence["warning"])
    
    # 5. Calculate adjusted signal strength
    adjusted_strength = base_signal_strength
    adjusted_strength *= valuation_signal["valuation_factor"]
    adjusted_strength *= peg_signal["peg_factor"]
    adjusted_strength *= rsi_signal["rsi_factor"]
    adjusted_strength *= volume_signal["volume_factor"]
    adjusted_strength *= (1 - divergence.get("macd_penalty", 0))
    
    # Clamp to 0-100
    adjusted_strength = max(0, min(100, adjusted_strength))
    
    # 6. Determine signal
    if adjusted_strength >= 70:
        signal = "STRONG_BUY"
    elif adjusted_strength >= 55:
        signal = "BUY"
    elif adjusted_strength >= 45:
        signal = "HOLD"
    elif adjusted_strength >= 30:
        signal = "SELL"
    else:
        signal = "STRONG_SELL"
    
    # 7. Entry recommendation
    support_levels = technical_data.get("support_levels", [])
    resistance_levels = technical_data.get("resistance_levels", [])
    atr = technical_data.get("atr", current_price * 0.02)  # Default 2% if missing
    
    entry_rec = calculate_entry_recommendation(
        current_price=current_price,
        support_levels=support_levels,
        resistance_levels=resistance_levels,
        atr=atr,
        fifty_two_week_high=fundamental_data.get("fifty_two_week_high")
    )
    
    # Override signal if pullback recommended
    if entry_rec["wait_for_pullback"] and signal in ["BUY", "STRONG_BUY"]:
        signal = "WAIT_FOR_PULLBACK"
        warnings.extend(entry_rec.get("rationale", []))
    
    # 8. Risk management
    volatility = technical_data.get("volatility_annualized")
    risk_mgmt = calculate_risk_management(
        current_price=current_price,
        atr=atr,
        support_levels=support_levels,
        volatility_annualized=volatility,
        risk_tolerance=risk_tolerance
    )
    
    # 9. Staged Entry Calculation
    # Calculate ideal entry price from support levels, fair value, or SMA
    ideal_entry = None
    fair_value = fundamental_data.get("fair_value")
    
    # First priority: Use nearest support level (most actionable for entry)
    if support_levels:
        valid_supports = [s for s in support_levels if s < current_price and s > current_price * 0.8]
        if valid_supports:
            ideal_entry = max(valid_supports)  # Nearest support below price
    
    # Second priority: Fair value (if reasonable - within 20% of current price)
    if not ideal_entry and fair_value and fair_value > 0:
        # Validate fair value is reasonable (not book value, not 1000% off)
        if fair_value > current_price * 0.8 and fair_value < current_price * 1.2:
            ideal_entry = fair_value
    
    # Third priority: 50-day SMA (if available and reasonable)
    if not ideal_entry and sma_50:
        if sma_50 > current_price * 0.8 and sma_50 < current_price * 1.1:
            ideal_entry = sma_50
    
    # Final fallback: 5-8% below current price (standard pullback entry)
    if not ideal_entry:
        ideal_entry = current_price * 0.93
    
    # Calculate intrinsic value for DCF overvaluation check
    # This is separate from ideal_entry - intrinsic value is the true DCF fair value
    # which may be far below current price for overvalued stocks
    intrinsic_value = fundamental_data.get("dcf_intrinsic_value")
    if not intrinsic_value:
        # Fallback: use fair_value from analyst targets or DCF if available
        intrinsic_value = fundamental_data.get("fair_value")
    
    staged_entry = calculate_staged_entry(
        current_price=current_price,
        ideal_entry_price=ideal_entry,
        regime=market_regime,
        valuation_status=valuation_signal["valuation_status"],
        rsi_status=rsi_signal["rsi_status"],
        intrinsic_value=intrinsic_value  # NEW: Pass DCF value for overvaluation check
    )
    
    return {
        "signal": signal,
        "signal_strength": round(adjusted_strength, 1),
        "base_signal_strength": round(base_signal_strength, 1),
        "confidence_adjustments": {
            "valuation_factor": valuation_signal["valuation_factor"],
            "valuation_status": valuation_signal["valuation_status"],
            "peg_factor": peg_signal["peg_factor"],
            "peg_status": peg_signal["peg_status"],
            "volume_factor": volume_signal["volume_factor"],
            "volume_status": volume_signal["volume_status"],
            "volume_conviction": volume_signal["conviction"],
            "rsi_factor": rsi_signal["rsi_factor"],
            "rsi_status": rsi_signal["rsi_status"],
            "macd_divergence": divergence["divergence_type"],
            "macd_penalty": divergence.get("macd_penalty", 0)
        },
        "entry_recommendation": {
            "immediate_entry_ok": entry_rec["immediate_entry_ok"],
            "wait_for_pullback": entry_rec["wait_for_pullback"],
            "pullback_target": entry_rec.get("pullback_target"),
            "pullback_pct": entry_rec.get("pullback_pct"),
            "near_all_time_high": entry_rec.get("near_all_time_high", False)
        },
        "staged_entry": {
            "recommended_position_pct": staged_entry["recommended_position_pct"],
            "price_vs_ideal_pct": staged_entry["price_vs_ideal_pct"],
            "ideal_entry_price": staged_entry["ideal_entry_price"],
            "regime_override_applied": staged_entry["regime_override_applied"],
            "base_position_pct": staged_entry["base_position_pct"],
            "rationale": staged_entry["rationale"],
            "entry_tiers": staged_entry["entry_tiers"]
        },
        "risk_management": risk_mgmt,
        "warnings": warnings
    }
