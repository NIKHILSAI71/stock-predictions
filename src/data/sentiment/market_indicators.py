"""
Market Sentiment Indicators Module
Put/Call Ratio, VIX, Short Interest
"""

from typing import Dict, Any, Optional
import yfinance as yf


def get_vix_data() -> Dict[str, Any]:
    """
    Get VIX (CBOE Volatility Index) data - the "fear gauge".

    VIX Interpretation:
        < 12: Extremely low volatility, complacency
        12-20: Low volatility, stable market
        20-30: Moderate volatility
        30-40: High volatility, fear in market
        > 40: Extreme fear, potential market crash

    Returns:
        Dictionary with VIX data and interpretation
    """
    try:
        vix = yf.Ticker("^VIX")
        hist = vix.history(period="5d")

        if hist.empty:
            return {'error': 'Unable to fetch VIX data'}

        current_vix = hist['Close'].iloc[-1]
        prev_vix = hist['Close'].iloc[-2] if len(hist) > 1 else current_vix
        change = current_vix - prev_vix

        # Interpretation
        if current_vix < 12:
            sentiment = 'Extremely Low - Complacency'
            signal = 'caution'
        elif current_vix < 20:
            sentiment = 'Low - Stable Market'
            signal = 'bullish'
        elif current_vix < 30:
            sentiment = 'Moderate - Normal Volatility'
            signal = 'neutral'
        elif current_vix < 40:
            sentiment = 'High - Fear in Market'
            signal = 'bearish'
        else:
            sentiment = 'Extreme Fear - Potential Bottom'
            signal = 'contrarian_bullish'

        return {
            'current_vix': round(current_vix, 2),
            'change': round(change, 2),
            'change_pct': round(change / prev_vix * 100, 2) if prev_vix else 0,
            'interpretation': sentiment,
            'signal': signal
        }
    except Exception as e:
        return {'error': str(e)}


def fear_greed_indicator(
    vix_level: float,
    put_call_ratio: float,
    market_momentum: float,  # % above/below 125-day MA
    safe_haven_demand: float,  # Bond/stock relative performance
    junk_bond_demand: float  # Junk vs investment grade spread
) -> Dict[str, Any]:
    """
    Calculate a simplified Fear & Greed Index (similar to CNN's).

    Scale: 0 (Extreme Fear) to 100 (Extreme Greed)

    Args:
        vix_level: Current VIX level
        put_call_ratio: Put/Call ratio
        market_momentum: Market momentum indicator
        safe_haven_demand: Safe haven demand indicator
        junk_bond_demand: Junk bond demand indicator

    Returns:
        Dictionary with Fear & Greed index
    """
    scores = []

    # VIX component (inverted - high VIX = fear)
    if vix_level < 15:
        vix_score = 90
    elif vix_level < 20:
        vix_score = 70
    elif vix_level < 25:
        vix_score = 50
    elif vix_level < 35:
        vix_score = 30
    else:
        vix_score = 10
    scores.append(vix_score)

    # Put/Call component (inverted - high P/C = fear)
    if put_call_ratio < 0.7:
        pc_score = 80
    elif put_call_ratio < 0.9:
        pc_score = 60
    elif put_call_ratio < 1.1:
        pc_score = 40
    else:
        pc_score = 20
    scores.append(pc_score)

    # Momentum component
    if market_momentum > 5:
        mom_score = 80
    elif market_momentum > 0:
        mom_score = 60
    elif market_momentum > -5:
        mom_score = 40
    else:
        mom_score = 20
    scores.append(mom_score)

    # Average all components
    index_value = sum(scores) / len(scores)

    # Classify
    if index_value >= 75:
        classification = 'Extreme Greed'
    elif index_value >= 55:
        classification = 'Greed'
    elif index_value >= 45:
        classification = 'Neutral'
    elif index_value >= 25:
        classification = 'Fear'
    else:
        classification = 'Extreme Fear'

    return {
        'index': round(index_value, 1),
        'classification': classification,
        'components': {
            'vix_score': vix_score,
            'put_call_score': pc_score,
            'momentum_score': mom_score
        }
    }
