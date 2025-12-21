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


def calculate_put_call_ratio(
    put_volume: int,
    call_volume: int
) -> Dict[str, Any]:
    """
    Calculate Put/Call ratio.
    
    Formula:
        Put/Call Ratio = Put Volume / Call Volume
    
    Interpretation:
        < 0.7: Bullish sentiment (more calls)
        0.7 - 1.0: Neutral
        > 1.0: Bearish sentiment (more puts)
        > 1.5: Extreme fear (potential contrarian buy)
    
    Args:
        put_volume: Total put option volume
        call_volume: Total call option volume
    
    Returns:
        Dictionary with ratio and interpretation
    """
    if call_volume == 0:
        return {'error': 'Call volume is zero'}
    
    ratio = put_volume / call_volume
    
    if ratio < 0.7:
        sentiment = 'Bullish'
        signal = 'bullish'
    elif ratio < 1.0:
        sentiment = 'Neutral'
        signal = 'neutral'
    elif ratio < 1.5:
        sentiment = 'Bearish'
        signal = 'bearish'
    else:
        sentiment = 'Extreme Fear - Contrarian Bullish'
        signal = 'contrarian_bullish'
    
    return {
        'put_call_ratio': round(ratio, 3),
        'put_volume': put_volume,
        'call_volume': call_volume,
        'interpretation': sentiment,
        'signal': signal
    }


def analyze_short_interest(
    shares_short: int,
    shares_outstanding: int,
    avg_daily_volume: int
) -> Dict[str, Any]:
    """
    Analyze short interest metrics.
    
    Metrics:
        Short Interest Ratio = Shares Short / Shares Outstanding
        Days to Cover = Shares Short / Average Daily Volume
    
    Args:
        shares_short: Number of shares sold short
        shares_outstanding: Total shares outstanding
        avg_daily_volume: Average daily trading volume
    
    Returns:
        Dictionary with short interest analysis
    """
    short_interest_ratio = shares_short / shares_outstanding if shares_outstanding > 0 else 0
    days_to_cover = shares_short / avg_daily_volume if avg_daily_volume > 0 else 0
    
    # Interpretation
    if short_interest_ratio > 0.20:
        interpretation = 'Very High Short Interest - Potential Squeeze'
    elif short_interest_ratio > 0.10:
        interpretation = 'High Short Interest'
    elif short_interest_ratio > 0.05:
        interpretation = 'Moderate Short Interest'
    else:
        interpretation = 'Low Short Interest'
    
    return {
        'shares_short': shares_short,
        'short_interest_pct': round(short_interest_ratio * 100, 2),
        'days_to_cover': round(days_to_cover, 1),
        'interpretation': interpretation
    }


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
