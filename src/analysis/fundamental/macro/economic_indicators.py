"""
Macroeconomic Indicators Module
"""

import yfinance as yf
from typing import Dict, Any, List
from datetime import datetime


def get_treasury_yields() -> Dict[str, Any]:
    """
    Get current US Treasury yields.
    
    Returns:
        Dictionary with various Treasury yields
    """
    symbols = {
        '3_month': '^IRX',   # 13-week T-bill
        '2_year': '^TYX',    # Approximation
        '10_year': '^TNX',   # 10-year Treasury
        '30_year': '^TYX'    # 30-year Treasury
    }
    
    yields = {}
    
    for name, symbol in symbols.items():
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="5d")
            if not hist.empty:
                yields[name] = round(hist['Close'].iloc[-1], 2)
        except:
            yields[name] = None
    
    # Calculate yield curve (10Y - 2Y spread)
    if yields.get('10_year') and yields.get('2_year'):
        spread = yields['10_year'] - yields['2_year']
        yields['yield_curve_spread'] = round(spread, 2)
        yields['yield_curve_status'] = 'Inverted (Recession Signal)' if spread < 0 else 'Normal'
    
    return yields


def analyze_yield_curve(
    short_rate: float,
    long_rate: float
) -> Dict[str, Any]:
    """
    Analyze yield curve shape.
    
    Args:
        short_rate: Short-term yield (e.g., 2-year)
        long_rate: Long-term yield (e.g., 10-year)
    
    Returns:
        Dictionary with yield curve analysis
    """
    spread = long_rate - short_rate
    
    if spread > 2:
        shape = 'Steep'
        implication = 'Strong economic growth expected'
    elif spread > 0.5:
        shape = 'Normal'
        implication = 'Healthy economic conditions'
    elif spread > 0:
        shape = 'Flat'
        implication = 'Slowing growth, uncertainty'
    else:
        shape = 'Inverted'
        implication = 'Recession warning signal'
    
    return {
        'spread': round(spread, 2),
        'shape': shape,
        'implication': implication,
        'short_rate': short_rate,
        'long_rate': long_rate
    }


def get_market_indices() -> Dict[str, Any]:
    """
    Get major market indices performance.
    
    Returns:
        Dictionary with index data
    """
    indices = {
        'sp500': '^GSPC',
        'dow': '^DJI',
        'nasdaq': '^IXIC',
        'russell2000': '^RUT',
        'vix': '^VIX'
    }
    
    result = {}
    
    for name, symbol in indices.items():
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="5d")
            if not hist.empty:
                current = hist['Close'].iloc[-1]
                prev = hist['Close'].iloc[-2] if len(hist) > 1 else current
                change = ((current - prev) / prev) * 100
                
                result[name] = {
                    'value': round(current, 2),
                    'change_pct': round(change, 2)
                }
        except:
            result[name] = None
    
    return result


def economic_calendar_impact(
    indicator: str,
    actual: float,
    forecast: float,
    previous: float
) -> Dict[str, Any]:
    """
    Analyze economic indicator release impact.
    
    Args:
        indicator: Name of indicator (e.g., 'GDP', 'CPI', 'NFP')
        actual: Actual released value
        forecast: Consensus forecast
        previous: Previous period value
    
    Returns:
        Dictionary with impact analysis
    """
    surprise = actual - forecast
    change_from_previous = actual - previous
    
    # Determine beat/miss
    if abs(surprise) < 0.01 * abs(forecast):
        result = 'In-line'
    elif surprise > 0:
        result = 'Beat'
    else:
        result = 'Miss'
    
    # Trend
    if change_from_previous > 0:
        trend = 'Improving'
    elif change_from_previous < 0:
        trend = 'Declining'
    else:
        trend = 'Stable'
    
    # Market impact interpretation (simplified)
    positive_indicators = ['GDP', 'NFP', 'Retail Sales', 'PMI', 'Consumer Confidence']
    negative_indicators = ['Unemployment', 'CPI', 'Initial Claims']
    
    if indicator in positive_indicators:
        market_impact = 'Bullish' if surprise > 0 else 'Bearish'
    elif indicator in negative_indicators:
        market_impact = 'Bearish' if surprise > 0 else 'Bullish'
    else:
        market_impact = 'Neutral'
    
    return {
        'indicator': indicator,
        'actual': actual,
        'forecast': forecast,
        'previous': previous,
        'surprise': round(surprise, 2),
        'result': result,
        'trend': trend,
        'market_impact': market_impact
    }


def sector_performance() -> Dict[str, Any]:
    """
    Get sector performance data.
    
    Returns:
        Dictionary with sector ETF performance
    """
    sectors = {
        'Technology': 'XLK',
        'Healthcare': 'XLV',
        'Financials': 'XLF',
        'Consumer Discretionary': 'XLY',
        'Consumer Staples': 'XLP',
        'Energy': 'XLE',
        'Utilities': 'XLU',
        'Materials': 'XLB',
        'Industrials': 'XLI',
        'Real Estate': 'XLRE',
        'Communication Services': 'XLC'
    }
    
    result = {}
    
    for sector, symbol in sectors.items():
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="1mo")
            if not hist.empty:
                current = hist['Close'].iloc[-1]
                month_ago = hist['Close'].iloc[0]
                month_return = ((current - month_ago) / month_ago) * 100
                
                result[sector] = {
                    'symbol': symbol,
                    'price': round(current, 2),
                    'month_return_pct': round(month_return, 2)
                }
        except:
            pass
    
    # Sort by performance
    sorted_sectors = dict(sorted(result.items(), key=lambda x: x[1]['month_return_pct'], reverse=True))
    
    return {
        'sectors': sorted_sectors,
        'top_performers': list(sorted_sectors.keys())[:3],
        'worst_performers': list(sorted_sectors.keys())[-3:]
    }
