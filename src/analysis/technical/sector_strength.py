"""
Sector Relative Strength Module

Compares individual stock performance against sector ETF benchmarks
to identify outperformers and underperformers.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import yfinance as yf


# Sector to ETF mapping
SECTOR_ETFS = {
    "Technology": "XLK",
    "Healthcare": "XLV",
    "Financial Services": "XLF",
    "Financials": "XLF",
    "Consumer Cyclical": "XLY",
    "Consumer Discretionary": "XLY",
    "Consumer Defensive": "XLP",
    "Consumer Staples": "XLP",
    "Industrials": "XLI",
    "Energy": "XLE",
    "Utilities": "XLU",
    "Real Estate": "XLRE",
    "Materials": "XLB",
    "Basic Materials": "XLB",
    "Communication Services": "XLC",
    "Communication": "XLC",
    # Broader market
    "Market": "SPY"
}

# Default to SPY if sector unknown
DEFAULT_ETF = "SPY"


def get_sector_etf(sector: str) -> str:
    """Get the appropriate sector ETF for comparison."""
    if not sector:
        return DEFAULT_ETF
    
    # Try exact match first
    if sector in SECTOR_ETFS:
        return SECTOR_ETFS[sector]
    
    # Try partial match
    sector_lower = sector.lower()
    for key, etf in SECTOR_ETFS.items():
        if key.lower() in sector_lower or sector_lower in key.lower():
            return etf
    
    return DEFAULT_ETF


def calculate_relative_strength(
    stock_data: pd.DataFrame,
    sector_data: pd.DataFrame,
    lookback_periods: list = [5, 10, 20, 60]
) -> Dict[str, Any]:
    """
    Calculate relative strength of stock vs sector ETF.
    
    Args:
        stock_data: Stock price DataFrame with 'Close' column
        sector_data: Sector ETF price DataFrame with 'Close' column
        lookback_periods: Periods for RS calculation
    
    Returns:
        Relative strength metrics
    """
    if 'Close' not in stock_data.columns or 'Close' not in sector_data.columns:
        return {"error": "Missing Close column in data"}
    
    if len(stock_data) < max(lookback_periods) or len(sector_data) < max(lookback_periods):
        return {"error": "Insufficient data for RS calculation"}
    
    # Align data
    stock_close = stock_data['Close']
    sector_close = sector_data['Close']
    
    # Make sure we have the same index
    common_idx = stock_close.index.intersection(sector_close.index)
    stock_close = stock_close.loc[common_idx]
    sector_close = sector_close.loc[common_idx]
    
    if len(common_idx) < max(lookback_periods):
        return {"error": "Insufficient overlapping data"}
    
    # Calculate relative strength ratio (stock price / sector price)
    rs_ratio = stock_close / sector_close
    
    # Calculate returns for different periods
    results = {}
    
    for period in lookback_periods:
        stock_return = (stock_close.iloc[-1] / stock_close.iloc[-period] - 1) * 100
        sector_return = (sector_close.iloc[-1] / sector_close.iloc[-period] - 1) * 100
        
        # Relative performance (stock - sector)
        relative_perf = stock_return - sector_return
        
        # RS ratio change
        rs_change = (rs_ratio.iloc[-1] / rs_ratio.iloc[-period] - 1) * 100
        
        results[f"{period}d"] = {
            "stock_return": round(stock_return, 2),
            "sector_return": round(sector_return, 2),
            "relative_performance": round(relative_perf, 2),
            "rs_ratio_change": round(rs_change, 2),
            "outperforming": relative_perf > 0
        }
    
    # Calculate RS line trend (is the RS ratio trending up?)
    rs_sma20 = rs_ratio.rolling(20).mean().iloc[-1]
    rs_sma50 = rs_ratio.rolling(50).mean().iloc[-1] if len(rs_ratio) >= 50 else rs_sma20
    
    rs_trend = "IMPROVING" if rs_sma20 > rs_sma50 else "WEAKENING"
    
    # Current RS percentile (where is current RS vs history)
    rs_percentile = (rs_ratio <= rs_ratio.iloc[-1]).mean() * 100
    
    return {
        "current_rs_ratio": round(rs_ratio.iloc[-1], 4),
        "rs_trend": rs_trend,
        "rs_percentile": round(rs_percentile, 1),
        "periods": results,
        "timestamp": datetime.now().isoformat()
    }


def get_sector_performance(sector_etf: str, period: str = "1mo") -> Dict[str, Any]:
    """
    Get sector ETF performance data.
    
    Args:
        sector_etf: ETF ticker (e.g., XLK)
        period: Time period for analysis
    
    Returns:
        Sector performance metrics
    """
    try:
        ticker = yf.Ticker(sector_etf)
        data = ticker.history(period=period)
        
        if data.empty:
            return {"error": f"No data for {sector_etf}"}
        
        current_price = data['Close'].iloc[-1]
        period_start = data['Close'].iloc[0]
        period_return = (current_price / period_start - 1) * 100
        
        # Volatility
        returns = data['Close'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252) * 100
        
        # Get info
        info = ticker.info
        
        return {
            "etf": sector_etf,
            "name": info.get("longName", sector_etf),
            "current_price": round(current_price, 2),
            "period_return": round(period_return, 2),
            "volatility": round(volatility, 2),
            "volume": int(data['Volume'].iloc[-1]),
            "avg_volume": int(data['Volume'].mean()),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {"error": str(e), "etf": sector_etf}


def get_relative_strength_rating(
    stock_data: pd.DataFrame,
    sector: str,
    period: str = "3mo"
) -> Dict[str, Any]:
    """
    Get comprehensive relative strength rating for a stock.
    
    Args:
        stock_data: Stock OHLCV DataFrame
        sector: Stock's sector name
        period: Analysis period
    
    Returns:
        Relative strength rating and analysis
    """
    # Get sector ETF
    sector_etf = get_sector_etf(sector)
    
    try:
        # Fetch sector data
        ticker = yf.Ticker(sector_etf)
        sector_data = ticker.history(period=period)
        
        if sector_data.empty:
            return {
                "available": False,
                "sector_etf": sector_etf,
                "error": "Could not fetch sector data"
            }
        
        # Calculate relative strength
        rs = calculate_relative_strength(stock_data, sector_data)
        
        if "error" in rs:
            return {
                "available": False,
                "sector_etf": sector_etf,
                "error": rs["error"]
            }
        
        # Get sector performance
        sector_perf = get_sector_performance(sector_etf, period)
        
        # Determine rating
        rs_20d = rs["periods"].get("20d", {}).get("relative_performance", 0)
        rs_60d = rs["periods"].get("60d", {}).get("relative_performance", 0)
        rs_trend = rs["rs_trend"]
        
        # Scoring
        score = 0
        
        if rs_20d > 10:
            score += 2
        elif rs_20d > 5:
            score += 1
        elif rs_20d < -10:
            score -= 2
        elif rs_20d < -5:
            score -= 1
        
        if rs_60d > 15:
            score += 2
        elif rs_60d > 5:
            score += 1
        elif rs_60d < -15:
            score -= 2
        elif rs_60d < -5:
            score -= 1
        
        if rs_trend == "IMPROVING":
            score += 1
        else:
            score -= 1
        
        # Rating
        if score >= 4:
            rating = "STRONG_OUTPERFORM"
            interpretation = "Significantly outperforming sector - strong momentum"
        elif score >= 2:
            rating = "OUTPERFORM"
            interpretation = "Outperforming sector - positive relative strength"
        elif score <= -4:
            rating = "STRONG_UNDERPERFORM"
            interpretation = "Significantly lagging sector - weak momentum"
        elif score <= -2:
            rating = "UNDERPERFORM"
            interpretation = "Underperforming sector - negative relative strength"
        else:
            rating = "MARKET_PERFORM"
            interpretation = "Performing in line with sector"
        
        return {
            "available": True,
            "sector": sector,
            "sector_etf": sector_etf,
            "rating": rating,
            "score": score,
            "interpretation": interpretation,
            "relative_strength": rs,
            "sector_performance": sector_perf,
            "key_metrics": {
                "20d_relative_perf": rs_20d,
                "60d_relative_perf": rs_60d,
                "rs_trend": rs_trend,
                "rs_percentile": rs["rs_percentile"]
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            "available": False,
            "sector_etf": sector_etf,
            "error": str(e)
        }


def compare_to_market(
    stock_data: pd.DataFrame,
    period: str = "3mo"
) -> Dict[str, Any]:
    """
    Compare stock to overall market (SPY).
    
    Args:
        stock_data: Stock OHLCV DataFrame
        period: Analysis period
    
    Returns:
        Market-relative performance
    """
    return get_relative_strength_rating(stock_data, "Market", period)
