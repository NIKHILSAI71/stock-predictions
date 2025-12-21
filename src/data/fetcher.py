"""
Stock Data Fetcher Module
Retrieves stock data from Yahoo Finance using yfinance
"""

import yfinance as yf
import pandas as pd
from typing import Optional, Dict, Any
from datetime import datetime, timedelta


from functools import lru_cache

@lru_cache(maxsize=32)
def get_stock_data(
    symbol: str,
    period: str = "1y",
    interval: str = "1d"
) -> pd.DataFrame:
    """
    Fetch historical OHLCV data for a stock.
    
    Args:
        symbol: Stock ticker symbol (e.g., 'AAPL', 'MSFT')
        period: Data period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
        interval: Data interval ('1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo')
    
    Returns:
        DataFrame with columns: Open, High, Low, Close, Volume, Adj Close
    """
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=period, interval=interval)
        
        if data.empty:
            raise ValueError(f"No data found for symbol: {symbol}")
        
        # Ensure column names are standardized
        data.columns = [col.title() for col in data.columns]
        
        return data
    except Exception as e:
        raise ValueError(f"Error fetching data for {symbol}: {str(e)}")


def get_current_price(symbol: str) -> Dict[str, Any]:
    """
    Get the current/latest price information for a stock.
    
    Args:
        symbol: Stock ticker symbol
    
    Returns:
        Dictionary with current price information
    """
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        return {
            "symbol": symbol.upper(),
            "current_price": info.get("regularMarketPrice", info.get("currentPrice")),
            "previous_close": info.get("previousClose"),
            "open": info.get("regularMarketOpen", info.get("open")),
            "day_high": info.get("dayHigh"),
            "day_low": info.get("dayLow"),
            "volume": info.get("volume"),
            "market_cap": info.get("marketCap"),
            "fifty_two_week_high": info.get("fiftyTwoWeekHigh"),
            "fifty_two_week_low": info.get("fiftyTwoWeekLow"),
            "currency": info.get("currency", "USD"),
            "last_updated": datetime.now().isoformat()
        }
    except Exception as e:
        raise ValueError(f"Error fetching current price for {symbol}: {str(e)}")


def get_financial_data(symbol: str) -> Dict[str, Any]:
    """
    Get financial statements data for fundamental analysis.
    
    Args:
        symbol: Stock ticker symbol
    
    Returns:
        Dictionary with balance sheet, income statement, and cash flow data
    """
    try:
        ticker = yf.Ticker(symbol)
        
        # Get financial statements
        balance_sheet = ticker.balance_sheet
        income_stmt = ticker.income_stmt
        cash_flow = ticker.cashflow
        
        return {
            "symbol": symbol.upper(),
            "balance_sheet": balance_sheet.to_dict() if not balance_sheet.empty else {},
            "income_statement": income_stmt.to_dict() if not income_stmt.empty else {},
            "cash_flow": cash_flow.to_dict() if not cash_flow.empty else {},
            "last_updated": datetime.now().isoformat()
        }
    except Exception as e:
        raise ValueError(f"Error fetching financial data for {symbol}: {str(e)}")


@lru_cache(maxsize=64)
def get_company_info(symbol: str) -> Dict[str, Any]:
    """
    Get company profile and key statistics.
    
    Args:
        symbol: Stock ticker symbol
    
    Returns:
        Dictionary with company information
    """
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        data = {
            "symbol": symbol.upper(),
            "name": info.get("longName", info.get("shortName", symbol)),
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "country": info.get("country"),
            "website": info.get("website"),
            "description": info.get("longBusinessSummary"),
            "employees": info.get("fullTimeEmployees"),
            "market_cap": info.get("marketCap"),
            "enterprise_value": info.get("enterpriseValue"),
            "trailing_pe": info.get("trailingPE"),
            "forward_pe": info.get("forwardPE"),
            "peg_ratio": info.get("pegRatio"),
            "price_to_book": info.get("priceToBook"),
            "price_to_sales": info.get("priceToSalesTrailing12Months"),
            "dividend_yield": info.get("dividendYield"),
            "dividend_rate": info.get("dividendRate"),
            "beta": info.get("beta"),
            "eps_trailing": info.get("trailingEps"),
            "eps_forward": info.get("forwardEps"),
            "revenue": info.get("totalRevenue"),
            "gross_profit": info.get("grossProfits"),
            "ebitda": info.get("ebitda"),
            "net_income": info.get("netIncomeToCommon"),
            "total_debt": info.get("totalDebt"),
            "total_cash": info.get("totalCash"),
            "book_value": info.get("bookValue"),
            "shares_outstanding": info.get("sharesOutstanding"),
            "float_shares": info.get("floatShares"),
            # New Metrics for Dashboard Expansion
            "return_on_equity": info.get("returnOnEquity"),
            "return_on_assets": info.get("returnOnAssets"),
            "current_ratio": info.get("currentRatio"),
            "quick_ratio": info.get("quickRatio"),
            "debt_to_equity": info.get("debtToEquity"),
            "revenue_growth": info.get("revenueGrowth"),
            "earnings_growth": info.get("earningsGrowth"),
            "free_cash_flow": info.get("freeCashflow"),
            "operating_margins": info.get("operatingMargins"),
            "target_mean_price": info.get("targetMeanPrice"),
            "target_high_price": info.get("targetHighPrice"),
            "target_low_price": info.get("targetLowPrice"),
            "recommendation_key": info.get("recommendationKey"),
            "number_of_analysts": info.get("numberOfAnalystOpinions"),
            "last_updated": datetime.now().isoformat()
        }
        
        # Manual PEG calculation if missing
        if not data.get("peg_ratio") and data.get("trailing_pe") and data.get("earnings_growth"):
            try:
                # PEG = PE / (Earnings Growth Rate * 100)
                growth_rate = data["earnings_growth"] * 100
                if growth_rate > 0:
                    data["peg_ratio"] = round(data["trailing_pe"] / growth_rate, 2)
            except:
                pass
                
        return data
    except Exception as e:
        raise ValueError(f"Error fetching company info for {symbol}: {str(e)}")


def get_dividends(symbol: str) -> pd.DataFrame:
    """
    Get dividend history for a stock.
    
    Args:
        symbol: Stock ticker symbol
    
    Returns:
        DataFrame with dividend history
    """
    try:
        ticker = yf.Ticker(symbol)
        dividends = ticker.dividends
        
        return dividends
    except Exception as e:
        raise ValueError(f"Error fetching dividends for {symbol}: {str(e)}")


def validate_symbol(symbol: str) -> bool:
    """
    Validate if a stock symbol exists.
    
    Args:
        symbol: Stock ticker symbol
    
    Returns:
        True if symbol exists, False otherwise
    """
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        return "regularMarketPrice" in info or "currentPrice" in info
    except:
        return False
