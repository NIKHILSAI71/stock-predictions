from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse
from typing import Optional

from src.core.utils import sanitize_for_json
from src.data import get_company_info, validate_symbol
from src.analysis.technical.strategies import analyze_growth_metrics, analyze_value_metrics
from src.data.news_fetcher import get_market_sentiment_search

router = APIRouter()

@router.get("/screening")
async def screen_stocks(
    min_pe: Optional[float] = None,
    max_pe: Optional[float] = None,
    min_market_cap: Optional[float] = None,
    min_dividend_yield: Optional[float] = None,
    symbols: str = "AAPL,MSFT,GOOGL,AMZN,META,NVDA,TSLA"
):
    """Screen stocks based on criteria."""
    try:
        symbol_list = [s.strip() for s in symbols.split(',')]
        results = []
        
        for symbol in symbol_list:
            try:
                info = get_company_info(symbol)
                
                # Apply filters
                if min_pe and info.get('trailing_pe', 0) < min_pe:
                    continue
                if max_pe and info.get('trailing_pe', float('inf')) > max_pe:
                    continue
                if min_market_cap and info.get('market_cap', 0) < min_market_cap:
                    continue
                if min_dividend_yield and (info.get('dividend_yield', 0) or 0) * 100 < min_dividend_yield:
                    continue
                
                results.append({
                    'symbol': symbol,
                    'name': info.get('name'),
                    'price': info.get('current_price'),
                    'pe_ratio': info.get('trailing_pe'),
                    'market_cap': info.get('market_cap'),
                    'dividend_yield': round(info.get('dividend_yield', 0) * 100, 2) if info.get('dividend_yield') else 0
                })
            except:
                continue
        
        return {
            "status": "success",
            "count": len(results),
            "results": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/strategy/analysis/{symbol}")
async def get_strategy_analysis(symbol: str):
    """Analyze stock using Growth and Value strategies."""
    try:
        info = get_company_info(symbol)
        
        # Prepare metrics (Mapping info keys to strategy expected keys)
        metrics = {
            'pe_ratio': info.get('trailing_pe') or 0.0,
            'pb_ratio': info.get('price_to_book') or 0.0,
            'current_ratio': info.get('current_ratio') or 0.0, 
            'debt_to_equity': info.get('debt_to_equity') or 0.0, 
            'dividend_yield': (info.get('dividend_yield') or 0) * 100,
            'eps_growth': (info.get('earnings_growth') or 0.15) * 100,
            'revenue_growth': (info.get('revenue_growth') or 0) * 100,
            'roe': (info.get('return_on_equity') or 0) * 100,
            'relative_strength': 50  # Neutral default - requires technical calculation
        }
        
        growth = analyze_growth_metrics(metrics)
        value = analyze_value_metrics(metrics)
        
        return {
            "status": "success",
            "symbol": symbol.upper(),
            "strategies": {
                "growth": growth,
                "value": value
            }
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "detail": str(e)})

@router.get("/alternative/{symbol}")
async def get_alternative_data(symbol: str):
    """Get alternative data (Sentiment, News)."""
    try:
        # Fetch real-time sentiment (No mocks)
        sentiment = await get_market_sentiment_search(symbol)
        
        news_vol = sentiment.get('news_volume', 0)
        traffic_level = "Very High" if news_vol >= 8 else ("High" if news_vol >= 5 else ("Medium" if news_vol >= 3 else "Low"))
        
        return {
            "status": "success",
            "symbol": symbol.upper(),
            "alternative_data": {
                "social_sentiment": sentiment,
                "web_traffic": {
                    "level": traffic_level,
                    "source": "Search & Media Volume",
                    "value": f"{news_vol} active sources"
                }
            }
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "detail": str(e)})
