from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse
from typing import Optional

from src.core.utils import sanitize_for_json
from src.data import get_stock_data, get_current_price, get_company_info, validate_symbol
from src.analysis.technical import (
    sma, ema, rsi, macd, bollinger_bands
)

router = APIRouter()

@router.get("/stock/{symbol}")
async def get_stock_overview(symbol: str):
    """Get comprehensive stock overview."""
    try:
        if not validate_symbol(symbol):
            return JSONResponse(
                status_code=404,
                content={"status": "error", "detail": f"Symbol '{symbol}' not found. Please check the symbol and try again."}
            )
        
        info = get_company_info(symbol)
        price = get_current_price(symbol)
        
        return {
            "status": "success",
            "data": {
                **info,
                "price_info": price
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "detail": str(e)}
        )

@router.get("/chart/{symbol}")
async def get_chart_data(
    symbol: str,
    period: str = "6mo",
    interval: str = "1d",
    indicators: Optional[str] = "sma,bb,volume"
):
    """Get chart data with optional indicators overlay."""
    try:
        data = get_stock_data(symbol, period, interval)
        
        # Convert to JSON-serializable format
        chart_data = {
            "dates": data.index.strftime('%Y-%m-%d').tolist(),
            "open": data['Open'].round(2).tolist(),
            "high": data['High'].round(2).tolist(),
            "low": data['Low'].round(2).tolist(),
            "close": data['Close'].round(2).tolist(),
            "volume": data['Volume'].tolist()
        }
        
        # Add requested indicators
        indicator_list = indicators.split(',') if indicators else []
        
        if 'sma' in indicator_list:
            chart_data['sma_20'] = sma(data, 20).round(2).tolist()
            chart_data['sma_50'] = sma(data, 50).round(2).tolist()
        
        if 'ema' in indicator_list:
            chart_data['ema_12'] = ema(data, 12).round(2).tolist()
            chart_data['ema_26'] = ema(data, 26).round(2).tolist()
        
        if 'bb' in indicator_list:
            bb = bollinger_bands(data)
            chart_data['bb_upper'] = bb['upper'].round(2).tolist()
            chart_data['bb_middle'] = bb['middle'].round(2).tolist()
            chart_data['bb_lower'] = bb['lower'].round(2).tolist()
        
        if 'rsi' in indicator_list:
            chart_data['rsi'] = rsi(data).round(2).tolist()
        
        if 'macd' in indicator_list:
            macd_data = macd(data)
            chart_data['macd'] = macd_data['macd'].round(2).tolist()
            chart_data['macd_signal'] = macd_data['signal'].round(2).tolist()
            chart_data['macd_histogram'] = macd_data['histogram'].round(2).tolist()
        
        return {
            "status": "success",
            "symbol": symbol.upper(),
            "chart_data": sanitize_for_json(chart_data)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stock/history/{symbol}")
async def get_stock_history(
    symbol: str,
    period: str = "6mo",
    interval: str = "1d"
):
    """Get historical stock data formatted for frontend chart."""
    try:
        if not validate_symbol(symbol):
            raise HTTPException(status_code=404, detail=f"Symbol '{symbol}' not found")
            
        data = get_stock_data(symbol, period=period, interval=interval)
        
        if data.empty:
            raise HTTPException(status_code=404, detail="No data found for symbol")
            
        # Format for Plotly (Frontend expects this structure)
        # See static/js/app.js: loadChart function
        formatted_data = {
            "dates": data.index.strftime('%Y-%m-%d').tolist(),
            "opens": data['Open'].round(2).tolist(),
            "highs": data['High'].round(2).tolist(),
            "lows": data['Low'].round(2).tolist(),
            "closes": data['Close'].round(2).tolist(),
            "volumes": data['Volume'].tolist()
        }
        
        return {
            "status": "success",
            "data": formatted_data
        }
    except HTTPException:
        raise
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "detail": str(e)}
        )
