from fastapi import APIRouter
from fastapi.responses import JSONResponse

from src.core.utils import sanitize_for_json
from src.data import get_stock_data, validate_symbol
from src.analysis.quantitative import (
    comprehensive_risk_analysis,
    arima_forecast
)

router = APIRouter()

@router.get("/risk/{symbol}")
async def get_risk_analysis(
    symbol: str,
    period: str = "1y"
):
    """Get comprehensive risk metrics for a stock."""
    try:
        if not validate_symbol(symbol):
            return JSONResponse(
                status_code=404,
                content={"status": "error", "detail": f"Symbol '{symbol}' not found."}
            )
        
        data = get_stock_data(symbol, period, "1d")
        
        # Clean data: Remove zero/negative prices causing massive % change spikes
        data = data[data['Close'] > 0]
        
        returns = data['Close'].pct_change().dropna()
        
        # Filter extreme outliers (e.g., > 500% daily move implies data bad tick)
        returns = returns[returns.abs() < 5.0] 
        
        # Get benchmark (SPY) for comparison
        try:
            benchmark_data = get_stock_data("SPY", period, "1d")
            # Same cleanup for benchmark
            benchmark_data = benchmark_data[benchmark_data['Close'] > 0]
            benchmark_returns = benchmark_data['Close'].pct_change().dropna()
            benchmark_returns = benchmark_returns[benchmark_returns.abs() < 5.0]
            
            risk_metrics = comprehensive_risk_analysis(returns, benchmark_returns)
        except:
            risk_metrics = comprehensive_risk_analysis(returns)
        
        return {
            "status": "success",
            "symbol": symbol.upper(),
            "risk_analysis": sanitize_for_json(risk_metrics)
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "detail": str(e)}
        )

@router.get("/forecast/{symbol}")
async def get_forecast(
    symbol: str,
    steps: int = 5
):
    """Get ARIMA forecast for stock price."""
    try:
        data = get_stock_data(symbol, "2y", "1d")
        forecast = arima_forecast(data['Close'], steps=steps)
        return {
            "status": "success",
            "symbol": symbol.upper(),
            "forecast": sanitize_for_json(forecast)
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "detail": str(e)})
