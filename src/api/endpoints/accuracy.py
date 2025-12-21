"""
Model Accuracy API Endpoints
Provides model performance metrics, prediction history, and backtesting results.
"""

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from src.core.utils import sanitize_for_json
from src.data import get_stock_data, get_company_info
from src.analysis.quantitative import (
    get_model_accuracy_summary, generate_price_targets, get_backtest_summary,
    calculate_mape, calculate_rmse, get_volatility_forecast
)

router = APIRouter()


@router.get("/model-accuracy/{symbol}")
async def get_model_accuracy(symbol: str):
    """
    Get comprehensive model accuracy metrics for a symbol.
    
    Returns:
        - Per-model accuracy percentages
        - MAPE scores for each model
        - System-wide accuracy
        - Validated prediction count
    """
    try:
        # Get accuracy summary (will use symbol-specific or system-wide)
        accuracy_data = get_model_accuracy_summary(symbol.upper())
        
        return JSONResponse(content={
            "status": "success",
            "symbol": symbol.upper(),
            "accuracy_data": sanitize_for_json(accuracy_data)
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"status": "error", "detail": str(e)})


@router.get("/price-targets/{symbol}")
async def get_price_targets(symbol: str):
    """
    Get multi-timeframe price targets with confidence intervals.
    
    Returns:
        - 7-day, 30-day, 90-day price targets
        - Confidence intervals for each timeframe
        - Direction indicator (Bullish/Bearish/Neutral)
    """
    try:
        # Get stock data for calculations
        stock_data = get_stock_data(symbol, period="1y", interval="1d")
        current_price = stock_data['Close'].iloc[-1]
        
        # Calculate returns for momentum
        returns = stock_data['Close'].pct_change().dropna()
        
        # Get volatility for confidence intervals
        volatility = returns.std() * (252 ** 0.5)  # Annualized
        
        # Calculate trend-based predictions (simple momentum)
        recent_return_7d = stock_data['Close'].iloc[-1] / stock_data['Close'].iloc[-7] - 1 if len(stock_data) >= 7 else 0
        recent_return_30d = stock_data['Close'].iloc[-1] / stock_data['Close'].iloc[-30] - 1 if len(stock_data) >= 30 else 0
        recent_return_90d = stock_data['Close'].iloc[-1] / stock_data['Close'].iloc[-90] - 1 if len(stock_data) >= 90 else 0
        
        # Dampen the predictions (momentum reversion)
        pred_7d = recent_return_7d * 0.3  # Expect 30% continuation
        pred_30d = recent_return_30d * 0.4
        pred_90d = recent_return_90d * 0.5
        
        # Generate targets
        targets = generate_price_targets(
            current_price=current_price,
            predicted_return_7d=pred_7d,
            predicted_return_30d=pred_30d,
            predicted_return_90d=pred_90d,
            volatility=volatility
        )
        
        return JSONResponse(content={
            "status": "success",
            "symbol": symbol.upper(),
            "current_price": round(current_price, 2),
            "volatility": round(volatility * 100, 1),
            "price_targets": sanitize_for_json(targets)
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"status": "error", "detail": str(e)})


@router.get("/backtest-summary/{symbol}")
async def get_backtest_results(symbol: str):
    """
    Get backtesting performance summary.
    
    Returns:
        - Win rate
        - Average gain/loss
        - Profit factor
        - Max drawdown
        - Sharpe ratio
    """
    try:
        backtest_data = get_backtest_summary(symbol.upper())
        
        return JSONResponse(content={
            "status": "success",
            "symbol": symbol.upper(),
            "backtest_summary": sanitize_for_json(backtest_data)
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"status": "error", "detail": str(e)})


@router.get("/prediction-history/{symbol}")
async def get_prediction_history_endpoint(symbol: str):
    """
    Get historical predictions with validation status.
    """
    try:
        from src.analysis.quantitative.model_accuracy import get_prediction_history
        
        all_predictions = get_prediction_history()
        
        # Filter for this symbol
        symbol_predictions = [
            p for p in all_predictions 
            if p.get('symbol', '').upper() == symbol.upper()
        ]
        
        # Get last 50 predictions
        recent = symbol_predictions[-50:] if len(symbol_predictions) > 50 else symbol_predictions
        
        # Calculate stats
        validated = [p for p in symbol_predictions if p.get('validated', False)]
        correct = sum(1 for p in validated if p.get('was_correct', False))
        
        return JSONResponse(content={
            "status": "success",
            "symbol": symbol.upper(),
            "total_predictions": len(symbol_predictions),
            "validated_count": len(validated),
            "accuracy": round(correct / len(validated) * 100, 1) if validated else 0,
            "recent_predictions": sanitize_for_json(recent[-10:])  # Last 10
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"status": "error", "detail": str(e)})
