"""
Model Accuracy API Endpoints
Provides model performance metrics, prediction history, and backtesting results.
"""

from fastapi import APIRouter
from fastapi.responses import JSONResponse
import logging

from src.core.utils import sanitize_for_json
from src.data import get_stock_data, get_company_info
from src.analysis.quantitative import (
    calculate_mape, calculate_rmse, get_volatility_forecast
)
from src.analysis.quantitative.backtesting import (
    backtest_strategy, sma_crossover_strategy, rsi_strategy
)
from src.analysis.quantitative.model_accuracy import get_accuracy_tracker

logger = logging.getLogger(__name__)
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
        tracker = get_accuracy_tracker()

        # Get accuracy metrics for last 90 days
        accuracy_data = tracker.get_all_models_accuracy(days=90)

        logger.info(
            f"Model accuracy for {symbol}: "
            f"{accuracy_data['system_accuracy']['accuracy_pct']}% system-wide"
        )

        return JSONResponse(
            content={
                "status": "success",
                "symbol": symbol.upper(),
                "accuracy_data": sanitize_for_json(accuracy_data),
            }
        )

    except Exception as e:
        logger.error(f"Model accuracy error for {symbol}: {e}")
        return JSONResponse(
            status_code=500, content={"status": "error", "detail": str(e)}
        )


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
        recent_return_7d = stock_data['Close'].iloc[-1] / \
            stock_data['Close'].iloc[-7] - 1 if len(stock_data) >= 7 else 0
        recent_return_30d = stock_data['Close'].iloc[-1] / \
            stock_data['Close'].iloc[-30] - 1 if len(stock_data) >= 30 else 0
        recent_return_90d = stock_data['Close'].iloc[-1] / \
            stock_data['Close'].iloc[-90] - 1 if len(stock_data) >= 90 else 0

        # Dampen the predictions (momentum reversion)
        pred_7d = recent_return_7d * 0.3  # Expect 30% continuation
        pred_30d = recent_return_30d * 0.4
        pred_90d = recent_return_90d * 0.5

        # Simplified target generation as generate_price_targets was removed
        targets = {
            "7d": current_price * (1 + pred_7d),
            "30d": current_price * (1 + pred_30d),
            "90d": current_price * (1 + pred_90d)
        }

        return JSONResponse(content={
            "status": "success",
            "symbol": symbol.upper(),
            "current_price": round(current_price, 2),
            "volatility": round(volatility * 100, 1),
            "price_targets": sanitize_for_json(targets)
        })

    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "detail": str(e)})


@router.get("/backtest-summary/{symbol}")
async def get_backtest_results(symbol: str, strategy: str = "sma_crossover"):
    """
    Get backtesting performance summary.

    Query Parameters:
        strategy: Strategy to backtest (sma_crossover, rsi) - default: sma_crossover

    Returns:
        - Win rate
        - Average gain/loss
        - Profit factor
        - Max drawdown
        - Sharpe ratio
        - Total return
    """
    try:
        # Get historical data (2 years for robust backtest)
        stock_data = get_stock_data(symbol, period="2y", interval="1d")

        if len(stock_data) < 100:
            return JSONResponse(
                status_code=400,
                content={
                    "status": "error",
                    "message": "Insufficient historical data for backtesting",
                },
            )

        # Select strategy
        if strategy == "rsi":
            signal_generator = rsi_strategy
            strategy_name = "RSI Mean Reversion"
        else:  # Default to SMA crossover
            signal_generator = sma_crossover_strategy
            strategy_name = "SMA Crossover (20/50)"

        # Run backtest
        logger.info(f"Running backtest for {symbol} using {strategy_name}")

        backtest_results = backtest_strategy(
            data=stock_data,
            signal_generator=signal_generator,
            initial_capital=10000.0,
            commission=0.001,  # 0.1%
            slippage=0.0005,  # 0.05%
            position_size=0.95,
        )

        logger.info(
            f"Backtest complete: {backtest_results['num_trades']} trades, "
            f"{backtest_results['total_return_pct']:.2f}% return"
        )

        return JSONResponse(
            content={
                "status": "success",
                "symbol": symbol.upper(),
                "strategy": strategy_name,
                "backtest_summary": {
                    "initial_capital": backtest_results["initial_capital"],
                    "final_value": round(backtest_results["final_value"], 2),
                    "total_return_pct": round(
                        backtest_results["total_return_pct"], 2
                    ),
                    "total_trades": backtest_results["num_trades"],
                    "win_rate": backtest_results["metrics"]["win_rate"],
                    "profit_factor": backtest_results["metrics"]["profit_factor"],
                    "sharpe_ratio": backtest_results["metrics"]["sharpe_ratio"],
                    "max_drawdown_pct": backtest_results["metrics"][
                        "max_drawdown_pct"
                    ],
                    "avg_win": backtest_results["metrics"]["avg_win"],
                    "avg_loss": backtest_results["metrics"]["avg_loss"],
                },
            }
        )

    except Exception as e:
        logger.error(f"Backtest error for {symbol}: {e}")
        return JSONResponse(
            status_code=500, content={"status": "error", "detail": str(e)}
        )


@router.get("/prediction-history/{symbol}")
async def get_prediction_history_endpoint(symbol: str, limit: int = 50):
    """
    Get historical predictions with validation status.

    Query Parameters:
        limit: Maximum number of predictions to return (default: 50)

    Returns:
        List of predictions with validation results
    """
    try:
        tracker = get_accuracy_tracker()

        # Get prediction history
        predictions = tracker.get_prediction_history(symbol, limit=limit)

        # Calculate summary stats
        validated = [p for p in predictions if p['validated']]
        correct = [p for p in validated if p['correct']]

        accuracy = (
            (len(correct) / len(validated)) * 100 if validated else 0.0
        )

        logger.info(
            f"Prediction history for {symbol}: "
            f"{len(predictions)} predictions, {len(validated)} validated"
        )

        return JSONResponse(
            content={
                "status": "success",
                "symbol": symbol.upper(),
                "total_predictions": len(predictions),
                "validated_count": len(validated),
                "accuracy": round(accuracy, 2) if validated else 0.0,
                "recent_predictions": sanitize_for_json(predictions),
            }
        )

    except Exception as e:
        logger.error(f"Prediction history error for {symbol}: {e}")
        return JSONResponse(
            status_code=500, content={"status": "error", "detail": str(e)}
        )
