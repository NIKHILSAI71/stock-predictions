from fastapi import APIRouter
from fastapi.responses import JSONResponse
import logging
import numpy as np
from datetime import datetime

from src.core.utils import sanitize_for_json
from src.data import get_stock_data, validate_symbol

logger = logging.getLogger(__name__)
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
                content={"status": "error",
                         "detail": f"Symbol '{symbol}' not found."}
            )

        data = get_stock_data(symbol, period, "1d")

        # Calculate basic risk metrics locally since module was removed
        returns = data['Close'].pct_change().dropna()
        volatility = returns.std() * (252 ** 0.5)

        risk_metrics = {
            "volatility": volatility,
            "message": "Advanced risk metrics module currently unavailable."
        }

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
async def get_forecast(symbol: str, steps: int = 5, models: str = "lstm,xgboost,gru"):
    """
    Get multi-model forecast for stock price.

    Query Parameters:
        steps: Number of days to forecast (default: 5, max: 30)
        models: Comma-separated list of models to use
                (options: lstm, xgboost, gru, cnn_lstm, attention)

    Returns:
        - Aggregated forecast (average across models)
        - Individual model forecasts
        - Consensus direction and confidence
    """
    try:
        # Validate inputs
        if not validate_symbol(symbol):
            return JSONResponse(
                status_code=404,
                content={"status": "error",
                         "detail": f"Symbol '{symbol}' not found."},
            )

        if steps < 1 or steps > 30:
            return JSONResponse(
                status_code=400,
                content={"status": "error",
                         "detail": "steps must be between 1 and 30"},
            )

        # Get stock data
        stock_data = get_stock_data(symbol, period="2y", interval="1d")

        if len(stock_data) < 100:
            return JSONResponse(
                status_code=400,
                content={
                    "status": "error",
                    "message": "Insufficient historical data for forecasting",
                },
            )

        current_price = float(stock_data['Close'].iloc[-1])

        # Import model functions
        from src.analysis.quantitative import (
            get_lstm_prediction,
            get_xgboost_prediction,
            get_gru_prediction,
            get_cnn_lstm_prediction,
            get_attention_prediction,
        )

        # Parse requested models
        model_list = [m.strip().lower() for m in models.split(',')]
        forecasts = {}
        model_errors = {}

        logger.info(
            f"Running forecast for {symbol} using models: {model_list}")

        # Run requested models
        if 'lstm' in model_list:
            try:
                lstm_result = get_lstm_prediction(
                    stock_data, prediction_horizon=steps, symbol=symbol
                )
                if lstm_result and 'predictions' in lstm_result:
                    forecasts['LSTM'] = {
                        "predictions": lstm_result['predictions'][:steps],
                        "direction": lstm_result.get('direction', 'Neutral'),
                        "confidence": lstm_result.get('confidence', 50.0),
                    }
                    logger.debug(
                        f"LSTM forecast: {forecasts['LSTM']['direction']}")
            except Exception as e:
                logger.warning(f"LSTM forecast failed: {e}")
                model_errors['LSTM'] = str(e)[:100]

        if 'xgboost' in model_list:
            try:
                xgb_result = get_xgboost_prediction(
                    stock_data, prediction_horizon=steps, symbol=symbol
                )
                if xgb_result and 'predictions' in xgb_result:
                    forecasts['XGBoost'] = {
                        "predictions": xgb_result['predictions'][:steps],
                        "direction": xgb_result.get('direction', 'Neutral'),
                        "confidence": xgb_result.get('confidence', 50.0),
                    }
                    logger.debug(
                        f"XGBoost forecast: {forecasts['XGBoost']['direction']}")
            except Exception as e:
                logger.warning(f"XGBoost forecast failed: {e}")
                model_errors['XGBoost'] = str(e)[:100]

        if 'gru' in model_list:
            try:
                gru_result = get_gru_prediction(
                    stock_data, prediction_horizon=steps, symbol=symbol
                )
                if gru_result and 'predictions' in gru_result:
                    forecasts['GRU'] = {
                        "predictions": gru_result['predictions'][:steps],
                        "direction": gru_result.get('direction', 'Neutral'),
                        "confidence": gru_result.get('confidence', 50.0),
                    }
                    logger.debug(
                        f"GRU forecast: {forecasts['GRU']['direction']}")
            except Exception as e:
                logger.warning(f"GRU forecast failed: {e}")
                model_errors['GRU'] = str(e)[:100]

        if 'cnn_lstm' in model_list:
            try:
                cnn_result = get_cnn_lstm_prediction(
                    stock_data, train_epochs=3)
                if cnn_result and 'predictions' in cnn_result:
                    forecasts['CNN-LSTM'] = {
                        "predictions": cnn_result['predictions'][:steps],
                        "direction": cnn_result.get('direction', 'Neutral'),
                        "confidence": cnn_result.get('confidence', 50.0),
                    }
                    logger.debug(
                        f"CNN-LSTM forecast: {forecasts['CNN-LSTM']['direction']}")
            except Exception as e:
                logger.warning(f"CNN-LSTM forecast failed: {e}")
                model_errors['CNN-LSTM'] = str(e)[:100]

        if 'attention' in model_list:
            try:
                att_result = get_attention_prediction(
                    stock_data, train_epochs=3)
                if att_result and 'predictions' in att_result:
                    forecasts['Attention'] = {
                        "predictions": att_result['predictions'][:steps],
                        "direction": att_result.get('direction', 'Neutral'),
                        "confidence": att_result.get('confidence', 50.0),
                    }
                    logger.debug(
                        f"Attention forecast: {forecasts['Attention']['direction']}")
            except Exception as e:
                logger.warning(f"Attention forecast failed: {e}")
                model_errors['Attention'] = str(e)[:100]

        if not forecasts:
            return JSONResponse(
                status_code=500,
                content={
                    "status": "error",
                    "message": "All forecasting models failed",
                    "errors": model_errors,
                },
            )

        # Aggregate forecasts (average predictions across models)
        aggregated = []
        for step in range(steps):
            step_predictions = []
            for model_forecast in forecasts.values():
                if len(model_forecast['predictions']) > step:
                    step_predictions.append(
                        model_forecast['predictions'][step])

            if step_predictions:
                aggregated.append(float(np.mean(step_predictions)))
            else:
                # If no predictions for this step, extrapolate from last available
                if aggregated:
                    aggregated.append(aggregated[-1])
                else:
                    aggregated.append(current_price)

        # Calculate consensus direction
        directions = [f['direction']
                      for f in forecasts.values() if f.get('direction')]
        bullish_count = sum(1 for d in directions if d == 'Bullish')
        bearish_count = sum(1 for d in directions if d == 'Bearish')

        if bullish_count > bearish_count:
            consensus = "Bullish"
        elif bearish_count > bullish_count:
            consensus = "Bearish"
        else:
            consensus = "Neutral"

        # Calculate consensus confidence (weighted by model agreement)
        agreement_ratio = (
            max(bullish_count, bearish_count) /
            len(directions) if directions else 0.5
        )
        avg_confidence = sum(
            f['confidence'] for f in forecasts.values() if f.get('confidence', 0)
        ) / len(forecasts)
        consensus_confidence = avg_confidence * agreement_ratio

        logger.info(
            f"Forecast complete: {len(forecasts)} models, "
            f"consensus={consensus} ({consensus_confidence:.1f}% confidence)"
        )

        return {
            "status": "success",
            "symbol": symbol.upper(),
            "current_price": round(current_price, 2),
            "forecast": {
                "steps": steps,
                "predictions": [round(p, 2) for p in aggregated],
                "consensus_direction": consensus,
                "consensus_confidence": round(consensus_confidence, 1),
                "model_agreement": f"{max(bullish_count, bearish_count)}/{len(directions)}"
                if directions
                else "N/A",
            },
            "individual_models": {
                name: {
                    "predictions": [round(p, 2) for p in data['predictions']],
                    "direction": data['direction'],
                    "confidence": round(data['confidence'], 1),
                }
                for name, data in forecasts.items()
            },
            "models_used": list(forecasts.keys()),
            "model_errors": model_errors if model_errors else None,
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"Forecast error for {symbol}: {e}", exc_info=True)
        return JSONResponse(
            status_code=500, content={"status": "error", "detail": str(e)}
        )
