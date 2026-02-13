"""
Random Forest Predictor for Stock Price Direction
Uses ensemble of decision trees for classification-based prediction.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import logging
import hashlib
from datetime import datetime

logger = logging.getLogger(__name__)


def prepare_rf_features(stock_data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Prepare features for Random Forest model.

    Creates 25+ engineered features including:
    - Price momentum (multiple timeframes)
    - Moving average distances
    - RSI, MACD, Bollinger Band positions
    - Volume ratios and trends
    - Volatility metrics
    - Rate of change indicators
    - Lag features

    Args:
        stock_data: DataFrame with OHLCV data

    Returns:
        Tuple of (features DataFrame, target Series)
    """
    close = stock_data['Close'].copy()
    high = stock_data['High'].copy() if 'High' in stock_data.columns else close
    low = stock_data['Low'].copy() if 'Low' in stock_data.columns else close
    volume = stock_data['Volume'].copy() if 'Volume' in stock_data.columns else pd.Series([
        1] * len(close), index=close.index)

    # Calculate returns
    returns = close.pct_change()

    features = pd.DataFrame(index=stock_data.index)

    # 1. Price Momentum Features (5, 10, 20, 50 days)
    for period in [5, 10, 20, 50]:
        if len(close) > period:
            features[f'momentum_{period}d'] = (
                close - close.shift(period)) / close.shift(period)

    # 2. Moving Average Distance Features
    for period in [5, 10, 20, 50, 200]:
        if len(close) > period:
            sma = close.rolling(period).mean()
            features[f'sma_{period}_dist'] = (close - sma) / sma

    # 3. RSI (Relative Strength Index - 14 period)
    if len(returns) > 14:
        gains = returns.copy()
        gains[gains < 0] = 0
        losses = -returns.copy()
        losses[losses < 0] = 0

        avg_gain = gains.rolling(14).mean()
        avg_loss = losses.rolling(14).mean()

        rs = avg_gain / (avg_loss + 1e-10)
        features['rsi'] = 100 - (100 / (1 + rs))

    # 4. MACD (12, 26, 9)
    if len(close) > 26:
        ema_12 = close.ewm(span=12, adjust=False).mean()
        ema_26 = close.ewm(span=26, adjust=False).mean()
        macd_line = ema_12 - ema_26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        features['macd'] = macd_line
        features['macd_signal'] = signal_line
        features['macd_histogram'] = macd_line - signal_line

    # 5. Bollinger Bands (20-period, 2 std dev)
    if len(close) > 20:
        sma_20 = close.rolling(20).mean()
        std_20 = close.rolling(20).std()
        upper_band = sma_20 + (2 * std_20)
        lower_band = sma_20 - (2 * std_20)
        features['bb_position'] = (
            close - lower_band) / (upper_band - lower_band + 1e-10)
        features['bb_width'] = (upper_band - lower_band) / sma_20

    # 6. Volume Features
    if len(volume) > 20:
        features['volume_ratio'] = volume / volume.rolling(20).mean()
        features['volume_trend'] = volume.rolling(
            5).mean() / volume.rolling(20).mean()

    # 7. Volatility Features (5, 20, 60 days)
    for period in [5, 20, 60]:
        if len(returns) > period:
            features[f'volatility_{period}d'] = returns.rolling(period).std()

    # 8. Rate of Change (5, 10, 20)
    for period in [5, 10, 20]:
        if len(close) > period:
            features[f'roc_{period}d'] = (
                close - close.shift(period)) / close.shift(period)

    # 9. Lag Features (1, 2, 3, 5, 10 days)
    features['returns'] = returns
    for lag in [1, 2, 3, 5, 10]:
        features[f'returns_lag{lag}'] = returns.shift(lag)
        if len(volume) > lag:
            features[f'volume_lag{lag}'] = (
                volume / volume.rolling(20).mean()).shift(lag)

    # 10. Price Range Features
    if 'High' in stock_data.columns and 'Low' in stock_data.columns:
        features['daily_range'] = (high - low) / close
        features['range_10d_avg'] = features['daily_range'].rolling(10).mean()

    # Create target: 1 if price goes up in next 5 days, 0 if down
    features['target'] = (close.shift(-5) > close).astype(int)

    # Drop NaN rows
    features = features.dropna()

    if len(features) < 50:
        raise ValueError(
            f"Insufficient clean data: {len(features)} rows (need at least 50)")

    # Separate features and target
    target = features['target']
    feature_cols = [col for col in features.columns if col != 'target']
    features = features[feature_cols]

    return features, target


def get_rf_prediction(
    stock_data: pd.DataFrame,
    symbol: Optional[str] = None,
    n_estimators: int = 200,
    max_depth: int = 15
) -> Dict[str, Any]:
    """
    Get Random Forest prediction for stock price direction.

    Uses cached models when available for performance.

    Args:
        stock_data: DataFrame with OHLCV data
        symbol: Stock symbol for caching
        n_estimators: Number of trees (default: 200)
        max_depth: Maximum tree depth (default: 15)

    Returns:
        Dictionary with prediction results
    """
    try:
        from src.analysis.quantitative.ml_models import train_random_forest
        from src.analysis.quantitative.model_cache import get_model_cache
        import sklearn

        # Prepare features
        try:
            features, target = prepare_rf_features(stock_data)
        except ValueError as e:
            return {
                "error": str(e),
                "direction": "Neutral",
                "confidence": 50,
                "status": "failed"
            }

        # Generate cache key with feature information to avoid mismatches
        cache = get_model_cache()
        feature_cols_hash = hashlib.md5(
            ('_'.join(sorted(features.columns))).encode()).hexdigest()[:8]
        last_date_hash = hashlib.md5(
            str(stock_data.index[-1]).encode()).hexdigest()[:8]
        cache_key = f"{symbol}_{last_date_hash}_{feature_cols_hash}" if symbol else f"{last_date_hash}_{feature_cols_hash}"

        # Try to get cached model
        cached_model_data = cache.get('random_forest', cache_key, stock_data)

        if cached_model_data and 'model' in cached_model_data:
            logger.info(f"Using cached Random Forest model for {symbol}")
            rf_result = cached_model_data
            from_cache = True
        else:
            # Train new model
            logger.info(
                f"Training new Random Forest model for {symbol} with {len(features)} samples")
            rf_result = train_random_forest(
                features,
                target,
                n_estimators=n_estimators
            )

            if rf_result.get('status') == 'success' and rf_result.get('model'):
                # Cache the trained model
                cache.set('random_forest', cache_key,
                          rf_result, stock_data, {})
                from_cache = False
            else:
                return {
                    "error": rf_result.get('error', 'Training failed'),
                    "direction": "Neutral",
                    "confidence": 50,
                    "status": "failed"
                }

        # Make prediction on latest data
        model = rf_result['model']
        scaler = rf_result.get('scaler')

        # Get last row of features
        last_features = features.iloc[[-1]].copy()

        # Scale if scaler available
        if scaler:
            last_features_scaled = pd.DataFrame(
                scaler.transform(last_features.fillna(0)),
                columns=last_features.columns,
                index=last_features.index
            )
        else:
            last_features_scaled = last_features.fillna(0)

        # Predict
        prediction = model.predict(last_features_scaled)[0]
        probabilities = model.predict_proba(last_features_scaled)[0]

        # Get confidence (max probability)
        confidence = max(probabilities) * 100

        # Direction
        direction = "Bullish" if prediction == 1 else "Bearish"

        # Calculate expected change based on historical volatility
        recent_returns = stock_data['Close'].pct_change().tail(20)
        avg_volatility = recent_returns.std()
        expected_change_pct = avg_volatility * \
            100 * (1 if prediction == 1 else -1)

        # Get current price
        current_price = stock_data['Close'].iloc[-1]

        # Get feature importance (top 10)
        feature_importance = rf_result.get('feature_importance', {})
        if feature_importance:
            # Sort by importance and take top 10
            sorted_importance = sorted(
                feature_importance.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]
            top_features = dict(sorted_importance)
        else:
            top_features = {}

        return {
            "direction": direction,
            "confidence": round(confidence, 1),
            "predicted_change_pct": round(expected_change_pct, 2),
            "current_price": float(current_price),
            "prediction_value": int(prediction),
            "probabilities": {
                "down": round(float(probabilities[0]) * 100, 1),
                "up": round(float(probabilities[1]) * 100, 1)
            },
            "model_metrics": {
                "accuracy_train": rf_result.get('accuracy_train', 0),
                "accuracy_test": rf_result.get('accuracy_test', 0),
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "training_samples": len(features)
            },
            "feature_importance": top_features,
            "from_cache": from_cache,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "method": "sklearn_random_forest",
            "status": "success"
        }

    except ImportError as e:
        logger.error(f"Import error in Random Forest predictor: {e}")
        return {
            "error": "sklearn or dependencies not installed",
            "direction": "Neutral",
            "confidence": 50,
            "status": "failed"
        }
    except Exception as e:
        logger.error(f"Error in Random Forest prediction: {e}", exc_info=True)
        return {
            "error": str(e),
            "direction": "Neutral",
            "confidence": 50,
            "status": "failed"
        }
