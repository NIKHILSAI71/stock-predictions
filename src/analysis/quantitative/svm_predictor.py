"""
Support Vector Machine (SVM) Predictor for Stock Price Direction
Uses Support Vector Classification with RBF kernel for non-linear pattern recognition.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import logging
import hashlib
from datetime import datetime

logger = logging.getLogger(__name__)


def prepare_svm_features(stock_data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Prepare features for SVM model.

    Uses the same 25+ features as Random Forest for consistency.
    Features will be scaled using StandardScaler (critical for SVM).

    Args:
        stock_data: DataFrame with OHLCV data

    Returns:
        Tuple of (features DataFrame, target Series)
    """
    # Import the shared feature preparation
    from src.analysis.quantitative.random_forest_predictor import prepare_rf_features

    # Use same features as Random Forest
    return prepare_rf_features(stock_data)


def get_svm_prediction(
    stock_data: pd.DataFrame,
    symbol: Optional[str] = None,
    kernel: str = 'rbf',
    C: float = 1.0
) -> Dict[str, Any]:
    """
    Get SVM prediction for stock price direction.

    SVM is particularly good at finding complex non-linear decision boundaries.
    Uses RBF kernel by default for non-linear pattern recognition.

    Args:
        stock_data: DataFrame with OHLCV data
        symbol: Stock symbol for caching
        kernel: SVM kernel ('rbf', 'poly', 'linear') - default 'rbf'
        C: Regularization parameter (default: 1.0)

    Returns:
        Dictionary with prediction results
    """
    try:
        from src.analysis.quantitative.ml_models import train_svm
        from src.analysis.quantitative.model_cache import get_model_cache
        import sklearn

        # Prepare features
        try:
            features, target = prepare_svm_features(stock_data)
        except ValueError as e:
            return {
                "error": str(e),
                "direction": "Neutral",
                "confidence": 50,
                "status": "failed"
            }

        # Generate cache key
        cache = get_model_cache()
        last_date_hash = hashlib.md5(
            str(stock_data.index[-1]).encode()).hexdigest()[:8]
        cache_key = f"{symbol}_{last_date_hash}_{kernel}" if symbol else f"{last_date_hash}_{kernel}"

        # Try to get cached model
        cached_model_data = cache.get('svm', cache_key, stock_data)

        if cached_model_data and 'model' in cached_model_data:
            logger.info(f"Using cached SVM model for {symbol}")
            svm_result = cached_model_data
            from_cache = True
        else:
            # Train new model
            logger.info(
                f"Training new SVM model for {symbol} with {len(features)} samples (kernel={kernel})")
            svm_result = train_svm(
                features,
                target,
                kernel=kernel
            )

            if svm_result.get('status') == 'success' and svm_result.get('model'):
                # Cache the trained model
                cache.set('svm', cache_key,
                          svm_result, stock_data, {})
                from_cache = False
            else:
                return {
                    "error": svm_result.get('error', 'Training failed'),
                    "direction": "Neutral",
                    "confidence": 50,
                    "status": "failed"
                }

        # Make prediction on latest data
        model = svm_result['model']
        scaler = svm_result.get('scaler')

        # Get last row of features
        last_features = features.iloc[[-1]].copy()

        # SVM REQUIRES scaling - scaler should always be present
        if scaler:
            last_features_scaled = pd.DataFrame(
                scaler.transform(last_features.fillna(0)),
                columns=last_features.columns,
                index=last_features.index
            )
        else:
            logger.warning("SVM scaler not found - results may be suboptimal")
            # Fallback: manual standardization
            from sklearn.preprocessing import StandardScaler
            temp_scaler = StandardScaler()
            temp_scaler.fit(features.fillna(0))
            last_features_scaled = pd.DataFrame(
                temp_scaler.transform(last_features.fillna(0)),
                columns=last_features.columns,
                index=last_features.index
            )

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

        # Get decision function (distance from hyperplane)
        try:
            decision_function = model.decision_function(
                last_features_scaled)[0]
            # Normalize to 0-100 scale
            decision_confidence = min(
                100, max(0, 50 + decision_function * 10))
        except Exception as e:
            logger.warning(f"Could not compute decision function: {e}")
            decision_function = None
            decision_confidence = confidence

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
            "decision_function": round(float(decision_function), 4) if decision_function is not None else None,
            "decision_confidence": round(decision_confidence, 1),
            "model_metrics": {
                "accuracy": svm_result.get('accuracy', 0),
                "kernel": kernel,
                "C": C,
                "training_samples": len(features)
            },
            "from_cache": from_cache,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "method": "sklearn_svm",
            "status": "success"
        }

    except ImportError as e:
        logger.error(f"Import error in SVM predictor: {e}")
        return {
            "error": "sklearn or dependencies not installed",
            "direction": "Neutral",
            "confidence": 50,
            "status": "failed"
        }
    except Exception as e:
        logger.error(f"Error in SVM prediction: {e}", exc_info=True)
        return {
            "error": str(e),
            "direction": "Neutral",
            "confidence": 50,
            "status": "failed"
        }
