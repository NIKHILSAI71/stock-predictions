"""
LSTM-based Time Series Prediction Module

Uses a lightweight pure-NumPy implementation for portability.
Falls back gracefully if TensorFlow is not available.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime
import logging
from .preprocessing_utils import prepare_sequences, normalize_data, denormalize_data, sigmoid, tanh

logger = logging.getLogger(__name__)


class SimpleLSTMCell:
    """
    Simplified LSTM cell using NumPy.
    For inference only - uses pre-computed weights.
    """

    def __init__(self, input_size: int, hidden_size: int):
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Initialize weights with Xavier initialization
        scale = np.sqrt(2.0 / (input_size + hidden_size))

        # Gates: forget, input, cell, output
        self.Wf = np.random.randn(
            input_size + hidden_size, hidden_size) * scale
        self.Wi = np.random.randn(
            input_size + hidden_size, hidden_size) * scale
        self.Wc = np.random.randn(
            input_size + hidden_size, hidden_size) * scale
        self.Wo = np.random.randn(
            input_size + hidden_size, hidden_size) * scale

        self.bf = np.zeros(hidden_size)
        self.bi = np.zeros(hidden_size)
        self.bc = np.zeros(hidden_size)
        self.bo = np.zeros(hidden_size)

    def forward(self, x: np.ndarray, h_prev: np.ndarray, c_prev: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Single forward pass."""
        combined = np.concatenate([x, h_prev], axis=-1)

        f = sigmoid(np.dot(combined, self.Wf) + self.bf)
        i = sigmoid(np.dot(combined, self.Wi) + self.bi)
        c_tilde = tanh(np.dot(combined, self.Wc) + self.bc)
        o = sigmoid(np.dot(combined, self.Wo) + self.bo)

        c = f * c_prev + i * c_tilde
        h = o * tanh(c)

        return h, c


class LSTMPredictor:
    """
    LSTM-based price predictor.
    Uses TensorFlow if available, falls back to NumPy implementation.
    """

    def __init__(self, lookback: int = 60, hidden_size: int = 50):
        self.lookback = lookback
        self.hidden_size = hidden_size
        self.use_tensorflow = False
        self.model = None
        self.min_val = 0
        self.max_val = 1

        # Try importing TensorFlow
        try:
            import tensorflow as tf
            self.tf = tf
            self.use_tensorflow = True
        except ImportError:
            self.tf = None
            self.use_tensorflow = False

    def _build_tf_model(self, input_shape: Tuple[int, int]) -> Any:
        """Build TensorFlow LSTM model."""
        if not self.use_tensorflow:
            return None

        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout

        model = Sequential([
            LSTM(self.hidden_size, return_sequences=True,
                 input_shape=input_shape),
            Dropout(0.2),
            LSTM(self.hidden_size, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(5)  # Predict 5 days
        ])
        model.compile(optimizer='adam', loss='mse')
        return model

    def _numpy_predict(self, sequence: np.ndarray) -> np.ndarray:
        """
        Simple prediction using momentum-weighted average.
        Used when TensorFlow is not available.
        """
        # Use exponential weighted average for prediction
        weights = np.exp(np.linspace(-1, 0, len(sequence)))
        weights /= weights.sum()

        trend = np.polyfit(range(len(sequence)), sequence.flatten(), 1)[0]
        last_price = sequence[-1, 0]

        # Generate predictions with trend continuation
        predictions = []
        for i in range(1, 6):
            pred = last_price + trend * i
            # Add mean reversion factor
            mean_price = np.mean(sequence)
            reversion = 0.1 * (mean_price - pred)
            predictions.append(pred + reversion)

        return np.array(predictions)

    def train(self, prices: np.ndarray, epochs: int = 10, verbose: int = 0) -> Dict[str, Any]:
        """
        Train the LSTM model.

        Args:
            prices: Array of historical prices
            epochs: Training epochs (TensorFlow only)
            verbose: Verbosity level

        Returns:
            Training history/metrics
        """
        if len(prices) < self.lookback + 5:
            return {"error": "Insufficient data for training"}

        # Normalize
        normalized, self.min_val, self.max_val = normalize_data(prices)

        # Prepare sequences
        X, y = prepare_sequences(normalized, self.lookback, forecast_horizon=5)

        if len(X) == 0:
            return {"error": "Could not create training sequences"}

        if self.use_tensorflow:
            try:
                self.model = self._build_tf_model((self.lookback, 1))
                history = self.model.fit(X, y, epochs=epochs, batch_size=32,
                                         validation_split=0.1, verbose=verbose)
                return {
                    "method": "tensorflow",
                    "epochs": epochs,
                    "final_loss": float(history.history['loss'][-1]),
                    "final_val_loss": float(history.history.get('val_loss', [0])[-1])
                }
            except Exception as e:
                self.use_tensorflow = False
                return {"method": "numpy_fallback", "reason": str(e)}

        # NumPy fallback - just store parameters
        return {"method": "numpy", "lookback": self.lookback}

    def predict(self, prices: np.ndarray) -> Dict[str, Any]:
        """
        Predict next 5 days.

        Args:
            prices: Recent price history (at least lookback days)

        Returns:
            Dictionary with predictions and confidence
        """
        if len(prices) < self.lookback:
            return {
                "error": "Insufficient price history",
                "required": self.lookback,
                "provided": len(prices)
            }

        # Take last lookback days
        recent = prices[-self.lookback:]

        # Normalize
        normalized, min_val, max_val = normalize_data(recent)

        # Reshape for prediction
        sequence = normalized.reshape(1, self.lookback, 1)

        # Ensure consistent tensor type to prevent TensorFlow retracing
        sequence = np.asarray(sequence, dtype=np.float32)

        if self.use_tensorflow and self.model is not None:
            try:
                pred_normalized = self.model.predict(sequence, verbose=0)[0]
                predictions = denormalize_data(
                    pred_normalized, min_val, max_val)
            except Exception:
                predictions = denormalize_data(
                    self._numpy_predict(sequence[0]), min_val, max_val
                )
        else:
            predictions = denormalize_data(
                self._numpy_predict(sequence[0]), min_val, max_val
            )

        current_price = prices[-1]

        return {
            "current_price": float(current_price),
            "predictions": {
                "day_1": float(predictions[0]) if not np.isnan(predictions[0]) else float(current_price),
                "day_2": float(predictions[1]) if not np.isnan(predictions[1]) else float(current_price),
                "day_3": float(predictions[2]) if not np.isnan(predictions[2]) else float(current_price),
                "day_4": float(predictions[3]) if not np.isnan(predictions[3]) else float(current_price),
                "day_5": float(predictions[4]) if not np.isnan(predictions[4]) else float(current_price)
            },
            "predicted_change_pct": float((predictions[4] - current_price) / current_price * 100) if not np.isnan(predictions[4]) else 0.0,
            "direction": "Bullish" if predictions[4] > current_price else ("Bearish" if predictions[4] < current_price else "Neutral"),
            "confidence": self._calculate_confidence(prices, predictions),
            "method": "tensorflow" if self.use_tensorflow else "numpy"
        }

    def _calculate_confidence(self, prices: np.ndarray, predictions: np.ndarray) -> float:
        """Calculate prediction confidence based on recent volatility and trend consistency."""
        # Lower confidence if high volatility
        # Use pct_change calculation that doesn't cause broadcast issues
        recent_prices = prices[-31:] if len(prices) >= 31 else prices
        if len(recent_prices) > 1:
            returns = np.diff(recent_prices) / recent_prices[:-1]
            volatility = np.std(returns) if len(returns) > 0 else 0.02
        else:
            volatility = 0.02

        # Base confidence
        base = 70

        # Reduce for high volatility
        vol_penalty = min(30, volatility * 1000)

        # Check trend consistency
        trend = np.polyfit(range(len(prices[-20:])), prices[-20:], 1)[0]
        pred_direction = predictions[-1] - predictions[0]

        if (trend > 0 and pred_direction > 0) or (trend < 0 and pred_direction < 0):
            trend_bonus = 10
        else:
            trend_bonus = -10

        confidence = max(30, min(95, base - vol_penalty + trend_bonus))
        return round(confidence, 1)


def get_lstm_prediction(stock_data: pd.DataFrame, train_epochs: int = 10, symbol: str = None) -> Dict[str, Any]:
    """
    Main function to get LSTM prediction for a stock with caching support.

    Args:
        stock_data: DataFrame with 'Close' column
        train_epochs: Number of training epochs
        symbol: Stock symbol (for caching)

    Returns:
        LSTM prediction results
    """
    from .model_cache import get_model_cache

    if 'Close' not in stock_data.columns:
        return {"error": "No 'Close' column in data"}

    prices = stock_data['Close'].values.astype(float)

    if len(prices) < 100:
        return {"error": "Need at least 100 days of data", "available": len(prices)}

    # Try to load from cache
    cache = get_model_cache()
    cached_model_data = None

    if symbol:
        cached_model_data = cache.get('lstm', symbol, stock_data)

    if cached_model_data is not None and 'model' in cached_model_data:
        # Use cached TensorFlow model
        predictor = LSTMPredictor(lookback=60)
        predictor.model = cached_model_data['model']
        predictor.min_val = cached_model_data.get(
            'scaler_min', prices.min())
        predictor.max_val = cached_model_data.get(
            'scaler_max', prices.max())
        logger.info(f"Using cached LSTM model for {symbol}")
        train_result = {"method": "cached", "epochs": train_epochs}
    else:
        # Train new model
        predictor = LSTMPredictor(lookback=60)
        train_result = predictor.train(
            prices[:-5], epochs=train_epochs, verbose=0)

        # Cache only the model data, not the predictor object
        if symbol and predictor.model is not None:
            cache.set('lstm', symbol, {
                'model': predictor.model,
                'scaler_min': predictor.min_val,
                'scaler_max': predictor.max_val
            }, stock_data, train_result)

    # Predict
    prediction = predictor.predict(prices)
    prediction["training"] = train_result
    prediction["timestamp"] = datetime.now().isoformat()
    prediction["from_cache"] = (cached_model_data is not None)

    return prediction
