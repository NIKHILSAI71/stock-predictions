"""
N-BEATS (Neural Basis Expansion Analysis for Time Series)

Published at ICLR 2020, N-BEATS is specifically designed for univariate time series forecasting.

Key advantages:
- Interpretable: Decomposes predictions into trend + seasonality
- No feature engineering needed: Works directly on raw price sequences
- State-of-the-art accuracy for financial forecasting
- Doubly residual stacking for deep architectures

Architecture:
- Generic Stack: Learns general patterns (polynomial basis)
- Trend Stack: Captures trend components
- Seasonality Stack: Captures seasonal/cyclical patterns (Fourier basis)
- Each stack has multiple blocks with backcast + forecast
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List
import logging
import hashlib
from datetime import datetime
import sys

logger = logging.getLogger(__name__)

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    # Handle Keras version differences for serialization
    if hasattr(keras, 'saving'):
        register_serializable = keras.saving.register_keras_serializable
    else:
        register_serializable = keras.utils.register_keras_serializable

    @register_serializable()
    class PolynomialBasis(layers.Layer):
        """Custom layer for polynomial trend basis"""

        def __init__(self, theta_size, output_size, **kwargs):
            super().__init__(**kwargs)
            self.theta_size = theta_size
            self.output_size = output_size

        def get_config(self):
            config = super().get_config()
            config.update({
                "theta_size": self.theta_size,
                "output_size": self.output_size,
            })
            return config

        def call(self, theta):
            # Compute polynomial basis matrix: shape (theta_size, output_size)
            t = tf.linspace(0.0, 1.0, self.output_size)  # (output_size,)
            # (theta_size,)
            powers = tf.range(self.theta_size, dtype=tf.float32)
            # Create basis: t^p for each power p
            # Use meshgrid to get (theta_size, output_size) shape
            # (theta_size, output_size)
            basis = tf.pow(t[None, :], powers[:, None])
            # Matrix multiplication: (batch, theta_size) @ (theta_size, output_size) -> (batch, output_size)
            return tf.matmul(theta, basis)

    @register_serializable()
    class FourierBasis(layers.Layer):
        """Custom layer for seasonality Fourier basis"""

        def __init__(self, theta_size, output_size, **kwargs):
            super().__init__(**kwargs)
            self.theta_size = theta_size
            self.output_size = output_size

        def get_config(self):
            config = super().get_config()
            config.update({
                "theta_size": self.theta_size,
                "output_size": self.output_size,
            })
            return config

        def call(self, theta):
            # Create Fourier basis: shape (theta_size, output_size)
            freqs = tf.range(self.theta_size // 2 + 1, dtype=tf.float32)
            t = tf.linspace(0.0, 1.0, self.output_size)
            # Compute 2π * freq * t for all frequencies and time points
            t_matrix = 2 * np.pi * freqs[:, None] * \
                t[None, :]  # (n_freqs, output_size)
            # Concatenate cos and sin components
            S = tf.concat([tf.cos(t_matrix), tf.sin(t_matrix)], axis=0)[
                :self.theta_size]  # (theta_size, output_size)
            # Matrix multiplication: (batch, theta_size) @ (theta_size, output_size) -> (batch, output_size)
            return tf.matmul(theta, S)

    @register_serializable()
    class GenericBasis(layers.Layer):
        """Custom layer for learned generic basis"""

        def __init__(self, theta_size, output_size, **kwargs):
            super().__init__(**kwargs)
            self.theta_size = theta_size
            self.output_size = output_size

        def get_config(self):
            config = super().get_config()
            config.update({
                "theta_size": self.theta_size,
                "output_size": self.output_size,
            })
            return config

        def build(self, input_shape):
            self.basis_matrix = self.add_weight(
                name='basis_matrix',
                shape=(self.theta_size, self.output_size),
                initializer='random_normal',
                trainable=True
            )

        def call(self, theta):
            # Matrix multiplication: (batch, theta_size) @ (theta_size, output_size)
            return tf.matmul(theta, self.basis_matrix)

    @register_serializable()
    class ZeroInitializer(layers.Layer):
        """Custom layer for zero initialization with proper serialization"""

        def __init__(self, output_size, **kwargs):
            super().__init__(**kwargs)
            self.output_size = output_size

        def get_config(self):
            config = super().get_config()
            config.update({"output_size": self.output_size})
            return config

        def call(self, inputs):
            # Create zero tensor with shape (batch_size, output_size)
            batch_size = tf.shape(inputs)[0]
            return tf.zeros([batch_size, self.output_size], dtype=tf.float32)

        def compute_output_shape(self, input_shape):
            return (input_shape[0], self.output_size)

except ImportError:
    logger.warning("TensorFlow not available. N-BEATS will be disabled.")
    # Dummy classes to avoid NameError if TF is missing but code tries to use them
    PolynomialBasis = object
    FourierBasis = object
    GenericBasis = object
    ZeroInitializer = object


def build_tensorflow_nbeats(
    lookback: int = 60,
    forecast_horizon: int = 5,
    n_blocks_per_stack: int = 4,
    layer_sizes: List[int] = None
) -> Any:
    """
    Build N-BEATS model with TensorFlow/Keras.

    Architecture (from Phase 3 plan):
    - Generic Stack (4 blocks with polynomial basis)
    - Trend Stack (4 blocks with trend basis)
    - Seasonality Stack (4 blocks with Fourier basis)
    - Each block: FC layers [512, 512, 512, 512] → Backcast + Forecast

    Args:
        lookback: Input sequence length (default: 60)
        forecast_horizon: Number of steps to forecast (default: 5)
        n_blocks_per_stack: Blocks per stack (default: 4)
        layer_sizes: Hidden layer sizes (default: [512, 512, 512, 512])

    Returns:
        Compiled Keras model
    """
    try:
        # TensorFlow is already imported at module level
        # Just reference it directly
        if layer_sizes is None:
            layer_sizes = [512, 512, 512, 512]

        # Custom layers are now defined at module level

        # Input
        inputs = layers.Input(shape=(lookback,))
        residual = inputs

        forecasts = []

        # Helper function to create a block
        def create_nbeats_block(x, block_type='generic'):
            """Create a single N-BEATS block."""
            # Fully connected layers
            h = x
            for layer_size in layer_sizes:
                h = layers.Dense(layer_size, activation='relu')(h)

            # Basis parameters
            if block_type == 'trend':
                # Polynomial basis for trend
                theta_b_size = 2 + 1  # Polynomial degree + 1
                theta_f_size = 2 + 1
            elif block_type == 'seasonality':
                # Fourier basis for seasonality
                theta_b_size = forecast_horizon
                theta_f_size = forecast_horizon
            else:  # generic
                theta_b_size = lookback
                theta_f_size = forecast_horizon

            # Backcast and forecast basis coefficients
            theta_b = layers.Dense(theta_b_size, use_bias=False)(h)
            theta_f = layers.Dense(theta_f_size, use_bias=False)(h)

            # Generate backcast and forecast using custom layers
            if block_type == 'trend':
                backcast = PolynomialBasis(theta_b_size, lookback)(theta_b)
                forecast = PolynomialBasis(
                    theta_f_size, forecast_horizon)(theta_f)

            elif block_type == 'seasonality':
                backcast = FourierBasis(theta_b_size, lookback)(theta_b)
                forecast = FourierBasis(
                    theta_f_size, forecast_horizon)(theta_f)

            else:  # generic
                backcast = GenericBasis(theta_b_size, lookback)(theta_b)
                forecast = GenericBasis(
                    theta_f_size, forecast_horizon)(theta_f)

            return backcast, forecast

        # Stack 1: Generic
        stack_residual = residual
        # Initialize forecast accumulator using custom ZeroInitializer layer
        stack_forecast = ZeroInitializer(forecast_horizon)(inputs)

        for _ in range(n_blocks_per_stack):
            backcast, forecast = create_nbeats_block(stack_residual, 'generic')
            stack_residual = layers.Subtract()([stack_residual, backcast])
            stack_forecast = layers.Add()([stack_forecast, forecast])

        residual = stack_residual
        forecasts.append(stack_forecast)

        # Stack 2: Trend
        stack_residual = residual
        stack_forecast = ZeroInitializer(forecast_horizon)(inputs)

        for _ in range(n_blocks_per_stack):
            backcast, forecast = create_nbeats_block(stack_residual, 'trend')
            stack_residual = layers.Subtract()([stack_residual, backcast])
            stack_forecast = layers.Add()([stack_forecast, forecast])

        residual = stack_residual
        forecasts.append(stack_forecast)

        # Stack 3: Seasonality
        stack_residual = residual
        stack_forecast = ZeroInitializer(forecast_horizon)(inputs)

        for _ in range(n_blocks_per_stack):
            backcast, forecast = create_nbeats_block(
                stack_residual, 'seasonality')
            stack_residual = layers.Subtract()([stack_residual, backcast])
            stack_forecast = layers.Add()([stack_forecast, forecast])

        forecasts.append(stack_forecast)

        # Sum all forecasts
        final_forecast = layers.Add()(forecasts)

        model = keras.Model(inputs=inputs, outputs=final_forecast)

        # Compile
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )

        return model

    except ImportError:
        return None
    except Exception as e:
        logger.error(f"Error building N-BEATS model: {e}")
        return None


def get_nbeats_prediction(
    stock_data: pd.DataFrame,
    train_epochs: int = 50,
    symbol: Optional[str] = None
) -> Dict[str, Any]:
    """
    Get N-BEATS prediction for stock price.

    N-BEATS is particularly good for:
    - Pure forecasting (no manual features needed)
    - Interpretable trend + seasonality decomposition
    - Univariate time series

    Args:
        stock_data: DataFrame with OHLCV data
        train_epochs: Number of training epochs (default: 50)
        symbol: Stock symbol for caching

    Returns:
        N-BEATS prediction results
    """
    try:
        from tensorflow import keras
        from src.analysis.quantitative.model_cache import get_model_cache

        if 'Close' not in stock_data.columns:
            return {"error": "No 'Close' column in data", "status": "failed"}

        if len(stock_data) < 100:
            return {
                "error": "Need at least 100 days of data",
                "available": len(stock_data),
                "direction": "Neutral",
                "confidence": 50.0,
                "status": "failed"
            }

        lookback = 60
        forecast_horizon = 5

        # Prepare data (N-BEATS works on raw prices, normalized)
        close_prices = stock_data['Close'].values
        prices_min, prices_max = close_prices.min(), close_prices.max()
        prices_normalized = (close_prices - prices_min) / \
            (prices_max - prices_min + 1e-10)

        # Create sequences
        X, y = [], []
        for i in range(lookback, len(prices_normalized) - forecast_horizon):
            X.append(prices_normalized[i-lookback:i])
            y.append(prices_normalized[i:i+forecast_horizon])

        if len(X) < 50:
            return {
                "error": "Insufficient sequences for training",
                "direction": "Neutral",
                "confidence": 50.0,
                "status": "failed"
            }

        X = np.array(X)
        y = np.array(y)

        # Check cache
        cache = get_model_cache()
        last_date_hash = hashlib.md5(
            str(stock_data.index[-1]).encode()).hexdigest()[:8]
        cache_key = f"{symbol}_{last_date_hash}" if symbol else last_date_hash

        cached_model_data = cache.get('nbeats', cache_key, stock_data)

        if cached_model_data and 'model' in cached_model_data:
            logger.info(f"Using cached N-BEATS model for {symbol}")
            model = cached_model_data['model']
            train_history = cached_model_data.get('history', {})
            from_cache = True
        else:
            # Build model
            model = build_tensorflow_nbeats(
                lookback, forecast_horizon, n_blocks_per_stack=4)

            if model is None:
                return {
                    "error": "Failed to build N-BEATS model",
                    "direction": "Neutral",
                    "confidence": 50.0,
                    "status": "failed"
                }

            # Train-test split
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]

            # Train with early stopping
            early_stop = keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True
            )

            logger.info(
                f"Training N-BEATS model for {symbol} ({len(X_train)} samples)")
            history = model.fit(
                X_train, y_train,
                validation_data=(X_test, y_test),
                epochs=train_epochs,
                batch_size=32,
                callbacks=[early_stop],
                verbose=0
            )

            train_history = {
                'final_loss': float(history.history['loss'][-1]),
                'final_val_loss': float(history.history['val_loss'][-1]),
                'epochs_trained': len(history.history['loss']),
                'samples': len(X_train)
            }

            # Cache model
            cache.set('nbeats', cache_key, {'model': model, 'history': train_history, 'scaler': (
                prices_min, prices_max)}, stock_data, {})
            from_cache = False

        # Make prediction
        latest_sequence = prices_normalized[-lookback:].reshape(1, lookback)
        pred_normalized = model.predict(latest_sequence, verbose=0)[0]

        # Denormalize predictions
        scaler = cached_model_data.get(
            'scaler', (prices_min, prices_max)) if from_cache else (prices_min, prices_max)
        predictions = pred_normalized * (scaler[1] - scaler[0]) + scaler[0]

        # Current price
        current_price = stock_data['Close'].iloc[-1]

        # Direction and confidence
        total_change = (predictions[-1] - current_price) / current_price * 100
        direction = "Bullish" if total_change > 0.5 else (
            "Bearish" if total_change < -0.5 else "Neutral")

        # Confidence based on prediction smoothness (N-BEATS decomposition quality)
        pred_smoothness = 1.0 - \
            np.std(np.diff(predictions)) / (np.std(predictions) + 1e-10)
        market_volatility = stock_data['Close'].pct_change().tail(20).std()
        confidence = max(55, min(88, 75 + pred_smoothness *
                         20 - market_volatility * 400))

        return {
            "current_price": float(current_price),
            "predictions": {
                "day_1": float(predictions[0]),
                "day_2": float(predictions[1]),
                "day_3": float(predictions[2]),
                "day_4": float(predictions[3]),
                "day_5": float(predictions[4])
            },
            "predicted_change_pct": round(total_change, 2),
            "direction": direction,
            "confidence": round(confidence, 1),
            "model_type": "N-BEATS (Neural Basis Expansion Analysis)",
            "method": "tensorflow_nbeats",
            "architecture": {
                "stacks": ["generic", "trend", "seasonality"],
                "blocks_per_stack": 4,
                "layer_sizes": [512, 512, 512, 512],
                "interpretable": True
            },
            "training": train_history,
            "from_cache": from_cache,
            "timestamp": datetime.now().isoformat(),
            "status": "success"
        }

    except ImportError as e:
        logger.error(f"TensorFlow not available for N-BEATS: {e}")
        return {
            "error": "TensorFlow required for N-BEATS model",
            "direction": "Neutral",
            "confidence": 50.0,
            "status": "failed"
        }
    except Exception as e:
        logger.error(f"Error in N-BEATS prediction: {e}", exc_info=True)
        return {
            "error": str(e),
            "direction": "Neutral",
            "confidence": 50.0,
            "status": "failed"
        }
