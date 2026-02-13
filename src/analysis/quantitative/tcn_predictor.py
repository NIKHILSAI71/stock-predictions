"""
Temporal Convolutional Network (TCN) for Stock Price Prediction

TCN advantages over LSTM/GRU (2024 research):
- 50% faster training (parallelizable convolutions)
- Longer effective history (exponential receptive field: 2^n)
- No vanishing gradient issues
- Captures multiple time scales simultaneously
- Often matches or exceeds LSTM accuracy

Architecture:
- Dilated causal convolutions with increasing dilation rates
- Residual connections for deep networks
- Layer normalization and dropout for regularization
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List
import logging
import hashlib
from datetime import datetime

logger = logging.getLogger(__name__)


def build_tensorflow_tcn(lookback: int, n_features: int, n_blocks: int = 4, n_filters: List[int] = None, kernel_size: int = 3) -> Any:
    """
    Build Temporal Convolutional Network with TensorFlow/Keras.

    Architecture (from Phase 3 plan):
    - 4 residual TCN blocks with dilation rates [1, 2, 4, 8]
    - Filters progression: [64, 128, 256, 512]
    - Causal convolutions (no future information leakage)
    - Residual connections + dropout

    Args:
        lookback: Sequence length
        n_features: Number of input features
        n_blocks: Number of TCN blocks (default: 4)
        n_filters: Filters per block (default: [64, 128, 256, 512])
        kernel_size: Convolution kernel size (default: 3)

    Returns:
        Compiled Keras model
    """
    try:
        from tensorflow import keras
        from tensorflow.keras import layers

        if n_filters is None:
            n_filters = [64, 128, 256, 512]

        # Input
        inputs = layers.Input(shape=(lookback, n_features))
        x = inputs

        # TCN blocks with increasing dilation
        for i in range(n_blocks):
            dilation_rate = 2 ** i  # 1, 2, 4, 8
            filters = n_filters[i] if i < len(n_filters) else n_filters[-1]

            # Residual block
            residual = x

            # Dilated causal conv 1
            x = layers.Conv1D(
                filters=filters,
                kernel_size=kernel_size,
                dilation_rate=dilation_rate,
                padding='causal',
                activation='relu'
            )(x)
            x = layers.LayerNormalization()(x)
            x = layers.Dropout(0.2)(x)

            # Dilated causal conv 2
            x = layers.Conv1D(
                filters=filters,
                kernel_size=kernel_size,
                dilation_rate=dilation_rate,
                padding='causal',
                activation='relu'
            )(x)
            x = layers.LayerNormalization()(x)
            x = layers.Dropout(0.2)(x)

            # Match dimensions for residual connection
            if residual.shape[-1] != filters:
                residual = layers.Conv1D(filters, 1, padding='same')(residual)

            # Add residual
            x = layers.Add()([x, residual])
            x = layers.Activation('relu')(x)

        # Global pooling and output
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        outputs = layers.Dense(5)(x)  # 5-day predictions

        model = keras.Model(inputs=inputs, outputs=outputs)

        # Compile
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )

        return model
    except ImportError:
        return None


def get_tcn_prediction(
    stock_data: pd.DataFrame,
    train_epochs: int = 50,
    symbol: Optional[str] = None
) -> Dict[str, Any]:
    """
    Get TCN prediction for stock price direction.

    TCN is particularly good for:
    - Long-range dependencies (receptive field = 2^n)
    - Fast training (parallelizable)
    - Multiple time scale patterns

    Args:
        stock_data: DataFrame with OHLCV data
        train_epochs: Number of training epochs (default: 50)
        symbol: Stock symbol for caching

    Returns:
        TCN prediction results
    """
    try:
        from tensorflow import keras
        from src.analysis.quantitative.model_cache import get_model_cache
        from src.analysis.quantitative.cnn_lstm_hybrid import prepare_cnn_features

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
        n_features = 10

        # Prepare features
        features = prepare_cnn_features(
            stock_data.tail(len(stock_data)), lookback)

        # Create sequences
        X, y = [], []
        close_prices = stock_data['Close'].values

        for i in range(lookback, len(features) - 5):
            X.append(features[i-lookback:i])
            future_prices = close_prices[i:i+5]
            current_price = close_prices[i-1]
            y.append((future_prices - current_price) / current_price)

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

        cached_model_data = cache.get('tcn', cache_key, stock_data)

        if cached_model_data and 'model' in cached_model_data:
            logger.info(f"Using cached TCN model for {symbol}")
            model = cached_model_data['model']
            train_history = cached_model_data.get('history', {})
            from_cache = True
        else:
            # Build model
            model = build_tensorflow_tcn(
                lookback, n_features, n_blocks=4, n_filters=[64, 128, 256, 512])

            # Train-test split
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]

            # Train with early stopping
            early_stop = keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            )

            logger.info(
                f"Training TCN model for {symbol} ({len(X_train)} samples)")
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
            cache.set('tcn', cache_key, {
                      'model': model, 'history': train_history}, stock_data, {})
            from_cache = False

        # Make prediction
        latest_features = features[-lookback:].reshape(1, lookback, n_features)

        # Ensure consistent tensor type to prevent TensorFlow retracing
        latest_features = np.asarray(latest_features, dtype=np.float32)

        pred_normalized = model.predict(latest_features, verbose=0)[0]

        # Convert to actual prices
        current_price = stock_data['Close'].iloc[-1]
        predictions = current_price * (1 + pred_normalized)

        # Direction and confidence
        total_change = (predictions[-1] - current_price) / current_price * 100
        direction = "Bullish" if total_change > 0.5 else (
            "Bearish" if total_change < -0.5 else "Neutral")

        # Confidence based on prediction consistency
        pred_volatility = np.std(pred_normalized)
        market_volatility = stock_data['Close'].pct_change().tail(20).std()
        confidence = max(50, min(90, 75 - pred_volatility *
                         700 - market_volatility * 400))

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
            "model_type": "Temporal Convolutional Network (TCN)",
            "method": "tensorflow_tcn",
            "architecture": {
                "n_blocks": 4,
                "dilation_rates": [1, 2, 4, 8],
                "filters": [64, 128, 256, 512],
                "receptive_field": lookback  # Effective history
            },
            "training": train_history,
            "from_cache": from_cache,
            "timestamp": datetime.now().isoformat(),
            "status": "success"
        }

    except ImportError as e:
        logger.error(f"TensorFlow not available for TCN: {e}")
        return {
            "error": "TensorFlow required for TCN model",
            "direction": "Neutral",
            "confidence": 50.0,
            "status": "failed"
        }
    except Exception as e:
        logger.error(f"Error in TCN prediction: {e}", exc_info=True)
        return {
            "error": str(e),
            "direction": "Neutral",
            "confidence": 50.0,
            "status": "failed"
        }
