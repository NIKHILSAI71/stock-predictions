"""
XGBoost Model for Stock Prediction

Based on research (2024), XGBoost often outperforms Random Forest for stock prediction
with proper hyperparameter tuning, achieving up to 99% accuracy for close price prediction
in some studies.

Key advantages:
- L1/L2 regularization to prevent overfitting
- Handles missing values automatically
- Feature importance analysis
- Sequential error correction (boosting)
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
from datetime import datetime
import logging
import warnings

# Configure logging
logger = logging.getLogger("uvicorn.info")

# Filter annoying sklearn parallel warning if present
warnings.filterwarnings("ignore", message=".*sklearn.utils.parallel.delayed.*")


def prepare_xgboost_features(
    stock_data: pd.DataFrame,
    include_lag_features: bool = True,
    lag_periods: list = [1, 2, 3, 5, 10]
) -> pd.DataFrame:
    """
    Prepare features optimized for XGBoost prediction.

    XGBoost benefits from:
    - Technical indicators
    - Lagged returns
    - Rolling statistics
    - Price momentum features

    Args:
        stock_data: DataFrame with OHLCV data
        include_lag_features: Whether to include lagged features
        lag_periods: Periods for lag features

    Returns:
        DataFrame with features
    """
    features = pd.DataFrame(index=stock_data.index)
    close = stock_data['Close']
    high = stock_data['High']
    low = stock_data['Low']
    volume = stock_data['Volume'] if 'Volume' in stock_data.columns else pd.Series(
        1, index=stock_data.index)

    # Price-based features
    features['returns_1d'] = close.pct_change()
    features['returns_5d'] = close.pct_change(5)
    features['returns_10d'] = close.pct_change(10)
    features['returns_20d'] = close.pct_change(20)

    # Volatility features
    features['volatility_5d'] = features['returns_1d'].rolling(5).std()
    features['volatility_20d'] = features['returns_1d'].rolling(20).std()
    features['volatility_ratio'] = features['volatility_5d'] / \
        features['volatility_20d']

    # Moving average features
    sma_5 = close.rolling(5).mean()
    sma_10 = close.rolling(10).mean()
    sma_20 = close.rolling(20).mean()
    sma_50 = close.rolling(50).mean()

    features['sma_5_dist'] = (close - sma_5) / sma_5
    features['sma_10_dist'] = (close - sma_10) / sma_10
    features['sma_20_dist'] = (close - sma_20) / sma_20
    features['sma_50_dist'] = (close - sma_50) / sma_50
    features['sma_cross_5_20'] = (sma_5 - sma_20) / sma_20

    # RSI (Relative Strength Index)
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta).where(delta < 0, 0).rolling(14).mean()
    rs = gain / loss
    features['rsi'] = 100 - (100 / (1 + rs))

    # MACD
    ema_12 = close.ewm(span=12).mean()
    ema_26 = close.ewm(span=26).mean()
    macd_line = ema_12 - ema_26
    signal_line = macd_line.ewm(span=9).mean()
    features['macd'] = macd_line
    features['macd_signal'] = signal_line
    features['macd_histogram'] = macd_line - signal_line

    # Bollinger Bands position
    bb_middle = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    bb_upper = bb_middle + 2 * bb_std
    bb_lower = bb_middle - 2 * bb_std
    features['bb_position'] = (close - bb_lower) / (bb_upper - bb_lower)
    features['bb_width'] = (bb_upper - bb_lower) / bb_middle

    # Volume features
    features['volume_ratio'] = volume / volume.rolling(20).mean()
    features['volume_trend'] = volume.rolling(
        5).mean() / volume.rolling(20).mean()

    # Price range features
    features['daily_range'] = (high - low) / close
    features['daily_range_avg'] = features['daily_range'].rolling(10).mean()

    # Momentum features
    features['momentum_5'] = close / close.shift(5) - 1
    features['momentum_10'] = close / close.shift(10) - 1
    features['momentum_20'] = close / close.shift(20) - 1

    # Add lagged features if requested
    if include_lag_features:
        for lag in lag_periods:
            features[f'return_lag_{lag}'] = features['returns_1d'].shift(lag)
            features[f'volume_ratio_lag_{lag}'] = features['volume_ratio'].shift(
                lag)

    # Day of week (cyclical encoding)
    if hasattr(stock_data.index, 'dayofweek'):
        features['day_sin'] = np.sin(
            2 * np.pi * stock_data.index.dayofweek / 5)
        features['day_cos'] = np.cos(
            2 * np.pi * stock_data.index.dayofweek / 5)

    return features


def create_xgboost_target(
    close_prices: pd.Series,
    horizon: int = 5,
    threshold: float = 0.0
) -> pd.Series:
    """
    Create binary classification target for XGBoost.

    Args:
        close_prices: Series of close prices
        horizon: Days ahead to predict
        threshold: Minimum % change to classify as up (helps reduce noise)

    Returns:
        Binary series: 1 = price goes up, 0 = price goes down
    """
    future_returns = close_prices.shift(-horizon) / close_prices - 1
    target = (future_returns > threshold).astype(int)
    return target


def train_xgboost(
    X: pd.DataFrame,
    y: pd.Series,
    n_estimators: int = 100,
    max_depth: int = 6,
    learning_rate: float = 0.1,
    early_stopping_rounds: int = 10,
    eval_metric: str = 'logloss'
) -> Dict[str, Any]:
    """
    Train XGBoost classifier for stock direction prediction.

    XGBoost advantages over Random Forest:
    - Better handling of imbalanced data
    - Built-in regularization (L1/L2)
    - Sequential learning corrects previous errors
    - Usually faster training

    Args:
        X: Feature matrix
        y: Target variable (0/1 for direction)
        n_estimators: Number of boosting rounds
        max_depth: Maximum tree depth
        learning_rate: Step size shrinkage (eta)
        early_stopping_rounds: Stop if no improvement
        eval_metric: Evaluation metric

    Returns:
        Dictionary with trained model and metrics
    """
    try:
        import xgboost as xgb
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        from sklearn.preprocessing import StandardScaler
        import logging

        logger = logging.getLogger("uvicorn.info")
        logger.info(f"Training XGBoost with {n_estimators} estimators...")

        # Handle NaN values
        X_clean = X.fillna(0).replace([np.inf, -np.inf], 0)
        y_clean = y.fillna(0)

        # Align indices
        common_idx = X_clean.index.intersection(y_clean.index)
        X_clean = X_clean.loc[common_idx]
        y_clean = y_clean.loc[common_idx]

        if len(X_clean) < 50:
            return {"error": "Insufficient data for training", "status": "failed"}

        # Scale features
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(
            scaler.fit_transform(X_clean),
            columns=X_clean.columns,
            index=X_clean.index
        )

        # Time-series split (no shuffle to preserve temporal order)
        train_size = int(len(X_scaled) * 0.8)
        X_train = X_scaled.iloc[:train_size]
        X_test = X_scaled.iloc[train_size:]
        y_train = y_clean.iloc[:train_size]
        y_test = y_clean.iloc[train_size:]

        # Create XGBoost classifier with regularization
        model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=0.8,  # Prevent overfitting
            colsample_bytree=0.8,  # Use 80% of features per tree
            reg_alpha=0.1,  # L1 regularization
            reg_lambda=1.0,  # L2 regularization
            random_state=42,
            eval_metric=eval_metric,
            n_jobs=-1
        )

        # Train with early stopping
        logger.info(f"Training XGBoost with {n_estimators} estimators...")
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )
        logger.info("XGBoost training complete.")

        # Predictions
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        test_proba = model.predict_proba(X_test)

        # Get confidence for latest prediction
        if len(test_proba) > 0:
            latest_proba = test_proba[-1]
            confidence = max(latest_proba) * 100
        else:
            confidence = 50.0

        # Calculate metrics
        train_accuracy = accuracy_score(y_train, train_pred)
        test_accuracy = accuracy_score(y_test, test_pred)
        precision = precision_score(y_test, test_pred, zero_division=0)
        recall = recall_score(y_test, test_pred, zero_division=0)
        f1 = f1_score(y_test, test_pred, zero_division=0)

        # Feature importance
        feature_importance = dict(
            zip(X_clean.columns, model.feature_importances_))
        top_features = dict(sorted(
            feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10])

        # Latest prediction
        latest_prediction = int(test_pred[-1]) if len(test_pred) > 0 else None
        direction = "Bullish" if latest_prediction == 1 else "Bearish"

        return {
            "model": model,
            "scaler": scaler,
            "accuracy_train": round(train_accuracy, 4),
            "accuracy_test": round(test_accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "feature_importance": top_features,
            "latest_prediction": latest_prediction,
            "direction": direction,
            "confidence": round(confidence, 1),
            "n_estimators_used": model.best_iteration if hasattr(model, 'best_iteration') else n_estimators,
            "status": "success"
        }

    except ImportError:
        return {"error": "xgboost not installed. Run: pip install xgboost", "status": "failed"}
    except Exception as e:
        return {"error": str(e), "status": "failed"}


def get_xgboost_prediction(
    stock_data: pd.DataFrame,
    prediction_horizon: int = 5,
    lookback_days: int = 252,
    symbol: str = None
) -> Dict[str, Any]:
    """
    Main function to get XGBoost prediction for a stock with caching support.

    Args:
        stock_data: DataFrame with OHLCV data
        prediction_horizon: Days ahead to predict
        lookback_days: Training window
        symbol: Stock symbol (for caching)

    Returns:
        XGBoost prediction results
    """
    from .model_cache import get_model_cache

    # Try to load from cache
    cache = get_model_cache()
    cached_result = None

    if symbol:
        cached_result = cache.get('xgboost', symbol, stock_data)

    if cached_result is not None:
        logger.info(f"Using cached XGBoost model for {symbol}")
        cached_result["from_cache"] = True
        return cached_result

    # Train new model
    if len(stock_data) < lookback_days:
        lookback_days = len(stock_data) - prediction_horizon - 10

    if lookback_days < 50:
        return {
            "error": "Insufficient historical data",
            "direction": "Neutral",
            "confidence": 50.0,
            "status": "failed"
        }

    # Use recent data for training
    data = stock_data.tail(lookback_days + prediction_horizon + 50)

    # Prepare features and target
    features = prepare_xgboost_features(data)
    target = create_xgboost_target(data['Close'], horizon=prediction_horizon)

    # Align features and target
    common_idx = features.index.intersection(target.index)
    features = features.loc[common_idx]
    target = target.loc[common_idx]

    # Remove rows with NaN target (future data we can't train on)
    valid_idx = ~target.isna()
    features = features[valid_idx]
    target = target[valid_idx]

    # Drop rows with too many NaN features
    features = features.dropna()
    target = target.loc[features.index]

    if len(features) < 50:
        return {
            "error": "Insufficient valid data after feature preparation",
            "direction": "Neutral",
            "confidence": 50.0,
            "status": "failed"
        }

    # Train model
    result = train_xgboost(
        X=features,
        y=target,
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1
    )

    if result.get("status") == "failed":
        return result

    # Add prediction horizon info
    result["prediction_horizon_days"] = prediction_horizon
    result["training_samples"] = len(features)
    result["from_cache"] = False
    result["method"] = "xgboost"

    # Add predicted_change_pct estimate based on direction and confidence
    if result.get("direction") == "Bullish":
        result["predicted_change_pct"] = result.get(
            "confidence", 50) / 10  # Scale confidence to reasonable %
    elif result.get("direction") == "Bearish":
        result["predicted_change_pct"] = -result.get("confidence", 50) / 10
    else:
        result["predicted_change_pct"] = 0.0

    # Cache the result
    if symbol and result.get("status") == "success":
        cache.set('xgboost', symbol, result, stock_data,
                  {"lookback_days": lookback_days})

    return result
