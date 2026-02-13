"""
LightGBM Predictor for Stock Price Direction
Microsoft's optimized gradient boosting framework - often outperforms XGBoost.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import logging
import hashlib
from datetime import datetime

logger = logging.getLogger(__name__)


def prepare_lightgbm_features(stock_data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Prepare features for LightGBM model.

    Uses the same 25+ features as XGBoost/Random Forest for consistency,
    but LightGBM can handle additional categorical features efficiently.

    Args:
        stock_data: DataFrame with OHLCV data

    Returns:
        Tuple of (features DataFrame, target Series)
    """
    # Import the shared feature preparation
    from src.analysis.quantitative.random_forest_predictor import prepare_rf_features

    features, target = prepare_rf_features(stock_data)

    # Add day-of-week as categorical feature (LightGBM handles this well)
    try:
        features['day_of_week'] = pd.to_datetime(
            stock_data.index).dayofweek.values[:len(features)]
    except:
        pass  # If index isn't datetime, skip this feature

    return features, target


def get_lightgbm_prediction(
    stock_data: pd.DataFrame,
    symbol: Optional[str] = None,
    n_estimators: int = 500,
    learning_rate: float = 0.05,
    num_leaves: int = 31,
    max_depth: int = 7
) -> Dict[str, Any]:
    """
    Get LightGBM prediction for stock price direction.

    LightGBM advantages over XGBoost:
    - Faster training (leaf-wise growth vs level-wise)
    - Better handling of categorical features
    - Lower memory usage
    - Often 2-5% better accuracy

    Args:
        stock_data: DataFrame with OHLCV data
        symbol: Stock symbol for caching
        n_estimators: Number of boosting rounds (default: 500)
        learning_rate: Learning rate (default: 0.05)
        num_leaves: Max number of leaves per tree (default: 31)
        max_depth: Maximum tree depth (default: 7)

    Returns:
        Dictionary with prediction results
    """
    try:
        import lightgbm as lgb
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        from sklearn.preprocessing import StandardScaler
        from src.analysis.quantitative.model_cache import get_model_cache

        # Prepare features
        try:
            features, target = prepare_lightgbm_features(stock_data)
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
        cache_key = f"{symbol}_{last_date_hash}" if symbol else last_date_hash

        # Try to get cached model
        cached_model_data = cache.get('lightgbm', cache_key, stock_data)

        if cached_model_data and 'model' in cached_model_data:
            logger.info(f"Using cached LightGBM model for {symbol}")
            lgb_result = cached_model_data
            from_cache = True
        else:
            # Train new model
            logger.info(
                f"Training new LightGBM model for {symbol} with {len(features)} samples")

            # Handle NaN values
            X_clean = features.fillna(0)
            y_clean = target.fillna(0)

            # Scale features for consistency (though LightGBM doesn't strictly require it)
            scaler = StandardScaler()
            X_scaled = pd.DataFrame(
                scaler.fit_transform(X_clean),
                columns=X_clean.columns,
                index=X_clean.index
            )

            # Train-test split (time-series: no shuffle)
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y_clean, test_size=0.2, shuffle=False
            )

            # LightGBM parameters
            params = {
                'objective': 'binary',
                'boosting_type': 'gbdt',
                'num_leaves': num_leaves,
                'learning_rate': learning_rate,
                'max_depth': max_depth,
                'min_data_in_leaf': 20,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'lambda_l1': 0.1,
                'lambda_l2': 0.1,
                'verbose': -1,
                'random_state': 42
            }

            # Create datasets
            train_data = lgb.Dataset(X_train, label=y_train)
            test_data = lgb.Dataset(
                X_test, label=y_test, reference=train_data)

            # Train model
            model = lgb.train(
                params,
                train_data,
                num_boost_round=n_estimators,
                valid_sets=[test_data],
                callbacks=[lgb.early_stopping(
                    stopping_rounds=50), lgb.log_evaluation(period=0)]
            )

            # Evaluate
            train_pred = (model.predict(X_train) > 0.5).astype(int)
            test_pred = (model.predict(X_test) > 0.5).astype(int)

            accuracy_train = accuracy_score(y_train, train_pred)
            accuracy_test = accuracy_score(y_test, test_pred)
            precision = precision_score(
                y_test, test_pred, zero_division=0)
            recall = recall_score(y_test, test_pred, zero_division=0)
            f1 = f1_score(y_test, test_pred, zero_division=0)

            # Get feature importance
            feature_importance = dict(
                zip(X_clean.columns, model.feature_importance(importance_type='gain')))

            lgb_result = {
                'model': model,
                'scaler': scaler,
                'accuracy_train': accuracy_train,
                'accuracy_test': accuracy_test,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'feature_importance': feature_importance,
                'n_estimators_used': model.num_trees(),
                'status': 'success'
            }

            # Cache the trained model
            cache.set('lightgbm', cache_key, lgb_result, stock_data, {})
            from_cache = False

        # Make prediction on latest data
        model = lgb_result['model']
        scaler = lgb_result.get('scaler')

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

        # Predict (LightGBM returns probabilities for binary classification)
        prediction_proba = model.predict(last_features_scaled)[0]
        prediction = 1 if prediction_proba > 0.5 else 0

        # Confidence (prediction probability)
        confidence = max(prediction_proba, 1 - prediction_proba) * 100

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
        feature_importance = lgb_result.get('feature_importance', {})
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
            "confidence": round(confidence,  1),
            "predicted_change_pct": round(expected_change_pct, 2),
            "current_price": float(current_price),
            "prediction_value": int(prediction),
            "prediction_probability": round(float(prediction_proba), 4),
            "probabilities": {
                "down": round((1 - prediction_proba) * 100, 1),
                "up": round(prediction_proba * 100, 1)
            },
            "model_metrics": {
                "accuracy_train": lgb_result.get('accuracy_train', 0),
                "accuracy_test": lgb_result.get('accuracy_test', 0),
                "precision": lgb_result.get('precision', 0),
                "recall": lgb_result.get('recall', 0),
                "f1_score": lgb_result.get('f1_score', 0),
                "n_estimators": n_estimators,
                "n_estimators_used": lgb_result.get('n_estimators_used', 0),
                "num_leaves": num_leaves,
                "learning_rate": learning_rate,
                "max_depth": max_depth,
                "training_samples": len(features)
            },
            "feature_importance": top_features,
            "from_cache": from_cache,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "method": "lightgbm",
            "status": "success"
        }

    except ImportError as e:
        logger.error(f"Import error in LightGBM predictor: {e}")
        return {
            "error": "lightgbm not installed - run 'pip install lightgbm'",
            "direction": "Neutral",
            "confidence": 50,
            "status": "failed"
        }
    except Exception as e:
        logger.error(f"Error in LightGBM prediction: {e}", exc_info=True)
        return {
            "error": str(e),
            "direction": "Neutral",
            "confidence": 50,
            "status": "failed"
        }
