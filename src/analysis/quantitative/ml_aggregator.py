"""
ML Prediction Aggregator Module
Combines Random Forest, SVM, PCA analysis, K-Means clustering, and momentum prediction
into a unified ensemble prediction for stock direction and confidence.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

from .ml_models import (
    pca_analysis, kmeans_clustering, simple_momentum_prediction,
    feature_importance_analysis, regime_detection, 
    train_random_forest, train_svm
)


def prepare_ml_features(stock_data: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare feature set for ML models from raw stock data.
    Uses both technical and fundamental features if available.
    
    Features:
    - RSI, MACD components, Bollinger Band %B
    - Moving average distances (5, 10, 20, 50 day)
    - Volume ratio
    - Price momentum (5, 10, 20 day returns)
    - Volatility (rolling std)
    """
    df = stock_data.copy()
    
    if 'Close' not in df.columns:
        return pd.DataFrame()
    
    close = df['Close']
    high = df.get('High', close)
    low = df.get('Low', close)
    volume = df.get('Volume', pd.Series([1] * len(close), index=close.index))
    
    features = pd.DataFrame(index=df.index)
    
    # Price returns for different periods
    for period in [1, 5, 10, 20]:
        features[f'return_{period}d'] = close.pct_change(period)
    
    # Moving averages distance from price
    for period in [5, 10, 20, 50]:
        if len(close) >= period:
            ma = close.rolling(window=period).mean()
            features[f'ma_{period}_dist'] = (close - ma) / ma
    
    # RSI (14-period)
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss.replace(0, 1)
    features['rsi'] = 100 - (100 / (1 + rs))
    
    # MACD components
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    features['macd'] = macd
    features['macd_signal'] = signal
    features['macd_hist'] = macd - signal
    
    # Bollinger Bands %B
    sma20 = close.rolling(window=20).mean()
    std20 = close.rolling(window=20).std()
    upper_band = sma20 + (std20 * 2)
    lower_band = sma20 - (std20 * 2)
    features['bb_pct_b'] = (close - lower_band) / (upper_band - lower_band)
    
    # Volume features
    avg_volume = volume.rolling(window=20).mean()
    features['volume_ratio'] = volume / avg_volume.replace(0, 1)
    
    # Volatility
    features['volatility_10d'] = close.pct_change().rolling(window=10).std() * np.sqrt(252)
    features['volatility_20d'] = close.pct_change().rolling(window=20).std() * np.sqrt(252)
    
    # ATR (Average True Range) normalized
    tr = pd.concat([
        high - low,
        abs(high - close.shift(1)),
        abs(low - close.shift(1))
    ], axis=1).max(axis=1)
    features['atr_pct'] = tr.rolling(window=14).mean() / close
    
    # Stochastic
    lowest_14 = low.rolling(window=14).min()
    highest_14 = high.rolling(window=14).max()
    features['stoch_k'] = 100 * (close - lowest_14) / (highest_14 - lowest_14 + 0.0001)
    
    # On-Balance Volume trend
    obv = (np.sign(close.diff()) * volume).cumsum()
    features['obv_change'] = obv.pct_change(5)
    
    return features.replace([np.inf, -np.inf], np.nan)


def create_target(close_prices: pd.Series, horizon: int = 1, threshold: float = 0.0) -> pd.Series:
    """
    Create binary classification target.
    1 = price goes up, 0 = price goes down
    
    Args:
        close_prices: Series of close prices
        horizon: Days ahead to predict
        threshold: Minimum % change to count as up/down
    """
    future_return = close_prices.pct_change(horizon).shift(-horizon)
    target = (future_return > threshold).astype(int)
    return target


def get_ml_ensemble_prediction(
    stock_data: pd.DataFrame,
    lookback_days: int = 252,
    prediction_horizon: int = 5,
    include_fundamentals: bool = True
) -> Dict[str, Any]:
    """
    Generate ensemble ML prediction combining multiple models.
    
    Args:
        stock_data: DataFrame with OHLCV data
        lookback_days: Training window (252 = 1 year of trading days)
        prediction_horizon: Days ahead to predict
        include_fundamentals: Whether to include fundamental features
        
    Returns:
        Dictionary with:
        - rf_prediction: Random Forest prediction (BULLISH/BEARISH/NEUTRAL)
        - rf_probability: Probability of upward move
        - svm_prediction: SVM prediction
        - svm_confidence: SVM confidence
        - momentum_prediction: Momentum-based prediction
        - pca_factors: Dominant factors from PCA
        - cluster_profile: Stock behavior cluster
        - regime: Current market regime
        - ensemble_direction: Combined prediction
        - ensemble_confidence: Combined confidence (0-100)
    """
    result = {
        'rf_prediction': 'NEUTRAL',
        'rf_probability': 0.5,
        'svm_prediction': 'NEUTRAL',
        'svm_confidence': 0.5,
        'momentum_prediction': 'Neutral',
        'momentum_confidence': 50,
        'pca_factors': [],
        'pca_explained_variance': 0,
        'cluster_profile': 'Unknown',
        'regime': 'Unknown',
        'regime_stability': 0,
        'ensemble_direction': 'NEUTRAL',
        'ensemble_confidence': 50,
        'models_agree': False,
        'feature_importance': {},
        'error': None
    }
    
    try:
        if len(stock_data) < 60:
            result['error'] = 'Insufficient data for ML prediction'
            return result
        
        # Prepare features
        features = prepare_ml_features(stock_data)
        if features.empty:
            result['error'] = 'Could not prepare features'
            return result
        
        # Create target (5-day forward return direction)
        close = stock_data['Close']
        target = create_target(close, horizon=prediction_horizon)
        
        # Align features and target, drop NaN
        combined = pd.concat([features, target.rename('target')], axis=1).dropna()
        
        if len(combined) < 60:
            result['error'] = 'Insufficient clean data after preprocessing'
            return result
        
        # Use lookback window for training
        train_data = combined.iloc[-min(lookback_days, len(combined)-1):-1]
        latest_features = combined.iloc[[-1]].drop(columns=['target'])
        
        X_train = train_data.drop(columns=['target'])
        y_train = train_data['target']
        
        # ============ 1. RANDOM FOREST ============
        try:
            rf_result = train_random_forest(X_train, y_train, n_estimators=50)
            if 'error' not in rf_result and 'model' in rf_result:
                rf_model = rf_result['model']
                rf_proba = rf_model.predict_proba(latest_features)[0]
                rf_pred_class = rf_model.predict(latest_features)[0]
                
                result['rf_probability'] = float(rf_proba[1]) if len(rf_proba) > 1 else 0.5
                if rf_pred_class == 1 and result['rf_probability'] > 0.55:
                    result['rf_prediction'] = 'BULLISH'
                elif rf_pred_class == 0 and result['rf_probability'] < 0.45:
                    result['rf_prediction'] = 'BEARISH'
                else:
                    result['rf_prediction'] = 'NEUTRAL'
                    
                # Feature importance
                if hasattr(rf_model, 'feature_importances_'):
                    importances = dict(zip(X_train.columns, rf_model.feature_importances_))
                    result['feature_importance'] = {
                        k: round(v, 4) for k, v in 
                        sorted(importances.items(), key=lambda x: -x[1])[:5]
                    }
        except Exception as e:
            result['rf_prediction'] = 'NEUTRAL'
            
        # ============ 2. SVM ============
        try:
            svm_result = train_svm(X_train, y_train.astype(int), kernel='rbf')
            if 'error' not in svm_result and 'model' in svm_result:
                svm_model = svm_result['model']
                svm_pred = svm_model.predict(latest_features)[0]
                
                # Get decision function for confidence
                if hasattr(svm_model, 'decision_function'):
                    decision = svm_model.decision_function(latest_features)[0]
                    svm_conf = 1 / (1 + np.exp(-abs(decision)))  # Sigmoid
                else:
                    svm_conf = 0.5
                    
                result['svm_confidence'] = float(svm_conf)
                if svm_pred == 1 and svm_conf > 0.55:
                    result['svm_prediction'] = 'BULLISH'
                elif svm_pred == 0 and svm_conf > 0.55:
                    result['svm_prediction'] = 'BEARISH'
                else:
                    result['svm_prediction'] = 'NEUTRAL'
        except Exception as e:
            result['svm_prediction'] = 'NEUTRAL'
            
        # ============ 3. MOMENTUM PREDICTION ============
        try:
            momentum_result = simple_momentum_prediction(stock_data, lookback=20)
            result['momentum_prediction'] = momentum_result.get('prediction', 'Neutral')
            result['momentum_confidence'] = momentum_result.get('confidence', 50)
        except:
            pass
            
        # ============ 4. PCA ANALYSIS ============
        try:
            # Transpose to get stocks as rows for meaningful PCA
            returns = stock_data['Close'].pct_change().dropna()
            if len(returns) > 30:
                # Create pseudo multi-asset returns for PCA
                returns_df = pd.DataFrame({
                    'price': returns,
                    'volume': stock_data['Volume'].pct_change().fillna(0).values[-len(returns):],
                    'range': ((stock_data['High'] - stock_data['Low']) / stock_data['Close']).values[-len(returns):]
                })
                pca_result = pca_analysis(returns_df, n_components=2)
                
                if pca_result.get('explained_variance_ratio'):
                    result['pca_explained_variance'] = round(sum(pca_result['explained_variance_ratio']) * 100, 1)
                    
                if pca_result.get('loadings') is not None:
                    # Identify dominant factors
                    loadings = pca_result['loadings']
                    if hasattr(loadings, 'iloc'):
                        top_factors = loadings.iloc[0].abs().nlargest(3).index.tolist()
                        result['pca_factors'] = top_factors
        except:
            pass
            
        # ============ 5. REGIME DETECTION ============
        try:
            returns = stock_data['Close'].pct_change().dropna()
            if len(returns) > 60:
                regime_result = regime_detection(returns, window=60)
                result['regime'] = regime_result.get('current_regime', 'Unknown')
                result['regime_stability'] = regime_result.get('regime_stability', 0)
        except:
            pass
            
        # ============ 6. CLUSTER PROFILE ============
        try:
            # Simple profiling based on momentum and volatility
            recent_return = float(close.pct_change(20).iloc[-1]) if len(close) > 20 else 0
            recent_vol = float(close.pct_change().rolling(20).std().iloc[-1]) * np.sqrt(252) if len(close) > 20 else 0
            
            if recent_return > 0.05 and recent_vol < 0.3:
                result['cluster_profile'] = 'Steady Gainer'
            elif recent_return > 0.05 and recent_vol >= 0.3:
                result['cluster_profile'] = 'Volatile Momentum'
            elif recent_return < -0.05 and recent_vol < 0.3:
                result['cluster_profile'] = 'Steady Decliner'
            elif recent_return < -0.05 and recent_vol >= 0.3:
                result['cluster_profile'] = 'High Risk Falling'
            elif abs(recent_return) <= 0.05 and recent_vol < 0.2:
                result['cluster_profile'] = 'Low Volatility Range'
            else:
                result['cluster_profile'] = 'Mixed Signals'
        except:
            pass
            
        # ============ 7. ENSEMBLE COMBINATION ============
        votes = {
            'BULLISH': 0,
            'BEARISH': 0,
            'NEUTRAL': 0
        }
        weights = {
            'rf': 0.35,  # Random Forest has good accuracy
            'svm': 0.25,  # SVM for pattern recognition
            'momentum': 0.25,  # Momentum for trend
            'regime': 0.15  # Regime for context
        }
        
        # RF vote
        if result['rf_prediction'] == 'BULLISH':
            votes['BULLISH'] += weights['rf'] * result['rf_probability']
        elif result['rf_prediction'] == 'BEARISH':
            votes['BEARISH'] += weights['rf'] * (1 - result['rf_probability'])
        else:
            votes['NEUTRAL'] += weights['rf'] * 0.5
            
        # SVM vote
        if result['svm_prediction'] == 'BULLISH':
            votes['BULLISH'] += weights['svm'] * result['svm_confidence']
        elif result['svm_prediction'] == 'BEARISH':
            votes['BEARISH'] += weights['svm'] * result['svm_confidence']
        else:
            votes['NEUTRAL'] += weights['svm'] * 0.5
            
        # Momentum vote
        if result['momentum_prediction'] == 'Bullish':
            votes['BULLISH'] += weights['momentum'] * (result['momentum_confidence'] / 100)
        elif result['momentum_prediction'] == 'Bearish':
            votes['BEARISH'] += weights['momentum'] * (result['momentum_confidence'] / 100)
        else:
            votes['NEUTRAL'] += weights['momentum'] * 0.5
            
        # Regime adjustment
        regime = result['regime'].lower() if result['regime'] else ''
        if 'bull' in regime:
            votes['BULLISH'] += weights['regime'] * 0.7
        elif 'bear' in regime:
            votes['BEARISH'] += weights['regime'] * 0.7
        else:
            votes['NEUTRAL'] += weights['regime'] * 0.5
        
        # Determine winner
        max_vote = max(votes.values())
        if votes['BULLISH'] == max_vote and votes['BULLISH'] > 0.4:
            result['ensemble_direction'] = 'BULLISH'
        elif votes['BEARISH'] == max_vote and votes['BEARISH'] > 0.4:
            result['ensemble_direction'] = 'BEARISH'
        else:
            result['ensemble_direction'] = 'NEUTRAL'
            
        # Confidence based on agreement
        total_vote = sum(votes.values())
        winning_pct = (max_vote / total_vote * 100) if total_vote > 0 else 50
        result['ensemble_confidence'] = round(min(95, max(30, winning_pct)), 1)
        
        # Check if models agree
        predictions = [result['rf_prediction'], result['svm_prediction'], result['momentum_prediction'].upper()]
        bullish_count = sum(1 for p in predictions if 'BULL' in p.upper())
        bearish_count = sum(1 for p in predictions if 'BEAR' in p.upper())
        result['models_agree'] = bullish_count >= 2 or bearish_count >= 2
        
        # Boost confidence if models agree
        if result['models_agree']:
            result['ensemble_confidence'] = min(95, result['ensemble_confidence'] * 1.1)
            
    except Exception as e:
        result['error'] = str(e)
        
    return result


def get_ml_prediction_summary(ml_result: Dict[str, Any]) -> str:
    """
    Generate human-readable summary of ML predictions.
    """
    if ml_result.get('error'):
        return f"ML Prediction: Unable to generate ({ml_result['error']})"
    
    direction = ml_result['ensemble_direction']
    confidence = ml_result['ensemble_confidence']
    
    model_signals = [
        f"RF: {ml_result['rf_prediction']}",
        f"SVM: {ml_result['svm_prediction']}",
        f"Momentum: {ml_result['momentum_prediction']}"
    ]
    
    agreement = "Models agree" if ml_result['models_agree'] else "Mixed signals"
    
    summary = f"ML Ensemble: {direction} ({confidence:.0f}% confidence) | {agreement} | {', '.join(model_signals)}"
    
    if ml_result.get('regime') and ml_result['regime'] != 'Unknown':
        summary += f" | Regime: {ml_result['regime']}"
        
    return summary
