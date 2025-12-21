"""
Machine Learning Models for Stock Analysis
PCA, Clustering, and basic ML predictions.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import logging
import warnings

# Configure logging
logger = logging.getLogger("uvicorn.info")

# Filter annoying sklearn parallel warning if explicit fix isn't possible in user code
warnings.filterwarnings("ignore", message=".*sklearn.utils.parallel.delayed.*")



def pca_analysis(
    returns: pd.DataFrame,
    n_components: int = 3
) -> Dict[str, Any]:
    """
    Principal Component Analysis for factor decomposition.
    
    Reduces dimensionality and identifies key factors driving returns.
    Uses numpy's SVD for stability (no sklearn dependency).
    
    Args:
        returns: DataFrame of stock returns
        n_components: Number of principal components
    
    Returns:
        Dictionary with components, explained variance, and loadings
    """
    # Standardize the data
    data = returns.dropna()
    if len(data) < 2:
        return {'error': 'Insufficient data for PCA'}
    
    mean = data.mean()
    std = data.std()
    std = std.replace(0, 1)  # Avoid division by zero
    standardized = (data - mean) / std
    
    # SVD decomposition
    U, s, Vt = np.linalg.svd(standardized.values, full_matrices=False)
    
    # Limit to n_components
    n_components = min(n_components, len(s))
    
    # Calculate explained variance
    explained_variance = (s ** 2) / (len(data) - 1)
    total_variance = explained_variance.sum()
    explained_variance_ratio = explained_variance / total_variance
    
    # Component loadings (correlations with original variables)
    loadings = Vt[:n_components].T * s[:n_components] / np.sqrt(len(data) - 1)
    
    # Create loading DataFrame
    loading_df = pd.DataFrame(
        loadings,
        index=data.columns,
        columns=[f'PC{i+1}' for i in range(n_components)]
    )
    
    return {
        'n_components': n_components,
        'explained_variance_ratio': [round(r * 100, 2) for r in explained_variance_ratio[:n_components]],
        'cumulative_variance': round(sum(explained_variance_ratio[:n_components]) * 100, 2),
        'loadings': loading_df.round(4).to_dict(),
        'interpretation': {
            f'PC{i+1}': _interpret_pc(loading_df[f'PC{i+1}']) 
            for i in range(n_components)
        }
    }


def _interpret_pc(loadings: pd.Series, top_n: int = 3) -> Dict[str, Any]:
    """Interpret a principal component based on loadings."""
    top_positive = loadings.nlargest(top_n)
    top_negative = loadings.nsmallest(top_n)
    
    return {
        'top_positive': dict(top_positive.round(4)),
        'top_negative': dict(top_negative.round(4))
    }


def kmeans_clustering(
    returns: pd.DataFrame,
    n_clusters: int = 3,
    max_iterations: int = 100
) -> Dict[str, Any]:
    """
    K-Means clustering for stock grouping.
    
    Groups stocks by similar behavior patterns.
    Pure numpy implementation.
    
    Args:
        returns: DataFrame of stock returns
        n_clusters: Number of clusters
        max_iterations: Maximum iterations for convergence
    
    Returns:
        Dictionary with cluster assignments and centers
    """
    data = returns.dropna().T  # Transpose: rows = stocks
    
    if len(data) < n_clusters:
        return {'error': f'Need at least {n_clusters} stocks for clustering'}
    
    # Feature engineering: mean, std, skew
    features = pd.DataFrame({
        'mean_return': data.mean(axis=1),
        'volatility': data.std(axis=1),
        'skewness': data.skew(axis=1)
    })
    
    # Standardize features
    mean = features.mean()
    std = features.std().replace(0, 1)
    X = ((features - mean) / std).values
    
    # Initialize centroids randomly
    np.random.seed(42)
    indices = np.random.choice(len(X), n_clusters, replace=False)
    centroids = X[indices].copy()
    
    # K-Means iteration
    for _ in range(max_iterations):
        # Assign points to nearest centroid
        distances = np.sqrt(((X[:, np.newaxis] - centroids) ** 2).sum(axis=2))
        labels = distances.argmin(axis=1)
        
        # Update centroids
        new_centroids = np.array([
            X[labels == k].mean(axis=0) if (labels == k).sum() > 0 else centroids[k]
            for k in range(n_clusters)
        ])
        
        # Check convergence
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids
    
    # Create results
    cluster_assignments = dict(zip(features.index, labels))
    
    # Analyze clusters
    cluster_profiles = {}
    for k in range(n_clusters):
        cluster_stocks = [s for s, c in cluster_assignments.items() if c == k]
        if cluster_stocks:
            cluster_features = features.loc[cluster_stocks]
            cluster_profiles[f'Cluster_{k}'] = {
                'count': len(cluster_stocks),
                'stocks': cluster_stocks,
                'avg_return': round(cluster_features['mean_return'].mean() * 100, 4),
                'avg_volatility': round(cluster_features['volatility'].mean() * 100, 2),
                'profile': _cluster_profile(cluster_features)
            }
    
    return {
        'n_clusters': n_clusters,
        'assignments': cluster_assignments,
        'profiles': cluster_profiles
    }


def _cluster_profile(features: pd.DataFrame) -> str:
    """Generate a descriptive profile for a cluster."""
    avg_return = features['mean_return'].mean()
    avg_vol = features['volatility'].mean()
    
    if avg_return > 0.001 and avg_vol < 0.02:
        return 'Low Risk, High Return'
    elif avg_return > 0.001 and avg_vol >= 0.02:
        return 'High Risk, High Return'
    elif avg_return <= 0.001 and avg_vol < 0.02:
        return 'Low Risk, Low Return'
    else:
        return 'High Risk, Low Return'


def simple_momentum_prediction(
    data: pd.DataFrame,
    lookback: int = 20,
    momentum_periods: List[int] = [5, 10, 20]
) -> Dict[str, Any]:
    """
    Simple momentum-based prediction model.
    
    Uses momentum indicators to predict direction.
    
    Args:
        data: DataFrame with Close prices
        lookback: Historical lookback period
        momentum_periods: Momentum calculation periods
    
    Returns:
        Dictionary with prediction and confidence
    """
    close = data['Close'] if 'Close' in data.columns else data.iloc[:, 0]
    
    # Calculate momentum signals
    signals = []
    
    for period in momentum_periods:
        if len(close) > period:
            mom = (close.iloc[-1] - close.iloc[-period]) / close.iloc[-period]
            signals.append(1 if mom > 0 else -1)
    
    if not signals:
        return {'error': 'Insufficient data'}
    
    # Average signal
    avg_signal = sum(signals) / len(signals)
    
    # Calculate trend strength
    sma_short = close.tail(5).mean()
    sma_long = close.tail(20).mean()
    trend_strength = (sma_short - sma_long) / sma_long if sma_long != 0 else 0
    
    # Calculate volatility for confidence
    returns = close.pct_change().dropna()
    recent_vol = returns.tail(20).std()
    
    # Lower volatility = higher confidence
    confidence = max(0.3, min(0.9, 0.8 - recent_vol * 10))
    
    if avg_signal > 0.5:
        prediction = 'Bullish'
    elif avg_signal < -0.5:
        prediction = 'Bearish'
    else:
        prediction = 'Neutral'
    
    return {
        'prediction': prediction,
        'confidence': round(confidence * 100, 1),
        'momentum_signals': dict(zip([f'{p}d' for p in momentum_periods], signals)),
        'trend_strength': round(trend_strength * 100, 2),
        'recent_volatility': round(recent_vol * 100, 2)
    }


def feature_importance_analysis(
    returns: pd.DataFrame,
    target_column: str
) -> Dict[str, float]:
    """
    Analyze feature importance using correlation.
    
    Simpler alternative to tree-based feature importance.
    
    Args:
        returns: DataFrame with features
        target_column: Column to predict
    
    Returns:
        Dictionary with feature importance scores
    """
    if target_column not in returns.columns:
        return {'error': f'Target column {target_column} not found'}
    
    target = returns[target_column]
    features = returns.drop(columns=[target_column])
    
    importance = {}
    for col in features.columns:
        # Use absolute correlation as importance
        corr = target.corr(features[col])
        if not np.isnan(corr):
            importance[col] = round(abs(corr) * 100, 2)
    
    # Sort by importance
    importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
    
    return importance


def regime_detection(
    returns: pd.Series,
    window: int = 60
) -> Dict[str, Any]:
    """
    Detect market regimes based on volatility and returns.
    
    Regimes:
        - Bull High Vol: Strong up, volatile
        - Bull Low Vol: Steady up, calm
        - Bear High Vol: Strong down, volatile
        - Bear Low Vol: Steady down, calm
    
    Args:
        returns: Series of returns
        window: Rolling window for calculations
    
    Returns:
        Dictionary with current regime, history, and strategy adjustments
    """
    if len(returns) < window:
        return {'error': 'Insufficient data for regime detection'}
    
    # Calculate rolling statistics
    rolling_mean = returns.rolling(window).mean()
    rolling_vol = returns.rolling(window).std()
    
    # Thresholds (based on historical distribution)
    mean_threshold = rolling_mean.median()
    vol_threshold = rolling_vol.median()
    
    def classify_regime(mean_val, vol_val):
        high_return = mean_val > mean_threshold
        high_vol = vol_val > vol_threshold
        
        if high_return and high_vol:
            return 'Bull High Volatility'
        elif high_return and not high_vol:
            return 'Bull Low Volatility'
        elif not high_return and high_vol:
            return 'Bear High Volatility'
        else:
            return 'Bear Low Volatility'
    
    current_regime = classify_regime(
        rolling_mean.iloc[-1],
        rolling_vol.iloc[-1]
    )
    
    # Regime history (last 5 periods)
    history = []
    for i in range(-5, 0):
        if len(rolling_mean) > abs(i):
            regime = classify_regime(rolling_mean.iloc[i], rolling_vol.iloc[i])
            history.append(regime)
    
    # Regime stability
    if history:
        stability = history.count(current_regime) / len(history) * 100
    else:
        stability = 0
    
    # Strategy adjustments based on regime (research-backed)
    # Reference: Adaptive models with regime detection improve Sharpe ratio by 81.8%
    strategy_adjustments = {
        'Bull High Volatility': {
            'momentum_weight': 0.7,
            'fundamental_weight': 0.3,
            'position_size_multiplier': 0.8,  # Reduce size in high vol
            'stop_loss_multiplier': 1.5,  # Wider stops
            'entry_strategy': 'scale_in',  # Don't go all-in
            'confidence_adjustment': 0.85,
            'recommended_action': 'BUY with caution, use limit orders'
        },
        'Bull Low Volatility': {
            'momentum_weight': 0.5,
            'fundamental_weight': 0.5,
            'position_size_multiplier': 1.0,  # Full size OK
            'stop_loss_multiplier': 1.0,  # Normal stops
            'entry_strategy': 'immediate',  # Can enter now
            'confidence_adjustment': 1.0,
            'recommended_action': 'BUY - optimal conditions'
        },
        'Bear High Volatility': {
            'momentum_weight': 0.2,  # Momentum fails in bear
            'fundamental_weight': 0.8,  # Focus on value
            'position_size_multiplier': 0.5,  # Half size max
            'stop_loss_multiplier': 2.0,  # Much wider stops
            'entry_strategy': 'wait_or_scale',  # Be patient
            'confidence_adjustment': 0.6,
            'recommended_action': 'WAIT or defensive positions only'
        },
        'Bear Low Volatility': {
            'momentum_weight': 0.3,
            'fundamental_weight': 0.7,
            'position_size_multiplier': 0.7,  # Reduced size
            'stop_loss_multiplier': 1.3,
            'entry_strategy': 'selective',
            'confidence_adjustment': 0.75,
            'recommended_action': 'Selective buying, focus on quality'
        }
    }
    
    current_adjustments = strategy_adjustments.get(
        current_regime, 
        strategy_adjustments['Bull Low Volatility']
    )
    
    # Regime-specific sector preferences
    sector_preferences = {
        'Bull High Volatility': ['Technology', 'Consumer Discretionary', 'Financials'],
        'Bull Low Volatility': ['Technology', 'Communication Services', 'Healthcare'],
        'Bear High Volatility': ['Utilities', 'Consumer Staples', 'Healthcare'],
        'Bear Low Volatility': ['Consumer Staples', 'Utilities', 'Financials']
    }
    
    return {
        'current_regime': current_regime,
        'regime_stability': round(stability, 1),
        'recent_history': history,
        'rolling_return': round(rolling_mean.iloc[-1] * 252 * 100, 2),  # Annualized %
        'rolling_volatility': round(rolling_vol.iloc[-1] * np.sqrt(252) * 100, 2),  # Annualized %
        # NEW: Strategy adjustments
        'strategy_adjustments': current_adjustments,
        'momentum_weight': current_adjustments['momentum_weight'],
        'fundamental_weight': current_adjustments['fundamental_weight'],
        'position_size_multiplier': current_adjustments['position_size_multiplier'],
        'stop_loss_multiplier': current_adjustments['stop_loss_multiplier'],
        'confidence_adjustment': current_adjustments['confidence_adjustment'],
        'entry_strategy': current_adjustments['entry_strategy'],
        'recommended_action': current_adjustments['recommended_action'],
        'preferred_sectors': sector_preferences.get(current_regime, [])
    }


def train_random_forest(
    X: pd.DataFrame,
    y: pd.Series,
    n_estimators: int = 100
) -> Dict[str, Any]:
    """
    Train Random Forest for price direction prediction.
    
    Args:
        X: Feature matrix
        y: Target variable (0/1 for direction)
        n_estimators: Number of trees
    
    Returns:
        Dictionary with trained model and evaluation metrics
    """
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score
        from sklearn.preprocessing import StandardScaler
        
        # Handle NaN values
        X_clean = X.fillna(0)
        y_clean = y.fillna(0)
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X_clean), columns=X_clean.columns, index=X_clean.index)
        
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_clean, test_size=0.2, shuffle=False)
        
        model = RandomForestClassifier(
            n_estimators=n_estimators, 
            random_state=42,
            max_depth=10,
            min_samples_split=5,
            n_jobs=-1
        )
        
        logger.info(f"Training Random Forest with {n_estimators} estimators...")
        model.fit(X_train, y_train)
        logger.info("Random Forest training complete.")
        
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        
        return {
            "model": model,  # Return trained model for prediction
            "scaler": scaler,
            "accuracy_train": round(train_score, 4),
            "accuracy_test": round(test_score, 4),
            "accuracy": round(accuracy, 4),
            "feature_importance": dict(zip(X.columns, model.feature_importances_)),
            "status": "success"
        }
    except ImportError:
        return {"error": "sklearn not installed", "status": "failed"}
    except Exception as e:
        return {"error": str(e), "status": "failed"}


def train_svm(
    X: pd.DataFrame,
    y: pd.Series,
    kernel: str = 'rbf'
) -> Dict[str, Any]:
    """
    Train Support Vector Machine for trend prediction.
    
    Args:
        X: Feature matrix
        y: Target variable (classification)
        kernel: Kernel type ('linear', 'poly', 'rbf')
    
    Returns:
        Dictionary with trained model and evaluation metrics
    """
    try:
        from sklearn.svm import SVC
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score
        from sklearn.preprocessing import StandardScaler
        
        # Handle NaN values
        X_clean = X.fillna(0)
        y_clean = y.fillna(0)
        
        # Scale features - critical for SVM
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X_clean), columns=X_clean.columns, index=X_clean.index)
        
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_clean, test_size=0.2, shuffle=False)
        
        model = SVC(
            kernel=kernel, 
            random_state=42,
            probability=True,  # Enable probability estimates
            C=1.0,
            gamma='scale'
        )
        
        logger.info(f"Training SVM with {kernel} kernel...")
        model.fit(X_train, y_train)
        logger.info("SVM training complete.")
        
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        
        return {
            "model": model,  # Return trained model for prediction
            "scaler": scaler,
            "accuracy": round(accuracy, 4),
            "status": "success"
        }
    except ImportError:
        return {"error": "sklearn not installed", "status": "failed"}
    except Exception as e:
        return {"error": str(e), "status": "failed"}
