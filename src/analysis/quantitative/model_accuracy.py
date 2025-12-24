"""
Model Accuracy and Performance Metrics Module
Calculates MAPE, RMSE, confidence intervals, and tracks prediction accuracy.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime, timedelta
import json
import os


def calculate_mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """
    Calculate Mean Absolute Percentage Error.
    
    Args:
        actual: Array of actual values
        predicted: Array of predicted values
        
    Returns:
        MAPE as percentage (e.g., 4.5 means 4.5% average error)
    """
    actual = np.array(actual)
    predicted = np.array(predicted)
    
    # Avoid division by zero
    mask = actual != 0
    if not np.any(mask):
        return 0.0
        
    mape = np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100
    return round(mape, 2)


def calculate_rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """
    Calculate Root Mean Square Error.
    
    Args:
        actual: Array of actual values
        predicted: Array of predicted values
        
    Returns:
        RMSE value
    """
    actual = np.array(actual)
    predicted = np.array(predicted)
    
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))
    return round(rmse, 4)


def calculate_mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    """
    Calculate Mean Absolute Error.
    """
    actual = np.array(actual)
    predicted = np.array(predicted)
    
    mae = np.mean(np.abs(actual - predicted))
    return round(mae, 4)


def calculate_directional_accuracy(actual_returns: np.ndarray, predicted_returns: np.ndarray) -> float:
    """
    Calculate directional accuracy (% of correct direction predictions).
    
    Args:
        actual_returns: Array of actual returns
        predicted_returns: Array of predicted returns
        
    Returns:
        Accuracy as percentage (0-100)
    """
    actual = np.array(actual_returns)
    predicted = np.array(predicted_returns)
    
    # Check if direction matches
    correct = np.sign(actual) == np.sign(predicted)
    accuracy = np.mean(correct) * 100
    
    return round(accuracy, 1)


def calculate_confidence_interval(
    predictions: List[float], 
    confidence: float = 0.95
) -> Tuple[float, float]:
    """
    Calculate confidence interval for predictions.
    
    Args:
        predictions: List of prediction values
        confidence: Confidence level (default 95%)
        
    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    if not predictions or len(predictions) < 2:
        return (0.0, 0.0)
    
    predictions = np.array(predictions)
    mean = np.mean(predictions)
    std = np.std(predictions, ddof=1)
    
    # Z-score for confidence level
    z_scores = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}
    z = z_scores.get(confidence, 1.96)
    
    margin = z * (std / np.sqrt(len(predictions)))
    
    return (round(mean - margin, 2), round(mean + margin, 2))


def calculate_price_confidence_interval(
    base_price: float,
    volatility: float,
    days: int,
    confidence: float = 0.95
) -> Tuple[float, float]:
    """
    Calculate confidence interval for future price based on volatility.
    
    Args:
        base_price: Current price
        volatility: Annualized volatility (as decimal, e.g., 0.25 for 25%)
        days: Number of days ahead
        confidence: Confidence level
        
    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    # Z-score for confidence level
    z_scores = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}
    z = z_scores.get(confidence, 1.96)
    
    # Scale volatility to time period
    time_factor = np.sqrt(days / 252)  # 252 trading days per year
    scaled_vol = volatility * time_factor
    
    # Calculate bounds
    lower = base_price * np.exp(-z * scaled_vol)
    upper = base_price * np.exp(z * scaled_vol)
    
    return (round(lower, 2), round(upper, 2))


def get_model_metrics_from_predictions(
    predictions: List[Dict], 
    current_prices: Dict[str, float]
) -> Dict[str, Any]:
    """
    Calculate accuracy metrics from historical predictions.
    
    Args:
        predictions: List of prediction records
        current_prices: Dict of symbol -> current price
        
    Returns:
        Dict with accuracy metrics per model
    """
    model_stats = {}
    
    for pred in predictions:
        if not pred.get('validated', False):
            continue
            
        model = pred.get('model', 'unknown')
        if model not in model_stats:
            model_stats[model] = {
                'total': 0,
                'correct': 0,
                'predicted_changes': [],
                'actual_changes': []
            }
        
        model_stats[model]['total'] += 1
        
        if pred.get('was_correct', False):
            model_stats[model]['correct'] += 1
            
        if 'change_pct' in pred and 'actual_change_pct' in pred:
            model_stats[model]['predicted_changes'].append(pred['change_pct'])
            model_stats[model]['actual_changes'].append(pred['actual_change_pct'])
    
    # Calculate final metrics
    results = {}
    for model, stats in model_stats.items():
        accuracy = (stats['correct'] / stats['total'] * 100) if stats['total'] > 0 else 50.0
        
        mape = 0.0
        if stats['predicted_changes'] and stats['actual_changes']:
            mape = calculate_mape(
                np.array(stats['actual_changes']), 
                np.array(stats['predicted_changes'])
            )
        
        results[model] = {
            'accuracy': round(accuracy, 1),
            'mape': mape,
            'total_predictions': stats['total'],
            'correct_predictions': stats['correct']
        }
    
    return results


def get_prediction_history(data_file: str = "data/predictions_history.json") -> List[Dict]:
    """Load prediction history from file."""
    try:
        if os.path.exists(data_file):
            with open(data_file, 'r') as f:
                return json.load(f)
    except Exception as e:
        print(f"Error loading prediction history: {e}")
    return []


def get_model_accuracy_summary(symbol: str = None) -> Dict[str, Any]:
    """
    Get comprehensive model accuracy summary.
    
    Args:
        symbol: Optional symbol to filter by
        
    Returns:
        Dict with model accuracy metrics
    """
    predictions = get_prediction_history()
    
    # Filter by symbol if provided
    if symbol:
        predictions = [p for p in predictions if p.get('symbol') == symbol]
    
    # Only use validated predictions
    validated = [p for p in predictions if p.get('validated', False)]
    
    if not validated:
        # No validated predictions yet - return pending status
        return {
            'status': 'pending',
            'system_accuracy': None,
            'total_predictions': len(predictions),
            'validated_predictions': 0,
            'models': {},
            'note': 'Accuracy metrics will appear after predictions are verified against actual outcomes'
        }
    
    # Calculate actual metrics
    correct = sum(1 for p in validated if p.get('was_correct', False))
    system_accuracy = (correct / len(validated) * 100) if validated else 50.0
    
    # Group by model
    model_metrics = get_model_metrics_from_predictions(validated, {})
    
    return {
        'system_accuracy': round(system_accuracy, 1),
        'total_predictions': len(predictions),
        'validated_predictions': len(validated),
        'models': model_metrics,
        'last_updated': datetime.now().isoformat()
    }


def generate_price_targets(
    current_price: float,
    predicted_return_7d: float,
    predicted_return_30d: float,
    predicted_return_90d: float,
    volatility: float = 0.25,
    garch_volatility: float = None
) -> Dict[str, Any]:
    """
    Generate volatility-adjusted price targets with confidence intervals.
    
    Args:
        current_price: Current stock price
        predicted_return_7d: Predicted return for 7 days (as decimal)
        predicted_return_30d: Predicted return for 30 days
        predicted_return_90d: Predicted return for 90 days
        volatility: Annualized volatility
        garch_volatility: GARCH-forecasted volatility (if available, uses this for forward-looking)
        
    Returns:
        Dict with price targets and confidence intervals
    """
    # Use GARCH volatility if available, otherwise historical
    effective_volatility = garch_volatility if garch_volatility else volatility
    
    # Volatility regime classification
    if effective_volatility > 0.50:
        vol_regime = "EXTREME"
        vol_multiplier = 1.5  # Widen targets significantly
    elif effective_volatility > 0.35:
        vol_regime = "HIGH"
        vol_multiplier = 1.25
    elif effective_volatility > 0.20:
        vol_regime = "NORMAL"
        vol_multiplier = 1.0
    else:
        vol_regime = "LOW"
        vol_multiplier = 0.85  # Tighter targets in calm markets
    
    targets = {}
    
    for days, pred_return, label in [
        (7, predicted_return_7d, 'day_7'),
        (30, predicted_return_30d, 'day_30'),
        (90, predicted_return_90d, 'day_90')
    ]:
        target_price = current_price * (1 + pred_return)
        
        # Apply volatility multiplier to get adjusted intervals
        adjusted_vol = effective_volatility * vol_multiplier
        ci_lower, ci_upper = calculate_price_confidence_interval(
            target_price, adjusted_vol, days
        )
        
        # Calculate range width as percentage
        range_width_pct = ((ci_upper - ci_lower) / target_price) * 100
        
        direction = "Bullish" if pred_return > 0.01 else ("Bearish" if pred_return < -0.01 else "Neutral")
        
        targets[label] = {
            'price': round(target_price, 2),
            'change_pct': round(pred_return * 100, 2),
            'confidence_interval': [ci_lower, ci_upper],
            'range_width_pct': round(range_width_pct, 2),
            'direction': direction
        }
    
    # Add volatility context
    targets['volatility_context'] = {
        'regime': vol_regime,
        'historical_volatility': round(volatility * 100, 1) if volatility else None,
        'garch_forecast': round(garch_volatility * 100, 1) if garch_volatility else None,
        'multiplier_applied': vol_multiplier,
        'note': f"Targets {'widened' if vol_multiplier > 1 else 'tightened'} due to {vol_regime.lower()} volatility"
    }
    
    return targets


def get_backtest_summary(symbol: str = None) -> Dict[str, Any]:
    """
    Get backtesting performance summary from actual prediction history.
    
    Returns dynamically calculated backtest metrics.
    """
    predictions = get_prediction_history()
    
    # Filter by symbol if provided
    if symbol:
        predictions = [p for p in predictions if p.get('symbol') == symbol.upper()]
    
    # Only use validated predictions
    validated = [p for p in predictions if p.get('validated', False)]
    
    if not validated:
        # No validated predictions yet - return status indicating this
        return {
            'status': 'pending',
            'message': 'No validated predictions yet. Metrics will appear after predictions are verified.',
            'total_predictions': len(predictions),
            'validated_count': 0
        }
    
    # Calculate metrics from validated predictions
    total_trades = len(validated)
    correct = sum(1 for p in validated if p.get('was_correct', False))
    win_rate = (correct / total_trades * 100) if total_trades > 0 else 0
    
    # Calculate average gains/losses
    gains = [p.get('actual_change_pct', 0) for p in validated if p.get('was_correct', False) and p.get('actual_change_pct', 0) > 0]
    losses = [p.get('actual_change_pct', 0) for p in validated if not p.get('was_correct', False) and p.get('actual_change_pct', 0) < 0]
    
    avg_gain = np.mean(gains) if gains else 0
    avg_loss = np.mean(losses) if losses else 0
    
    # Profit factor = Total Gains / Total Losses (in absolute terms)
    total_gains = sum(gains) if gains else 0
    total_losses = abs(sum(losses)) if losses else 1  # Avoid division by zero
    profit_factor = total_gains / total_losses if total_losses > 0 else 0
    
    # Simple Sharpe ratio approximation (returns / volatility)
    returns = [p.get('actual_change_pct', 0) for p in validated]
    avg_return = np.mean(returns) if returns else 0
    std_return = np.std(returns) if len(returns) > 1 else 1
    sharpe_ratio = (avg_return / std_return) * np.sqrt(252 / 5) if std_return > 0 else 0  # Annualized for 5-day predictions
    
    return {
        'status': 'active',
        'total_predictions': len(predictions),
        'validated_count': total_trades,
        'win_rate': round(win_rate, 1),
        'avg_gain': round(avg_gain, 2),
        'avg_loss': round(avg_loss, 2),
        'profit_factor': round(profit_factor, 2),
        'sharpe_ratio': round(sharpe_ratio, 2),
        'last_updated': datetime.now().isoformat()
    }

