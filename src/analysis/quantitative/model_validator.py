"""
Model Validator Module

Proper validation for time series models with walk-forward testing,
early stopping, and production performance tracking.

Fixes:
- Overfitting to training data
- Improper train/validation/test splits
- Missing cross-validation
- No detection of model degradation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Callable, Any, Optional
from sklearn.model_selection import TimeSeriesSplit
import logging

logger = logging.getLogger(__name__)


class TimeSeriesValidator:
    """
    Proper validation for time series models.
    
    Uses walk-forward validation which is the ONLY correct 
    way to validate time series ML models.
    
    Features:
    - TimeSeriesSplit for proper temporal ordering
    - Directional accuracy (most important for trading)
    - Realistic performance estimates with conservative adjustments
    - Reliability scoring
    
    Usage:
        validator = TimeSeriesValidator(n_splits=5)
        
        def train_func(X_train, y_train):
            model = XGBRegressor()
            model.fit(X_train, y_train)
            return model
        
        def predict_func(model, X_test):
            return model.predict(X_test)
        
        metrics = validator.walk_forward_validation(X, y, train_func, predict_func)
    """
    
    def __init__(self, n_splits: int = 5, gap: int = 0):
        """
        Initialize the validator.
        
        Args:
            n_splits: Number of folds for cross-validation
            gap: Number of samples to exclude between train and test sets
                 (useful to prevent look-ahead bias)
        """
        self.n_splits = n_splits
        self.gap = gap
        self.tscv = TimeSeriesSplit(n_splits=n_splits, gap=gap)
        self.fold_results: List[Dict[str, Any]] = []
        
    def walk_forward_validation(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model_train_func: Callable,
        model_predict_func: Callable
    ) -> Dict[str, float]:
        """
        Walk-forward validation for time series.
        
        This is the MOST REALISTIC evaluation method for trading models.
        Each fold uses only past data for training.
        
        Args:
            X: Features array (n_samples, n_features)
            y: Target array (n_samples,)
            model_train_func: Function(X_train, y_train) -> trained_model
            model_predict_func: Function(model, X_test) -> predictions
            
        Returns:
            Dictionary with validation metrics
        """
        predictions = []
        actuals = []
        fold_accuracies = []
        fold_errors = []
        self.fold_results = []
        
        for fold, (train_idx, test_idx) in enumerate(self.tscv.split(X)):
            # Split data (respects temporal order)
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            try:
                # Train model on this fold
                model = model_train_func(X_train, y_train)
                
                # Predict on test set
                y_pred = model_predict_func(model, X_test)
                
                # Calculate fold metrics
                fold_accuracy = self._calculate_directional_accuracy(y_test, y_pred)
                fold_rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
                fold_mae = np.mean(np.abs(y_test - y_pred))
                
                fold_accuracies.append(fold_accuracy)
                fold_errors.append(fold_rmse)
                
                predictions.extend(y_pred)
                actuals.extend(y_test)
                
                self.fold_results.append({
                    'fold': fold,
                    'train_size': len(train_idx),
                    'test_size': len(test_idx),
                    'accuracy': fold_accuracy,
                    'rmse': fold_rmse,
                    'mae': fold_mae
                })
                
                logger.debug(
                    f"Fold {fold}: accuracy={fold_accuracy:.2%}, "
                    f"rmse={fold_rmse:.4f}, train_size={len(train_idx)}"
                )
                
            except Exception as e:
                logger.error(f"Fold {fold} failed: {e}")
                self.fold_results.append({
                    'fold': fold,
                    'error': str(e)
                })
        
        if not fold_accuracies:
            return {
                'error': 'All folds failed',
                'mean_accuracy': 0.5,
                'std_accuracy': 0.0
            }
        
        # Calculate overall metrics
        predictions_arr = np.array(predictions)
        actuals_arr = np.array(actuals)
        
        metrics = {
            'mean_accuracy': np.mean(fold_accuracies),
            'std_accuracy': np.std(fold_accuracies),
            'min_accuracy': np.min(fold_accuracies),
            'max_accuracy': np.max(fold_accuracies),
            'rmse': np.sqrt(np.mean((actuals_arr - predictions_arr) ** 2)),
            'mae': np.mean(np.abs(actuals_arr - predictions_arr)),
            'directional_accuracy': self._calculate_directional_accuracy(
                actuals_arr, 
                predictions_arr
            ),
            'n_folds': len(fold_accuracies),
            'total_predictions': len(predictions)
        }
        
        return metrics
    
    def _calculate_directional_accuracy(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray
    ) -> float:
        """
        Most important metric for trading: Did we predict direction correctly?
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            
        Returns:
            Fraction of correct directional predictions
        """
        if len(y_true) < 2:
            return 0.5
        
        # Calculate returns/changes
        true_direction = np.sign(np.diff(y_true))
        pred_direction = np.sign(np.diff(y_pred))
        
        # Handle zero changes (no direction)
        mask = (true_direction != 0) & (pred_direction != 0)
        
        if mask.sum() == 0:
            return 0.5
        
        correct = (true_direction[mask] == pred_direction[mask]).sum()
        total = mask.sum()
        
        return correct / total if total > 0 else 0.5
    
    def get_realistic_performance_estimate(
        self,
        validation_metrics: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Adjust metrics to be more realistic for production.
        
        Training/validation accuracy is ALWAYS higher than real-world.
        We apply conservative adjustment factors based on research.
        
        Args:
            validation_metrics: Output from walk_forward_validation()
            
        Returns:
            Conservative, realistic performance estimates
        """
        mean_acc = validation_metrics.get('mean_accuracy', 0.5)
        min_acc = validation_metrics.get('min_accuracy', 0.5)
        max_acc = validation_metrics.get('max_accuracy', 0.5)
        std_acc = validation_metrics.get('std_accuracy', 0.0)
        
        # Apply conservative adjustment factors
        # Based on empirical research: production is ~10-20% worse
        adjusted_metrics = {
            'expected_accuracy': mean_acc * 0.85,  # Expect 15% drop
            'worst_case_accuracy': min_acc * 0.80,  # 20% drop
            'best_case_accuracy': max_acc * 0.90,  # 10% drop
            'confidence_interval': (
                max(0.5, mean_acc - 2 * std_acc),
                min(0.95, mean_acc + 2 * std_acc)
            ),
            'reliability_score': self._calculate_reliability(validation_metrics),
            'production_ready': self._is_production_ready(validation_metrics)
        }
        
        return adjusted_metrics
    
    def _calculate_reliability(self, metrics: Dict[str, float]) -> float:
        """
        How reliable is this model?
        
        High consistency across folds = High reliability.
        """
        std_acc = metrics.get('std_accuracy', 0.0)
        min_acc = metrics.get('min_accuracy', 0.5)
        
        # Low std = High consistency = High reliability
        consistency = max(0, 100 - std_acc * 200)
        
        # High min accuracy = doesn't fail badly = High reliability
        min_threshold = max(0, (min_acc - 0.5) * 200)
        
        # Combine with weights
        reliability = (consistency * 0.6 + min_threshold * 0.4)
        
        return np.clip(reliability, 0, 100)
    
    def _is_production_ready(self, metrics: Dict[str, float]) -> bool:
        """
        Conservative check for production readiness.
        """
        mean_acc = metrics.get('mean_accuracy', 0)
        min_acc = metrics.get('min_accuracy', 0)
        std_acc = metrics.get('std_accuracy', 1)
        
        # Must meet ALL criteria
        return (
            mean_acc >= 0.55 and      # Better than random
            min_acc >= 0.50 and       # No terrible folds
            std_acc <= 0.10 and       # Consistent performance
            metrics.get('n_folds', 0) >= 3  # Enough validation
        )
    
    def get_fold_results(self) -> List[Dict[str, Any]]:
        """Get detailed results for each fold."""
        return self.fold_results.copy()


class EarlyStoppingMonitor:
    """
    Prevents overfitting during model training.
    
    Monitors validation loss and stops training when it stops improving.
    
    Usage:
        monitor = EarlyStoppingMonitor(patience=15)
        
        for epoch in range(100):
            train_loss = train_one_epoch()
            val_loss = validate()
            
            if monitor.check(val_loss):
                print(f"Early stopping at epoch {epoch}")
                break
    """
    
    def __init__(
        self, 
        patience: int = 10, 
        min_delta: float = 0.001,
        mode: str = 'min'
    ):
        """
        Initialize early stopping monitor.
        
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'min' if lower is better (loss), 'max' if higher is better
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        
        if mode == 'min':
            self.best_value = float('inf')
            self.is_better = lambda curr, best: curr < best - min_delta
        else:
            self.best_value = float('-inf')
            self.is_better = lambda curr, best: curr > best + min_delta
            
        self.counter = 0
        self.should_stop = False
        self.best_epoch = 0
        self.history: List[float] = []
        
    def check(self, value: float) -> bool:
        """
        Check if training should stop.
        
        Args:
            value: Current validation metric
            
        Returns:
            True if training should stop
        """
        self.history.append(value)
        
        if self.is_better(value, self.best_value):
            self.best_value = value
            self.counter = 0
            self.best_epoch = len(self.history) - 1
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            self.should_stop = True
            
        return self.should_stop
    
    def reset(self):
        """Reset the monitor for a new training run."""
        if self.mode == 'min':
            self.best_value = float('inf')
        else:
            self.best_value = float('-inf')
        self.counter = 0
        self.should_stop = False
        self.best_epoch = 0
        self.history = []
    
    def get_summary(self) -> Dict[str, Any]:
        """Get training summary."""
        return {
            'best_value': self.best_value,
            'best_epoch': self.best_epoch,
            'total_epochs': len(self.history),
            'stopped_early': self.should_stop,
            'patience_used': self.counter
        }


class ModelPerformanceTracker:
    """
    Track model performance over time to detect degradation.
    
    Essential for production systems where models can become stale
    as market conditions change.
    
    Usage:
        tracker = ModelPerformanceTracker()
        
        # After each prediction
        tracker.add_prediction(predicted=105.5, actual=106.2)
        
        if tracker.should_retrain():
            retrain_model()
    """
    
    def __init__(self, window_size: int = 30, max_history: int = 1000):
        """
        Initialize the tracker.
        
        Args:
            window_size: Number of recent predictions to compare
            max_history: Maximum history to keep in memory
        """
        self.window_size = window_size
        self.max_history = max_history
        self.performance_history: List[Dict[str, Any]] = []
        
    def add_prediction(
        self, 
        predicted: float, 
        actual: float, 
        timestamp: Optional[pd.Timestamp] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Record a prediction outcome.
        
        Args:
            predicted: The predicted value
            actual: The actual outcome value
            timestamp: When the prediction was made
            metadata: Any additional info to track
        """
        if timestamp is None:
            timestamp = pd.Timestamp.now()
            
        # Directional correctness
        correct = (np.sign(predicted) == np.sign(actual))
        error = abs(predicted - actual)
        
        record = {
            'timestamp': timestamp,
            'predicted': predicted,
            'actual': actual,
            'correct': correct,
            'error': error,
            **(metadata or {})
        }
        
        self.performance_history.append(record)
        
        # Keep only recent history for memory efficiency
        if len(self.performance_history) > self.max_history:
            self.performance_history = self.performance_history[-self.max_history:]
    
    def get_recent_accuracy(self, n: int = 30) -> float:
        """
        Get accuracy over recent N predictions.
        
        Args:
            n: Number of recent predictions to consider
            
        Returns:
            Accuracy as fraction (0-1)
        """
        if not self.performance_history:
            return 0.5
        
        recent = self.performance_history[-n:]
        if not recent:
            return 0.5
            
        correct = sum(1 for r in recent if r['correct'])
        return correct / len(recent)
    
    def get_recent_error(self, n: int = 30) -> float:
        """Get mean absolute error of recent predictions."""
        if not self.performance_history:
            return 0.0
        
        recent = self.performance_history[-n:]
        return np.mean([r['error'] for r in recent])
    
    def is_degrading(self, threshold: float = 0.10) -> bool:
        """
        Check if model performance is degrading.
        
        Compares recent performance vs historical performance.
        
        Args:
            threshold: Minimum accuracy drop to be considered degrading
            
        Returns:
            True if performance has dropped significantly
        """
        if len(self.performance_history) < self.window_size * 2:
            return False
        
        # Recent performance
        recent = self.performance_history[-self.window_size:]
        recent_accuracy = sum(1 for r in recent if r['correct']) / len(recent)
        
        # Historical performance (previous window)
        historical = self.performance_history[-self.window_size*2:-self.window_size]
        historical_accuracy = sum(1 for r in historical if r['correct']) / len(historical)
        
        # Check for significant drop
        return (historical_accuracy - recent_accuracy) > threshold
    
    def should_retrain(
        self, 
        min_accuracy: float = 0.55,
        max_error_increase: float = 0.20
    ) -> bool:
        """
        Determine if model needs retraining.
        
        Triggers retrain if:
        1. Performance is degrading
        2. Accuracy below threshold
        3. Error increasing significantly
        
        Returns:
            True if retraining is recommended
        """
        if len(self.performance_history) < 50:
            return False
        
        recent_accuracy = self.get_recent_accuracy(30)
        degrading = self.is_degrading()
        
        # Check if error is increasing
        if len(self.performance_history) >= 60:
            recent_errors = [r['error'] for r in self.performance_history[-30:]]
            historical_errors = [r['error'] for r in self.performance_history[-60:-30]]
            
            error_increasing = (
                np.mean(recent_errors) > 
                np.mean(historical_errors) * (1 + max_error_increase)
            )
        else:
            error_increasing = False
        
        should_retrain = (
            (recent_accuracy < min_accuracy) or 
            degrading or 
            error_increasing
        )
        
        if should_retrain:
            logger.warning(
                f"Retrain recommended: accuracy={recent_accuracy:.2%}, "
                f"degrading={degrading}, error_increasing={error_increasing}"
            )
        
        return should_retrain
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        if not self.performance_history:
            return {'error': 'No predictions tracked'}
        
        recent_accuracy = self.get_recent_accuracy(30)
        recent_error = self.get_recent_error(30)
        
        all_accuracies = [r['correct'] for r in self.performance_history]
        all_errors = [r['error'] for r in self.performance_history]
        
        return {
            'total_predictions': len(self.performance_history),
            'recent_accuracy_30': recent_accuracy,
            'recent_avg_error_30': recent_error,
            'overall_accuracy': sum(all_accuracies) / len(all_accuracies),
            'overall_avg_error': np.mean(all_errors),
            'is_degrading': self.is_degrading(),
            'should_retrain': self.should_retrain(),
            'oldest_prediction': self.performance_history[0]['timestamp'],
            'newest_prediction': self.performance_history[-1]['timestamp']
        }
    
    def reset(self):
        """Clear all tracking history."""
        self.performance_history = []


class OverfitDetector:
    """
    Detect potential overfitting during training.
    """
    
    def __init__(self, gap_threshold: float = 0.10):
        """
        Args:
            gap_threshold: Max acceptable gap between train and val performance
        """
        self.gap_threshold = gap_threshold
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        
    def update(self, train_loss: float, val_loss: float):
        """Record training and validation losses."""
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
    
    def is_overfitting(self) -> bool:
        """
        Check if model is overfitting.
        
        Signs of overfitting:
        1. Train loss decreasing while val loss increasing
        2. Large gap between train and val loss
        """
        if len(self.train_losses) < 5:
            return False
        
        # Check recent trend (last 5 epochs)
        recent_train = self.train_losses[-5:]
        recent_val = self.val_losses[-5:]
        
        # Train improving, val getting worse
        train_improving = recent_train[-1] < recent_train[0]
        val_degrading = recent_val[-1] > recent_val[0] * 1.05
        
        # Large gap between train and val
        gap = abs(recent_val[-1] - recent_train[-1])
        large_gap = gap > self.gap_threshold
        
        return (train_improving and val_degrading) or large_gap
    
    def get_train_val_gap(self) -> float:
        """Get current gap between training and validation loss."""
        if not self.train_losses or not self.val_losses:
            return 0.0
        return abs(self.val_losses[-1] - self.train_losses[-1])
    
    def reset(self):
        """Reset for new training run."""
        self.train_losses = []
        self.val_losses = []
