"""
Ensemble Prediction Scorer Module
Combines multiple ML models and technical indicators with Bayesian-Meta reliability tracking.

Data Storage Structure:
data/predictions/ensemble/history.json
data/predictions/ensemble/by_model/
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import json
import os
import logging
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import BaggingClassifier
from src import config
from src.data.sentiment.news_reasoner import get_news_reasoner
from .uncertainty_quantification import get_ensemble_uncertainty, UncertaintyQuantifier
import pickle

logger = logging.getLogger(__name__)


class PredictionStore:
    """
    Store and validate predictions for accuracy tracking.

    Features:
    - Human-readable JSON with metadata
    - Organized folder structure
    - Per-model performance tracking
    """

    def __init__(self, storage_dir: str = "data/predictions/ensemble"):
        self.storage_dir = storage_dir
        self.history_file = os.path.join(storage_dir, "history.json")
        self.by_model_dir = os.path.join(storage_dir, "by_model")
        self.predictions: List[Dict] = []

        # Ensure directories exist
        os.makedirs(self.storage_dir, exist_ok=True)
        os.makedirs(self.by_model_dir, exist_ok=True)

        self._load()

    def _load(self):
        """Load predictions from file."""
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Handle both old format (list) and new format (dict with metadata)
                    if isinstance(data, dict) and 'data' in data:
                        self.predictions = data['data']
                    else:
                        self.predictions = data
            except Exception as e:
                logger.error(
                    f"[PredictionStore] Error loading {self.history_file}: {e}")
                self.predictions = []

    def _save(self):
        """Save predictions to file with human-readable formatting."""
        try:
            # Create output with metadata
            output = {
                "_metadata": {
                    "description": "Ensemble model prediction history",
                    "last_updated": datetime.now().isoformat(),
                    "total_predictions": len(self.predictions),
                    "validated_count": sum(1 for p in self.predictions if p.get('validated')),
                    "version": "2.0"
                },
                "data": self.predictions[-1000:]  # Keep last 1000
            }

            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(output, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(
                f"[PredictionStore] Error saving {self.history_file}: {e}")

    def record_prediction(
        self,
        symbol: str,
        model_name: str,
        predicted_direction: str,
        predicted_change_pct: float,
        confidence: float,
        current_price: float,
        regime: str = "sideways",
        volatility: float = 0.2
    ):
        """Record a new prediction."""
        prediction = {
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol,
            "model": model_name,
            "direction": predicted_direction,
            "change_pct": predicted_change_pct,
            "confidence": confidence,
            "price_at_prediction": current_price,
            "regime": regime,
            "volatility": volatility,
            "validated": False,
            "was_correct": None
        }
        self.predictions.append(prediction)
        self._save()

    def validate_predictions(self, symbol: str, current_price: float, days_ago: int = 5):
        """Validate predictions made N days ago."""
        cutoff = datetime.now() - timedelta(days=days_ago)

        for pred in self.predictions:
            if pred["symbol"] == symbol and not pred["validated"]:
                pred_time = datetime.fromisoformat(pred["timestamp"])
                if pred_time < cutoff:
                    # Check if prediction was correct
                    actual_change = (
                        current_price - pred["price_at_prediction"]) / pred["price_at_prediction"] * 100
                    predicted_direction = pred["direction"]

                    if predicted_direction == "Bullish" and actual_change > 0:
                        pred["was_correct"] = True
                    elif predicted_direction == "Bearish" and actual_change < 0:
                        pred["was_correct"] = True
                    else:
                        pred["was_correct"] = False

                    pred["validated"] = True
                    pred["actual_change_pct"] = actual_change

        self._save()

    def get_model_accuracy(self, model_name: str, symbol: str = None, lookback_days: int = 30) -> Dict[str, Any]:
        """Get accuracy stats for a model."""
        cutoff = datetime.now() - timedelta(days=lookback_days)

        relevant = [
            p for p in self.predictions
            if p["model"] == model_name
            and p["validated"]
            and datetime.fromisoformat(p["timestamp"]) > cutoff
            and (symbol is None or p["symbol"] == symbol)
        ]

        if not relevant:
            return {"accuracy": 0.5, "sample_size": 0, "confidence_calibration": 1.0}

        correct = sum(1 for p in relevant if p["was_correct"])
        total = len(relevant)
        accuracy = correct / total

        # Calculate confidence calibration
        avg_confidence = np.mean([p["confidence"] for p in relevant])
        calibration = accuracy / \
            (avg_confidence / 100) if avg_confidence > 0 else 1.0

        return {
            "accuracy": round(accuracy, 3),
            "sample_size": total,
            "correct": correct,
            "confidence_calibration": round(calibration, 2)
        }

    def get_training_data(self, model_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """Extract training features and labels for the Meta-Learner."""
        relevant = [p for p in self.predictions if p["model"]
                    == model_name and p["validated"]]
        if not relevant:
            return np.array([]), np.array([])

        X = []
        y = []
        regime_map = {"bullish": 1, "bearish": -1, "sideways": 0}

        for p in relevant:
            # Features: [confidence, regime_code, volatility]
            X.append([
                p.get("confidence", 50) / 100,
                regime_map.get(p.get("regime", "sideways"), 0),
                p.get("volatility", 0.2)
            ])
            y.append(1 if p.get("was_correct") else 0)

        return np.array(X), np.array(y)


class MetaLearner:
    """
    Meta-Learning Aggregator with Bayesian-style uncertainty tracking.

    Uses a Bagging ensemble of Logistic Regressors to estimate both
    the probability of correctness and the uncertainty (variance) of that estimate.
    """

    def __init__(self, model_name: str, cache_dir: str = "data/models/meta"):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.model_path = os.path.join(cache_dir, f"{model_name}_meta.pkl")

        # Use a bagging ensemble for uncertainty estimation

        self.base_clf = LogisticRegression(max_iter=100)
        self.clf = BaggingClassifier(
            estimator=self.base_clf,
            n_estimators=10,
            max_samples=0.8,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.is_trained = False
        self._last_uncertainty = 0.5

        os.makedirs(cache_dir, exist_ok=True)
        self._load_model()

    def _load_model(self):
        """Load trained model from disk with legacy support and resilience."""
        if not os.path.exists(self.model_path):
            return  # Normal state for new system

        try:
            with open(self.model_path, 'rb') as f:
                data = pickle.load(f)

                # Support both 'model' (new) and 'clf' (legacy) keys
                self.clf = data.get('model', data.get('clf'))
                self.scaler = data.get('scaler')
                # Default to True if loaded
                self.is_trained = data.get('is_trained', True)

                if self.clf and self.scaler:
                    # Validate model type - must have estimators_ for BaggingClassifier
                    if not hasattr(self.clf, 'estimators_'):
                        logger.warning(
                            f"Loaded model for {self.model_name} is invalid (missing estimators_). Resetting.")
                        self.clf = BaggingClassifier(
                            estimator=self.base_clf,
                            n_estimators=10,
                            max_samples=0.8,
                            random_state=42
                        )
                        self.is_trained = False
                    else:
                        logger.info(
                            f"Loaded Meta-Learner for {self.model_name}")
                else:
                    logger.warning(
                        f"Meta-Learner file for {self.model_name} is invalid or empty.")
        except (EOFError, KeyError, pickle.UnpicklingError, AttributeError) as e:
            logger.warning(
                f"Meta-Learner file for {self.model_name} is corrupted or incompatible: {e}. It will be re-trained.")
            # Reset on error
            self.clf = BaggingClassifier(
                estimator=self.base_clf,
                n_estimators=10,
                max_samples=0.8,
                random_state=42
            )
            self.is_trained = False
        except Exception as e:
            logger.error(
                f"Unexpected error loading Meta-Learner {self.model_name}: {e}")

    def _save_model(self):
        try:
            with open(self.model_path, 'wb') as f:
                pickle.dump({
                    'model': self.clf,
                    'scaler': self.scaler,
                    'is_trained': self.is_trained
                }, f)
        except Exception as e:
            logger.error(f"Failed to save Meta-Learner {self.model_name}: {e}")

    def train(self, X: np.ndarray, y: np.ndarray):
        if len(X) < 10:
            return

        try:
            # Check if we have at least 2 classes (Correct and Incorrect)
            unique_classes = np.unique(y)
            if len(unique_classes) < 2:
                # If everything is correct, model is perfectly reliable anyway.
                # If everything is incorrect, model is perfectly unreliable.
                # We skip training as classifier needs 2 classes.
                return

            X_scaled = self.scaler.fit_transform(X)
            self.clf.fit(X_scaled, y)
            self.is_trained = True
            self._save_model()
            logger.info(
                f"Successfully trained Meta-Learner for {self.model_name} on {len(X)} samples")
        except Exception as e:
            logger.error(
                f"Training failed for Meta-Learner {self.model_name}: {e}")

    def predict_reliability(self, confidence: float, regime: str, volatility: float) -> Tuple[float, float]:
        """
        Predict probability of correctness and uncertainty.

        Returns:
            Tuple of (probability_correct, uncertainty)
        """
        if not self.is_trained:
            return 0.5, 0.5

        try:
            regime_map = {"bullish": 1, "bearish": -1, "sideways": 0}
            features = np.array([[
                confidence / 100,
                regime_map.get(regime, 0),
                volatility
            ]])

            X_scaled = self.scaler.transform(features)

            # Use ensemble variance as uncertainty
            # Each estimator gives a probability
            probas = [est.predict_proba(X_scaled)[0, 1]
                      for est in self.clf.estimators_]

            mean_proba = np.mean(probas)
            uncertainty = np.std(probas) * 2  # Spread (0 to ~1)

            self._last_uncertainty = float(uncertainty)
            return float(mean_proba), float(uncertainty)
        except Exception as e:
            logger.error(
                f"Reliability prediction failed for {self.model_name}: {e}")
            return 0.5, 0.5


class EnsembleScorer:
    """
    Combines multiple prediction sources into a unified score.

    Features stable weight management to prevent dramatic fluctuations:
    - Bayesian smoothing for gradual weight transitions
    - Minimum weight floors to prevent model abandonment
    - Maximum weight adjustment limits per update cycle
    - Market regime-aware weight profiles
    """

    # Updated weights including ALL 12+ models (2024 comprehensive ensemble)
    # Deep Learning Models (45%): CNN-LSTM, Attention, TCN, LSTM, GRU, N-BEATS
    # Gradient Boosting (25%): XGBoost, LightGBM, Random Forest
    # Support Models (20%): SVM, Momentum, Technical
    # Fundamental (5%): Long-term value assessment
    DEFAULT_WEIGHTS = {
        # Deep Learning Models (45%)
        "cnn_lstm": 0.11,       # Hybrid CNN-LSTM (optimized TensorFlow)
        "attention": 0.10,       # Transformer with multi-head attention
        # Temporal Convolutional Network (fast, long-range)
        "tcn": 0.09,
        "lstm": 0.09,           # Traditional LSTM
        "gru": 0.08,            # GRU (faster alternative)
        "nbeats": 0.08,         # N-BEATS (interpretable forecasting)

        # Gradient Boosting (25%)
        "xgboost": 0.11,        # Strong tree-based learner
        "lightgbm": 0.10,       # Often outperforms XGBoost, faster training
        "random_forest": 0.09,   # Ensemble of decision trees

        # Support Models (20%)
        "svm": 0.07,            # Support Vector Machine
        "momentum": 0.07,       # Multi-timeframe momentum with regime detection
        "technical": 0.06,      # Multi-indicator technical signals

        # Fundamental (5%)
        "fundamental": 0.04     # Valuation + quality + growth scoring
    }

    # Market regime-specific weight adjustments
    # Different models perform better in different market conditions
    REGIME_WEIGHT_PROFILES = {
        "bullish": {
            "momentum": 1.3,      # Momentum works well in bull markets
            "lstm": 1.2,          # Trend-following models benefit
            "gru": 1.2,
            "xgboost": 1.1,
            "technical": 0.9,     # Less effective when trends dominate
            "fundamental": 0.8,   # Valuations often stretched
        },
        "bearish": {
            "fundamental": 1.4,   # Value matters in bear markets
            "svm": 1.2,           # Classification works for regime detection
            "random_forest": 1.2,
            "momentum": 0.7,      # Momentum can lag in reversals
            "lstm": 0.9,
        },
        "sideways": {
            "technical": 1.3,     # Mean reversion works in ranges
            "svm": 1.2,
            "random_forest": 1.1,
            "momentum": 0.7,      # False signals in ranging markets
            "lstm": 0.9,
        }
    }

    # Stability configuration constants
    MIN_WEIGHT_FLOOR = 0.02           # No model weight can go below 2%
    MAX_WEIGHT_MULTIPLIER = 2.0       # Maximum adjustment: 2x default
    MIN_WEIGHT_MULTIPLIER = 0.5       # Minimum adjustment: 0.5x default
    # EMA smoothing factor (lower = more stable)
    SMOOTHING_ALPHA = 0.3
    MIN_SAMPLES_FOR_ADJUSTMENT = 10   # Minimum predictions before adjusting weights
    # Prior accuracy (50%) for Bayesian smoothing
    BAYESIAN_PRIOR_WEIGHT = 0.5
    BAYESIAN_PRIOR_SAMPLES = 10       # Equivalent sample count for prior

    def __init__(self, prediction_store: Optional[PredictionStore] = None):
        self.weights = self.DEFAULT_WEIGHTS.copy()
        self.store = prediction_store or PredictionStore()
        self._previous_weights = self.DEFAULT_WEIGHTS.copy()  # For EMA smoothing
        self._current_regime = "sideways"  # Default regime
        self._current_volatility = 0.2
        # Track last predictions for correlation
        self._model_history: Dict[str, List[float]] = {}
        self._correlation_penalties: Dict[str, float] = {}

        # Initialize Meta-Learners for each model
        self.meta_learners = {
            model: MetaLearner(model)
            for model in self.DEFAULT_WEIGHTS.keys()
        }

        # === NEW: Online Learning System ===
        # Gradient descent with momentum for online weight updates
        self.learning_rate = 0.05  # Conservative learning rate
        self.momentum_factor = 0.9  # Momentum for smooth updates
        self.model_gradients: Dict[str, float] = {model: 0.0 for model in self.DEFAULT_WEIGHTS.keys()}
        self.model_momentum: Dict[str, float] = {model: 0.0 for model in self.DEFAULT_WEIGHTS.keys()}

        # Regime-specific meta-learners
        self.regime_meta_learners: Dict[str, Dict[str, MetaLearner]] = {
            'bullish': {},
            'bearish': {},
            'sideways': {}
        }

        # Performance tracking for online learning
        self.online_performance: Dict[str, Dict[str, Any]] = {
            model: {
                'correct': 0,
                'total': 0,
                'recent_accuracy': 0.5,
                'weight_updates': 0
            }
            for model in self.DEFAULT_WEIGHTS.keys()
        }

    def detect_market_regime(self, stock_data: Optional[pd.DataFrame] = None) -> Tuple[str, float]:
        """
        Detect current market regime and volatility.

        Returns: (regime_name, volatility)
        """
        if stock_data is None or len(stock_data) < 50:
            return "sideways", 0.2

        try:
            # Calculate 20-day and 50-day moving averages
            close = stock_data['Close'].values
            sma_20 = np.mean(close[-20:])
            sma_50 = np.mean(close[-50:])
            current_price = close[-1]

            # Calculate recent volatility
            returns = np.diff(close[-20:]) / close[-21:-1]
            volatility = np.std(returns) * np.sqrt(252)  # Annualized

            # Store current volatility
            self._current_volatility = float(volatility)

            # Calculate trend strength
            price_change_20d = (current_price - close[-20]) / close[-20] * 100

            # Determine regime
            if current_price > sma_20 > sma_50 and price_change_20d > 3:
                return "bullish", volatility
            elif current_price < sma_20 < sma_50 and price_change_20d < -3:
                return "bearish", volatility
            else:
                return "sideways", volatility
        except Exception:
            return "sideways", 0.2

    def _apply_bayesian_smoothing(self, accuracy: float, sample_size: int) -> float:
        """
        Apply Bayesian smoothing to accuracy estimates.

        This prevents extreme weight changes based on small sample sizes
        by incorporating a prior belief of 50% accuracy.
        """
        # Weighted average of prior and observed accuracy
        # More samples = more weight to observed accuracy
        total_samples = sample_size + self.BAYESIAN_PRIOR_SAMPLES
        smoothed_accuracy = (
            (self.BAYESIAN_PRIOR_WEIGHT * self.BAYESIAN_PRIOR_SAMPLES) +
            (accuracy * sample_size)
        ) / total_samples

        return smoothed_accuracy

    def _apply_ema_smoothing(self, new_weights: Dict[str, float]) -> Dict[str, float]:
        """
        Apply Exponential Moving Average smoothing to weight updates.

        This prevents sudden jumps in weights by blending with previous weights.
        """
        smoothed_weights = {}
        for model in self.DEFAULT_WEIGHTS.keys():
            old_weight = self._previous_weights.get(
                model, self.DEFAULT_WEIGHTS[model])
            new_weight = new_weights.get(model, self.DEFAULT_WEIGHTS[model])
            # EMA formula: new = alpha * current + (1 - alpha) * previous
            smoothed_weights[model] = (
                self.SMOOTHING_ALPHA * new_weight +
                (1 - self.SMOOTHING_ALPHA) * old_weight
            )
        return smoothed_weights

    def _apply_weight_bounds(self, weights: Dict[str, float]) -> Dict[str, float]:
        """
        Apply minimum floor and maximum multiplier constraints to weights.

        This prevents any model from being completely ignored or over-weighted.
        """
        bounded_weights = {}
        for model, weight in weights.items():
            default = self.DEFAULT_WEIGHTS.get(model, 0.1)
            min_allowed = max(self.MIN_WEIGHT_FLOOR,
                              default * self.MIN_WEIGHT_MULTIPLIER)
            max_allowed = default * self.MAX_WEIGHT_MULTIPLIER
            bounded_weights[model] = max(min_allowed, min(max_allowed, weight))
        return bounded_weights

    def _apply_regime_adjustments(self, weights: Dict[str, float], regime: str) -> Dict[str, float]:
        """
        Apply market regime-specific weight adjustments.

        Different models perform better in different market conditions.
        """
        if regime not in self.REGIME_WEIGHT_PROFILES:
            return weights

        regime_multipliers = self.REGIME_WEIGHT_PROFILES[regime]
        adjusted_weights = {}

        for model, weight in weights.items():
            multiplier = regime_multipliers.get(model, 1.0)
            adjusted_weights[model] = weight * multiplier

        return adjusted_weights

    def update_weights_from_accuracy(
        self,
        symbol: str = None,
        lookback_days: int = 30,
        stock_data: Optional[pd.DataFrame] = None
    ):
        """
        Adjust weights based on historical accuracy with stability mechanisms.

        Features:
        - Bayesian smoothing prevents overreaction to small samples
        - EMA smoothing ensures gradual weight transitions
        - Weight bounds prevent extreme adjustments
        - Regime awareness adjusts for market conditions
        """
        # Detect current market regime
        self._current_regime, self._current_volatility = self.detect_market_regime(
            stock_data)

        accuracies = {}
        sample_sizes = {}

        for model in self.weights.keys():
            # Train Meta-Learner if we have symbol-specific data or global data
            X, y = self.store.get_training_data(model)
            if len(X) >= 20:
                self.meta_learners[model].train(X, y)

            stats = self.store.get_model_accuracy(model, symbol, lookback_days)
            sample_sizes[model] = stats["sample_size"]

            # Only adjust if we have enough samples (increased threshold)
            if stats["sample_size"] >= self.MIN_SAMPLES_FOR_ADJUSTMENT:
                # Apply Bayesian smoothing to the accuracy
                smoothed_accuracy = self._apply_bayesian_smoothing(
                    stats["accuracy"],
                    stats["sample_size"]
                )
                accuracies[model] = smoothed_accuracy
            else:
                # Use prior (default to 0.5 = neutral)
                accuracies[model] = self.BAYESIAN_PRIOR_WEIGHT

        # === NEW: Train regime-specific meta-learners ===
        # Do this periodically to specialize models by market condition
        try:
            regime_training_stats = self.train_regime_specific_meta_learners(
                min_samples_per_regime=50
            )
            logger.debug(f"Regime-specific meta-learner training: {regime_training_stats}")
        except Exception as e:
            logger.warning(f"Regime-specific meta-learner training failed: {e}")

        # Calculate raw accuracy-based weights
        total_accuracy = sum(accuracies.values())
        if total_accuracy > 0:
            raw_weights = {}
            for model in self.weights:
                # Blend accuracy-based weight with default weight
                # Using 70% default, 30% accuracy to prioritize stability
                accuracy_weight = accuracies.get(model, 0.5) / total_accuracy
                default_weight = self.DEFAULT_WEIGHTS[model]
                raw_weights[model] = 0.7 * \
                    default_weight + 0.3 * accuracy_weight
        else:
            raw_weights = self.DEFAULT_WEIGHTS.copy()

        # Apply regime-specific adjustments
        regime_adjusted = self._apply_regime_adjustments(
            raw_weights, self._current_regime)

        # Apply weight bounds (floor and ceiling)
        bounded = self._apply_weight_bounds(regime_adjusted)

        # Apply EMA smoothing for gradual transitions
        smoothed = self._apply_ema_smoothing(bounded)

        # Normalize to sum to 1
        weight_total = sum(smoothed.values())
        if weight_total > 0:
            for model in smoothed:
                smoothed[model] /= weight_total

        # Store previous weights for next EMA calculation
        self._previous_weights = self.weights.copy()

        # Update current weights
        self.weights = smoothed

    def update_weights_online(
        self,
        model_predictions: Dict[str, Dict[str, Any]],
        actual_outcome: str,
        current_price: float,
        future_price: float,
        symbol: str = None
    ) -> Dict[str, Any]:
        """
        Online learning: Update weights based on prediction outcome (real-time).

        Uses gradient descent with momentum for smooth, continuous learning.

        Args:
            model_predictions: Dict of {model_name: prediction_dict}
            actual_outcome: Actual direction ('Bullish', 'Bearish', 'Neutral')
            current_price: Price at prediction time
            future_price: Price after prediction horizon (e.g., 5 days later)
            symbol: Stock symbol for logging

        Returns:
            Dict with update statistics
        """
        # Determine actual direction from price movement
        price_change_pct = ((future_price - current_price) / current_price) * 100

        if price_change_pct > 2.0:
            actual_direction = "Bullish"
        elif price_change_pct < -2.0:
            actual_direction = "Bearish"
        else:
            actual_direction = "Neutral"

        # Override with provided outcome if available
        if actual_outcome:
            actual_direction = actual_outcome

        updates = {}
        weight_changes = {}

        for model_name, pred in model_predictions.items():
            if model_name not in self.weights:
                continue

            predicted_direction = pred.get('direction', 'Neutral')
            confidence = pred.get('confidence', 50) / 100.0

            # Calculate loss: 0 if correct, 1 if wrong
            is_correct = (predicted_direction == actual_direction)
            loss = 0.0 if is_correct else 1.0

            # Weight confidence into loss (higher confidence mistakes hurt more)
            weighted_loss = loss * confidence

            # Update online performance tracking
            self.online_performance[model_name]['total'] += 1
            if is_correct:
                self.online_performance[model_name]['correct'] += 1

            # Calculate recent accuracy with exponential decay
            alpha = 0.1  # Decay factor
            old_acc = self.online_performance[model_name]['recent_accuracy']
            new_acc = alpha * (1.0 if is_correct else 0.0) + (1 - alpha) * old_acc
            self.online_performance[model_name]['recent_accuracy'] = new_acc

            # Gradient descent with momentum update
            gradient = weighted_loss

            # Update momentum term
            old_momentum = self.model_momentum.get(model_name, 0.0)
            new_momentum = (
                self.momentum_factor * old_momentum +
                (1 - self.momentum_factor) * gradient
            )
            self.model_momentum[model_name] = new_momentum

            # Update weight using momentum
            current_weight = self.weights.get(model_name, 0.1)
            delta_weight = -self.learning_rate * new_momentum

            # Apply update
            new_weight = current_weight + delta_weight

            # Clip to reasonable range (2% to 20%)
            new_weight = max(0.02, min(0.20, new_weight))

            weight_changes[model_name] = {
                'old_weight': round(current_weight, 4),
                'new_weight': round(new_weight, 4),
                'delta': round(delta_weight, 4),
                'was_correct': is_correct,
                'confidence': round(confidence * 100, 1)
            }

            self.weights[model_name] = new_weight
            self.online_performance[model_name]['weight_updates'] += 1

            updates[model_name] = {
                'correct': is_correct,
                'loss': round(weighted_loss, 4),
                'gradient': round(gradient, 4),
                'momentum': round(new_momentum, 4),
                'weight_change': round(delta_weight, 4)
            }

        # Normalize weights to sum to 1.0
        total_weight = sum(self.weights.values())
        if total_weight > 0:
            for model_name in self.weights:
                self.weights[model_name] /= total_weight

        logger.info(
            f"[{symbol}] Online weight update: actual={actual_direction}, "
            f"price_change={price_change_pct:.2f}%, "
            f"models_updated={len(updates)}"
        )

        return {
            'actual_direction': actual_direction,
            'price_change_pct': round(price_change_pct, 2),
            'models_updated': len(updates),
            'weight_changes': weight_changes,
            'update_details': updates,
            'normalized_weights': {k: round(v, 4) for k, v in self.weights.items()}
        }

    def train_regime_specific_meta_learners(
        self,
        min_samples_per_regime: int = 50
    ) -> Dict[str, Any]:
        """
        Train separate meta-learners for each market regime (bull/bear/sideways).

        This allows models to specialize based on market conditions.

        Args:
            min_samples_per_regime: Minimum predictions needed per regime

        Returns:
            Dict with training statistics per regime
        """
        training_stats = {
            'bullish': {},
            'bearish': {},
            'sideways': {}
        }

        for regime in ['bullish', 'bearish', 'sideways']:
            regime_predictions = [
                p for p in self.store.predictions
                if p.get('validated') and p.get('regime') == regime
            ]

            if len(regime_predictions) < min_samples_per_regime:
                logger.info(
                    f"Skipping regime-specific training for {regime}: "
                    f"only {len(regime_predictions)} samples (need {min_samples_per_regime})"
                )
                training_stats[regime] = {
                    'status': 'insufficient_data',
                    'samples': len(regime_predictions),
                    'required': min_samples_per_regime
                }
                continue

            # Train meta-learner for each model in this regime
            models_trained = 0
            for model_name in self.DEFAULT_WEIGHTS.keys():
                # Filter predictions for this model in this regime
                model_regime_preds = [
                    p for p in regime_predictions
                    if p.get('model') == model_name
                ]

                if len(model_regime_preds) < 10:
                    continue

                # Prepare training data
                X = []
                y = []
                regime_map = {"bullish": 1, "bearish": -1, "sideways": 0}

                for p in model_regime_preds:
                    X.append([
                        p.get("confidence", 50) / 100,
                        regime_map.get(p.get("regime", "sideways"), 0),
                        p.get("volatility", 0.2)
                    ])
                    y.append(1 if p.get("was_correct") else 0)

                X = np.array(X)
                y = np.array(y)

                # Check if we have both classes
                if len(np.unique(y)) < 2:
                    continue

                # Create regime-specific meta-learner
                if model_name not in self.regime_meta_learners[regime]:
                    self.regime_meta_learners[regime][model_name] = MetaLearner(
                        f"{model_name}_{regime}"
                    )

                # Train it
                self.regime_meta_learners[regime][model_name].train(X, y)
                models_trained += 1

            training_stats[regime] = {
                'status': 'trained',
                'samples': len(regime_predictions),
                'models_trained': models_trained,
                'accuracy': round(
                    sum(1 for p in regime_predictions if p.get('was_correct')) / len(regime_predictions),
                    3
                )
            }

            logger.info(
                f"Trained regime-specific meta-learners for {regime}: "
                f"{models_trained} models, {len(regime_predictions)} samples, "
                f"regime accuracy: {training_stats[regime]['accuracy']:.1%}"
            )

        return training_stats

    def get_regime_specific_reliability(
        self,
        model_name: str,
        confidence: float,
        regime: str,
        volatility: float
    ) -> Tuple[float, float]:
        """
        Get reliability prediction using regime-specific meta-learner if available.

        Falls back to global meta-learner if regime-specific not trained.

        Args:
            model_name: Name of model
            confidence: Model confidence (0-100)
            regime: Current market regime
            volatility: Current volatility

        Returns:
            (reliability, uncertainty) tuple
        """
        # Try regime-specific meta-learner first
        if regime in self.regime_meta_learners:
            if model_name in self.regime_meta_learners[regime]:
                regime_ml = self.regime_meta_learners[regime][model_name]
                if regime_ml.is_trained:
                    return regime_ml.predict_reliability(confidence, regime, volatility)

        # Fallback to global meta-learner
        if model_name in self.meta_learners:
            return self.meta_learners[model_name].predict_reliability(
                confidence, regime, volatility
            )

        # Default
        return 0.5, 0.5

    def record_prediction_outcome(
        self,
        prediction_id: str,
        actual_price: float,
        predicted_price: float,
        prediction_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Record the actual outcome of a prediction for continuous learning.

        This is called after the prediction horizon has passed to enable
        online weight updates.

        Args:
            prediction_id: Unique ID for prediction
            actual_price: Actual stock price after horizon
            predicted_price: Price at time of prediction
            prediction_data: Original prediction dict with model predictions

        Returns:
            Dict with outcome statistics
        """
        # Extract model predictions from ensemble result
        model_predictions = prediction_data.get('model_results', [])

        if not model_predictions:
            logger.warning(f"No model predictions found for outcome recording")
            return {'error': 'no_model_predictions'}

        # Convert model_results list to predictions dict
        predictions_dict = {}
        for model_result in model_predictions:
            model_name = model_result.get('model')
            if model_name:
                predictions_dict[model_name] = {
                    'direction': model_result.get('direction'),
                    'confidence': model_result.get('confidence'),
                    'change_pct': 0  # Not stored in model_results
                }

        # Determine actual outcome
        predicted_direction = prediction_data.get('direction', 'Neutral')
        price_change = ((actual_price - predicted_price) / predicted_price) * 100

        if price_change > 2.0:
            actual_direction = "Bullish"
        elif price_change < -2.0:
            actual_direction = "Bearish"
        else:
            actual_direction = "Neutral"

        was_correct = (predicted_direction == actual_direction)

        # Update weights online
        update_result = self.update_weights_online(
            model_predictions=predictions_dict,
            actual_outcome=actual_direction,
            current_price=predicted_price,
            future_price=actual_price,
            symbol=prediction_data.get('symbol', 'UNKNOWN')
        )

        return {
            'prediction_id': prediction_id,
            'was_correct': was_correct,
            'predicted_direction': predicted_direction,
            'actual_direction': actual_direction,
            'price_change_pct': round(price_change, 2),
            'weight_update': update_result,
            'timestamp': datetime.now().isoformat()
        }

    def get_online_learning_stats(self) -> Dict[str, Any]:
        """
        Get statistics about online learning performance.

        Returns:
            Dict with learning stats per model
        """
        stats = {}

        for model_name, perf in self.online_performance.items():
            total = perf['total']
            correct = perf['correct']
            accuracy = (correct / total) if total > 0 else 0.5

            stats[model_name] = {
                'online_accuracy': round(accuracy, 3),
                'total_updates': total,
                'correct_predictions': correct,
                'recent_accuracy': round(perf['recent_accuracy'], 3),
                'weight_updates': perf['weight_updates'],
                'current_weight': round(self.weights.get(model_name, 0.1), 4)
            }

        # Add regime-specific meta-learner info
        regime_info = {}
        for regime in ['bullish', 'bearish', 'sideways']:
            trained_models = list(self.regime_meta_learners[regime].keys())
            regime_info[regime] = {
                'models_trained': len(trained_models),
                'models': trained_models
            }

        return {
            'model_stats': stats,
            'regime_meta_learners': regime_info,
            'learning_rate': self.learning_rate,
            'momentum_factor': self.momentum_factor,
            'total_online_updates': sum(perf['total'] for perf in self.online_performance.values())
        }

    def get_weight_stability_info(self) -> Dict[str, Any]:
        """
        Get information about weight stability and adjustments.

        Returns metrics useful for debugging weight behavior.
        """
        weight_changes = {}
        for model in self.weights:
            default = self.DEFAULT_WEIGHTS[model]
            current = self.weights[model]
            change_pct = ((current - default) / default) * 100
            weight_changes[model] = round(change_pct, 1)

        return {
            "current_regime": self._current_regime,
            "weight_changes_from_default_pct": weight_changes,
            "stability_config": {
                "smoothing_alpha": self.SMOOTHING_ALPHA,
                "min_samples_required": self.MIN_SAMPLES_FOR_ADJUSTMENT,
                "weight_bounds": f"{self.MIN_WEIGHT_MULTIPLIER}x - {self.MAX_WEIGHT_MULTIPLIER}x",
                "min_floor": self.MIN_WEIGHT_FLOOR
            }
        }

    def apply_sentiment_multiplier(self, symbol: str, direction: str, confidence: float) -> float:
        """
        Apply sentiment-based confidence multiplier.

        Args:
            symbol: Stock symbol
            direction: Prediction direction ('Bullish', 'Bearish', 'Neutral')
            confidence: Base confidence score

        Returns:
            Adjusted confidence score
        """
        try:
            from src.data.sentiment.storage.cache_manager import get_cache_manager
            cache = get_cache_manager()

            # Try to get cached sentiment
            cached_sentiment = cache.get_api_response(
                'sentiment_analyze', {'symbol': symbol})

            if cached_sentiment:
                overall_sent = cached_sentiment.get('overall_sentiment', {})
                sent_label = overall_sent.get('label', 'neutral')
                sent_confidence = overall_sent.get('confidence', 0.0)

                # Only apply multiplier if sentiment has high confidence
                if sent_confidence > 0.7:
                    # Convert sentiment label to match direction format
                    if sent_label == 'positive':
                        sent_direction = 'Bullish'
                    elif sent_label == 'negative':
                        sent_direction = 'Bearish'
                    else:
                        sent_direction = 'Neutral'

                    # If sentiment agrees with prediction, boost confidence
                    if sent_direction == direction and direction != 'Neutral':
                        multiplier = 1.15  # 15% boost
                        logger.info(
                            f"[{symbol}] Sentiment AGREES ({sent_direction}): boosting confidence by 15%")
                        return min(100, confidence * multiplier)

                    # If sentiment disagrees, penalize confidence
                    elif sent_direction != direction and direction != 'Neutral' and sent_direction != 'Neutral':
                        multiplier = 0.85  # 15% penalty
                        logger.info(
                            f"[{symbol}] Sentiment DISAGREES ({sent_direction} vs {direction}): reducing confidence by 15%")
                        return confidence * multiplier

            # No adjustment if no sentiment data or low confidence
            return confidence

        except Exception as e:
            logger.debug(f"Sentiment multiplier error: {e}")
            return confidence

    def calculate_ensemble_score(
        self,
        predictions: Dict[str, Dict[str, Any]],
        symbol: str = None,  # NEW: Optional symbol for sentiment integration
        stock_data: Optional[pd.DataFrame] = None  # NEW: For uncertainty quantification
    ) -> Dict[str, Any]:
        """
        Calculate ensemble prediction from multiple model outputs.

        Args:
            predictions: Dict mapping model name to prediction dict.
                         Each prediction should have 'direction', 'confidence', 'change_pct'

        Returns:
            Ensemble prediction with aggregated score
        """
        if not predictions:
            return {
                "direction": "Neutral",
                "confidence": 0,
                "change_pct": 0,
                "models_agree": False,
                "explanation": "No predictions available"
            }

        bullish_score = 0
        bearish_score = 0
        total_weight = 0
        weighted_change = 0
        model_results = []

        # === NEW: Correlation Pruning (Diversity Filter) ===
        # Reduce weights of models that are too similar to others
        self._calculate_correlation_penalties()

        for model_name, pred in predictions.items():
            base_weight = self.weights.get(model_name, 0.1)
            # Apply diversity penalty
            div_penalty = self._correlation_penalties.get(model_name, 1.0)
            base_weight *= div_penalty

            confidence = pred.get("confidence", 50) / 100
            direction = pred.get("direction", "Neutral")
            change_pct = pred.get(
                "change_pct", pred.get("predicted_change_pct", 0))

            # === NEW: Bayesian Meta-Learning with Uncertainty ===
            # Use regime-specific meta-learner if available, fallback to global
            reliability, uncertainty = self.get_regime_specific_reliability(
                model_name=model_name,
                confidence=pred.get("confidence", 50),
                regime=self._current_regime,
                volatility=self._current_volatility
            )

            # Combine base weight with meta-reliability and PENALIZE uncertainty
            # uncertainty is 0-~1. We want to reduce weight as uncertainty increases.
            # penalty_factor: 1.0 (no penalty) to 0.5 (high uncertainty)
            penalty_factor = 1.0 - (uncertainty * 0.5)
            weight = base_weight * (reliability * 2) * penalty_factor

            if direction == "Bullish":
                bullish_score += weight * confidence
            elif direction == "Bearish":
                bearish_score += weight * confidence

            total_weight += weight
            weighted_change += weight * change_pct

            model_results.append({
                "model": model_name,
                "direction": direction,
                "confidence": pred.get("confidence", 50),
                "weight": round(weight, 3),
                "meta_reliability": round(reliability, 2),
                "uncertainty": round(uncertainty, 2),
                "diversity_score": round(div_penalty, 2)
            })

            # Record in history for future correlation calculation
            if model_name not in self._model_history:
                self._model_history[model_name] = []

            signal_val = confidence if direction == "Bullish" else - \
                confidence if direction == "Bearish" else 0
            self._model_history[model_name].append(signal_val)
            if len(self._model_history[model_name]) > 50:
                self._model_history[model_name].pop(0)

        # Normalize scores
        if total_weight > 0:
            bullish_score /= total_weight
            bearish_score /= total_weight
            weighted_change /= total_weight

        # Determine final direction
        if bullish_score > bearish_score + 0.1:
            direction = "Bullish"
            confidence = bullish_score * 100
        elif bearish_score > bullish_score + 0.1:
            direction = "Bearish"
            confidence = bearish_score * 100
        else:
            direction = "Neutral"
            confidence = 50

        # === NEW: Sentiment-Based Confidence Multiplier ===
        # Apply sentiment analysis to boost/penalize confidence
        if symbol:
            confidence = self.apply_sentiment_multiplier(
                symbol, direction, confidence)

        # Check model agreement
        directions = [p.get("direction", "Neutral")
                      for p in predictions.values()]
        unique_directions = set(d for d in directions if d != "Neutral")
        models_agree = len(unique_directions) <= 1

        # Generate explanation
        if models_agree and len(unique_directions) == 1:
            explanation = f"All models agree: {list(unique_directions)[0]}"
        elif len(unique_directions) == 0:
            explanation = "Models are neutral/uncertain"
        else:
            bull_count = sum(1 for d in directions if d == "Bullish")
            bear_count = sum(1 for d in directions if d == "Bearish")
            explanation = f"Mixed signals: {bull_count} bullish, {bear_count} bearish"

        # === NEW: Signal Quality Rating ===
        # Based on confidence, model agreement, and number of models
        if confidence >= 75 and models_agree and len(predictions) >= 5:
            signal_quality = "VERY_HIGH"
            quality_score = 95
        elif confidence >= 65 and len(predictions) >= 4:
            signal_quality = "HIGH"
            quality_score = 75
        elif confidence >= 55:
            signal_quality = "MEDIUM"
            quality_score = 55
        else:
            signal_quality = "LOW"
            quality_score = 35

        # === NEW: Confidence Threshold Check ===
        # Only recommend action when confidence meets threshold
        CONFIDENCE_THRESHOLD = 60  # Minimum confidence to signal
        if confidence < CONFIDENCE_THRESHOLD:
            actionable_signal = False
            trade_recommendation = "NO_SIGNAL - Confidence below threshold"
        elif direction == "Neutral":
            actionable_signal = False
            trade_recommendation = "HOLD - No clear direction"
        else:
            actionable_signal = True
            strength = "STRONG" if confidence >= 70 else "MODERATE" if confidence >= 60 else "WEAK"
            trade_recommendation = f"{strength}_{direction.upper()}"

        # === NEW: Position Sizing Recommendation ===
        # Scale position based on confidence and signal quality
        if not actionable_signal:
            position_size_pct = 0
        elif signal_quality == "VERY_HIGH":
            position_size_pct = min(100, int(confidence * 1.2))
        elif signal_quality == "HIGH":
            position_size_pct = min(80, int(confidence * 1.0))
        elif signal_quality == "MEDIUM":
            position_size_pct = min(50, int(confidence * 0.7))
        else:  # LOW
            position_size_pct = min(25, int(confidence * 0.4))

        # === NEW: Uncertainty Quantification ===
        # Calculate prediction intervals and uncertainty decomposition
        uncertainty_data = None
        if stock_data is not None:
            try:
                # Convert predictions dict to list for uncertainty calculation
                pred_list = list(predictions.values())
                uncertainty_result = get_ensemble_uncertainty(pred_list, stock_data)

                if 'error' not in uncertainty_result:
                    uncertainty_data = uncertainty_result
                    logger.info(f"[{symbol}] Uncertainty: epistemic={uncertainty_result['uncertainty_decomposition']['epistemic_pct']:.1f}%, aleatoric={uncertainty_result['uncertainty_decomposition']['aleatoric_pct']:.1f}%")
            except Exception as e:
                logger.warning(f"Failed to calculate uncertainty: {e}")

        return {
            "direction": direction,
            "confidence": round(confidence, 1),
            "predicted_change_pct": round(weighted_change, 2),
            "models_agree": models_agree,
            "bullish_score": round(bullish_score * 100, 1),
            "bearish_score": round(bearish_score * 100, 1),
            "explanation": explanation,
            "model_results": model_results,
            "weights_used": {k: round(v, 3) for k, v in self.weights.items()},
            "timestamp": datetime.now().isoformat(),
            # NEW FIELDS
            "signal_quality": signal_quality,
            "quality_score": quality_score,
            "actionable_signal": actionable_signal,
            "trade_recommendation": trade_recommendation,
            "position_size_pct": position_size_pct,
            "confidence_threshold": CONFIDENCE_THRESHOLD,
            "meets_threshold": confidence >= CONFIDENCE_THRESHOLD,
            # UNCERTAINTY QUANTIFICATION
            "uncertainty": uncertainty_data
        }

    def get_prediction_with_confidence(
        self,
        symbol: str,
        current_price: float,
        lstm_pred: Optional[Dict] = None,
        gru_pred: Optional[Dict] = None,
        xgboost_pred: Optional[Dict] = None,
        rf_pred: Optional[Dict] = None,
        svm_pred: Optional[Dict] = None,
        momentum_pred: Optional[Dict] = None,
        technical_signals: Optional[Dict] = None,
        fundamental_score: Optional[Dict] = None,
        stock_data: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        Generate unified prediction from all available sources.

        Args:
            symbol: Stock ticker symbol
            current_price: Current stock price
            stock_data: Price DataFrame for market regime detection
            *_pred: Individual model predictions

        Returns comprehensive prediction with confidence and explanation.
        """
        # === NEW: LLM News Reasoning ===
        systemic_impact = {"impact_score": 50,
                           "reasoning": "No news available"}
        if technical_signals and "news" in technical_signals:
            reasoner = get_news_reasoner()
            try:
                # analyze_systemic_impact is async — run it safely from sync context
                import asyncio
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as pool:
                        systemic_impact = pool.submit(
                            asyncio.run,
                            reasoner.analyze_systemic_impact(
                                symbol, technical_signals.get("news", []))
                        ).result(timeout=30)
                else:
                    systemic_impact = loop.run_until_complete(
                        reasoner.analyze_systemic_impact(
                            symbol, technical_signals.get("news", [])))
            except Exception as e:
                logger.warning(f"Systemic impact analysis failed: {e}")
                systemic_impact = {"impact_score": 50,
                                   "reasoning": f"Analysis unavailable: {e}"}

        predictions = {}

        # XGBoost (NEW - highest priority based on research)
        if xgboost_pred and "error" not in xgboost_pred:
            predictions["xgboost"] = {
                "direction": xgboost_pred.get("direction", "Neutral"),
                "confidence": xgboost_pred.get("confidence", 50),
                "change_pct": xgboost_pred.get("predicted_change_pct", 0)
            }

        # LSTM
        if lstm_pred and "error" not in lstm_pred:
            predictions["lstm"] = {
                "direction": lstm_pred.get("direction", "Neutral"),
                "confidence": lstm_pred.get("confidence", 50),
                "change_pct": lstm_pred.get("predicted_change_pct", 0)
            }

        # GRU (NEW - alternative to LSTM)
        if gru_pred and "error" not in gru_pred:
            predictions["gru"] = {
                "direction": gru_pred.get("direction", "Neutral"),
                "confidence": gru_pred.get("confidence", 50),
                "change_pct": gru_pred.get("expected_change_pct", 0)
            }

        # Random Forest
        if rf_pred and "error" not in rf_pred:
            predictions["random_forest"] = {
                "direction": "Bullish" if rf_pred.get("prediction", 0) == 1 else "Bearish",
                "confidence": rf_pred.get("confidence", 50),
                "change_pct": rf_pred.get("predicted_change", 0)
            }

        # SVM
        if svm_pred and "error" not in svm_pred:
            predictions["svm"] = {
                "direction": "Bullish" if svm_pred.get("prediction", 0) == 1 else "Bearish",
                "confidence": svm_pred.get("confidence", 50),
                "change_pct": svm_pred.get("predicted_change", 0)
            }

        # Momentum
        if momentum_pred and "error" not in momentum_pred:
            predictions["momentum"] = {
                "direction": momentum_pred.get("prediction", "Neutral"),
                "confidence": momentum_pred.get("confidence", 50),
                "change_pct": momentum_pred.get("expected_move", 0)
            }

        # Technical signals
        if technical_signals:
            signal = technical_signals.get("signal", "HOLD")
            if "BUY" in signal.upper():
                direction = "Bullish"
            elif "SELL" in signal.upper():
                direction = "Bearish"
            else:
                direction = "Neutral"

            predictions["technical"] = {
                "direction": direction,
                "confidence": technical_signals.get("signal_strength", 50),
                "change_pct": 0
            }

        # Fundamental score
        if fundamental_score:
            # Simple heuristic based on valuation
            pe_ratio = fundamental_score.get("pe_ratio", 20)
            peg_ratio = fundamental_score.get("peg_ratio", 1.5)

            if pe_ratio and pe_ratio < 15 and (not peg_ratio or peg_ratio < 1):
                direction = "Bullish"
                confidence = 60
            elif pe_ratio and pe_ratio > 30 and peg_ratio and peg_ratio > 2:
                direction = "Bearish"
                confidence = 60
            else:
                direction = "Neutral"
                confidence = 50

            predictions["fundamental"] = {
                "direction": direction,
                "confidence": confidence,
                "change_pct": 0
            }

        # Update weights based on historical accuracy (with market regime detection)
        self.update_weights_from_accuracy(symbol, stock_data=stock_data)

        # Calculate ensemble
        ensemble = self.calculate_ensemble_score(predictions)

        # Add weight stability info to the result
        stability_info = self.get_weight_stability_info()
        ensemble["market_regime"] = stability_info["current_regime"]
        ensemble["weight_stability"] = stability_info
        ensemble["systemic_impact"] = systemic_impact

        # Apply impact score as a multiplier to confidence if significant
        # impact_score 0-100, 50 is neutral.
        impact_multiplier = 1.0
        if systemic_impact.get("impact_score"):
            score = systemic_impact["impact_score"]
            if score > 70 or score < 30:
                # Strong impact detected - shift confidence
                impact_multiplier = 1.0 + (abs(score - 50) / 50) * 0.2
                ensemble["confidence"] = min(
                    98, ensemble["confidence"] * impact_multiplier)
                ensemble["explanation"] += f" | Systemic Impact: {systemic_impact.get('reasoning')}"

        # Record prediction for future validation
        self.store.record_prediction(
            symbol=symbol,
            model_name="ensemble",
            predicted_direction=ensemble["direction"],
            predicted_change_pct=ensemble["predicted_change_pct"],
            confidence=ensemble["confidence"],
            current_price=current_price,
            regime=self._current_regime,
            volatility=self._current_volatility
        )

        # Validate old predictions
        self.store.validate_predictions(symbol, current_price)

        return ensemble

    def _calculate_correlation_penalties(self):
        """
        Detect highly correlated models and penalize redundancy.
        Ensures ensemble diversity.
        """
        models = list(self._model_history.keys())
        self._correlation_penalties = {
            m: 1.0 for m in self.DEFAULT_WEIGHTS.keys()}

        if len(models) < 2:
            return

        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                m1, m2 = models[i], models[j]
                h1, h2 = self._model_history[m1], self._model_history[m2]

                if len(h1) >= 10 and len(h2) >= 10:
                    try:
                        # Simple correlation
                        correlation = np.corrcoef(h1[-10:], h2[-10:])[0, 1]
                        if not np.isnan(correlation) and correlation > 0.85:
                            # Too much overlap. Penalize one with lower base weight
                            w1 = self.weights.get(m1, 0.1)
                            w2 = self.weights.get(m2, 0.1)

                            if w1 > w2:
                                self._correlation_penalties[m2] *= 0.8
                            else:
                                self._correlation_penalties[m1] *= 0.8
                    except:
                        pass


def get_ensemble_prediction(
    symbol: str,
    stock_data: pd.DataFrame,
    ml_prediction: Optional[Dict] = None,
    technical_signals: Optional[Dict] = None,
    fundamental_data: Optional[Dict] = None
) -> Dict[str, Any]:
    """
    Main function to get ensemble prediction for a stock.

    Args:
        symbol: Stock ticker
        stock_data: Price DataFrame
        ml_prediction: Dictionary containing ML model predictions
        technical_signals: Output from signals endpoint
        fundamental_data: Fundamental metrics

    Returns:
        Unified ensemble prediction
    """
    scorer = get_ensemble_scorer()  # Use singleton instead of creating new instance

    current_price = stock_data['Close'].iloc[-1] if len(stock_data) > 0 else 0

    # Extract individual predictions from ml_prediction
    lstm_pred = None
    rf_pred = None
    svm_pred = None
    momentum_pred = None
    xgboost_pred = None
    gru_pred = None

    if ml_prediction:
        # LSTM
        lstm_pred = ml_prediction.get("lstm_prediction")

        # XGBoost
        if ml_prediction.get("xgboost_prediction") is not None:
            xgboost_pred = ml_prediction.get("xgboost_prediction")

        # GRU
        if ml_prediction.get("gru_prediction") is not None:
            gru_pred = ml_prediction.get("gru_prediction")

        # Random Forest
        if ml_prediction.get("rf_prediction") is not None:
            rf_pred = {
                "prediction": 1 if ml_prediction.get("rf_prediction") == "Bullish" else 0,
                "confidence": ml_prediction.get("rf_confidence", 50)
            }

        # SVM
        if ml_prediction.get("svm_prediction") is not None:
            svm_pred = {
                "prediction": 1 if ml_prediction.get("svm_prediction") == "Bullish" else 0,
                "confidence": ml_prediction.get("svm_confidence", 50)
            }

        # Momentum
        if ml_prediction.get("momentum_prediction") is not None:
            momentum_pred = {
                "prediction": ml_prediction.get("momentum_prediction"),
                "confidence": ml_prediction.get("momentum_confidence", 50),
                "expected_move": ml_prediction.get("momentum_expected_change", 0)
            }

    return scorer.get_prediction_with_confidence(
        symbol=symbol,
        current_price=current_price,
        lstm_pred=lstm_pred,
        gru_pred=gru_pred,
        xgboost_pred=xgboost_pred,
        rf_pred=rf_pred,
        svm_pred=svm_pred,
        momentum_pred=momentum_pred,
        technical_signals=technical_signals,
        fundamental_score=fundamental_data,
        stock_data=stock_data
    )


# Global singleton for ensemble scorer
_ensemble_scorer = None


def get_ensemble_scorer() -> EnsembleScorer:
    """
    Get global ensemble scorer instance (singleton pattern).

    Returns:
        Singleton EnsembleScorer instance
    """
    global _ensemble_scorer
    if _ensemble_scorer is None:
        _ensemble_scorer = EnsembleScorer()
    return _ensemble_scorer
