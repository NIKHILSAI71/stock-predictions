"""
Ensemble Prediction Scorer Module

Combines multiple ML models and technical indicators into a unified prediction
with dynamic weighting based on historical accuracy.

Data Storage Structure:
    data/
    └── predictions/
        └── ensemble/
            ├── history.json      # All ensemble predictions (last 1000)
            └── by_model/         # Performance by model type
                ├── lstm.json
                ├── xgboost.json
                └── ...
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import json
import os


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
                print(f"[PredictionStore] Error loading {self.history_file}: {e}")
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
            print(f"[PredictionStore] Error saving {self.history_file}: {e}")
    
    def record_prediction(
        self,
        symbol: str,
        model_name: str,
        predicted_direction: str,
        predicted_change_pct: float,
        confidence: float,
        current_price: float
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
                    actual_change = (current_price - pred["price_at_prediction"]) / pred["price_at_prediction"] * 100
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
        calibration = accuracy / (avg_confidence / 100) if avg_confidence > 0 else 1.0
        
        return {
            "accuracy": round(accuracy, 3),
            "sample_size": total,
            "correct": correct,
            "confidence_calibration": round(calibration, 2)
        }


class EnsembleScorer:
    """
    Combines multiple prediction sources into a unified score.
    
    Features stable weight management to prevent dramatic fluctuations:
    - Bayesian smoothing for gradual weight transitions
    - Minimum weight floors to prevent model abandonment
    - Maximum weight adjustment limits per update cycle
    - Market regime-aware weight profiles
    """
    
    # Updated weights including XGBoost, GRU, CNN-LSTM, and Attention based on research (2024)
    # XGBoost often outperforms RF, CNN-LSTM combines spatial+temporal learning
    # Attention-based models are state-of-the-art for sequence modeling
    DEFAULT_WEIGHTS = {
        "xgboost": 0.15,        # Strong tree-based learner
        "cnn_lstm": 0.14,       # NEW: Hybrid CNN-LSTM model
        "attention": 0.12,      # NEW: Transformer attention
        "lstm": 0.11,
        "gru": 0.10,            # Alternative to LSTM
        "random_forest": 0.10,
        "svm": 0.08,
        "momentum": 0.08,
        "technical": 0.07,
        "fundamental": 0.05
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
    SMOOTHING_ALPHA = 0.3             # EMA smoothing factor (lower = more stable)
    MIN_SAMPLES_FOR_ADJUSTMENT = 10   # Minimum predictions before adjusting weights
    BAYESIAN_PRIOR_WEIGHT = 0.5       # Prior accuracy (50%) for Bayesian smoothing
    BAYESIAN_PRIOR_SAMPLES = 10       # Equivalent sample count for prior
    
    def __init__(self, prediction_store: Optional[PredictionStore] = None):
        self.weights = self.DEFAULT_WEIGHTS.copy()
        self.store = prediction_store or PredictionStore()
        self._previous_weights = self.DEFAULT_WEIGHTS.copy()  # For EMA smoothing
        self._current_regime = "sideways"  # Default regime
    
    def detect_market_regime(self, stock_data: Optional[pd.DataFrame] = None) -> str:
        """
        Detect current market regime based on price action.
        
        Returns: "bullish", "bearish", or "sideways"
        """
        if stock_data is None or len(stock_data) < 50:
            return "sideways"
        
        try:
            # Calculate 20-day and 50-day moving averages
            close = stock_data['Close'].values
            sma_20 = np.mean(close[-20:])
            sma_50 = np.mean(close[-50:])
            current_price = close[-1]
            
            # Calculate recent volatility
            returns = np.diff(close[-20:]) / close[-21:-1]
            volatility = np.std(returns) * np.sqrt(252)  # Annualized
            
            # Calculate trend strength
            price_change_20d = (current_price - close[-20]) / close[-20] * 100
            
            # Determine regime
            if current_price > sma_20 > sma_50 and price_change_20d > 3:
                return "bullish"
            elif current_price < sma_20 < sma_50 and price_change_20d < -3:
                return "bearish"
            else:
                return "sideways"
        except Exception:
            return "sideways"
    
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
            old_weight = self._previous_weights.get(model, self.DEFAULT_WEIGHTS[model])
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
            min_allowed = max(self.MIN_WEIGHT_FLOOR, default * self.MIN_WEIGHT_MULTIPLIER)
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
        self._current_regime = self.detect_market_regime(stock_data)
        
        accuracies = {}
        sample_sizes = {}
        
        for model in self.weights.keys():
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
        
        # Calculate raw accuracy-based weights
        total_accuracy = sum(accuracies.values())
        if total_accuracy > 0:
            raw_weights = {}
            for model in self.weights:
                # Blend accuracy-based weight with default weight
                # Using 70% default, 30% accuracy to prioritize stability
                accuracy_weight = accuracies.get(model, 0.5) / total_accuracy
                default_weight = self.DEFAULT_WEIGHTS[model]
                raw_weights[model] = 0.7 * default_weight + 0.3 * accuracy_weight
        else:
            raw_weights = self.DEFAULT_WEIGHTS.copy()
        
        # Apply regime-specific adjustments
        regime_adjusted = self._apply_regime_adjustments(raw_weights, self._current_regime)
        
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
    
    def calculate_ensemble_score(
        self,
        predictions: Dict[str, Dict[str, Any]]
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
        
        for model_name, pred in predictions.items():
            weight = self.weights.get(model_name, 0.1)
            confidence = pred.get("confidence", 50) / 100
            direction = pred.get("direction", "Neutral")
            change_pct = pred.get("change_pct", pred.get("predicted_change_pct", 0))
            
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
                "weight": round(weight, 3)
            })
        
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
        
        # Check model agreement
        directions = [p.get("direction", "Neutral") for p in predictions.values()]
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
        
        return {
            "direction": direction,
            "confidence": round(confidence, 1),
            "predicted_change_pct": round(weighted_change, 2),
            "models_agree": models_agree,
            "bullish_score": round(bullish_score * 100, 1),
            "bearish_score": round(bearish_score * 100, 1),
            "explanation": explanation,
            "model_breakdown": model_results,
            "weights_used": {k: round(v, 3) for k, v in self.weights.items()},
            "timestamp": datetime.now().isoformat(),
            # NEW FIELDS
            "signal_quality": signal_quality,
            "quality_score": quality_score,
            "actionable_signal": actionable_signal,
            "trade_recommendation": trade_recommendation,
            "position_size_pct": position_size_pct,
            "confidence_threshold": CONFIDENCE_THRESHOLD,
            "meets_threshold": confidence >= CONFIDENCE_THRESHOLD
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
        
        # Record prediction for future validation
        self.store.record_prediction(
            symbol=symbol,
            model_name="ensemble",
            predicted_direction=ensemble["direction"],
            predicted_change_pct=ensemble["predicted_change_pct"],
            confidence=ensemble["confidence"],
            current_price=current_price
        )
        
        # Validate old predictions
        self.store.validate_predictions(symbol, current_price)
        
        return ensemble


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
        ml_prediction: Output from ml_aggregator
        technical_signals: Output from signals endpoint
        fundamental_data: Fundamental metrics
    
    Returns:
        Unified ensemble prediction
    """
    scorer = EnsembleScorer()
    
    current_price = stock_data['Close'].iloc[-1] if len(stock_data) > 0 else 0
    
    # Extract individual predictions from ml_prediction
    lstm_pred = None
    rf_pred = None
    svm_pred = None
    momentum_pred = None
    
    if ml_prediction:
        # Check if LSTM is included
        lstm_pred = ml_prediction.get("lstm_prediction")
        
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
        rf_pred=rf_pred,
        svm_pred=svm_pred,
        momentum_pred=momentum_pred,
        technical_signals=technical_signals,
        fundamental_score=fundamental_data,
        stock_data=stock_data
    )
