"""
Robust Confidence Calculator

Calculates stable, meaningful confidence scores for ensemble predictions.

Fixes:
- Erratic confidence scores
- Overconfidence issues
- Poor calibration between confidence and actual accuracy

Multi-factor confidence based on:
1. Model Agreement (30%) - How much do models agree?
2. Historical Accuracy (25%) - How accurate have models been?
3. Market Conditions (20%) - Is the market predictable?
4. Data Quality (15%) - Is the data reliable?
5. Temporal Consistency (10%) - Are predictions stable?
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging

logger = logging.getLogger(__name__)


class RobustConfidenceCalculator:
    """
    Calculates stable, meaningful confidence scores.
    
    Features:
    - Multi-factor confidence calculation
    - Hard bounds to prevent overconfidence (50-95%)
    - Calibration using sigmoid compression
    - Validation for adaptive learning
    
    Usage:
        calculator = RobustConfidenceCalculator()
        
        confidence, breakdown = calculator.calculate_ensemble_confidence(
            model_predictions={'xgboost': 105.5, 'lstm': 104.2},
            model_weights={'xgboost': 0.15, 'lstm': 0.12},
            model_historical_accuracy={'xgboost': 0.68, 'lstm': 0.72},
            market_data={'volatility': 0.025, 'volume_ratio': 1.1}
        )
    """
    
    # Confidence bounds - NEVER claim 100% or 0% certainty
    MIN_CONFIDENCE = 50.0
    MAX_CONFIDENCE = 95.0
    
    # Factor weights for ensemble confidence
    FACTOR_WEIGHTS = {
        'agreement': 0.30,
        'accuracy': 0.25,
        'market': 0.20,
        'data_quality': 0.15,
        'consistency': 0.10
    }
    
    def __init__(
        self, 
        min_confidence: float = 50.0, 
        max_confidence: float = 95.0
    ):
        """
        Initialize the confidence calculator.
        
        Args:
            min_confidence: Minimum confidence score (default 50%)
            max_confidence: Maximum confidence score (default 95%)
        """
        self.min_confidence = min_confidence
        self.max_confidence = max_confidence
        
        # Track historical calibration
        self._calibration_history: List[Dict[str, float]] = []
        
    def calculate_ensemble_confidence(
        self,
        model_predictions: Dict[str, float],
        model_weights: Dict[str, float],
        model_historical_accuracy: Dict[str, float],
        market_data: Dict[str, Any]
    ) -> Tuple[float, Dict[str, float]]:
        """
        Multi-factor confidence calculation.
        
        Args:
            model_predictions: Dict of model_name -> predicted_value
            model_weights: Dict of model_name -> weight
            model_historical_accuracy: Dict of model_name -> accuracy (0-1)
            market_data: Dict containing volatility, volume_ratio, regime, etc.
        
        Returns:
            Tuple of (confidence_score, breakdown_dict)
        """
        if not model_predictions:
            return self.min_confidence, {'error': 'No predictions provided'}
        
        # Factor 1: Model Agreement (30% weight)
        agreement_score = self._calculate_agreement(model_predictions, model_weights)
        
        # Factor 2: Historical Accuracy (25% weight)
        accuracy_score = self._calculate_accuracy_confidence(
            model_predictions, 
            model_weights, 
            model_historical_accuracy
        )
        
        # Factor 3: Market Conditions (20% weight)
        market_score = self._calculate_market_confidence(market_data)
        
        # Factor 4: Data Quality (15% weight)
        data_quality_score = self._calculate_data_quality(market_data)
        
        # Factor 5: Prediction Consistency (10% weight)
        consistency_score = self._calculate_temporal_consistency(
            model_predictions, 
            market_data.get('recent_predictions', [])
        )
        
        # Weighted combination
        raw_confidence = (
            agreement_score * self.FACTOR_WEIGHTS['agreement'] +
            accuracy_score * self.FACTOR_WEIGHTS['accuracy'] +
            market_score * self.FACTOR_WEIGHTS['market'] +
            data_quality_score * self.FACTOR_WEIGHTS['data_quality'] +
            consistency_score * self.FACTOR_WEIGHTS['consistency']
        )
        
        # Apply bounds and calibration
        final_confidence = self._calibrate_confidence(raw_confidence)
        
        breakdown = {
            'agreement': round(agreement_score, 1),
            'historical_accuracy': round(accuracy_score, 1),
            'market_conditions': round(market_score, 1),
            'data_quality': round(data_quality_score, 1),
            'temporal_consistency': round(consistency_score, 1),
            'raw_score': round(raw_confidence, 1),
            'final_score': round(final_confidence, 1)
        }
        
        return final_confidence, breakdown
    
    def _calculate_agreement(
        self, 
        predictions: Dict[str, float], 
        weights: Dict[str, float]
    ) -> float:
        """
        How much do models agree?
        High agreement = High confidence.
        
        Uses weighted standard deviation of predictions.
        """
        pred_values = np.array(list(predictions.values()))
        
        if len(pred_values) < 2:
            return 50.0  # Not enough models to measure agreement
        
        # Get weights aligned with predictions
        weight_values = np.array([
            weights.get(model, 1.0) for model in predictions.keys()
        ])
        weight_values = weight_values / weight_values.sum()  # Normalize
        
        # Weighted mean
        weighted_mean = np.average(pred_values, weights=weight_values)
        
        # Weighted variance
        weighted_variance = np.average(
            (pred_values - weighted_mean) ** 2, 
            weights=weight_values
        )
        weighted_std = np.sqrt(weighted_variance)
        
        # Normalize by mean to get coefficient of variation
        if weighted_mean != 0:
            cv = weighted_std / abs(weighted_mean)
        else:
            cv = weighted_std
        
        # Convert CV to agreement score (0-100)
        # Lower CV = higher agreement
        # CV of 0.15 (15% disagreement) is considered "normal"
        max_expected_cv = 0.15
        agreement = max(0, 100 * (1 - cv / max_expected_cv))
        
        return np.clip(agreement, 0, 100)
    
    def _calculate_accuracy_confidence(
        self,
        predictions: Dict[str, float],
        weights: Dict[str, float],
        historical_accuracy: Dict[str, float]
    ) -> float:
        """
        Weight predictions by how accurate models have been historically.
        """
        if not historical_accuracy:
            return 60.0  # Neutral if no history
        
        # Collect accuracies and weights for models with history
        accuracies = []
        weights_list = []
        
        for model in predictions.keys():
            if model in historical_accuracy:
                acc = historical_accuracy[model]
                # Ensure accuracy is in range [0, 1]
                if acc > 1:
                    acc = acc / 100  # Assume it was given as percentage
                accuracies.append(acc)
                weights_list.append(weights.get(model, 1.0))
        
        if not accuracies:
            return 60.0  # Neutral
        
        # Weighted average of historical accuracies
        weights_arr = np.array(weights_list)
        weights_arr = weights_arr / weights_arr.sum()  # Normalize
        weighted_accuracy = np.average(accuracies, weights=weights_arr)
        
        # Scale to 0-100
        return weighted_accuracy * 100
    
    def _calculate_market_confidence(self, market_data: Dict[str, Any]) -> float:
        """
        Market conditions affect prediction reliability.
        Volatile/unusual markets = Lower confidence.
        """
        confidence_adjustments = []
        
        # 1. Volatility check (CRITICAL factor)
        volatility = market_data.get('volatility', 0.02)
        if volatility < 0.01:  # Very low volatility - easy to predict
            confidence_adjustments.append(90)
        elif volatility < 0.03:  # Normal volatility
            confidence_adjustments.append(75)
        elif volatility < 0.05:  # High volatility
            confidence_adjustments.append(55)
        else:  # Extreme volatility - hard to predict
            confidence_adjustments.append(40)
        
        # 2. Volume check (unusual volume = unusual behavior)
        volume_ratio = market_data.get('volume_ratio', 1.0)
        if 0.7 <= volume_ratio <= 1.3:  # Normal volume
            confidence_adjustments.append(80)
        elif volume_ratio < 0.5 or volume_ratio > 2.0:  # Very abnormal
            confidence_adjustments.append(50)
        else:  # Somewhat abnormal
            confidence_adjustments.append(65)
        
        # 3. Market regime
        regime = market_data.get('regime', 'normal')
        regime_scores = {
            'bull': 85,       # Trending up - momentum works well
            'bullish': 85,
            'normal': 75,     # Range-bound - harder to predict
            'sideways': 70,
            'bear': 60,       # Trending down - more volatile
            'bearish': 60,
            'crisis': 40      # High uncertainty
        }
        confidence_adjustments.append(regime_scores.get(regime, 70))
        
        # 4. Trend strength (strong trend = easier to predict)
        trend_strength = market_data.get('trend_strength', 0.5)
        trend_strength = np.clip(trend_strength, 0, 1)
        confidence_adjustments.append(50 + trend_strength * 50)
        
        return np.mean(confidence_adjustments)
    
    def _calculate_data_quality(self, market_data: Dict[str, Any]) -> float:
        """
        Poor data quality = Lower confidence.
        """
        quality_score = 100.0
        
        # Check for missing data
        missing_ratio = market_data.get('missing_data_ratio', 0)
        quality_score -= missing_ratio * 50  # Heavy penalty for missing data
        
        # Check data recency
        days_since_update = market_data.get('days_since_update', 0)
        if days_since_update > 2:
            quality_score -= (days_since_update - 2) * 10
        
        # Check sample size (need enough history for ML)
        data_points = market_data.get('data_points', 100)
        if data_points < 60:  # Less than 60 days
            quality_score -= (60 - data_points) * 0.5
        
        return np.clip(quality_score, 0, 100)
    
    def _calculate_temporal_consistency(
        self,
        current_predictions: Dict[str, float],
        recent_predictions: List[float]
    ) -> float:
        """
        Are predictions stable over time?
        Wild swings in predictions = Lower confidence.
        """
        if not recent_predictions or len(recent_predictions) < 3:
            return 70.0  # Neutral when no history
        
        # Current weighted average
        current_avg = np.mean(list(current_predictions.values()))
        
        # Check if current prediction is within reasonable range of recent
        recent_mean = np.mean(recent_predictions)
        recent_std = np.std(recent_predictions)
        
        # Z-score of current vs recent predictions
        if recent_std == 0 or np.isnan(recent_std):
            z_score = 0
        else:
            z_score = abs((current_avg - recent_mean) / recent_std)
        
        # Convert z-score to confidence
        # z=0 (perfect consistency) = 100
        # z=3 (3 std away) = 30
        consistency = max(30, 100 - z_score * 23)
        
        return consistency
    
    def _calibrate_confidence(self, raw_confidence: float) -> float:
        """
        Apply final calibration to prevent overconfidence.
        
        Uses tanh transformation to compress extremes.
        Maps raw scores to realistic confidence levels.
        """
        # Center around 70% (typical average confidence)
        centered = (raw_confidence - 70) / 15
        
        # Apply tanh transformation to compress extremes
        # This ensures we never get too close to 0% or 100%
        compressed = 70 + 15 * np.tanh(centered)
        
        # Enforce hard bounds
        return np.clip(compressed, self.min_confidence, self.max_confidence)
    
    def validate_confidence(
        self, 
        confidence: float, 
        actual_outcome: bool
    ) -> float:
        """
        Check if confidence was well-calibrated.
        Use this in adaptive learning to improve calibration.
        
        Args:
            confidence: The predicted confidence (50-95)
            actual_outcome: Whether the prediction was correct
            
        Returns:
            Calibration error score (-30 to +20)
            Positive = well calibrated
            Negative = poorly calibrated
        """
        if actual_outcome:
            # Correct prediction
            if confidence > 75:
                score = 20  # Reward high confidence when correct
            elif confidence < 60:
                score = -10  # Penalize low confidence when correct
            else:
                score = 10  # Neutral reward
        else:
            # Wrong prediction
            if confidence > 75:
                score = -30  # Heavy penalty for overconfidence
            elif confidence < 60:
                score = -5  # Small penalty for uncertainty
            else:
                score = -15  # Medium penalty
        
        # Track for calibration analysis
        self._calibration_history.append({
            'confidence': confidence,
            'correct': actual_outcome,
            'score': score
        })
        
        return score
    
    def get_calibration_stats(self) -> Dict[str, Any]:
        """
        Get calibration statistics from validation history.
        """
        if not self._calibration_history:
            return {'error': 'No calibration history'}
        
        confidences = [h['confidence'] for h in self._calibration_history]
        correct = [h['correct'] for h in self._calibration_history]
        scores = [h['score'] for h in self._calibration_history]
        
        # Calculate calibration by confidence bucket
        buckets = {'50-60': [], '60-70': [], '70-80': [], '80-95': []}
        for conf, corr in zip(confidences, correct):
            if conf < 60:
                buckets['50-60'].append(corr)
            elif conf < 70:
                buckets['60-70'].append(corr)
            elif conf < 80:
                buckets['70-80'].append(corr)
            else:
                buckets['80-95'].append(corr)
        
        calibration_by_bucket = {}
        for bucket, outcomes in buckets.items():
            if outcomes:
                calibration_by_bucket[bucket] = {
                    'count': len(outcomes),
                    'accuracy': sum(outcomes) / len(outcomes)
                }
        
        return {
            'total_predictions': len(self._calibration_history),
            'average_confidence': np.mean(confidences),
            'overall_accuracy': sum(correct) / len(correct),
            'average_calibration_score': np.mean(scores),
            'calibration_by_bucket': calibration_by_bucket
        }


# Convenience function for quick confidence calculation
def calculate_confidence(
    model_predictions: Dict[str, float],
    model_weights: Dict[str, float],
    model_historical_accuracy: Dict[str, float] = None,
    market_data: Dict[str, Any] = None
) -> float:
    """
    Quick confidence calculation with sensible defaults.
    
    Usage:
        confidence = calculate_confidence(
            {'xgboost': 105, 'lstm': 104},
            {'xgboost': 0.15, 'lstm': 0.12}
        )
    """
    calculator = RobustConfidenceCalculator()
    
    if model_historical_accuracy is None:
        model_historical_accuracy = {}
    
    if market_data is None:
        market_data = {}
    
    confidence, _ = calculator.calculate_ensemble_confidence(
        model_predictions,
        model_weights,
        model_historical_accuracy,
        market_data
    )
    
    return confidence
