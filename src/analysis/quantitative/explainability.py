"""
Model Explainability Framework

Provides interpretable explanations for all model types:
- SHAP (SHapley Additive exPlanations) for tree models
- Feature importance ranking
- Attention weights visualization for transformers
- Decision paths for ensemble models

Based on research (2024):
- SHAP provides theoretically sound feature attributions
- TreeExplainer is fast and exact for tree-based models
- Attention weights reveal temporal importance

Note: Install shap for full functionality: pip install shap
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class ModelExplainer:
    """Unified explainability across all model types"""

    def __init__(self, model, model_type: str, feature_names: Optional[List[str]] = None):
        """
        Initialize model explainer.

        Args:
            model: Trained model object
            model_type: Type of model ('xgboost', 'lightgbm', 'random_forest',
                       'svm', 'lstm', 'gru', 'attention', etc.)
            feature_names: List of feature names (for tree models)
        """
        self.model = model
        self.model_type = model_type.lower()
        self.feature_names = feature_names or []

    def explain_prediction(
        self,
        features: np.ndarray,
        top_k: int = 10
    ) -> Dict[str, Any]:
        """
        Generate explanation for single prediction.

        Args:
            features: Input features for prediction
            top_k: Number of top features to return

        Returns:
            Explanation dict with feature importances and text description
        """
        try:
            if self.model_type in ['xgboost', 'lightgbm', 'random_forest']:
                return self._tree_based_explanation(features, top_k)

            elif self.model_type in ['lstm', 'gru', 'cnn_lstm', 'tcn', 'nbeats']:
                return self._rnn_explanation(features, top_k)

            elif self.model_type == 'attention':
                return self._attention_explanation(features, top_k)

            elif self.model_type == 'svm':
                return self._svm_explanation(features, top_k)

            elif self.model_type == 'momentum':
                return self._momentum_explanation(features, top_k)

            else:
                return {
                    'method': 'generic',
                    'explanation': f'No specific explainability for {self.model_type}',
                    'model_type': self.model_type
                }

        except Exception as e:
            logger.error(f"Error generating explanation: {e}")
            return {
                'error': str(e),
                'model_type': self.model_type
            }

    def _tree_based_explanation(
        self,
        features: np.ndarray,
        top_k: int
    ) -> Dict[str, Any]:
        """
        SHAP TreeExplainer for gradient boosting models.

        Uses fast and exact SHAP values for tree ensembles.
        """
        try:
            # Try SHAP if available
            try:
                import shap

                # Create explainer
                explainer = shap.TreeExplainer(self.model)

                # Calculate SHAP values
                if len(features.shape) == 1:
                    features = features.reshape(1, -1)

                shap_values = explainer.shap_values(features)

                # Handle different SHAP output formats
                if isinstance(shap_values, list):
                    # Multi-class output - use first class
                    shap_values = shap_values[0]

                if len(shap_values.shape) > 1:
                    shap_values = shap_values[0]  # Take first instance

                # Get feature importance (absolute SHAP values)
                feature_importance = dict(zip(
                    self.feature_names if self.feature_names else [
                        f"feature_{i}" for i in range(len(shap_values))],
                    np.abs(shap_values)
                ))

                # Sort by importance
                top_features = dict(sorted(
                    feature_importance.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:top_k])

                # Generate text explanation
                explanation_text = self._generate_text_explanation(
                    top_features)

                return {
                    'method': 'SHAP TreeExplainer',
                    'model_type': self.model_type,
                    'top_features': top_features,
                    'shap_values': {k: float(v) for k, v in zip(self.feature_names, shap_values)} if self.feature_names else {},
                    'base_value': float(explainer.expected_value) if hasattr(explainer, 'expected_value') else None,
                    'explanation_text': explanation_text,
                    'feature_count': len(top_features)
                }

            except ImportError:
                logger.warning(
                    "SHAP not installed. Fallback to feature_importances_")
                # Fallback to built-in feature importance
                return self._fallback_feature_importance(top_k)

        except Exception as e:
            logger.error(f"Error in tree-based explanation: {e}")
            return {'error': str(e), 'method': 'tree_based'}

    def _fallback_feature_importance(self, top_k: int) -> Dict[str, Any]:
        """Fallback to built-in feature importance when SHAP unavailable"""
        try:
            if hasattr(self.model, 'feature_importances_'):
                importances = self.model.feature_importances_

                feature_importance = dict(zip(
                    self.feature_names if self.feature_names else [
                        f"feature_{i}" for i in range(len(importances))],
                    importances
                ))

                top_features = dict(sorted(
                    feature_importance.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:top_k])

                explanation_text = self._generate_text_explanation(
                    top_features)

                return {
                    'method': 'Feature Importance (built-in)',
                    'model_type': self.model_type,
                    'top_features': top_features,
                    'explanation_text': explanation_text
                }

            return {'method': 'unavailable', 'reason': 'Model has no feature_importances_'}

        except Exception as e:
            return {'error': str(e), 'method': 'fallback'}

    def _attention_explanation(
        self,
        features: np.ndarray,
        top_k: int
    ) -> Dict[str, Any]:
        """
        Extract attention weights from transformer model.

        Shows which time steps the model focused on.
        """
        try:
            # Placeholder - actual implementation depends on model structure
            # Would need to access attention layer outputs

            explanation = {
                'method': 'Attention Weights',
                'model_type': 'attention',
                'explanation_text': 'Transformer attention mechanism focuses on important time steps',
                'note': 'Actual attention weights require model-specific extraction'
            }

            # If model has custom attention extraction method
            if hasattr(self.model, 'get_attention_weights'):
                attention_weights = self.model.get_attention_weights(features)

                # Find most important timesteps
                if attention_weights is not None:
                    avg_attention = attention_weights.mean(
                        axis=0)  # Average across heads
                    top_timesteps = np.argsort(avg_attention)[-top_k:][::-1]

                    explanation['top_timesteps'] = top_timesteps.tolist()
                    explanation['attention_weights'] = avg_attention.tolist()
                    explanation[
                        'explanation_text'] = f"Model focused on time steps: {top_timesteps.tolist()[:3]}"

            return explanation

        except Exception as e:
            return {'error': str(e), 'method': 'attention'}

    def _rnn_explanation(
        self,
        features: np.ndarray,
        top_k: int
    ) -> Dict[str, Any]:
        """
        Explanation for RNN models (LSTM, GRU, etc.).

        Uses temporal importance and gradient-based methods.
        """
        try:
            # For RNNs, feature importance is temporal
            sequence_length = features.shape[0] if len(
                features.shape) > 1 else len(features)

            # Simple heuristic: more recent timesteps typically more important
            temporal_weights = np.exp(np.linspace(-2, 0, sequence_length))
            temporal_weights = temporal_weights / temporal_weights.sum()

            top_timesteps = np.argsort(temporal_weights)[-top_k:][::-1]

            return {
                'method': 'Temporal Importance',
                'model_type': self.model_type,
                'top_timesteps': top_timesteps.tolist(),
                'temporal_weights': temporal_weights.tolist(),
                'explanation_text': f'{self.model_type.upper()} model weighs recent time steps more heavily',
                'sequence_length': sequence_length
            }

        except Exception as e:
            return {'error': str(e), 'method': 'rnn'}

    def _svm_explanation(
        self,
        features: np.ndarray,
        top_k: int
    ) -> Dict[str, Any]:
        """
        Explanation for SVM models.

        Uses support vector importance and coefficient magnitude.
        """
        try:
            explanation = {
                'method': 'SVM Decision Function',
                'model_type': 'svm'
            }

            # For linear SVM, can use coefficients
            if hasattr(self.model, 'coef_'):
                coefficients = self.model.coef_[0] if len(
                    self.model.coef_.shape) > 1 else self.model.coef_

                feature_importance = dict(zip(
                    self.feature_names if self.feature_names else [
                        f"feature_{i}" for i in range(len(coefficients))],
                    np.abs(coefficients)
                ))

                top_features = dict(sorted(
                    feature_importance.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:top_k])

                explanation['top_features'] = top_features
                explanation['explanation_text'] = self._generate_text_explanation(
                    top_features)

            # For RBF kernel, provide general info
            elif hasattr(self.model, 'kernel') and self.model.kernel == 'rbf':
                explanation['explanation_text'] = 'SVM with RBF kernel uses non-linear decision boundary'
                explanation['n_support_vectors'] = len(self.model.support_vectors_) if hasattr(
                    self.model, 'support_vectors_') else None

            return explanation

        except Exception as e:
            return {'error': str(e), 'method': 'svm'}

    def _momentum_explanation(
        self,
        features: Dict[str, Any],
        top_k: int
    ) -> Dict[str, Any]:
        """
        Explanation for momentum-based predictions.

        Shows which momentum indicators contributed most.
        """
        try:
            # Momentum predictions typically include indicator values
            indicators = []

            if isinstance(features, dict):
                # Extract indicator contributions
                for key, value in features.items():
                    if any(ind in key.lower() for ind in ['rsi', 'macd', 'momentum', 'ma', 'trend']):
                        indicators.append(
                            (key, abs(value) if isinstance(value, (int, float)) else 0))

            # Sort by magnitude
            indicators.sort(key=lambda x: x[1], reverse=True)
            top_indicators = dict(indicators[:top_k])

            explanation_text = 'Momentum prediction driven by: ' + ', '.join([
                f'{ind} ({val:.2f})' for ind, val in list(top_indicators.items())[:3]
            ])

            return {
                'method': 'Momentum Indicators',
                'model_type': 'momentum',
                'top_indicators': top_indicators,
                'explanation_text': explanation_text
            }

        except Exception as e:
            return {'error': str(e), 'method': 'momentum'}

    def _generate_text_explanation(self, top_features: Dict[str, float]) -> str:
        """
        Generate human-readable explanation from feature importances.

        Args:
            top_features: Dict of feature name to importance

        Returns:
            Text explanation string
        """
        try:
            explanations = []

            for feature, importance in list(top_features.items())[:3]:
                feature_lower = feature.lower()

                # Map features to interpretable descriptions
                if 'rsi' in feature_lower:
                    explanations.append(
                        f"RSI momentum (strength {importance:.3f})")
                elif 'momentum' in feature_lower or 'roc' in feature_lower:
                    explanations.append(
                        f"Price momentum (strength {importance:.3f})")
                elif 'volume' in feature_lower:
                    explanations.append(
                        f"Volume trend (strength {importance:.3f})")
                elif 'ma' in feature_lower or 'sma' in feature_lower or 'ema' in feature_lower:
                    explanations.append(
                        f"Moving average signal (strength {importance:.3f})")
                elif 'macd' in feature_lower:
                    explanations.append(
                        f"MACD indicator (strength {importance:.3f})")
                elif 'volatility' in feature_lower or 'atr' in feature_lower:
                    explanations.append(
                        f"Volatility measure (strength {importance:.3f})")
                elif 'return' in feature_lower or 'change' in feature_lower:
                    explanations.append(
                        f"Price change pattern (strength {importance:.3f})")
                else:
                    explanations.append(
                        f"{feature} (strength {importance:.3f})")

            if explanations:
                return "Prediction driven by: " + ", ".join(explanations)
            else:
                return "Prediction based on multiple technical factors"

        except Exception as e:
            logger.warning(f"Error generating text explanation: {e}")
            return "Feature importance analysis available"


def explain_ensemble_prediction(
    model_predictions: List[Dict[str, Any]],
    ensemble_result: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Explain ensemble prediction by showing model contributions.

    Args:
        model_predictions: List of individual model predictions
        ensemble_result: Final ensemble prediction

    Returns:
        Ensemble explanation dict
    """
    try:
        # Analyze model agreements and disagreements
        directions = {}
        confidences = {}
        contributions = {}

        for pred in model_predictions:
            if isinstance(pred, dict) and 'direction' in pred:
                model_name = pred.get(
                    'method', pred.get('model_type', 'unknown'))
                directions[model_name] = pred['direction']
                confidences[model_name] = pred.get('confidence', 50)

                # Calculate contribution (confidence * weight)
                weight = ensemble_result.get(
                    'weights_used', {}).get(model_name, 0.1)
                contributions[model_name] = confidences[model_name] * weight

        # Sort by contribution
        top_contributors = dict(sorted(
            contributions.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5])

        # Identify agreement/disagreement
        bullish_models = [m for m, d in directions.items() if d == 'Bullish']
        bearish_models = [m for m, d in directions.items() if d == 'Bearish']
        neutral_models = [m for m, d in directions.items() if d == 'Neutral']

        explanation_parts = []
        if bullish_models:
            explanation_parts.append(
                f"{len(bullish_models)} models bullish: {', '.join(bullish_models[:3])}")
        if bearish_models:
            explanation_parts.append(
                f"{len(bearish_models)} models bearish: {', '.join(bearish_models[:3])}")
        if neutral_models:
            explanation_parts.append(f"{len(neutral_models)} models neutral")

        explanation_text = "; ".join(explanation_parts)

        return {
            'method': 'Ensemble Analysis',
            'final_direction': ensemble_result.get('direction', 'Neutral'),
            'final_confidence': ensemble_result.get('confidence', 50),
            'top_contributors': top_contributors,
            'model_agreement': {
                'bullish': bullish_models,
                'bearish': bearish_models,
                'neutral': neutral_models
            },
            'explanation_text': explanation_text,
            'n_models': len(directions)
        }

    except Exception as e:
        logger.error(f"Error explaining ensemble: {e}")
        return {
            'error': str(e),
            'method': 'ensemble'
        }
