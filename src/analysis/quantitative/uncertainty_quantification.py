"""
Uncertainty Quantification System

Provides prediction intervals and uncertainty decomposition for all model types:
- Quantile regression for neural networks (LSTM, GRU, Attention, etc.)
- Bootstrap confidence intervals for tree models (XGBoost, LightGBM, RF)
- Ensemble uncertainty aggregation
- Epistemic (model) vs Aleatoric (data) uncertainty separation

Based on research:
- Quantile regression provides better calibrated intervals than simple std dev
- Bootstrap sampling captures model uncertainty in tree ensembles
- Epistemic uncertainty: variance across different models
- Aleatoric uncertainty: inherent data noise (volatility proxy)
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class UncertaintyQuantifier:
    """Unified uncertainty estimation for all model types"""

    def __init__(self, model_type: str = 'ensemble'):
        """
        Initialize uncertainty quantifier.

        Args:
            model_type: Type of model ('lstm', 'gru', 'xgboost', 'rf', 'lightgbm',
                       'svm', 'ensemble')
        """
        self.model_type = model_type
        self.confidence_levels = {
            '1_sigma': 0.68,  # 68% confidence (±1σ)
            '2_sigma': 0.95,  # 95% confidence (±2σ)
            '3_sigma': 0.997  # 99.7% confidence (±3σ)
        }

    def calculate_prediction_intervals(
        self,
        predictions: np.ndarray,
        confidence_level: float = 0.68,
        method: str = 'auto',
        historical_errors: Optional[np.ndarray] = None,
        volatility: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Calculate prediction intervals using appropriate method.

        Args:
            predictions: Array of predictions (can be single or multiple)
            confidence_level: Desired confidence level (0.68 for 1σ, 0.95 for 2σ)
            method: 'quantile', 'bootstrap', 'ensemble', or 'auto'
            historical_errors: Historical prediction errors for calibration
            volatility: Market volatility for scaling

        Returns:
            Dictionary with lower, median, upper bounds and interval width
        """
        if method == 'auto':
            # Auto-select method based on model type
            if self.model_type in ['lstm', 'gru', 'attention', 'cnn_lstm', 'tcn', 'nbeats']:
                method = 'quantile'
            elif self.model_type in ['xgboost', 'lightgbm', 'random_forest']:
                method = 'bootstrap'
            else:
                method = 'ensemble'

        if method == 'quantile':
            return self._quantile_intervals(predictions, confidence_level, historical_errors, volatility)
        elif method == 'bootstrap':
            return self._bootstrap_intervals(predictions, confidence_level, historical_errors, volatility)
        elif method == 'ensemble':
            return self._ensemble_intervals(predictions, confidence_level)
        else:
            raise ValueError(f"Unknown method: {method}")

    def _quantile_intervals(
        self,
        predictions: np.ndarray,
        confidence_level: float,
        historical_errors: Optional[np.ndarray] = None,
        volatility: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Quantile regression intervals for neural networks.

        Uses historical prediction errors to estimate quantiles.
        If not available, uses volatility-based approximation.
        """
        median = np.median(predictions) if len(
            predictions.shape) > 0 and predictions.size > 1 else float(predictions)

        # Calculate quantiles
        lower_q = (1 - confidence_level) / 2
        upper_q = 1 - lower_q

        if historical_errors is not None and len(historical_errors) > 5:
            # Use empirical quantiles from historical errors
            lower_quantile = np.quantile(historical_errors, lower_q)
            upper_quantile = np.quantile(historical_errors, upper_q)

            lower = median + lower_quantile
            upper = median + upper_quantile
        else:
            # Fallback: Use volatility-based approximation
            # For 68% CI: ±1σ, for 95% CI: ±1.96σ
            if volatility is None:
                # Estimate from prediction variance
                if len(predictions.shape) > 0 and predictions.size > 1:
                    volatility = np.std(predictions)
                else:
                    volatility = abs(median) * 0.02  # 2% default volatility

            # Calculate z-score for confidence level
            if confidence_level == 0.68:
                z_score = 1.0
            elif confidence_level == 0.95:
                z_score = 1.96
            elif confidence_level == 0.997:
                z_score = 3.0
            else:
                # Inverse normal CDF approximation
                from scipy import stats
                z_score = stats.norm.ppf(upper_q)

            interval_width = z_score * volatility
            lower = median - interval_width
            upper = median + interval_width

        return {
            'lower': float(lower),
            'median': float(median),
            'upper': float(upper),
            'interval_width': float(upper - lower),
            'confidence_level': confidence_level,
            'method': 'quantile_regression'
        }

    def _bootstrap_intervals(
        self,
        predictions: np.ndarray,
        confidence_level: float,
        historical_errors: Optional[np.ndarray] = None,
        volatility: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Bootstrap confidence intervals for tree models.

        Simulates model uncertainty by resampling predictions
        or using out-of-bag estimates.
        """
        median = np.median(predictions) if len(
            predictions.shape) > 0 and predictions.size > 1 else float(predictions)

        # Calculate quantiles
        lower_q = (1 - confidence_level) / 2
        upper_q = 1 - lower_q

        # Bootstrap sampling
        n_bootstrap = 1000
        bootstrap_preds = []

        if len(predictions.shape) > 0 and predictions.size > 1:
            # Have multiple predictions (e.g., from ensemble)
            for _ in range(n_bootstrap):
                sample = np.random.choice(
                    predictions, size=len(predictions), replace=True)
                bootstrap_preds.append(np.mean(sample))
        else:
            # Single prediction - use volatility-based perturbation
            if volatility is None:
                volatility = abs(median) * 0.02  # 2% default

            for _ in range(n_bootstrap):
                noise = np.random.normal(0, volatility)
                bootstrap_preds.append(median + noise)

        bootstrap_preds = np.array(bootstrap_preds)

        lower = np.quantile(bootstrap_preds, lower_q)
        upper = np.quantile(bootstrap_preds, upper_q)

        return {
            'lower': float(lower),
            'median': float(median),
            'upper': float(upper),
            'interval_width': float(upper - lower),
            'confidence_level': confidence_level,
            'method': 'bootstrap',
            'n_samples': n_bootstrap
        }

    def _ensemble_intervals(
        self,
        predictions: np.ndarray,
        confidence_level: float
    ) -> Dict[str, Any]:
        """
        Ensemble-based intervals using prediction variance.

        Uses the spread of predictions from multiple models.
        """
        if len(predictions.shape) == 0 or predictions.size == 1:
            # Single prediction - no variance
            return {
                'lower': float(predictions),
                'median': float(predictions),
                'upper': float(predictions),
                'interval_width': 0.0,
                'confidence_level': confidence_level,
                'method': 'ensemble',
                'warning': 'Single prediction - no uncertainty estimation possible'
            }

        median = np.median(predictions)
        std = np.std(predictions)

        # Calculate z-score for confidence level
        if confidence_level == 0.68:
            z_score = 1.0
        elif confidence_level == 0.95:
            z_score = 1.96
        elif confidence_level == 0.997:
            z_score = 3.0
        else:
            from scipy import stats
            z_score = stats.norm.ppf((1 + confidence_level) / 2)

        lower = median - z_score * std
        upper = median + z_score * std

        return {
            'lower': float(lower),
            'median': float(median),
            'upper': float(upper),
            'interval_width': float(upper - lower),
            'confidence_level': confidence_level,
            'method': 'ensemble',
            'n_models': len(predictions)
        }

    def decompose_uncertainty(
        self,
        model_predictions: List[Dict[str, Any]],
        market_volatility: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Separate epistemic (model) and aleatoric (data) uncertainty.

        Args:
            model_predictions: List of prediction dicts from different models
            market_volatility: Current market volatility (annualized)

        Returns:
            Dictionary with epistemic, aleatoric, and total uncertainty
        """
        # Extract confidence scores and predictions
        confidences = []
        predicted_changes = []

        for pred in model_predictions:
            if isinstance(pred, dict) and 'confidence' in pred:
                confidences.append(pred['confidence'])

            if isinstance(pred, dict) and 'predicted_change_pct' in pred:
                change = pred['predicted_change_pct']
                if change not in [None, 'N/A', 'nan'] and not (isinstance(change, float) and np.isnan(change)):
                    predicted_changes.append(float(change))

        # Epistemic uncertainty: Variance across different models
        if len(confidences) > 1:
            # Normalize confidences to [0, 1]
            conf_normalized = np.array(confidences) / 100.0
            epistemic = float(np.var(conf_normalized))
        else:
            epistemic = 0.0

        # Aleatoric uncertainty: Inherent data noise
        if market_volatility is not None:
            # Use market volatility as proxy for data uncertainty
            aleatoric = float(market_volatility / 100.0)  # Normalize to [0, 1]
        elif len(predicted_changes) > 1:
            # Use variance of predicted changes
            # Scale to reasonable range
            aleatoric = float(np.var(predicted_changes) / 10000.0)
        else:
            aleatoric = 0.02  # Default 2% uncertainty

        # Total uncertainty (assuming independence)
        total = float(np.sqrt(epistemic**2 + aleatoric**2))

        return {
            'epistemic': epistemic,           # Model uncertainty
            'aleatoric': aleatoric,           # Data uncertainty
            'total': total,                   # Combined uncertainty
            'epistemic_pct': epistemic * 100,  # As percentage
            'aleatoric_pct': aleatoric * 100,  # As percentage
            'total_pct': total * 100          # As percentage
        }

    def calibrate_intervals(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
        confidence_level: float = 0.68
    ) -> Dict[str, Any]:
        """
        Calibrate prediction intervals using historical data.

        Checks if actual values fall within predicted intervals
        at the specified confidence level.

        Args:
            predictions: Historical predictions
            actuals: Actual observed values
            confidence_level: Target confidence level

        Returns:
            Calibration metrics and adjustment factors
        """
        if len(predictions) != len(actuals):
            raise ValueError("Predictions and actuals must have same length")

        # Calculate errors
        errors = actuals - predictions

        # Calculate empirical quantiles
        lower_q = (1 - confidence_level) / 2
        upper_q = 1 - lower_q

        empirical_lower = np.quantile(errors, lower_q)
        empirical_upper = np.quantile(errors, upper_q)

        # Check coverage (% of actuals within intervals)
        intervals_lower = predictions + empirical_lower
        intervals_upper = predictions + empirical_upper

        within_interval = np.sum(
            (actuals >= intervals_lower) & (actuals <= intervals_upper))
        coverage = within_interval / len(actuals)

        # Calibration score (how close to target confidence level)
        calibration_error = abs(coverage - confidence_level)

        return {
            'target_confidence': confidence_level,
            'empirical_coverage': float(coverage),
            'calibration_error': float(calibration_error),
            'lower_quantile': float(empirical_lower),
            'upper_quantile': float(empirical_upper),
            'is_well_calibrated': calibration_error < 0.05,  # Within 5%
            'mean_absolute_error': float(np.mean(np.abs(errors))),
            'rmse': float(np.sqrt(np.mean(errors**2)))
        }


def add_uncertainty_to_prediction(
    prediction: Dict[str, Any],
    model_type: str,
    stock_data: Optional[pd.DataFrame] = None,
    confidence_level: float = 0.68
) -> Dict[str, Any]:
    """
    Add uncertainty quantification to a model prediction.

    Args:
        prediction: Model prediction dictionary
        model_type: Type of model
        stock_data: Historical stock data for volatility calculation
        confidence_level: Desired confidence level

    Returns:
        Updated prediction dict with uncertainty intervals
    """
    quantifier = UncertaintyQuantifier(model_type)

    # Get current price and predicted change
    current_price = prediction.get('current_price', 100.0)
    predicted_change_pct = prediction.get('predicted_change_pct', 0.0)

    # Handle N/A or nan values
    if predicted_change_pct in [None, 'N/A', 'nan'] or (isinstance(predicted_change_pct, float) and np.isnan(predicted_change_pct)):
        predicted_change_pct = 0.0

    # Calculate predicted price
    predicted_price = current_price * (1 + predicted_change_pct / 100.0)

    # Calculate market volatility if stock data available
    volatility = None
    if stock_data is not None and 'Close' in stock_data.columns:
        returns = stock_data['Close'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252) * 100  # Annualized volatility

    # Calculate prediction intervals
    intervals = quantifier.calculate_prediction_intervals(
        predictions=np.array([predicted_price]),
        confidence_level=confidence_level,
        volatility=volatility
    )

    # Add to prediction
    prediction['uncertainty'] = {
        'confidence_level': confidence_level,
        'predicted_price': {
            'lower': intervals['lower'],
            'median': intervals['median'],
            'upper': intervals['upper']
        },
        'predicted_change_pct': {
            'lower': (intervals['lower'] / current_price - 1) * 100,
            'median': predicted_change_pct,
            'upper': (intervals['upper'] / current_price - 1) * 100
        },
        'interval_width': intervals['interval_width'],
        'interval_width_pct': (intervals['interval_width'] / current_price) * 100,
        'method': intervals['method']
    }

    return prediction


def get_ensemble_uncertainty(
    model_predictions: List[Dict[str, Any]],
    stock_data: Optional[pd.DataFrame] = None
) -> Dict[str, Any]:
    """
    Calculate ensemble-level uncertainty from multiple model predictions.

    Args:
        model_predictions: List of predictions from all models
        stock_data: Historical stock data for volatility

    Returns:
        Comprehensive uncertainty analysis
    """
    quantifier = UncertaintyQuantifier('ensemble')

    # Extract predictions
    valid_predictions = []
    for pred in model_predictions:
        if isinstance(pred, dict) and 'current_price' in pred:
            current_price = pred['current_price']
            change_pct = pred.get('predicted_change_pct', 0.0)

            if change_pct not in [None, 'N/A', 'nan'] and not (isinstance(change_pct, float) and np.isnan(change_pct)):
                predicted_price = current_price * \
                    (1 + float(change_pct) / 100.0)
                valid_predictions.append(predicted_price)

    if len(valid_predictions) < 2:
        return {
            'error': 'Insufficient predictions for ensemble uncertainty',
            'n_predictions': len(valid_predictions)
        }

    # Calculate market volatility
    volatility = None
    if stock_data is not None and 'Close' in stock_data.columns:
        returns = stock_data['Close'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252) * 100

    # Get intervals for different confidence levels
    intervals_68 = quantifier.calculate_prediction_intervals(
        np.array(valid_predictions), confidence_level=0.68
    )

    intervals_95 = quantifier.calculate_prediction_intervals(
        np.array(valid_predictions), confidence_level=0.95
    )

    # Decompose uncertainty
    uncertainty_decomp = quantifier.decompose_uncertainty(
        model_predictions,
        market_volatility=volatility
    )

    return {
        'intervals_1_sigma': intervals_68,
        'intervals_2_sigma': intervals_95,
        'uncertainty_decomposition': uncertainty_decomp,
        'n_models': len(valid_predictions),
        'prediction_std': float(np.std(valid_predictions)),
        'coefficient_of_variation': float(np.std(valid_predictions) / abs(np.mean(valid_predictions)))
    }
