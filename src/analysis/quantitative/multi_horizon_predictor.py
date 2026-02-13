"""
Multi-Horizon Prediction System

Enables predictions across multiple time horizons:
- 1-day: Day trading signals
- 5-day: Swing trading (current default)
- 20-day: Monthly outlook
- 60-day: Quarterly outlook

Features:
- Horizon-specific model configurations
- Uncertainty scaling by time horizon
- Different indicator periods per horizon
- Adaptive confidence adjustment

Based on research (2024):
- Prediction accuracy decreases with sqrt(time)
- Uncertainty scales roughly with sqrt(horizon)
- Different horizons require different technical indicators
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)


class MultiHorizonPredictor:
    """Unified interface for multi-timeframe predictions"""

    HORIZONS = {
        '1d': 1,     # Day trading
        '5d': 5,     # Swing trading (default)
        '20d': 20,   # Monthly outlook
        '60d': 60    # Quarterly outlook
    }

    # Horizon-specific configuration
    HORIZON_CONFIG = {
        '1d': {
            'ma_periods': [5, 10, 20],
            'rsi_period': 7,
            'volatility_window': 5,
            'confidence_scale': 1.0
        },
        '5d': {
            'ma_periods': [10, 20, 50],
            'rsi_period': 14,
            'volatility_window': 20,
            'confidence_scale': 0.9
        },
        '20d': {
            'ma_periods': [20, 50, 100],
            'rsi_period': 21,
            'volatility_window': 60,
            'confidence_scale': 0.75
        },
        '60d': {
            'ma_periods': [50, 100, 200],
            'rsi_period': 28,
            'volatility_window': 120,
            'confidence_scale': 0.6
        }
    }

    def __init__(self, base_predictor=None):
        """
        Initialize multi-horizon predictor.

        Args:
            base_predictor: Base prediction function/class to wrap
        """
        self.predictor = base_predictor
        self.horizons = self.HORIZONS

    def predict_all_horizons(
        self,
        stock_data: pd.DataFrame,
        model_name: str = 'ensemble',
        include_uncertainty: bool = True
    ) -> Dict[str, Dict]:
        """
        Generate predictions for all timeframes.

        Args:
            stock_data: Historical price data
            model_name: Name of model for configuration
            include_uncertainty: Whether to calculate uncertainty

        Returns:
            Dict mapping horizon name to prediction dict
        """
        results = {}

        for horizon_name, horizon_days in self.horizons.items():
            try:
                logger.info(
                    f"Generating {horizon_name} ({horizon_days}-day) prediction")

                # Get horizon-specific prediction
                pred = self._predict_single_horizon(
                    stock_data,
                    horizon_name,
                    horizon_days,
                    model_name
                )

                # Scale uncertainty by horizon
                if include_uncertainty and 'confidence' in pred:
                    pred['confidence'] = self._scale_confidence_by_horizon(
                        pred['confidence'],
                        horizon_days
                    )

                    # Add horizon-adjusted uncertainty
                    pred['uncertainty_adjusted'] = self._calculate_horizon_uncertainty(
                        horizon_days,
                        stock_data
                    )

                results[horizon_name] = pred

            except Exception as e:
                logger.error(f"Failed to predict {horizon_name}: {e}")
                results[horizon_name] = {
                    'error': str(e),
                    'horizon_days': horizon_days,
                    'status': 'failed'
                }

        return results

    def _predict_single_horizon(
        self,
        stock_data: pd.DataFrame,
        horizon_name: str,
        horizon_days: int,
        model_name: str
    ) -> Dict[str, Any]:
        """
        Generate prediction for a single time horizon.

        Args:
            stock_data: Historical price data
            horizon_name: Name of horizon (1d, 5d, etc.)
            horizon_days: Number of days in horizon
            model_name: Model name for configuration

        Returns:
            Prediction dict for this horizon
        """
        config = self.HORIZON_CONFIG[horizon_name]

        # Calculate horizon-specific features
        features = self._calculate_horizon_features(stock_data, config)

        # Simple trend-based prediction (placeholder for model integration)
        current_price = stock_data['Close'].iloc[-1]

        # Use multiple MAs for trend detection
        trend_signals = []
        for ma_period in config['ma_periods']:
            if len(stock_data) > ma_period:
                ma = stock_data['Close'].rolling(ma_period).mean().iloc[-1]
                trend_signals.append(1 if current_price > ma else -1)

        # Aggregate trend
        trend_score = sum(trend_signals) / len(trend_signals)

        # Calculate volatility for uncertainty
        returns = stock_data['Close'].pct_change().tail(
            config['volatility_window'])
        volatility = returns.std() * np.sqrt(252) * 100  # Annualized

        # Predict price change (scaled by horizon)
        # Rule of thumb: longer horizons have larger magnitude changes
        base_change = trend_score * 2.0  # Base 2% per signal
        horizon_multiplier = np.sqrt(horizon_days / 5)  # Scale from base 5-day
        predicted_change_pct = base_change * horizon_multiplier

        # Calculate predicted price
        predicted_price = current_price * (1 + predicted_change_pct / 100)

        # Determine direction
        if predicted_change_pct > 0.5:
            direction = "Bullish"
        elif predicted_change_pct < -0.5:
            direction = "Bearish"
        else:
            direction = "Neutral"

        # Base confidence (higher for shorter horizons)
        base_confidence = 70.0 * config['confidence_scale']

        # Adjust by volatility (higher volatility = lower confidence)
        volatility_penalty = min(volatility / 2, 20)  # Cap at 20% penalty
        confidence = max(30, base_confidence - volatility_penalty)

        return {
            'horizon': horizon_name,
            'horizon_days': horizon_days,
            'current_price': float(current_price),
            'predicted_price': float(predicted_price),
            'predicted_change_pct': round(predicted_change_pct, 2),
            'direction': direction,
            'confidence': round(confidence, 1),
            'volatility': round(volatility, 2),
            'trend_score': round(trend_score, 2),
            'features': features,
            'config': config
        }

    def _calculate_horizon_features(
        self,
        stock_data: pd.DataFrame,
        config: Dict
    ) -> Dict[str, float]:
        """Calculate features specific to a time horizon"""
        features = {}

        try:
            # Moving averages
            for i, period in enumerate(config['ma_periods']):
                if len(stock_data) > period:
                    ma = stock_data['Close'].rolling(period).mean().iloc[-1]
                    features[f'ma_{period}'] = float(ma)
                    features[f'price_to_ma_{period}'] = float(
                        stock_data['Close'].iloc[-1] / ma - 1
                    ) * 100

            # RSI
            rsi_period = config['rsi_period']
            if len(stock_data) > rsi_period:
                delta = stock_data['Close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(rsi_period).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(rsi_period).mean()
                rs = gain / (loss + 1e-10)
                rsi = 100 - (100 / (1 + rs))
                features['rsi'] = float(rsi.iloc[-1])

            # Volatility
            vol_window = config['volatility_window']
            if len(stock_data) > vol_window:
                returns = stock_data['Close'].pct_change().tail(vol_window)
                features['volatility'] = float(
                    returns.std() * np.sqrt(252) * 100)

            # Volume trend (if available)
            if 'Volume' in stock_data.columns and len(stock_data) > 20:
                vol_ma = stock_data['Volume'].rolling(20).mean()
                features['volume_ratio'] = float(
                    stock_data['Volume'].iloc[-1] / vol_ma.iloc[-1]
                )

        except Exception as e:
            logger.warning(f"Error calculating horizon features: {e}")

        return features

    def _scale_confidence_by_horizon(
        self,
        base_confidence: float,
        horizon_days: int
    ) -> float:
        """
        Scale confidence by time horizon.

        Uncertainty increases with sqrt(time), so confidence decreases.

        Args:
            base_confidence: Original confidence (0-100)
            horizon_days: Forecast horizon in days

        Returns:
            Scaled confidence
        """
        # Base horizon (5 days)
        base_horizon = 5

        # Scaling factor: sqrt(horizon/base)
        scaling_factor = np.sqrt(horizon_days / base_horizon)

        # Scale confidence down for longer horizons
        scaled_confidence = base_confidence / scaling_factor

        # Keep within reasonable bounds
        return max(30, min(95, scaled_confidence))

    def _calculate_horizon_uncertainty(
        self,
        horizon_days: int,
        stock_data: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Calculate horizon-adjusted uncertainty metrics.

        Args:
            horizon_days: Forecast horizon in days
            stock_data: Historical price data

        Returns:
            Dict with uncertainty metrics
        """
        try:
            # Calculate daily volatility
            returns = stock_data['Close'].pct_change().dropna()
            daily_vol = returns.std()

            # Scale to horizon (sqrt rule)
            horizon_vol = daily_vol * np.sqrt(horizon_days)

            # Convert to percentage
            horizon_uncertainty_pct = horizon_vol * 100

            # Calculate prediction interval width
            # 68% confidence interval (±1σ)
            interval_width_pct = horizon_uncertainty_pct * 2  # ±1σ

            return {
                'daily_volatility_pct': round(daily_vol * 100, 2),
                'horizon_volatility_pct': round(horizon_uncertainty_pct, 2),
                'interval_width_68pct': round(interval_width_pct, 2),
                # ±2σ
                'interval_width_95pct': round(interval_width_pct * 1.96, 2),
                'sqrt_time_scaling': round(np.sqrt(horizon_days), 2)
            }

        except Exception as e:
            logger.warning(f"Error calculating horizon uncertainty: {e}")
            return {
                'error': str(e)
            }

    def get_best_horizon_for_strategy(
        self,
        strategy_type: str
    ) -> str:
        """
        Recommend best horizon for trading strategy.

        Args:
            strategy_type: 'day_trading', 'swing_trading', 'position_trading', 'long_term'

        Returns:
            Recommended horizon name ('1d', '5d', '20d', or '60d')
        """
        strategy_mapping = {
            'day_trading': '1d',
            'scalping': '1d',
            'swing_trading': '5d',
            'short_term': '5d',
            'position_trading': '20d',
            'medium_term': '20d',
            'long_term': '60d',
            'investment': '60d'
        }

        # Default to 5d
        return strategy_mapping.get(strategy_type.lower(), '5d')


def create_multi_horizon_summary(
    horizon_predictions: Dict[str, Dict],
    current_price: float
) -> Dict[str, Any]:
    """
    Create summary across all time horizons.

    Args:
        horizon_predictions: Dict of predictions by horizon
        current_price: Current stock price

    Returns:
        Summary dict with consensus and divergence metrics
    """
    # Extract directions
    directions = {}
    confidences = {}
    changes = {}

    for horizon, pred in horizon_predictions.items():
        if 'error' not in pred:
            directions[horizon] = pred.get('direction', 'Neutral')
            confidences[horizon] = pred.get('confidence', 50)
            changes[horizon] = pred.get('predicted_change_pct', 0)

    if not directions:
        return {'error': 'No valid predictions across horizons'}

    # Calculate consensus
    bullish_count = sum(1 for d in directions.values() if d == 'Bullish')
    bearish_count = sum(1 for d in directions.values() if d == 'Bearish')
    neutral_count = sum(1 for d in directions.values() if d == 'Neutral')

    total = len(directions)
    if bullish_count > bearish_count and bullish_count > neutral_count:
        consensus = 'Bullish'
    elif bearish_count > bullish_count and bearish_count > neutral_count:
        consensus = 'Bearish'
    else:
        consensus = 'Mixed'

    # Calculate agreement strength
    max_count = max(bullish_count, bearish_count, neutral_count)
    agreement_pct = (max_count / total) * 100 if total > 0 else 0

    # Average confidence
    avg_confidence = np.mean(list(confidences.values())) if confidences else 50

    # Trend analysis (short vs long term)
    short_term_change = changes.get('1d', changes.get('5d', 0))
    long_term_change = changes.get('60d', changes.get('20d', 0))

    if short_term_change > 0 and long_term_change > 0:
        trend_alignment = 'Both bullish'
    elif short_term_change < 0 and long_term_change < 0:
        trend_alignment = 'Both bearish'
    elif abs(short_term_change) < 0.5 and abs(long_term_change) < 0.5:
        trend_alignment = 'Both neutral'
    else:
        trend_alignment = 'Divergent'

    return {
        'consensus': consensus,
        'agreement_pct': round(agreement_pct, 1),
        'avg_confidence': round(avg_confidence, 1),
        'direction_breakdown': {
            'bullish': bullish_count,
            'bearish': bearish_count,
            'neutral': neutral_count
        },
        'trend_alignment': trend_alignment,
        'short_term': {
            'change_pct': short_term_change,
            'direction': directions.get('1d', directions.get('5d', 'N/A'))
        },
        'long_term': {
            'change_pct': long_term_change,
            'direction': directions.get('60d', directions.get('20d', 'N/A'))
        },
        'current_price': current_price,
        'horizons_analyzed': list(directions.keys())
    }
