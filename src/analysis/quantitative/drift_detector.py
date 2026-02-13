"""
Drift Detection System

Monitors for distribution shifts in financial data that indicate model retraining is needed.

Detection Methods:
- Statistical tests (Kolmogorov-Smirnov test)
- Volatility regime changes
- Return distribution shifts
- Volume pattern changes
- Trend breaks

Based on research (2024):
- Markets shift regimes every 3-12 months
- Volatility clustering indicates distribution change
- Drift detection improves long-term accuracy by 5-10%
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class DriftDetector:
    """Detects distribution shifts in financial data"""

    def __init__(self, baseline_window: int = 252):
        """
        Initialize drift detector.

        Args:
            baseline_window: Days of historical data for baseline (default 1 year)
        """
        self.baseline_window = baseline_window
        self.baselines = {}  # Store baseline stats per symbol

    def detect_drift(
        self,
        symbol: str,
        current_data: pd.DataFrame,
        thresholds: Optional[Dict[str, float]] = None
    ) -> float:
        """
        Calculate comprehensive drift score.

        Args:
            symbol: Stock ticker
            current_data: Recent price data
            thresholds: Custom drift thresholds

        Returns:
            Drift score (0-1, higher = more drift)
        """
        if thresholds is None:
            thresholds = {
                'volatility_change': 0.3,     # 30% volatility change
                'ks_statistic': 0.15,         # KS test threshold
                'volume_change': 0.5,          # 50% volume change
                'trend_correlation': 0.7       # Trend correlation threshold
            }

        # Get or create baseline
        baseline = self._get_baseline(symbol, current_data)

        if baseline is None:
            logger.warning(f"No baseline for {symbol}, creating new baseline")
            self._update_baseline(symbol, current_data)
            return 0.0

        drift_scores = []
        weights = []

        # 1. Volatility drift (40% weight)
        vol_drift, vol_score = self._detect_volatility_drift(
            current_data, baseline, thresholds['volatility_change']
        )
        drift_scores.append(vol_score)
        weights.append(0.4)

        # 2. Return distribution drift (30% weight)
        ks_drift, ks_score = self._detect_distribution_drift(
            current_data, baseline, thresholds['ks_statistic']
        )
        drift_scores.append(ks_score)
        weights.append(0.3)

        # 3. Volume drift (20% weight)
        vol_change, vol_score = self._detect_volume_drift(
            current_data, baseline, thresholds['volume_change']
        )
        drift_scores.append(vol_score)
        weights.append(0.2)

        # 4. Trend change (10% weight)
        trend_corr, trend_score = self._detect_trend_change(
            current_data, baseline, thresholds['trend_correlation']
        )
        drift_scores.append(trend_score)
        weights.append(0.1)

        # Weighted average drift score
        total_drift = sum(s * w for s, w in zip(drift_scores, weights))

        # Tiered logging based on drift severity
        if total_drift > 0.25:  # Severe drift (>25%)
            logger.warning(f"[{symbol}] SEVERE DRIFT: {total_drift:.3f} "
                           f"(vol:{drift_scores[0]:.2f}, dist:{drift_scores[1]:.2f}, "
                           f"volume:{drift_scores[2]:.2f}, trend:{drift_scores[3]:.2f})")
        elif total_drift > 0.15:  # Elevated drift (above threshold)
            logger.info(f"[{symbol}] Drift above threshold: {total_drift:.3f} "
                        f"(vol:{drift_scores[0]:.2f}, dist:{drift_scores[1]:.2f}, "
                        f"volume:{drift_scores[2]:.2f}, trend:{drift_scores[3]:.2f})")
        else:  # Normal operation
            logger.debug(f"[{symbol}] Drift Score: {total_drift:.3f} "
                         f"(vol:{drift_scores[0]:.2f}, dist:{drift_scores[1]:.2f}, "
                         f"volume:{drift_scores[2]:.2f}, trend:{drift_scores[3]:.2f})")

        return total_drift

    def _detect_volatility_drift(
        self,
        current_data: pd.DataFrame,
        baseline: Dict[str, Any],
        threshold: float
    ) -> Tuple[float, float]:
        """
        Detect volatility regime change.

        Returns:
            (volatility_change, drift_score)
        """
        try:
            # Current volatility (annualized)
            current_returns = current_data['Close'].pct_change().dropna()
            current_vol = current_returns.std() * np.sqrt(252)

            # Baseline volatility
            baseline_vol = baseline['volatility']

            # Calculate relative change
            vol_change = abs(current_vol - baseline_vol) / baseline_vol

            # Normalize to drift score (0-1)
            drift_score = min(vol_change / threshold, 1.0)

            return vol_change, drift_score

        except Exception as e:
            logger.warning(f"Volatility drift detection failed: {e}")
            return 0.0, 0.0

    def _detect_distribution_drift(
        self,
        current_data: pd.DataFrame,
        baseline: Dict[str, Any],
        threshold: float
    ) -> Tuple[float, float]:
        """
        Detect distribution shift using Kolmogorov-Smirnov test.

        Returns:
            (ks_statistic, drift_score)
        """
        try:
            from scipy import stats

            # Current returns
            current_returns = current_data['Close'].pct_change(
            ).dropna().values

            # Baseline returns
            baseline_returns = baseline['returns']

            # Perform KS test
            ks_statistic, p_value = stats.ks_2samp(
                current_returns, baseline_returns)

            # Normalize to drift score
            drift_score = min(ks_statistic / threshold, 1.0)

            return ks_statistic, drift_score

        except Exception as e:
            logger.warning(f"Distribution drift detection failed: {e}")
            return 0.0, 0.0

    def _detect_volume_drift(
        self,
        current_data: pd.DataFrame,
        baseline: Dict[str, Any],
        threshold: float
    ) -> Tuple[float, float]:
        """
        Detect volume pattern change.

        Returns:
            (volume_change, drift_score)
        """
        try:
            if 'Volume' not in current_data.columns:
                return 0.0, 0.0

            # Current average volume
            current_vol_avg = current_data['Volume'].mean()

            # Baseline average volume
            baseline_vol_avg = baseline.get('volume_avg', current_vol_avg)

            # Calculate relative change
            vol_change = abs(current_vol_avg -
                             baseline_vol_avg) / baseline_vol_avg

            # Normalize and cap
            drift_score = min(vol_change / threshold, 1.0)

            return vol_change, drift_score

        except Exception as e:
            logger.warning(f"Volume drift detection failed: {e}")
            return 0.0, 0.0

    def _detect_trend_change(
        self,
        current_data: pd.DataFrame,
        baseline: Dict[str, Any],
        threshold: float
    ) -> Tuple[float, float]:
        """
        Detect trend direction change.

        Returns:
            (trend_correlation, drift_score)
        """
        try:
            # Current trend (linear regression slope)
            current_prices = current_data['Close'].values
            current_trend = np.polyfit(
                range(len(current_prices)), current_prices, 1)[0]

            # Baseline trend
            baseline_trend = baseline.get('trend', 0.0)

            # Calculate correlation between trends
            if baseline_trend != 0:
                # Sign agreement and magnitude similarity
                sign_agreement = 1.0 if np.sign(
                    current_trend) == np.sign(baseline_trend) else 0.0
                magnitude_ratio = min(abs(current_trend), abs(
                    baseline_trend)) / max(abs(current_trend), abs(baseline_trend))

                trend_correlation = sign_agreement * magnitude_ratio
            else:
                trend_correlation = 1.0  # No baseline trend to compare

            # Drift score: inverse of correlation
            drift_score = max(0.0, 1.0 - trend_correlation / threshold)

            return trend_correlation, drift_score

        except Exception as e:
            logger.warning(f"Trend drift detection failed: {e}")
            return 1.0, 0.0

    def _get_baseline(self, symbol: str, data: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Get baseline statistics for symbol"""
        if symbol in self.baselines:
            return self.baselines[symbol]

        # Try to create from historical data
        if len(data) >= self.baseline_window:
            baseline_data = data.iloc[-self.baseline_window:]
            self._update_baseline(symbol, baseline_data)
            return self.baselines.get(symbol)

        return None

    def _update_baseline(self, symbol: str, data: pd.DataFrame):
        """Update baseline statistics"""
        try:
            # Calculate baseline stats
            returns = data['Close'].pct_change().dropna()

            baseline = {
                'volatility': returns.std() * np.sqrt(252),  # Annualized
                'returns': returns.values,
                'volume_avg': data['Volume'].mean() if 'Volume' in data.columns else 0,
                'trend': np.polyfit(range(len(data)), data['Close'].values, 1)[0],
                'timestamp': datetime.now().isoformat(),
                'n_samples': len(data)
            }

            self.baselines[symbol] = baseline
            logger.info(f"Updated baseline for {symbol}")

        except Exception as e:
            logger.error(f"Failed to update baseline for {symbol}: {e}")

    def should_update_baseline(
        self,
        symbol: str,
        days_since_update: int,
        max_age_days: int = 90
    ) -> bool:
        """
        Check if baseline should be updated.

        Args:
            symbol: Stock ticker
            days_since_update: Days since last baseline update
            max_age_days: Maximum baseline age before refresh

        Returns:
            True if baseline should be updated
        """
        return days_since_update >= max_age_days

    def get_drift_report(self, symbol: str) -> Dict[str, Any]:
        """Get detailed drift analysis report"""
        baseline = self.baselines.get(symbol)

        if baseline is None:
            return {'error': f'No baseline for {symbol}'}

        return {
            'symbol': symbol,
            'baseline_stats': {
                'volatility': baseline['volatility'],
                'volume_avg': baseline['volume_avg'],
                'trend': baseline['trend'],
                'n_samples': baseline['n_samples'],
                'timestamp': baseline['timestamp']
            },
            'age_days': (datetime.now() - datetime.fromisoformat(baseline['timestamp'])).days
        }


def detect_regime_change(
    stock_data: pd.DataFrame,
    window: int = 60
) -> Dict[str, Any]:
    """
    Detect market regime changes (bull/bear/sideways transitions).

    Args:
        stock_data: Price data
        window: Rolling window for regime detection

    Returns:
        Dict with regime info and change detection
    """
    try:
        # Calculate returns and volatility
        returns = stock_data['Close'].pct_change().dropna()

        # Rolling statistics
        rolling_return = returns.rolling(window).mean()
        rolling_vol = returns.rolling(window).std()

        # Current regime
        current_return = rolling_return.iloc[-1]
        current_vol = rolling_vol.iloc[-1]

        # Classify regime
        if current_return > 0.001 and current_vol < 0.02:  # Positive return, low vol
            current_regime = 'bullish'
        elif current_return < -0.001 and current_vol > 0.02:  # Negative return, high vol
            current_regime = 'bearish'
        else:
            current_regime = 'sideways'

        # Historical regime
        historical_return = rolling_return.iloc[-window:-1].mean()
        historical_vol = rolling_vol.iloc[-window:-1].mean()

        if historical_return > 0.001 and historical_vol < 0.02:
            historical_regime = 'bullish'
        elif historical_return < -0.001 and historical_vol > 0.02:
            historical_regime = 'bearish'
        else:
            historical_regime = 'sideways'

        # Detect change
        regime_changed = current_regime != historical_regime

        return {
            'current_regime': current_regime,
            'historical_regime': historical_regime,
            'regime_changed': regime_changed,
            'current_return': float(current_return),
            'current_volatility': float(current_vol),
            'change_magnitude': abs(current_return - historical_return)
        }

    except Exception as e:
        logger.error(f"Regime change detection failed: {e}")
        return {
            'current_regime': 'unknown',
            'error': str(e)
        }
