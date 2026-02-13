"""
Model Accuracy Tracking
Track and validate ML model predictions over time to measure real-world accuracy.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from pathlib import Path
import json
import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class Prediction:
    """Store a single prediction with metadata."""

    symbol: str
    model_name: str
    prediction_date: str  # ISO format
    target_date: str  # ISO format - when prediction is for
    predicted_direction: str  # Bullish/Bearish/Neutral
    predicted_price: float
    confidence: float  # 0-100
    actual_price: Optional[float] = None
    validated: bool = False
    correct: Optional[bool] = None
    error_pct: Optional[float] = None  # MAPE

    def to_dict(self):
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


class ModelAccuracyTracker:
    """
    Track model predictions and calculate accuracy metrics.

    Stores predictions in JSONL format for persistence and easy querying.
    """

    def __init__(self, storage_path: str = "data/predictions"):
        """
        Initialize tracker with storage path.

        Args:
            storage_path: Directory to store prediction data
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.predictions_file = self.storage_path / "predictions.jsonl"

        logger.info(
            f"ModelAccuracyTracker initialized: {self.predictions_file}")

    def record_prediction(
        self,
        symbol: str,
        model_name: str,
        predicted_direction: str,
        predicted_price: float,
        confidence: float,
        horizon_days: int = 7,
    ) -> str:
        """
        Record a new prediction.

        Args:
            symbol: Stock symbol
            model_name: Name of the ML model
            predicted_direction: Bullish/Bearish/Neutral
            predicted_price: Predicted price
            confidence: Confidence score (0-100)
            horizon_days: Prediction horizon in days

        Returns:
            Prediction date (ISO format)
        """
        prediction = Prediction(
            symbol=symbol.upper(),
            model_name=model_name,
            prediction_date=datetime.now().isoformat(),
            target_date=(datetime.now() +
                         timedelta(days=horizon_days)).isoformat(),
            predicted_direction=predicted_direction,
            predicted_price=predicted_price,
            confidence=confidence,
        )

        # Append to JSONL file
        try:
            with open(self.predictions_file, 'a') as f:
                f.write(json.dumps(prediction.to_dict()) + '\n')

            logger.debug(
                f"Recorded prediction: {model_name} for {symbol} "
                f"({predicted_direction}, ${predicted_price:.2f})"
            )
        except Exception as e:
            logger.error(f"Failed to record prediction: {e}")

        return prediction.prediction_date

    def validate_predictions(self, current_prices: Dict[str, float]):
        """
        Validate predictions that have reached their target date.

        Args:
            current_prices: Dictionary of {symbol: current_price}
        """
        if not self.predictions_file.exists():
            return

        predictions = self._load_predictions()
        updated = False

        for pred in predictions:
            # Skip already validated predictions
            if pred['validated']:
                continue

            # Check if target date has been reached
            target_date = datetime.fromisoformat(pred['target_date'])
            if datetime.now() >= target_date:
                symbol = pred['symbol']

                if symbol in current_prices:
                    actual_price = current_prices[symbol]
                    pred['actual_price'] = actual_price
                    pred['validated'] = True

                    # Calculate error percentage (MAPE)
                    pred['error_pct'] = (
                        abs(actual_price - pred['predicted_price'])
                        / actual_price
                        * 100
                    )

                    # Check if direction was correct
                    predicted_change = (
                        pred['predicted_price'] - actual_price
                    )  # If positive, we predicted higher

                    if pred['predicted_direction'] == 'Bullish':
                        # Bullish means we predicted price would go UP
                        pred['correct'] = actual_price > pred['predicted_price']
                    elif pred['predicted_direction'] == 'Bearish':
                        # Bearish means we predicted price would go DOWN
                        pred['correct'] = actual_price < pred['predicted_price']
                    else:  # Neutral
                        # Neutral means price should stay relatively flat (within 2%)
                        pred['correct'] = pred['error_pct'] < 2.0

                    updated = True

                    logger.debug(
                        f"Validated prediction: {pred['model_name']} for {symbol} - "
                        f"{'Correct' if pred['correct'] else 'Incorrect'} "
                        f"(error: {pred['error_pct']:.2f}%)"
                    )

        if updated:
            self._save_predictions(predictions)
            logger.info(f"Validated predictions updated")

    def get_model_accuracy(
        self, model_name: Optional[str] = None, days: int = 90
    ) -> Dict[str, Any]:
        """
        Calculate accuracy metrics for a specific model or all models.

        Args:
            model_name: Name of model to analyze (None for all models)
            days: Number of days to look back

        Returns:
            Dictionary with accuracy metrics
        """
        predictions = self._load_predictions()

        # Filter by date range
        cutoff = datetime.now() - timedelta(days=days)
        recent = [
            p
            for p in predictions
            if datetime.fromisoformat(p['prediction_date']) > cutoff
        ]

        # Filter by model if specified
        if model_name:
            recent = [p for p in recent if p['model_name'] == model_name]

        # Get validated predictions only
        validated = [p for p in recent if p['validated']]
        correct = [p for p in validated if p['correct']]

        if len(validated) == 0:
            return {
                "model_name": model_name or "all_models",
                "total_predictions": len(recent),
                "validated": 0,
                "accuracy_pct": 0.0,
                "mape": 0.0,
                "avg_confidence": 0.0,
                "date_range_days": days,
                "note": "No validated predictions in timeframe",
            }

        # Calculate MAPE (Mean Absolute Percentage Error)
        errors = [p['error_pct']
                  for p in validated if p['error_pct'] is not None]
        mape = sum(errors) / len(errors) if errors else 100.0

        # Calculate average confidence
        avg_confidence = sum(p['confidence']
                             for p in validated) / len(validated)

        return {
            "model_name": model_name or "all_models",
            "total_predictions": len(recent),
            "validated": len(validated),
            "correct_predictions": len(correct),
            "accuracy_pct": round((len(correct) / len(validated)) * 100, 2),
            "mape": round(mape, 2),
            "avg_confidence": round(avg_confidence, 1),
            "date_range_days": days,
        }

    def get_all_models_accuracy(self, days: int = 90) -> Dict[str, Any]:
        """
        Get accuracy metrics for all models.

        Args:
            days: Number of days to look back

        Returns:
            Dictionary with system-wide and per-model metrics
        """
        predictions = self._load_predictions()

        # Get unique model names
        models = set(p['model_name'] for p in predictions)

        results = {}
        for model in models:
            results[model] = self.get_model_accuracy(model, days)

        # Calculate system-wide metrics
        system_metrics = self.get_model_accuracy(None, days)

        return {
            "system_accuracy": system_metrics,
            "per_model": results,
            "timestamp": datetime.now().isoformat(),
        }

    def get_prediction_history(
        self, symbol: str, limit: int = 50
    ) -> List[Dict]:
        """
        Get prediction history for a specific symbol.

        Args:
            symbol: Stock symbol
            limit: Maximum number of predictions to return

        Returns:
            List of predictions, most recent first
        """
        predictions = self._load_predictions()

        # Filter by symbol
        symbol_preds = [
            p for p in predictions if p['symbol'] == symbol.upper()]

        # Sort by date, most recent first
        symbol_preds.sort(key=lambda x: x['prediction_date'], reverse=True)

        return symbol_preds[:limit]

    def _load_predictions(self) -> List[Dict]:
        """Load all predictions from JSONL file."""
        if not self.predictions_file.exists():
            return []

        predictions = []
        try:
            with open(self.predictions_file, 'r') as f:
                for line in f:
                    predictions.append(json.loads(line))
        except Exception as e:
            logger.error(f"Failed to load predictions: {e}")

        return predictions

    def _save_predictions(self, predictions: List[Dict]):
        """Save predictions to JSONL file (overwrites)."""
        try:
            with open(self.predictions_file, 'w') as f:
                for pred in predictions:
                    f.write(json.dumps(pred) + '\n')
        except Exception as e:
            logger.error(f"Failed to save predictions: {e}")


# Singleton instance
_tracker = None


def get_accuracy_tracker() -> ModelAccuracyTracker:
    """
    Get singleton accuracy tracker instance.

    Returns:
        ModelAccuracyTracker instance
    """
    global _tracker
    if _tracker is None:
        _tracker = ModelAccuracyTracker()
    return _tracker
