"""
Model Accuracy Tests
Unit tests for model accuracy tracking.
"""

import pytest
from src.analysis.quantitative.model_accuracy import (
    ModelAccuracyTracker,
    Prediction
)
from pathlib import Path
import tempfile
import shutil
import json
from datetime import datetime, timedelta


@pytest.fixture
def temp_tracker():
    """Create temporary tracker for testing."""
    temp_dir = tempfile.mkdtemp()
    tracker = ModelAccuracyTracker(storage_path=temp_dir)
    yield tracker
    # Cleanup
    shutil.rmtree(temp_dir)


@pytest.mark.unit
def test_record_prediction(temp_tracker):
    """Test recording a prediction."""
    pred_date = temp_tracker.record_prediction(
        symbol="AAPL",
        model_name="LSTM",
        predicted_direction="Bullish",
        predicted_price=150.0,
        confidence=75.5,
        horizon_days=7
    )

    assert pred_date is not None

    # Verify prediction was saved
    predictions = temp_tracker._load_predictions()
    assert len(predictions) == 1
    assert predictions[0]['symbol'] == 'AAPL'
    assert predictions[0]['model_name'] == 'LSTM'
    assert predictions[0]['predicted_direction'] == 'Bullish'
    assert predictions[0]['predicted_price'] == 150.0
    assert predictions[0]['confidence'] == 75.5


@pytest.mark.unit
def test_validate_predictions(temp_tracker):
    """Test prediction validation."""
    # Record prediction in the past (for immediate validation)
    temp_tracker.record_prediction(
        symbol="AAPL",
        model_name="LSTM",
        predicted_direction="Bullish",
        predicted_price=150.0,
        confidence=75.0,
        horizon_days=-1  # In the past
    )

    # Validate with actual price
    temp_tracker.validate_predictions({"AAPL": 155.0})

    predictions = temp_tracker._load_predictions()
    assert predictions[0]['validated'] == True
    assert predictions[0]['actual_price'] == 155.0
    assert predictions[0]['correct'] == True  # Bullish and price went up
    assert predictions[0]['error_pct'] is not None


@pytest.mark.unit
def test_validate_bearish_prediction(temp_tracker):
    """Test bearish prediction validation."""
    temp_tracker.record_prediction(
        symbol="AAPL",
        model_name="XGBoost",
        predicted_direction="Bearish",
        predicted_price=150.0,
        confidence=70.0,
        horizon_days=-1
    )

    # Price went down, bearish was correct
    temp_tracker.validate_predictions({"AAPL": 145.0})

    predictions = temp_tracker._load_predictions()
    assert predictions[0]['correct'] == True


@pytest.mark.unit
def test_get_model_accuracy(temp_tracker):
    """Test accuracy calculation."""
    # Record multiple predictions
    for i in range(10):
        temp_tracker.record_prediction(
            symbol="AAPL",
            model_name="LSTM",
            predicted_direction="Bullish" if i % 2 == 0 else "Bearish",
            predicted_price=150.0 + i,
            confidence=70.0,
            horizon_days=-1
        )

    # Validate all (simulate 70% accuracy - bullish predictions correct)
    temp_tracker.validate_predictions({"AAPL": 155.0})

    accuracy = temp_tracker.get_model_accuracy("LSTM", days=90)
    assert accuracy['total_predictions'] == 10
    assert accuracy['validated'] == 10
    assert accuracy['model_name'] == "LSTM"


@pytest.mark.unit
def test_get_prediction_history(temp_tracker):
    """Test retrieving prediction history."""
    # Record predictions for different symbols
    temp_tracker.record_prediction(
        symbol="AAPL",
        model_name="LSTM",
        predicted_direction="Bullish",
        predicted_price=150.0,
        confidence=75.0,
        horizon_days=7
    )

    temp_tracker.record_prediction(
        symbol="GOOGL",
        model_name="LSTM",
        predicted_direction="Bearish",
        predicted_price=2800.0,
        confidence=80.0,
        horizon_days=7
    )

    # Get history for AAPL
    aapl_history = temp_tracker.get_prediction_history("AAPL", limit=10)
    assert len(aapl_history) == 1
    assert aapl_history[0]['symbol'] == 'AAPL'

    # Get history for GOOGL
    googl_history = temp_tracker.get_prediction_history("GOOGL", limit=10)
    assert len(googl_history) == 1
    assert googl_history[0]['symbol'] == 'GOOGL'


@pytest.mark.unit
def test_empty_accuracy(temp_tracker):
    """Test accuracy calculation with no predictions."""
    accuracy = temp_tracker.get_model_accuracy("NonExistent", days=90)
    assert accuracy['total_predictions'] == 0
    assert accuracy['validated'] == 0
    assert accuracy['accuracy_pct'] == 0.0


@pytest.mark.unit
def test_all_models_accuracy(temp_tracker):
    """Test getting accuracy for all models."""
    # Record predictions from multiple models
    temp_tracker.record_prediction(
        symbol="AAPL",
        model_name="LSTM",
        predicted_direction="Bullish",
        predicted_price=150.0,
        confidence=75.0,
        horizon_days=-1
    )

    temp_tracker.record_prediction(
        symbol="AAPL",
        model_name="XGBoost",
        predicted_direction="Bearish",
        predicted_price=150.0,
        confidence=80.0,
        horizon_days=-1
    )

    # Validate
    temp_tracker.validate_predictions({"AAPL": 155.0})

    all_accuracy = temp_tracker.get_all_models_accuracy(days=90)
    assert "system_accuracy" in all_accuracy
    assert "per_model" in all_accuracy
    assert "LSTM" in all_accuracy["per_model"]
    assert "XGBoost" in all_accuracy["per_model"]
