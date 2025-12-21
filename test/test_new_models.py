"""
Test new prediction models: XGBoost, GRU, GARCH, CNN-LSTM, Attention, Wavelet
"""
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.analysis.quantitative.xgboost_model import get_xgboost_prediction
from src.analysis.quantitative.gru_predictor import get_gru_prediction
from src.analysis.quantitative.time_series import get_volatility_forecast
from src.analysis.quantitative.cnn_lstm_hybrid import get_cnn_lstm_prediction
from src.analysis.quantitative.attention_predictor import get_attention_prediction
from src.analysis.quantitative.wavelet_denoising import get_wavelet_denoised_data, denoise_stock_data

def create_test_data(n_days=300):
    """Create synthetic stock data for testing."""
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=n_days, freq='D')
    prices = 100 + np.cumsum(np.random.randn(n_days) * 2)
    volume = np.random.randint(1000000, 5000000, n_days)
    return pd.DataFrame({
        'Open': prices - 0.5,
        'High': prices + 1,
        'Low': prices - 1,
        'Close': prices,
        'Volume': volume
    }, index=dates)

def test_xgboost():
    """Test XGBoost prediction model."""
    print("Testing XGBoost...")
    test_data = create_test_data()
    result = get_xgboost_prediction(test_data, prediction_horizon=5)
    
    if result.get("status") == "success":
        print(f"  Direction: {result.get('direction')}")
        print(f"  Confidence: {result.get('confidence')}%")
        print(f"  Accuracy (test): {result.get('accuracy_test')}")
        print("  PASSED")
        return True
    else:
        print(f"  Error: {result.get('error')}")
        return False

def test_gru():
    """Test GRU prediction model."""
    print("Testing GRU...")
    test_data = create_test_data()
    result = get_gru_prediction(test_data, train_epochs=2)
    
    if "error" not in result or result.get("direction"):
        print(f"  Direction: {result.get('direction')}")
        print(f"  Confidence: {result.get('confidence')}%")
        print(f"  Model type: {result.get('model_type')}")
        print("  PASSED")
        return True
    else:
        print(f"  Error: {result.get('error')}")
        return False

def test_garch():
    """Test GARCH volatility forecasting."""
    print("Testing GARCH...")
    test_data = create_test_data()
    result = get_volatility_forecast(test_data, forecast_horizon=5)
    
    if result.get("recommended_model"):
        print(f"  Recommended model: {result.get('recommended_model')}")
        print(f"  Historical volatility (20d): {result.get('historical_volatility_20d')}%")
        regime = result.get("volatility_regime", {})
        print(f"  Volatility regime: {regime.get('regime')}")
        print("  PASSED")
        return True
    else:
        print("  Failed to get volatility forecast")
        return False

def test_cnn_lstm():
    """Test CNN-LSTM Hybrid model."""
    print("Testing CNN-LSTM Hybrid...")
    test_data = create_test_data()
    result = get_cnn_lstm_prediction(test_data, train_epochs=2)
    
    if result.get("status") == "success":
        print(f"  Direction: {result.get('direction')}")
        print(f"  Confidence: {result.get('confidence')}%")
        print(f"  Model type: {result.get('model_type')}")
        print("  PASSED")
        return True
    else:
        print(f"  Error: {result.get('error')}")
        return False

def test_attention():
    """Test Attention-based Transformer model."""
    print("Testing Attention Transformer...")
    test_data = create_test_data()
    result = get_attention_prediction(test_data, train_epochs=2)
    
    if result.get("status") == "success":
        print(f"  Direction: {result.get('direction')}")
        print(f"  Confidence: {result.get('confidence')}%")
        print(f"  Model type: {result.get('model_type')}")
        attention_focus = result.get("attention_focus", {})
        print(f"  Attention focus: {attention_focus.get('interpretation', 'N/A')}")
        print("  PASSED")
        return True
    else:
        print(f"  Error: {result.get('error')}")
        return False

def test_wavelet():
    """Test Wavelet Denoising preprocessing."""
    print("Testing Wavelet Denoising...")
    test_data = create_test_data()
    result = get_wavelet_denoised_data(test_data, column='Close')
    
    if result.get("status") == "success":
        print(f"  Noise removed: {result.get('noise_removed_pct')}%")
        print(f"  Trend clarity: {result.get('trend_clarity')}")
        print(f"  Volatility reduction: {result.get('volatility_reduction_pct')}%")
        print(f"  Method: {result.get('method')}")
        print("  PASSED")
        return True
    else:
        print(f"  Error: {result.get('error')}")
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("Testing All Prediction Models")
    print("=" * 50)
    
    results = []
    results.append(("XGBoost", test_xgboost()))
    print()
    results.append(("GRU", test_gru()))
    print()
    results.append(("GARCH", test_garch()))
    print()
    results.append(("CNN-LSTM", test_cnn_lstm()))
    print()
    results.append(("Attention", test_attention()))
    print()
    results.append(("Wavelet", test_wavelet()))
    print()
    
    print("=" * 50)
    print("Summary:")
    for name, passed in results:
        status = "PASSED" if passed else "FAILED"
        print(f"  {name}: {status}")
    
    all_passed = all(r[1] for r in results)
    print("=" * 50)
    print("All tests passed!" if all_passed else "Some tests failed!")
