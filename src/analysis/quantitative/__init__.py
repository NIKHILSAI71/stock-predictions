# Quantitative Analysis Module
from .monte_carlo import (
    geometric_brownian_motion, monte_carlo_simulation,
    value_at_risk, expected_shortfall
)
from .ml_models import (
    pca_analysis, kmeans_clustering, simple_momentum_prediction,
    feature_importance_analysis, regime_detection, train_random_forest, train_svm
)
from .lstm_predictor import (
    get_lstm_prediction, LSTMPredictor
)
from .ensemble_scorer import (
    get_ensemble_prediction, EnsembleScorer, PredictionStore
)
# NEW: XGBoost model
from .xgboost_model import (
    get_xgboost_prediction, train_xgboost, prepare_xgboost_features
)
# NEW: GRU predictor
from .gru_predictor import (
    get_gru_prediction, GRUPredictor, compare_lstm_gru
)
# NEW: Wavelet Denoising preprocessing
from .wavelet_denoising import (
    get_wavelet_denoised_data, denoise_stock_data, apply_denoising_to_features
)
# NEW: CNN-LSTM Hybrid model
from .cnn_lstm_hybrid import (
    get_cnn_lstm_prediction, CNNLSTMPredictor
)
# NEW: Attention-based predictor
from .attention_predictor import (
    get_attention_prediction, AttentionPredictor
)
# NEW: Statistical Arbitrage / Pair Trading
from .statistical_arbitrage import (
    test_cointegration, calculate_spread, calculate_zscore,
    generate_pair_signals, find_cointegrated_pairs, get_pair_trading_analysis
)
# NEW: Metrics and Volatility (Restored)
from .metrics import (
    calculate_mape, calculate_rmse, get_volatility_forecast
)
# NEW: Random Forest Predictor
from .random_forest_predictor import (
    get_rf_prediction, prepare_rf_features
)
# NEW: SVM Predictor
from .svm_predictor import (
    get_svm_prediction, prepare_svm_features
)
# NEW: Enhanced Momentum Predictor
from .momentum_predictor import (
    get_momentum_prediction, calculate_rsi, detect_market_regime
)
# NEW: LightGBM Predictor
from .lightgbm_predictor import (
    get_lightgbm_prediction, prepare_lightgbm_features
)
# NEW: TCN Predictor
from .tcn_predictor import (
    get_tcn_prediction, build_tensorflow_tcn
)
# NEW: N-BEATS Predictor
from .nbeats_predictor import (
    get_nbeats_prediction, build_tensorflow_nbeats
)
# NEW: Technical Signals
from .technical_signal_predictor import (
    get_technical_signals
)
# NEW: Fundamental Analysis
from .fundamental_predictor import (
    get_fundamental_prediction
)

__all__ = [

    # Monte Carlo
    'geometric_brownian_motion', 'monte_carlo_simulation',
    'value_at_risk', 'expected_shortfall',
    # ML Models
    'pca_analysis', 'kmeans_clustering', 'simple_momentum_prediction',
    'feature_importance_analysis', 'regime_detection', 'train_random_forest', 'train_svm',
    # LSTM
    'get_lstm_prediction', 'LSTMPredictor',
    # Ensemble
    'get_ensemble_prediction', 'EnsembleScorer', 'PredictionStore',
    # NEW: XGBoost
    'get_xgboost_prediction', 'train_xgboost', 'prepare_xgboost_features',
    # NEW: GRU
    'get_gru_prediction', 'GRUPredictor', 'compare_lstm_gru',
    # NEW: Wavelet Denoising
    'get_wavelet_denoised_data', 'denoise_stock_data', 'apply_denoising_to_features',
    # NEW: CNN-LSTM Hybrid
    'get_cnn_lstm_prediction', 'CNNLSTMPredictor',
    # NEW: Attention Predictor
    'get_attention_prediction', 'AttentionPredictor',
    # NEW: Statistical Arbitrage / Pair Trading
    'test_cointegration', 'calculate_spread', 'calculate_zscore',
    'generate_pair_signals', 'find_cointegrated_pairs', 'get_pair_trading_analysis',
    # Metrics
    'calculate_mape', 'calculate_rmse', 'get_volatility_forecast',
    # NEW: Random Forest Predictor
    'get_rf_prediction', 'prepare_rf_features',
    # NEW: SVM Predictor
    'get_svm_prediction', 'prepare_svm_features',
    # NEW: Enhanced Momentum Predictor
    'get_momentum_prediction', 'calculate_rsi', 'detect_market_regime',
    # NEW: LightGBM Predictor
    'get_lightgbm_prediction', 'prepare_lightgbm_features',
    # NEW: TCN Predictor
   'get_tcn_prediction', 'build_tensorflow_tcn',
    # NEW: N-BEATS Predictor
    'get_nbeats_prediction', 'build_tensorflow_nbeats',
    # NEW: Technical Signals
    'get_technical_signals',
    # NEW: Fundamental Analysis
    'get_fundamental_prediction'
]
