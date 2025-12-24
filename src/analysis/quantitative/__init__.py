# Quantitative Analysis Module
from .monte_carlo import (
    geometric_brownian_motion, monte_carlo_simulation,
    value_at_risk, expected_shortfall
)
from .backtesting import (
    backtest_strategy, calculate_performance_metrics,
    sma_crossover_strategy, rsi_strategy, macd_strategy, Trade
)
from .correlation import (
    calculate_correlation_matrix, rolling_correlation,
    beta_calculation, portfolio_correlation_risk
)
from .risk_metrics import (
    sharpe_ratio, sortino_ratio, calmar_ratio, maximum_drawdown,
    treynor_ratio, information_ratio, beta, comprehensive_risk_analysis
)
from .portfolio_optimization import (
    mean_variance_optimization, min_variance_portfolio, risk_parity,
    kelly_criterion, black_litterman_returns, portfolio_rebalance_signals
)
from .ml_models import (
    pca_analysis, kmeans_clustering, simple_momentum_prediction,
    feature_importance_analysis, regime_detection, train_random_forest, train_svm
)
from .time_series import (
    arima_forecast, check_cointegration, stationarity_test,
    garch_volatility_forecast, gjr_garch_volatility_forecast, get_volatility_forecast
)
from .ml_aggregator import (
    get_ml_ensemble_prediction, get_ml_prediction_summary,
    prepare_ml_features, create_target
)
from .lstm_predictor import (
    get_lstm_prediction, LSTMPredictor
)
from .ensemble_scorer import (
    get_ensemble_prediction, EnsembleScorer, PredictionStore
)
from .anomaly_detection import (
    get_anomaly_alerts, detect_volume_spikes, detect_price_gaps,
    detect_volatility_cluster, detect_momentum_divergence
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
# NEW: Model Accuracy metrics
from .model_accuracy import (
    calculate_mape, calculate_rmse, calculate_directional_accuracy,
    calculate_confidence_interval, calculate_price_confidence_interval,
    get_model_accuracy_summary, generate_price_targets, get_backtest_summary
)
# NEW: Statistical Arbitrage / Pair Trading
from .statistical_arbitrage import (
    test_cointegration, calculate_spread, calculate_zscore,
    generate_pair_signals, find_cointegrated_pairs, get_pair_trading_analysis
)

__all__ = [

    # Monte Carlo
    'geometric_brownian_motion', 'monte_carlo_simulation',
    'value_at_risk', 'expected_shortfall',
    # Backtesting
    'backtest_strategy', 'calculate_performance_metrics',
    'sma_crossover_strategy', 'rsi_strategy', 'macd_strategy', 'Trade',
    # Correlation
    'calculate_correlation_matrix', 'rolling_correlation',
    'beta_calculation', 'portfolio_correlation_risk',
    # Risk Metrics
    'sharpe_ratio', 'sortino_ratio', 'calmar_ratio', 'maximum_drawdown',
    'treynor_ratio', 'information_ratio', 'beta', 'comprehensive_risk_analysis',
    # Portfolio Optimization
    'mean_variance_optimization', 'min_variance_portfolio', 'risk_parity',
    'kelly_criterion', 'black_litterman_returns', 'portfolio_rebalance_signals',
    # ML Models
    'pca_analysis', 'kmeans_clustering', 'simple_momentum_prediction',
    'feature_importance_analysis', 'regime_detection', 'train_random_forest', 'train_svm',
    # ML Aggregator
    'get_ml_ensemble_prediction', 'get_ml_prediction_summary',
    'prepare_ml_features', 'create_target',
    # Time Series / GARCH
    'arima_forecast', 'check_cointegration', 'stationarity_test',
    'garch_volatility_forecast', 'gjr_garch_volatility_forecast', 'get_volatility_forecast',
    # LSTM
    'get_lstm_prediction', 'LSTMPredictor',
    # Ensemble
    'get_ensemble_prediction', 'EnsembleScorer', 'PredictionStore',
    # Anomaly Detection
    'get_anomaly_alerts', 'detect_volume_spikes', 'detect_price_gaps',
    'detect_volatility_cluster', 'detect_momentum_divergence',
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
    # NEW: Model Accuracy
    'calculate_mape', 'calculate_rmse', 'calculate_directional_accuracy',
    'calculate_confidence_interval', 'calculate_price_confidence_interval',
    'get_model_accuracy_summary', 'generate_price_targets', 'get_backtest_summary',
    # NEW: Statistical Arbitrage / Pair Trading
    'test_cointegration', 'calculate_spread', 'calculate_zscore',
    'generate_pair_signals', 'find_cointegrated_pairs', 'get_pair_trading_analysis'
]

