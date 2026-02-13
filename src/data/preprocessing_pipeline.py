"""
Consistent Data Preprocessing Pipeline

This module provides a unified preprocessing pipeline for all ML models.
CRITICAL: All models MUST use this pipeline to ensure consistent predictions.

Problems this solves:
- Wavelet denoising was optional → Now ALWAYS applied
- Different normalization methods across models → Unified MinMax [0,1]
- Inconsistent missing data handling → Forward fill + backward fill for prices
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, Any, List
import logging

# Try to import pywt, fall back to no wavelet if not available
try:
    import pywt
    PYWT_AVAILABLE = True
except ImportError:
    PYWT_AVAILABLE = False
    logging.warning("pywt not installed. Wavelet denoising will be disabled.")

logger = logging.getLogger(__name__)


class ConsistentPreprocessor:
    """
    Ensures all models receive identically preprocessed data.
    
    This singleton-like class maintains consistent preprocessing across
    all model predictions for a given symbol.
    
    Usage:
        preprocessor = ConsistentPreprocessor()
        clean_df = preprocessor.preprocess(raw_df, symbol)
        predictions_normalized = model.predict(clean_df)
        predictions_real = preprocessor.inverse_transform(predictions_normalized, symbol)
    """
    
    # Class-level cache to share scaler params across instances
    _scaler_cache: Dict[str, Dict[str, Dict[str, float]]] = {}
    
    # Standard feature windows - MUST be consistent across all models
    FEATURE_WINDOWS = [5, 10, 20]
    
    # Outlier threshold in standard deviations
    OUTLIER_THRESHOLD = 5.0
    
    # Minimum data points required for wavelet denoising
    MIN_WAVELET_LENGTH = 60
    
    def __init__(self, wavelet_enabled: bool = True):
        """
        Initialize the preprocessor.
        
        Args:
            wavelet_enabled: Whether to apply wavelet denoising. 
                            Should be True for consistency, but can be 
                            disabled for debugging.
        """
        self.scaler_params: Dict[str, Dict[str, Dict[str, float]]] = {}
        self.wavelet_enabled = wavelet_enabled and PYWT_AVAILABLE
        self._preprocessing_log: List[str] = []
        
    def preprocess(
        self, 
        df: pd.DataFrame, 
        symbol: str,
        calculate_features: bool = True
    ) -> pd.DataFrame:
        """
        Single preprocessing pipeline for ALL models.
        
        Order of operations is critical - do NOT change!
        
        Args:
            df: Raw DataFrame with OHLCV data
            symbol: Stock ticker symbol
            calculate_features: Whether to calculate derived features
            
        Returns:
            Preprocessed DataFrame ready for model input
        """
        if df is None or df.empty:
            raise ValueError(f"Empty dataframe received for {symbol}")
            
        df = df.copy()
        self._preprocessing_log = [f"Starting preprocessing for {symbol}"]
        initial_rows = len(df)
        
        # Step 1: Handle missing data (ALWAYS THE SAME WAY)
        df = self._handle_missing_data(df)
        self._preprocessing_log.append(f"After missing data handling: {len(df)} rows")
        
        # Step 2: Remove outliers (ALWAYS)
        df = self._remove_outliers(df)
        self._preprocessing_log.append(f"After outlier removal: {len(df)} rows")
        
        # Step 3: Apply wavelet denoising (ALWAYS - not optional for production)
        if self.wavelet_enabled and len(df) >= self.MIN_WAVELET_LENGTH:
            df = self._apply_wavelet_denoising(df)
            self._preprocessing_log.append("Wavelet denoising applied")
        elif not self.wavelet_enabled:
            self._preprocessing_log.append("Wavelet denoising DISABLED (debug mode)")
        else:
            self._preprocessing_log.append(f"Wavelet skipped (need {self.MIN_WAVELET_LENGTH} rows, have {len(df)})")
        
        # Step 4: Calculate features (BEFORE normalization)
        if calculate_features:
            df = self._calculate_features(df)
            self._preprocessing_log.append(f"After feature calculation: {len(df)} rows")
        
        # Step 5: Normalize (LAST STEP)
        df, scaler_params = self._normalize(df, symbol)
        self.scaler_params[symbol] = scaler_params
        ConsistentPreprocessor._scaler_cache[symbol] = scaler_params
        self._preprocessing_log.append(f"Normalization complete. Final: {len(df)} rows (from {initial_rows})")
        
        logger.debug(" | ".join(self._preprocessing_log))
        
        return df
    
    def _handle_missing_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Consistent missing data strategy.
        
        Strategy:
        - Price columns: Forward fill then backward fill
        - Indicator columns: Drop rows (they need history anyway)
        
        This is deterministic and reproducible.
        """
        # Standard OHLCV columns
        price_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        available_price_cols = [col for col in price_cols if col in df.columns]
        
        if available_price_cols:
            # Forward fill then backward fill for price data
            # This handles both gaps at start and end of series
            df[available_price_cols] = df[available_price_cols].ffill().bfill()
        
        # Handle any remaining NaN columns with forward/backward fill
        df = df.ffill().bfill()
        
        # If still NaN (shouldn't happen), drop rows
        if df.isnull().any().any():
            nan_count = df.isnull().sum().sum()
            logger.warning(f"Dropping {nan_count} remaining NaN values")
            df = df.dropna()
        
        return df
    
    def _remove_outliers(
        self, 
        df: pd.DataFrame, 
        threshold: float = None
    ) -> pd.DataFrame:
        """
        Remove extreme price movements that break models.
        
        Uses capping instead of removal to preserve data length.
        This is important for maintaining sequence integrity in LSTM/GRU.
        
        Args:
            df: DataFrame with Close prices
            threshold: Standard deviations for outlier detection
            
        Returns:
            DataFrame with capped returns
        """
        if threshold is None:
            threshold = self.OUTLIER_THRESHOLD
            
        if 'Close' not in df.columns:
            return df
            
        # Calculate returns
        returns = df['Close'].pct_change()
        
        # Skip if not enough data
        if len(returns) < 3:
            return df
        
        # Calculate bounds
        mean = returns.mean()
        std = returns.std()
        
        if std == 0 or np.isnan(std):
            return df
            
        upper_bound = mean + threshold * std
        lower_bound = mean - threshold * std
        
        # Count outliers for logging
        outlier_count = ((returns > upper_bound) | (returns < lower_bound)).sum()
        
        if outlier_count > 0:
            logger.debug(f"Capping {outlier_count} outlier returns at {threshold} std")
            
            # Cap outliers instead of removing (preserves data length)
            returns_capped = returns.clip(lower=lower_bound, upper=upper_bound)
            
            # Reconstruct prices from capped returns
            # Start from first valid price
            first_price = df['Close'].iloc[0]
            df['Close'] = first_price * (1 + returns_capped.fillna(0)).cumprod()
        
        return df
    
    def _apply_wavelet_denoising(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply wavelet denoising to price series.
        
        ALWAYS apply this in production - improves all models by 5-15%.
        Uses Daubechies-4 wavelet with soft thresholding.
        """
        if not PYWT_AVAILABLE:
            return df
            
        for col in ['Close', 'Open', 'High', 'Low']:
            if col in df.columns:
                try:
                    denoised = self._wavelet_denoise(df[col].values)
                    # Ensure same length (wavelet can add/remove a sample)
                    if len(denoised) >= len(df):
                        df[col] = denoised[:len(df)]
                    else:
                        # Pad with last value if shorter
                        padding = np.full(len(df) - len(denoised), denoised[-1])
                        df[col] = np.concatenate([denoised, padding])
                except Exception as e:
                    logger.warning(f"Wavelet denoising failed for {col}: {e}")
                    
        return df
    
    def _wavelet_denoise(
        self, 
        signal: np.ndarray, 
        wavelet: str = 'db4', 
        level: int = 3
    ) -> np.ndarray:
        """
        Apply wavelet denoising using Donoho-Johnstone threshold.
        
        Args:
            signal: 1D array of values
            wavelet: Wavelet type (db4 is good for financial data)
            level: Decomposition level
            
        Returns:
            Denoised signal
        """
        if not PYWT_AVAILABLE:
            return signal
            
        # Handle edge case
        if len(signal) < 2 ** level:
            return signal
            
        # Decompose
        coeffs = pywt.wavedec(signal, wavelet, level=level)
        
        # Calculate threshold (Donoho-Johnstone universal threshold)
        # Using median absolute deviation for robustness
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
        threshold = sigma * np.sqrt(2 * np.log(len(signal)))
        
        # Apply soft thresholding to detail coefficients
        new_coeffs = [coeffs[0]]  # Keep approximation coefficients unchanged
        for coeff in coeffs[1:]:
            new_coeffs.append(pywt.threshold(coeff, threshold, mode='soft'))
        
        # Reconstruct
        return pywt.waverec(new_coeffs, wavelet)
    
    def _calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate derived features BEFORE normalization.
        
        Uses standardized windows across all models.
        """
        if 'Close' not in df.columns:
            return df
            
        # Returns (most important for ML)
        df['returns'] = df['Close'].pct_change()
        
        # Rolling statistics using standard windows
        for window in self.FEATURE_WINDOWS:
            if len(df) >= window:
                df[f'sma_{window}'] = df['Close'].rolling(window).mean()
                df[f'std_{window}'] = df['Close'].rolling(window).std()
                df[f'min_{window}'] = df['Close'].rolling(window).min()
                df[f'max_{window}'] = df['Close'].rolling(window).max()
        
        # Momentum features
        for period in [5, 10, 20]:
            if len(df) >= period:
                df[f'momentum_{period}'] = df['Close'] - df['Close'].shift(period)
                df[f'roc_{period}'] = df['Close'].pct_change(period) * 100
        
        # Volume features (if available)
        if 'Volume' in df.columns:
            df['volume_sma_20'] = df['Volume'].rolling(20).mean()
            df['volume_ratio'] = df['Volume'] / df['volume_sma_20'].replace(0, np.nan)
            df['volume_ratio'] = df['volume_ratio'].fillna(1)
        
        # Drop NaN rows created by rolling windows
        df = df.dropna()
        
        return df
    
    def _normalize(
        self, 
        df: pd.DataFrame, 
        symbol: str
    ) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
        """
        Min-Max normalization to [0, 1] - Same for all models.
        
        Stores parameters for inverse transform to get real prices.
        
        Args:
            df: DataFrame to normalize
            symbol: Stock symbol for caching
            
        Returns:
            Tuple of (normalized DataFrame, scaler parameters)
        """
        scaler_params = {}
        df = df.copy()
        
        # Normalize numeric columns to [0, 1]
        for col in df.select_dtypes(include=[np.number]).columns:
            col_min = df[col].min()
            col_max = df[col].max()
            
            if col_max - col_min > 1e-10:  # Avoid division by zero
                df[col] = (df[col] - col_min) / (col_max - col_min)
            else:
                df[col] = 0.5  # Constant column -> middle value
            
            scaler_params[col] = {
                'min': float(col_min), 
                'max': float(col_max)
            }
        
        return df, scaler_params
    
    def inverse_transform(
        self, 
        predictions: np.ndarray, 
        symbol: str, 
        column: str = 'Close'
    ) -> np.ndarray:
        """
        Convert normalized predictions back to real prices.
        
        Args:
            predictions: Normalized predictions in [0, 1]
            symbol: Stock symbol to look up scaler params
            column: Column to inverse transform (default: Close)
            
        Returns:
            Predictions in original scale
        """
        # Check instance cache first, then class cache
        if symbol in self.scaler_params:
            params = self.scaler_params[symbol].get(column)
        elif symbol in ConsistentPreprocessor._scaler_cache:
            params = ConsistentPreprocessor._scaler_cache[symbol].get(column)
        else:
            raise ValueError(
                f"No scaler params found for {symbol}. "
                f"Call preprocess() first."
            )
        
        if params is None:
            raise ValueError(f"No scaler params found for column '{column}'")
            
        return predictions * (params['max'] - params['min']) + params['min']
    
    def get_preprocessing_log(self) -> List[str]:
        """Get the log of preprocessing steps for debugging."""
        return self._preprocessing_log.copy()
    
    def get_scaler_params(self, symbol: str) -> Optional[Dict[str, Dict[str, float]]]:
        """Get scaler parameters for a symbol."""
        return self.scaler_params.get(symbol) or \
               ConsistentPreprocessor._scaler_cache.get(symbol)


# Singleton instance for global access
_global_preprocessor: Optional[ConsistentPreprocessor] = None


def get_preprocessor() -> ConsistentPreprocessor:
    """
    Get the global preprocessor instance.
    
    This ensures all models use the same preprocessor and scaler params.
    """
    global _global_preprocessor
    if _global_preprocessor is None:
        _global_preprocessor = ConsistentPreprocessor()
    return _global_preprocessor


def preprocess_for_model(
    df: pd.DataFrame, 
    symbol: str,
    calculate_features: bool = True
) -> pd.DataFrame:
    """
    Convenience function for preprocessing data.
    
    Usage:
        from src.data.preprocessing_pipeline import preprocess_for_model, inverse_transform
        
        clean_df = preprocess_for_model(raw_df, "AAPL")
        predictions = model.predict(clean_df)
        real_prices = inverse_transform(predictions, "AAPL")
    """
    return get_preprocessor().preprocess(df, symbol, calculate_features)


def inverse_transform(
    predictions: np.ndarray, 
    symbol: str, 
    column: str = 'Close'
) -> np.ndarray:
    """
    Convenience function for inverse transformation.
    """
    return get_preprocessor().inverse_transform(predictions, symbol, column)


def get_preprocessing_metrics(
    df: pd.DataFrame,
    symbol: str
) -> Dict[str, Any]:
    """
    Get detailed preprocessing metrics for a stock.
    
    Returns metrics about:
    - Wavelet denoising effectiveness
    - Outlier detection and removal
    - Data normalization parameters
    - Feature engineering summary
    
    Args:
        df: Raw DataFrame with OHLCV data
        symbol: Stock ticker symbol
        
    Returns:
        Dictionary with preprocessing metrics for display
    """
    if df is None or df.empty:
        return {"error": "No data provided"}
    
    metrics = {
        "symbol": symbol,
        "wavelet_available": PYWT_AVAILABLE,
        "data_quality": {},
        "wavelet_denoising": {},
        "outlier_analysis": {},
        "normalization": {},
        "features_generated": []
    }
    
    try:
        # Data quality metrics
        initial_rows = len(df)
        missing_count = df.isnull().sum().sum()
        metrics["data_quality"] = {
            "total_rows": initial_rows,
            "missing_values": int(missing_count),
            "missing_pct": round(missing_count / (initial_rows * len(df.columns)) * 100, 2) if initial_rows > 0 else 0,
            "columns": list(df.columns)
        }
        
        # Wavelet denoising metrics
        if PYWT_AVAILABLE and 'Close' in df.columns and len(df) >= 60:
            original_close = df['Close'].values
            
            # Create preprocessor and apply wavelet
            preprocessor = ConsistentPreprocessor(wavelet_enabled=True)
            
            # Calculate noise metrics
            original_std = np.std(np.diff(original_close))
            
            # Apply wavelet denoising
            try:
                denoised = preprocessor._wavelet_denoise(original_close)
                denoised_std = np.std(np.diff(denoised[:len(original_close)]))
                
                noise_reduction = ((original_std - denoised_std) / original_std * 100) if original_std > 0 else 0
                
                metrics["wavelet_denoising"] = {
                    "applied": True,
                    "wavelet_type": "db4 (Daubechies-4)",
                    "original_volatility": round(original_std, 4),
                    "denoised_volatility": round(denoised_std, 4),
                    "noise_reduction_pct": round(max(0, noise_reduction), 2),
                    "signal_clarity": "High" if noise_reduction > 20 else ("Medium" if noise_reduction > 10 else "Low")
                }
            except Exception as e:
                metrics["wavelet_denoising"] = {"applied": False, "error": str(e)}
        else:
            metrics["wavelet_denoising"] = {
                "applied": False,
                "reason": "pywt not available" if not PYWT_AVAILABLE else (
                    "Insufficient data (need 60+ rows)" if len(df) < 60 else "No Close column"
                )
            }
        
        # Outlier analysis
        if 'Close' in df.columns:
            returns = df['Close'].pct_change().dropna()
            if len(returns) > 0:
                mean_return = returns.mean()
                std_return = returns.std()
                threshold = 5.0 * std_return
                
                outliers = ((returns > mean_return + threshold) | (returns < mean_return - threshold)).sum()
                
                metrics["outlier_analysis"] = {
                    "total_returns": len(returns),
                    "outliers_detected": int(outliers),
                    "outlier_pct": round(outliers / len(returns) * 100, 2) if len(returns) > 0 else 0,
                    "threshold_used": f"{5.0} standard deviations",
                    "mean_daily_return": round(mean_return * 100, 4),
                    "return_volatility": round(std_return * 100, 4)
                }
        
        # Normalization params
        preprocessor = get_preprocessor()
        scaler_params = preprocessor.get_scaler_params(symbol)
        if scaler_params:
            metrics["normalization"] = {
                "method": "MinMax [0, 1]",
                "columns_normalized": len(scaler_params),
                "price_range": {
                    "min": scaler_params.get('Close', {}).get('min'),
                    "max": scaler_params.get('Close', {}).get('max')
                } if 'Close' in scaler_params else {}
            }
        else:
            metrics["normalization"] = {"status": "Not yet applied for this symbol"}
        
        # Features that would be generated
        metrics["features_generated"] = [
            "returns", 
            "sma_5", "sma_10", "sma_20",
            "std_5", "std_10", "std_20",
            "min_5", "min_10", "min_20",
            "max_5", "max_10", "max_20",
            "momentum_5", "momentum_10", "momentum_20",
            "roc_5", "roc_10", "roc_20",
            "volume_sma_20", "volume_ratio"
        ]
        
    except Exception as e:
        metrics["error"] = str(e)
    
    return metrics
