"""
Wavelet Denoising for Stock Price Data

Based on 2024 research, wavelet transform denoising is a critical preprocessing step
that can improve prediction accuracy by 5-15% by removing high-frequency noise
while preserving underlying trends and patterns.

Key concepts:
- Decompose signal into approximation (low-frequency trend) and detail (high-frequency noise)
- Apply thresholding to remove noise from detail coefficients
- Reconstruct clean signal for better model training
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional, List


def _soft_threshold(data: np.ndarray, threshold: float) -> np.ndarray:
    """
    Apply soft thresholding to wavelet coefficients.
    
    Soft thresholding shrinks coefficients toward zero, providing
    smoother denoised signals than hard thresholding.
    
    Formula: sign(x) * max(|x| - threshold, 0)
    """
    return np.sign(data) * np.maximum(np.abs(data) - threshold, 0)


def _hard_threshold(data: np.ndarray, threshold: float) -> np.ndarray:
    """
    Apply hard thresholding to wavelet coefficients.
    
    Hard thresholding sets small coefficients to zero while keeping
    large ones unchanged. Can preserve sharp features but may introduce
    discontinuities.
    """
    return data * (np.abs(data) >= threshold)


def _calculate_universal_threshold(coefficients: np.ndarray, n: int) -> float:
    """
    Calculate universal threshold using Donoho and Johnstone's method.
    
    threshold = sigma * sqrt(2 * log(n))
    
    where sigma is estimated from median absolute deviation (MAD)
    of the finest detail coefficients.
    """
    # Estimate noise standard deviation using MAD
    sigma = np.median(np.abs(coefficients)) / 0.6745
    threshold = sigma * np.sqrt(2 * np.log(n))
    return threshold


def wavelet_decompose(data: np.ndarray, wavelet: str = 'db4', level: int = 4) -> Dict[str, Any]:
    """
    Decompose signal using Discrete Wavelet Transform.
    
    Uses PyWavelets if available, otherwise falls back to simple
    moving average decomposition for basic noise reduction.
    
    Args:
        data: 1D array of prices
        wavelet: Wavelet type ('db4', 'db6', 'sym4', etc.)
        level: Decomposition level (higher = more smoothing)
    
    Returns:
        Dictionary with coefficients and metadata
    """
    try:
        import pywt
        
        # Perform multilevel decomposition
        coeffs = pywt.wavedec(data, wavelet, level=level)
        
        return {
            "coefficients": coeffs,
            "wavelet": wavelet,
            "level": level,
            "method": "pywt",
            "original_length": len(data)
        }
        
    except ImportError:
        # Fallback: Simple moving average decomposition
        # Approximation = MA smoothed, Detail = original - MA
        window = min(2 ** level, len(data) // 4)
        approximation = pd.Series(data).rolling(window, center=True, min_periods=1).mean().values
        detail = data - approximation
        
        return {
            "coefficients": [approximation, detail],
            "wavelet": "ma_fallback",
            "level": 1,
            "method": "numpy_ma",
            "original_length": len(data)
        }


def wavelet_reconstruct(decomposition: Dict[str, Any], denoised_coeffs: List[np.ndarray]) -> np.ndarray:
    """
    Reconstruct signal from wavelet coefficients.
    
    Args:
        decomposition: Output from wavelet_decompose
        denoised_coeffs: Thresholded coefficients
    
    Returns:
        Reconstructed denoised signal
    """
    if decomposition["method"] == "pywt":
        try:
            import pywt
            return pywt.waverec(denoised_coeffs, decomposition["wavelet"])[:decomposition["original_length"]]
        except ImportError:
            pass
    
    # Fallback reconstruction
    if len(denoised_coeffs) >= 2:
        return denoised_coeffs[0] + denoised_coeffs[1]
    return denoised_coeffs[0]


def denoise_stock_data(
    prices: np.ndarray,
    wavelet: str = 'db4',
    level: int = 4,
    threshold_type: str = 'soft',
    threshold_mode: str = 'universal'
) -> Dict[str, Any]:
    """
    Denoise stock price data using wavelet transform.
    
    This is the main function for preprocessing stock data before
    feeding it to prediction models.
    
    Args:
        prices: 1D array of stock prices
        wavelet: Wavelet type ('db4', 'db6', 'sym4', 'sym6')
                 - db4/db6: Daubechies, good for financial data with trends
                 - sym4/sym6: Symlets, better symmetry for some applications
        level: Decomposition level (2-6 recommended)
               Higher = more smoothing, lower = preserve more detail
        threshold_type: 'soft' (smoother) or 'hard' (preserves peaks)
        threshold_mode: 'universal' (conservative) or 'adaptive'
    
    Returns:
        Dictionary with denoised data and analysis
    """
    if len(prices) < 50:
        return {
            "denoised": prices,
            "noise_removed": 0.0,
            "snr_improvement": 0.0,
            "status": "insufficient_data",
            "error": "Need at least 50 data points for effective denoising"
        }
    
    # Decompose
    decomposition = wavelet_decompose(prices, wavelet, level)
    coeffs = decomposition["coefficients"]
    
    # Threshold selection function
    threshold_fn = _soft_threshold if threshold_type == 'soft' else _hard_threshold
    
    # Apply thresholding to detail coefficients (keep approximation unchanged)
    denoised_coeffs = [coeffs[0]]  # Keep approximation
    
    for i, detail in enumerate(coeffs[1:], 1):
        if threshold_mode == 'universal':
            threshold = _calculate_universal_threshold(detail, len(prices))
        else:
            # Adaptive: level-dependent threshold
            threshold = _calculate_universal_threshold(detail, len(prices)) / (2 ** i)
        
        denoised_detail = threshold_fn(detail, threshold)
        denoised_coeffs.append(denoised_detail)
    
    # Reconstruct
    denoised = wavelet_reconstruct(decomposition, denoised_coeffs)
    
    # Calculate metrics
    noise = prices - denoised
    noise_energy = np.sum(noise ** 2)
    signal_energy = np.sum(prices ** 2)
    
    noise_removed_pct = (noise_energy / signal_energy) * 100 if signal_energy > 0 else 0
    
    # Signal-to-noise ratio improvement
    original_snr = 10 * np.log10(signal_energy / (np.var(prices) + 1e-10))
    denoised_var = np.var(denoised) if len(denoised) > 1 else 1e-10
    denoised_snr = 10 * np.log10(np.sum(denoised ** 2) / (denoised_var + 1e-10))
    snr_improvement = denoised_snr - original_snr
    
    return {
        "denoised": denoised,
        "original": prices,
        "noise": noise,
        "noise_removed_pct": round(noise_removed_pct, 2),
        "snr_improvement_db": round(snr_improvement, 2),
        "wavelet": decomposition["wavelet"],
        "level": decomposition["level"],
        "method": decomposition["method"],
        "threshold_type": threshold_type,
        "status": "success"
    }


def get_wavelet_denoised_data(
    stock_data: pd.DataFrame,
    column: str = 'Close',
    **kwargs
) -> Dict[str, Any]:
    """
    Main function to get wavelet-denoised stock data.
    
    Args:
        stock_data: DataFrame with OHLCV data
        column: Column to denoise (default: 'Close')
        **kwargs: Additional arguments for denoise_stock_data
    
    Returns:
        Dictionary with denoised data and analysis
    """
    if column not in stock_data.columns:
        return {
            "error": f"Column '{column}' not found in data",
            "status": "failed"
        }
    
    prices = stock_data[column].values.astype(float)
    result = denoise_stock_data(prices, **kwargs)
    
    # Add denoised as DataFrame column for easy use
    if result["status"] == "success":
        result["denoised_series"] = pd.Series(
            result["denoised"], 
            index=stock_data.index,
            name=f"{column}_denoised"
        )
        
        # Calculate trend clarity improvement
        original_returns = np.diff(prices) / prices[:-1]
        denoised_returns = np.diff(result["denoised"]) / result["denoised"][:-1]
        
        original_volatility = np.std(original_returns)
        denoised_volatility = np.std(denoised_returns)
        
        result["volatility_reduction_pct"] = round(
            (1 - denoised_volatility / original_volatility) * 100, 2
        ) if original_volatility > 0 else 0
        
        result["trend_clarity"] = "high" if result["volatility_reduction_pct"] > 30 else (
            "medium" if result["volatility_reduction_pct"] > 15 else "low"
        )
    
    return result


def apply_denoising_to_features(
    stock_data: pd.DataFrame,
    columns: List[str] = ['Close', 'High', 'Low'],
    **kwargs
) -> pd.DataFrame:
    """
    Apply wavelet denoising to multiple columns for ML preprocessing.
    
    Args:
        stock_data: DataFrame with OHLCV data
        columns: List of columns to denoise
        **kwargs: Additional denoising arguments
    
    Returns:
        DataFrame with original and denoised columns
    """
    result_df = stock_data.copy()
    
    for col in columns:
        if col in stock_data.columns:
            denoised_result = get_wavelet_denoised_data(stock_data, column=col, **kwargs)
            if denoised_result.get("status") == "success":
                result_df[f"{col}_denoised"] = denoised_result["denoised_series"]
    
    return result_df
