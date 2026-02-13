"""
Shared Preprocessing Utilities for Time Series Models

Common functions used by LSTM, GRU, and other time series prediction models.
Consolidates duplicate code to improve maintainability.
"""

import numpy as np
from typing import Tuple


def prepare_sequences(
    data: np.ndarray,
    lookback: int = 60,
    forecast_horizon: int = 5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare sequences for time series training (LSTM, GRU, etc.).

    Args:
        data: 1D array of prices
        lookback: Number of past days to use
        forecast_horizon: Number of days to predict

    Returns:
        X: (samples, lookback, 1) sequences
        y: (samples, forecast_horizon) targets
    """
    X, y = [], []
    for i in range(lookback, len(data) - forecast_horizon):
        X.append(data[i - lookback:i])
        y.append(data[i:i + forecast_horizon])

    X = np.array(X).reshape(-1, lookback, 1)
    y = np.array(y)
    return X, y


def normalize_data(data: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """
    Normalize data to [0, 1] range using min-max normalization.

    Args:
        data: Input array to normalize

    Returns:
        normalized: Normalized data in [0, 1]
        min_val: Original minimum value
        max_val: Original maximum value
    """
    min_val = np.min(data)
    max_val = np.max(data)

    # Handle edge case where all values are the same
    if max_val - min_val == 0:
        return data, min_val, max_val

    normalized = (data - min_val) / (max_val - min_val)
    return normalized, min_val, max_val


def denormalize_data(data: np.ndarray, min_val: float, max_val: float) -> np.ndarray:
    """
    Denormalize data back to original scale.

    Args:
        data: Normalized data in [0, 1]
        min_val: Original minimum value
        max_val: Original maximum value

    Returns:
        Denormalized data in original scale
    """
    return data * (max_val - min_val) + min_val


def sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Sigmoid activation function with clipping to prevent overflow.

    Args:
        x: Input array

    Returns:
        Sigmoid-transformed array
    """
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def tanh(x: np.ndarray) -> np.ndarray:
    """
    Tanh activation function with clipping to prevent overflow.

    Args:
        x: Input array

    Returns:
        Tanh-transformed array
    """
    return np.tanh(np.clip(x, -500, 500))
