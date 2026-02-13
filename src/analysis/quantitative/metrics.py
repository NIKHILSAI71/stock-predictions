
"""
Quantitative Analysis Metrics
"""
import numpy as np
import pandas as pd
from typing import Union, List, Dict, Any

def calculate_mape(y_true: Union[List, np.ndarray], y_pred: Union[List, np.ndarray]) -> float:
    """
    Calculate Mean Absolute Percentage Error.
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # Avoid division by zero
    mask = y_true != 0
    if not np.any(mask):
        return 0.0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def calculate_rmse(y_true: Union[List, np.ndarray], y_pred: Union[List, np.ndarray]) -> float:
    """
    Calculate Root Mean Squared Error.
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.sqrt(((y_pred - y_true) ** 2).mean())

def get_volatility_forecast(data: Union[pd.DataFrame, pd.Series, List], forecast_horizon: int = 5) -> Dict[str, Any]:
    """
    Generate a simple volatility forecast based on historical data.
    """
    try:
        if isinstance(data, pd.DataFrame):
            if 'Close' in data.columns:
                series = data['Close']
            else:
                series = data.iloc[:, 0]
        elif isinstance(data, list):
            series = pd.Series(data)
        else:
            series = data
            
        returns = series.pct_change().dropna()
        if len(returns) < 2:
            return {"forecast": 0.0, "trend": "Unknown", "error": "Insufficient data"}
            
        # Annualized volatility
        current_vol = returns.tail(20).std() * np.sqrt(252)
        
        # Simple mean reversion forecast (very basic)
        long_term_vol = returns.std() * np.sqrt(252)
        
        forecasts = []
        for i in range(forecast_horizon):
            # Slowly revert to mean
            vol = current_vol * (0.9 ** i) + long_term_vol * (1 - 0.9 ** i)
            forecasts.append(vol)
            
        trend = "Stable"
        if forecasts[-1] > current_vol * 1.05:
            trend = "Rising"
        elif forecasts[-1] < current_vol * 0.95:
            trend = "Falling"
            
        return {
            "forecast": forecasts,  # List of values
            "trend": trend,
            "current_volatility": current_vol,
            "long_term_volatility": long_term_vol,
            # For compatibility with legacy code expecting single value or different format
            "volatility_forecast": forecasts,
            "recommended_model": "GARCH (Simulated)",
            "historical_volatility_20d": current_vol
        }
    except Exception as e:
        return {"forecast": [], "error": str(e), "trend": "Error"}
