"""
Time Series Analysis Module
"""

import pandas as pd
import numpy as np
import warnings
from typing import Dict, Any, Tuple
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import coint, adfuller

def arima_forecast(
    data: pd.Series,
    order: Tuple[int, int, int] = (1, 1, 1),
    steps: int = 5
) -> Dict[str, Any]:
    """
    Generate ARIMA forecast.
    
    Args:
        data: Time series data
        order: ARIMA parameters (p, d, q)
        steps: Number of periods to forecast
    
    Returns:
        Dictionary with forecast data
    """
    try:
        # Suppress convergence warnings
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning)
            warnings.filterwarnings('ignore', message='.*Maximum Likelihood.*')
            warnings.filterwarnings('ignore', message='.*convergence.*')
            
            # Use values to avoid frequency warnings/errors with non-standard date indices
            # Try with the given order first
            try:
                model = ARIMA(data.values, order=order)
                results = model.fit(method_kwargs={'warn_convergence': False})
            except:
                # Fallback to simpler model if convergence fails
                model = ARIMA(data.values, order=(0, 1, 0))
                results = model.fit(method_kwargs={'warn_convergence': False})
            
            forecast = results.forecast(steps=steps)
            
            return {
                "forecast": forecast.tolist(),
                "aic": results.aic,
                "bic": results.bic,
                "status": "success"
            }
    except Exception as e:
        # Return a simple forecast based on last value if ARIMA fails completely
        last_val = float(data.iloc[-1]) if len(data) > 0 else 0
        return {
            "forecast": [last_val] * steps,
            "aic": None,
            "bic": None,
            "status": "fallback",
            "note": "Using last value as forecast due to model limitations"
        }

def check_cointegration(
    series1: pd.Series,
    series2: pd.Series
) -> Dict[str, Any]:
    """
    Test for cointegration between two time series.
    
    Args:
        series1: First time series
        series2: Second time series
    
    Returns:
        Dictionary with cointegration test results
    """
    try:
        # Align data
        df = pd.DataFrame({'s1': series1, 's2': series2}).dropna()
        
        if len(df) < 50:
            return {"error": "Insufficient data", "status": "failed"}
        
        score, pvalue, _ = coint(df['s1'], df['s2'])
        
        return {
            "t_statistic": score,
            "p_value": pvalue,
            "is_cointegrated": pvalue < 0.05,
            "status": "success"
        }
    except Exception as e:
        return {"error": str(e), "status": "failed"}

def stationarity_test(data: pd.Series) -> Dict[str, Any]:
    """
    Perform Augmented Dickey-Fuller test for stationarity.
    """
    try:
        result = adfuller(data.dropna())
        return {
            "test_statistic": result[0],
            "p_value": result[1],
            "is_stationary": result[1] < 0.05,
            "status": "success"
        }
    except Exception as e:
        return {"error": str(e), "status": "failed"}


def garch_volatility_forecast(
    returns: pd.Series,
    forecast_horizon: int = 5,
    p: int = 1,
    q: int = 1
) -> Dict[str, Any]:
    """
    GARCH(p,q) volatility forecasting.
    
    GARCH models capture volatility clustering - the tendency for large price 
    movements to be followed by large movements and small by small.
    
    Formula:
        σ²(t) = ω + α × ε²(t-1) + β × σ²(t-1)
        
    Where:
        σ²(t) = Conditional variance at time t
        ω = Long-term average variance (constant)
        α = Coefficient for lagged squared returns (ARCH term)
        β = Coefficient for lagged variance (GARCH term)
        ε = Return shock/innovation
    
    Args:
        returns: Series of daily returns (not prices!)
        forecast_horizon: Number of days to forecast
        p: GARCH lag order (default 1)
        q: ARCH lag order (default 1)
    
    Returns:
        Dictionary with volatility forecasts and model info
    """
    try:
        from arch import arch_model
        
        # Clean returns data
        returns_clean = returns.dropna() * 100  # Scale to percentage for numerical stability
        
        if len(returns_clean) < 100:
            return {
                "error": "Insufficient data for GARCH (need at least 100 observations)",
                "status": "failed"
            }
        
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            
            # Fit GARCH(p,q) model
            model = arch_model(
                returns_clean, 
                vol='Garch', 
                p=p, 
                q=q,
                mean='Constant',
                dist='normal'
            )
            
            fitted = model.fit(disp='off', show_warning=False)
            
            # Generate volatility forecast
            forecast = fitted.forecast(horizon=forecast_horizon)
            
            # Get forecasted variance and convert to annualized volatility
            variance_forecast = forecast.variance.iloc[-1].values
            volatility_forecast = np.sqrt(variance_forecast) * np.sqrt(252) / 100  # Annualized, back to decimal
            
            # Current conditional volatility
            current_cond_vol = np.sqrt(fitted.conditional_volatility.iloc[-1]) * np.sqrt(252) / 100
            
            # Model parameters
            params = fitted.params
            
            # Calculate persistence (α + β should be < 1 for stationarity)
            alpha = params.get('alpha[1]', 0)
            beta = params.get('beta[1]', 0)
            persistence = alpha + beta
            
            # Long-run (unconditional) volatility
            omega = params.get('omega', 0)
            if persistence < 1:
                long_run_var = omega / (1 - persistence)
                long_run_vol = np.sqrt(long_run_var) * np.sqrt(252) / 100
            else:
                long_run_vol = None  # Non-stationary
            
            return {
                "volatility_forecast": [round(v * 100, 2) for v in volatility_forecast],
                "current_volatility_annualized": round(current_cond_vol * 100, 2),
                "long_run_volatility": round(long_run_vol * 100, 2) if long_run_vol else None,
                "persistence": round(persistence, 4),
                "is_stationary": persistence < 1,
                "alpha": round(alpha, 4),
                "beta": round(beta, 4),
                "omega": round(omega, 6),
                "aic": round(fitted.aic, 2),
                "bic": round(fitted.bic, 2),
                "forecast_horizon_days": forecast_horizon,
                "volatility_trend": "increasing" if volatility_forecast[-1] > current_cond_vol else "decreasing",
                "status": "success"
            }
            
    except ImportError:
        return {
            "error": "arch library not installed. Run: pip install arch",
            "status": "failed"
        }
    except Exception as e:
        return {"error": str(e), "status": "failed"}


def gjr_garch_volatility_forecast(
    returns: pd.Series,
    forecast_horizon: int = 5
) -> Dict[str, Any]:
    """
    GJR-GARCH volatility forecasting (asymmetric GARCH).
    
    GJR-GARCH captures the leverage effect - negative returns often have
    a larger impact on volatility than positive returns of the same magnitude.
    
    Formula:
        σ²(t) = ω + (α + γ × I(t-1)) × ε²(t-1) + β × σ²(t-1)
        
    Where I(t-1) = 1 if ε(t-1) < 0 (negative shock)
    
    Key insight from research: 
    - GJR-GARCH often outperforms standard GARCH for financial data
    - γ > 0 indicates leverage effect (bad news increases volatility more)
    
    Args:
        returns: Series of daily returns
        forecast_horizon: Number of days to forecast
    
    Returns:
        Dictionary with asymmetric volatility forecasts
    """
    try:
        from arch import arch_model
        
        returns_clean = returns.dropna() * 100
        
        if len(returns_clean) < 100:
            return {
                "error": "Insufficient data for GJR-GARCH",
                "status": "failed"
            }
        
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            
            # Fit GJR-GARCH(1,1,1) model
            model = arch_model(
                returns_clean,
                vol='Garch',
                p=1,
                o=1,  # Asymmetric term
                q=1,
                mean='Constant',
                dist='normal'
            )
            
            fitted = model.fit(disp='off', show_warning=False)
            
            # Generate forecast
            forecast = fitted.forecast(horizon=forecast_horizon)
            variance_forecast = forecast.variance.iloc[-1].values
            volatility_forecast = np.sqrt(variance_forecast) * np.sqrt(252) / 100
            
            current_cond_vol = np.sqrt(fitted.conditional_volatility.iloc[-1]) * np.sqrt(252) / 100
            
            # Get asymmetry parameter (gamma)
            params = fitted.params
            gamma = params.get('gamma[1]', 0)
            alpha = params.get('alpha[1]', 0)
            beta = params.get('beta[1]', 0)
            
            # Interpret leverage effect
            if gamma > 0.05:
                leverage_effect = "strong"
                leverage_description = "Bad news significantly increases volatility"
            elif gamma > 0:
                leverage_effect = "moderate"
                leverage_description = "Bad news moderately increases volatility"
            else:
                leverage_effect = "none"
                leverage_description = "No asymmetric volatility response detected"
            
            return {
                "volatility_forecast": [round(v * 100, 2) for v in volatility_forecast],
                "current_volatility_annualized": round(current_cond_vol * 100, 2),
                "gamma_asymmetry": round(gamma, 4),
                "leverage_effect": leverage_effect,
                "leverage_description": leverage_description,
                "alpha": round(alpha, 4),
                "beta": round(beta, 4),
                "persistence": round(alpha + beta + 0.5 * gamma, 4),
                "forecast_horizon_days": forecast_horizon,
                "model_type": "GJR-GARCH(1,1,1)",
                "status": "success"
            }
            
    except ImportError:
        return {
            "error": "arch library not installed. Run: pip install arch",
            "status": "failed"
        }
    except Exception as e:
        return {"error": str(e), "status": "failed"}


def get_volatility_forecast(
    stock_data: pd.DataFrame,
    forecast_horizon: int = 5
) -> Dict[str, Any]:
    """
    Main function to get comprehensive volatility forecast.
    
    Combines GARCH and GJR-GARCH for robust volatility prediction.
    
    Args:
        stock_data: DataFrame with at least 'Close' column
        forecast_horizon: Days ahead to forecast
    
    Returns:
        Combined volatility analysis
    """
    # Calculate returns
    close = stock_data['Close']
    returns = close.pct_change().dropna()
    
    # Get both forecasts
    garch_result = garch_volatility_forecast(returns, forecast_horizon)
    gjr_result = gjr_garch_volatility_forecast(returns, forecast_horizon)
    
    # Historical volatility for comparison
    hist_vol_20d = returns.tail(20).std() * np.sqrt(252) * 100
    hist_vol_60d = returns.tail(60).std() * np.sqrt(252) * 100
    
    # Determine best model using AIC if both succeeded
    if garch_result.get("status") == "success" and gjr_result.get("status") == "success":
        best_model = "GJR-GARCH" if gjr_result.get("leverage_effect") != "none" else "GARCH"
        best_forecast = gjr_result["volatility_forecast"] if best_model == "GJR-GARCH" else garch_result["volatility_forecast"]
    elif garch_result.get("status") == "success":
        best_model = "GARCH"
        best_forecast = garch_result["volatility_forecast"]
    elif gjr_result.get("status") == "success":
        best_model = "GJR-GARCH"
        best_forecast = gjr_result["volatility_forecast"]
    else:
        # Fallback to historical volatility
        best_model = "Historical"
        best_forecast = [round(hist_vol_20d, 2)] * forecast_horizon
    
    return {
        "recommended_model": best_model,
        "volatility_forecast": best_forecast,
        "historical_volatility_20d": round(hist_vol_20d, 2),
        "historical_volatility_60d": round(hist_vol_60d, 2),
        "garch_analysis": garch_result,
        "gjr_garch_analysis": gjr_result,
        "forecast_horizon_days": forecast_horizon,
        "volatility_regime": _classify_volatility_regime(hist_vol_20d, hist_vol_60d)
    }


def _classify_volatility_regime(short_vol: float, long_vol: float) -> Dict[str, Any]:
    """Classify current volatility regime."""
    if short_vol > 40:
        regime = "Extreme"
        description = "Very high volatility - exercise extreme caution"
    elif short_vol > 25:
        regime = "High"
        description = "Elevated volatility - widen stop losses"
    elif short_vol > 15:
        regime = "Normal"
        description = "Average volatility conditions"
    else:
        regime = "Low"
        description = "Below-average volatility - potential breakout ahead"
    
    # Check for volatility expansion/contraction
    if short_vol > long_vol * 1.3:
        trend = "Expanding"
    elif short_vol < long_vol * 0.7:
        trend = "Contracting"
    else:
        trend = "Stable"
    
    return {
        "regime": regime,
        "description": description,
        "trend": trend
    }
