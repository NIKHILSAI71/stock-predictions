"""
Monte Carlo Simulation Module
Price simulation using Geometric Brownian Motion
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional


def geometric_brownian_motion(
    S0: float,
    mu: float,
    sigma: float,
    T: float,
    dt: float,
    num_simulations: int = 1000
) -> np.ndarray:
    """
    Simulate stock prices using Geometric Brownian Motion.
    
    Formula:
        dS = μSdt + σSdW
        S(t+dt) = S(t) * exp((μ - σ²/2)dt + σ√dt * Z)
    
    Where:
        S = Stock price
        μ = Expected return (drift)
        σ = Volatility
        W = Wiener process
        Z = Standard normal random variable
    
    Args:
        S0: Initial stock price
        mu: Expected annual return (decimal)
        sigma: Annual volatility (decimal)
        T: Time horizon in years
        dt: Time step in years (e.g., 1/252 for daily)
        num_simulations: Number of simulation paths
    
    Returns:
        Array of simulated prices (num_simulations x num_steps)
    """
    num_steps = int(T / dt)
    
    # Generate random normal samples
    Z = np.random.standard_normal((num_simulations, num_steps))
    
    # Calculate drift and diffusion components
    drift = (mu - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt) * Z
    
    # Calculate price paths
    log_returns = drift + diffusion
    log_returns = np.insert(log_returns, 0, 0, axis=1)  # Add starting point
    
    prices = S0 * np.exp(np.cumsum(log_returns, axis=1))
    
    return prices


def monte_carlo_simulation(
    current_price: float,
    historical_returns: pd.Series,
    days_forward: int = 252,
    num_simulations: int = 10000,
    confidence_levels: List[float] = [0.05, 0.25, 0.5, 0.75, 0.95]
) -> Dict[str, Any]:
    """
    Run Monte Carlo simulation for stock price forecasting.
    
    Args:
        current_price: Current stock price
        historical_returns: Series of historical daily returns
        days_forward: Number of trading days to simulate
        num_simulations: Number of simulation paths
        confidence_levels: Quantiles for confidence intervals
    
    Returns:
        Dictionary with simulation results
    """
    # Calculate parameters from historical data
    mu = historical_returns.mean() * 252  # Annualized mean return
    sigma = historical_returns.std() * np.sqrt(252)  # Annualized volatility
    
    # Run simulation
    dt = 1 / 252  # Daily steps
    T = days_forward / 252  # Time horizon in years
    
    simulated_prices = geometric_brownian_motion(
        current_price, mu, sigma, T, dt, num_simulations
    )
    
    # Final prices
    final_prices = simulated_prices[:, -1]
    
    # Calculate statistics
    mean_final = np.mean(final_prices)
    std_final = np.std(final_prices)
    
    # Confidence intervals
    percentiles = {}
    for level in confidence_levels:
        percentiles[f'p{int(level*100)}'] = np.percentile(final_prices, level * 100)
    
    # Probability calculations
    prob_above_current = (final_prices > current_price).sum() / num_simulations * 100
    prob_up_10pct = (final_prices > current_price * 1.1).sum() / num_simulations * 100
    prob_down_10pct = (final_prices < current_price * 0.9).sum() / num_simulations * 100
    
    # Expected return
    expected_return = (mean_final - current_price) / current_price * 100
    
    return {
        'current_price': round(current_price, 2),
        'days_forward': days_forward,
        'num_simulations': num_simulations,
        'annualized_return': round(mu * 100, 2),
        'annualized_volatility': round(sigma * 100, 2),
        'mean_final_price': round(mean_final, 2),
        'std_final_price': round(std_final, 2),
        'expected_return_pct': round(expected_return, 2),
        'percentiles': {k: round(v, 2) for k, v in percentiles.items()},
        'prob_above_current': round(prob_above_current, 2),
        'prob_up_10pct': round(prob_up_10pct, 2),
        'prob_down_10pct': round(prob_down_10pct, 2),
        'median_price': round(np.median(final_prices), 2),
        'min_price': round(np.min(final_prices), 2),
        'max_price': round(np.max(final_prices), 2)
    }


def value_at_risk(
    portfolio_value: float,
    returns: pd.Series,
    confidence_level: float = 0.95,
    time_horizon: int = 1,
    method: str = 'historical'
) -> Dict[str, Any]:
    """
    Calculate Value at Risk (VaR).
    
    Args:
        portfolio_value: Current portfolio value
        returns: Historical returns series
        confidence_level: Confidence level (e.g., 0.95 for 95%)
        time_horizon: Time horizon in days
        method: 'historical', 'parametric', or 'monte_carlo'
    
    Returns:
        Dictionary with VaR results
    """
    if method == 'historical':
        # Sort returns and find percentile
        sorted_returns = returns.sort_values()
        percentile_idx = int((1 - confidence_level) * len(sorted_returns))
        var_return = sorted_returns.iloc[percentile_idx]
        
    elif method == 'parametric':
        # Assume normal distribution
        from scipy.stats import norm
        z_score = norm.ppf(1 - confidence_level)
        var_return = returns.mean() + z_score * returns.std()
        
    else:  # monte_carlo
        num_sims = 10000
        simulated_returns = np.random.choice(returns, size=(num_sims, time_horizon))
        portfolio_returns = simulated_returns.sum(axis=1)
        var_return = np.percentile(portfolio_returns, (1 - confidence_level) * 100)
    
    # Scale for time horizon
    var_return = var_return * np.sqrt(time_horizon)
    var_value = portfolio_value * abs(var_return)
    
    return {
        'var_percentage': round(abs(var_return) * 100, 2),
        'var_value': round(var_value, 2),
        'confidence_level': confidence_level * 100,
        'time_horizon_days': time_horizon,
        'method': method
    }


def expected_shortfall(
    returns: pd.Series,
    confidence_level: float = 0.95
) -> float:
    """
    Calculate Expected Shortfall (Conditional VaR).
    
    The average loss in the worst (1-confidence_level)% of cases.
    
    Args:
        returns: Historical returns series
        confidence_level: Confidence level
    
    Returns:
        Expected shortfall as a decimal
    """
    sorted_returns = returns.sort_values()
    cutoff_idx = int((1 - confidence_level) * len(sorted_returns))
    
    # Average of returns below VaR threshold
    es = sorted_returns[:cutoff_idx].mean()
    
    return abs(es)
