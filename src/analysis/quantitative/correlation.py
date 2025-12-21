"""
Correlation Analysis Module
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any


def calculate_correlation_matrix(
    price_data: Dict[str, pd.Series]
) -> pd.DataFrame:
    """
    Calculate correlation matrix for multiple assets.
    
    Args:
        price_data: Dictionary of symbol -> price series
    
    Returns:
        DataFrame with correlation matrix
    """
    # Combine into DataFrame
    df = pd.DataFrame(price_data)
    
    # Calculate returns
    returns = df.pct_change().dropna()
    
    # Calculate correlation matrix
    correlation = returns.corr()
    
    return correlation


def rolling_correlation(
    series1: pd.Series,
    series2: pd.Series,
    window: int = 60
) -> pd.Series:
    """
    Calculate rolling correlation between two price series.
    
    Args:
        series1: First price series
        series2: Second price series
        window: Rolling window size
    
    Returns:
        Series with rolling correlation values
    """
    returns1 = series1.pct_change()
    returns2 = series2.pct_change()
    
    return returns1.rolling(window).corr(returns2)


def beta_calculation(
    stock_returns: pd.Series,
    market_returns: pd.Series
) -> Dict[str, float]:
    """
    Calculate beta (systematic risk) of a stock relative to market.
    
    Formula:
        β = Cov(stock, market) / Var(market)
    
    Args:
        stock_returns: Stock return series
        market_returns: Market (e.g., S&P 500) return series
    
    Returns:
        Dictionary with beta and related statistics
    """
    # Align series
    aligned = pd.concat([stock_returns, market_returns], axis=1).dropna()
    stock = aligned.iloc[:, 0]
    market = aligned.iloc[:, 1]
    
    # Calculate beta
    covariance = stock.cov(market)
    market_variance = market.var()
    beta = covariance / market_variance if market_variance != 0 else 1
    
    # Calculate alpha (intercept)
    alpha = stock.mean() - beta * market.mean()
    
    # R-squared
    correlation = stock.corr(market)
    r_squared = correlation ** 2
    
    return {
        'beta': round(beta, 3),
        'alpha': round(alpha * 252 * 100, 2),  # Annualized alpha in %
        'correlation': round(correlation, 3),
        'r_squared': round(r_squared, 3)
    }


def portfolio_correlation_risk(
    weights: Dict[str, float],
    correlation_matrix: pd.DataFrame,
    volatilities: Dict[str, float]
) -> Dict[str, Any]:
    """
    Analyze portfolio risk based on correlations.
    
    Args:
        weights: Dictionary of symbol -> weight
        correlation_matrix: Correlation matrix DataFrame
        volatilities: Dictionary of symbol -> annualized volatility
    
    Returns:
        Portfolio risk metrics
    """
    symbols = list(weights.keys())
    w = np.array([weights[s] for s in symbols])
    vol = np.array([volatilities[s] for s in symbols])
    corr = correlation_matrix.loc[symbols, symbols].values
    
    # Covariance matrix
    D = np.diag(vol)
    cov = D @ corr @ D
    
    # Portfolio variance and volatility
    portfolio_variance = w @ cov @ w
    portfolio_volatility = np.sqrt(portfolio_variance)
    
    # Diversification ratio
    weighted_vol = np.sum(w * vol)
    diversification_ratio = weighted_vol / portfolio_volatility if portfolio_volatility > 0 else 1
    
    return {
        'portfolio_volatility': round(portfolio_volatility * 100, 2),
        'diversification_ratio': round(diversification_ratio, 2),
        'diversification_benefit': round((1 - 1/diversification_ratio) * 100, 2) if diversification_ratio > 0 else 0
    }
