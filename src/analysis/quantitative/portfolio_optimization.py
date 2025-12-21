"""
Portfolio Optimization Module
Mean-variance optimization, Black-Litterman, and position sizing.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional


def calculate_portfolio_returns(
    weights: np.ndarray,
    mean_returns: np.ndarray
) -> float:
    """Calculate expected portfolio return."""
    return np.sum(weights * mean_returns)


def calculate_portfolio_volatility(
    weights: np.ndarray,
    cov_matrix: np.ndarray
) -> float:
    """Calculate portfolio volatility (standard deviation)."""
    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))


def mean_variance_optimization(
    returns: pd.DataFrame,
    risk_free_rate: float = 0.02,
    num_portfolios: int = 5000
) -> Dict[str, Any]:
    """
    Perform Mean-Variance Optimization (Markowitz).
    
    Generates random portfolios to find the efficient frontier.
    
    Args:
        returns: DataFrame of asset returns (columns = assets)
        risk_free_rate: Annual risk-free rate
        num_portfolios: Number of random portfolios to generate
    
    Returns:
        Dictionary with optimal portfolios and efficient frontier
    """
    # Calculate statistics
    mean_returns = returns.mean() * 252  # Annualize
    cov_matrix = returns.cov() * 252     # Annualize
    num_assets = len(returns.columns)
    
    # Store results
    results = {
        'returns': [],
        'volatility': [],
        'sharpe': [],
        'weights': []
    }
    
    # Generate random portfolios
    for _ in range(num_portfolios):
        # Random weights
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        
        # Calculate portfolio metrics
        port_return = calculate_portfolio_returns(weights, mean_returns.values)
        port_vol = calculate_portfolio_volatility(weights, cov_matrix.values)
        sharpe = (port_return - risk_free_rate) / port_vol if port_vol > 0 else 0
        
        results['returns'].append(port_return)
        results['volatility'].append(port_vol)
        results['sharpe'].append(sharpe)
        results['weights'].append(weights)
    
    # Find optimal portfolios
    max_sharpe_idx = np.argmax(results['sharpe'])
    min_vol_idx = np.argmin(results['volatility'])
    
    return {
        'max_sharpe_portfolio': {
            'return': round(results['returns'][max_sharpe_idx] * 100, 2),
            'volatility': round(results['volatility'][max_sharpe_idx] * 100, 2),
            'sharpe_ratio': round(results['sharpe'][max_sharpe_idx], 4),
            'weights': dict(zip(returns.columns, np.round(results['weights'][max_sharpe_idx], 4)))
        },
        'min_volatility_portfolio': {
            'return': round(results['returns'][min_vol_idx] * 100, 2),
            'volatility': round(results['volatility'][min_vol_idx] * 100, 2),
            'sharpe_ratio': round(results['sharpe'][min_vol_idx], 4),
            'weights': dict(zip(returns.columns, np.round(results['weights'][min_vol_idx], 4)))
        },
        'efficient_frontier': {
            'returns': [round(r * 100, 2) for r in results['returns']],
            'volatilities': [round(v * 100, 2) for v in results['volatility']]
        },
        'assets': list(returns.columns)
    }


def min_variance_portfolio(
    returns: pd.DataFrame
) -> Dict[str, Any]:
    """
    Calculate the minimum variance portfolio analytically.
    
    Args:
        returns: DataFrame of asset returns
    
    Returns:
        Dictionary with weights and metrics
    """
    cov_matrix = returns.cov() * 252
    num_assets = len(returns.columns)
    
    # Analytical solution: w = Σ^(-1) * 1 / (1' * Σ^(-1) * 1)
    ones = np.ones(num_assets)
    cov_inv = np.linalg.pinv(cov_matrix.values)  # Use pseudo-inverse for stability
    
    numerator = np.dot(cov_inv, ones)
    denominator = np.dot(ones, numerator)
    
    weights = numerator / denominator if denominator != 0 else np.ones(num_assets) / num_assets
    
    # Calculate metrics
    mean_returns = returns.mean() * 252
    port_return = calculate_portfolio_returns(weights, mean_returns.values)
    port_vol = calculate_portfolio_volatility(weights, cov_matrix.values)
    
    return {
        'weights': dict(zip(returns.columns, np.round(weights, 4))),
        'expected_return': round(port_return * 100, 2),
        'volatility': round(port_vol * 100, 2),
        'sharpe_ratio': round((port_return - 0.02) / port_vol, 4) if port_vol > 0 else 0
    }


def risk_parity(
    returns: pd.DataFrame
) -> Dict[str, Any]:
    """
    Calculate Risk Parity portfolio weights.
    
    Equal risk contribution from each asset.
    Simplified approach: inverse volatility weighting.
    
    Args:
        returns: DataFrame of asset returns
    
    Returns:
        Dictionary with weights and metrics
    """
    # Calculate volatilities
    volatilities = returns.std() * np.sqrt(252)
    
    # Inverse volatility weights
    inv_vol = 1 / volatilities
    weights = inv_vol / inv_vol.sum()
    
    # Calculate metrics
    cov_matrix = returns.cov() * 252
    mean_returns = returns.mean() * 252
    
    port_return = calculate_portfolio_returns(weights.values, mean_returns.values)
    port_vol = calculate_portfolio_volatility(weights.values, cov_matrix.values)
    
    # Calculate risk contribution
    marginal_contrib = np.dot(cov_matrix.values, weights.values)
    risk_contrib = weights.values * marginal_contrib / port_vol
    
    return {
        'weights': dict(zip(returns.columns, np.round(weights.values, 4))),
        'expected_return': round(port_return * 100, 2),
        'volatility': round(port_vol * 100, 2),
        'risk_contribution': dict(zip(returns.columns, np.round(risk_contrib * 100, 2)))
    }


def kelly_criterion(
    win_rate: float,
    avg_win: float,
    avg_loss: float
) -> Dict[str, Any]:
    """
    Calculate Kelly Criterion for optimal position sizing.
    
    Formula:
        Kelly % = W - [(1 - W) / R]
        Where: W = Win probability, R = Win/Loss ratio
    
    Args:
        win_rate: Probability of winning trade (0-1)
        avg_win: Average winning trade amount
        avg_loss: Average losing trade amount (positive number)
    
    Returns:
        Dictionary with Kelly percentage and recommendations
    """
    if avg_loss == 0:
        return {'error': 'Average loss cannot be zero'}
    
    # Win/Loss ratio
    win_loss_ratio = avg_win / avg_loss
    
    # Kelly formula
    kelly_pct = win_rate - ((1 - win_rate) / win_loss_ratio)
    
    # Half-Kelly (more conservative)
    half_kelly = kelly_pct / 2
    
    # Quarter-Kelly (very conservative)
    quarter_kelly = kelly_pct / 4
    
    return {
        'kelly_percentage': round(kelly_pct * 100, 2),
        'half_kelly': round(half_kelly * 100, 2),
        'quarter_kelly': round(quarter_kelly * 100, 2),
        'win_rate': round(win_rate * 100, 2),
        'win_loss_ratio': round(win_loss_ratio, 2),
        'recommendation': 'Half-Kelly is recommended for most traders',
        'should_trade': kelly_pct > 0
    }


def black_litterman_returns(
    market_cap_weights: Dict[str, float],
    expected_views: List[Dict[str, Any]],
    cov_matrix: pd.DataFrame,
    risk_aversion: float = 2.5,
    tau: float = 0.05
) -> Dict[str, float]:
    """
    Calculate Black-Litterman expected returns.
    
    Combines market equilibrium with investor views.
    
    Args:
        market_cap_weights: Market capitalization weights
        expected_views: List of views with 'asset', 'view', 'confidence'
        cov_matrix: Covariance matrix of returns
        risk_aversion: Market risk aversion (default 2.5)
        tau: Scaling factor (default 0.05)
    
    Returns:
        Dictionary with Black-Litterman expected returns
    """
    assets = list(market_cap_weights.keys())
    n = len(assets)
    
    # Market cap weights as array
    weights = np.array([market_cap_weights[a] for a in assets])
    sigma = cov_matrix.loc[assets, assets].values
    
    # Equilibrium returns (implied by market)
    pi = risk_aversion * np.dot(sigma, weights)
    
    # If no views, return equilibrium
    if not expected_views:
        return dict(zip(assets, np.round(pi * 100, 2)))
    
    # Process views
    num_views = len(expected_views)
    P = np.zeros((num_views, n))  # View matrix
    Q = np.zeros(num_views)       # View returns
    omega_diag = []               # View uncertainty
    
    for i, view in enumerate(expected_views):
        asset_idx = assets.index(view['asset'])
        P[i, asset_idx] = 1
        Q[i] = view['view'] / 100  # Convert to decimal
        # Uncertainty inversely proportional to confidence
        conf = view.get('confidence', 0.5)
        omega_diag.append(tau * sigma[asset_idx, asset_idx] / conf)
    
    omega = np.diag(omega_diag)
    
    # Black-Litterman formula
    tau_sigma = tau * sigma
    
    # Combined expected returns
    inv_tau_sigma = np.linalg.pinv(tau_sigma)
    inv_omega = np.linalg.pinv(omega)
    
    combined_precision = inv_tau_sigma + np.dot(P.T, np.dot(inv_omega, P))
    combined_precision_inv = np.linalg.pinv(combined_precision)
    
    bl_returns = np.dot(
        combined_precision_inv,
        np.dot(inv_tau_sigma, pi) + np.dot(P.T, np.dot(inv_omega, Q))
    )
    
    return dict(zip(assets, np.round(bl_returns * 100, 2)))


def portfolio_rebalance_signals(
    current_weights: Dict[str, float],
    target_weights: Dict[str, float],
    threshold: float = 0.05
) -> Dict[str, Any]:
    """
    Generate rebalancing signals based on drift from target.
    
    Args:
        current_weights: Current portfolio weights
        target_weights: Target portfolio weights
        threshold: Rebalance if drift exceeds this (default 5%)
    
    Returns:
        Dictionary with rebalancing recommendations
    """
    drifts = {}
    actions = {}
    needs_rebalance = False
    
    for asset in target_weights:
        current = current_weights.get(asset, 0)
        target = target_weights[asset]
        drift = current - target
        drifts[asset] = round(drift * 100, 2)
        
        if abs(drift) > threshold:
            needs_rebalance = True
            if drift > 0:
                actions[asset] = f"SELL {round(drift * 100, 1)}%"
            else:
                actions[asset] = f"BUY {round(abs(drift) * 100, 1)}%"
        else:
            actions[asset] = "HOLD"
    
    return {
        'needs_rebalance': needs_rebalance,
        'drifts': drifts,
        'actions': actions,
        'threshold': f"{threshold * 100}%"
    }
