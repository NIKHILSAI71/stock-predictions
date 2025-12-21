"""
Risk Metrics Module
Portfolio and investment risk measurement tools.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Union, Optional


def sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.02,
    periods_per_year: int = 252
) -> float:
    """
    Calculate Sharpe Ratio - risk-adjusted return measure.
    
    Formula:
        Sharpe Ratio = (Portfolio Return - Risk-Free Rate) / Portfolio Std Dev
    
    Interpretation:
        < 1.0: Suboptimal
        1.0 - 2.0: Good
        2.0 - 3.0: Very Good
        > 3.0: Excellent
    
    Args:
        returns: Series of periodic returns
        risk_free_rate: Annual risk-free rate (default 2%)
        periods_per_year: Trading periods per year (252 for daily)
    
    Returns:
        Annualized Sharpe Ratio
    """
    if len(returns) < 2:
        return 0.0
    
    # Annualize returns and volatility
    mean_return = returns.mean() * periods_per_year
    std_dev = returns.std() * np.sqrt(periods_per_year)
    
    if std_dev < 1e-9:
        return 0.0
    
    sharpe = (mean_return - risk_free_rate) / std_dev
    return round(sharpe, 4)


def sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.02,
    periods_per_year: int = 252
) -> float:
    """
    Calculate Sortino Ratio - downside risk-adjusted return.
    
    Formula:
        Sortino Ratio = (Portfolio Return - Risk-Free Rate) / Downside Deviation
    
    Unlike Sharpe, only penalizes downside volatility.
    
    Args:
        returns: Series of periodic returns
        risk_free_rate: Annual risk-free rate (default 2%)
        periods_per_year: Trading periods per year
    
    Returns:
        Annualized Sortino Ratio
    """
    if len(returns) < 2:
        return 0.0
    
    # Calculate downside returns (negative only)
    downside_returns = returns[returns < 0]
    
    if len(downside_returns) == 0:
        return float('inf')  # No downside
    
    # Downside deviation
    downside_std = downside_returns.std() * np.sqrt(periods_per_year)
    
    if downside_std < 1e-9:
        return 0.0
    
    # Annualized return
    mean_return = returns.mean() * periods_per_year
    
    sortino = (mean_return - risk_free_rate) / downside_std
    return round(sortino, 4)


def calmar_ratio(
    returns: pd.Series,
    periods_per_year: int = 252
) -> float:
    """
    Calculate Calmar Ratio - return vs maximum drawdown.
    
    Formula:
        Calmar Ratio = Annualized Return / Maximum Drawdown
    
    Args:
        returns: Series of periodic returns
        periods_per_year: Trading periods per year
    
    Returns:
        Calmar Ratio
    """
    if len(returns) < 2:
        return 0.0
    
    # Annualized return
    total_return = (1 + returns).prod()
    years = len(returns) / periods_per_year
    annualized_return = total_return ** (1 / years) - 1 if years > 0 else 0
    
    # Maximum drawdown
    max_dd = maximum_drawdown(returns)['max_drawdown']
    
    if max_dd == 0:
        return float('inf')
    
    calmar = annualized_return / abs(max_dd)
    return round(calmar, 4)


def maximum_drawdown(
    returns: pd.Series
) -> Dict[str, Any]:
    """
    Calculate Maximum Drawdown and related metrics.
    
    Formula:
        Drawdown = (Peak - Trough) / Peak
        Max Drawdown = Maximum of all drawdowns
    
    Args:
        returns: Series of periodic returns
    
    Returns:
        Dictionary with max drawdown, duration, and recovery info
    """
    # Convert returns to cumulative wealth
    cumulative = (1 + returns).cumprod()
    
    # Running maximum
    running_max = cumulative.cummax()
    
    # Drawdown series
    drawdown = (cumulative - running_max) / running_max
    
    # Maximum drawdown
    max_dd = drawdown.min()
    max_dd_idx = drawdown.idxmin()
    
    # Find the peak before max drawdown
    peak_idx = cumulative[:max_dd_idx].idxmax()
    
    # Find recovery (if any)
    recovery_idx = None
    if max_dd_idx is not None:
        post_dd = cumulative[max_dd_idx:]
        recovered = post_dd[post_dd >= running_max[peak_idx]]
        if len(recovered) > 0:
            recovery_idx = recovered.index[0]
    
    # Calculate duration
    if isinstance(peak_idx, pd.Timestamp) and isinstance(max_dd_idx, pd.Timestamp):
        drawdown_duration = (max_dd_idx - peak_idx).days
    else:
        drawdown_duration = None
    
    return {
        'max_drawdown': round(max_dd, 4),
        'max_drawdown_pct': round(max_dd * 100, 2),
        'peak_date': peak_idx,
        'trough_date': max_dd_idx,
        'recovery_date': recovery_idx,
        'drawdown_duration_days': drawdown_duration,
        'current_drawdown': round(drawdown.iloc[-1], 4)
    }


def treynor_ratio(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    risk_free_rate: float = 0.02,
    periods_per_year: int = 252
) -> float:
    """
    Calculate Treynor Ratio - beta-adjusted excess return.
    
    Formula:
        Treynor Ratio = (Portfolio Return - Risk-Free Rate) / Beta
    
    Args:
        returns: Portfolio returns
        benchmark_returns: Benchmark returns (e.g., S&P 500)
        risk_free_rate: Annual risk-free rate
        periods_per_year: Trading periods per year
    
    Returns:
        Treynor Ratio
    """
    if len(returns) < 2 or len(benchmark_returns) < 2:
        return 0.0
    
    # Align the series
    aligned = pd.concat([returns, benchmark_returns], axis=1).dropna()
    if len(aligned) < 2:
        return 0.0
    
    port_returns = aligned.iloc[:, 0]
    bench_returns = aligned.iloc[:, 1]
    
    # Calculate beta
    covariance = port_returns.cov(bench_returns)
    variance = bench_returns.var()
    beta = covariance / variance if variance != 0 else 1.0
    
    if beta == 0:
        return 0.0
    
    # Annualized excess return
    mean_return = port_returns.mean() * periods_per_year
    excess_return = mean_return - risk_free_rate
    
    treynor = excess_return / beta
    return round(treynor, 4)


def information_ratio(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    periods_per_year: int = 252
) -> float:
    """
    Calculate Information Ratio - active return per unit of active risk.
    
    Formula:
        IR = (Portfolio Return - Benchmark Return) / Tracking Error
    
    Interpretation:
        > 0.5: Good
        > 1.0: Excellent
    
    Args:
        returns: Portfolio returns
        benchmark_returns: Benchmark returns
        periods_per_year: Trading periods per year
    
    Returns:
        Information Ratio
    """
    if len(returns) < 2 or len(benchmark_returns) < 2:
        return 0.0
    
    # Align series
    aligned = pd.concat([returns, benchmark_returns], axis=1).dropna()
    if len(aligned) < 2:
        return 0.0
    
    port_returns = aligned.iloc[:, 0]
    bench_returns = aligned.iloc[:, 1]
    
    # Active return (tracking difference)
    active_return = port_returns - bench_returns
    
    # Annualize
    mean_active = active_return.mean() * periods_per_year
    tracking_error = active_return.std() * np.sqrt(periods_per_year)
    
    if tracking_error == 0:
        return 0.0
    
    ir = mean_active / tracking_error
    return round(ir, 4)


def beta(
    returns: pd.Series,
    benchmark_returns: pd.Series
) -> Dict[str, float]:
    """
    Calculate Beta and Alpha.
    
    Beta measures systematic risk relative to market.
    Alpha measures excess return vs expected return based on beta.
    
    Args:
        returns: Portfolio returns
        benchmark_returns: Benchmark returns
    
    Returns:
        Dictionary with beta and alpha
    """
    aligned = pd.concat([returns, benchmark_returns], axis=1).dropna()
    if len(aligned) < 2:
        return {'beta': 1.0, 'alpha': 0.0}
    
    port_returns = aligned.iloc[:, 0]
    bench_returns = aligned.iloc[:, 1]
    
    # Calculate beta
    covariance = port_returns.cov(bench_returns)
    variance = bench_returns.var()
    
    beta_val = covariance / variance if variance != 0 else 1.0
    
    # Calculate alpha (Jensen's Alpha)
    alpha_val = port_returns.mean() - (beta_val * bench_returns.mean())
    
    return {
        'beta': round(beta_val, 4),
        'alpha': round(alpha_val * 252, 4)  # Annualized
    }


def comprehensive_risk_analysis(
    returns: pd.Series,
    benchmark_returns: pd.Series = None,
    risk_free_rate: float = 0.02
) -> Dict[str, Any]:
    """
    Comprehensive risk analysis report.
    
    Args:
        returns: Portfolio returns
        benchmark_returns: Optional benchmark returns
        risk_free_rate: Annual risk-free rate
    
    Returns:
        Dictionary with all risk metrics
    """
    # Pre-calculate common stats
    periods = 252
    mean_ret = returns.mean()
    std_dev = returns.std()
    
    # Annualized stats
    ann_ret = mean_ret * periods
    ann_vol = std_dev * np.sqrt(periods)
    
    # Sharpe (Manual calc to reuse stats)
    sharpe = (ann_ret - risk_free_rate) / ann_vol if ann_vol > 1e-9 else 0.0
    
    # Sortino
    downside_returns = returns[returns < 0]
    downside_std = downside_returns.std() * np.sqrt(periods) if not downside_returns.empty else 0.0
    sortino = (ann_ret - risk_free_rate) / downside_std if downside_std > 1e-9 else (float('inf') if downside_returns.empty else 0.0)
    
    # Calmar & Drawdown
    dd_stats = maximum_drawdown(returns)
    max_dd = dd_stats['max_drawdown']
    calmar = ann_ret / abs(max_dd) if max_dd != 0 else float('inf')
    
    result = {
        'sharpe_ratio': round(sharpe, 4),
        'sortino_ratio': round(sortino, 4),
        'calmar_ratio': round(calmar, 4),
        'max_drawdown': dd_stats,
        'volatility': round(ann_vol * 100, 2),
        'annualized_return': round(ann_ret * 100, 2),
        'positive_periods': round((returns > 0).sum() / len(returns) * 100, 2),
        'skewness': round(returns.skew(), 4),
        'kurtosis': round(returns.kurtosis(), 4)
    }
    
    if benchmark_returns is not None:
        result['treynor_ratio'] = treynor_ratio(returns, benchmark_returns, risk_free_rate)
        result['information_ratio'] = information_ratio(returns, benchmark_returns)
        result['beta_alpha'] = beta(returns, benchmark_returns)
    
    # Risk rating
    if sharpe >= 2.0:
        result['risk_rating'] = 'Excellent'
    elif sharpe >= 1.0:
        result['risk_rating'] = 'Good'
    elif sharpe >= 0.5:
        result['risk_rating'] = 'Moderate'
    else:
        result['risk_rating'] = 'Poor'
    
    return result
