"""
Dividend Discount Model (DDM) - Gordon Growth Model
"""

from typing import Dict, Any, List


def gordon_growth_model(
    current_dividend: float,
    growth_rate: float,
    required_return: float
) -> Dict[str, Any]:
    """
    Calculate intrinsic value using Gordon Growth Model (Constant Growth DDM).
    
    Formula:
        P = D₁ / (r - g)
        D₁ = D₀ × (1 + g)
    
    Where:
        P = Intrinsic value (fair price)
        D₁ = Expected dividend next year
        D₀ = Current dividend
        r = Required rate of return (cost of equity)
        g = Constant dividend growth rate
    
    Args:
        current_dividend: Current annual dividend per share (D₀)
        growth_rate: Constant annual dividend growth rate (decimal, e.g., 0.05 for 5%)
        required_return: Required rate of return (decimal, e.g., 0.10 for 10%)
    
    Returns:
        Dictionary with valuation results
    """
    if required_return <= growth_rate:
        return {
            'fair_value': None,
            'error': 'Required return must be greater than growth rate',
            'current_dividend': current_dividend,
            'growth_rate': growth_rate * 100,
            'required_return': required_return * 100
        }
    
    if current_dividend <= 0:
        return {
            'fair_value': None,
            'error': 'Company must pay dividends for DDM',
            'current_dividend': current_dividend
        }
    
    # Calculate D₁ (next year's dividend)
    next_dividend = current_dividend * (1 + growth_rate)
    
    # Calculate fair value
    fair_value = next_dividend / (required_return - growth_rate)
    
    # Calculate implied dividend yield
    implied_yield = (next_dividend / fair_value) * 100
    
    return {
        'fair_value': round(fair_value, 2),
        'current_dividend': current_dividend,
        'next_dividend': round(next_dividend, 2),
        'growth_rate': round(growth_rate * 100, 2),
        'required_return': round(required_return * 100, 2),
        'implied_dividend_yield': round(implied_yield, 2)
    }


def two_stage_ddm(
    current_dividend: float,
    high_growth_rate: float,
    high_growth_years: int,
    stable_growth_rate: float,
    required_return: float
) -> Dict[str, Any]:
    """
    Calculate intrinsic value using Two-Stage Dividend Discount Model.
    
    Stage 1: High growth period with varying dividends
    Stage 2: Stable (perpetual) growth period
    
    Args:
        current_dividend: Current annual dividend per share
        high_growth_rate: Growth rate during high growth phase
        high_growth_years: Number of years in high growth phase
        stable_growth_rate: Perpetual growth rate in stable phase
        required_return: Required rate of return
    
    Returns:
        Dictionary with valuation results
    """
    if required_return <= stable_growth_rate:
        return {
            'fair_value': None,
            'error': 'Required return must be greater than stable growth rate'
        }
    
    if current_dividend <= 0:
        return {
            'fair_value': None,
            'error': 'Company must pay dividends'
        }
    
    # Stage 1: Calculate present value of dividends during high growth
    stage1_dividends = []
    stage1_pv = []
    dividend = current_dividend
    
    for year in range(1, high_growth_years + 1):
        dividend = dividend * (1 + high_growth_rate)
        stage1_dividends.append(round(dividend, 2))
        pv = dividend / ((1 + required_return) ** year)
        stage1_pv.append(round(pv, 2))
    
    pv_stage1 = sum(stage1_pv)
    
    # Stage 2: Calculate terminal value using Gordon Growth Model
    # First dividend in stable phase
    first_stable_dividend = dividend * (1 + stable_growth_rate)
    
    # Terminal value at the end of high growth phase
    terminal_value = first_stable_dividend / (required_return - stable_growth_rate)
    
    # Present value of terminal value
    pv_terminal = terminal_value / ((1 + required_return) ** high_growth_years)
    
    # Total fair value
    fair_value = pv_stage1 + pv_terminal
    
    return {
        'fair_value': round(fair_value, 2),
        'stage1_dividends': stage1_dividends,
        'stage1_pv': stage1_pv,
        'pv_stage1': round(pv_stage1, 2),
        'terminal_value': round(terminal_value, 2),
        'pv_terminal': round(pv_terminal, 2),
        'current_dividend': current_dividend,
        'high_growth_rate': round(high_growth_rate * 100, 2),
        'stable_growth_rate': round(stable_growth_rate * 100, 2),
        'required_return': round(required_return * 100, 2),
        'high_growth_years': high_growth_years
    }


def h_model_ddm(
    current_dividend: float,
    initial_growth_rate: float,
    stable_growth_rate: float,
    half_life_years: float,
    required_return: float
) -> Dict[str, Any]:
    """
    Calculate intrinsic value using H-Model (gradual growth decline).
    
    The H-Model assumes growth rate declines linearly from initial to stable rate.
    
    Formula:
        P = D₀ × (1 + gₛ) / (r - gₛ) + D₀ × H × (gᵢ - gₛ) / (r - gₛ)
    
    Where:
        gₛ = Stable growth rate
        gᵢ = Initial high growth rate
        H = Half-life of high growth period
    
    Args:
        current_dividend: Current annual dividend
        initial_growth_rate: Initial high growth rate
        stable_growth_rate: Long-term stable growth rate
        half_life_years: Years until growth rate is halfway between initial and stable
        required_return: Required rate of return
    
    Returns:
        Dictionary with valuation results
    """
    if required_return <= stable_growth_rate:
        return {
            'fair_value': None,
            'error': 'Required return must be greater than stable growth rate'
        }
    
    if current_dividend <= 0:
        return {
            'fair_value': None,
            'error': 'Company must pay dividends'
        }
    
    # Gordon Growth component (stable growth value)
    stable_value = (current_dividend * (1 + stable_growth_rate)) / (required_return - stable_growth_rate)
    
    # H-Model extraordinary growth component
    extraordinary_value = (current_dividend * half_life_years * (initial_growth_rate - stable_growth_rate)) / (required_return - stable_growth_rate)
    
    fair_value = stable_value + extraordinary_value
    
    return {
        'fair_value': round(fair_value, 2),
        'stable_value_component': round(stable_value, 2),
        'extraordinary_value_component': round(extraordinary_value, 2),
        'current_dividend': current_dividend,
        'initial_growth_rate': round(initial_growth_rate * 100, 2),
        'stable_growth_rate': round(stable_growth_rate * 100, 2),
        'half_life_years': half_life_years,
        'required_return': round(required_return * 100, 2)
    }


def estimate_growth_rate(
    roe: float,
    retention_ratio: float
) -> float:
    """
    Estimate sustainable dividend growth rate.
    
    Formula:
        g = ROE × Retention Ratio
        g = ROE × (1 - Payout Ratio)
    
    Args:
        roe: Return on Equity (decimal)
        retention_ratio: Earnings retention ratio (1 - payout ratio)
    
    Returns:
        Estimated growth rate as decimal
    """
    return roe * retention_ratio


def implied_growth_rate(
    stock_price: float,
    current_dividend: float,
    required_return: float
) -> float:
    """
    Calculate implied growth rate from current stock price using DDM.
    
    Rearranging Gordon Growth Model:
        g = r - D₁/P
    
    Args:
        stock_price: Current stock price
        current_dividend: Current annual dividend
        required_return: Required rate of return
    
    Returns:
        Implied growth rate as decimal
    """
    if stock_price <= 0 or current_dividend <= 0:
        return 0
    
    # Assume next dividend is current * (1 + estimated growth)
    # Using iterative approach
    implied_yield = current_dividend / stock_price
    implied_growth = required_return - implied_yield
    
    return round(implied_growth, 4)
