"""
Discounted Cash Flow (DCF) Valuation Model
"""

import numpy as np
from typing import Dict, List, Any, Optional


def calculate_wacc(
    equity_value: float,
    debt_value: float,
    cost_of_equity: float,
    cost_of_debt: float,
    tax_rate: float
) -> float:
    """
    Calculate Weighted Average Cost of Capital (WACC).
    
    Formula:
        WACC = (E/V × Re) + (D/V × Rd × (1-T))
        
    Where:
        E = Market value of equity
        D = Market value of debt
        V = Total value (E + D)
        Re = Cost of equity
        Rd = Cost of debt
        T = Tax rate
    
    Args:
        equity_value: Market value of equity
        debt_value: Market value of debt
        cost_of_equity: Expected return for equity holders (decimal, e.g., 0.10 for 10%)
        cost_of_debt: Interest rate on debt (decimal)
        tax_rate: Corporate tax rate (decimal)
    
    Returns:
        WACC as a decimal
    """
    total_value = equity_value + debt_value
    
    if total_value == 0:
        return cost_of_equity  # Default to cost of equity if no data
    
    equity_weight = equity_value / total_value
    debt_weight = debt_value / total_value
    
    wacc = (equity_weight * cost_of_equity) + (debt_weight * cost_of_debt * (1 - tax_rate))
    
    return wacc


def calculate_cost_of_equity(
    risk_free_rate: float,
    beta: float,
    market_risk_premium: float
) -> float:
    """
    Calculate Cost of Equity using Capital Asset Pricing Model (CAPM).
    
    Formula:
        Re = Rf + β × (Rm - Rf)
        
    Where:
        Rf = Risk-free rate
        β = Beta (systematic risk)
        Rm - Rf = Market risk premium
    
    Args:
        risk_free_rate: Risk-free rate (e.g., 10-year Treasury yield)
        beta: Stock's beta coefficient
        market_risk_premium: Expected market return minus risk-free rate
    
    Returns:
        Cost of equity as a decimal
    """
    return risk_free_rate + (beta * market_risk_premium)


def calculate_terminal_value_perpetuity(
    final_fcf: float,
    perpetual_growth_rate: float,
    wacc: float
) -> float:
    """
    Calculate Terminal Value using Perpetuity Growth Method (Gordon Growth).
    
    Formula:
        TV = FCF × (1 + g) / (WACC - g)
    
    Args:
        final_fcf: Free cash flow in the final projection year
        perpetual_growth_rate: Long-term growth rate (should be < GDP growth)
        wacc: Weighted average cost of capital
    
    Returns:
        Terminal value
    """
    if wacc <= perpetual_growth_rate:
        raise ValueError("WACC must be greater than perpetual growth rate")
    
    return (final_fcf * (1 + perpetual_growth_rate)) / (wacc - perpetual_growth_rate)


def calculate_terminal_value_exit_multiple(
    final_ebitda: float,
    exit_multiple: float
) -> float:
    """
    Calculate Terminal Value using Exit Multiple Method.
    
    Formula:
        TV = EBITDA × Exit Multiple
    
    Args:
        final_ebitda: EBITDA in the final projection year
        exit_multiple: EV/EBITDA multiple
    
    Returns:
        Terminal value
    """
    return final_ebitda * exit_multiple


def discount_cash_flows(
    cash_flows: List[float],
    discount_rate: float
) -> List[float]:
    """
    Discount future cash flows to present value.
    
    Formula:
        PV = CF / (1 + r)^n
    
    Args:
        cash_flows: List of future cash flows
        discount_rate: Discount rate (WACC)
    
    Returns:
        List of present values
    """
    present_values = []
    for i, cf in enumerate(cash_flows):
        year = i + 1
        pv = cf / ((1 + discount_rate) ** year)
        present_values.append(pv)
    
    return present_values


def dcf_valuation(
    current_fcf: float,
    growth_rates: List[float],
    wacc: float,
    terminal_growth_rate: float = 0.025,
    shares_outstanding: float = 1,
    net_debt: float = 0
) -> Dict[str, Any]:
    """
    Perform complete DCF valuation.
    
    Formula:
        DCF = Σ(CFn / (1+r)^n) + TV / (1+r)^n
    
    Args:
        current_fcf: Current free cash flow
        growth_rates: List of projected growth rates for each year
        wacc: Weighted average cost of capital
        terminal_growth_rate: Perpetual growth rate after projection period
        shares_outstanding: Number of shares outstanding
        net_debt: Net debt (total debt - cash)
    
    Returns:
        Dictionary with valuation details
    """
    # Project future cash flows
    projected_fcf = [current_fcf]
    for growth_rate in growth_rates:
        next_fcf = projected_fcf[-1] * (1 + growth_rate)
        projected_fcf.append(next_fcf)
    
    # Remove current FCF (index 0), keep only projected
    projected_fcf = projected_fcf[1:]
    
    # Calculate present value of projected cash flows
    pv_cash_flows = discount_cash_flows(projected_fcf, wacc)
    
    # Calculate terminal value
    terminal_value = calculate_terminal_value_perpetuity(
        projected_fcf[-1], 
        terminal_growth_rate, 
        wacc
    )
    
    # Discount terminal value to present
    projection_years = len(projected_fcf)
    pv_terminal_value = terminal_value / ((1 + wacc) ** projection_years)
    
    # Calculate enterprise value
    enterprise_value = sum(pv_cash_flows) + pv_terminal_value
    
    # Calculate equity value
    equity_value = enterprise_value - net_debt
    
    # Calculate per share value
    fair_value_per_share = equity_value / shares_outstanding if shares_outstanding > 0 else 0
    
    return {
        'projected_fcf': [round(fcf, 2) for fcf in projected_fcf],
        'pv_cash_flows': [round(pv, 2) for pv in pv_cash_flows],
        'terminal_value': round(terminal_value, 2),
        'pv_terminal_value': round(pv_terminal_value, 2),
        'enterprise_value': round(enterprise_value, 2),
        'equity_value': round(equity_value, 2),
        'fair_value_per_share': round(fair_value_per_share, 2),
        'wacc': round(wacc * 100, 2),
        'terminal_growth_rate': round(terminal_growth_rate * 100, 2),
        'projection_years': projection_years,
        'tv_as_percentage_of_ev': round((pv_terminal_value / enterprise_value) * 100, 1) if enterprise_value > 0 else 0
    }


def sensitivity_analysis(
    base_dcf: Dict[str, Any],
    current_fcf: float,
    growth_rates: List[float],
    shares_outstanding: float,
    net_debt: float,
    wacc_range: List[float] = None,
    growth_range: List[float] = None
) -> Dict[str, List[List[float]]]:
    """
    Perform sensitivity analysis on DCF valuation.
    
    Varies WACC and terminal growth rate to show range of fair values.
    
    Args:
        base_dcf: Base DCF results
        current_fcf: Current free cash flow
        growth_rates: Projected growth rates
        shares_outstanding: Shares outstanding
        net_debt: Net debt
        wacc_range: List of WACC values to test
        growth_range: List of terminal growth rates to test
    
    Returns:
        Sensitivity table with fair values
    """
    if wacc_range is None:
        base_wacc = base_dcf['wacc'] / 100
        wacc_range = [base_wacc - 0.02, base_wacc - 0.01, base_wacc, base_wacc + 0.01, base_wacc + 0.02]
    
    if growth_range is None:
        base_growth = base_dcf['terminal_growth_rate'] / 100
        growth_range = [base_growth - 0.01, base_growth - 0.005, base_growth, base_growth + 0.005, base_growth + 0.01]
    
    sensitivity_table = []
    
    for growth in growth_range:
        row = []
        for wacc in wacc_range:
            if wacc > growth:
                result = dcf_valuation(
                    current_fcf, growth_rates, wacc, growth,
                    shares_outstanding, net_debt
                )
                row.append(result['fair_value_per_share'])
            else:
                row.append(None)  # Invalid combination
        sensitivity_table.append(row)
    
    return {
        'wacc_values': [round(w * 100, 1) for w in wacc_range],
        'growth_values': [round(g * 100, 1) for g in growth_range],
        'fair_values': sensitivity_table
    }
