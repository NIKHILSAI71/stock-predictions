"""
Solvency and Leverage Ratios Module
"""

from typing import Dict, Any


def calculate_debt_to_equity(
    total_debt: float,
    shareholders_equity: float
) -> Dict[str, Any]:
    """
    Calculate Debt-to-Equity (D/E) Ratio.
    
    Formula:
        D/E = Total Debt / Shareholders' Equity
    
    Args:
        total_debt: Total debt (short-term + long-term)
        shareholders_equity: Total shareholders' equity
    
    Returns:
        Dictionary with ratio and interpretation
    """
    if shareholders_equity <= 0:
        return {
            'debt_to_equity': None,
            'interpretation': 'N/A (negative or zero equity)'
        }
    
    ratio = total_debt / shareholders_equity
    
    # Interpretation
    if ratio < 0.5:
        interp = 'Low leverage - conservative capital structure'
    elif ratio < 1:
        interp = 'Moderate leverage'
    elif ratio < 2:
        interp = 'High leverage'
    else:
        interp = 'Very high leverage - significant debt load'
    
    return {
        'debt_to_equity': round(ratio, 2),
        'total_debt': total_debt,
        'equity': shareholders_equity,
        'interpretation': interp
    }


def calculate_debt_to_assets(
    total_debt: float,
    total_assets: float
) -> Dict[str, Any]:
    """
    Calculate Debt-to-Assets Ratio.
    
    Formula:
        D/A = Total Debt / Total Assets
    
    Args:
        total_debt: Total debt
        total_assets: Total assets
    
    Returns:
        Dictionary with ratio and interpretation
    """
    if total_assets <= 0:
        return {
            'debt_to_assets': None,
            'interpretation': 'N/A'
        }
    
    ratio = total_debt / total_assets
    percentage = ratio * 100
    
    # Interpretation
    if ratio < 0.3:
        interp = 'Low debt relative to assets'
    elif ratio < 0.5:
        interp = 'Moderate debt levels'
    elif ratio < 0.7:
        interp = 'High debt levels'
    else:
        interp = 'Very high debt - assets largely financed by debt'
    
    return {
        'debt_to_assets': round(ratio, 2),
        'debt_percentage': round(percentage, 1),
        'total_debt': total_debt,
        'total_assets': total_assets,
        'interpretation': interp
    }


def calculate_interest_coverage(
    ebit: float,
    interest_expense: float
) -> Dict[str, Any]:
    """
    Calculate Interest Coverage Ratio.
    
    Formula:
        Interest Coverage = EBIT / Interest Expense
    
    Measures ability to pay interest on debt.
    
    Args:
        ebit: Earnings before interest and taxes
        interest_expense: Interest expense
    
    Returns:
        Dictionary with ratio and interpretation
    """
    if interest_expense <= 0:
        return {
            'interest_coverage': None,
            'interpretation': 'No interest expense (debt-free or minimal debt)'
        }
    
    ratio = ebit / interest_expense
    
    # Interpretation
    if ratio < 1:
        interp = 'Critical - cannot cover interest payments'
    elif ratio < 1.5:
        interp = 'Weak coverage - high risk'
    elif ratio < 3:
        interp = 'Adequate coverage'
    elif ratio < 5:
        interp = 'Good coverage'
    else:
        interp = 'Strong coverage - low debt risk'
    
    return {
        'interest_coverage': round(ratio, 2),
        'ebit': ebit,
        'interest_expense': interest_expense,
        'interpretation': interp
    }


def calculate_equity_ratio(
    shareholders_equity: float,
    total_assets: float
) -> Dict[str, Any]:
    """
    Calculate Equity Ratio.
    
    Formula:
        Equity Ratio = Shareholders' Equity / Total Assets
    
    Shows proportion of assets financed by equity.
    
    Args:
        shareholders_equity: Total shareholders' equity
        total_assets: Total assets
    
    Returns:
        Dictionary with ratio
    """
    if total_assets <= 0:
        return {
            'equity_ratio': None,
            'interpretation': 'N/A'
        }
    
    ratio = shareholders_equity / total_assets
    
    return {
        'equity_ratio': round(ratio, 2),
        'equity_percentage': round(ratio * 100, 1),
        'interpretation': f'{ratio * 100:.1f}% of assets financed by equity'
    }


def calculate_debt_service_coverage(
    net_operating_income: float,
    total_debt_service: float
) -> Dict[str, Any]:
    """
    Calculate Debt Service Coverage Ratio (DSCR).
    
    Formula:
        DSCR = Net Operating Income / Total Debt Service
    
    Args:
        net_operating_income: Net operating income
        total_debt_service: Total debt service (principal + interest)
    
    Returns:
        Dictionary with ratio
    """
    if total_debt_service <= 0:
        return {
            'dscr': None,
            'interpretation': 'No debt service'
        }
    
    ratio = net_operating_income / total_debt_service
    
    # Interpretation
    if ratio < 1:
        interp = 'Insufficient income to cover debt service'
    elif ratio < 1.25:
        interp = 'Marginal coverage'
    elif ratio < 1.5:
        interp = 'Adequate coverage'
    else:
        interp = 'Strong debt service coverage'
    
    return {
        'dscr': round(ratio, 2),
        'interpretation': interp
    }


def calculate_financial_leverage(
    total_assets: float,
    shareholders_equity: float
) -> Dict[str, Any]:
    """
    Calculate Financial Leverage (Equity Multiplier).
    
    Formula:
        Financial Leverage = Total Assets / Shareholders' Equity
    
    Args:
        total_assets: Total assets
        shareholders_equity: Total shareholders' equity
    
    Returns:
        Dictionary with leverage ratio
    """
    if shareholders_equity <= 0:
        return {
            'leverage': None,
            'interpretation': 'N/A'
        }
    
    leverage = total_assets / shareholders_equity
    
    return {
        'leverage': round(leverage, 2),
        'interpretation': f'{leverage:.1f}x equity multiplier'
    }


def comprehensive_solvency(financials: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate all solvency ratios from financial data.
    
    Args:
        financials: Dictionary with financial data
    
    Returns:
        Dictionary with all solvency metrics
    """
    results = {}
    
    debt = financials.get('total_debt', 0)
    equity = financials.get('shareholders_equity', 0)
    assets = financials.get('total_assets', 0)
    
    if debt and equity:
        results['debt_to_equity'] = calculate_debt_to_equity(debt, equity)
    
    if debt and assets:
        results['debt_to_assets'] = calculate_debt_to_assets(debt, assets)
    
    if equity and assets:
        results['equity_ratio'] = calculate_equity_ratio(equity, assets)
        results['leverage'] = calculate_financial_leverage(assets, equity)
    
    if ebit := financials.get('ebit'):
        if interest := financials.get('interest_expense'):
            results['interest_coverage'] = calculate_interest_coverage(ebit, interest)
    
    return results
