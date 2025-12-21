"""
Profitability Ratios Module
"""

from typing import Dict, Any, Optional


def calculate_roe(
    net_income: float,
    shareholders_equity: float,
    beginning_equity: Optional[float] = None
) -> Dict[str, Any]:
    """
    Calculate Return on Equity (ROE).
    
    Formula:
        ROE = Net Income / Shareholders' Equity
        
    For more accuracy, use average equity:
        ROE = Net Income / ((Beginning Equity + Ending Equity) / 2)
    
    Args:
        net_income: Net income for the period
        shareholders_equity: Total shareholders' equity (ending)
        beginning_equity: Beginning shareholders' equity (optional, for averaging)
    
    Returns:
        Dictionary with ROE and interpretation
    """
    if beginning_equity is not None:
        avg_equity = (beginning_equity + shareholders_equity) / 2
    else:
        avg_equity = shareholders_equity
    
    if avg_equity <= 0:
        return {
            'roe': None,
            'interpretation': 'N/A (negative or zero equity)'
        }
    
    roe = (net_income / avg_equity) * 100
    
    # Interpretation
    if roe < 0:
        interp = 'Negative return (company losing money)'
    elif roe < 10:
        interp = 'Below average return on equity'
    elif roe < 15:
        interp = 'Average return on equity'
    elif roe < 20:
        interp = 'Good return on equity'
    else:
        interp = 'Excellent return on equity'
    
    return {
        'roe': round(roe, 2),
        'net_income': net_income,
        'equity': avg_equity,
        'interpretation': interp
    }


def calculate_roa(
    net_income: float,
    total_assets: float,
    beginning_assets: Optional[float] = None
) -> Dict[str, Any]:
    """
    Calculate Return on Assets (ROA).
    
    Formula:
        ROA = Net Income / Average Total Assets
    
    Args:
        net_income: Net income for the period
        total_assets: Total assets (ending)
        beginning_assets: Beginning total assets (optional)
    
    Returns:
        Dictionary with ROA and interpretation
    """
    if beginning_assets is not None:
        avg_assets = (beginning_assets + total_assets) / 2
    else:
        avg_assets = total_assets
    
    if avg_assets <= 0:
        return {
            'roa': None,
            'interpretation': 'N/A (no assets)'
        }
    
    roa = (net_income / avg_assets) * 100
    
    # Interpretation varies by industry
    if roa < 0:
        interp = 'Negative return (company losing money)'
    elif roa < 5:
        interp = 'Low asset efficiency'
    elif roa < 10:
        interp = 'Average asset efficiency'
    elif roa < 20:
        interp = 'Good asset efficiency'
    else:
        interp = 'Excellent asset efficiency'
    
    return {
        'roa': round(roa, 2),
        'net_income': net_income,
        'total_assets': avg_assets,
        'interpretation': interp
    }


def calculate_roic(
    operating_income: float,
    tax_rate: float,
    invested_capital: float
) -> Dict[str, Any]:
    """
    Calculate Return on Invested Capital (ROIC).
    
    Formula:
        ROIC = NOPAT / Invested Capital
        NOPAT = Operating Income × (1 - Tax Rate)
        Invested Capital = Equity + Debt - Cash
    
    Args:
        operating_income: Operating income (EBIT)
        tax_rate: Effective tax rate (decimal)
        invested_capital: Total invested capital
    
    Returns:
        Dictionary with ROIC and interpretation
    """
    if invested_capital <= 0:
        return {
            'roic': None,
            'interpretation': 'N/A (no invested capital)'
        }
    
    nopat = operating_income * (1 - tax_rate)
    roic = (nopat / invested_capital) * 100
    
    # Interpretation
    if roic < 0:
        interp = 'Destroying value (below cost of capital)'
    elif roic < 8:
        interp = 'Likely below cost of capital'
    elif roic < 15:
        interp = 'Generating adequate returns'
    elif roic < 25:
        interp = 'Strong returns on capital'
    else:
        interp = 'Exceptional returns (competitive advantage)'
    
    return {
        'roic': round(roic, 2),
        'nopat': round(nopat, 2),
        'invested_capital': invested_capital,
        'interpretation': interp
    }


def calculate_net_profit_margin(
    net_income: float,
    revenue: float
) -> Dict[str, Any]:
    """
    Calculate Net Profit Margin.
    
    Formula:
        Net Profit Margin = (Net Income / Revenue) × 100
    
    Args:
        net_income: Net income
        revenue: Total revenue
    
    Returns:
        Dictionary with margin and interpretation
    """
    if revenue <= 0:
        return {
            'net_margin': None,
            'interpretation': 'N/A (no revenue)'
        }
    
    margin = (net_income / revenue) * 100
    
    return {
        'net_margin': round(margin, 2),
        'net_income': net_income,
        'revenue': revenue,
        'interpretation': f'{margin:.1f}% of revenue becomes profit'
    }


def calculate_gross_profit_margin(
    revenue: float,
    cost_of_goods_sold: float
) -> Dict[str, Any]:
    """
    Calculate Gross Profit Margin.
    
    Formula:
        Gross Margin = ((Revenue - COGS) / Revenue) × 100
    
    Args:
        revenue: Total revenue
        cost_of_goods_sold: Cost of goods sold
    
    Returns:
        Dictionary with margin
    """
    if revenue <= 0:
        return {
            'gross_margin': None,
            'interpretation': 'N/A'
        }
    
    gross_profit = revenue - cost_of_goods_sold
    margin = (gross_profit / revenue) * 100
    
    return {
        'gross_margin': round(margin, 2),
        'gross_profit': gross_profit,
        'revenue': revenue
    }


def calculate_operating_margin(
    operating_income: float,
    revenue: float
) -> Dict[str, Any]:
    """
    Calculate Operating Profit Margin.
    
    Formula:
        Operating Margin = (Operating Income / Revenue) × 100
    
    Args:
        operating_income: Operating income (EBIT)
        revenue: Total revenue
    
    Returns:
        Dictionary with margin
    """
    if revenue <= 0:
        return {
            'operating_margin': None,
            'interpretation': 'N/A'
        }
    
    margin = (operating_income / revenue) * 100
    
    return {
        'operating_margin': round(margin, 2),
        'operating_income': operating_income,
        'revenue': revenue
    }


def calculate_ebitda_margin(
    ebitda: float,
    revenue: float
) -> Dict[str, Any]:
    """
    Calculate EBITDA Margin.
    
    Formula:
        EBITDA Margin = (EBITDA / Revenue) × 100
    
    Args:
        ebitda: Earnings before interest, taxes, depreciation, amortization
        revenue: Total revenue
    
    Returns:
        Dictionary with margin
    """
    if revenue <= 0:
        return {
            'ebitda_margin': None,
            'interpretation': 'N/A'
        }
    
    margin = (ebitda / revenue) * 100
    
    return {
        'ebitda_margin': round(margin, 2),
        'ebitda': ebitda,
        'revenue': revenue
    }


def calculate_eps(
    net_income: float,
    preferred_dividends: float,
    weighted_avg_shares: float
) -> Dict[str, Any]:
    """
    Calculate Earnings Per Share (EPS).
    
    Formula:
        EPS = (Net Income - Preferred Dividends) / Weighted Average Shares Outstanding
    
    Args:
        net_income: Net income
        preferred_dividends: Dividends paid to preferred shareholders
        weighted_avg_shares: Weighted average common shares outstanding
    
    Returns:
        Dictionary with EPS
    """
    if weighted_avg_shares <= 0:
        return {
            'eps': None,
            'interpretation': 'N/A'
        }
    
    eps = (net_income - preferred_dividends) / weighted_avg_shares
    
    return {
        'eps': round(eps, 2),
        'net_income': net_income,
        'shares': weighted_avg_shares
    }


def comprehensive_profitability(financials: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate all profitability ratios from financial data.
    
    Args:
        financials: Dictionary with financial data
    
    Returns:
        Dictionary with all profitability metrics
    """
    results = {}
    
    # ROE
    if 'net_income' in financials and 'shareholders_equity' in financials:
        results['roe'] = calculate_roe(
            financials['net_income'],
            financials['shareholders_equity'],
            financials.get('beginning_equity')
        )
    
    # ROA
    if 'net_income' in financials and 'total_assets' in financials:
        results['roa'] = calculate_roa(
            financials['net_income'],
            financials['total_assets'],
            financials.get('beginning_assets')
        )
    
    # ROIC
    if all(k in financials for k in ['operating_income', 'tax_rate', 'invested_capital']):
        results['roic'] = calculate_roic(
            financials['operating_income'],
            financials['tax_rate'],
            financials['invested_capital']
        )
    
    # Margins
    if 'net_income' in financials and 'revenue' in financials:
        results['net_margin'] = calculate_net_profit_margin(
            financials['net_income'],
            financials['revenue']
        )
    
    if 'revenue' in financials and 'cogs' in financials:
        results['gross_margin'] = calculate_gross_profit_margin(
            financials['revenue'],
            financials['cogs']
        )
    
    if 'operating_income' in financials and 'revenue' in financials:
        results['operating_margin'] = calculate_operating_margin(
            financials['operating_income'],
            financials['revenue']
        )
    
    if 'ebitda' in financials and 'revenue' in financials:
        results['ebitda_margin'] = calculate_ebitda_margin(
            financials['ebitda'],
            financials['revenue']
        )
    
    return results
