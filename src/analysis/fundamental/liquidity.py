"""
Liquidity Ratios Module
"""

from typing import Dict, Any


def calculate_current_ratio(
    current_assets: float,
    current_liabilities: float
) -> Dict[str, Any]:
    """
    Calculate Current Ratio.
    
    Formula:
        Current Ratio = Current Assets / Current Liabilities
    
    Measures ability to pay short-term obligations.
    
    Args:
        current_assets: Total current assets
        current_liabilities: Total current liabilities
    
    Returns:
        Dictionary with ratio and interpretation
    """
    if current_liabilities <= 0:
        return {
            'current_ratio': None,
            'interpretation': 'N/A (no current liabilities)'
        }
    
    ratio = current_assets / current_liabilities
    
    # Interpretation
    if ratio < 1:
        interp = 'Poor liquidity - may struggle to meet short-term obligations'
    elif ratio < 1.5:
        interp = 'Adequate liquidity'
    elif ratio < 3:
        interp = 'Good liquidity'
    else:
        interp = 'Very high liquidity (may indicate inefficient asset use)'
    
    return {
        'current_ratio': round(ratio, 2),
        'current_assets': current_assets,
        'current_liabilities': current_liabilities,
        'interpretation': interp
    }


def calculate_quick_ratio(
    current_assets: float,
    inventory: float,
    current_liabilities: float
) -> Dict[str, Any]:
    """
    Calculate Quick Ratio (Acid Test).
    
    Formula:
        Quick Ratio = (Current Assets - Inventory) / Current Liabilities
    
    More conservative liquidity measure excluding inventory.
    
    Args:
        current_assets: Total current assets
        inventory: Total inventory
        current_liabilities: Total current liabilities
    
    Returns:
        Dictionary with ratio and interpretation
    """
    if current_liabilities <= 0:
        return {
            'quick_ratio': None,
            'interpretation': 'N/A'
        }
    
    quick_assets = current_assets - inventory
    ratio = quick_assets / current_liabilities
    
    # Interpretation
    if ratio < 0.5:
        interp = 'Poor quick liquidity'
    elif ratio < 1:
        interp = 'Below ideal quick liquidity'
    elif ratio < 1.5:
        interp = 'Good quick liquidity'
    else:
        interp = 'Strong quick liquidity'
    
    return {
        'quick_ratio': round(ratio, 2),
        'quick_assets': quick_assets,
        'current_liabilities': current_liabilities,
        'interpretation': interp
    }


def calculate_cash_ratio(
    cash_and_equivalents: float,
    current_liabilities: float
) -> Dict[str, Any]:
    """
    Calculate Cash Ratio.
    
    Formula:
        Cash Ratio = Cash and Cash Equivalents / Current Liabilities
    
    Most conservative liquidity measure.
    
    Args:
        cash_and_equivalents: Cash and cash equivalents
        current_liabilities: Total current liabilities
    
    Returns:
        Dictionary with ratio and interpretation
    """
    if current_liabilities <= 0:
        return {
            'cash_ratio': None,
            'interpretation': 'N/A'
        }
    
    ratio = cash_and_equivalents / current_liabilities
    
    # Interpretation
    if ratio < 0.2:
        interp = 'Low cash coverage'
    elif ratio < 0.5:
        interp = 'Adequate cash coverage'
    elif ratio < 1:
        interp = 'Good cash coverage'
    else:
        interp = 'Strong cash coverage (may be holding too much cash)'
    
    return {
        'cash_ratio': round(ratio, 2),
        'cash': cash_and_equivalents,
        'current_liabilities': current_liabilities,
        'interpretation': interp
    }


def calculate_working_capital(
    current_assets: float,
    current_liabilities: float
) -> Dict[str, Any]:
    """
    Calculate Working Capital.
    
    Formula:
        Working Capital = Current Assets - Current Liabilities
    
    Args:
        current_assets: Total current assets
        current_liabilities: Total current liabilities
    
    Returns:
        Dictionary with working capital amount
    """
    working_capital = current_assets - current_liabilities
    
    # Interpretation
    if working_capital < 0:
        interp = 'Negative working capital (short-term liabilities exceed assets)'
    elif working_capital == 0:
        interp = 'Zero working capital'
    else:
        interp = 'Positive working capital'
    
    return {
        'working_capital': round(working_capital, 2),
        'current_assets': current_assets,
        'current_liabilities': current_liabilities,
        'interpretation': interp
    }


def calculate_operating_cash_flow_ratio(
    operating_cash_flow: float,
    current_liabilities: float
) -> Dict[str, Any]:
    """
    Calculate Operating Cash Flow Ratio.
    
    Formula:
        OCF Ratio = Operating Cash Flow / Current Liabilities
    
    Measures ability to pay current liabilities from operating cash.
    
    Args:
        operating_cash_flow: Cash flow from operations
        current_liabilities: Total current liabilities
    
    Returns:
        Dictionary with ratio
    """
    if current_liabilities <= 0:
        return {
            'ocf_ratio': None,
            'interpretation': 'N/A'
        }
    
    ratio = operating_cash_flow / current_liabilities
    
    return {
        'ocf_ratio': round(ratio, 2),
        'operating_cash_flow': operating_cash_flow,
        'current_liabilities': current_liabilities,
        'interpretation': f"Operating cash can cover {ratio:.1f}x current liabilities"
    }


def comprehensive_liquidity(financials: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate all liquidity ratios from financial data.
    
    Args:
        financials: Dictionary with financial data
    
    Returns:
        Dictionary with all liquidity metrics
    """
    results = {}
    
    ca = financials.get('current_assets', 0)
    cl = financials.get('current_liabilities', 0)
    
    if ca and cl:
        results['current_ratio'] = calculate_current_ratio(ca, cl)
        results['working_capital'] = calculate_working_capital(ca, cl)
    
    if inv := financials.get('inventory'):
        if ca and cl:
            results['quick_ratio'] = calculate_quick_ratio(ca, inv, cl)
    
    if cash := financials.get('cash'):
        if cl:
            results['cash_ratio'] = calculate_cash_ratio(cash, cl)
    
    if ocf := financials.get('operating_cash_flow'):
        if cl:
            results['ocf_ratio'] = calculate_operating_cash_flow_ratio(ocf, cl)
    
    return results
