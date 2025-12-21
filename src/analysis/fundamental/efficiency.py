"""
Efficiency (Activity) Ratios Module
"""

from typing import Dict, Any


def calculate_asset_turnover(
    revenue: float,
    average_total_assets: float
) -> Dict[str, Any]:
    """
    Calculate Asset Turnover Ratio.
    
    Formula:
        Asset Turnover = Revenue / Average Total Assets
    
    Measures efficiency of using assets to generate sales.
    
    Args:
        revenue: Total revenue
        average_total_assets: Average total assets
    
    Returns:
        Dictionary with ratio and interpretation
    """
    if average_total_assets <= 0:
        return {
            'asset_turnover': None,
            'interpretation': 'N/A'
        }
    
    ratio = revenue / average_total_assets
    
    return {
        'asset_turnover': round(ratio, 2),
        'revenue': revenue,
        'avg_assets': average_total_assets,
        'interpretation': f"${ratio:.2f} revenue per $1 of assets"
    }


def calculate_inventory_turnover(
    cost_of_goods_sold: float,
    average_inventory: float
) -> Dict[str, Any]:
    """
    Calculate Inventory Turnover Ratio.
    
    Formula:
        Inventory Turnover = COGS / Average Inventory
    
    Measures how quickly inventory is sold.
    
    Args:
        cost_of_goods_sold: Cost of goods sold
        average_inventory: Average inventory
    
    Returns:
        Dictionary with ratio and interpretation
    """
    if average_inventory <= 0:
        return {
            'inventory_turnover': None,
            'interpretation': 'N/A (no inventory)'
        }
    
    turnover = cost_of_goods_sold / average_inventory
    days_in_inventory = 365 / turnover if turnover > 0 else 0
    
    return {
        'inventory_turnover': round(turnover, 2),
        'days_in_inventory': round(days_in_inventory, 1),
        'interpretation': f"Inventory turns over {turnover:.1f}x per year ({days_in_inventory:.0f} days)"
    }


def calculate_receivables_turnover(
    revenue: float,
    average_receivables: float
) -> Dict[str, Any]:
    """
    Calculate Receivables Turnover Ratio.
    
    Formula:
        Receivables Turnover = Revenue / Average Accounts Receivable
    
    Measures efficiency in collecting receivables.
    
    Args:
        revenue: Total revenue (or net credit sales)
        average_receivables: Average accounts receivable
    
    Returns:
        Dictionary with ratio and interpretation
    """
    if average_receivables <= 0:
        return {
            'receivables_turnover': None,
            'interpretation': 'N/A'
        }
    
    turnover = revenue / average_receivables
    dso = 365 / turnover if turnover > 0 else 0
    
    return {
        'receivables_turnover': round(turnover, 2),
        'days_sales_outstanding': round(dso, 1),
        'interpretation': f"Collects receivables {turnover:.1f}x per year ({dso:.0f} days average)"
    }


def calculate_payables_turnover(
    cost_of_goods_sold: float,
    average_payables: float
) -> Dict[str, Any]:
    """
    Calculate Payables Turnover Ratio.
    
    Formula:
        Payables Turnover = COGS / Average Accounts Payable
    
    Measures how quickly company pays suppliers.
    
    Args:
        cost_of_goods_sold: Cost of goods sold
        average_payables: Average accounts payable
    
    Returns:
        Dictionary with ratio and interpretation
    """
    if average_payables <= 0:
        return {
            'payables_turnover': None,
            'interpretation': 'N/A'
        }
    
    turnover = cost_of_goods_sold / average_payables
    dpo = 365 / turnover if turnover > 0 else 0
    
    return {
        'payables_turnover': round(turnover, 2),
        'days_payable_outstanding': round(dpo, 1),
        'interpretation': f"Pays suppliers {turnover:.1f}x per year ({dpo:.0f} days average)"
    }


def calculate_cash_conversion_cycle(
    days_inventory: float,
    days_receivables: float,
    days_payables: float
) -> Dict[str, Any]:
    """
    Calculate Cash Conversion Cycle.
    
    Formula:
        CCC = Days Inventory + Days Receivables - Days Payable
    
    Measures time to convert inventory investment back to cash.
    
    Args:
        days_inventory: Days inventory outstanding
        days_receivables: Days sales outstanding
        days_payables: Days payable outstanding
    
    Returns:
        Dictionary with CCC
    """
    ccc = days_inventory + days_receivables - days_payables
    
    # Interpretation
    if ccc < 0:
        interp = 'Negative CCC - company receives cash before paying suppliers'
    elif ccc < 30:
        interp = 'Efficient cash conversion'
    elif ccc < 60:
        interp = 'Average cash conversion'
    else:
        interp = 'Long cash conversion cycle'
    
    return {
        'cash_conversion_cycle': round(ccc, 1),
        'days_inventory': days_inventory,
        'days_receivables': days_receivables,
        'days_payables': days_payables,
        'interpretation': interp
    }


def calculate_fixed_asset_turnover(
    revenue: float,
    average_fixed_assets: float
) -> Dict[str, Any]:
    """
    Calculate Fixed Asset Turnover Ratio.
    
    Formula:
        Fixed Asset Turnover = Revenue / Average Net Fixed Assets
    
    Args:
        revenue: Total revenue
        average_fixed_assets: Average net property, plant & equipment
    
    Returns:
        Dictionary with ratio
    """
    if average_fixed_assets <= 0:
        return {
            'fixed_asset_turnover': None,
            'interpretation': 'N/A'
        }
    
    ratio = revenue / average_fixed_assets
    
    return {
        'fixed_asset_turnover': round(ratio, 2),
        'interpretation': f"${ratio:.2f} revenue per $1 of fixed assets"
    }


def comprehensive_efficiency(financials: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate all efficiency ratios from financial data.
    
    Args:
        financials: Dictionary with financial data
    
    Returns:
        Dictionary with all efficiency metrics
    """
    results = {}
    
    revenue = financials.get('revenue', 0)
    
    if revenue and (assets := financials.get('avg_total_assets')):
        results['asset_turnover'] = calculate_asset_turnover(revenue, assets)
    
    if cogs := financials.get('cogs'):
        if inv := financials.get('avg_inventory'):
            results['inventory_turnover'] = calculate_inventory_turnover(cogs, inv)
    
    if revenue and (rec := financials.get('avg_receivables')):
        results['receivables_turnover'] = calculate_receivables_turnover(revenue, rec)
    
    if cogs := financials.get('cogs'):
        if pay := financials.get('avg_payables'):
            results['payables_turnover'] = calculate_payables_turnover(cogs, pay)
    
    # Calculate CCC if we have all components
    if all(k in results for k in ['inventory_turnover', 'receivables_turnover', 'payables_turnover']):
        results['cash_conversion_cycle'] = calculate_cash_conversion_cycle(
            results['inventory_turnover']['days_in_inventory'],
            results['receivables_turnover']['days_sales_outstanding'],
            results['payables_turnover']['days_payable_outstanding']
        )
    
    if revenue and (fa := financials.get('avg_fixed_assets')):
        results['fixed_asset_turnover'] = calculate_fixed_asset_turnover(revenue, fa)
    
    return results
