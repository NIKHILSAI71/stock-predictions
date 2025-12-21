"""
Valuation Ratios Module
Calculates P/E, P/B, P/S, EV/EBITDA, and other valuation metrics
"""

from typing import Dict, Any, Optional
import math


def calculate_pe_ratio(
    stock_price: float,
    eps: float,
    eps_type: str = 'trailing'
) -> Dict[str, Any]:
    """
    Calculate Price-to-Earnings (P/E) Ratio.
    
    Formula:
        P/E = Stock Price / Earnings Per Share
    
    Args:
        stock_price: Current stock price
        eps: Earnings per share (trailing or forward)
        eps_type: 'trailing' for TTM or 'forward' for estimated
    
    Returns:
        Dictionary with P/E ratio and interpretation
    """
    if eps <= 0:
        return {
            'pe_ratio': None,
            'eps': eps,
            'eps_type': eps_type,
            'interpretation': 'N/A (negative or zero earnings)'
        }
    
    pe = stock_price / eps
    
    # Interpretation
    if pe < 10:
        interp = 'Low valuation (potentially undervalued or declining business)'
    elif pe < 20:
        interp = 'Moderate valuation'
    elif pe < 30:
        interp = 'High valuation (growth expectations)'
    else:
        interp = 'Very high valuation (high growth or overvalued)'
    
    return {
        'pe_ratio': round(pe, 2),
        'eps': eps,
        'eps_type': eps_type,
        'interpretation': interp
    }


def calculate_peg_ratio(
    pe_ratio: float,
    earnings_growth_rate: float
) -> Dict[str, Any]:
    """
    Calculate Price/Earnings to Growth (PEG) Ratio.
    
    Formula:
        PEG = P/E Ratio / Annual EPS Growth Rate
    
    Args:
        pe_ratio: Price-to-Earnings ratio
        earnings_growth_rate: Estimated annual EPS growth rate (as percentage, e.g., 15 for 15%)
    
    Returns:
        Dictionary with PEG ratio and interpretation
    """
    if earnings_growth_rate <= 0:
        return {
            'peg_ratio': None,
            'pe_ratio': pe_ratio,
            'growth_rate': earnings_growth_rate,
            'interpretation': 'N/A (negative or zero growth)'
        }
    
    peg = pe_ratio / earnings_growth_rate
    
    # Interpretation (Peter Lynch's rule)
    if peg < 1:
        interp = 'Potentially undervalued (PEG < 1)'
    elif peg == 1:
        interp = 'Fairly valued (PEG = 1)'
    elif peg < 2:
        interp = 'Moderately valued'
    else:
        interp = 'Potentially overvalued (PEG > 2)'
    
    return {
        'peg_ratio': round(peg, 2),
        'pe_ratio': pe_ratio,
        'growth_rate': earnings_growth_rate,
        'interpretation': interp
    }


def calculate_pb_ratio(
    stock_price: float,
    book_value_per_share: float
) -> Dict[str, Any]:
    """
    Calculate Price-to-Book (P/B) Ratio.
    
    Formula:
        P/B = Stock Price / Book Value per Share
    
    Args:
        stock_price: Current stock price
        book_value_per_share: Book value per share
    
    Returns:
        Dictionary with P/B ratio and interpretation
    """
    if book_value_per_share <= 0:
        return {
            'pb_ratio': None,
            'book_value_per_share': book_value_per_share,
            'interpretation': 'N/A (negative book value)'
        }
    
    pb = stock_price / book_value_per_share
    
    # Interpretation
    if pb < 1:
        interp = 'Trading below book value (potential value investment or distressed)'
    elif pb < 3:
        interp = 'Moderate valuation'
    else:
        interp = 'High premium over book value (growth/quality premium or overvalued)'
    
    return {
        'pb_ratio': round(pb, 2),
        'book_value_per_share': book_value_per_share,
        'interpretation': interp
    }


def calculate_ps_ratio(
    market_cap: float,
    total_revenue: float
) -> Dict[str, Any]:
    """
    Calculate Price-to-Sales (P/S) Ratio.
    
    Formula:
        P/S = Market Capitalization / Total Revenue (TTM)
    
    Args:
        market_cap: Market capitalization
        total_revenue: Total revenue (trailing twelve months)
    
    Returns:
        Dictionary with P/S ratio and interpretation
    """
    if total_revenue <= 0:
        return {
            'ps_ratio': None,
            'market_cap': market_cap,
            'revenue': total_revenue,
            'interpretation': 'N/A (no revenue)'
        }
    
    ps = market_cap / total_revenue
    
    # Interpretation
    if ps < 1:
        interp = 'Low valuation relative to sales'
    elif ps < 3:
        interp = 'Moderate valuation'
    elif ps < 10:
        interp = 'High valuation (growth company premium)'
    else:
        interp = 'Very high valuation relative to sales'
    
    return {
        'ps_ratio': round(ps, 2),
        'market_cap': market_cap,
        'revenue': total_revenue,
        'interpretation': interp
    }


def calculate_ev_ebitda(
    enterprise_value: float,
    ebitda: float
) -> Dict[str, Any]:
    """
    Calculate Enterprise Value to EBITDA ratio.
    
    Formula:
        EV/EBITDA = Enterprise Value / EBITDA
        
    Enterprise Value = Market Cap + Total Debt - Cash
    
    Args:
        enterprise_value: Enterprise value
        ebitda: Earnings before interest, taxes, depreciation, amortization
    
    Returns:
        Dictionary with EV/EBITDA ratio and interpretation
    """
    if ebitda <= 0:
        return {
            'ev_ebitda': None,
            'enterprise_value': enterprise_value,
            'ebitda': ebitda,
            'interpretation': 'N/A (negative EBITDA)'
        }
    
    ev_ebitda = enterprise_value / ebitda
    
    # Interpretation (varies by industry)
    if ev_ebitda < 8:
        interp = 'Low valuation (potentially undervalued)'
    elif ev_ebitda < 14:
        interp = 'Moderate valuation'
    elif ev_ebitda < 20:
        interp = 'High valuation'
    else:
        interp = 'Very high valuation'
    
    return {
        'ev_ebitda': round(ev_ebitda, 2),
        'enterprise_value': enterprise_value,
        'ebitda': ebitda,
        'interpretation': interp
    }


def calculate_ev_sales(
    enterprise_value: float,
    total_revenue: float
) -> Dict[str, Any]:
    """
    Calculate Enterprise Value to Sales ratio.
    
    Formula:
        EV/Sales = Enterprise Value / Total Revenue
    
    Args:
        enterprise_value: Enterprise value
        total_revenue: Total revenue
    
    Returns:
        Dictionary with EV/Sales ratio
    """
    if total_revenue <= 0:
        return {
            'ev_sales': None,
            'interpretation': 'N/A (no revenue)'
        }
    
    ev_sales = enterprise_value / total_revenue
    
    return {
        'ev_sales': round(ev_sales, 2),
        'enterprise_value': enterprise_value,
        'revenue': total_revenue
    }


def calculate_dividend_yield(
    annual_dividend: float,
    stock_price: float
) -> Dict[str, Any]:
    """
    Calculate Dividend Yield.
    
    Formula:
        Dividend Yield = (Annual Dividend / Stock Price) × 100
    
    Args:
        annual_dividend: Annual dividend per share
        stock_price: Current stock price
    
    Returns:
        Dictionary with dividend yield percentage
    """
    if stock_price <= 0:
        return {
            'dividend_yield': None,
            'interpretation': 'N/A'
        }
    
    if annual_dividend <= 0:
        return {
            'dividend_yield': 0,
            'annual_dividend': 0,
            'interpretation': 'No dividend'
        }
    
    yield_pct = (annual_dividend / stock_price) * 100
    
    # Interpretation
    if yield_pct < 2:
        interp = 'Low yield (growth focus)'
    elif yield_pct < 4:
        interp = 'Moderate yield'
    elif yield_pct < 6:
        interp = 'High yield'
    else:
        interp = 'Very high yield (verify sustainability)'
    
    return {
        'dividend_yield': round(yield_pct, 2),
        'annual_dividend': annual_dividend,
        'interpretation': interp
    }


def comprehensive_valuation(company_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate comprehensive valuation metrics from company data.
    
    Args:
        company_data: Dictionary with company financial data
    
    Returns:
        Dictionary with all valuation metrics
    """
    results = {}
    
    stock_price = company_data.get('current_price', 0)
    
    # P/E Ratio
    if eps := company_data.get('eps_trailing'):
        results['pe_trailing'] = calculate_pe_ratio(stock_price, eps, 'trailing')
    
    if eps := company_data.get('eps_forward'):
        results['pe_forward'] = calculate_pe_ratio(stock_price, eps, 'forward')
    
    # PEG Ratio
    if pe := results.get('pe_trailing', {}).get('pe_ratio'):
        if growth := company_data.get('earnings_growth_rate'):
            results['peg'] = calculate_peg_ratio(pe, growth)
    
    # P/B Ratio
    if bv := company_data.get('book_value'):
        results['pb'] = calculate_pb_ratio(stock_price, bv)
    
    # P/S Ratio
    if mc := company_data.get('market_cap'):
        if rev := company_data.get('revenue'):
            results['ps'] = calculate_ps_ratio(mc, rev)
    
    # EV/EBITDA
    if ev := company_data.get('enterprise_value'):
        if ebitda := company_data.get('ebitda'):
            results['ev_ebitda'] = calculate_ev_ebitda(ev, ebitda)
    
    # Dividend Yield
    if div := company_data.get('dividend_rate'):
        results['dividend_yield'] = calculate_dividend_yield(div, stock_price)
    
    return results
