"""
Comparable Company Analysis (Comps)
"""

import yfinance as yf
import pandas as pd
from typing import Dict, List, Any, Optional


def get_peer_metrics(symbols: List[str]) -> pd.DataFrame:
    """
    Fetch valuation metrics for peer companies.
    
    Args:
        symbols: List of stock symbols to compare
    
    Returns:
        DataFrame with comparative metrics
    """
    metrics = []
    
    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            metrics.append({
                'Symbol': symbol,
                'Name': info.get('shortName', symbol),
                'Market Cap': info.get('marketCap', 0),
                'P/E': info.get('trailingPE'),
                'Forward P/E': info.get('forwardPE'),
                'P/B': info.get('priceToBook'),
                'P/S': info.get('priceToSalesTrailing12Months'),
                'EV/EBITDA': info.get('enterpriseToEbitda'),
                'EV/Revenue': info.get('enterpriseToRevenue'),
                'Revenue': info.get('totalRevenue', 0),
                'Net Margin': (info.get('profitMargins', 0) or 0) * 100,
                'ROE': (info.get('returnOnEquity', 0) or 0) * 100,
                'Debt/Equity': info.get('debtToEquity'),
                'Dividend Yield': (info.get('dividendYield', 0) or 0) * 100
            })
        except Exception as e:
            print(f"Error fetching {symbol}: {e}")
    
    return pd.DataFrame(metrics)


def comparable_analysis(
    target_symbol: str,
    peer_symbols: List[str]
) -> Dict[str, Any]:
    """
    Perform comparable company analysis.
    
    Args:
        target_symbol: The stock to analyze
        peer_symbols: List of peer company symbols
    
    Returns:
        Dictionary with comparison results
    """
    all_symbols = [target_symbol] + peer_symbols
    df = get_peer_metrics(all_symbols)
    
    if df.empty:
        return {'error': 'Could not fetch data'}
    
    # Calculate peer averages (excluding target)
    peer_df = df[df['Symbol'] != target_symbol]
    target_row = df[df['Symbol'] == target_symbol].iloc[0] if target_symbol in df['Symbol'].values else None
    
    if target_row is None:
        return {'error': 'Target symbol not found'}
    
    # Calculate averages and medians
    metrics_to_compare = ['P/E', 'Forward P/E', 'P/B', 'P/S', 'EV/EBITDA', 'Net Margin', 'ROE']
    
    comparison = {}
    valuation_summary = []
    
    for metric in metrics_to_compare:
        target_value = target_row[metric]
        peer_avg = peer_df[metric].mean()
        peer_median = peer_df[metric].median()
        
        if pd.notna(target_value) and pd.notna(peer_avg):
            premium_discount = ((target_value - peer_avg) / peer_avg) * 100 if peer_avg != 0 else 0
            
            comparison[metric] = {
                'target': round(target_value, 2) if pd.notna(target_value) else None,
                'peer_avg': round(peer_avg, 2),
                'peer_median': round(peer_median, 2),
                'premium_discount_pct': round(premium_discount, 1)
            }
            
            # Interpretation
            if metric in ['P/E', 'Forward P/E', 'P/B', 'P/S', 'EV/EBITDA']:
                if premium_discount < -15:
                    valuation_summary.append(f"{metric}: Undervalued by {abs(premium_discount):.1f}%")
                elif premium_discount > 15:
                    valuation_summary.append(f"{metric}: Premium of {premium_discount:.1f}%")
    
    # Calculate implied fair value based on peer multiples
    implied_values = []
    
    if pd.notna(target_row.get('P/E')) and peer_df['P/E'].notna().any():
        eps = target_row.get('Market Cap', 0) / target_row.get('P/E', 1) / 1e9 if target_row.get('P/E') else 0
        if eps > 0:
            implied_pe = eps * peer_df['P/E'].median() * 1e9
            implied_values.append({'method': 'P/E Multiple', 'value': implied_pe})
    
    return {
        'target': {
            'symbol': target_symbol,
            'name': target_row['Name'],
            'market_cap': target_row['Market Cap']
        },
        'peers': peer_df[['Symbol', 'Name', 'Market Cap']].to_dict('records'),
        'comparison': comparison,
        'valuation_summary': valuation_summary,
        'overall_assessment': 'Undervalued' if len([s for s in valuation_summary if 'Undervalued' in s]) > len([s for s in valuation_summary if 'Premium' in s]) else 'Premium Valuation'
    }


def sector_peers(symbol: str, limit: int = 5) -> List[str]:
    """
    Find peer companies in the same sector.
    
    Args:
        symbol: Stock symbol
        limit: Maximum number of peers
    
    Returns:
        List of peer symbols
    """
    # Common sector peer mappings
    sector_peers_map = {
        'Technology': ['AAPL', 'MSFT', 'GOOGL', 'META', 'NVDA', 'ADBE', 'CRM', 'ORCL'],
        'Financials': ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'USB', 'PNC'],
        'Healthcare': ['JNJ', 'UNH', 'PFE', 'ABBV', 'MRK', 'LLY', 'TMO', 'ABT'],
        'Consumer': ['AMZN', 'WMT', 'HD', 'MCD', 'NKE', 'SBUX', 'TGT', 'COST'],
        'Energy': ['XOM', 'CVX', 'COP', 'SLB', 'EOG', 'PXD', 'MPC', 'PSX']
    }
    
    try:
        ticker = yf.Ticker(symbol)
        sector = ticker.info.get('sector', '')
        
        for key, peers in sector_peers_map.items():
            if key.lower() in sector.lower():
                # Return peers excluding the target
                return [p for p in peers if p != symbol][:limit]
        
        # Default to tech giants if sector not found
        return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META'][:limit]
        
    except:
        return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META'][:limit]
