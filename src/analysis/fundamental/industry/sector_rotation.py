"""
Sector Rotation Strategy Module
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List
import yfinance as yf


# Sector ETFs
SECTOR_ETFS = {
    'Technology': 'XLK',
    'Healthcare': 'XLV',
    'Financials': 'XLF',
    'Consumer Discretionary': 'XLY',
    'Consumer Staples': 'XLP',
    'Energy': 'XLE',
    'Utilities': 'XLU',
    'Materials': 'XLB',
    'Industrials': 'XLI',
    'Real Estate': 'XLRE',
    'Communication Services': 'XLC'
}

# Economic cycle sector preferences
ECONOMIC_CYCLE_SECTORS = {
    'early_expansion': ['Technology', 'Consumer Discretionary', 'Industrials', 'Materials'],
    'mid_expansion': ['Technology', 'Communication Services', 'Industrials'],
    'late_expansion': ['Energy', 'Materials', 'Financials'],
    'early_recession': ['Utilities', 'Healthcare', 'Consumer Staples'],
    'late_recession': ['Financials', 'Technology', 'Consumer Discretionary']
}


def get_sector_performance(period: str = '3mo') -> Dict[str, Any]:
    """
    Get sector performance for a given period.
    
    Args:
        period: Time period ('1mo', '3mo', '6mo', '1y')
    
    Returns:
        Dictionary with sector performance data
    """
    performance = {}
    
    for sector, symbol in SECTOR_ETFS.items():
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period)
            
            if not hist.empty:
                start_price = hist['Close'].iloc[0]
                end_price = hist['Close'].iloc[-1]
                return_pct = ((end_price - start_price) / start_price) * 100
                
                # Calculate volatility
                returns = hist['Close'].pct_change().dropna()
                volatility = returns.std() * np.sqrt(252) * 100
                
                performance[sector] = {
                    'symbol': symbol,
                    'return_pct': round(return_pct, 2),
                    'volatility': round(volatility, 2),
                    'sharpe': round(return_pct / volatility if volatility > 0 else 0, 2)
                }
        except Exception as e:
            performance[sector] = {'error': str(e)}
    
    # Sort by performance to get top/worst performers
    valid_sectors = {k: v for k, v in performance.items() if 'return_pct' in v}
    sorted_sectors = sorted(valid_sectors.keys(), key=lambda x: valid_sectors[x]['return_pct'], reverse=True)
    
    return {
        'sectors': performance,
        'top_performers': sorted_sectors[:3] if sorted_sectors else [],
        'worst_performers': sorted_sectors[-3:] if len(sorted_sectors) >= 3 else sorted_sectors
    }


def rank_sectors(performance: Dict[str, Any]) -> List[str]:
    """
    Rank sectors by performance.
    
    Args:
        performance: Sector performance dictionary (can be raw or wrapped in 'sectors' key)
    
    Returns:
        List of sectors ordered by return
    """
    # Handle both old format (direct dict) and new format (wrapped in 'sectors')
    sectors_data = performance.get('sectors', performance)
    valid = {k: v for k, v in sectors_data.items() if isinstance(v, dict) and 'return_pct' in v}
    sorted_sectors = sorted(valid.keys(), key=lambda x: valid[x]['return_pct'], reverse=True)
    return sorted_sectors


def sector_momentum_strategy(
    short_lookback: int = 20,
    long_lookback: int = 60
) -> Dict[str, Any]:
    """
    Sector momentum strategy using relative performance.
    
    Args:
        short_lookback: Short-term lookback days
        long_lookback: Long-term lookback days
    
    Returns:
        Dictionary with momentum signals
    """
    signals = {}
    
    for sector, symbol in SECTOR_ETFS.items():
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period='6mo')
            
            if len(hist) >= long_lookback:
                current = hist['Close'].iloc[-1]
                short_ma = hist['Close'].rolling(short_lookback).mean().iloc[-1]
                long_ma = hist['Close'].rolling(long_lookback).mean().iloc[-1]
                
                # Calculate momentum score
                short_momentum = (current - short_ma) / short_ma * 100
                long_momentum = (current - long_ma) / long_ma * 100
                
                # Combined signal
                if short_momentum > 0 and long_momentum > 0:
                    signal = 'Strong Buy'
                    score = (short_momentum + long_momentum) / 2
                elif short_momentum > 0:
                    signal = 'Buy'
                    score = short_momentum
                elif long_momentum > 0:
                    signal = 'Hold'
                    score = 0
                else:
                    signal = 'Avoid'
                    score = (short_momentum + long_momentum) / 2
                
                signals[sector] = {
                    'symbol': symbol,
                    'signal': signal,
                    'momentum_score': round(score, 2),
                    'short_momentum': round(short_momentum, 2),
                    'long_momentum': round(long_momentum, 2)
                }
        except:
            pass
    
    return signals


def economic_cycle_recommendation(
    cycle_phase: str
) -> Dict[str, Any]:
    """
    Get sector recommendations based on economic cycle.
    
    Args:
        cycle_phase: Current economic cycle phase
            Options: 'early_expansion', 'mid_expansion', 'late_expansion',
                    'early_recession', 'late_recession'
    
    Returns:
        Dictionary with sector recommendations
    """
    recommended = ECONOMIC_CYCLE_SECTORS.get(cycle_phase, [])
    
    all_sectors = list(SECTOR_ETFS.keys())
    avoid = [s for s in all_sectors if s not in recommended]
    
    return {
        'cycle_phase': cycle_phase,
        'recommended_sectors': recommended,
        'neutral_sectors': [],
        'avoid_sectors': avoid,
        'strategy': f"Focus on {', '.join(recommended[:3])} during {cycle_phase.replace('_', ' ')}"
    }


def sector_relative_strength(
    benchmark: str = 'SPY'
) -> Dict[str, Any]:
    """
    Calculate sector relative strength vs benchmark.
    
    Args:
        benchmark: Benchmark ETF symbol
    
    Returns:
        Dictionary with relative strength data
    """
    try:
        # Get benchmark data
        bench = yf.Ticker(benchmark)
        bench_hist = bench.history(period='3mo')
        bench_return = (bench_hist['Close'].iloc[-1] / bench_hist['Close'].iloc[0] - 1) * 100
        
        relative_strength = {}
        
        for sector, symbol in SECTOR_ETFS.items():
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period='3mo')
                
                if not hist.empty:
                    sector_return = (hist['Close'].iloc[-1] / hist['Close'].iloc[0] - 1) * 100
                    rs = sector_return - bench_return
                    
                    relative_strength[sector] = {
                        'sector_return': round(sector_return, 2),
                        'relative_strength': round(rs, 2),
                        'outperforming': rs > 0
                    }
            except:
                pass
        
        # Sort by relative strength
        sorted_rs = dict(sorted(
            relative_strength.items(),
            key=lambda x: x[1]['relative_strength'],
            reverse=True
        ))
        
        return {
            'benchmark': benchmark,
            'benchmark_return': round(bench_return, 2),
            'sectors': sorted_rs,
            'leaders': list(sorted_rs.keys())[:3],
            'laggards': list(sorted_rs.keys())[-3:]
        }
    except Exception as e:
        return {'error': str(e)}
