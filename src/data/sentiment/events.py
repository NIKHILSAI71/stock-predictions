"""
Event-Driven Analytics Module
Earnings surprises, dividend changes, and corporate catalyst tracking.
"""

from typing import Dict, Any, List, Optional
import yfinance as yf
from datetime import datetime, timedelta


def earnings_surprise_analysis(symbol: str) -> Dict[str, Any]:
    """
    Analyze earnings surprises (actual vs expected EPS).
    
    Earnings surprises can cause significant price movements.
    
    Args:
        symbol: Stock ticker symbol
    
    Returns:
        Dictionary with earnings surprise history and analysis
    """
    try:
        ticker = yf.Ticker(symbol)
        
        # Get earnings history
        earnings = ticker.earnings_history
        
        if earnings is None or earnings.empty:
            # Fallback to basic info
            info = ticker.info
            return {
                'symbol': symbol,
                'trailing_eps': info.get('trailingEps'),
                'forward_eps': info.get('forwardEps'),
                'note': 'Detailed earnings history not available'
            }
        
        # Analyze surprises
        surprises = []
        for idx, row in earnings.iterrows():
            actual = row.get('epsActual', 0)
            estimate = row.get('epsEstimate', 0)
            
            if estimate and estimate != 0:
                surprise_pct = ((actual - estimate) / abs(estimate)) * 100
                
                surprises.append({
                    'date': str(idx),
                    'actual_eps': actual,
                    'estimated_eps': estimate,
                    'surprise_pct': round(surprise_pct, 2),
                    'beat': actual > estimate
                })
        
        # Calculate statistics
        if surprises:
            beat_rate = sum(1 for s in surprises if s['beat']) / len(surprises) * 100
            avg_surprise = sum(s['surprise_pct'] for s in surprises) / len(surprises)
        else:
            beat_rate = 0
            avg_surprise = 0
        
        # Recent trend
        recent_trend = 'Positive' if avg_surprise > 0 else 'Negative' if avg_surprise < 0 else 'Neutral'
        
        return {
            'symbol': symbol,
            'history': surprises[-4:] if len(surprises) > 4 else surprises,  # Last 4 quarters
            'statistics': {
                'beat_rate': round(beat_rate, 1),
                'average_surprise': round(avg_surprise, 2),
                'recent_trend': recent_trend
            },
            'signal': 'Positive' if beat_rate > 75 else 'Neutral' if beat_rate > 50 else 'Negative'
        }
        
    except Exception as e:
        return {'error': str(e), 'symbol': symbol}


def dividend_tracking(symbol: str) -> Dict[str, Any]:
    """
    Track dividend changes and announcements.
    
    Dividend increases are bullish signals; cuts are bearish.
    
    Args:
        symbol: Stock ticker symbol
    
    Returns:
        Dictionary with dividend analysis
    """
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        # Current dividend info
        current_yield = info.get('dividendYield', 0) or 0
        dividend_rate = info.get('dividendRate', 0) or 0
        payout_ratio = info.get('payoutRatio', 0) or 0
        
        if dividend_rate == 0:
            return {
                'symbol': symbol,
                'pays_dividend': False,
                'note': 'Company does not currently pay dividends'
            }
        
        # Get dividend history
        dividends = ticker.dividends
        
        if dividends is None or dividends.empty:
            return {
                'symbol': symbol,
                'pays_dividend': True,
                'current_yield': round(current_yield * 100, 2),
                'annual_rate': dividend_rate,
                'history': 'Not available'
            }
        
        # Analyze dividend changes
        recent_dividends = dividends.tail(8)  # Last 2 years quarterly
        
        changes = []
        prev_dividend = None
        
        for date, amount in recent_dividends.items():
            if prev_dividend is not None and prev_dividend != 0:
                change_pct = ((amount - prev_dividend) / prev_dividend) * 100
                if abs(change_pct) > 1:  # Significant change
                    changes.append({
                        'date': str(date.date()),
                        'from': prev_dividend,
                        'to': amount,
                        'change_pct': round(change_pct, 2)
                    })
            prev_dividend = amount
        
        # Dividend growth
        if len(recent_dividends) >= 4:
            first_div = recent_dividends.iloc[0]
            last_div = recent_dividends.iloc[-1]
            if first_div > 0:
                total_growth = ((last_div - first_div) / first_div) * 100
            else:
                total_growth = 0
        else:
            total_growth = 0
        
        # Sustainability
        if payout_ratio > 0.9:
            sustainability = 'At Risk'
            sustainability_note = 'Payout ratio above 90% may not be sustainable'
        elif payout_ratio > 0.7:
            sustainability = 'Moderate'
            sustainability_note = 'Payout ratio is elevated'
        elif payout_ratio > 0:
            sustainability = 'Healthy'
            sustainability_note = 'Payout ratio indicates sustainable dividend'
        else:
            sustainability = 'Unknown'
            sustainability_note = 'Payout ratio not available'
        
        return {
            'symbol': symbol,
            'pays_dividend': True,
            'current_yield': round(current_yield * 100, 2),
            'annual_rate': dividend_rate,
            'payout_ratio': round(payout_ratio * 100, 2),
            'recent_changes': changes,
            'growth': {
                'total_growth_pct': round(total_growth, 2),
                'trend': 'Growing' if total_growth > 5 else 'Stable' if total_growth > -5 else 'Declining'
            },
            'sustainability': {
                'rating': sustainability,
                'note': sustainability_note
            }
        }
        
    except Exception as e:
        return {'error': str(e), 'symbol': symbol}


def analyst_estimates(symbol: str) -> Dict[str, Any]:
    """
    Track analyst estimate revisions.
    
    Rising estimates are bullish; falling estimates are bearish.
    
    Args:
        symbol: Stock ticker symbol
    
    Returns:
        Dictionary with analyst estimates and revisions
    """
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        # Current estimates
        current_price = info.get('currentPrice', 0) or info.get('regularMarketPrice', 0)
        target_mean = info.get('targetMeanPrice', 0)
        target_high = info.get('targetHighPrice', 0)
        target_low = info.get('targetLowPrice', 0)
        num_analysts = info.get('numberOfAnalystOpinions', 0)
        recommendation = info.get('recommendationKey', 'none')
        
        # Calculate upside/downside
        if current_price and target_mean:
            upside = ((target_mean - current_price) / current_price) * 100
        else:
            upside = 0
        
        # EPS estimates
        eps_trailing = info.get('trailingEps', 0)
        eps_forward = info.get('forwardEps', 0)
        
        if eps_trailing and eps_forward and eps_trailing > 0:
            eps_growth = ((eps_forward - eps_trailing) / abs(eps_trailing)) * 100
        else:
            eps_growth = 0
        
        return {
            'symbol': symbol,
            'current_price': current_price,
            'analyst_targets': {
                'mean': target_mean,
                'high': target_high,
                'low': target_low,
                'upside_pct': round(upside, 2)
            },
            'coverage': {
                'num_analysts': num_analysts,
                'recommendation': recommendation.upper() if recommendation else 'N/A'
            },
            'eps_estimates': {
                'trailing': eps_trailing,
                'forward': eps_forward,
                'expected_growth': round(eps_growth, 2)
            },
            'signal': 'Bullish' if upside > 20 else 'Neutral' if upside > 0 else 'Bearish'
        }
        
    except Exception as e:
        return {'error': str(e), 'symbol': symbol}


def insider_activity(symbol: str) -> Dict[str, Any]:
    """
    Track insider trading activity.
    
    Insider buying is generally bullish; heavy selling is bearish.
    
    Args:
        symbol: Stock ticker symbol
    
    Returns:
        Dictionary with insider activity summary
    """
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        # Get insider transactions
        insider_trans = ticker.insider_transactions
        
        insider_pct = info.get('heldPercentInsiders', 0) or 0
        
        if insider_trans is None or insider_trans.empty:
            return {
                'symbol': symbol,
                'insider_ownership_pct': round(insider_pct * 100, 2),
                'recent_transactions': 'Not available',
                'note': 'Use SEC filings for detailed insider activity'
            }
        
        # Analyze recent transactions
        recent = insider_trans.head(10)
        
        buys = 0
        sells = 0
        buy_value = 0
        sell_value = 0
        
        transactions = []
        
        for _, row in recent.iterrows():
            trans_type = str(row.get('Text', '')).lower()
            shares = row.get('Shares', 0) or 0
            value = row.get('Value', 0) or 0
            
            if 'buy' in trans_type or 'purchase' in trans_type:
                buys += 1
                buy_value += value if value else 0
                action = 'BUY'
            elif 'sell' in trans_type or 'sale' in trans_type:
                sells += 1
                sell_value += value if value else 0
                action = 'SELL'
            else:
                action = 'OTHER'
            
            transactions.append({
                'insider': row.get('Insider', 'Unknown'),
                'action': action,
                'shares': shares,
                'value': value
            })
        
        # Determine signal
        if buys > sells * 2:
            signal = 'Strong Insider Buying'
        elif buys > sells:
            signal = 'Net Insider Buying'
        elif sells > buys * 2:
            signal = 'Heavy Insider Selling'
        elif sells > buys:
            signal = 'Net Insider Selling'
        else:
            signal = 'Neutral'
        
        return {
            'symbol': symbol,
            'insider_ownership_pct': round(insider_pct * 100, 2),
            'recent_activity': {
                'buys': buys,
                'sells': sells,
                'buy_value': buy_value,
                'sell_value': sell_value
            },
            'transactions': transactions[:5],  # Top 5
            'signal': signal
        }
        
    except Exception as e:
        return {'error': str(e), 'symbol': symbol}


def corporate_events_summary(symbol: str) -> Dict[str, Any]:
    """
    Comprehensive corporate events and catalysts summary.
    
    Args:
        symbol: Stock ticker symbol
    
    Returns:
        Dictionary with all event tracking data
    """
    return {
        'symbol': symbol,
        'earnings': earnings_surprise_analysis(symbol),
        'dividends': dividend_tracking(symbol),
        'analyst_estimates': analyst_estimates(symbol),
        'insider_activity': insider_activity(symbol),
        'generated_at': datetime.now().isoformat()
    }
