"""
Alternative Data Sources Module

Provides additional data signals beyond traditional price/volume:
- Insider trading activity
- Institutional holdings
- Options flow indicators
- Social sentiment (VADER)
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import yfinance as yf


def get_insider_transactions(symbol: str) -> Dict[str, Any]:
    """
    Get insider trading activity from yfinance.
    
    Args:
        symbol: Stock ticker
    
    Returns:
        Insider transaction summary and signal
    """
    try:
        ticker = yf.Ticker(symbol)
        
        # Get insider transactions
        insider_purchases = ticker.insider_purchases
        insider_roster = ticker.insider_roster_holders
        
        # Analyze purchases
        net_insider_value = 0
        recent_transactions = []
        buy_count = 0
        sell_count = 0
        
        if insider_purchases is not None and not insider_purchases.empty:
            # Parse insider data
            for _, row in insider_purchases.head(10).iterrows():
                shares = row.get('Shares', 0) or 0
                value = row.get('Value', 0) or 0
                
                if shares > 0:
                    buy_count += 1
                    net_insider_value += value
                elif shares < 0:
                    sell_count += 1
                    net_insider_value -= abs(value)
                
                recent_transactions.append({
                    "position": str(row.get('Position', 'Unknown')),
                    "shares": int(shares) if shares else 0,
                    "value": float(value) if value else 0
                })
        
        # Determine signal
        if buy_count > sell_count * 2:
            signal = "STRONG_BUY"
            interpretation = "Heavy insider buying - bullish signal"
        elif buy_count > sell_count:
            signal = "BUY"
            interpretation = "Net insider buying detected"
        elif sell_count > buy_count * 2:
            signal = "STRONG_SELL"
            interpretation = "Heavy insider selling - bearish signal"
        elif sell_count > buy_count:
            signal = "SELL"
            interpretation = "Net insider selling detected"
        else:
            signal = "NEUTRAL"
            interpretation = "Balanced insider activity"
        
        return {
            "available": True,
            "buy_transactions": buy_count,
            "sell_transactions": sell_count,
            "net_value": round(net_insider_value, 2),
            "signal": signal,
            "interpretation": interpretation,
            "recent_transactions": recent_transactions[:5],
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            "available": False,
            "signal": "NEUTRAL",
            "interpretation": "Insider data unavailable",
            "error": str(e)
        }


def get_institutional_holdings(symbol: str) -> Dict[str, Any]:
    """
    Get institutional ownership data.
    
    Args:
        symbol: Stock ticker
    
    Returns:
        Institutional holdings summary
    """
    try:
        ticker = yf.Ticker(symbol)
        
        # Get institutional holders
        inst_holders = ticker.institutional_holders
        major_holders = ticker.major_holders
        
        # Parse major holders percentage
        inst_pct = 0.0
        insider_pct = 0.0
        
        if major_holders is not None and not major_holders.empty:
            for _, row in major_holders.iterrows():
                value = row.iloc[0] if len(row) > 0 else ""
                desc = str(row.iloc[1]) if len(row) > 1 else ""
                
                if "institution" in desc.lower():
                    try:
                        inst_pct = float(str(value).replace('%', ''))
                    except:
                        pass
                elif "insider" in desc.lower():
                    try:
                        insider_pct = float(str(value).replace('%', ''))
                    except:
                        pass
        
        # Top holders
        top_holders = []
        if inst_holders is not None and not inst_holders.empty:
            for _, row in inst_holders.head(5).iterrows():
                top_holders.append({
                    "holder": str(row.get('Holder', 'Unknown')),
                    "shares": int(row.get('Shares', 0) or 0),
                    "pct_out": float(row.get('% Out', 0) or 0)
                })
        
        # Determine signal based on institutional ownership
        if inst_pct > 80:
            signal = "HIGH_INST"
            interpretation = "Very high institutional ownership - stable but limited upside"
        elif inst_pct > 60:
            signal = "GOOD_INST"
            interpretation = "Healthy institutional presence - quality validation"
        elif inst_pct > 30:
            signal = "MODERATE_INST"
            interpretation = "Moderate institutional interest"
        elif inst_pct > 10:
            signal = "LOW_INST"
            interpretation = "Low institutional ownership - higher volatility risk"
        else:
            signal = "RETAIL"
            interpretation = "Retail-dominated - can be volatile"
        
        return {
            "available": True,
            "institutional_pct": round(inst_pct, 2),
            "insider_pct": round(insider_pct, 2),
            "public_float_pct": round(100 - inst_pct - insider_pct, 2),
            "top_holders": top_holders,
            "holder_count": len(top_holders),
            "signal": signal,
            "interpretation": interpretation,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            "available": False,
            "signal": "UNKNOWN",
            "interpretation": "Institutional data unavailable",
            "error": str(e)
        }


def get_options_flow_indicator(symbol: str) -> Dict[str, Any]:
    """
    Get options flow indicator (put/call ratio proxy).
    
    Args:
        symbol: Stock ticker
    
    Returns:
        Options flow indicator
    """
    try:
        ticker = yf.Ticker(symbol)
        
        # Get options expiration dates
        expirations = ticker.options
        
        if not expirations:
            return {
                "available": False,
                "signal": "NEUTRAL",
                "interpretation": "No options data available"
            }
        
        # Get nearest expiration
        nearest_exp = expirations[0]
        opt_chain = ticker.option_chain(nearest_exp)
        
        calls = opt_chain.calls
        puts = opt_chain.puts
        
        # Calculate volumes
        call_volume = calls['volume'].sum() if 'volume' in calls.columns else 0
        put_volume = puts['volume'].sum() if 'volume' in puts.columns else 0
        
        call_oi = calls['openInterest'].sum() if 'openInterest' in calls.columns else 0
        put_oi = puts['openInterest'].sum() if 'openInterest' in puts.columns else 0
        
        # Put/Call ratio
        if call_volume > 0:
            pc_ratio_volume = put_volume / call_volume
        else:
            pc_ratio_volume = 1.0
        
        if call_oi > 0:
            pc_ratio_oi = put_oi / call_oi
        else:
            pc_ratio_oi = 1.0
        
        # Determine signal
        # High put/call = bearish sentiment (but can be contrarian bullish)
        if pc_ratio_volume < 0.5:
            signal = "BULLISH"
            interpretation = "Low put/call ratio - bullish options flow"
        elif pc_ratio_volume > 1.5:
            signal = "BEARISH"
            interpretation = "High put/call ratio - bearish hedging activity"
        elif pc_ratio_volume > 1.0:
            signal = "CAUTIOUS"
            interpretation = "Elevated puts - some hedging activity"
        else:
            signal = "NEUTRAL"
            interpretation = "Balanced options activity"
        
        return {
            "available": True,
            "put_call_ratio_volume": round(pc_ratio_volume, 2),
            "put_call_ratio_oi": round(pc_ratio_oi, 2),
            "call_volume": int(call_volume) if not np.isnan(call_volume) else 0,
            "put_volume": int(put_volume) if not np.isnan(put_volume) else 0,
            "call_open_interest": int(call_oi) if not np.isnan(call_oi) else 0,
            "put_open_interest": int(put_oi) if not np.isnan(put_oi) else 0,
            "expiration": nearest_exp,
            "signal": signal,
            "interpretation": interpretation,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            "available": False,
            "signal": "NEUTRAL",
            "interpretation": "Options data unavailable",
            "error": str(e)
        }


def analyze_sentiment_vader(texts: List[str]) -> Dict[str, Any]:
    """
    Analyze sentiment using VADER (Valence Aware Dictionary and sEntiment Reasoner).
    
    Args:
        texts: List of text strings to analyze
    
    Returns:
        Aggregated sentiment analysis
    """
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        
        analyzer = SentimentIntensityAnalyzer()
        
        scores = []
        sentiments = []
        
        for text in texts:
            if not text:
                continue
            
            vs = analyzer.polarity_scores(text)
            scores.append(vs['compound'])
            
            # Classify
            if vs['compound'] >= 0.05:
                sentiments.append('positive')
            elif vs['compound'] <= -0.05:
                sentiments.append('negative')
            else:
                sentiments.append('neutral')
        
        if not scores:
            return {
                "available": True,
                "method": "vader",
                "sentiment_score": 0,
                "sentiment": "neutral",
                "sample_size": 0
            }
        
        avg_score = np.mean(scores)
        positive_count = sentiments.count('positive')
        negative_count = sentiments.count('negative')
        neutral_count = sentiments.count('neutral')
        
        # Overall sentiment
        if avg_score >= 0.15:
            overall = "bullish"
        elif avg_score >= 0.05:
            overall = "slightly_bullish"
        elif avg_score <= -0.15:
            overall = "bearish"
        elif avg_score <= -0.05:
            overall = "slightly_bearish"
        else:
            overall = "neutral"
        
        return {
            "available": True,
            "method": "vader",
            "sentiment_score": round(avg_score, 3),
            "sentiment": overall,
            "positive_count": positive_count,
            "negative_count": negative_count,
            "neutral_count": neutral_count,
            "sample_size": len(scores),
            "bullish_ratio": round(positive_count / len(scores) * 100, 1) if scores else 0
        }
        
    except ImportError:
        return {
            "available": False,
            "method": "vader",
            "sentiment_score": 0,
            "sentiment": "neutral",
            "error": "VADER not installed. Run: pip install vaderSentiment"
        }
    except Exception as e:
        return {
            "available": False,
            "method": "vader",
            "sentiment_score": 0,
            "sentiment": "neutral",
            "error": str(e)
        }


def get_social_sentiment(symbol: str, news_headlines: List[str] = None) -> Dict[str, Any]:
    """
    Get social sentiment for a stock using VADER.
    
    Args:
        symbol: Stock ticker
        news_headlines: Optional list of recent news headlines
    
    Returns:
        Social sentiment analysis
    """
    # If no headlines provided, try to get from yfinance
    if not news_headlines:
        try:
            ticker = yf.Ticker(symbol)
            news = ticker.news
            
            if news:
                news_headlines = []
                for item in news[:20]:
                    # Handle new yfinance structure: title is in content.title
                    title = None
                    if isinstance(item, dict):
                        # New structure: {'id': ..., 'content': {'title': ...}}
                        content = item.get('content', {})
                        if isinstance(content, dict):
                            title = content.get('title')
                        # Fallback to old structure
                        if not title:
                            title = item.get('title', '')
                    
                    if title:
                        news_headlines.append(title)
        except Exception as e:
            print(f"News fetch error: {e}")
            news_headlines = []
    
    if not news_headlines:
        return {
            "available": False,
            "sentiment_score": 0,
            "sentiment": "neutral",
            "interpretation": "No news data available for sentiment analysis"
        }
    
    # Analyze with VADER
    sentiment = analyze_sentiment_vader(news_headlines)
    
    # Add interpretation
    score = sentiment.get("sentiment_score", 0)
    
    if score >= 0.2:
        interpretation = "Strong positive sentiment in news coverage"
        signal = "BULLISH"
    elif score >= 0.05:
        interpretation = "Mild positive sentiment"
        signal = "SLIGHTLY_BULLISH"
    elif score <= -0.2:
        interpretation = "Strong negative sentiment - caution advised"
        signal = "BEARISH"
    elif score <= -0.05:
        interpretation = "Mild negative sentiment"
        signal = "SLIGHTLY_BEARISH"
    else:
        interpretation = "Neutral news sentiment"
        signal = "NEUTRAL"
    
    sentiment["signal"] = signal
    sentiment["interpretation"] = interpretation
    sentiment["headline_count"] = len(news_headlines)
    sentiment["timestamp"] = datetime.now().isoformat()
    
    return sentiment


def get_all_alternative_data(symbol: str, news_headlines: List[str] = None) -> Dict[str, Any]:
    """
    Get all alternative data for a stock.
    
    Args:
        symbol: Stock ticker
        news_headlines: Optional news headlines for sentiment
    
    Returns:
        Comprehensive alternative data
    """
    return {
        "symbol": symbol,
        "insider_activity": get_insider_transactions(symbol),
        "institutional_holdings": get_institutional_holdings(symbol),
        "options_flow": get_options_flow_indicator(symbol),
        "social_sentiment": get_social_sentiment(symbol, news_headlines),
        "timestamp": datetime.now().isoformat()
    }


def get_alternative_data_signal(alt_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Aggregate alternative data into a unified signal.
    
    Args:
        alt_data: Output from get_all_alternative_data
    
    Returns:
        Unified alternative data signal
    """
    signals = {
        "STRONG_BUY": 2,
        "BUY": 1,
        "BULLISH": 1,
        "SLIGHTLY_BULLISH": 0.5,
        "NEUTRAL": 0,
        "SLIGHTLY_BEARISH": -0.5,
        "CAUTIOUS": -0.5,
        "BEARISH": -1,
        "SELL": -1,
        "STRONG_SELL": -2
    }
    
    total_score = 0
    components = 0
    
    # Insider activity
    insider = alt_data.get("insider_activity", {})
    if insider.get("available"):
        signal = insider.get("signal", "NEUTRAL")
        total_score += signals.get(signal, 0)
        components += 1
    
    # Options flow
    options = alt_data.get("options_flow", {})
    if options.get("available"):
        signal = options.get("signal", "NEUTRAL")
        total_score += signals.get(signal, 0)
        components += 1
    
    # Social sentiment
    sentiment = alt_data.get("social_sentiment", {})
    if sentiment.get("available"):
        signal = sentiment.get("signal", "NEUTRAL")
        total_score += signals.get(signal, 0)
        components += 1
    
    # Calculate average score
    if components > 0:
        avg_score = total_score / components
    else:
        avg_score = 0
    
    # Determine overall signal
    if avg_score >= 1.5:
        overall_signal = "STRONG_BUY"
    elif avg_score >= 0.5:
        overall_signal = "BUY"
    elif avg_score <= -1.5:
        overall_signal = "STRONG_SELL"
    elif avg_score <= -0.5:
        overall_signal = "SELL"
    else:
        overall_signal = "NEUTRAL"
    
    return {
        "overall_signal": overall_signal,
        "score": round(avg_score, 2),
        "components_used": components,
        "insider_signal": insider.get("signal", "N/A"),
        "options_signal": options.get("signal", "N/A"),
        "sentiment_signal": sentiment.get("signal", "N/A"),
        "confidence": round(min(components / 3 * 100, 100), 0)
    }
