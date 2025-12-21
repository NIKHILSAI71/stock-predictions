"""
News Sentiment Analysis Module
"""

import requests
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta


def analyze_text_sentiment(text: str) -> Dict[str, Any]:
    """
    Analyze sentiment of a text using TextBlob.
    
    Args:
        text: Text to analyze
    
    Returns:
        Dictionary with polarity and subjectivity scores
    """
    try:
        from textblob import TextBlob
        
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity  # -1 to 1
        subjectivity = blob.sentiment.subjectivity  # 0 to 1
        
        # Classify sentiment
        if polarity > 0.1:
            sentiment = 'positive'
        elif polarity < -0.1:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
        
        return {
            'polarity': round(polarity, 3),
            'subjectivity': round(subjectivity, 3),
            'sentiment': sentiment
        }
    except ImportError:
        return {
            'polarity': 0,
            'subjectivity': 0,
            'sentiment': 'neutral',
            'error': 'TextBlob not installed'
        }


def aggregate_sentiment_scores(
    sentiments: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Aggregate multiple sentiment scores into overall sentiment.
    
    Args:
        sentiments: List of sentiment dictionaries
    
    Returns:
        Aggregated sentiment scores
    """
    if not sentiments:
        return {
            'avg_polarity': 0,
            'avg_subjectivity': 0,
            'overall_sentiment': 'neutral',
            'positive_count': 0,
            'negative_count': 0,
            'neutral_count': 0,
            'total_count': 0
        }
    
    polarities = [s['polarity'] for s in sentiments if 'polarity' in s]
    subjectivities = [s['subjectivity'] for s in sentiments if 'subjectivity' in s]
    
    avg_polarity = sum(polarities) / len(polarities) if polarities else 0
    avg_subjectivity = sum(subjectivities) / len(subjectivities) if subjectivities else 0
    
    positive_count = sum(1 for s in sentiments if s.get('sentiment') == 'positive')
    negative_count = sum(1 for s in sentiments if s.get('sentiment') == 'negative')
    neutral_count = sum(1 for s in sentiments if s.get('sentiment') == 'neutral')
    
    # Determine overall sentiment
    if avg_polarity > 0.1:
        overall = 'bullish'
    elif avg_polarity < -0.1:
        overall = 'bearish'
    else:
        overall = 'neutral'
    
    return {
        'avg_polarity': round(avg_polarity, 3),
        'avg_subjectivity': round(avg_subjectivity, 3),
        'overall_sentiment': overall,
        'positive_count': positive_count,
        'negative_count': negative_count,
        'neutral_count': neutral_count,
        'total_count': len(sentiments),
        'bullish_ratio': round(positive_count / len(sentiments) * 100, 1) if sentiments else 0
    }


def sentiment_score_to_signal(
    sentiment_score: float,
    threshold_buy: float = 0.2,
    threshold_sell: float = -0.2
) -> str:
    """
    Convert sentiment score to trading signal.
    
    Args:
        sentiment_score: Polarity score (-1 to 1)
        threshold_buy: Threshold for buy signal
        threshold_sell: Threshold for sell signal
    
    Returns:
        Trading signal: 'buy', 'sell', or 'hold'
    """
    if sentiment_score >= threshold_buy:
        return 'buy'
    elif sentiment_score <= threshold_sell:
        return 'sell'
    else:
        return 'hold'
