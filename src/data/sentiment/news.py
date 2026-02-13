"""
News Sentiment Analysis Module
"""

import requests
import re
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

# Keywords for categorization
CATEGORY_KEYWORDS = {
    'earnings': ['earnings', 'quarterly', 'eps', 'revenue', 'guidance', 'beat', 'miss', 'profit', 'loss'],
    'm&a': ['merger', 'acquisition', 'buyout', 'acquire', 'takeover', 'deal', 'transaction'],
    'regulatory/fda': ['fda', 'approval', 'clinical trial', 'sec', 'lawsuit', 'investigation', 'fine', 'regulation'],
    'macro': ['fed', 'interest rate', 'inflation', 'cpi', 'gdp', 'economic', 'market-wide'],
    'product': ['launch', 'new product', 'innovation', 'update', 'patent']
}

# Severity weights by category
SEVERITY_WEIGHTS = {
    'earnings': 2.0,
    'm&a': 2.5,
    'regulatory/fda': 1.8,
    'macro': 1.2,
    'product': 1.3,
    'general': 1.0
}


def categorize_news(text: str) -> str:
    """Categorize news based on keyword matching."""
    text_lower = text.lower()
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(keyword in text_lower for keyword in keywords):
            return category
    return 'general'


def analyze_text_sentiment(text: str) -> Dict[str, Any]:
    """
    Analyze sentiment of a text using VADER with category-based severity.
    """
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

        analyzer = SentimentIntensityAnalyzer()
        scores = analyzer.polarity_scores(text)
        polarity = scores['compound']

        category = categorize_news(text)
        severity = SEVERITY_WEIGHTS.get(category, 1.0)

        subjectivity = 0.5

        if polarity >= 0.05:
            sentiment = 'positive'
        elif polarity <= -0.05:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'

        return {
            'polarity': round(polarity, 3),
            'subjectivity': round(subjectivity, 3),
            'sentiment': sentiment,
            'category': category,
            'severity': severity,
            'weighted_polarity': round(polarity * severity, 3),
            'details': scores
        }
    except ImportError:
        return {
            'polarity': 0,
            'subjectivity': 0,
            'sentiment': 'neutral',
            'category': 'general',
            'severity': 1.0,
            'error': 'vaderSentiment not installed'
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
    weighted_polarities = [s.get('weighted_polarity', s['polarity'])
                           for s in sentiments if 'polarity' in s]
    total_weights = sum(s.get('severity', 1.0) for s in sentiments)

    avg_polarity = sum(polarities) / len(polarities) if polarities else 0
    weighted_avg_polarity = sum(weighted_polarities) / \
        total_weights if total_weights > 0 else avg_polarity

    avg_subjectivity = sum(s['subjectivity']
                           for s in sentiments if 'subjectivity' in s) / len(sentiments) if sentiments else 0

    positive_count = sum(
        1 for s in sentiments if s.get('sentiment') == 'positive')
    negative_count = sum(
        1 for s in sentiments if s.get('sentiment') == 'negative')
    neutral_count = sum(
        1 for s in sentiments if s.get('sentiment') == 'neutral')

    # Track categories for diagnostic info
    categories = {}
    for s in sentiments:
        cat = s.get('category', 'general')
        categories[cat] = categories.get(cat, 0) + 1

    # Determine overall sentiment based on WEIGHTED polarity
    if weighted_avg_polarity >= 0.05:
        overall = 'bullish'
    elif weighted_avg_polarity <= -0.05:
        overall = 'bearish'
    else:
        overall = 'neutral'

    return {
        'avg_polarity': round(avg_polarity, 3),
        'weighted_avg_polarity': round(weighted_avg_polarity, 3),
        'avg_subjectivity': round(avg_subjectivity, 3),
        'overall_sentiment': overall,
        'positive_count': positive_count,
        'negative_count': negative_count,
        'neutral_count': neutral_count,
        'total_count': len(sentiments),
        'categories': categories,
        'bullish_ratio': round(positive_count / len(sentiments) * 100, 1) if sentiments else 0
    }
