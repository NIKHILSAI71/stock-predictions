# Sentiment Analysis Module

# Legacy exports (backward compatibility)
from .news import (
    analyze_text_sentiment, aggregate_sentiment_scores
)
from .market_indicators import (
    get_vix_data, fear_greed_indicator
)
from .events import (
    earnings_surprise_analysis, dividend_tracking, analyst_estimates,
    insider_activity, corporate_events_summary
)

# New sentiment engine (main interface)
from .sentiment_engine import get_sentiment_engine, SentimentEngine

# Engines
from .engines import (
    VADEREngine, FinBERTEngine, RoBERTaEngine, TextBlobEngine, GeminiEngine
)

# Aggregators
from .aggregators.ensemble_aggregator import get_ensemble_aggregator
from .aggregators.controversy_detector import get_controversy_detector

# Analyzers
from .analyzers.aspect_analyzer import get_aspect_analyzer
from .analyzers.entity_detector import get_entity_detector
from .analyzers.event_classifier import get_event_classifier

# Sources
from .sources.news_source import get_news_source
from .sources.twitter_source import get_twitter_source
from .sources.reddit_source import get_reddit_source
from .sources.stocktwits_source import get_stocktwits_source

# Storage
from .storage.sentiment_store import get_sentiment_store
from .storage.cache_manager import get_cache_manager


__all__ = [
    # Legacy
    'analyze_text_sentiment', 'aggregate_sentiment_scores',
    'get_vix_data', 'fear_greed_indicator',
    'earnings_surprise_analysis', 'dividend_tracking', 'analyst_estimates',
    'insider_activity', 'corporate_events_summary',

    # New main interface
    'get_sentiment_engine', 'SentimentEngine',

    # Engines
    'VADEREngine', 'FinBERTEngine', 'RoBERTaEngine', 'TextBlobEngine', 'GeminiEngine',

    # Aggregators
    'get_ensemble_aggregator', 'get_controversy_detector',

    # Analyzers
    'get_aspect_analyzer', 'get_entity_detector', 'get_event_classifier',

    # Sources
    'get_news_source', 'get_twitter_source', 'get_reddit_source', 'get_stocktwits_source',

    # Storage
    'get_sentiment_store', 'get_cache_manager'
]
