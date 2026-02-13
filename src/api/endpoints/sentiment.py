"""
Sentiment Analysis API Endpoints
New endpoints for comprehensive sentiment analysis.
"""

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse, StreamingResponse
from typing import Optional
import logging
import asyncio
import json

from src.data.sentiment import get_sentiment_engine
from src.data.sentiment.storage import get_sentiment_store
from src.data.sentiment.aggregators import get_controversy_detector

logger = logging.getLogger("uvicorn.info")

router = APIRouter()


@router.get("/sentiment/analyze/{symbol}")
async def analyze_sentiment(
    symbol: str,
    timeframe: str = Query("1day", regex="^(1hr|1day|1week|1month)$"),
    sources: str = Query("all", regex="^(all|news|social|analyst)$")
):
    """
    Get comprehensive sentiment analysis for a symbol.

    Args:
        symbol: Stock symbol (e.g., 'TSLA')
        timeframe: Time period ('1hr', '1day', '1week', '1month')
        sources: Data sources ('all', 'news', 'social', 'analyst')

    Returns:
        Complete sentiment analysis with:
        - Overall sentiment (label, polarity, confidence)
        - Temporal analysis (1hr, 1day, 1week, 1month trends)
        - Aspect breakdown (product, management, financials, etc.)
        - Model scores (individual engine results)
        - Sources breakdown
        - Events detected
        - Entities extracted
        - Controversy analysis
    """
    try:
        engine = get_sentiment_engine()

        result = await engine.analyze(
            symbol=symbol,
            timeframe=timeframe,
            sources=sources
        )

        return JSONResponse(content=result)

    except Exception as e:
        logger.error(f"Sentiment analysis error for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sentiment/stream/{symbol}")
async def stream_sentiment(symbol: str):
    """
    Real-time sentiment updates via Server-Sent Events (SSE).

    Args:
        symbol: Stock symbol

    Returns:
        SSE stream with sentiment updates every 60 seconds
    """
    async def event_generator():
        engine = get_sentiment_engine()

        while True:
            try:
                # Fetch latest sentiment
                result = await engine.analyze(symbol, timeframe="1hr", sources="all")

                # Format as SSE
                data = {
                    'type': 'sentiment_update',
                    'data': result
                }

                yield f"data: {json.dumps(data)}\n\n"

                # Wait 60 seconds before next update
                await asyncio.sleep(60)

            except Exception as e:
                logger.error(f"Stream error for {symbol}: {e}")
                error_data = {
                    'type': 'error',
                    'message': str(e)
                }
                yield f"data: {json.dumps(error_data)}\n\n"
                await asyncio.sleep(60)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream"
    )


@router.get("/sentiment/controversy/{symbol}")
async def detect_controversy(symbol: str):
    """
    Analyze for contradictory narratives and manipulation patterns.

    Args:
        symbol: Stock symbol

    Returns:
        Controversy analysis with:
        - Controversy score (0-100)
        - Flags (high_disagreement, bimodal_distribution, etc.)
        - Metrics (std_dev, mean_polarity, volume)
        - Explanation
    """
    try:
        engine = get_sentiment_engine()
        detector = get_controversy_detector()

        # Get recent sentiment data
        result = await engine.analyze(symbol, timeframe="1day", sources="all")

        # Additional controversy analysis
        controversy = result.get('controversy', {})

        return JSONResponse(content={
            'symbol': symbol,
            'controversy': controversy
        })

    except Exception as e:
        logger.error(f"Controversy detection error for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sentiment/temporal/{symbol}")
async def get_temporal_sentiment(
    symbol: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
):
    """
    Historical sentiment trends over time.

    Args:
        symbol: Stock symbol
        start_date: Start date (ISO format, optional)
        end_date: End date (ISO format, optional)

    Returns:
        Multi-timeframe sentiment analysis with trends
    """
    try:
        store = get_sentiment_store()

        # Get multi-timeframe analysis
        temporal = store.get_multi_timeframe_sentiment(symbol)

        return JSONResponse(content={
            'symbol': symbol,
            'temporal_analysis': temporal
        })

    except Exception as e:
        logger.error(f"Temporal sentiment error for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sentiment/health")
async def health_check():
    """
    Health check endpoint for sentiment engine components.

    Returns:
        Status of all engines, sources, and analyzers
    """
    try:
        engine = get_sentiment_engine()

        health = {
            'engines': {
                'vader': engine.vader.is_available,
                'finbert': engine.finbert.is_available,
                'roberta': engine.roberta.is_available,
                'textblob': engine.textblob.is_available,
                'gemini': engine.gemini.is_available
            },
            'sources': {
                'news': True,  # Always available
                'twitter': engine.twitter_source.is_available(),
                'reddit': engine.reddit_source.is_available(),
                'stocktwits': engine.stocktwits_source.is_available()
            },
            'analyzers': {
                'entity_detector': engine.entity_detector.is_available()
            },
            'cache': engine.cache.get_stats()
        }

        return JSONResponse(content=health)

    except Exception as e:
        logger.error(f"Health check error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sentiment/stocktwits/{symbol}")
async def get_stocktwits_sentiment(symbol: str, limit: int = Query(30, ge=1, le=30)):
    """
    Get StockTwits social sentiment for a symbol (legacy endpoint).

    Args:
        symbol: Stock symbol (e.g., 'AAPL')
        limit: Number of messages to fetch (1-30)

    Returns:
        StockTwits sentiment analysis with messages and sentiment breakdown
    """
    try:
        from src.data.sentiment.sources.stocktwits_source import get_stocktwits_streams

        result = get_stocktwits_streams(symbol, limit=limit)

        return JSONResponse(content=result)

    except Exception as e:
        logger.error(f"StockTwits sentiment error for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =========================================================================
# STOCKTWITS GATEWAY API ENDPOINTS (New)
# =========================================================================

@router.get("/sentiment/stocktwits/trending")
async def get_stocktwits_trending_enhanced(
    regions: str = Query("US", description="Region filter: US, CA, IN, ALL"),
    asset_class: str = Query("equities", description="Asset class: equities, crypto, all"),
    limit: int = Query(20, ge=1, le=100)
):
    """
    Get top trending symbols with fundamentals, prices, and trending context.

    Returns enhanced trending data including sector, industry, trending scores,
    price data, and fundamental metrics.
    """
    try:
        from src.data.sentiment.sources.stocktwits_source import get_stocktwits_source

        source = get_stocktwits_source()
        result = await source.fetch_trending_enhanced(
            regions=regions, asset_class=asset_class, limit=limit
        )

        return JSONResponse(content=result)

    except Exception as e:
        logger.error(f"StockTwits trending error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sentiment/stocktwits/{symbol}/sentiment")
async def get_stocktwits_sentiment_detail(symbol: str):
    """
    Get real-time sentiment analysis detail for a symbol.

    Returns normalized sentiment labels (Bullish/Bearish) and message volume
    indicators from StockTwits' 15-minute intervals.
    """
    try:
        from src.data.sentiment.sources.stocktwits_source import get_stocktwits_source

        source = get_stocktwits_source()
        result = await source.fetch_sentiment_detail(symbol)

        return JSONResponse(content=result)

    except Exception as e:
        logger.error(f"StockTwits sentiment detail error for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sentiment/stocktwits/{symbol}/popular")
async def get_stocktwits_popular_messages(symbol: str):
    """
    Get trending/popular messages for a specific symbol.

    Returns the most engaged messages with user info and engagement metrics.
    """
    try:
        from src.data.sentiment.sources.stocktwits_source import get_stocktwits_source

        source = get_stocktwits_source()
        result = await source.fetch_popular_messages(symbol)

        return JSONResponse(content=result)

    except Exception as e:
        logger.error(f"StockTwits popular messages error for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sentiment/stocktwits/{symbol}/why-trending")
async def get_stocktwits_why_trending(symbol: str):
    """
    Understand why a symbol is trending with AI-generated context.

    Returns trending context summary explaining the driver behind a symbol's
    increased activity (e.g., earnings beat, guidance raise, analyst upgrade).
    """
    try:
        from src.data.sentiment.sources.stocktwits_source import get_stocktwits_source

        source = get_stocktwits_source()
        result = await source.fetch_why_trending(symbol)

        return JSONResponse(content=result)

    except Exception as e:
        logger.error(f"StockTwits why-trending error for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sentiment/stocktwits/featured")
async def get_stocktwits_featured():
    """
    Get highlighted/featured messages (top posts across all symbols).
    """
    try:
        from src.data.sentiment.sources.stocktwits_source import get_stocktwits_source

        source = get_stocktwits_source()
        result = await source.fetch_featured_messages()

        return JSONResponse(content=result)

    except Exception as e:
        logger.error(f"StockTwits featured error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sentiment/stocktwits/suggested")
async def get_stocktwits_suggested():
    """
    Get suggested message stream from various users and symbols.
    """
    try:
        from src.data.sentiment.sources.stocktwits_source import get_stocktwits_source

        source = get_stocktwits_source()
        result = await source.fetch_suggested_messages()

        return JSONResponse(content=result)

    except Exception as e:
        logger.error(f"StockTwits suggested error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sentiment/stocktwits/conversation/{thread_id}")
async def get_stocktwits_conversation(thread_id: str):
    """
    Get a conversation thread with messages and replies.
    """
    try:
        from src.data.sentiment.sources.stocktwits_source import get_stocktwits_source

        source = get_stocktwits_source()
        result = await source.fetch_conversation(thread_id)

        return JSONResponse(content=result)

    except Exception as e:
        logger.error(f"StockTwits conversation error for {thread_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sentiment/stocktwits/news")
async def get_stocktwits_news(
    limit: int = Query(20, ge=1, le=100),
    published_after: Optional[str] = Query(None, description="ISO 8601 date filter")
):
    """
    Get StockTwits news articles in JSON format.

    Returns full article content with symbols, categories, and tags.
    """
    try:
        from src.data.sentiment.sources.stocktwits_source import get_stocktwits_source

        source = get_stocktwits_source()
        result = await source.fetch_news_articles(limit=limit, published_after=published_after)

        return JSONResponse(content=result)

    except Exception as e:
        logger.error(f"StockTwits news error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sentiment/stocktwits/{symbol}/latest")
async def get_stocktwits_latest_messages(symbol: str):
    """
    Get live stream data (latest messages) for a symbol via the new gateway.
    """
    try:
        from src.data.sentiment.sources.stocktwits_source import get_stocktwits_source

        source = get_stocktwits_source()
        result = await source.fetch_latest_messages(symbol)

        return JSONResponse(content=result)

    except Exception as e:
        logger.error(f"StockTwits latest error for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sentiment/stocktwits/nse/gainers")
async def get_nse_top_gainers():
    """
    Get top performing NSE (National Stock Exchange) stocks.
    """
    try:
        from src.data.sentiment.sources.stocktwits_source import get_stocktwits_source

        source = get_stocktwits_source()
        result = await source.fetch_nse_gainers()

        return JSONResponse(content=result)

    except Exception as e:
        logger.error(f"NSE gainers error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sentiment/stocktwits/nse/losers")
async def get_nse_top_losers():
    """
    Get worst performing NSE (National Stock Exchange) stocks.
    """
    try:
        from src.data.sentiment.sources.stocktwits_source import get_stocktwits_source

        source = get_stocktwits_source()
        result = await source.fetch_nse_losers()

        return JSONResponse(content=result)

    except Exception as e:
        logger.error(f"NSE losers error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
