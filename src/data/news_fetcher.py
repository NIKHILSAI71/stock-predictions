"""
News Fetcher Module - Enhanced Multi-Source Research
Retrieves comprehensive financial intelligence from 20+ official sources.
Mimics professional human research methodology.
"""


import logging
import asyncio
from ddgs import DDGS
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime

from crawl4ai import AsyncWebCrawler
import hashlib
import feedparser
from urllib.parse import quote
import random

# Configure Logger
logger = logging.getLogger(__name__)

# Official and trusted financial news sources for validation
TRUSTED_SOURCES = [
    "reuters.com", "bloomberg.com", "wsj.com", "cnbc.com", "marketwatch.com",
    "seekingalpha.com", "fool.com", "investopedia.com", "barrons.com", "ft.com",
    "finance.yahoo.com", "businessinsider.com", "thestreet.com", "zacks.com",
    "sec.gov", "nasdaq.com", "nyse.com", "benzinga.com", "tipranks.com",
    "morningstar.com", "investing.com", "kiplinger.com", "moneycontrol.com",
    "economictimes.com", "livemint.com", "hindustantimes.com", "business-standard.com"  # India sources
]

# Low quality / auto-generated / spam domains to ignore
BLACKLISTED_DOMAINS = [
    "stockstory.org", "simplestocks.com", "dailyoverview.com", "investorsobserver.com",
    "walletinvestor.com", "gov.capital", "stockinvest.us", "marketbeat.com",
    # Often considered clickbait by some pro traders, but popular. Keeping optional.
    "zacks.com", "motleyfool.com", "fool.com",
    # controversial but often spammy
    "insidermonkey.com", "investorplace.com", "thestreet.com",
    "barchart.com", "guru.com"
]


async def safe_callback(status_callback: Optional[Callable], msg: str):
    """Safely execute status callback, handling both async and sync callbacks."""
    if not status_callback or not callable(status_callback):
        return
    try:
        result = status_callback(msg)
        # If it returns an awaitable, await it
        if hasattr(result, '__await__'):
            await result
    except (TypeError, AttributeError) as e:
        # Silently ignore if callback can't be called properly
        logger.debug(f"Status callback failed: {e}")
        pass


def _is_trusted_source(url: str) -> bool:
    """Check if URL is from a trusted financial source."""
    if not url:
        return False
    url_lower = url.lower()
    return any(source in url_lower for source in TRUSTED_SOURCES)


def _is_blacklisted(url: str) -> bool:
    """Check if URL is from a blacklisted or low-quality source."""
    if not url:
        return True
    url_lower = url.lower()
    return any(source in url_lower for source in BLACKLISTED_DOMAINS)


def _deduplicate_results(results: List[Dict]) -> List[Dict]:
    """Remove duplicate articles by URL hash."""
    seen_urls = set()
    unique = []
    for r in results:
        url = r.get('link', '')
        url_hash = hashlib.md5(url.encode()).hexdigest()
        if url_hash not in seen_urls:
            seen_urls.add(url_hash)
            unique.append(r)
    return unique


async def scrape_article(url: str, max_retries: int = 2, status_callback: Optional[Callable[[str], Any]] = None) -> str:
    """
    Helper to scrape full article text from a URL using Crawl4AI.
    Includes retry logic and fixes for deprecated attributes.
    """
    if not url:
        return ""

    # SKIP YAHOO FINANCE CONSENT WALLS
    if "yahoo.com" in url or "finance.yahoo.com" in url:
        msg = f"[SCRAPER] Skipping Yahoo Finance URL (Consent Wall): {url}"
        await safe_callback(status_callback, msg)
        logger.info(msg)
        return ""

    for attempt in range(max_retries):
        try:
            msg = f"[SCRAPER] Attempt {attempt+1}: {url}"
            await safe_callback(status_callback, msg)
            logger.info(msg)

            # Configure crawler with timeout settings
            async with AsyncWebCrawler(
                verbose=(attempt == 0),
                browser_type="chromium",
                headless=True
            ) as crawler:
                # Increase timeout to 60 seconds and add retry logic
                try:
                    result = await asyncio.wait_for(
                        # 60 second timeout
                        crawler.arun(url=url, page_timeout=60000),
                        timeout=70  # Overall timeout slightly higher
                    )
                except asyncio.TimeoutError:
                    wait_time = (attempt + 1) * 3
                    msg = f"[SCRAPER] Timeout fetching {url}. Retrying in {wait_time}s..."
                    await safe_callback(status_callback, msg)
                    logger.warning(msg)
                    await asyncio.sleep(wait_time)
                    continue

                if not result.success:
                    wait_time = (attempt + 1) * 2
                    msg = f"[SCRAPER] Failed to fetch {url} - {result.error_message}. Retrying in {wait_time}s..."
                    await safe_callback(status_callback, msg)
                    logger.info(msg)
                    await asyncio.sleep(wait_time)
                    continue

                # Fix for deprecated 'fit_markdown' attribute (now in result.markdown.fit_markdown)
                # Fallback chain for text extraction
                text = ""
                if hasattr(result, 'markdown_v2') and result.markdown_v2:
                    text = result.markdown_v2.fit_markdown or result.markdown_v2.raw_markdown
                elif hasattr(result, 'markdown') and result.markdown:
                    # New version: result.markdown is often an object with fit_markdown
                    if isinstance(result.markdown, str):
                        text = result.markdown
                    else:
                        text = getattr(result.markdown, 'fit_markdown', '') or getattr(
                            result.markdown, 'raw_markdown', '')

                if not text:
                    text = result.cleaned_html or ""

                # Remove excessive whitespace and clean
                import re
                text = re.sub(r'\s+', ' ', text)
                text = text.strip()

                if len(text) > 100:
                    msg = f"[SCRAPER] Success: {url} ({len(text)} chars)"
                    await safe_callback(status_callback, msg)
                    logger.info(msg)
                    # Truncate to reasonable size for LLM context
                    return text[:4000]

        except Exception as e:
            msg = f"[SCRAPER] Error on attempt {attempt+1} for {url}: {e}"
            await safe_callback(status_callback, msg)
            logger.info(msg)
            await asyncio.sleep(1)

    return ""


async def get_google_news_rss(query: str, max_results: int = 10, status_callback: Optional[Callable[[str], Any]] = None) -> List[Dict[str, Any]]:
    """
    Fetch news from Google News RSS feed for improved reliability.
    Accepts a generic query string or symbol.
    """
    try:
        # Construct RSS URL
        # If query doesn't look like a search query, assume it's a symbol
        if len(query.split()) == 1 and query.isupper():
            search_query = f"{query} stock news"
        else:
            search_query = query

        rss_url = f"https://news.google.com/rss/search?q={quote(search_query)}&hl=en-US&gl=US&ceid=US:en"

        msg = f"[RSS ENGINE] Fetching Google News RSS for: {search_query}"
        await safe_callback(status_callback, msg)
        logger.info(msg)

        # Parse feed (sync operation, run in thread)
        def parse_feed():
            return feedparser.parse(rss_url)

        feed = await asyncio.to_thread(parse_feed)

        results = []
        for entry in feed.entries[:max_results]:
            # Google news links are often redirects, scrape_article will handle them if crawl4ai supports redirects
            # or we might need to resolve them. Crawl4AI usually handles redirects.

            results.append({
                "title": entry.title,
                "link": entry.link,
                "source": entry.source.title if hasattr(entry, 'source') else "Google News",
                "date": entry.published if hasattr(entry, 'published') else datetime.now().strftime("%Y-%m-%d"),
                "summary": entry.summary if hasattr(entry, 'summary') else entry.title,
                "content": "",  # Will be filled by scraper
                # Google News generally aggregates reputable sources, or we can check domain
                "is_trusted": True
            })

        logger.info(f"Google RSS success: {len(results)} articles found")
        return results

    except Exception as e:
        logger.error(f"Google RSS fetch failed: {e}")
        return []


async def get_latest_news(symbol: str, max_results: int = 10, custom_query: str = None, status_callback: Optional[Callable[[str], Any]] = None) -> List[Dict[str, Any]]:
    """
    Fetch latest news for a stock symbol using DuckDuckGo.
    Includes DEEP READING of top results via Crawl4AI.
    """
    clean_symbol = symbol.upper()
    exchange_name = ""

    if clean_symbol.endswith('.NS'):
        clean_symbol = clean_symbol.replace('.NS', '')
        exchange_name = "NSE India"
    elif clean_symbol.endswith('.BO'):
        clean_symbol = clean_symbol.replace('.BO', '')
        exchange_name = "BSE India"
    elif clean_symbol.endswith('.L'):
        clean_symbol = clean_symbol.replace('.L', '')
        exchange_name = "London"
    elif clean_symbol.endswith('.TO'):
        clean_symbol = clean_symbol.replace('.TO', '')
        exchange_name = "Toronto"

    if custom_query:
        query = custom_query.replace(symbol, clean_symbol)
    else:
        query = f"{clean_symbol} stock {exchange_name} news latest analysis".strip()

    results = []

    logger.info(f"Generated Web Query: {query}")
    msg = f"\n[NEWS ENGINE] Searching: '{query}'"
    await safe_callback(status_callback, msg)
    logger.info(msg)

    success_count = 0

    try:
        # PRIORITY 1: Google News RSS (More reliable than DDGS usually)
        rss_results = await get_google_news_rss(query, max_results, status_callback)
        if rss_results:
            results.extend(rss_results)
            msg = f"[NEWS] Found {len(rss_results)} articles via Google RSS"
            await safe_callback(status_callback, msg)
            logger.info(msg)

        # PRIORITY 2: DDGS if RSS low on results
        if len(results) < 5:
            # DDGS is synchronous; run in thread with retry logic
            async def fetch_ddgs_with_retry(q, limit, retries=3):
                for i in range(retries):
                    try:
                        def _fetch():
                            search_res = []
                            with DDGS(timeout=10) as ddgs:
                                try:
                                    ddgs_news = ddgs.news(
                                        query=q, region="us-en", safesearch="off", max_results=limit)
                                    if ddgs_news:
                                        for r in ddgs_news:
                                            search_res.append(r)
                                except Exception as inner_e:
                                    error_msg = str(inner_e).lower()
                                    if "503" in error_msg or "ratelimit" in error_msg:
                                        logger.warning(
                                            f"DDGS News Ratelimit on {q}")
                                    elif "gzip" in error_msg or "content-length" in error_msg:
                                        logger.warning(
                                            f"DDGS returned empty/malformed response for {q}: {inner_e}")
                                    else:
                                        logger.warning(
                                            f"DDGS News error: {inner_e}")
                                    raise inner_e
                            return search_res

                        return await asyncio.to_thread(_fetch)
                    except Exception as e:
                        error_msg = str(e).lower()
                        # Check for gzip errors specifically
                        if "gzip" in error_msg or "content-length" in error_msg:
                            logger.warning(
                                f"DDGS gzip/empty response error (attempt {i+1}/{retries}): {e}")
                            # For gzip errors, wait longer before retry
                            wait = (i + 1) * 5
                        else:
                            wait = (i + 1) * 3

                        if i == retries - 1:
                            logger.error(
                                f"DDGS ultimate failure for {q} after {retries} attempts: {e}")
                            return []

                        msg = f"[DDGS] News failed (attempt {i+1}/{retries}), retry in {wait}s..."
                        await safe_callback(status_callback, msg)
                        logger.info(msg)
                        await asyncio.sleep(wait)
                return []

            ddgs_news_results = await fetch_ddgs_with_retry(query, max_results)

            new_articles = []
            for r in ddgs_news_results:
                url = r.get("url")
                if not url or _is_blacklisted(url):
                    continue

                new_articles.append({
                    "title": r.get("title"),
                    "link": url,
                    "source": r.get("source"),
                    "date": r.get("date"),
                    "summary": r.get("body"),
                    "content": "",
                    "is_trusted": _is_trusted_source(url)
                })

            results.extend(new_articles)
            if new_articles:
                msg = f"[NEWS] Added {len(new_articles)} articles from DDGS News"
                await safe_callback(status_callback, msg)
                logger.info(msg)

    except Exception as e:
        logger.warning(f"Major news engine failure for {symbol}: {e}")

    except Exception as e:
        logger.warning(
            f"News query timed out or failed for {symbol}, trying fallback... {e}")

    if not results:
        try:
            def fetch_ddgs_text():
                search_res = []
                with DDGS(timeout=5) as ddgs:
                    ddgs_text = ddgs.text(
                        query=query, region="us-en", safesearch="off", max_results=max_results)
                    for r in ddgs_text:
                        search_res.append(r)
                return search_res

            ddgs_text_results = await asyncio.to_thread(fetch_ddgs_text)

            for r in ddgs_text_results:
                results.append({
                    "title": r.get("title"),
                    "link": r.get("href"),
                    "source": "Web Search",
                    "summary": r.get("body"),
                    "content": "",
                    "is_trusted": _is_trusted_source(r.get("href", ""))
                })

                if _is_blacklisted(r.get("href", "")):
                    continue

                msg = f"[NEWS-FALLBACK] Found: {r.get('title')} - {r.get('source', 'Web')}"
                await safe_callback(status_callback, msg)
                logger.info(msg)
            success_count = len(results)
            logger.info(
                f"Fallback search SUCCESS: {success_count} results")
        except Exception as e2:
            logger.warning(f"Fallback search also failed for {symbol}")

    # Deep read top articles (Semi-parallel with concurrency limit)
    if results:
        try:
            # Deduplicate before scraping
            results = _deduplicate_results(results)
            urls = [r['link'] for r in results[:5]]

            # Use Semaphore to avoid overwhelming the network/CPU
            semaphore = asyncio.Semaphore(3)

            async def limited_scrape(u):
                async with semaphore:
                    # Random jitter before scrape to avoid bot detection
                    await asyncio.sleep(random.uniform(0.5, 2.0))
                    return await scrape_article(u, status_callback=status_callback)

            scraped_contents = await asyncio.gather(*[limited_scrape(url) for url in urls], return_exceptions=True)

            success_count = 0
            for i, content in enumerate(scraped_contents):
                if i >= len(results):
                    break

                if isinstance(content, str) and content:
                    results[i]['content'] = content.strip()
                    success_count += 1
                else:
                    results[i]['content'] = results[i].get(
                        'summary', '') or results[i].get('title', '')

            msg = f"[SCRAPE] Deep read complete: {success_count}/5 successful"
            await safe_callback(status_callback, msg)
            logger.info(msg)
        except Exception as e:
            logger.error(f"Error in deep read: {e}")

    if success_count > 0:
        logger.info(
            f"Content extraction: {success_count}/{min(5, len(results))} articles scraped")

    return results


async def get_comprehensive_news(
    symbol: str,
    queries: List[str],
    max_per_query: int = 5,
    status_callback: Optional[Callable[[str], Any]] = None
) -> List[Dict[str, Any]]:
    """
    Comprehensive news aggregation using multiple AI-generated queries.
    Fetches from 20+ official sources like a human researcher.

    Args:
        symbol: Stock ticker
        queries: List of AI-generated search queries covering all aspects
        max_per_query: Results per query (total = queries * max_per_query)
        status_callback: Optional callback for status updates

    Returns:
        Deduplicated, prioritized list of news articles
    """
    all_results = []

    logger.info(f"COMPREHENSIVE RESEARCH MODE - {len(queries)} queries")
    msg = f"\n[DEEP RESEARCH] Starting multi-agent research with {len(queries)} queries..."
    await safe_callback(status_callback, msg)
    logger.info(msg)

    # Define fetcher mostly sync to run in thread, but calls blocking DDGS
    def fetch_query_sync(query):
        """Fetch results for a single query."""
        results = []
        try:
            with DDGS(timeout=5) as ddgs:
                # News search - limited to last 7 days
                news_results = ddgs.news(
                    query=query,
                    region="us-en",
                    safesearch="off",
                    # 'w' = last week (7 days), 'd' = last day, 'm' = last month
                    timelimit="w",
                    max_results=max_per_query
                )
                if news_results:
                    for r in news_results:
                        results.append({
                            "title": r.get("title"),
                            "link": r.get("url"),
                            "source": r.get("source"),
                            "date": r.get("date"),
                            "summary": r.get("body"),
                            "content": "",
                            "query_type": query[:50],
                            "is_trusted": _is_trusted_source(r.get("url", ""))
                        })

                        if _is_blacklisted(r.get("url", "")):
                            continue

                # Also do text search for broader coverage
                text_results = ddgs.text(
                    query=query,
                    region="us-en",
                    safesearch="off",
                    max_results=3
                )
                if text_results:
                    for r in text_results:
                        results.append({
                            "title": r.get("title"),
                            "link": r.get("href"),
                            "source": "Web",
                            "date": "",
                            "summary": r.get("body"),
                            "content": "",
                            "query_type": query[:50],
                            "is_trusted": _is_trusted_source(r.get("href", ""))
                        })

                        if _is_blacklisted(r.get("href", "")):
                            continue
        except Exception as e:
            if "503" in str(e) or "Ratelimit" in str(e):
                raise e  # Re-raise for retry logic
            logger.warning(f"Query failed: {query[:40]}... - {e}")

        return results

    # Execute queries with controlled concurrency
    msg = f"[DEEP RESEARCH] Executing {len(queries)} queries with rate control..."
    await safe_callback(status_callback, msg)
    logger.info(msg)

    # Use semaphore for queries to avoid DDGS 503 errors
    query_semaphore = asyncio.Semaphore(2)

    async def fetch_with_backoff(q, attempt=0):
        async with query_semaphore:
            if attempt > 0:
                wait = (attempt * 5) + (attempt * 2)
                await asyncio.sleep(wait)

            try:
                return await asyncio.to_thread(fetch_query_sync, q)
            except Exception as e:
                if attempt < 2 and ("503" in str(e) or "Ratelimit" in str(e)):
                    return await fetch_with_backoff(q, attempt + 1)
                return []

    tasks = [fetch_with_backoff(q) for q in queries]
    query_results_list = await asyncio.gather(*tasks)

    for results in query_results_list:
        all_results.extend(results)

    # Deduplicate
    unique_results = _deduplicate_results(all_results)

    # Prioritize trusted sources
    clean_results = [
        r for r in unique_results if not _is_blacklisted(r.get('link', ''))]
    trusted = [r for r in clean_results if r.get('is_trusted')]
    untrusted = [r for r in clean_results if not r.get('is_trusted')]

    prioritized = trusted + untrusted[:15]

    msg = f"[DEEP RESEARCH] Gathered {len(prioritized)} sources. Scrutinizing top 8 for deep reading..."
    await safe_callback(status_callback, msg)
    logger.info(msg)

    # Deep read top articles (limited concurrency)
    articles_to_read = min(8, len(prioritized))
    scrape_semaphore = asyncio.Semaphore(3)

    async def scrape_task(idx):
        url = prioritized[idx].get('link')
        if not url:
            return
        async with scrape_semaphore:
            content = await scrape_article(url, status_callback=status_callback)
            if content:
                prioritized[idx]['content'] = content
                return True
            return False

    scrape_results = await asyncio.gather(*[scrape_task(i) for i in range(articles_to_read)])
    success_count = sum(1 for r in scrape_results if r)

    logger.info(
        f"RESEARCH COMPLETE: {len(prioritized)} sources, {success_count} deep reads.")
    return prioritized[:25]


async def get_market_sentiment_search(symbol: str, custom_query: str = None, status_callback: Optional[Callable[[str], Any]] = None) -> Dict[str, Any]:
    """
    Get sentiment proxy by searching for recent sentiment keywords.
    In a real system, this would analyze the text.
    Here we return the raw search snippets for the AI to analyze.
    """
    news = await get_latest_news(symbol, max_results=10, custom_query=custom_query, status_callback=status_callback)

    # Simple keyword counting for basic proxy (AI will do better analysis)
    text_content = " ".join(
        [n.get("title", "") + " " + n.get("summary", "") for n in news]).lower()

    positive_words = ["surge", "buy", "outperform",
                      "growth", "record", "bull", "gain", "upgrade"]
    negative_words = ["plunge", "sell", "underperform",
                      "decline", "miss", "bear", "loss", "downgrade"]

    pos_count = sum(1 for word in positive_words if word in text_content)
    neg_count = sum(1 for word in negative_words if word in text_content)

    sentiment_score = 0
    if pos_count + neg_count > 0:
        sentiment_score = (pos_count - neg_count) / (pos_count + neg_count)

    return {
        "symbol": symbol,
        "sentiment_score": round(sentiment_score, 2),
        "news_volume": len(news),
        "recent_headlines": [n["title"] for n in news[:3]],
        "news_context": news  # Pass full context for AI
    }
