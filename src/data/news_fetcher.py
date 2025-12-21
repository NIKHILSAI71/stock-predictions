"""
News Fetcher Module - Enhanced Multi-Source Research
Retrieves comprehensive financial intelligence from 20+ official sources.
Mimics professional human research methodology.
"""

from ddgs import DDGS
from typing import List, Dict, Any, Optional
from datetime import datetime
import requests
from bs4 import BeautifulSoup
import concurrent.futures
import hashlib

# ANSI Colors
GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
RESET = "\033[0m"

# Official and trusted financial news sources for validation
TRUSTED_SOURCES = [
    "reuters.com", "bloomberg.com", "wsj.com", "cnbc.com", "marketwatch.com",
    "seekingalpha.com", "fool.com", "investopedia.com", "barrons.com", "ft.com",
    "finance.yahoo.com", "businessinsider.com", "thestreet.com", "zacks.com",
    "sec.gov", "nasdaq.com", "nyse.com", "benzinga.com", "tipranks.com",
    "morningstar.com", "investing.com", "kiplinger.com", "moneycontrol.com",
    "economictimes.com", "livemint.com"  # India sources
]


def _is_trusted_source(url: str) -> bool:
    """Check if URL is from a trusted financial source."""
    if not url:
        return False
    url_lower = url.lower()
    return any(source in url_lower for source in TRUSTED_SOURCES)


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


def scrape_article(url: str) -> str:
    """
    Helper to scrape full article text from a URL.
    Uses multiple extraction strategies for maximum success.
    """
    if not url:
        return ""
    
    try:
        # More complete browser headers to avoid blocks
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Cache-Control': 'max-age=0'
        }
        
        response = requests.get(url, headers=headers, timeout=8, allow_redirects=True)
        
        if response.status_code != 200:
            return ""
            
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove unwanted elements
        for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside', 
                            'advertisement', 'iframe', 'noscript', 'form']):
            element.decompose()
        
        # Strategy 1: Common article selectors (site-specific)
        article_selectors = [
            # Generic
            ('article', {}),
            ('div', {'class': ['article-body', 'article-content', 'story-body', 
                              'post-content', 'entry-content', 'content-body',
                              'ArticleBody', 'article__body', 'caas-body']}),
            ('div', {'itemprop': 'articleBody'}),
            ('div', {'id': ['article-body', 'story-body', 'main-content']}),
            # News sites
            ('div', {'class': ['zn-body__paragraph', 'body-content']}),  # CNN
            ('div', {'class': ['paywall']}),  # Some paywalled content
            ('section', {'class': 'article-body'}),
        ]
        
        text = ""
        
        for tag, attrs in article_selectors:
            if attrs:
                elements = soup.find_all(tag, attrs)
            else:
                elements = soup.find_all(tag)
            
            for element in elements:
                paragraphs = element.find_all(['p', 'li'])
                extracted = ' '.join([
                    p.get_text().strip() 
                    for p in paragraphs 
                    if len(p.get_text().strip()) > 40
                ])
                if len(extracted) > len(text):
                    text = extracted
        
        # Strategy 2: If no article found, get all paragraphs
        if len(text) < 200:
            all_paragraphs = soup.find_all('p')
            text = ' '.join([
                p.get_text().strip() 
                for p in all_paragraphs 
                if len(p.get_text().strip()) > 50
            ])
        
        # Strategy 3: Get main content div
        if len(text) < 200:
            main = soup.find('main') or soup.find('div', {'role': 'main'})
            if main:
                text = main.get_text(separator=' ', strip=True)
        
        # Clean up the text
        if text:
            # Remove excessive whitespace
            import re
            text = re.sub(r'\s+', ' ', text)
            text = text.strip()
            
            # Truncate to reasonable size
            return text[:4000] if len(text) > 4000 else text
        
        return ""
        
    except Exception as e:
        print(f"{YELLOW}WARN:     Scrape failed for {url[:50]}... - {str(e)[:30]}{RESET}")
        return ""


def get_latest_news(symbol: str, max_results: int = 10, custom_query: str = None) -> List[Dict[str, Any]]:
    """
    Fetch latest news for a stock symbol using DuckDuckGo.
    Includes DEEP READING of top results.
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
    
    print(f"{GREEN}INFO:     Generated Web Query: {query}{RESET}")
    
    try:
        with DDGS() as ddgs:
            ddgs_news = ddgs.news(query=query, region="us-en", safesearch="off", max_results=max_results)
            if ddgs_news:
                for r in ddgs_news:
                    results.append({
                        "title": r.get("title"),
                        "link": r.get("url"),
                        "source": r.get("source"),
                        "date": r.get("date"),
                        "summary": r.get("body"),
                        "content": "",
                        "is_trusted": _is_trusted_source(r.get("url", ""))
                    })
    except Exception as e:
        print(f"Error fetching news for {symbol}: {e}")

    if not results:
        try:
            with DDGS() as ddgs:
                ddgs_text = ddgs.text(query=query, region="us-en", safesearch="off", max_results=max_results)
                for r in ddgs_text:
                    results.append({
                        "title": r.get("title"),
                        "link": r.get("href"),
                        "source": "Web Search",
                        "summary": r.get("body"),
                        "content": "",
                        "is_trusted": _is_trusted_source(r.get("href", ""))
                    })
        except Exception as e2:
            print(f"Error in fallback search for {symbol}: {e2}")

    # Deep read top articles
    if results:
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            urls = [r['link'] for r in results[:8]]
            contents = list(executor.map(scrape_article, urls))
            
            for i, content in enumerate(contents):
                if content:
                    results[i]['content'] = content.strip()

    return results


def get_comprehensive_news(
    symbol: str, 
    queries: List[str],
    max_per_query: int = 5
) -> List[Dict[str, Any]]:
    """
    Comprehensive news aggregation using multiple AI-generated queries.
    Fetches from 20+ official sources like a human researcher.
    
    Args:
        symbol: Stock ticker
        queries: List of AI-generated search queries covering all aspects
        max_per_query: Results per query (total = queries * max_per_query)
    
    Returns:
        Deduplicated, prioritized list of news articles
    """
    all_results = []
    
    print(f"{CYAN}{'='*60}{RESET}")
    print(f"{CYAN}INFO:     COMPREHENSIVE RESEARCH MODE - {len(queries)} queries{RESET}")
    print(f"{CYAN}{'='*60}{RESET}")
    
    def fetch_query(query):
        """Fetch results for a single query."""
        results = []
        try:
            with DDGS() as ddgs:
                # News search - limited to last 7 days
                news_results = ddgs.news(
                    query=query, 
                    region="us-en", 
                    safesearch="off", 
                    timelimit="w",  # 'w' = last week (7 days), 'd' = last day, 'm' = last month
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
        except Exception as e:
            print(f"{YELLOW}WARN:     Query failed: {query[:40]}... - {e}{RESET}")
        
        return results
    
    # Execute all queries in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
        futures = {executor.submit(fetch_query, q): q for q in queries}
        
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            query = futures[future]
            try:
                results = future.result()
                all_results.extend(results)
                print(f"{GREEN}INFO:     Query {i+1}/{len(queries)}: {len(results)} results - {query[:50]}...{RESET}")
            except Exception as e:
                print(f"{YELLOW}WARN:     Query failed: {query[:30]}... - {e}{RESET}")
    
    # Deduplicate
    unique_results = _deduplicate_results(all_results)
    
    # Prioritize trusted sources
    trusted = [r for r in unique_results if r.get('is_trusted')]
    untrusted = [r for r in unique_results if not r.get('is_trusted')]
    prioritized = trusted + untrusted
    
    print(f"{CYAN}INFO:     Total unique articles: {len(prioritized)} ({len(trusted)} from trusted sources){RESET}")
    
    # Deep read top 20 articles (prioritizing trusted sources)
    articles_to_read = min(20, len(prioritized))
    print(f"{GREEN}INFO:     Deep reading top {articles_to_read} articles for full content...{RESET}")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=15) as executor:
        urls = [r['link'] for r in prioritized[:articles_to_read] if r.get('link')]
        contents = list(executor.map(scrape_article, urls))
        
        success_count = 0
        for i, content in enumerate(contents):
            if content and i < len(prioritized):
                prioritized[i]['content'] = content
                success_count += 1
                print(f"{GREEN}INFO:     Deep Read SUCCESS ({len(content)} chars): {prioritized[i]['title'][:35]}...{RESET}")
        
        print(f"{CYAN}INFO:     Deep reading complete: {success_count}/{articles_to_read} articles extracted{RESET}")
    
    print(f"{CYAN}{'='*60}{RESET}")
    print(f"{CYAN}INFO:     RESEARCH COMPLETE: {len(prioritized)} sources gathered{RESET}")
    print(f"{CYAN}{'='*60}{RESET}")
    
    return prioritized[:25]  # Return top 25 sources

def get_market_sentiment_search(symbol: str, custom_query: str = None) -> Dict[str, Any]:
    """
    Get sentiment proxy by searching for recent sentiment keywords.
    In a real system, this would analyze the text. 
    Here we return the raw search snippets for the AI to analyze.
    """
    news = get_latest_news(symbol, max_results=10, custom_query=custom_query)
    
    # Simple keyword counting for basic proxy (AI will do better analysis)
    text_content = " ".join([n.get("title", "") + " " + n.get("summary", "") for n in news]).lower()
    
    positive_words = ["surge", "buy", "outperform", "growth", "record", "bull", "gain", "upgrade"]
    negative_words = ["plunge", "sell", "underperform", "decline", "miss", "bear", "loss", "downgrade"]
    
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
        "news_context": news # Pass full context for AI
    }
