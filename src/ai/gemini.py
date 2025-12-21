
import google.generativeai as genai
import os
import json
import hashlib
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta

from src import config


logger = logging.getLogger(__name__)

# Response cache for consistency: cache_key -> (response, timestamp)
_response_cache: Dict[str, Tuple[Dict, datetime]] = {}
CACHE_TTL = 300  # 5 minutes

# Consistency config - low temperature for stable outputs
CONSISTENCY_CONFIG = {
    'temperature': 0.1,  # CRITICAL: Low temp for consistent outputs
    'top_p': 0.8,
    'top_k': 40,
    'max_output_tokens': 4096,
}


def _create_cache_key(symbol: str, data: Dict) -> str:
    """Create deterministic cache key for response caching."""
    key_data = {
        'symbol': symbol,
        'signal': str(data.get('universal_system_signal', {}).get('signal', 'UNKNOWN')),
        'price': round(data.get('technicals', {}).get('current_price', 0), 2),
    }
    key_str = json.dumps(key_data, sort_keys=True)
    return hashlib.md5(key_str.encode()).hexdigest()


def _get_cached_response(cache_key: str) -> Optional[Dict]:
    """Get cached response if still valid."""
    if cache_key not in _response_cache:
        return None
    response, timestamp = _response_cache[cache_key]
    if datetime.now() - timestamp > timedelta(seconds=CACHE_TTL):
        del _response_cache[cache_key]
        return None
    return response.copy()


# Configure API key safely
def configure_gemini():
    api_key = config.GEMINI_API_KEY
    if not api_key:
        logger.warning("GEMINI_API_KEY not found in configuration.")
        return False
    
    genai.configure(api_key=api_key)
    return True



def _build_sector_context(stock_classification: Dict[str, Any]) -> str:
    """
    Build sector-specific context for AI prompt.
    Returns formatted string with classification details and sector-specific rules.
    """
    if not stock_classification:
        return "No classification data available - use default S&P 500 benchmarks."
    
    sector = stock_classification.get('sector', 'Unknown')
    stock_type = stock_classification.get('stock_type', 'growth')
    market_cap_tier = stock_classification.get('market_cap_tier', 'unknown')
    sector_avg_pe = stock_classification.get('sector_avg_pe', 17.8)
    volatility_profile = stock_classification.get('volatility_profile', 'moderate')
    is_commodity_linked = stock_classification.get('is_commodity_linked', False)
    is_rate_sensitive = stock_classification.get('is_interest_rate_sensitive', False)
    
    # Sector-specific analysis rules
    sector_rules = {
        'Technology': """
SECTOR-SPECIFIC RULES FOR TECHNOLOGY:
- High P/E (25-40x) is ACCEPTABLE for high-growth tech with wide moat
- Prioritize: Revenue growth rate, Gross margin, R&D spending
- Wide moat is CRITICAL - no moat = downgrade valuation tolerance
- RSI overbought thresholds can be higher (75+) due to momentum nature
- MACD divergence is more significant - tech trends strongly
""",
        'Communication Services': """
SECTOR-SPECIFIC RULES FOR COMMUNICATION SERVICES:
- P/E of 20-35x is normal for this sector
- Check subscriber/user growth, ARPU trends
- Network effects create moats - evaluate platform stickiness
- Ad revenue growth is key for Alphabet/Meta type companies
""",
        'Financials': """
SECTOR-SPECIFIC RULES FOR FINANCIALS:
- P/E > 15x is EXPENSIVE for banks (use 8-15x as normal range)
- P/B (Price to Book) is MORE important than P/E
- Interest rate sensitivity is CRITICAL - check yield curve
- Check: Net Interest Margin, ROE, Tier 1 Capital Ratio
- Credit quality and loan loss provisions matter
- In rate-hiking cycles: POSITIVE for net interest income
""",
        'Energy': """
SECTOR-SPECIFIC RULES FOR ENERGY:
- P/E is LESS RELEVANT - focus on commodity price correlation
- This stock moves with OIL/GAS prices - track crude oil trends
- Check: Production volume, Debt levels, Breakeven oil price
- Cyclical sector - avoid buying at peak commodity prices
- High debt is dangerous in price downturns
- Dividends may be cut if commodity prices crash
""",
        'Materials': """
SECTOR-SPECIFIC RULES FOR MATERIALS:
- Commodity-driven like Energy - P/E less relevant
- Track: Industrial metal prices, China demand, Inventory levels
- Cyclical - tied to economic growth
- Check debt levels and cost structure
""",
        'Healthcare': """
SECTOR-SPECIFIC RULES FOR HEALTHCARE:
- For large pharma: P/E of 15-25x is normal
- For BIOTECH (no profits): P/E is NOT APPLICABLE
  * Evaluate by: Pipeline value, FDA calendar, Cash runway
  * Binary events (trial results) create EXTREME volatility
  * Use small position sizes
- Patent cliff is a major risk - check expiration dates
- R&D productivity matters
""",
        'Consumer Discretionary': """
SECTOR-SPECIFIC RULES FOR CONSUMER DISCRETIONARY:
- Cyclical sector - performance tied to consumer confidence
- P/E of 15-30x depending on growth
- Check: Same-store sales, Inventory turnover, E-commerce mix
- Interest rate sensitive (consumer credit)
- Recession risk: downgrade in bear markets
""",
        'Consumer Staples': """
SECTOR-SPECIFIC RULES FOR CONSUMER STAPLES (DEFENSIVE):
- Lower P/E tolerance: 15-22x is normal
- Dividend yield is KEY - this is an income sector
- Focus on: Pricing power, Market share stability
- Defensive - outperforms in bear markets
- Low beta expected
""",
        'Utilities': """
SECTOR-SPECIFIC RULES FOR UTILITIES (DEFENSIVE):
- P/E of 12-22x is normal for utilities
- Dividend yield is PRIMARY consideration
- Interest rate VERY sensitive (bond proxy)
- Check: Rate base growth, Regulatory environment
- In rate-hiking cycles: utilities UNDERPERFORM
- Low volatility expected - stability is the appeal
""",
        'Real Estate': """
SECTOR-SPECIFIC RULES FOR REAL ESTATE (REITs):
- DO NOT USE P/E - use P/FFO (Funds From Operations) instead
- Dividend yield is critical (REITs must pay 90% of income)
- Interest rate VERY sensitive
- Check: Occupancy rate, Cap rate, Debt maturity schedule
- Different REIT types behave differently (office vs industrial vs residential)
""",
        'Industrials': """
SECTOR-SPECIFIC RULES FOR INDUSTRIALS:
- Cyclical sector - tied to economic cycle
- P/E of 15-25x depending on cycle position
- Check: Order backlog, Capacity utilization, PMI trends
- Late cycle performer typically
- Capex trends matter
"""
    }
    
    # Build context string
    context = f"""
Stock Classification:
- Market Cap Tier: {market_cap_tier.upper()}
- Sector: {sector}
- Stock Type: {stock_type}
- Sector Average P/E: {sector_avg_pe}x (USE THIS, NOT 17.8x)
- Volatility Profile: {volatility_profile}
- Commodity-Linked: {'YES' if is_commodity_linked else 'No'}
- Interest Rate Sensitive: {'YES' if is_rate_sensitive else 'No'}

{sector_rules.get(sector, 'Use balanced fundamental + technical approach.')}
"""
    
    # Add warnings if any
    warnings = stock_classification.get('warnings', [])
    if warnings:
        context += "\nWARNINGS:\n" + "\n".join(f"- {w}" for w in warnings)
    
    return context



def generate_search_query(symbol: str, data_context: Dict[str, Any]) -> str:
    """
    Agentic step: Analyzes data to formulate the best search query.
    DEPRECATED: Use generate_search_queries() for comprehensive research.
    """
    if not configure_gemini():
        return f"{symbol} stock news latest analysis"
        
    model = genai.GenerativeModel(config.GEMINI_MODEL)
    
    # Create a mini-summary for the prompt
    signal = data_context.get('universal_system_signal', {}).get('signal', 'UNKNOWN')
    technicals = data_context.get('technicals', {})
    price_change = technicals.get('price_change_pct', 0)
    
    prompt = f"""
    You are a Financial News Researcher. 
    Stock: {symbol}
    Current Signal: {signal}
    Price Change Today: {price_change}%
    
    Based on this, what is the SINGLE best search query to find the explanation or validity of this move?
    If big move -> "Why did {symbol} move X% today"
    If earnings season -> "{symbol} earnings analysis"
    If unknown -> "{symbol} stock latest news catalyst"
    
    Return ONLY the query string. No quotes.
    """
    
    try:
        response = model.generate_content(prompt)
        query = response.text.strip().replace('"', '')
        return query if len(query) < 100 else f"{symbol} stock news"
    except:
        return f"{symbol} stock news analysis"


def generate_search_queries(symbol: str, data_context: Dict[str, Any]) -> List[str]:
    """
    AI-powered comprehensive research query generator.
    Generates 8-12 targeted queries covering ALL aspects a professional analyst would research.
    
    Query Categories:
    1. Breaking news & recent catalysts
    2. Earnings & revenue analysis
    3. SEC filings & regulatory news
    4. Analyst ratings & price targets
    5. Sector & competitive dynamics  
    6. Insider & institutional activity
    7. Technical analysis perspectives
    8. Risk factors & concerns
    9. Management & corporate news
    10. Macro economic impact
    11. Options & derivatives activity
    12. Valuation & fair value analysis
    
    Returns:
        List of 8-12 specific search queries for comprehensive research
    """
    if not configure_gemini():
        # Fallback queries if AI unavailable
        return [
            f"{symbol} stock news today",
            f"{symbol} earnings report analysis",
            f"{symbol} analyst rating upgrade downgrade",
            f"{symbol} SEC filing 10-K 10-Q",
            f"{symbol} insider buying selling",
            f"{symbol} sector competitors comparison",
            f"{symbol} price target forecast",
            f"{symbol} risk factors concerns"
        ]
    
    model = genai.GenerativeModel(
        config.GEMINI_MODEL,
        generation_config={'temperature': 0.3, 'max_output_tokens': 1024}
    )
    
    # Extract context for prompt
    signal_info = data_context.get('universal_system_signal', {})
    signal = signal_info.get('signal', 'UNKNOWN')
    technicals = data_context.get('technicals', {})
    price_change = technicals.get('price_change_pct', 0)
    rsi = technicals.get('rsi', 50)
    classification = data_context.get('classification', {})
    sector = classification.get('sector', 'Unknown')
    
    prompt = f"""
You are a Senior Financial Research Analyst conducting comprehensive due diligence on {symbol}.
Your task is to generate 15-18 highly specific search queries that will gather ALL information needed for a complete investment analysis.

CRITICAL: You must research UNPREDICTABLE RISK FACTORS that could cause sudden price movements.

=== STOCK CONTEXT ===
Symbol: {symbol}
Current Signal: {signal}
Price Change: {price_change}%
RSI: {rsi}
Sector: {sector}

=== MANDATORY RESEARCH AREAS ===

**STANDARD ANALYSIS (8-10 queries)**
1. BREAKING NEWS & CATALYSTS - Recent announcements, market-moving events
2. EARNINGS & FINANCIALS - Quarterly results, EPS surprises, guidance changes
3. SEC FILINGS & REGULATORY - 10-K, 10-Q, FDA approvals, regulatory actions
4. ANALYST COVERAGE - Price targets, rating changes, consensus
5. SECTOR & COMPETITION - Industry trends, competitor performance
6. INSIDER & INSTITUTIONAL - Insider buying/selling, 13F filings
7. TECHNICAL & OPTIONS - Unusual options activity, put/call ratios

**UNPREDICTABLE RISK FACTORS (5-7 queries) - CRITICAL**
8. LAWSUIT & LEGAL RISKS - Class action lawsuits, SEC investigations, DOJ probes
9. EXECUTIVE CHANGES - CEO resignation, CFO departure, board changes
10. BLACK SWAN INDICATORS - Supply chain disruptions, geopolitical exposure, pandemic impact
11. MARKET MANIPULATION SIGNALS - Short squeeze potential, meme stock activity, unusual volume
12. REGULATORY CRACKDOWNS - Antitrust, data privacy violations, environmental fines
13. SENTIMENT FRAGILITY - Social media controversy, boycotts, public relations crisis
14. DEBT & LIQUIDITY CRISIS - Bond downgrades, covenant breaches, cash burn rate

=== OUTPUT FORMAT ===
Return ONLY a JSON array of 15-18 search query strings. No other text.
Make each query specific and actionable. Include the stock symbol in each query.

Example output:
["{symbol} latest news today", "{symbol} lawsuit SEC investigation", "{symbol} CEO resignation executive changes", "{symbol} short squeeze potential", ...]
"""

    try:
        response = model.generate_content(prompt)
        text = response.text.strip()
        
        # Clean up response - extract JSON array
        if '[' in text and ']' in text:
            start = text.find('[')
            end = text.rfind(']') + 1
            json_str = text[start:end]
            queries = json.loads(json_str)
            
            if isinstance(queries, list) and len(queries) >= 5:
                logger.info(f"AI generated {len(queries)} research queries for {symbol}")
                return queries[:12]  # Cap at 12
        
        # Fallback if parsing fails
        raise ValueError("Could not parse AI response")
        
    except Exception as e:
        logger.warning(f"AI query generation failed: {e}. Using fallback queries.")
        # Comprehensive fallback queries
        return [
            f"{symbol} stock news today latest",
            f"{symbol} quarterly earnings results analysis",
            f"{symbol} analyst rating price target",
            f"{symbol} SEC filing 10-K 10-Q 8-K",
            f"{symbol} insider trading buying selling",
            f"{symbol} institutional investors holdings",
            f"{symbol} {sector} sector comparison competitors",
            f"{symbol} risk factors concerns warnings",
            f"{symbol} management CEO CFO executive resignation",
            f"{symbol} options unusual activity put call",
            f"{symbol} lawsuit SEC investigation legal",
            f"{symbol} short squeeze short interest manipulation",
            f"{symbol} earnings surprise miss beat",
            f"{symbol} controversy scandal boycott",
            f"{symbol} supply chain disruption risk",
            f"{symbol} analyst downgrade upgrade rating"
        ]

def generate_market_insights(
    stock_symbol: str,
    technical_data: Dict[str, Any],
    fundamental_data: Dict[str, Any],
    news_sentiment: Dict[str, Any],
    extra_metrics: Dict[str, Any] = None,
    macro_data: Dict[str, Any] = None,
    stock_classification: Dict[str, Any] = None,
    universal_signal: Dict[str, Any] = None,
    search_context: List[Dict[str, Any]] = None,
    use_cache: bool = True
) -> Dict[str, Any]:
    """
    Generate comprehensive market insights acting as an Autonomous Agent.
    
    Features:
    - Low temperature (0.1) for consistent outputs
    - Response caching (5 min TTL) for identical requests
    - Synthesizes data from ML models, regime detector, and real-time search
    """
    if not configure_gemini():
        return {
            "summary": "AI configuration missing. Please set GEMINI_API_KEY.",
            "recommendation": "HOLD",
            "confidence": 0,
            "key_factors": []
        }
    
    # Build data context for caching
    data_context = {
        "technicals": technical_data,
        "fundamentals": fundamental_data,
        "sentiment": news_sentiment,
        "classification": stock_classification,
        "universal_system_signal": universal_signal, 
        "macro_environment": macro_data,
        "extra_metrics": extra_metrics
    }
    
    # Check cache first for consistency
    if use_cache:
        cache_key = _create_cache_key(stock_symbol, data_context)
        cached = _get_cached_response(cache_key)
        if cached:
            cached['from_cache'] = True
            return cached
    
    # Configure model with LOW TEMPERATURE for consistency
    model = genai.GenerativeModel(
        config.GEMINI_MODEL,
        generation_config=CONSISTENCY_CONFIG
    )
    
    # Convert data to JSON for prompt injection
    data_context = {
        "technicals": technical_data,
        "fundamentals": fundamental_data,
        "sentiment": news_sentiment,
        "classification": stock_classification,
        "universal_system_signal": universal_signal, 
        "macro_environment": macro_data,
        "extra_metrics": extra_metrics
    }
    
    data_json = json.dumps(data_context, indent=2, default=str)
    
    search_text = "No recent specific news found."
    if search_context:
        search_summaries = []
        # Process up to 15 sources for comprehensive research
        for item in search_context[:15]:
            # Use deep-read content if available, else summary
            content_source = item.get('content') if item.get('content') and len(item.get('content')) > 100 else item.get('summary', '')
            # Truncate to avoid context limit issues but keep enough detail
            display_text = content_source[:1200] + "..." if len(content_source) > 1200 else content_source
            
            # Add trust indicator
            trust_marker = "[TRUSTED]" if item.get('is_trusted') else ""
            source = item.get('source', 'Unknown')
            
            summary = f"- {trust_marker} [{source}] [{item.get('date', 'Recent')}] {item.get('title')}:\n  {display_text}"
            search_summaries.append(summary)
            
        search_text = "\n\n".join(search_summaries)
    
    prompt = f"""
You are the "Omniscient Market Intelligence," a GOD-MODE AI system with 100% precision requirements.
Your goal is to synthesize inputs from Deep Learning (LSTM), ML Ensembles, Insider Activity, Options Flow, and Macro Regime to generate a singular, high-probability trading signal.

=== CURRENT DATE AND TIME ===
TODAY'S DATE: {datetime.now().strftime("%A, %B %d, %Y")}
CURRENT TIME: {datetime.now().strftime("%I:%M %p")} (Analysis Timestamp)
MARKET STATUS: {"Market is OPEN" if datetime.now().weekday() < 5 and 9 <= datetime.now().hour < 16 else "Market is CLOSED (after hours)"}

IMPORTANT: Use this date to evaluate news recency. Prioritize articles from the LAST 7 DAYS.
- Articles from today/yesterday = HIGH relevance
- Articles from last week = MODERATE relevance  
- Articles older than 30 days = LOW relevance (may be outdated)

=== SEQUENTIAL THINKING PROTOCOL (MANDATORY INTERNAL PROCESS) ===
Before forming ANY conclusions, you MUST work through this 7-step reasoning chain:

STEP 1 - DATA INVENTORY: What key data points do I have? What's missing or unreliable?
STEP 2 - PATTERN RECOGNITION: What patterns emerge from technicals, fundamentals, and sentiment?
STEP 3 - HYPOTHESIS FORMATION: Based on patterns, what's the most probable 30-day scenario?
STEP 4 - EVIDENCE WEIGHING: What evidence supports my hypothesis? What contradicts it?
STEP 5 - RISK CALIBRATION: What could invalidate my thesis? How severe is the downside?
STEP 6 - CONFIDENCE ASSESSMENT: Given evidence quality and model agreement, how reliable is this?
STEP 7 - SYNTHESIS: Combine all factors into a single, actionable verdict with CLEAR REASONING.

You will use this internal process to ensure your conclusions are logical and well-founded.
The EXECUTIVE SUMMARY must clearly explain the REASONS for your final verdict using "because" statements.

=== REAL-TIME WEB INTELLIGENCE (LIVE DATA - {len(search_context) if search_context else 0} SOURCES SCRAPED TODAY) ===
IMPORTANT: The following news articles were scraped in REAL-TIME just now. This is CURRENT, LIVE data.
You MUST read ALL of these sources carefully and summarize the key findings in your response.
Include specific headlines, dates, and sources in your web_intelligence_summary.
PRIORITIZE news from the LAST 7 DAYS based on today's date above.

{search_text}

=== SYSTEM DATA INPUTS (GOD MODE ENABLED) ===
{data_json}

=== MANDATORY GOD-MODE ANALYSIS PROTOCOL ===
You MUST process this data step-by-step with ruthless logic.
1. NO EMOJIS. Use professional financial terminology ONLY.
2. UNIVERSAL APPLICABILITY: This model must work for ANY stock (Growth, Value, Penny, Mega-Cap).
3. EXHAUSTIVE ONE-SHOT ANALYSIS: You must provide a COMPLETE, DETAILED thesis for Bull, Bear, and Base cases.
4. COLOR-NEUTRAL LANGUAGE: Do not use "green/red" imagery. Stick to "Bullish/Bearish" or "Upside/Downside".
5. FORMATTING: Write Key Drivers and Risk Factors as complete, natural sentences.
6. EXECUTIVE SUMMARY REASONING: The agent_summary MUST explain WHY the recommendation is made using clear "because X, therefore Y" logic. This is the most important output - investors need to understand the REASONING, not just the verdict.
7. DYNAMIC CONTENT: Provide AS MANY Key Drivers and Risk Factors as needed to fully explain the thesis.

**Phase 1: MARKET REGIME & SECTOR CONTEXT (Weight: 15%)**
- Identify the Market Regime (Bull/Bear + Volatility).
- Analyze Sector Relative Strength: Is {stock_symbol} leading or lagging its sector?
- IMPLICATION: If Sector is lagging AND Market is Bearish -> HARD SELL unless idiosyncratic catalyst exists.

**Phase 2: SMART MONEY & ALTERNATIVE DATA (Weight: 20%)**
- Insider Activity: Are executives buying? (Strongest bullish signal)
- Institutional Holdings: Is smart money accumulating?
- Options Flow: Put/Call Ratio and Unusual Whales.
- SHOW: "Insiders: [Action]. Institutions: [Trend]. Options: [Sentiment]. Smart Money Score: [0-100]."

**Phase 3: ADVANCED ML ENSEMBLE (Weight: 25%)**
- LSTM Deep Learning: What is the 5-day price target and confidence?
- XGBoost Gradient Boosting: Check direction and confidence (highest accuracy model per research).
- GRU Neural Network: Compare with LSTM - if both agree, HIGHER confidence.
- GARCH Volatility: Use volatility regime (Low/Normal/High) to size positions.
- ML Ensemble Consensus: Do Random Forest, SVM, XGBoost, LSTM, GRU, and Momentum models AGREE?
- IF XGBoost + LSTM + GRU all Agree -> CONVICTION MODE ON.
- IF Models Conflict -> PENALIZE Confidence significantly.
- High GARCH volatility -> Reduce position size recommendation.
- SHOW: "LSTM Target: [Price]. XGBoost: [Direction]. GRU: [Direction]. Volatility Regime: [Level]. Ensemble Agreement: [Yes/No]."

**Phase 4: VALUATION & FUNDAMENTALS (Weight: 20%)**
- Compare P/E vs Sector Average (Calculate Premium/Discount %).
- Check DCF Fair Value vs Current Price (Upside/Downside %).
- Is the Moat Wide or Narrow? (Wide moat allows higher valuation premium).
- SHOW: "Valuation: [Undervalued/Overvalued] by [X]%. Fair Value: $[Price]."

**Phase 5: TECHNICAL PRECISION (Weight: 10%)**
- RSI & MACD: Standard momentum check.
- Volume Analysis: Is the move supported by volume (OBV/MFI)?
- Anomaly Alerts: Any statistical anomalies triggering?
- SHOW: "Technicals: [Bullish/Bearish]. Volume: [Confirming/Diverging]."

**Phase 6: NEWS & ANALYST SENTIMENT (Weight: 10%)**
- Does breaking news override the data?
- Compare your view against Wall Street Consensus:
    - If you say SELL but Analysts say STRONG BUY -> You better have a massive "Idiosyncratic Risk" reason.
    - If you agree -> "Consensus Confirmation".

**Phase 7: UNPREDICTABLE RISK ASSESSMENT (CRITICAL - Confidence Penalty Zone)**
YOU MUST SCAN FOR THESE UNPREDICTABLE FACTORS THAT CAN INVALIDATE ALL OTHER ANALYSIS:

A. BLACK SWAN EVENTS (Immediate Confidence -30 if detected):
   - Pandemic/epidemic exposure in supply chain
   - Geopolitical risk (war, sanctions, trade wars)
   - Natural disaster vulnerability
   - Cybersecurity breach or data leak

B. RANDOM CORPORATE EVENTS (Confidence -20 if detected):
   - Recent CEO/CFO resignation or departure
   - Major lawsuit or SEC investigation pending
   - FDA rejection or regulatory crackdown
   - Unexpected earnings miss (>10% below consensus)
   - Accounting irregularities or restatements

C. MARKET MANIPULATION SIGNALS (Confidence -15 if detected):
   - Meme stock activity (WSB, Reddit mentions)
   - Unusual short interest (>30% of float)
   - Pump-and-dump patterns
   - Insider selling while recommending buy

D. SENTIMENT FRAGILITY (Confidence -10 if detected):
   - Social media controversy or boycott
   - ESG scandal (environmental, labor issues)
   - Customer backlash trending on social media
   - Analyst downgrades clustering in short period

E. EFFICIENT MARKET CONSIDERATIONS:
   - If stock has moved >15% recently, much of the opportunity may be priced in
   - If everyone is bullish, the easy money is made -> reduce conviction
   - Contrarian signals: extreme sentiment often precedes reversals

CONFIDENCE CALIBRATION RULES:
- Start with base confidence from ML models
- Apply penalties for EACH unpredictable risk detected
- Maximum confidence = 85% (never claim certainty in markets)
- If ANY black swan indicator detected -> cap confidence at 60%
- Be HONEST about what you cannot predict

**Phase 8: STRESS TEST ANALYSIS (CRISIS PATTERN DETECTION)**
Compare current market conditions to historical crisis patterns. Output warnings if similarities detected.

HISTORICAL CRISIS PATTERNS TO CHECK:
1. COVID CRASH (March 2020): VIX spike >40, rapid sector rotation, liquidity crisis
2. FED RATE HIKES (2022): Inverted yield curve, growth-to-value rotation, tech selloff
3. FLASH CRASHES: Unusual volume spikes, bid-ask spread widening
4. SECTOR BUBBLES: P/E ratios >50 in sector, extreme sentiment readings

STRESS TEST OUTPUT:
If current conditions match ANY crisis pattern (>60% similarity):
- Set stress_test_warning = true
- Identify which crisis pattern matches
- Estimate potential drawdown based on historical precedent
- Recommend defensive positioning


**Phase 7: DUAL-TIMEFRAME SYNTHESIS (CRITICAL UPGRADE)**
You must generate TWO distinct outputs:
1. TRADING VIEW (2-4 Weeks): Technicals + Momentum + Flows.
2. INVESTING VIEW (1-3 Years): Fundamentals + Moat + Growth.

=== RESPONSE FORMAT (Strict JSON) ===
{{
  "chain_of_thought": {{
    "phase_1_market_context": {{
      "regime": "<Analysis>",
      "sector_rank": "<Leader/Lagger>",
      "bias": "<Direction>"
    }},
    "phase_2_smart_money": {{
      "insider_signal": "<Buy/Sell/None>",
      "institutional_flow": "<Accumulation/Distribution>",
      "options_sentiment": "<Bullish/Bearish>",
      "score": <0-100>
    }},
    "phase_3_ml_ensemble": {{
      "ensemble_agreement": <true/false>,
      "ml_signal": "<Direction>"
    }},
    "phase_4_fundamental_thesis": {{
      "valuation_status": "<Cheap/Fair/Expensive>",
      "moat_status": "<Wide/Narrow/None>",
      "growth_outlook": "<Strong/Stable/Weak>"
    }},
    "phase_6_consensus_check": {{
      "analyst_view": "<Buy/Hold/Sell>",
      "ai_vs_street": "<Agree/Disagree>",
      "reasoning": "<Why you diverge or converge>"
    }}
  }},

  "agent_summary": "<CRITICAL: Write a comprehensive 3-5 sentence executive summary that EXPLAINS THE REASONING. Start with the verdict, then use 'because' statements to justify. Example: 'We recommend BUY because (1) technical indicators show oversold conditions with RSI at 32, (2) recent earnings beat consensus by 15% demonstrating strong execution, (3) insider purchases of $4.2M signal management confidence, and (4) ML models show 85% agreement on upward momentum. The primary risk is sector headwinds, but company-specific catalysts outweigh macro concerns.'>",
  
  "recommendation": {{
    "signal": "<STRONG BUY|BUY|HOLD|SELL|STRONG SELL>",
    "timeframe": "<Explicitly state timeframe, e.g. '2-4 Weeks' or '12 Months'>",
    "action": "<Enter Now / Wait for Pullback / Trim>"
  }},
  
  "confidence_score": <0-100>,
  
  "conclusion_drivers": [
    "<REASON 1: 'Because [specific evidence], we conclude [outcome]'>",
    "<REASON 2: 'Because [specific evidence], we conclude [outcome]'>",
    "<REASON 3: 'Because [specific evidence], we conclude [outcome]'>",
    "<Add 3-6 specific reasoning statements that justify the recommendation>"
  ],
  
  "scenarios": {{
    "bull_case": {{
      "probability": "<% eg 30%>",
      "target": "<$Price>",
      "catalyst": "<Driver>"
    }},
    "base_case": {{
      "probability": "<% eg 50%>",
      "target": "<$Price>",
      "thesis": "<Core Thesis>"
    }},
    "bear_case": {{
      "probability": "<% eg 20%>",
      "target": "<$Price>",
      "risk": "<Main Risk>"
    }}
  }},

  "web_intelligence_summary": {{
    "total_sources_analyzed": <number of sources you reviewed>,
    "key_news_items": [
      {{
        "headline": "<Exact headline from the news>",
        "source": "<Source name e.g. Reuters, Bloomberg>",
        "date": "<Date if available>",
        "significance": "<Why this matters for the investment thesis>"
      }},
      {{
        "headline": "<Another key news item>",
        "source": "<Source>",
        "date": "<Date>",
        "significance": "<Impact on stock>"
      }}
    ],
    "news_sentiment_summary": "<Overall sentiment from news: Positive/Negative/Mixed with brief explanation>",
    "key_catalysts_from_news": [
      "<Catalyst 1 identified from news>",
      "<Catalyst 2 identified from news>"
    ],
    "risks_from_news": [
      "<Risk 1 identified from news coverage>",
      "<Risk 2 identified from news coverage>"
    ]
  }},

  "key_drivers": [
    "<Driver 1 - substantive insight>",
    "<Driver 2 - substantive insight>",
    "<... PROVIDE AS MANY AS NEEDED - no minimum, no maximum ...>"
  ],
  
  "unpredictable_risks": {{
    "event_risk_score": <0-100, higher = more unpredictable events detected>,
    "confidence_penalty_applied": <total penalty applied to confidence, e.g. -25>,
    "black_swan_detected": <true/false>,
    "detected_risks": [
      {{
        "category": "<BLACK_SWAN|CORPORATE_EVENT|MANIPULATION|SENTIMENT_FRAGILITY>",
        "description": "<Specific risk detected from news or data>",
        "severity": "<CRITICAL|HIGH|MEDIUM|LOW>",
        "confidence_impact": <negative number, e.g. -15>
      }}
    ],
    "manipulation_warning": {{
      "detected": <true/false>,
      "signals": ["<Signal 1>", "<Signal 2>"],
      "short_interest_pct": <percentage if available, else null>
    }},
    "sentiment_fragility": {{
      "score": <0-100, higher = more fragile>,
      "reason": "<Why sentiment could shift suddenly>"
    }},
    "stress_test_warning": {{
      "triggered": <true/false>,
      "pattern_matched": "<COVID_CRASH|FED_HIKES|FLASH_CRASH|SECTOR_BUBBLE|NONE>",
      "similarity_pct": <0-100>,
      "estimated_drawdown": "<-5% to -10%|-10% to -20%|-20% to -40%|>-40%>",
      "defensive_action": "<Reduce position|Add hedges|Exit fully|Hold with stops>"
    }},
    "what_could_go_wrong": [
      "<Specific event 1 that could invalidate thesis>",
      "<Specific event 2 that could cause sudden loss>",
      "<Be brutally honest about unknowns>"
    ]
  }},
  
  "risk_assessment": {{
    "max_drawdown_risk": "<Low/Medium/High>",
    "risk_factors": [
      {{
        "factor": "<Risk Description - be comprehensive>",
        "impact": "<High/Medium/Low>",
        "probability": "<High/Medium/Low>"
      }},
      {{
        "factor": "<... ADD MORE RISK FACTORS AS NEEDED ...>",
        "impact": "<High/Medium/Low>",
        "probability": "<High/Medium/Low>"
      }}
    ]
  }}
}}

Return ONLY valid JSON.
"""

    try:
        response = model.generate_content(prompt)
        text = response.text.replace('```json', '').replace('```', '').strip()
        result = json.loads(text)
        
        # Cache successful response for consistency
        if use_cache:
            result['from_cache'] = False
            result['generated_at'] = datetime.now().isoformat()
            _response_cache[cache_key] = (result.copy(), datetime.now())
        
        return result
    except Exception as e:
        logger.error(f"Gemini AI failed: {e}")
        return {
            "error": str(e),
            "summary": "AI Agent failed to synthesize data.",
            "recommendation": "HOLD",
            "from_cache": False
        }
