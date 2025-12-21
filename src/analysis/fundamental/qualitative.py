"""
Qualitative Fundamental Analysis Module
ESG scoring, moat assessment, and management quality indicators.
"""

from typing import Dict, Any, List, Optional
import yfinance as yf


def esg_score_analysis(symbol: str) -> Dict[str, Any]:
    """
    Extract and analyze ESG (Environmental, Social, Governance) scores.
    
    Uses available data from yfinance to assess sustainability.
    
    Args:
        symbol: Stock ticker symbol
    
    Returns:
        Dictionary with ESG metrics and interpretation
    """
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        # Extract ESG-related data
        result = {
            'environmental': {},
            'social': {},
            'governance': {},
            'overall_score': None
        }
        
        # Environmental factors
        sector = info.get('sector', 'Unknown')
        industry = info.get('industry', 'Unknown')
        
        # High-impact sectors for environmental
        high_env_impact = ['Energy', 'Utilities', 'Basic Materials', 'Industrials']
        env_concern = sector in high_env_impact
        
        result['environmental'] = {
            'sector': sector,
            'high_impact_sector': env_concern,
            'note': 'High environmental scrutiny' if env_concern else 'Moderate environmental profile'
        }
        
        # Social factors
        employee_count = info.get('fullTimeEmployees', 0)
        
        result['social'] = {
            'employees': employee_count,
            'employee_scale': 'Large' if employee_count > 10000 else 'Medium' if employee_count > 1000 else 'Small'
        }
        
        # Governance factors
        result['governance'] = {
            'has_board_info': 'companyOfficers' in info,
            'public_company': True,  # Listed companies
            'country': info.get('country', 'Unknown'),
            'exchange': info.get('exchange', 'Unknown')
        }
        
        # Simple scoring (0-100)
        # This is a simplified model - real ESG requires specialized data
        scores = []
        
        # Environmental score
        env_score = 60 if not env_concern else 40
        scores.append(env_score)
        
        # Social score (larger companies usually have more HR policies)
        social_score = 70 if employee_count > 5000 else 60 if employee_count > 500 else 50
        scores.append(social_score)
        
        # Governance score (public companies have disclosure requirements)
        gov_score = 65
        scores.append(gov_score)
        
        result['scores'] = {
            'environmental': env_score,
            'social': social_score,
            'governance': gov_score
        }
        result['overall_score'] = round(sum(scores) / len(scores), 1)
        
        # Rating
        overall = result['overall_score']
        if overall >= 70:
            result['rating'] = 'A - Strong ESG'
        elif overall >= 60:
            result['rating'] = 'B - Good ESG'
        elif overall >= 50:
            result['rating'] = 'C - Average ESG'
        else:
            result['rating'] = 'D - Below Average ESG'
        
        result['disclaimer'] = 'Simplified ESG estimate. For accurate ESG data, use specialized providers like MSCI, Sustainalytics, or Refinitiv.'
        
        return result
        
    except Exception as e:
        return {'error': str(e)}


def moat_assessment(symbol: str) -> Dict[str, Any]:
    """
    Assess economic moat (competitive advantage durability).
    
    Analyzes factors that create sustainable competitive advantages:
    - Brand power
    - Switching costs
    - Network effects
    - Cost advantages
    - Efficient scale
    - Intangible assets
    
    Args:
        symbol: Stock ticker symbol
    
    Returns:
        Dictionary with moat assessment
    """
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        moat_factors = {}
        moat_score = 0
        max_score = 100
        
        # 1. Profitability (high margins suggest pricing power)
        gross_margin = info.get('grossMargins', 0) or 0
        operating_margin = info.get('operatingMargins', 0) or 0
        
        if gross_margin > 0.5:  # 50%+ gross margin
            moat_factors['high_margins'] = {
                'present': True,
                'gross_margin': round(gross_margin * 100, 1),
                'implies': 'Pricing power / brand strength'
            }
            moat_score += 20
        else:
            moat_factors['high_margins'] = {
                'present': False,
                'gross_margin': round(gross_margin * 100, 1)
            }
        
        # 2. Market position (market cap as proxy for scale)
        market_cap = info.get('marketCap', 0) or 0
        
        if market_cap > 100e9:  # $100B+
            moat_factors['scale_advantage'] = {
                'present': True,
                'market_cap': market_cap,
                'implies': 'Efficient scale / cost advantages'
            }
            moat_score += 15
        elif market_cap > 10e9:
            moat_factors['scale_advantage'] = {
                'present': 'Partial',
                'market_cap': market_cap
            }
            moat_score += 8
        else:
            moat_factors['scale_advantage'] = {'present': False}
        
        # 3. Return on Invested Capital (high ROIC = moat)
        # Using ROE as proxy
        roe = info.get('returnOnEquity', 0) or 0
        
        if roe > 0.20:  # 20%+
            moat_factors['high_roic'] = {
                'present': True,
                'roe': round(roe * 100, 1),
                'implies': 'Sustainable competitive advantage'
            }
            moat_score += 25
        elif roe > 0.10:
            moat_factors['high_roic'] = {
                'present': 'Moderate',
                'roe': round(roe * 100, 1)
            }
            moat_score += 12
        else:
            moat_factors['high_roic'] = {'present': False}
        
        # 4. Revenue growth (moatful companies can grow)
        revenue_growth = info.get('revenueGrowth', 0) or 0
        
        if revenue_growth > 0.15:
            moat_factors['growth_capability'] = {
                'present': True,
                'revenue_growth': round(revenue_growth * 100, 1)
            }
            moat_score += 15
        elif revenue_growth > 0:
            moat_factors['growth_capability'] = {
                'present': 'Moderate',
                'revenue_growth': round(revenue_growth * 100, 1)
            }
            moat_score += 7
        else:
            moat_factors['growth_capability'] = {'present': False}
        
        # 5. Industry characteristics (some industries have natural moats)
        sector = info.get('sector', '')
        moat_friendly_sectors = ['Technology', 'Healthcare', 'Consumer Cyclical', 'Financial Services']
        
        if sector in moat_friendly_sectors:
            moat_factors['favorable_industry'] = {
                'present': True,
                'sector': sector,
                'implies': 'Industry allows for durable advantages'
            }
            moat_score += 10
        else:
            moat_factors['favorable_industry'] = {
                'present': False,
                'sector': sector
            }
        
        # Calculate normalized score
        normalized_score = round(moat_score, 1)
        
        # Moat classification
        if normalized_score >= 70:
            moat_rating = 'Wide Moat'
            moat_description = 'Strong, durable competitive advantages likely to persist 20+ years'
        elif normalized_score >= 50:
            moat_rating = 'Narrow Moat'
            moat_description = 'Modest competitive advantages, may persist 10+ years'
        else:
            moat_rating = 'No Moat'
            moat_description = 'Limited competitive advantages, vulnerable to competition'
        
        return {
            'symbol': symbol,
            'moat_rating': moat_rating,
            'moat_score': normalized_score,
            'description': moat_description,
            'factors': moat_factors,
            'methodology': 'Analysis based on profitability, scale, returns, growth, and industry structure'
        }
        
    except Exception as e:
        return {'error': str(e)}


def management_quality_indicators(symbol: str) -> Dict[str, Any]:
    """
    Assess management quality based on available metrics.
    
    Considers:
    - Insider ownership (alignment with shareholders)
    - Capital allocation (dividends, buybacks)
    - Earnings quality
    - Executive compensation (relative to performance)
    
    Args:
        symbol: Stock ticker symbol
    
    Returns:
        Dictionary with management quality assessment
    """
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        indicators = {}
        quality_score = 50  # Start neutral
        
        # 1. Insider ownership (alignment)
        insider_pct = info.get('heldPercentInsiders', 0) or 0
        
        if insider_pct > 0.10:  # 10%+ insider ownership
            indicators['insider_alignment'] = {
                'status': 'Strong',
                'insider_ownership': round(insider_pct * 100, 2),
                'interpretation': 'Management has significant skin in the game'
            }
            quality_score += 15
        elif insider_pct > 0.02:
            indicators['insider_alignment'] = {
                'status': 'Moderate',
                'insider_ownership': round(insider_pct * 100, 2)
            }
            quality_score += 5
        else:
            indicators['insider_alignment'] = {
                'status': 'Low',
                'insider_ownership': round(insider_pct * 100, 2)
            }
        
        # 2. Capital allocation (dividends and debt management)
        dividend_yield = info.get('dividendYield', 0) or 0
        payout_ratio = info.get('payoutRatio', 0) or 0
        
        if dividend_yield > 0 and payout_ratio < 0.75:
            indicators['capital_allocation'] = {
                'status': 'Prudent',
                'dividend_yield': round(dividend_yield * 100, 2),
                'payout_ratio': round(payout_ratio * 100, 2),
                'interpretation': 'Sustainable dividend policy'
            }
            quality_score += 10
        elif dividend_yield > 0 and payout_ratio >= 0.75:
            indicators['capital_allocation'] = {
                'status': 'Aggressive',
                'dividend_yield': round(dividend_yield * 100, 2),
                'payout_ratio': round(payout_ratio * 100, 2),
                'interpretation': 'High payout may limit reinvestment'
            }
        else:
            indicators['capital_allocation'] = {
                'status': 'Growth-focused',
                'interpretation': 'Retaining earnings for reinvestment'
            }
            quality_score += 5
        
        # 3. Earnings quality (compare EPS trends)
        trailing_eps = info.get('trailingEps', 0) or 0
        forward_eps = info.get('forwardEps', 0) or 0
        
        if trailing_eps > 0 and forward_eps > trailing_eps:
            indicators['earnings_quality'] = {
                'status': 'Growing',
                'trailing_eps': trailing_eps,
                'forward_eps': forward_eps,
                'expected_growth': round((forward_eps - trailing_eps) / trailing_eps * 100, 1) if trailing_eps > 0 else 0
            }
            quality_score += 15
        elif trailing_eps > 0:
            indicators['earnings_quality'] = {
                'status': 'Stable',
                'trailing_eps': trailing_eps,
                'forward_eps': forward_eps
            }
            quality_score += 5
        else:
            indicators['earnings_quality'] = {
                'status': 'Unprofitable/Uncertain',
                'trailing_eps': trailing_eps
            }
            quality_score -= 10
        
        # 4. Track record (company age and consistency)
        # Using sector and industry presence as proxy
        employees = info.get('fullTimeEmployees', 0) or 0
        
        if employees > 50000:
            indicators['operational_scale'] = {
                'status': 'Established',
                'employees': employees,
                'interpretation': 'Large organization suggests mature management'
            }
            quality_score += 10
        elif employees > 1000:
            indicators['operational_scale'] = {
                'status': 'Growing',
                'employees': employees
            }
            quality_score += 5
        
        # Final rating
        if quality_score >= 80:
            rating = 'Excellent'
        elif quality_score >= 65:
            rating = 'Good'
        elif quality_score >= 50:
            rating = 'Average'
        else:
            rating = 'Below Average'
        
        return {
            'symbol': symbol,
            'quality_score': quality_score,
            'rating': rating,
            'indicators': indicators,
            'note': 'Based on quantitative proxies. For full assessment, review proxy statements and earnings calls.'
        }
        
    except Exception as e:
        return {'error': str(e)}


def business_model_durability(symbol: str) -> Dict[str, Any]:
    """
    Assess business model durability and adaptability.
    
    Args:
        symbol: Stock ticker symbol
    
    Returns:
        Dictionary with business model assessment
    """
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        assessment = {}
        durability_score = 50
        
        # Revenue diversification (single product risk)
        sector = info.get('sector', 'Unknown')
        industry = info.get('industry', 'Unknown')
        
        # Recurring revenue potential (based on industry)
        high_recurring = ['Software—Application', 'Software—Infrastructure', 
                          'Insurance', 'Banks', 'REITs', 'Utilities']
        
        if industry in high_recurring:
            assessment['revenue_model'] = {
                'type': 'Recurring/Subscription likely',
                'durability': 'High',
                'interpretation': 'Predictable revenue streams'
            }
            durability_score += 20
        else:
            assessment['revenue_model'] = {
                'type': 'Transaction-based likely',
                'durability': 'Moderate'
            }
        
        # Customer concentration risk (using market position as proxy)
        market_cap = info.get('marketCap', 0) or 0
        
        if market_cap > 50e9:
            assessment['market_position'] = {
                'status': 'Market leader',
                'interpretation': 'Diversified customer base likely'
            }
            durability_score += 15
        else:
            assessment['market_position'] = {
                'status': 'Challenger',
                'interpretation': 'May have customer concentration'
            }
        
        # Technology/Disruption risk
        tech_disruption_risk = ['Retail', 'Media', 'Traditional Banking', 'Print Media']
        low_disruption_risk = ['Utilities', 'Healthcare Providers', 'Defense']
        
        if industry in low_disruption_risk:
            assessment['disruption_risk'] = {
                'level': 'Low',
                'interpretation': 'Regulated or essential services'
            }
            durability_score += 15
        elif sector in ['Technology']:
            assessment['disruption_risk'] = {
                'level': 'Medium',
                'interpretation': 'Tech companies can be both disruptors and disrupted'
            }
        else:
            assessment['disruption_risk'] = {
                'level': 'Variable',
                'interpretation': 'Depends on specific competitive dynamics'
            }
        
        # Final classification
        if durability_score >= 75:
            durability_rating = 'High Durability'
        elif durability_score >= 55:
            durability_rating = 'Moderate Durability'
        else:
            durability_rating = 'Low Durability'
        
        return {
            'symbol': symbol,
            'durability_score': durability_score,
            'rating': durability_rating,
            'sector': sector,
            'industry': industry,
            'assessment': assessment
        }
        
    except Exception as e:
        return {'error': str(e)}
