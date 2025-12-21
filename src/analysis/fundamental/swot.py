"""
SWOT Analysis Framework
Dynamic SWOT generation based on financial metrics
"""

from typing import Dict, List, Any, Optional


def generate_swot_analysis(
    company_data: Dict[str, Any],
    industry_avg: Optional[Dict[str, float]] = None
) -> Dict[str, List[str]]:
    """
    Generate SWOT analysis from company financial data.
    
    Args:
        company_data: Company financial metrics
        industry_avg: Industry average metrics for comparison
    
    Returns:
        Dictionary with Strengths, Weaknesses, Opportunities, Threats
    """
    strengths = []
    weaknesses = []
    opportunities = []
    threats = []
    
    # Default industry averages
    if industry_avg is None:
        industry_avg = {
            'pe_ratio': 20,
            'profit_margin': 10,
            'roe': 15,
            'debt_to_equity': 1.0,
            'current_ratio': 1.5,
            'dividend_yield': 2.0,
            'revenue_growth': 10
        }
    
    # STRENGTHS Analysis
    # Profitability
    if company_data.get('profit_margin', 0) > industry_avg.get('profit_margin', 10):
        strengths.append(f"Strong profit margin ({company_data['profit_margin']:.1f}% vs industry avg {industry_avg['profit_margin']}%)")
    
    if company_data.get('roe', 0) > industry_avg.get('roe', 15):
        strengths.append(f"Superior return on equity ({company_data['roe']:.1f}%)")
    
    # Financial Health
    if company_data.get('current_ratio', 0) > 2:
        strengths.append("Strong liquidity position")
    
    if company_data.get('debt_to_equity', 1) < 0.5:
        strengths.append("Low debt levels provide financial flexibility")
    
    if company_data.get('cash_position', 0) > 0:
        strengths.append("Strong cash reserves")
    
    # Market Position
    if company_data.get('market_cap', 0) > 100e9:
        strengths.append("Large market cap provides stability and resources")
    
    if company_data.get('brand_value'):
        strengths.append("Strong brand recognition")
    
    # Dividends
    if company_data.get('dividend_yield', 0) > industry_avg.get('dividend_yield', 2):
        strengths.append(f"Attractive dividend yield ({company_data['dividend_yield']:.2f}%)")
    
    if company_data.get('dividend_growth_years', 0) > 10:
        strengths.append(f"Consistent dividend growth for {company_data['dividend_growth_years']} years")
    
    # WEAKNESSES Analysis
    if company_data.get('profit_margin', 100) < industry_avg.get('profit_margin', 10):
        weaknesses.append(f"Below-average profit margin ({company_data.get('profit_margin', 0):.1f}%)")
    
    if company_data.get('debt_to_equity', 0) > 2:
        weaknesses.append(f"High debt levels (D/E: {company_data['debt_to_equity']:.2f})")
    
    if company_data.get('current_ratio', 0) < 1:
        weaknesses.append("Potential liquidity concerns")
    
    if company_data.get('revenue_growth', 0) < 0:
        weaknesses.append("Declining revenue")
    
    if company_data.get('pe_ratio', 0) > 40:
        weaknesses.append("High valuation may limit upside")
    
    if company_data.get('concentration_risk'):
        weaknesses.append("Revenue concentration risk")
    
    # OPPORTUNITIES Analysis
    if company_data.get('revenue_growth', 0) > industry_avg.get('revenue_growth', 10):
        opportunities.append("Outpacing industry growth rate")
    
    if company_data.get('emerging_markets_exposure'):
        opportunities.append("Expansion potential in emerging markets")
    
    if company_data.get('rd_investment', 0) > 0:
        opportunities.append("R&D investment driving innovation pipeline")
    
    if company_data.get('digital_transformation'):
        opportunities.append("Digital transformation initiatives")
    
    if company_data.get('market_share', 0) < 50:
        opportunities.append("Room for market share gains")
    
    opportunities.append("Potential M&A or strategic partnerships")
    opportunities.append("AI and automation adoption for efficiency")
    
    # THREATS Analysis
    if company_data.get('competitive_intensity', 0) > 7:
        threats.append("Intense competitive environment")
    
    if company_data.get('regulatory_risk'):
        threats.append("Regulatory uncertainty and compliance costs")
    
    threats.append("Macroeconomic uncertainty and recession risk")
    threats.append("Supply chain disruption potential")
    
    if company_data.get('technology_disruption_risk'):
        threats.append("Technology disruption risk in the industry")
    
    if company_data.get('interest_rate_sensitivity'):
        threats.append("Interest rate sensitivity on borrowing costs")
    
    if company_data.get('currency_exposure'):
        threats.append("Foreign exchange rate fluctuations")
    
    return {
        'strengths': strengths[:5],  # Top 5
        'weaknesses': weaknesses[:5],
        'opportunities': opportunities[:5],
        'threats': threats[:5]
    }


def swot_score(swot: Dict[str, List[str]]) -> Dict[str, Any]:
    """
    Calculate overall SWOT score.
    
    Args:
        swot: SWOT analysis result
    
    Returns:
        Dictionary with scores
    """
    positive_score = len(swot['strengths']) + len(swot['opportunities'])
    negative_score = len(swot['weaknesses']) + len(swot['threats'])
    
    total_score = positive_score - negative_score
    
    if total_score >= 4:
        outlook = 'Strong Positive'
    elif total_score >= 1:
        outlook = 'Moderately Positive'
    elif total_score >= -2:
        outlook = 'Neutral'
    else:
        outlook = 'Cautious'
    
    return {
        'strengths_count': len(swot['strengths']),
        'weaknesses_count': len(swot['weaknesses']),
        'opportunities_count': len(swot['opportunities']),
        'threats_count': len(swot['threats']),
        'positive_score': positive_score,
        'negative_score': negative_score,
        'net_score': total_score,
        'outlook': outlook
    }


def create_swot_from_info(info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create SWOT analysis from yfinance company info.
    
    Args:
        info: Company info from yfinance
    
    Returns:
        Complete SWOT analysis with scores
    """
    # Extract relevant metrics
    company_data = {
        'profit_margin': (info.get('profitMargins', 0) or 0) * 100,
        'roe': (info.get('returnOnEquity', 0) or 0) * 100,
        'debt_to_equity': info.get('debtToEquity', 100) / 100 if info.get('debtToEquity') else 1,
        'current_ratio': info.get('currentRatio', 0) or 0,
        'dividend_yield': (info.get('dividendYield', 0) or 0) * 100,
        'pe_ratio': info.get('trailingPE', 0) or 0,
        'revenue_growth': (info.get('revenueGrowth', 0) or 0) * 100,
        'market_cap': info.get('marketCap', 0) or 0,
        'cash_position': info.get('totalCash', 0) or 0
    }
    
    swot = generate_swot_analysis(company_data)
    scores = swot_score(swot)
    
    return {
        'swot': swot,
        'scores': scores,
        'company_name': info.get('shortName', 'Unknown'),
        'sector': info.get('sector', 'Unknown'),
        'industry': info.get('industry', 'Unknown')
    }
