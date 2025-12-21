"""
Growth Investing Strategy Framework
"""

from typing import Dict, Any, List

def analyze_growth_metrics(
    metrics: Dict[str, float]
) -> Dict[str, Any]:
    """
    Analyze stock based on growth investing criteria.
    
    Criteria (CAN SLIM inspired):
    - Current earnings growth > 20%
    - Annual earnings growth > 25%
    - New product/management/highs
    - Supply/Demand (Volume)
    - Leader in industry
    - Institutional sponsorship
    - Market direction
    """
    score = 0
    max_score = 5
    
    eps_growth = metrics.get('eps_growth', 0)
    rev_growth = metrics.get('revenue_growth', 0)
    roe = metrics.get('roe', 0)
    relative_strength = metrics.get('relative_strength', 0)
    
    details = []
    
    if eps_growth > 20:
        score += 1
        details.append("Strong EPS growth (>20%)")
    
    if rev_growth > 25:
        score += 1
        details.append("Strong Revenue growth (>25%)")
        
    if roe > 17:
        score += 1
        details.append("High ROE (>17%)")
        
    if relative_strength > 80:
        score += 1
        details.append("Strong Relative Strength (>80)")
        
    prospects = "Strong Buy" if score >= 4 else "Buy" if score == 3 else "Hold" if score == 2 else "Avoid"
    
    return {
        "score": f"{score}/{max_score}",
        "rating": prospects,
        "criteria_met": details
    }
