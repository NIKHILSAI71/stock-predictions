"""
Value Investing Strategy Framework
"""

from typing import Dict, Any

def analyze_value_metrics(
    metrics: Dict[str, float]
) -> Dict[str, Any]:
    """
    Analyze stock based on value investing criteria.
    
    Criteria (Graham/Buffett inspired):
    - Low P/E (< 15 or < Industry Avg)
    - P/B < 1.5
    - Strong current ratio (> 1.5)
    - Consistent dividend history
    - Low Debt/Equity
    """
    score = 0
    max_score = 5
    
    pe = metrics.get('pe_ratio', float('inf'))
    pb = metrics.get('pb_ratio', float('inf'))
    current_ratio = metrics.get('current_ratio', 0)
    dept_equity = metrics.get('debt_to_equity', float('inf'))
    div_yield = metrics.get('dividend_yield', 0)
    
    details = []
    
    if pe < 15 and pe > 0:
        score += 1
        details.append("Attractive P/E (<15)")
        
    if pb < 1.5:
        score += 1
        details.append("Low P/B (<1.5)")
        
    if current_ratio > 1.5:
        score += 1
        details.append("Healthy Liquidity (Current Ratio > 1.5)")
        
    if dept_equity < 0.5:
        score += 1
        details.append("Low Debt (<0.5 D/E)")
        
    if div_yield > 2.0:
        score += 1
        details.append("Good Dividend Yield (>2%)")

    prospects = "Strong Buy" if score >= 4 else "Buy" if score == 3 else "Hold" if score == 2 else "Avoid"
    
    return {
        "score": f"{score}/{max_score}",
        "rating": prospects,
        "criteria_met": details
    }
