"""
Fundamental Analysis Predictor

Scores stocks based on fundamentals:
- Valuation Metrics (50%): P/E, PEG, P/B, P/S, EV/EBITDA
- Quality Metrics (30%): ROE, ROA, Profit Margins, Debt ratios
- Growth Metrics (20%): Revenue growth, Earnings growth, EPS growth

Scoring Logic:
- Undervalued + High Quality + Growing = Strong Buy
- Overvalued + Low Quality = Strong Sell
- Mixed signals = Neutral
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


def get_fundamental_prediction(symbol: str) -> Dict[str, Any]:
    """
    Generate fundamental analysis prediction for a stock.

    Args:
        symbol: Stock ticker symbol

    Returns:
        Fundamental analysis prediction
    """
    try:
        import yfinance as yf

        # Get company info
        ticker = yf.Ticker(symbol)
        info = ticker.info

        if not info or 'symbol' not in info:
            return {
                "error": f"Could not fetch data for {symbol}",
                "direction": "Neutral",
                "confidence": 50,
                "status": "failed"
            }

        scores = {}
        breakdown = {}

        # === VALUATION METRICS (50% weight) ===
        valuation_score = 0
        valuation_count = 0

        # P/E Ratio (lower is better, typically < 25 is good)
        pe_ratio = info.get('trailingPE') or info.get('forwardPE')
        if pe_ratio and pe_ratio > 0:
            if pe_ratio < 15:
                pe_score = 100
            elif pe_ratio < 25:
                pe_score = 70
            elif pe_ratio < 35:
                pe_score = 40
            else:
                pe_score = 20
            scores['pe_ratio'] = pe_score
            breakdown['pe_ratio'] = round(pe_ratio, 2)
            valuation_score += pe_score
            valuation_count += 1

        # PEG Ratio (< 1 is undervalued)
        peg_ratio = info.get('pegRatio')
        if peg_ratio and peg_ratio > 0:
            if peg_ratio < 1:
                peg_score = 100
            elif peg_ratio < 1.5:
                peg_score = 70
            elif peg_ratio < 2:
                peg_score = 40
            else:
                peg_score = 20
            scores['peg_ratio'] = peg_score
            breakdown['peg_ratio'] = round(peg_ratio, 2)
            valuation_score += peg_score
            valuation_count += 1

        # P/B Ratio (lower is better)
        pb_ratio = info.get('priceToBook')
        if pb_ratio and pb_ratio > 0:
            if pb_ratio < 1:
                pb_score = 100
            elif pb_ratio < 3:
                pb_score = 70
            elif pb_ratio < 5:
                pb_score = 40
            else:
                pb_score = 20
            scores['pb_ratio'] = pb_score
            breakdown['pb_ratio'] = round(pb_ratio, 2)
            valuation_score += pb_score
            valuation_count += 1

        # P/S Ratio
        ps_ratio = info.get('priceToSalesTrailing12Months')
        if ps_ratio and ps_ratio > 0:
            if ps_ratio < 1:
                ps_score = 100
            elif ps_ratio < 2:
                ps_score = 70
            elif ps_ratio < 5:
                ps_score = 40
            else:
                ps_score = 20
            scores['ps_ratio'] = ps_score
            breakdown['ps_ratio'] = round(ps_ratio, 2)
            valuation_score += ps_score
            valuation_count += 1

        valuation_score = (valuation_score /
                           valuation_count) if valuation_count > 0 else 50

        # === QUALITY METRICS (30% weight) ===
        quality_score = 0
        quality_count = 0

        # ROE (> 15% is good)
        roe = info.get('returnOnEquity')
        if roe:
            roe_pct = roe * 100
            if roe_pct > 20:
                roe_score = 100
            elif roe_pct > 15:
                roe_score = 80
            elif roe_pct > 10:
                roe_score = 60
            else:
                roe_score = 30
            scores['roe'] = roe_score
            breakdown['roe'] = round(roe_pct, 2)
            quality_score += roe_score
            quality_count += 1

        # ROA (> 5% is good)
        roa = info.get('returnOnAssets')
        if roa:
            roa_pct = roa * 100
            if roa_pct > 10:
                roa_score = 100
            elif roa_pct > 5:
                roa_score = 75
            elif roa_pct > 2:
                roa_score = 50
            else:
                roa_score = 25
            scores['roa'] = roa_score
            breakdown['roa'] = round(roa_pct, 2)
            quality_score += roa_score
            quality_count += 1

        # Profit Margin
        profit_margin = info.get('profitMargins')
        if profit_margin:
            margin_pct = profit_margin * 100
            if margin_pct > 20:
                margin_score = 100
            elif margin_pct > 10:
                margin_score = 75
            elif margin_pct > 5:
                margin_score = 50
            else:
                margin_score = 25
            scores['profit_margin'] = margin_score
            breakdown['profit_margin'] = round(margin_pct, 2)
            quality_score += margin_score
            quality_count += 1

        # Debt to Equity (lower is better, <1 is good)
        debt_to_equity = info.get('debtToEquity')
        if debt_to_equity is not None:
            if debt_to_equity < 50:
                debt_score = 100
            elif debt_to_equity < 100:
                debt_score = 75
            elif debt_to_equity < 200:
                debt_score = 40
            else:
                debt_score = 20
            scores['debt_to_equity'] = debt_score
            breakdown['debt_to_equity'] = round(debt_to_equity, 2)
            quality_score += debt_score
            quality_count += 1

        # Current Ratio (> 1.5 is healthy)
        current_ratio = info.get('currentRatio')
        if current_ratio:
            if current_ratio > 2:
                curr_score = 100
            elif current_ratio > 1.5:
                curr_score = 80
            elif current_ratio > 1:
                curr_score = 50
            else:
                curr_score = 25
            scores['current_ratio'] = curr_score
            breakdown['current_ratio'] = round(current_ratio, 2)
            quality_score += curr_score
            quality_count += 1

        quality_score = (
            quality_score / quality_count) if quality_count > 0 else 50

        # === GROWTH METRICS (20% weight) ===
        growth_score = 0
        growth_count = 0

        # Revenue Growth
        revenue_growth = info.get('revenueGrowth')
        if revenue_growth:
            rev_pct = revenue_growth * 100
            if rev_pct > 20:
                rev_score = 100
            elif rev_pct > 10:
                rev_score = 75
            elif rev_pct > 5:
                rev_score = 50
            elif rev_pct > 0:
                rev_score = 30
            else:
                rev_score = 10
            scores['revenue_growth'] = rev_score
            breakdown['revenue_growth'] = round(rev_pct, 2)
            growth_score += rev_score
            growth_count += 1

        # Earnings Growth
        earnings_growth = info.get('earningsGrowth')
        if earnings_growth:
            earn_pct = earnings_growth * 100
            if earn_pct > 20:
                earn_score = 100
            elif earn_pct > 10:
                earn_score = 75
            elif earn_pct > 5:
                earn_score = 50
            elif earn_pct > 0:
                earn_score = 30
            else:
                earn_score = 10
            scores['earnings_growth'] = earn_score
            breakdown['earnings_growth'] = round(earn_pct, 2)
            growth_score += earn_score
            growth_count += 1

        growth_score = (
            growth_score / growth_count) if growth_count > 0 else 50

        # === COMBINED SCORE ===
        total_score = (
            valuation_score * 0.5 +
            quality_score * 0.3 +
            growth_score * 0.2
        )

        # Direction based on combined score
        if total_score >= 75:
            direction = "Bullish"
            recommendation = "Strong Buy"
        elif total_score >= 60:
            direction = "Bullish"
            recommendation = "Buy"
        elif total_score >= 40:
            direction = "Neutral"
            recommendation = "Hold"
        elif total_score >= 25:
            direction = "Bearish"
            recommendation = "Sell"
        else:
            direction = "Bearish"
            recommendation = "Strong Sell"

        # Confidence based on data availability
        data_completeness = (valuation_count + quality_count +
                             growth_count) / 10  # Max 10 metrics
        confidence = min(85, max(40, total_score * data_completeness))

        return {
            "direction": direction,
            "confidence": round(confidence, 1),
            "recommendation": recommendation,
            "fundamental_score": round(total_score, 1),
            "breakdown": {
                "valuation_score": round(valuation_score, 1),
                "quality_score": round(quality_score, 1),
                "growth_score": round(growth_score, 1)
            },
            "metrics": breakdown,
            "company_name": info.get('longName', symbol),
            "sector": info.get('sector', 'Unknown'),
            "industry": info.get('industry', 'Unknown'),
            "timestamp": datetime.now().isoformat(),
            "method": "fundamental_analysis",
            "status": "success"
        }

    except ImportError:
        return {
            "error": "yfinance not installed",
            "direction": "Neutral",
            "confidence": 50,
            "status": "failed"
        }
    except Exception as e:
        logger.error(f"Error in fundamental analysis: {e}", exc_info=True)
        return {
            "error": str(e),
            "direction": "Neutral",
            "confidence": 50,
            "status": "failed"
        }
