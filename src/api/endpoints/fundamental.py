from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse

from src.core.utils import sanitize_for_json
from src.data import get_company_info, validate_symbol
from src.analysis.fundamental.valuation import (
    calculate_pe_ratio, calculate_peg_ratio, calculate_pb_ratio,
    calculate_ps_ratio, calculate_ev_ebitda, gordon_growth_model
)
from src.data.sentiment import corporate_events_summary
import math

router = APIRouter()


@router.get("/fundamental/{symbol}")
async def get_fundamental_analysis(symbol: str):
    """Get fundamental ratios for a stock."""
    try:
        info = get_company_info(symbol)

        fundamentals = {}

        # Valuation ratios
        if info.get('trailing_pe'):
            fundamentals['pe_ratio'] = info['trailing_pe']
        if info.get('forward_pe'):
            fundamentals['forward_pe'] = info['forward_pe']
        if info.get('peg_ratio'):
            fundamentals['peg_ratio'] = info['peg_ratio']
        if info.get('price_to_book'):
            fundamentals['pb_ratio'] = info['price_to_book']
        if info.get('price_to_sales'):
            fundamentals['ps_ratio'] = info['price_to_sales']

        # Profitability
        if info.get('net_income') and info.get('revenue'):
            margin = (info['net_income'] / info['revenue']) * 100
            fundamentals['net_margin'] = round(margin, 2)
        if info.get('return_on_equity'):
            fundamentals['roe'] = round(info['return_on_equity'] * 100, 2)
        if info.get('return_on_assets'):
            fundamentals['roa'] = round(info['return_on_assets'] * 100, 2)
        if info.get('operating_margins'):
            fundamentals['operating_margin'] = round(
                info['operating_margins'] * 100, 2)

        # Liquidity & Solvency
        if info.get('current_ratio'):
            fundamentals['current_ratio'] = round(info['current_ratio'], 2)
        if info.get('quick_ratio'):
            fundamentals['quick_ratio'] = round(info['quick_ratio'], 2)
        if info.get('debt_to_equity'):
            fundamentals['debt_to_equity'] = round(info['debt_to_equity'], 2)

        # Dividend
        if info.get('dividend_yield'):
            fundamentals['dividend_yield'] = round(
                info['dividend_yield'] * 100, 2)
        if info.get('dividend_rate'):
            fundamentals['dividend_rate'] = info['dividend_rate']

        # Market data
        fundamentals['market_cap'] = info.get('market_cap')
        fundamentals['enterprise_value'] = info.get('enterprise_value')
        fundamentals['beta'] = info.get('beta')
        fundamentals['shares_outstanding'] = info.get('shares_outstanding')

        # Earnings
        fundamentals['eps_trailing'] = info.get('eps_trailing')
        fundamentals['eps_forward'] = info.get('eps_forward')

        # Revenue & Profit
        fundamentals['revenue'] = info.get('revenue')
        fundamentals['ebitda'] = info.get('ebitda')
        fundamentals['net_income'] = info.get('net_income')

        # Balance sheet highlights
        fundamentals['total_debt'] = info.get('total_debt')
        fundamentals['total_cash'] = info.get('total_cash')
        fundamentals['book_value'] = info.get('book_value')

        return {
            "status": "success",
            "symbol": symbol.upper(),
            "fundamentals": fundamentals
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/valuation/{symbol}")
async def get_valuation_analysis(
    symbol: str,
    required_return: float = Query(
        0.10, description="Required rate of return (decimal)")
):
    """Get stock valuation using multiple methods."""
    try:
        info = get_company_info(symbol)

        valuation = {}

        # Current price
        current_price = info.get('current_price') or info.get(
            'price_info', {}).get('current_price')
        valuation['current_price'] = current_price

        # P/E based valuation
        if pe := info.get('trailing_pe'):
            if eps := info.get('eps_trailing'):
                valuation['pe_valuation'] = {
                    'pe_ratio': pe,
                    'eps': eps,
                    'implied_price': round(pe * eps, 2)
                }

        # DCF Valuation (Simple 2-stage model approximation if data allows)
        dcf_value = None
        if info.get('free_cash_flow') and info.get('beta'):
            try:
                fcf = info['free_cash_flow']
                beta = info['beta']
                rf_rate = 0.045  # 4.5% Risk Free (Approx)
                market_return = 0.10  # 10% Market Return
                cost_of_equity = rf_rate + beta * (market_return - rf_rate)

                # Assume 5% terminal growth, 10% short term growth (generic if missing)
                growth_rate = info.get('earnings_growth', 0.10)
                if not growth_rate:
                    growth_rate = 0.10

                # 5 Year Projection
                future_cash_flows = []
                for i in range(1, 6):
                    future_cash_flows.append(fcf * ((1 + growth_rate) ** i))

                # 3% Perp growth
                terminal_value = (
                    future_cash_flows[-1] * (1 + 0.03)) / (cost_of_equity - 0.03)

                dcf_total = sum([fcf / ((1 + cost_of_equity) ** i)
                                for i, fcf in enumerate(future_cash_flows, 1)])
                dcf_total += terminal_value / ((1 + cost_of_equity) ** 5)

                shares = info.get('shares_outstanding')
                if shares:
                    dcf_value = round(dcf_total / shares, 2)
                    valuation['dcf_valuation'] = {
                        'fair_value': dcf_value,
                        'method': 'DCF (FCF)',
                        'details': f"WACC: {round(cost_of_equity*100, 1)}%, Growth: {round(growth_rate*100, 1)}%"
                    }
            except Exception as e:
                pass

        # DDM valuation (if dividend paying and DCF failed or as secondary)
        if div := info.get('dividend_rate'):
            growth = 0.05  # Assume 5% growth if not available
            if div > 0 and required_return > growth:
                ddm = gordon_growth_model(div, growth, required_return)
                valuation['ddm_valuation'] = ddm

        # Relative valuation
        def get_peg():
            if info.get('peg_ratio'):
                return info.get('peg_ratio')
            # Try Calculate: PE / (Growth Rate * 100)
            if info.get('trailing_pe') and info.get('earnings_growth'):
                growth = info.get('earnings_growth') * 100
                if growth > 0:
                    return round(info['trailing_pe'] / growth, 2)
            # Try Forward PE / Forward Growth if available?
            if info.get('forward_pe') and info.get('earnings_growth'):
                growth = info.get('earnings_growth') * 100
                if growth > 0:
                    return round(info['forward_pe'] / growth, 2)
            return None

        valuation['valuation_metrics'] = {
            'pe_ratio': info.get('trailing_pe'),
            'forward_pe': info.get('forward_pe'),
            'peg_ratio': get_peg(),
            'pb_ratio': info.get('price_to_book'),
            'ps_ratio': info.get('price_to_sales'),
            'ev_ebitda': round(info['enterprise_value'] / info['ebitda'], 2) if info.get('enterprise_value') and info.get('ebitda') else None
        }

        return {
            "status": "success",
            "symbol": symbol.upper(),
            "valuation": valuation
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/events/{symbol}")
async def get_events_analysis(symbol: str):
    """Get event-driven analysis (Earnings, Dividends, Analyst estimates)."""
    try:
        if not validate_symbol(symbol):
            return JSONResponse(
                status_code=404,
                content={"status": "error",
                         "detail": f"Symbol '{symbol}' not found."}
            )

        events = corporate_events_summary(symbol)

        import json
        events_safe = json.loads(json.dumps(events, default=str))

        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "symbol": symbol.upper(),
                "events": sanitize_for_json(events_safe)
            }
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"status": "error", "detail": str(e)}
        )
