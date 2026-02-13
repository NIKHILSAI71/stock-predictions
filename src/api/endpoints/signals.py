from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse
import math
import asyncio
import logging

from src.core.utils import sanitize_for_json
from src.core.api_limiter import get_api_coordinator
from src.data import get_stock_data, get_company_info, validate_symbol
from src.analysis.technical import (
    sma, rsi, macd, atr, bollinger_bands,
    fibonacci_retracement
)
from src.analysis.technical.signals import generate_entry_signal, detect_macd_divergence, generate_universal_signal
from src.analysis.technical.missing_stubs import pivot_points, volume_analysis
from src.analysis.quantitative.ml_models import regime_detection
from src.analysis.quantitative.classification import get_sector_pe_benchmark
from src.adaptive import AdaptiveLearningSystem
from src.data.news_fetcher import get_comprehensive_news
from src.data.sentiment import get_sentiment_engine
from src.ai.gemini import generate_market_insights, generate_search_queries
from src.api.endpoints.ai import get_ml_ensemble_prediction

router = APIRouter()
adaptive_system = AdaptiveLearningSystem()
logger = logging.getLogger(__name__)


@router.get("/universal-signals/{symbol}")
async def get_universal_signals(
    symbol: str,
    risk_tolerance: str = Query("medium", enum=["low", "medium", "high"]),
    investment_horizon: str = Query(
        "medium_term", enum=["short_term", "medium_term", "long_term"])
):
    """
    Get UNIVERSAL trading signals adapted to stock type, sector, and market regime.
    Now with ADAPTIVE LEARNING weights.
    """
    # Acquire analysis slot to prevent server overload
    coordinator = get_api_coordinator()

    try:
        # Try to acquire slot with timeout (non-blocking check)
        acquired = await asyncio.wait_for(
            coordinator.analysis_semaphore.acquire(),
            timeout=5.0
        )
    except asyncio.TimeoutError:
        logger.warning(f"Analysis request for {symbol} throttled - server at capacity")
        return JSONResponse(
            status_code=503,
            content={
                "status": "throttled",
                "detail": "Server at capacity. Please try again in 10 seconds.",
                "retry_after": 10,
                "symbol": symbol
            }
        )

    try:
        if not validate_symbol(symbol):
            return JSONResponse(
                status_code=404,
                content={"status": "error",
                         "detail": f"Symbol '{symbol}' not found."}
            )

        # 1. Fetch Data
        stock_data = get_stock_data(symbol, period="2y", interval="1d")
        company_info = get_company_info(symbol)

        # 1b. Calculate technical data for universal signal
        rsi_vals = rsi(stock_data)
        macd_data = macd(stock_data)
        atr_vals = atr(stock_data)
        # sr = find_support_resistance(stock_data) # Removed
        # vol_analysis = volume_analysis(stock_data) # Removed

        technical_data = {
            "rsi": rsi_vals.iloc[-1] if len(rsi_vals) > 0 else 50,
            "macd_trend": "Bullish" if len(macd_data['histogram']) > 0 and macd_data['histogram'].iloc[-1] > 0 else "Bearish",
            "atr": atr_vals.iloc[-1] if len(atr_vals) > 0 else 0,
            "sma_50": sma(stock_data, 50).iloc[-1] if len(stock_data) >= 50 else None,
            "sma_200": sma(stock_data, 200).iloc[-1] if len(stock_data) >= 200 else None,
            "support_levels": [],
            "resistance_levels": [],
            "volume_ratio": 1.0
        }

        # 2. Generate Universal Signal
        result = generate_universal_signal(
            symbol=symbol,
            stock_data=stock_data,
            company_info=company_info,
            technical_data=technical_data,
            risk_tolerance=risk_tolerance
        )

        # 3. Apply Adaptive Weights
        sector = result['classification']['sector']
        regime = result['market_regime']['current_regime']
        adaptive_weight = adaptive_system.get_regime_adaptive_weight(
            sector, regime)

        # Adjust confidence based on historical accuracy for this sector + regime
        original_confidence = result['signal']['confidence']
        adjusted_confidence = min(1.0, original_confidence * adaptive_weight)

        result['signal']['confidence'] = round(adjusted_confidence, 2)
        result['adaptive_adjustment'] = {
            "sector_weight": adaptive_weight,
            "original_confidence": original_confidence,
            "note": "Adjusted based on past prediction accuracy" if adaptive_weight != 1.0 else "Neutral weight"
        }

        # 4. Record Prediction for future learning
        current_price = stock_data['Close'].iloc[-1]
        adaptive_system.record_prediction(
            symbol=symbol,
            signal=result['signal'],
            current_price=current_price,
            classification=result['classification'],
            regime=regime
        )

        # 5. COMPREHENSIVE ANALYSIS - News, Sentiment, ML Models, AI Insights
        logger.info(f"Starting comprehensive analysis for {symbol}")

        # 5a. Generate AI research queries
        query_context = {
            "technicals": technical_data,
            "universal_system_signal": result['signal'],
            "classification": result['classification']
        }
        research_queries = generate_search_queries(symbol, query_context)
        logger.info(
            f"Generated {len(research_queries)} research queries for {symbol}")

        # 5b. Fetch comprehensive news (async)
        try:
            comprehensive_news = await get_comprehensive_news(
                symbol=symbol,
                queries=research_queries,
                max_per_query=5
            )
            logger.info(
                f"Fetched {len(comprehensive_news)} news articles for {symbol}")
        except Exception as e:
            logger.error(f"News fetch error: {e}")
            comprehensive_news = []

        # 5c. Run sentiment analysis (async)
        try:
            sentiment_engine = get_sentiment_engine()
            comprehensive_sentiment = await sentiment_engine.analyze(
                symbol=symbol,
                timeframe="1day",
                sources="all"
            )
            logger.info(f"Completed sentiment analysis for {symbol}")
        except Exception as e:
            logger.error(f"Sentiment analysis error: {e}")
            comprehensive_sentiment = {}

        # 5d. Run ML ensemble predictions (parallel)
        try:
            loop = asyncio.get_event_loop()
            ml_predictions = await loop.run_in_executor(
                None,
                get_ml_ensemble_prediction,
                stock_data,
                symbol
            )
            logger.info(f"Completed ML ensemble predictions for {symbol}")
        except Exception as e:
            logger.error(f"ML ensemble error: {e}")
            ml_predictions = {}

        # 5e. Prepare fundamental data for AI synthesis
        fundamental_data = {
            "pe_ratio": company_info.get('trailing_pe'),
            "peg_ratio": company_info.get('peg_ratio'),
            "market_cap": company_info.get('market_cap'),
            "sector": company_info.get('sector'),
            "industry": company_info.get('industry')
        }

        # 5f. AI synthesis via Gemini
        try:
            ai_insights = generate_market_insights(
                stock_symbol=symbol,
                technical_data=technical_data,
                fundamental_data=fundamental_data,
                news_sentiment={
                    "news_context": comprehensive_news,
                    "comprehensive_sentiment": comprehensive_sentiment
                },
                extra_metrics={
                    "ml_ensemble": ml_predictions,
                    "universal_signal": result
                },
                stock_classification=result['classification'],
                universal_signal=result
            )
            logger.info(f"Completed AI synthesis for {symbol}")
        except Exception as e:
            logger.error(f"AI synthesis error: {e}")
            ai_insights = "AI analysis temporarily unavailable"

        # 6. Combine into comprehensive response
        result['ai_analysis'] = ai_insights
        result['comprehensive_sentiment'] = comprehensive_sentiment
        result['ml_predictions'] = ml_predictions
        result['news'] = comprehensive_news
        result['analysis_metadata'] = {
            "news_articles_analyzed": len(comprehensive_news),
            "sentiment_sources": ["news", "twitter", "reddit", "stocktwits"] if comprehensive_sentiment else [],
            "ml_models_used": len(ml_predictions.get("predictions", {})) if isinstance(ml_predictions, dict) else 0
        }

        return JSONResponse(content=sanitize_for_json({
            "status": "success",
            "symbol": symbol.upper(),
            "data": result
        }))
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"status": "error", "detail": str(e)})
    finally:
        # Always release the analysis slot
        coordinator.analysis_semaphore.release()


@router.get("/signals/{symbol}")
async def get_trading_signals(
    symbol: str,
    risk_tolerance: str = "medium"
):
    """
    Get comprehensive trading signals with valuation filters and risk management.
    """
    try:
        if not validate_symbol(symbol):
            return JSONResponse(
                status_code=404,
                content={"status": "error",
                         "detail": f"Symbol '{symbol}' not found."}
            )

        # Gather data
        stock_data = get_stock_data(symbol, "1y", "1d")
        company_info = get_company_info(symbol)

        # Calculate technical indicators
        rsi_values = rsi(stock_data)
        macd_data = macd(stock_data)
        atr_values = atr(stock_data)
        # sr = find_support_resistance(stock_data) # Removed

        current_price = stock_data['Close'].iloc[-1]

        # Calculate returns for volatility
        returns = stock_data['Close'].pct_change().dropna()
        volatility_annual = returns.std() * (252 ** 0.5)

        # Prepare technical data dict
        technical_data = {
            "rsi": rsi_values.iloc[-1],
            "macd_trend": "Bullish" if macd_data['histogram'].iloc[-1] > 0 else "Bearish",
            "macd_histogram_series": macd_data['histogram'],
            "price_series": stock_data['Close'],
            "atr": atr_values.iloc[-1],
            "sma_50": sma(stock_data, 50).iloc[-1] if len(stock_data) >= 50 else None,
            "sma_200": sma(stock_data, 200).iloc[-1] if len(stock_data) >= 200 else None,
            "support_levels": [],
            "resistance_levels": [],
            "volatility_annualized": volatility_annual
        }

        # Get volume analysis for confirmation
        volume_data = volume_analysis(stock_data)

        # Get pivot points for intermediate resistance/support levels
        pivot_data = pivot_points(stock_data, method='standard')

        # NEW: Calculate Fibonacci retracement levels
        fib_data = fibonacci_retracement(stock_data)
        fibonacci_levels = {
            "trend": fib_data.get('trend', 'unknown'),
            "swing_high": fib_data.get('swing_high'),
            "swing_low": fib_data.get('swing_low'),
            "fib_236": fib_data.get('23.6%'),
            "fib_382": fib_data.get('38.2%'),
            "fib_500": fib_data.get('50.0%'),
            "fib_618": fib_data.get('61.8%'),
            "fib_786": fib_data.get('78.6%'),
            "fib_100": fib_data.get('100.0%'),
            "current_price": round(float(current_price), 2)
        }

        # Calculate fair value with robust fallbacks
        fair_value = None
        dcf_intrinsic_value = None

        if company_info.get('free_cash_flow') and company_info.get('shares_outstanding'):
            try:
                fcf = company_info['free_cash_flow']
                shares = company_info['shares_outstanding']
                growth = company_info.get('earnings_growth', 0.10) or 0.10
                long_term_growth = min(abs(growth), 0.12)
                discount_rate = 0.10

                if fcf > 0 and shares > 0:
                    fcf_per_share = fcf / shares
                    stage1_value = 0
                    for year in range(1, 6):
                        stage1_value += (fcf_per_share * ((1 + long_term_growth)
                                         ** year)) / ((1 + discount_rate) ** year)

                    terminal_fcf = fcf_per_share * \
                        ((1 + long_term_growth) ** 5) * (1 + 0.03)
                    terminal_value = terminal_fcf / (discount_rate - 0.03)
                    terminal_pv = terminal_value / ((1 + discount_rate) ** 5)

                    dcf_value = stage1_value + terminal_pv

                    if dcf_value > current_price * 0.2 and dcf_value < current_price * 3:
                        dcf_intrinsic_value = dcf_value
                        fair_value = dcf_value
            except:
                pass

        if not dcf_intrinsic_value and company_info.get('target_mean_price'):
            target_price = company_info['target_mean_price']
            if target_price > 0 and target_price < current_price * 2:
                dcf_intrinsic_value = target_price
                if not fair_value:
                    fair_value = target_price

        if not fair_value and company_info.get('eps_trailing') and company_info.get('book_value'):
            try:
                eps = company_info['eps_trailing']
                bv = company_info['book_value']
                if eps > 0 and bv > 0:
                    fair_value = math.sqrt(22.5 * eps * bv)
            except:
                pass

        if not fair_value and company_info.get('eps_trailing') and company_info.get('trailing_pe'):
            fair_value = company_info['eps_trailing'] * 15.0

        dcf_overvaluation_pct = None
        if dcf_intrinsic_value and dcf_intrinsic_value > 0:
            dcf_overvaluation_pct = round(
                ((current_price - dcf_intrinsic_value) / dcf_intrinsic_value) * 100, 1)

        fundamental_data = {
            "pe_ratio": company_info.get('trailing_pe'),
            "forward_pe": company_info.get('forward_pe'),
            "peg_ratio": company_info.get('peg_ratio'),
            "fair_value": fair_value,
            "dcf_intrinsic_value": dcf_intrinsic_value,
            "dcf_overvaluation_pct": dcf_overvaluation_pct,
            "industry_pe": 17.8,
            "fifty_two_week_high": company_info.get('fifty_two_week_high')
        }

        regime_data = regime_detection(
            returns, window=min(60, len(returns) - 1))
        if 'error' in regime_data:
            regime_data = {
                'current_regime': 'Bull Low Volatility',
                'momentum_weight': 0.5,
                'fundamental_weight': 0.5,
                'position_size_multiplier': 1.0,
                'confidence_adjustment': 1.0,
                'regime_stability': 50.0
            }

        signal_result = generate_entry_signal(
            current_price=current_price,
            technical_data=technical_data,
            fundamental_data=fundamental_data,
            risk_tolerance=risk_tolerance,
            volume_data=volume_data,
            market_regime=regime_data
        )

        pe_ratio = company_info.get('trailing_pe')

        try:
            sector = company_info.get('sector', 'Unknown')
            industry_pe = get_sector_pe_benchmark(sector)
        except:
            industry_pe = 17.8

        pe_vs_industry = None
        pe_vs_industry_status = "N/A"

        if pe_ratio and pe_ratio > 0:
            pe_vs_industry = round(pe_ratio / industry_pe, 2)
            if pe_vs_industry > 1.5:
                pe_vs_industry_status = f"{pe_ratio:.1f}x vs {industry_pe}x (HIGH)"
            elif pe_vs_industry > 1.2:
                pe_vs_industry_status = f"{pe_ratio:.1f}x vs {industry_pe}x (ELEVATED)"
            elif pe_vs_industry < 0.8:
                pe_vs_industry_status = f"{pe_ratio:.1f}x vs {industry_pe}x (LOW)"
            else:
                pe_vs_industry_status = f"{pe_ratio:.1f}x vs {industry_pe}x (FAIR)"
        elif company_info.get('sector') == 'Energy' or company_info.get('sector') == 'Materials':
            pe_vs_industry_status = "Cyclical (P/E less relevant)"

        overvaluation_pct = None
        overvaluation_status = "N/A"

        if fair_value and fair_value > 0:
            overvaluation_pct = round(
                (current_price - fair_value) / fair_value * 100, 1)

            if overvaluation_pct > 20:
                overvaluation_status = f"+{overvaluation_pct}% (OVERVALUED)"
            elif overvaluation_pct > 5:
                overvaluation_status = f"+{overvaluation_pct}% (PREMIUM)"
            elif overvaluation_pct < -20:
                overvaluation_status = f"{overvaluation_pct}% (UNDERVALUED)"
            elif overvaluation_pct < -5:
                overvaluation_status = f"{overvaluation_pct}% (DISCOUNT)"
            else:
                overvaluation_status = f"{overvaluation_pct:+.1f}% (FAIR)"
        else:
            if pe_ratio and pe_ratio > 30:
                overvaluation_status = "Likely Overvalued (High P/E)"
            elif pe_ratio and pe_ratio < 10:
                overvaluation_status = "Likely Undervalued (Low P/E)"
            else:
                overvaluation_status = "Market Weight"

        macd_divergence = detect_macd_divergence(
            stock_data['Close'], macd_data['histogram'])
        macd_divergence_status = macd_divergence.get('divergence_type', 'NONE')
        if macd_divergence_status == "BEARISH":
            macd_divergence_status = "BEARISH"
        elif macd_divergence_status == "MOMENTUM_SHIFT_BEARISH":
            macd_divergence_status = "WEAKENING"
        elif macd_divergence_status == "BULLISH":
            macd_divergence_status = "BULLISH"

        pullback_target = signal_result.get(
            'entry_recommendation', {}).get('pullback_target')
        pullback_pct = signal_result.get(
            'entry_recommendation', {}).get('pullback_pct')
        pullback_status = "N/A"
        if pullback_target:
            pullback_status = f"${pullback_target} ({pullback_pct}% down)"
        elif not signal_result.get('entry_recommendation', {}).get('wait_for_pullback'):
            pullback_status = "Entry OK Now"

        response_data = {
            "status": "success",
            "symbol": symbol.upper(),
            "current_price": round(float(current_price), 2),
            "signals": signal_result,
            "analysis": {
                "pe_vs_industry": pe_vs_industry_status,
                "pe_vs_industry_ratio": pe_vs_industry,
                "overvaluation_pct": overvaluation_pct,
                "overvaluation_status": overvaluation_status,
                "macd_divergence": macd_divergence_status,
                "pullback_target": pullback_status,
                "near_52w_high": signal_result.get('entry_recommendation', {}).get('near_all_time_high', False)
            },
            "volume_confirmation": {
                "current_volume": volume_data.get('current_volume'),
                "avg_volume_20": volume_data.get('avg_volume_20'),
                "volume_ratio": volume_data.get('volume_ratio'),
                "volume_signal": volume_data.get('volume_signal'),
                "conviction": signal_result.get('confidence_adjustments', {}).get('volume_conviction', 'N/A'),
                "obv_trend": volume_data.get('obv_trend'),
                "mfi": volume_data.get('mfi'),
                "mfi_signal": volume_data.get('mfi_signal')
            },
            "intermediate_levels": {
                "pivot_point": pivot_data.get('pivot'),
                "resistance_r1": pivot_data.get('r1'),
                "resistance_r2": pivot_data.get('r2'),
                "resistance_r3": pivot_data.get('r3'),
                "support_s1": pivot_data.get('s1'),
                "support_s2": pivot_data.get('s2'),
                "support_s3": pivot_data.get('s3')
            },
            "technical_snapshot": {
                "rsi": round(float(rsi_values.iloc[-1]), 2),
                "macd_histogram": round(float(macd_data['histogram'].iloc[-1]), 4),
                "atr": round(float(atr_values.iloc[-1]), 2),
                "volatility_annualized_pct": round(float(volatility_annual) * 100, 2)
            },
            "valuation_snapshot": {
                "pe_ratio": company_info.get('trailing_pe'),
                "forward_pe": company_info.get('forward_pe'),
                "peg_ratio": company_info.get('peg_ratio'),
                "peg_status": signal_result.get('confidence_adjustments', {}).get('peg_status', 'N/A'),
                "estimated_fair_value": round(float(fair_value), 2) if fair_value else None,
                "dcf_intrinsic_value": round(float(dcf_intrinsic_value), 2) if dcf_intrinsic_value else None,
                "dcf_overvaluation_pct": dcf_overvaluation_pct,
                "industry_pe": industry_pe
            },
            "fibonacci_levels": fibonacci_levels,
            "staged_entry": signal_result.get('staged_entry', {}),
            "market_regime": {
                "current_regime": regime_data.get('current_regime', 'Unknown'),
                "regime_stability": regime_data.get('regime_stability'),
                "position_size_multiplier": regime_data.get('position_size_multiplier', 1.0)
            }
        }

        return JSONResponse(content=sanitize_for_json(response_data))
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"status": "error", "detail": str(e)})
