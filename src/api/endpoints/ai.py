from fastapi import APIRouter
from fastapi.responses import JSONResponse

from src.core.utils import sanitize_for_json
from src.data import get_stock_data, get_company_info, get_latest_news
from src.data.alternative_data import get_all_alternative_data
from src.analysis.technical import (
    rsi, macd, detect_patterns, ichimoku_signals, volume_analysis
)
from src.analysis.fundamental.valuation import gordon_growth_model
from src.analysis.fundamental import (
    esg_score_analysis, moat_assessment, management_quality_indicators
)
from src.analysis.quantitative import (
    comprehensive_risk_analysis, arima_forecast, get_lstm_prediction, 
    get_ml_ensemble_prediction, get_anomaly_alerts,
    get_xgboost_prediction, get_gru_prediction, get_volatility_forecast,
    get_cnn_lstm_prediction, get_attention_prediction, get_wavelet_denoised_data
)
from src.analysis.quantitative.classification import classify_stock
from src.analysis.technical.signals import generate_universal_signal, generate_entry_signal, detect_macd_divergence
from src.analysis.fundamental.macro.economic_indicators import get_treasury_yields, get_market_indices
from src.analysis.fundamental.industry.sector_rotation import get_sector_performance
from src.analysis.technical.strategies import analyze_growth_metrics, analyze_value_metrics
from src.data.news_fetcher import get_market_sentiment_search, get_comprehensive_news
from src.ai.gemini import generate_market_insights, generate_search_query, generate_search_queries
from src.adaptive import AdaptiveLearningSystem
from src.analysis.technical import get_relative_strength_rating

router = APIRouter()
adaptive_system = AdaptiveLearningSystem()

# Helper access to endpoints logic - simplified by calling libraries directly as per original code

@router.get("/ai/analyze/{symbol}")
async def get_ai_analysis(symbol: str):
    """Get AI-powered comprehensive analysis using ALL available tools."""
    try:
        # 1. Gather all data
        stock_data = get_stock_data(symbol, period="2y", interval="1d") # Data mostly reused
        company_info = get_company_info(symbol)
        
        # --- Technical ---
        rsi_vals = rsi(stock_data)
        macd_data = macd(stock_data)
        
        # --- Patterns ---
        patterns = detect_patterns(stock_data)
        
        # --- Ichimoku ---
        ichimoku = ichimoku_signals(stock_data)

        # --- Volume ---
        vol_analysis = volume_analysis(stock_data)
        
        # --- Qualitative (ESG, Moat) ---
        esg = esg_score_analysis(symbol)
        moat = moat_assessment(symbol)
        management = management_quality_indicators(symbol)
        qual_data = {
            "esg": esg,
            "moat": moat,
            "management": management
        }
        
        # --- Valuation ---
        valuation_metrics = {
             'pe_ratio': company_info.get('trailing_pe'),
             'peg_ratio': company_info.get('peg_ratio'),
             'fair_value_ddm': gordon_growth_model(company_info.get('dividend_rate', 0), 0.05, 0.10) if company_info.get('dividend_rate') else "N/A"
        }

        # --- Risk ---
        returns = stock_data['Close'].pct_change().dropna()
        risk_metrics = comprehensive_risk_analysis(returns)
        
        # --- Forecast ---
        forecast_data = arima_forecast(stock_data['Close'], steps=5)

        # --- Universal Signal (Calculated via Adaptive System) ---
        try:
             # STOCK CLASSIFICATION
             stock_classification = classify_stock(symbol)

             # GENERATE UNIVERSAL SIGNAL
             tech_data_simple = {
                "rsi": rsi_vals.iloc[-1] if len(rsi_vals) > 0 else 50,
                "macd_trend": "Bullish" if len(macd_data['histogram']) > 0 and macd_data['histogram'].iloc[-1] > 0 else "Bearish"
             }
             
             universal_signal_full = generate_universal_signal(
                 symbol=symbol,
                 stock_data=stock_data,
                 company_info=company_info,
                 technical_data=tech_data_simple,
                 risk_tolerance="medium"
             )
             
             universal_signal = universal_signal_full.get('signal', {})
             
             # RECORD PREDICTION
             current_price = stock_data['Close'].iloc[-1]
             adaptive_system.record_prediction(
                symbol=symbol,
                signal=universal_signal,
                current_price=current_price,
                classification=stock_classification
             )
             
        except Exception as e:
             print(f"Signal Generation Error: {e}")
             universal_signal = {}
             stock_classification = {}
             universal_signal_full = {}

        # --- Sentiment/Alternative (COMPREHENSIVE RESEARCH PIPELINE) ---
        # Build context for AI query generation
        query_context = {
            "technicals": tech_data_simple,
            "universal_system_signal": universal_signal,
            "classification": stock_classification
        }
        
        # AI generates 10-12 comprehensive research queries
        research_queries = generate_search_queries(symbol, query_context)
        print(f"\033[96mINFO:     AI Generated {len(research_queries)} research queries for {symbol}\033[0m")
        
        # Comprehensive news aggregation from 20+ sources
        comprehensive_news = get_comprehensive_news(
            symbol=symbol,
            queries=research_queries,
            max_per_query=5
        )
        
        # Also get basic sentiment score (but we'll override news_context with comprehensive data)
        sentiment = get_market_sentiment_search(symbol)
        
        # Calculate content extraction stats
        articles_with_content = sum(1 for n in comprehensive_news if n.get('content') and len(n.get('content', '')) > 100)
        
        news_vol = len(comprehensive_news)
        traffic_level = "Very High" if news_vol >= 15 else ("High" if news_vol >= 10 else ("Medium" if news_vol >= 5 else "Low"))
        
        # Override news_context with comprehensive news (which has deep content)
        sentiment['news_context'] = comprehensive_news
        sentiment['news_volume'] = news_vol
        sentiment['articles_with_deep_content'] = articles_with_content
        sentiment['recent_headlines'] = [n.get('title', '') for n in comprehensive_news[:5]]
        
        sent_data = {
            "social_sentiment": sentiment,
            "web_traffic": {
                "level": traffic_level,
                "source": "Comprehensive Research",
                "value": f"{news_vol} sources analyzed ({articles_with_content} with full content)"
            },
            "research_queries_used": len(research_queries)
        }
        
        # --- Strategy ---
        strat_metrics = {
            'pe_ratio': company_info.get('trailing_pe') or 0.0,
            'roe': (company_info.get('return_on_equity') or 0) * 100,
            'debt_to_equity': company_info.get('debt_to_equity') or 0.0, 
            'current_ratio': company_info.get('current_ratio') or 0.0,
            'dividend_yield': (company_info.get('dividend_yield') or 0) * 100,
            'eps_growth': 10.0, 
            'revenue_growth': 10.0 
        }
        growth_strat = analyze_growth_metrics(strat_metrics)
        value_strat = analyze_value_metrics(strat_metrics)

        # --- Macro & Industry ---
        treasury_yields = get_treasury_yields()
        market_indices = get_market_indices()
        sector_perf = get_sector_performance(period="1mo")
        
        macro_context = {
            "treasury_yields": treasury_yields,
            "market_indices": market_indices,
            "sector_performance": sector_perf
        }

        # --- NEW: Advanced Data Injection for God Mode ---
        # 1. LSTM Prediction
        try:
            lstm_pred = get_lstm_prediction(stock_data, train_epochs=5)
        except:
            lstm_pred = {}

        # 2. ML Ensemble
        try:
            from src.analysis.quantitative.ml_aggregator import get_ml_ensemble_prediction
            ml_prediction = get_ml_ensemble_prediction(stock_data)
        except:
            ml_prediction = {}

        # 3. Alternative Data
        try:
            alt_data = get_all_alternative_data(symbol)
            from src.data.alternative_data import get_alternative_data_signal
            alt_signal = get_alternative_data_signal(alt_data)
        except:
            alt_data = {}
            alt_signal = {}

        # 4. Sector Strength
        try:
            sector = company_info.get('sector', 'Unknown')
            sector_strength = get_relative_strength_rating(stock_data, sector, period="3mo")
        except:
            sector_strength = {}

        # 5. Anomaly Alerts
        try:
            anomaly_alerts = get_anomaly_alerts(stock_data)
        except:
            anomaly_alerts = {}

        # 6. NEW: XGBoost Prediction
        try:
            xgboost_pred = get_xgboost_prediction(stock_data, prediction_horizon=5)
        except:
            xgboost_pred = {}

        # 7. NEW: GRU Prediction
        try:
            gru_pred = get_gru_prediction(stock_data, train_epochs=3)
        except:
            gru_pred = {}

        # 8. NEW: GARCH Volatility Forecast
        try:
            volatility_forecast = get_volatility_forecast(stock_data, forecast_horizon=5)
        except:
            volatility_forecast = {}

        # 9. NEW: CNN-LSTM Hybrid Prediction
        try:
            cnn_lstm_pred = get_cnn_lstm_prediction(stock_data, train_epochs=3)
        except:
            cnn_lstm_pred = {}

        # 10. NEW: Attention-based Prediction
        try:
            attention_pred = get_attention_prediction(stock_data, train_epochs=3)
        except:
            attention_pred = {}

        # 11. NEW: Wavelet Denoising Analysis
        try:
            wavelet_analysis = get_wavelet_denoised_data(stock_data, column='Close')
        except:
            wavelet_analysis = {}
        
        # Use comprehensive news gathered earlier (20+ sources)
        
        ai_insights = generate_market_insights(
            stock_symbol=symbol,
            technical_data={
                "rsi": rsi_vals.iloc[-1],
                "macd": "Bullish" if macd_data['histogram'].iloc[-1] > 0 else "Bearish", 
                "close_price": current_price,
                "volume_conviction": universal_signal.get('volume_confirmation', 'Neutral'),
                "trend_strength": adx_val if 'adx_val' in locals() else 0 # Ensure ADX is passed if available
            },
            fundamental_data=valuation_metrics,
            news_sentiment=sent_data,
            extra_metrics={
               "risk_analysis": {
                   "volatility": risk_metrics.get('volatility'),
                   "sharpe": risk_metrics.get('sharpe_ratio'),
                   "drawdown": risk_metrics.get('max_drawdown') if isinstance(risk_metrics.get('max_drawdown'), (int, float)) else risk_metrics.get('max_drawdown', {}).get('max_drawdown_pct', 0)
               },
               "strategies": {
                   "growth_score": growth_strat['score'],
                   "value_score": value_strat['score']
               },
               "forecast": forecast_data,
               "patterns": patterns,
               "ichimoku": ichimoku,
               "qualitative": qual_data,
               "volume_analysis": {
                   "volume_ratio": vol_analysis.get('volume_ratio', 1.0),
                   "obv_trend": vol_analysis.get('obv_trend', 'Neutral'),
                   "mfi": vol_analysis.get('mfi')
               },
               # GOD MODE DATA INJECTION
               "ml_ensemble": ml_prediction,
               "lstm_forecast": lstm_pred,
               "alternative_data": {
                   "signal": alt_signal,
                   "insider": alt_data.get("insider_activity"),
                   "options": alt_data.get("options_flow")
               },
               "sector_strength": sector_strength,
               "anomaly_alerts": anomaly_alerts,
               "analyst_consensus": {
                   "recommendation": company_info.get('recommendation_key', 'none'),
                   "target_price": company_info.get('target_mean_price'),
                   "target_high": company_info.get('target_high_price'),
                   "target_low": company_info.get('target_low_price'),
                   "analyst_count": company_info.get('number_of_analysts')
               },
               # NEW: Advanced ML Models
               "xgboost_prediction": xgboost_pred,
               "gru_prediction": gru_pred,
               "volatility_forecast": volatility_forecast,
               "cnn_lstm_prediction": cnn_lstm_pred,
               "attention_prediction": attention_pred,
               "wavelet_analysis": {
                   "noise_removed_pct": wavelet_analysis.get('noise_removed_pct', 0),
                   "trend_clarity": wavelet_analysis.get('trend_clarity', 'N/A'),
                   "volatility_reduction": wavelet_analysis.get('volatility_reduction_pct', 0)
               }
            },
            macro_data=macro_context,
            stock_classification=stock_classification,
            universal_signal=universal_signal_full, 
            search_context=comprehensive_news  # 20+ sources from comprehensive research
        )
        
        return {
            "status": "success",
            "symbol": symbol.upper(),
            "ai_analysis": ai_insights,
            "macro_context": sanitize_for_json(macro_context),
            "alternative_data": sent_data 
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"status": "error", "detail": str(e)})


@router.get("/enhanced-prediction/{symbol}")
async def get_enhanced_prediction(symbol: str):
    """
    Get enhanced AI prediction combining:
    - LSTM deep learning predictions
    - Ensemble scoring from multiple ML models
    - Alternative data (insider trading, institutional, options, sentiment)
    - Sector relative strength
    - Anomaly detection alerts
    """
    try:
        # 1. Get stock data
        stock_data = get_stock_data(symbol, period="2y", interval="1d")
        company_info = get_company_info(symbol)
        sector = company_info.get('sector', 'Unknown')
        
        # 2. LSTM Prediction
        try:
            lstm_pred = get_lstm_prediction(stock_data, train_epochs=5)
        except Exception as e:
            lstm_pred = {"error": str(e), "method": "lstm_unavailable"}
        
        # 3. Get existing ML prediction for ensemble
        try:
            from src.analysis.quantitative.ml_aggregator import get_ml_ensemble_prediction
            ml_prediction = get_ml_ensemble_prediction(stock_data)
        except Exception as e:
            ml_prediction = {"error": str(e)}
        
        # 4. Technical signals for ensemble
        try:
            rsi_vals = rsi(stock_data)
            macd_data = macd(stock_data)
            technical_signals = {
                "signal": "BUY" if rsi_vals.iloc[-1] < 40 and macd_data['histogram'].iloc[-1] > 0 else 
                          ("SELL" if rsi_vals.iloc[-1] > 60 and macd_data['histogram'].iloc[-1] < 0 else "HOLD"),
                "signal_strength": 50 + (50 - rsi_vals.iloc[-1]) * 0.5
            }
        except:
            technical_signals = None
        
        # 5. Ensemble prediction
        try:
            from src.analysis.quantitative.ensemble_scorer import get_ensemble_prediction
            # Prepare predictions dict for ensemble
            predictions_for_ensemble = {}
            
            if lstm_pred and "error" not in lstm_pred:
                predictions_for_ensemble["lstm"] = lstm_pred
            
            if ml_prediction and "error" not in ml_prediction:
                if ml_prediction.get("rf_prediction"):
                    predictions_for_ensemble["random_forest"] = {
                        "direction": ml_prediction.get("rf_prediction", "Neutral"),
                        "confidence": ml_prediction.get("rf_confidence", 50)
                    }
                if ml_prediction.get("svm_prediction"):
                    predictions_for_ensemble["svm"] = {
                        "direction": ml_prediction.get("svm_prediction", "Neutral"),
                        "confidence": ml_prediction.get("svm_confidence", 50)
                    }
                if ml_prediction.get("momentum_prediction"):
                    predictions_for_ensemble["momentum"] = {
                        "direction": ml_prediction.get("momentum_prediction", "Neutral"),
                        "confidence": ml_prediction.get("momentum_confidence", 50)
                    }
            
            ensemble_result = get_ensemble_prediction(
                symbol=symbol,
                stock_data=stock_data,
                ml_prediction=ml_prediction,
                technical_signals=technical_signals,
                fundamental_data=company_info
            )
        except Exception as e:
            ensemble_result = {"error": str(e)}
        
        # 6. Alternative Data
        try:
            alt_data = get_all_alternative_data(symbol)
            from src.data.alternative_data import get_alternative_data_signal
            alt_signal = get_alternative_data_signal(alt_data)
        except Exception as e:
            alt_data = {"error": str(e)}
            alt_signal = {"overall_signal": "NEUTRAL", "error": str(e)}
        
        # 7. Sector Relative Strength
        try:
            
            sector_strength = get_relative_strength_rating(stock_data, sector, period="3mo")
        except Exception as e:
            sector_strength = {"available": False, "error": str(e)}
        
        # 8. Anomaly Detection
        try:
            anomaly_alerts = get_anomaly_alerts(stock_data)
        except Exception as e:
            anomaly_alerts = {"total_alerts": 0, "error": str(e)}

        # 9. NEW: XGBoost Prediction
        try:
            xgboost_pred = get_xgboost_prediction(stock_data, prediction_horizon=5)
        except Exception as e:
            xgboost_pred = {"error": str(e)}

        # 10. NEW: GRU Prediction
        try:
            gru_pred = get_gru_prediction(stock_data, train_epochs=3)
        except Exception as e:
            gru_pred = {"error": str(e)}

        # 11. NEW: GARCH Volatility Forecast
        try:
            volatility_forecast = get_volatility_forecast(stock_data, forecast_horizon=5)
        except Exception as e:
            volatility_forecast = {"error": str(e)}

        # 12. NEW: CNN-LSTM Hybrid Prediction
        try:
            cnn_lstm_pred = get_cnn_lstm_prediction(stock_data, train_epochs=3)
        except Exception as e:
            cnn_lstm_pred = {"error": str(e)}

        # 13. NEW: Attention-based Prediction
        try:
            attention_pred = get_attention_prediction(stock_data, train_epochs=3)
        except Exception as e:
            attention_pred = {"error": str(e)}
        
        
        current_price = stock_data['Close'].iloc[-1]
        
        # 14. NEW: Model Accuracy and Price Targets
        try:
            from src.analysis.quantitative.model_accuracy import (
                get_model_accuracy_summary, generate_price_targets
            )
            
            # Get accuracy metrics
            accuracy_data = get_model_accuracy_summary(symbol)
            
            # Generate multi-timeframe targets
            # Generate multi-timeframe targets
            # 1. 7-Day: Use LSTM/ML Prediction if available, else Momentum
            ml_return_7d = lstm_pred.get("predicted_change_pct", 0) / 100.0 if lstm_pred else 0
            if ml_return_7d == 0:
                 # Fallback to momentum if ML fails
                 ml_return_7d = (stock_data['Close'].iloc[-1] / stock_data['Close'].iloc[-7] - 1) * 0.3
            
            # 2. 90-Day: Convergence towards Analyst Target (Weighted)
            analyst_target = company_info.get('target_mean_price')
            if analyst_target and analyst_target > 0:
                # Assume 90 days captures ~25% of the move to the 1-year target
                gap_pct = (analyst_target - current_price) / current_price
                return_90d = gap_pct * 0.25
            else:
                return_90d = (stock_data['Close'].iloc[-1] / stock_data['Close'].iloc[-90] - 1) * 0.5
            
            # 3. 30-Day: Interpolate between 7d and 90d
            return_30d = (ml_return_7d * 0.7) + (return_90d * 0.3)
                
            volatility = stock_data['Close'].pct_change().std() * (252 ** 0.5)
            
            targets = generate_price_targets(
                current_price=current_price,
                predicted_return_7d=ml_return_7d,
                predicted_return_30d=return_30d,
                predicted_return_90d=return_90d,
                volatility=volatility
            )
        except Exception as e:
            accuracy_data = {}
            targets = {}
            print(f"Error calculating accuracy/targets: {e}")

        # Compile response

        
        response = {
            "status": "success",
            "symbol": symbol.upper(),
            "current_price": round(current_price, 2),
            "price_targets": targets,
            "model_metrics": accuracy_data.get('models', {}),
            "system_accuracy": accuracy_data.get('system_accuracy', 0),
            "enhanced_prediction": {
                "lstm_prediction": lstm_pred,
                "ensemble_prediction": ensemble_result,
                "ml_models": {
                    "random_forest": ml_prediction.get("rf_prediction", "N/A"),
                    "svm": ml_prediction.get("svm_prediction", "N/A"),
                    "momentum": ml_prediction.get("momentum_prediction", "N/A"),
                    "models_agree": ml_prediction.get("models_agree", False)
                },
                "direction": ensemble_result.get("direction", lstm_pred.get("direction", "Neutral")),
                "confidence": ensemble_result.get("confidence", lstm_pred.get("confidence", 50)),
                "predicted_change_pct": lstm_pred.get("predicted_change_pct", 0)
            },
            "alternative_data": {
                "summary": alt_signal,
                "insider_activity": alt_data.get("insider_activity", {}),
                "institutional_holdings": alt_data.get("institutional_holdings", {}),
                "options_flow": alt_data.get("options_flow", {}),
                "social_sentiment": alt_data.get("social_sentiment", {})
            },
            "sector_analysis": {
                "sector": sector,
                "relative_strength": sector_strength
            },
            "anomaly_alerts": anomaly_alerts,
            # NEW: Advanced ML Models
            "xgboost_prediction": {
                "direction": xgboost_pred.get("direction", "N/A"),
                "confidence": xgboost_pred.get("confidence", 50),
                "accuracy": xgboost_pred.get("accuracy_test", "N/A")
            },
            "gru_prediction": {
                "direction": gru_pred.get("direction", "N/A"),
                "confidence": gru_pred.get("confidence", 50),
                "predictions": gru_pred.get("predictions", [])
            },
            "volatility_forecast": {
                "model": volatility_forecast.get("recommended_model", "N/A"),
                "forecast": volatility_forecast.get("volatility_forecast", []),
                "historical_20d": volatility_forecast.get("historical_volatility_20d", 0),
                "regime": volatility_forecast.get("volatility_regime", {})
            },
            # NEW: CNN-LSTM Hybrid
            "cnn_lstm_prediction": {
                "direction": cnn_lstm_pred.get("direction", "N/A"),
                "confidence": cnn_lstm_pred.get("confidence", 50),
                "predicted_change_pct": cnn_lstm_pred.get("predicted_change_pct", 0),
                "model_type": cnn_lstm_pred.get("model_type", "CNN-LSTM")
            },
            # NEW: Attention-based Transformer
            "attention_prediction": {
                "direction": attention_pred.get("direction", "N/A"),
                "confidence": attention_pred.get("confidence", 50),
                "predicted_change_pct": attention_pred.get("predicted_change_pct", 0),
                "attention_focus": attention_pred.get("attention_focus", {})
            },
            "meta": {
                "models_used": [
                    "LSTM (Long Short-Term Memory)", 
                    "XGBoost (Gradient Boosting)", 
                    "GRU (Gated Recurrent Unit)", 
                    "CNN-LSTM Hybrid", 
                    "Attention Transformer", 
                    "Random Forest", 
                    "SVM (Support Vector Machine)", 
                    "Momentum Analysis", 
                    "GARCH Volatility",
                    "Ensemble Meta-Learner"
                ],
                "data_sources": ["Yahoo Finance", "News Sentiment", "Insider Trading", "Options Flow"]
            }
        }
        
        return JSONResponse(content=sanitize_for_json(response))
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"status": "error", "detail": str(e)})
