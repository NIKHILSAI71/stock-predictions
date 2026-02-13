import { useStock } from '../../context/StockContext';

const DataGrid = () => {
    const { expertData, signalsData, aiData, loading } = useStock();

    const { technical: tech, fundamental: fund, risk, patterns, valuation: val, alternative: alt } = expertData;

    // Safety check - if data isn't loaded yet
    if (!tech || !fund) return <section id="dataGrid" className="data-grid hidden"></section>;

    const signals = signalsData?.signals || signalsData || {};
    const analysis = signalsData?.analysis || signals?.analysis || {};
    const enhancedPred = signalsData?.enhanced_prediction || {};
    const prediction = enhancedPred.enhanced_prediction || {};
    const mlPrediction = signalsData?.ml_prediction || {};
    const volumeConf = signalsData?.volume_confirmation || {};
    const pivotLevels = signalsData?.intermediate_levels || {};
    const fibLevels = signalsData?.fibonacci_levels || {};
    const stagedEntry = signalsData?.staged_entry || {};
    const marketRegimeData = signalsData?.market_regime || {};
    const valSnapshot = signalsData?.valuation_snapshot || {};
    const statArb = aiData?.statistical_arbitrage || {};

    // Helper helpers
    const formatLarge = (num) => {
        if (!num && num !== 0) return '-';
        if (num >= 1e12) return (num / 1e12).toFixed(2) + 'T';
        if (num >= 1e9) return (num / 1e9).toFixed(2) + 'B';
        if (num >= 1e6) return (num / 1e6).toFixed(2) + 'M';
        if (num >= 1e3) return (num / 1e3).toFixed(2) + 'K';
        return num.toLocaleString();
    };

    const MetricRow = ({ label, value, suffix = '', isGood = null, isSignal = false, className = '' }) => {
        let valClass = isSignal ? 'signal-tag' : 'metric-value';

        // Auto color coding based on label and value
        let autoColor = null;
        const numVal = parseFloat(value);
        const labelLower = label.toLowerCase();

        // Define thresholds for specific metrics
        if (!isNaN(numVal) && isGood === null && !className) {
            // RSI thresholds
            if (labelLower.includes('rsi')) {
                if (numVal > 70) autoColor = 'text-red';        // Overbought
                else if (numVal < 30) autoColor = 'text-green';  // Oversold
                else if (numVal > 60 || numVal < 40) autoColor = 'text-yellow';
            }
            // ADX - Trend strength
            else if (labelLower === 'adx') {
                if (numVal > 40) autoColor = 'text-green';       // Strong trend
                else if (numVal > 25) autoColor = 'text-yellow'; // Moderate trend
                else autoColor = 'text-red';                      // Weak/No trend
            }
            // P/E Ratio
            else if (labelLower.includes('p/e') && !labelLower.includes('vs')) {
                if (numVal < 15) autoColor = 'text-green';       // Undervalued
                else if (numVal > 30) autoColor = 'text-red';    // Overvalued
                else if (numVal > 20) autoColor = 'text-yellow';
            }
            // Volatility - lower is better
            else if (labelLower === 'volatility') {
                if (numVal > 40) autoColor = 'text-red';         // High volatility
                else if (numVal > 25) autoColor = 'text-yellow';
                else autoColor = 'text-green';
            }
            // Beta
            else if (labelLower === 'beta') {
                if (numVal > 1.5) autoColor = 'text-red';        // High risk
                else if (numVal < 0.8) autoColor = 'text-green'; // Low risk
                else autoColor = 'text-yellow';
            }
            // Sharpe Ratio
            else if (labelLower.includes('sharpe')) {
                if (numVal > 1.5) autoColor = 'text-green';      // Excellent
                else if (numVal > 1) autoColor = 'text-yellow';  // Good
                else autoColor = 'text-red';                     // Poor
            }
            // Debt/Equity
            else if (labelLower.includes('debt')) {
                if (numVal < 0.5) autoColor = 'text-green';      // Low debt
                else if (numVal > 1.5) autoColor = 'text-red';   // High debt
                else autoColor = 'text-yellow';
            }
            // Current Ratio
            else if (labelLower.includes('current ratio')) {
                if (numVal > 2) autoColor = 'text-green';        // Strong liquidity
                else if (numVal < 1) autoColor = 'text-red';     // Weak liquidity
                else autoColor = 'text-yellow';
            }
            // MFI
            else if (labelLower === 'mfi') {
                if (numVal > 80) autoColor = 'text-red';         // Overbought
                else if (numVal < 20) autoColor = 'text-green';  // Oversold
            }
            // Z-Score
            else if (labelLower.includes('z-score')) {
                if (Math.abs(numVal) > 2) autoColor = 'text-green';  // Strong signal
                else if (Math.abs(numVal) > 1) autoColor = 'text-yellow';
            }
            // Confidence metrics
            else if (labelLower.includes('confidence') || labelLower.includes('conf.') || labelLower.includes('accuracy')) {
                if (numVal > 80) autoColor = 'text-green';       // High confidence
                else if (numVal > 60) autoColor = 'text-yellow';
                else autoColor = 'text-red';                     // Low confidence
            }
            // PEG Ratio
            else if (labelLower.includes('peg')) {
                if (numVal < 1) autoColor = 'text-green';        // Undervalued
                else if (numVal > 2) autoColor = 'text-red';     // Overvalued
            }
            // Volume Ratio
            else if (labelLower.includes('volume ratio')) {
                if (numVal > 1.5) autoColor = 'text-green';      // High volume
                else if (numVal < 0.7) autoColor = 'text-red';   // Low volume
            }
            // Max Drawdown
            else if (labelLower.includes('drawdown')) {
                if (numVal < -30) autoColor = 'text-red';
                else if (numVal < -15) autoColor = 'text-yellow';
            }
        }

        if (isGood === true) valClass += ' text-green';
        else if (isGood === false) valClass += ' text-red';
        else if (className) valClass += ` ${className}`;
        else if (autoColor) valClass += ` ${autoColor}`;

        // Handling special signal text coloring logic from app.js
        if (isSignal && !className && !autoColor) {
            const v = (value || "").toString().toUpperCase();
            if (v.includes("BUY") || v.includes("BULLISH") || v.includes("UNDERVALUED") || v.includes("FAIR") || v.includes("ENTRY OK") || v.includes("STRONG") || v.includes("CONFIRMED") || v.includes("YES") || v.includes("ACTIVE") || v === "UP" || v === "UPTREND") {
                valClass += " text-green";
            } else if (v.includes("SELL") || v.includes("BEARISH") || v.includes("OVERVALUED") || v.includes("OVERBOUGHT") || v.includes("HIGH") || v.includes("ELEVATED") || v.includes("NO") || v === "DOWN" || v === "DOWNTREND") {
                valClass += " text-red";
            } else if (v.includes("WAIT") || v.includes("CAUTION") || v.includes("NEUTRAL") || v.includes("WEAK") || v.includes("HOLD") || v.includes("MODERATE") || v === "NONE") {
                valClass += " text-yellow";
            }
        }

        return (
            <div className="metric-row">
                <span className="metric-label">{label}</span>
                <span className={valClass}>{value !== undefined && value !== null ? value + suffix : '-'}</span>
            </div>
        );
    };

    // Calculate complex values
    const activePatterns = Object.keys(patterns || {}).filter(k => patterns[k] !== 'No pattern detected');
    let patternText = 'None';
    if (activePatterns.length > 0) {
        patternText = activePatterns.slice(0, 2).join(', ').replace(/_/g, ' ');
        if (activePatterns.length > 2) patternText += ` (+${activePatterns.length - 2})`;
    }
    patternText = patternText.replace(/\b\w/g, l => l.toUpperCase());

    const maxDD = risk?.max_drawdown?.max_drawdown !== undefined ? risk.max_drawdown.max_drawdown * 100 : risk?.max_drawdown?.max_drawdown_pct;

    return (
        <section id="dataGrid" className="data-grid">
            {/* Technical Column */}
            <div className="data-column">
                <div className="col-header">TECHNICAL</div>
                <MetricRow label="RSI (14)" value={tech.rsi?.toFixed(2)} />
                <div className="metric-row">
                    <span className="metric-label">MACD</span>
                    {tech.macd !== undefined ? (
                        <span className={`metric-value ${tech.macd_histogram > 0 ? 'text-green' : tech.macd_histogram < 0 ? 'text-red' : ''}`}>
                            {tech.macd?.toFixed(2)} / {tech.macd_signal?.toFixed(2)}
                        </span>
                    ) : <span className="metric-value">-</span>}
                </div>
                <MetricRow label="ADX" value={tech.adx?.toFixed(2)} />
                <MetricRow label="ATR" value={tech.atr?.toFixed(2)} />
            </div>

            {/* Fundamental Column */}
            <div className="data-column">
                <div className="col-header">FUNDAMENTAL</div>
                <MetricRow label="P/E (TTM)" value={fund.pe_ratio?.toFixed(2)} />
                <MetricRow label="Forward P/E" value={fund.forward_pe?.toFixed(2)} />
                <MetricRow label="Market Cap" value={formatLarge(fund.market_cap)} />
                <MetricRow label="Div Yield" value={fund.dividend_yield} suffix="%" />
                <MetricRow label="Net Margin" value={fund.net_margin} suffix="%" isGood={fund.net_margin > 0} />
                <MetricRow label="Oper. Margin" value={fund.operating_margin} suffix="%" isGood={fund.operating_margin > 0} />
                <MetricRow label="ROE" value={fund.roe} suffix="%" isGood={fund.roe > 0} />
            </div>

            {/* Signals Column */}
            <div className="data-column">
                <div className="col-header">SIGNALS & RISK</div>
                <MetricRow label="Signal" value={signals.signal} isSignal={true} />
                <MetricRow label="Strength" value={signals.signal_strength} suffix="%" isGood={signals.signal_strength > 55} />
                <MetricRow label="PE vs Industry" value={analysis.pe_vs_industry} isSignal={true} />
                <MetricRow label="Overvaluation" value={analysis.overvaluation_status} isSignal={true} />
                <MetricRow label="MACD Divergence" value={analysis.macd_divergence} isSignal={true} />
                <MetricRow label="RSI Status" value={signals.confidence_adjustments?.rsi_status?.replace(/_/g, ' ')} isSignal={true} />
                <MetricRow label="Pullback Target" value={analysis.pullback_target} />
                <MetricRow label="Stop Loss" value={signals.risk_management ? `$${signals.risk_management.stop_loss_price} (${signals.risk_management.stop_loss_pct}%)` : null} isGood={false} />
                <MetricRow label="Position Size" value={signals.risk_management?.position_size_pct} suffix="%" />
            </div>

            {/* ML PREDICTIONS Column */}
            <div className="data-column">
                <div className="col-header">ML PREDICTIONS</div>
                <MetricRow label="ML Direction" value={mlPrediction?.ensemble_direction} isSignal={true} />
                <MetricRow label="ML Confidence" value={mlPrediction?.ensemble_confidence?.toFixed(1)} suffix="%" isGood={mlPrediction?.ensemble_confidence > 70} />
                <MetricRow label="Random Forest" value={mlPrediction?.rf_prediction} isSignal={true} />
                <MetricRow label="SVM" value={mlPrediction?.svm_prediction} isSignal={true} />
                <MetricRow label="Momentum" value={mlPrediction?.momentum_prediction} isSignal={true} />
                <MetricRow label="Models Agree" value={mlPrediction?.models_agree === true ? 'YES' : mlPrediction?.models_agree === false ? 'NO' : '-'} isSignal={true} />
                <MetricRow label="Top Feature" value={Object.keys(mlPrediction?.feature_importance || {})[0]?.replace(/_/g, ' ').toUpperCase()} />
                <MetricRow label="Cluster Profile" value={mlPrediction?.cluster_profile} />
            </div>

            {/* Staged Entry Column */}
            <div className="data-column">
                <div className="col-header">STAGED ENTRY</div>
                <MetricRow label="Position Size" value={stagedEntry?.recommended_position_pct} suffix="%" isSignal={true} />
                <MetricRow label="Market Regime" value={marketRegimeData?.current_regime} isSignal={true} />
                <MetricRow label="Regime Override" value={stagedEntry?.regime_override_applied ? "APPLIED" : "NO"} isSignal={stagedEntry?.regime_override_applied} />
                <MetricRow label="Ideal Entry" value={stagedEntry?.ideal_entry_price ? `$${stagedEntry.ideal_entry_price}` : '-'} className="text-green" />
                <MetricRow label="Price vs Ideal" value={stagedEntry?.price_vs_ideal_pct?.toFixed(1)} suffix="%" isGood={stagedEntry?.price_vs_ideal_pct <= 0} />
                <MetricRow label="Entry Tier 1" value={stagedEntry?.entry_tiers?.[0] ? `$${stagedEntry.entry_tiers[0].price} (${stagedEntry.entry_tiers[0].position_pct}%)` : '-'} className="text-yellow" />
                <MetricRow label="Entry Tier 2" value={stagedEntry?.entry_tiers?.[1] ? `$${stagedEntry.entry_tiers[1].price} (${stagedEntry.entry_tiers[1].position_pct}%)` : '-'} className="text-green" />
            </div>

            {/* Valuation Column */}
            <div className="data-column">
                <div className="col-header">VALUATION</div>
                <MetricRow label="DCF Intrinsic" value={valSnapshot?.dcf_intrinsic_value ? `$${(typeof valSnapshot.dcf_intrinsic_value === 'number' ? valSnapshot.dcf_intrinsic_value.toFixed(2) : valSnapshot.dcf_intrinsic_value)}` : '-'} className="text-green" />
                <MetricRow label="Overvaluation" value={valSnapshot?.dcf_overvaluation_pct ? `${valSnapshot.dcf_overvaluation_pct > 0 ? '+' : ''}${valSnapshot.dcf_overvaluation_pct.toFixed(1)}%` : '-'} isSignal={true} />
                <MetricRow label="PEG Ratio" value={valSnapshot?.peg_ratio?.toFixed(2) || val.valuation_metrics?.peg_ratio?.toFixed(2)} />
                <MetricRow label="PEG Status" value={signals.confidence_adjustments?.peg_status?.replace(/_/g, ' ')} isSignal={true} />
                <MetricRow label="Fair Value" value={val.dcf_valuation?.fair_value?.toFixed(2) || val.ddm_valuation?.value_per_share?.toFixed(2)} suffix={val.dcf_valuation ? ' (DCF)' : val.ddm_valuation ? ' (DDM)' : ''} />
                <MetricRow label="P/B Ratio" value={val.valuation_metrics?.pb_ratio?.toFixed(2)} />
                <MetricRow label="P/S Ratio" value={val.valuation_metrics?.ps_ratio?.toFixed(2)} />
                <MetricRow label="EV/EBITDA" value={val.valuation_metrics?.ev_ebitda?.toFixed(2)} />
            </div>

            {/* Volume Analysis */}
            <div className="data-column">
                <div className="col-header">VOLUME ANALYSIS</div>
                <MetricRow label="Volume Ratio" value={volumeConf?.volume_ratio?.toFixed(2)} suffix="x" isGood={volumeConf?.volume_ratio > 1} />
                <MetricRow label="Conviction" value={volumeConf?.volume_conviction || signals.confidence_adjustments?.volume_conviction} isSignal={true} />
                <MetricRow label="OBV Trend" value={volumeConf?.obv_trend} isSignal={true} />
                <MetricRow label="MFI" value={volumeConf?.mfi} />
                <MetricRow label="MFI Signal" value={volumeConf?.mfi_signal} isSignal={true} />
            </div>

            {/* Pivot Levels */}
            <div className="data-column">
                <div className="col-header">PIVOT LEVELS</div>
                <MetricRow label="Pivot Point" value={pivotLevels?.pivot_point ? `$${pivotLevels.pivot_point}` : '-'} />
                <MetricRow label="Resistance R1" value={pivotLevels?.resistance_r1 ? `$${pivotLevels.resistance_r1}` : '-'} className="text-red" />
                <MetricRow label="Resistance R2" value={pivotLevels?.resistance_r2 ? `$${pivotLevels.resistance_r2}` : '-'} className="text-red" />
                <MetricRow label="Support S1" value={pivotLevels?.support_s1 ? `$${pivotLevels.support_s1}` : '-'} className="text-green" />
                <MetricRow label="Support S2" value={pivotLevels?.support_s2 ? `$${pivotLevels.support_s2}` : '-'} className="text-green" />
            </div>

            {/* Fibonacci Levels */}
            <div className="data-column">
                <div className="col-header">FIBONACCI LEVELS</div>
                <MetricRow label="Trend" value={fibLevels?.trend} isSignal={true} />
                <MetricRow label="Swing High" value={fibLevels?.swing_high ? `$${fibLevels.swing_high.toFixed(2)}` : '-'} className="text-red" />
                <MetricRow label="Swing Low" value={fibLevels?.swing_low ? `$${fibLevels.swing_low.toFixed(2)}` : '-'} className="text-green" />
                <MetricRow label="Fib 23.6%" value={fibLevels?.fib_236 ? `$${fibLevels.fib_236.toFixed(2)}` : '-'} />
                <MetricRow label="Fib 38.2%" value={fibLevels?.fib_382 ? `$${fibLevels.fib_382.toFixed(2)}` : '-'} />
                <MetricRow label="Fib 50%" value={fibLevels?.fib_500 ? `$${fibLevels.fib_500.toFixed(2)}` : '-'} />
                <MetricRow label="Fib 61.8%" value={fibLevels?.fib_618 ? `$${fibLevels.fib_618.toFixed(2)}` : '-'} />
            </div>

            {/* Risk & Health */}
            <div className="data-column">
                <div className="col-header">RISK & HEALTH</div>
                <MetricRow label="Volatility" value={risk.volatility?.toFixed(2)} suffix="%" />
                <MetricRow label="Sharpe Ratio" value={risk.sharpe_ratio?.toFixed(2)} isGood={risk.sharpe_ratio > 1} />
                <MetricRow label="Max Drawdown" value={maxDD?.toFixed(2)} suffix="%" isGood={false} />
                <MetricRow label="Beta" value={fund.beta?.toFixed(2)} />
                <MetricRow label="Debt/Equity" value={fund.debt_to_equity} />
                <MetricRow label="Current Ratio" value={fund.current_ratio} />
            </div>

            {/* Enhanced AI */}
            <div className="data-column">
                <div className="col-header">ENHANCED AI</div>
                <MetricRow label="LSTM Forecast" value={prediction.lstm_prediction?.direction} isSignal={true} />
                <MetricRow label="LSTM Confidence" value={prediction.lstm_prediction?.confidence?.toFixed(1)} suffix="%" />
                <MetricRow label="5-Day Target" value={prediction.lstm_prediction?.predictions?.day_5 ? `$${prediction.lstm_prediction.predictions.day_5.toFixed(2)}` : '-'} />
                <MetricRow label="Ensemble Score" value={prediction.confidence?.toFixed(1)} suffix="%" />
                <MetricRow label="Sector Strength" value={enhancedPred.sector_analysis?.relative_strength?.rating?.replace(/_/g, ' ')} isSignal={true} />
                <MetricRow label="Insider Signal" value={enhancedPred.alternative_data?.insider_activity?.signal?.replace(/_/g, ' ')} isSignal={true} />
                <MetricRow label="Options Flow" value={enhancedPred.alternative_data?.options_flow?.signal} isSignal={true} />
                <MetricRow label="Anomaly Alerts" value={enhancedPred.anomaly_alerts?.total_alerts || 'NONE'} isSignal={true} />
            </div>

            {/* Advanced ML */}
            <div className="data-column">
                <div className="col-header">ADVANCED ML</div>
                <MetricRow label="XGBoost" value={enhancedPred.xgboost_prediction?.direction} isSignal={true} />
                <MetricRow label="XGBoost Conf." value={enhancedPred.xgboost_prediction?.confidence?.toFixed(1)} suffix="%" />
                <MetricRow label="GRU" value={enhancedPred.gru_prediction?.direction} isSignal={true} />
                <MetricRow label="GRU Conf." value={enhancedPred.gru_prediction?.confidence?.toFixed(1)} suffix="%" />
                <MetricRow label="CNN-LSTM" value={enhancedPred.cnn_lstm_prediction?.direction} isSignal={true} />
                <MetricRow label="CNN-LSTM Conf." value={enhancedPred.cnn_lstm_prediction?.confidence?.toFixed(1)} suffix="%" />
                <MetricRow label="Attention" value={enhancedPred.attention_prediction?.direction} isSignal={true} />
                <MetricRow label="Attention Conf." value={enhancedPred.attention_prediction?.confidence?.toFixed(1)} suffix="%" />
                <MetricRow label="GARCH Model" value={enhancedPred.volatility_forecast?.model} />
                <MetricRow label="Vol. Regime" value={enhancedPred.volatility_forecast?.regime?.regime} isSignal={true} />
            </div>

            {/* Pair Trading */}
            <div className="data-column">
                <div className="col-header">PAIR TRADING{!aiData && loading ? ' (LOADING...)' : ''}</div>
                {!aiData && !loading ? (
                    <MetricRow label="Status" value="AWAITING AI" isSignal={true} />
                ) : statArb.error ? (
                    <MetricRow label="Status" value="UNAVAILABLE" isSignal={true} />
                ) : statArb.market_pair ? (
                    <>
                        <MetricRow label="Market Pair" value={statArb.market_pair?.pair || statArb.sector_pair?.pair || (statArb.sector_etf ? `vs ${statArb.sector_etf}` : 'vs SPY')} />
                        <MetricRow label="Cointegrated" value={statArb.market_pair?.cointegration?.is_cointegrated ? 'YES' : 'NO'} isSignal={statArb.market_pair?.cointegration?.is_cointegrated} />
                        <MetricRow label="Z-Score" value={statArb.market_pair?.current_signal?.current_zscore?.toFixed(2)} />
                        <MetricRow label="Signal" value={statArb.market_pair?.current_signal?.signal?.replace(/_/g, ' ')} isSignal={true} />
                        <MetricRow label="Confidence" value={statArb.market_pair?.current_signal?.confidence?.toFixed(1)} suffix="%" />
                        <MetricRow label="Half-Life" value={statArb.market_pair?.cointegration?.half_life_days?.toFixed(0)} suffix=" days" />
                        <MetricRow label="Hedge Ratio" value={statArb.market_pair?.cointegration?.hedge_ratio?.toFixed(3)} />
                        <MetricRow label="Opportunity" value={statArb.has_opportunity ? 'ACTIVE' : 'NONE'} isSignal={true} />
                    </>
                ) : loading ? (
                    <MetricRow label="Status" value="ANALYZING..." isSignal={true} />
                ) : (
                    <MetricRow label="Status" value="NO DATA" isSignal={true} />
                )}
            </div>

            {/* Patterns & Sentiment */}
            <div className="data-column">
                <div className="col-header">PATTERNS & SENTIMENT</div>
                <MetricRow label="Candlestick Pat." value={patternText} />
                <MetricRow label="News Sentiment" value={enhancedPred.alternative_data?.social_sentiment?.signal?.replace(/_/g, ' ') || alt.social_sentiment?.news_sentiment} isSignal={true} />
                <MetricRow label="Sentiment Score" value={enhancedPred.alternative_data?.social_sentiment?.sentiment_score?.toFixed(2) || alt.social_sentiment?.sentiment_score?.toFixed(2)} />
                <MetricRow label="Web Traffic" value={formatLarge(alt.web_traffic?.value || alt.web_traffic?.monthly_visits)} />
            </div>

            {/* Market Context (Populated from AI Data) */}
            <div className="data-column">
                <div className="col-header">MARKET CONTEXT{!aiData && loading ? ' (LOADING...)' : ''}</div>
                {aiData?.error ? (
                    <div className="metric-row"><span className="metric-label text-yellow">AI Unavailable</span></div>
                ) : (() => {
                    // Helper to parse yield curve data: "-0.68 (Inverted (Recession Signal))"
                    const yieldCurveRaw = aiData?.macro_context?.yield_curve || '';
                    const isInverted = yieldCurveRaw.toLowerCase().includes('inverted');
                    // Extract value: first part before space or entire string
                    const firstSpaceIdx = yieldCurveRaw.indexOf(' ');
                    const yieldCurveValue = firstSpaceIdx > 0 ? yieldCurveRaw.substring(0, firstSpaceIdx) : yieldCurveRaw;
                    // Extract status: everything after the value, removing outer parentheses
                    let yieldCurveStatus = '';
                    if (firstSpaceIdx > 0) {
                        let statusPart = yieldCurveRaw.substring(firstSpaceIdx + 1).trim();
                        // Remove outer parentheses if present
                        if (statusPart.startsWith('(') && statusPart.endsWith(')')) {
                            statusPart = statusPart.slice(1, -1);
                        }
                        yieldCurveStatus = statusPart;
                    }

                    // VIX thresholds
                    const vixValue = aiData?.macro_context?.vix;
                    const vixIsHigh = vixValue && vixValue > 20;
                    const vixIsLow = vixValue && vixValue < 15;

                    return (
                        <>
                            <MetricRow label="S&P 500" value={aiData?.macro_context?.sp500 || (loading ? '...' : undefined)} />
                            <MetricRow
                                label="VIX"
                                value={vixValue || (loading ? '...' : undefined)}
                                isGood={vixIsLow}
                                className={vixIsHigh ? 'text-red' : vixIsLow ? 'text-green' : ''}
                            />
                            <MetricRow label="10Y Yield" value={aiData?.macro_context?.treasury_yield_10y ? `${aiData.macro_context.treasury_yield_10y}%` : (loading ? '...' : undefined)} />
                            <div className="metric-row">
                                <span className="metric-label">Yield Curve</span>
                                <span className={`metric-value ${isInverted ? 'text-red' : 'text-green'}`}>
                                    {loading && !yieldCurveRaw ? '...' : yieldCurveValue || '-'}
                                </span>
                            </div>
                            {yieldCurveStatus && yieldCurveRaw && (
                                <div className="metric-row">
                                    <span className="metric-label"></span>
                                    <span className={`metric-value ${isInverted ? 'text-red' : 'text-green'}`} style={{ fontSize: '0.75rem' }}>
                                        {yieldCurveStatus}
                                    </span>
                                </div>
                            )}
                            <MetricRow label="Top Sector" value={aiData?.macro_context?.top_sector || (loading ? '...' : undefined)} isSignal={true} />
                        </>
                    );
                })()}
            </div>
        </section>
    );
};

export default DataGrid;
