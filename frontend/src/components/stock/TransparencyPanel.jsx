import { useState } from 'react';
import { useStock } from '../../context/StockContext';

const TransparencyPanel = () => {
    const { signalsData } = useStock();
    const [isOpen, setIsOpen] = useState(true);

    const data = signalsData?.enhanced_prediction;
    if (!data) return <section id="modelTransparency" className="transparency-panel hidden"></section>;

    const toggle = () => setIsOpen(!isOpen);

    const metrics = data.model_metrics || {};
    const volatility = data.volatility_forecast || {};
    const sector = data.sector_analysis || {};
    const anomalies = data.anomaly_alerts || {};

    return (
        <section id="modelTransparency" className="transparency-panel">
            <div className="panel-header-small" onClick={toggle} style={{ cursor: 'pointer' }}>
                <h4>MODEL TRANSPARENCY & ACCURACY</h4>
                <svg className={`expand-icon ${isOpen ? 'rotate-180' : ''}`} width="24" height="24" viewBox="0 0 24 24" fill="none"
                    stroke="currentColor" strokeWidth="2">
                    <path d="M6 9l6 6 6-6"></path>
                </svg>
            </div>
            <div id="transparencyDetails" className={`transparency-content ${isOpen ? '' : 'hidden'}`}>
                {/* 0. Global Risk Metrics (New) */}
                <div style={{ marginBottom: '20px', padding: '15px', background: 'rgba(25, 30, 50, 0.6)', borderRadius: '8px', display: 'flex', justifyContent: 'space-between', flexWrap: 'wrap', gap: '10px' }}>
                    <div style={{ textAlign: 'center', flex: '1 1 30%' }}>
                        <div style={{ fontSize: '0.8rem', color: '#888' }}>SHARPE RATIO</div>
                        <div style={{ fontSize: '1.2rem', fontWeight: 'bold', color: data.risk_metrics?.sharpe_ratio >= 1 ? '#4caf50' : '#fff' }}>
                            {data.risk_metrics?.sharpe_ratio || 'N/A'}
                        </div>
                    </div>
                    <div style={{ textAlign: 'center', flex: '1 1 30%' }}>
                        <div style={{ fontSize: '0.8rem', color: '#888' }}>MAX DRAWDOWN</div>
                        <div style={{ fontSize: '1.2rem', fontWeight: 'bold', color: '#ef5350' }}>
                            {data.risk_metrics?.max_drawdown ? `${data.risk_metrics.max_drawdown}%` : 'N/A'}
                        </div>
                    </div>
                    <div style={{ textAlign: 'center', flex: '1 1 30%' }}>
                        <div style={{ fontSize: '0.8rem', color: '#888' }}>VOL. REGIME</div>
                        <div style={{ fontSize: '1.2rem', fontWeight: 'bold', color: '#fff' }}>
                            {data.risk_metrics?.volatility_regime || 'Unknown'}
                        </div>
                    </div>
                    <div style={{ textAlign: 'center', flex: '1 1 30%', marginTop: '10px' }}>
                        <div style={{ fontSize: '0.8rem', color: '#888' }}>MODELS AGREE</div>
                        <div style={{ fontSize: '1.1rem', fontWeight: 'bold', color: '#4facfe' }}>
                            {data.enhanced_prediction?.ml_models?.models_agree ? 'YES' : 'NO'}
                        </div>
                    </div>
                </div>

                {/* 1. Deep Learning Models */}
                <div className="models-grid" id="modelsGrid">
                    {Object.entries(metrics).map(([model, stats]) => (
                        <div key={model} className="model-card">
                            <div className="model-name">{model.toUpperCase().replace('_', ' ')}</div>

                            {/* Direction & Confidence (New) */}
                            {(stats.direction || stats.confidence) && (
                                <div className="model-stat-row" style={{ marginBottom: '8px', paddingBottom: '8px', borderBottom: '1px solid rgba(255,255,255,0.1)' }}>
                                    <span className={stats.direction === 'Bullish' ? 'text-green' : stats.direction === 'Bearish' ? 'text-red' : 'text-gray'}>
                                        {stats.direction?.toUpperCase() || 'NEUTRAL'}
                                    </span>
                                    <span style={{ fontSize: '0.8rem' }}>{stats.confidence}% Conf.</span>
                                </div>
                            )}

                            <div className="model-stat-row">
                                <span>Accuracy</span>
                                <span className="text-green">{stats.accuracy}%</span>
                            </div>
                            <div className="model-stat-row">
                                <span>MAPE</span>
                                <span className="mape-badge">{stats.mape}%</span>
                            </div>
                        </div>
                    ))}
                </div>

                {/* 2. Volatility Forecast (New) */}
                {volatility.forecast && (
                    <div style={{ marginTop: '20px', padding: '15px', background: 'rgba(25, 30, 50, 0.6)', borderRadius: '8px' }}>
                        <h5 style={{ margin: '0 0 10px 0', fontSize: '0.9rem', color: '#888' }}>VOLATILITY FORECAST ({volatility.model})</h5>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                            <div>
                                <div style={{ fontSize: '0.8rem', color: '#aaa' }}>Current Regime</div>
                                <div style={{ color: '#fff', fontWeight: 'bold' }}>{volatility.regime?.current_regime || 'Unknown'}</div>
                            </div>
                            <div>
                                <div style={{ fontSize: '0.8rem', color: '#aaa' }}>Forecast (5d)</div>
                                <div style={{ color: '#4facfe', fontWeight: 'bold' }}>
                                    {volatility.forecast && volatility.forecast.length > 0
                                        ? `${(volatility.forecast[0] * 100).toFixed(2)}%`
                                        : 'N/A'}
                                </div>
                            </div>
                        </div>
                    </div>
                )}

                {/* 3. Sector Analysis (New) */}
                {sector.sector_name && (
                    <div style={{ marginTop: '15px', padding: '15px', background: 'rgba(25, 30, 50, 0.6)', borderRadius: '8px' }}>
                        <h5 style={{ margin: '0 0 10px 0', fontSize: '0.9rem', color: '#888' }}>SECTOR: {sector.sector_name.toUpperCase()}</h5>
                        <div className="model-stat-row">
                            <span>Sector Strength</span>
                            <span className={(sector.sector_strength > 0 || sector.relative_strength > 0) ? 'text-green' : 'text-red'}>
                                {typeof sector.sector_strength === 'number' ? sector.sector_strength.toFixed(2) : (typeof sector.relative_strength === 'number' ? sector.relative_strength.toFixed(2) : 'N/A')}
                            </span>
                        </div>
                        <div className="model-stat-row">
                            <span>Rel. Strength (RSI)</span>
                            <span className="mape-badge">{sector.sector_rsi?.toFixed(1) || 'N/A'}</span>
                        </div>
                    </div>
                )}

                {/* 4. Anomaly Alerts (New) */}
                {anomalies.alerts && anomalies.alerts.length > 0 && (
                    <div style={{ marginTop: '15px', padding: '15px', background: 'rgba(50, 20, 20, 0.4)', borderRadius: '8px', border: '1px solid rgba(239, 83, 80, 0.3)' }}>
                        <h5 style={{ margin: '0 0 10px 0', fontSize: '0.9rem', color: '#ef5350' }}>ANOMALY ALERTS ({anomalies.total_alerts})</h5>
                        {anomalies.alerts.map((alert, index) => (
                            <div key={index} style={{ marginBottom: '8px', paddingBottom: '8px', borderBottom: index < anomalies.alerts.length - 1 ? '1px solid rgba(255,255,255,0.1)' : 'none' }}>
                                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                                    <span style={{ color: '#fff', fontWeight: 'bold', fontSize: '0.85rem' }}>
                                        {alert.type.replace('_', ' ').toUpperCase()}
                                    </span>
                                    <span style={{
                                        fontSize: '0.7rem',
                                        padding: '2px 6px',
                                        borderRadius: '4px',
                                        backgroundColor: alert.severity === 'high' ? 'rgba(239, 83, 80, 0.8)' : 'rgba(255, 167, 38, 0.8)',
                                        color: '#fff'
                                    }}>
                                        {alert.severity.toUpperCase()}
                                    </span>
                                </div>
                                <div style={{ fontSize: '0.8rem', color: '#ccc', marginTop: '4px' }}>
                                    {alert.description}
                                </div>
                                <div style={{ fontSize: '0.7rem', color: '#888', marginTop: '2px' }}>
                                    {alert.date}
                                </div>
                            </div>
                        ))}
                    </div>
                )}

                {/* 5. Alternative Data (New) */}
                {data.alternative_data && (
                    <div style={{ marginTop: '15px', padding: '15px', background: 'rgba(25, 30, 50, 0.6)', borderRadius: '8px' }}>
                        <h5 style={{ margin: '0 0 10px 0', fontSize: '0.9rem', color: '#888' }}>ALTERNATIVE SIGNALS</h5>
                        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px' }}>
                            <div className="model-stat-row">
                                <span>Insider Trading</span>
                                <span className={data.alternative_data.insider_activity?.trend === 'Buying' ? 'text-green' : data.alternative_data.insider_activity?.trend === 'Selling' ? 'text-red' : 'text-gray'}>
                                    {data.alternative_data.insider_activity?.trend || 'N/A'}
                                </span>
                            </div>
                            <div className="model-stat-row">
                                <span>Options Flow</span>
                                <span className={data.alternative_data.options_flow?.sentiment === 'Bullish' ? 'text-green' : data.alternative_data.options_flow?.sentiment === 'Bearish' ? 'text-red' : 'text-gray'}>
                                    {data.alternative_data.options_flow?.sentiment || 'N/A'}
                                </span>
                            </div>
                            <div className="model-stat-row">
                                <span>Instit. Holdings</span>
                                <span className={data.alternative_data.institutional_holdings?.change_pct > 0 ? 'text-green' : 'text-red'}>
                                    {data.alternative_data.institutional_holdings?.change_pct ? `${data.alternative_data.institutional_holdings.change_pct}%` : 'N/A'}
                                </span>
                            </div>
                            <div className="model-stat-row">
                                <span>Social Sentiment</span>
                                <span className={data.alternative_data.social_sentiment?.score > 0 ? 'text-green' : 'text-red'}>
                                    {data.alternative_data.social_sentiment?.score || 'N/A'}
                                </span>
                            </div>
                        </div>
                    </div>
                )}

                {/* 6. Statistical Arbitrage (New) */}
                {data.statistical_arbitrage && (
                    <div style={{ marginTop: '15px', padding: '15px', background: 'rgba(25, 30, 50, 0.6)', borderRadius: '8px' }}>
                        <h5 style={{ margin: '0 0 10px 0', fontSize: '0.9rem', color: '#888' }}>STATISTICAL ARBITRAGE</h5>
                        <div className="model-stat-row">
                            <span>Market Pair (SPY)</span>
                            <span style={{ color: '#fff' }}>
                                {data.statistical_arbitrage.market_pair?.current_signal?.signal || 'NO SIGNAL'}
                            </span>
                        </div>
                        {data.statistical_arbitrage.sector_pair && (
                            <div className="model-stat-row">
                                <span>Sector Pair ({data.statistical_arbitrage.sector_etf})</span>
                                <span style={{ color: '#fff' }}>
                                    {data.statistical_arbitrage.sector_pair?.current_signal?.signal || 'NO SIGNAL'}
                                </span>
                            </div>
                        )}
                    </div>
                )}

                {/* 7. Wavelet & Preprocessing (New) */}
                <div style={{ marginTop: '15px', display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '15px' }}>
                    {data.wavelet_analysis && (
                        <div style={{ padding: '15px', background: 'rgba(25, 30, 50, 0.6)', borderRadius: '8px' }}>
                            <h5 style={{ margin: '0 0 10px 0', fontSize: '0.9rem', color: '#888' }}>WAVELET DENOISING</h5>
                            <div className="model-stat-row">
                                <span>Noise Removed</span>
                                <span className="text-green">{data.wavelet_analysis.noise_removed_pct}%</span>
                            </div>
                            <div className="model-stat-row">
                                <span>Clarified Trend</span>
                                <span style={{ color: '#fff' }}>{data.wavelet_analysis.trend_clarity}</span>
                            </div>
                        </div>
                    )}

                    {data.preprocessing_metrics && (
                        <div style={{ padding: '15px', background: 'rgba(25, 30, 50, 0.6)', borderRadius: '8px' }}>
                            <h5 style={{ margin: '0 0 10px 0', fontSize: '0.9rem', color: '#888' }}>DATA QUALITY</h5>
                            <div className="model-stat-row">
                                <span>Quality Score</span>
                                <span className="text-green">{data.preprocessing_metrics.data_quality_score}%</span>
                            </div>
                            <div className="model-stat-row">
                                <span>Outliers Handled</span>
                                <span style={{ color: '#fff' }}>{data.preprocessing_metrics.outliers_handled}</span>
                            </div>
                        </div>
                    )}
                </div>
            </div>
        </section>
    );
};

export default TransparencyPanel;
