import { useState } from 'react';
import { useStock } from '../../context/StockContext';

const DataQualityPanel = () => {
    const { signalsData, aiData } = useStock();
    const [isOpen, setIsOpen] = useState(true);

    // Get preprocessing metrics from either endpoint
    const metrics = signalsData?.enhanced_prediction?.preprocessing_metrics ||
        aiData?.preprocessing_metrics;

    // Get market sentiment
    const marketSentiment = signalsData?.enhanced_prediction?.market_sentiment ||
        aiData?.macro_context;

    // Get quantified sentiment
    const quantifiedSentiment = aiData?.alternative_data?.quantified_sentiment;

    if (!metrics && !marketSentiment && !quantifiedSentiment) {
        return null;
    }

    const toggle = () => setIsOpen(!isOpen);

    const wavelet = metrics?.wavelet_denoising || {};
    const outliers = metrics?.outlier_analysis || {};
    const dataQuality = metrics?.data_quality || {};
    const fearGreed = marketSentiment?.fear_greed || {};
    const vix = marketSentiment?.vix || {};

    return (
        <section className="data-quality-panel">
            <div className="panel-header-small" onClick={toggle} style={{ cursor: 'pointer' }}>
                <h4>DATA QUALITY & MARKET SENTIMENT</h4>
                <svg className={`expand-icon ${isOpen ? 'rotate-180' : ''}`} width="24" height="24" viewBox="0 0 24 24" fill="none"
                    stroke="currentColor" strokeWidth="2">
                    <path d="M6 9l6 6 6-6"></path>
                </svg>
            </div>

            <div className={`transparency-content ${isOpen ? '' : 'hidden'}`}>
                <div className="models-grid">
                    {/* Fear & Greed Index */}
                    {(fearGreed?.index || marketSentiment?.fear_greed_index) && (
                        <div className="model-card">
                            <div className="model-name">FEAR & GREED</div>
                            <div className="model-stat-row">
                                <span>Index</span>
                                <span className={fearGreed?.index > 50 ? 'text-green' : 'text-red'}>
                                    {fearGreed?.index || marketSentiment?.fear_greed_index}
                                </span>
                            </div>
                            <div className="model-stat-row">
                                <span>Classification</span>
                                <span className="mape-badge">
                                    {fearGreed?.classification || marketSentiment?.fear_greed_classification}
                                </span>
                            </div>
                        </div>
                    )}

                    {/* VIX Data */}
                    {(vix?.current || marketSentiment?.vix_current) && (
                        <div className="model-card">
                            <div className="model-name">VIX INDEX</div>
                            <div className="model-stat-row">
                                <span>Level</span>
                                <span className={vix?.signal === 'bullish' ? 'text-green' : 'text-red'}>
                                    {vix?.current || marketSentiment?.vix_current}
                                </span>
                            </div>
                            <div className="model-stat-row">
                                <span>Signal</span>
                                <span className="mape-badge">
                                    {vix?.signal || marketSentiment?.vix_signal || 'N/A'}
                                </span>
                            </div>
                            <div className="model-stat-row">
                                <span>Status</span>
                                <span style={{ fontSize: '0.7rem' }}>
                                    {vix?.interpretation || marketSentiment?.vix_interpretation || 'N/A'}
                                </span>
                            </div>
                        </div>
                    )}

                    {/* Quantified News Sentiment */}
                    {quantifiedSentiment && (
                        <div className="model-card">
                            <div className="model-name">NEWS SENTIMENT</div>
                            <div className="model-stat-row">
                                <span>Polarity</span>
                                <span className={quantifiedSentiment.avg_polarity > 0 ? 'text-green' : 'text-red'}>
                                    {quantifiedSentiment.avg_polarity?.toFixed(2) || 'N/A'}
                                </span>
                            </div>
                            <div className="model-stat-row">
                                <span>Sentiment</span>
                                <span className="mape-badge">
                                    {quantifiedSentiment.overall_sentiment?.toUpperCase()}
                                </span>
                            </div>
                            <div className="model-stat-row">
                                <span>Bullish Ratio</span>
                                <span className="text-green">{quantifiedSentiment.bullish_ratio}%</span>
                            </div>
                        </div>
                    )}

                    {/* Wavelet Denoising */}
                    {wavelet?.applied && (
                        <div className="model-card">
                            <div className="model-name">WAVELET DENOISING</div>
                            <div className="model-stat-row">
                                <span>Noise Reduction</span>
                                <span className="text-green">{wavelet.noise_reduction_pct}%</span>
                            </div>
                            <div className="model-stat-row">
                                <span>Signal Clarity</span>
                                <span className="mape-badge">{wavelet.signal_clarity}</span>
                            </div>
                            <div className="model-stat-row">
                                <span>Type</span>
                                <span style={{ fontSize: '0.7rem' }}>{wavelet.wavelet_type}</span>
                            </div>
                        </div>
                    )}

                    {/* Data Quality */}
                    {dataQuality?.total_rows && (
                        <div className="model-card">
                            <div className="model-name">DATA QUALITY</div>
                            <div className="model-stat-row">
                                <span>Total Rows</span>
                                <span className="text-green">{dataQuality.total_rows}</span>
                            </div>
                            <div className="model-stat-row">
                                <span>Missing</span>
                                <span className={dataQuality.missing_values > 0 ? 'text-red' : 'text-green'}>
                                    {dataQuality.missing_values} ({dataQuality.missing_pct}%)
                                </span>
                            </div>
                        </div>
                    )}

                    {/* Outlier Analysis */}
                    {outliers?.total_returns && (
                        <div className="model-card">
                            <div className="model-name">OUTLIER ANALYSIS</div>
                            <div className="model-stat-row">
                                <span>Outliers Detected</span>
                                <span className={outliers.outliers_detected > 5 ? 'text-red' : 'text-green'}>
                                    {outliers.outliers_detected} ({outliers.outlier_pct}%)
                                </span>
                            </div>
                            <div className="model-stat-row">
                                <span>Daily Return</span>
                                <span className={outliers.mean_daily_return > 0 ? 'text-green' : 'text-red'}>
                                    {outliers.mean_daily_return}%
                                </span>
                            </div>
                        </div>
                    )}

                    {/* --- NEW: Alternative Data Section --- */}

                    {/* Insider Trading */}
                    {aiData?.alternative_data?.insider_trading && (
                        <div className="model-card">
                            <div className="model-name">INSIDER TRADING</div>
                            <div className="model-stat-row">
                                <span>Signal</span>
                                <span className={aiData.alternative_data.insider_trading.signal === 'Bullish' ? 'text-green' : aiData.alternative_data.insider_trading.signal === 'Bearish' ? 'text-red' : 'text-gray'}>
                                    {aiData.alternative_data.insider_trading.signal?.toUpperCase() || 'N/A'}
                                </span>
                            </div>
                            <div className="model-stat-row">
                                <span>Net Activity</span>
                                <span className="mape-badge" style={{ fontSize: '0.75rem' }}>
                                    {aiData.alternative_data.insider_trading.net_activity || 'N/A'}
                                </span>
                            </div>
                        </div>
                    )}

                    {/* Institutional Holdings */}
                    {aiData?.alternative_data?.institutional_holdings && (
                        <div className="model-card">
                            <div className="model-name">INSTITUTIONAL FLOW</div>
                            <div className="model-stat-row">
                                <span>Signal</span>
                                <span className={aiData.alternative_data.institutional_holdings.signal === 'Bullish' ? 'text-green' : aiData.alternative_data.institutional_holdings.signal === 'Bearish' ? 'text-red' : 'text-gray'}>
                                    {aiData.alternative_data.institutional_holdings.signal?.toUpperCase() || 'N/A'}
                                </span>
                            </div>
                            <div className="model-stat-row">
                                <span>Change</span>
                                <span className="mape-badge">
                                    {aiData.alternative_data.institutional_holdings.change_pct}%
                                </span>
                            </div>
                        </div>
                    )}

                    {/* Options Flow */}
                    {aiData?.alternative_data?.options_flow && (
                        <div className="model-card">
                            <div className="model-name">OPTIONS FLOW</div>
                            <div className="model-stat-row">
                                <span>Put/Call Ratio</span>
                                <span className={aiData.alternative_data.options_flow.put_call_ratio < 0.7 ? 'text-green' : aiData.alternative_data.options_flow.put_call_ratio > 1.0 ? 'text-red' : 'text-gray'}>
                                    {aiData.alternative_data.options_flow.put_call_ratio}
                                </span>
                            </div>
                            <div className="model-stat-row">
                                <span>Sentiment</span>
                                <span className="mape-badge">
                                    {aiData.alternative_data.options_flow.sentiment?.toUpperCase() || 'NEUTRAL'}
                                </span>
                            </div>
                        </div>
                    )}
                </div>
            </div>
        </section>
    );
};

export default DataQualityPanel;
