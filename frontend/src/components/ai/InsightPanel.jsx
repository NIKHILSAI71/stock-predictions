import { useStock } from '../../context/StockContext';
import NewsSources from './NewsSources';
import StressTestWarning from './StressTestWarning';
import RiskMetrics from './RiskMetrics';
import WhatCouldGoWrong from './WhatCouldGoWrong';
import Scenarios from './Scenarios';

const InsightPanel = () => {
    const { aiData, loading, expertData } = useStock();

    // Safety check for loading state
    if (loading) return (
        <section id="aiSection" className="ai-insight-panel">
            <div className="panel-header">
                <h3>GEMINI INTELLIGENCE</h3>
                <div className="ai-header-controls">
                    <span id="aiConfidenceBadge" className="confidence-badge hidden">CONFIDENCE: <span id="aiConfidence">-</span>%</span>
                    <button id="getActionBtn" className="get-action-btn">GET ACTION</button>
                </div>
            </div>
            <div id="aiSkeleton" className="ai-skeleton">
                <div className="skeleton-line title-line"></div>
                <div className="skeleton-line text-line"></div>
                <div className="skeleton-line text-line"></div>
                <div className="skeleton-line text-line short"></div>
                <div className="skeleton-grid">
                    <div className="skeleton-block"></div>
                    <div className="skeleton-block"></div>
                </div>
            </div>
        </section>
    );

    if (!aiData) return (
        <section id="aiSection" className="ai-insight-panel hidden">
            <div className="panel-header">
                <h3>GEMINI INTELLIGENCE</h3>
                <div className="ai-header-controls">
                    <span id="aiConfidenceBadge" className="confidence-badge hidden">CONFIDENCE: <span id="aiConfidence">-</span>%</span>
                    <button id="getActionBtn" className="get-action-btn">GET ACTION</button>
                </div>
            </div>
        </section>
    );

    const analysis = aiData.ai_analysis || {};
    const summary = analysis.agent_summary || analysis.summary || "Analysis unavailable.";
    const rec = analysis.recommendation || {};
    const drivers = analysis.key_drivers || [];

    // Normalize risks: check top-level risks, then risk_assessment.risk_factors
    const risks = analysis.risks || (analysis.risk_assessment?.risk_factors) || [];

    // Merge unpredictable risks into scenarios for StressTestWarning
    if (analysis.unpredictable_risks) {
        if (!analysis.scenarios) analysis.scenarios = {};
        analysis.scenarios.stress_test_warning = analysis.unpredictable_risks.stress_test_warning;
    }

    const signal = rec.signal || "HOLD";
    const confidence = analysis.confidence_score || 0;

    // Risk data for metrics section
    const riskData = expertData?.risk?.risk_analysis || {};

    return (
        <section id="aiSection" className="ai-insight-panel">
            <div className="panel-header">
                <h3>GEMINI INTELLIGENCE</h3>
                <div className="ai-header-controls">
                    <span id="aiConfidenceBadge" className="confidence-badge">CONFIDENCE: <span id="aiConfidence">{confidence}</span>%</span>
                    <button id="getActionBtn" className="get-action-btn">GET ACTION</button>
                </div>
            </div>

            <div id="aiContent">
                {/* 1. Top: Executive Summary */}
                <div className="ai-main-summary">
                    <h4>EXECUTIVE SUMMARY</h4>
                    {/* Render summary with clickable citations */}
                    <div id="aiSummary" className="typing-effect summary-text">
                        {(() => {
                            if (!summary) return null;

                            // Get the news context for looking up citations
                            const newsContext = aiData.alternative_data?.social_sentiment?.news_context ||
                                aiData.web_intelligence_summary?.key_news_items ||
                                analysis.news_context || [];

                            // Regex to find (1), (2), etc.
                            const parts = summary.split(/(\(\d+\))/g);

                            return parts.map((part, index) => {
                                const match = part.match(/\((\d+)\)/);
                                if (match) {
                                    const citationIndex = parseInt(match[1]) - 1; // Convert 1-based to 0-based
                                    const source = newsContext[citationIndex];

                                    if (source) {
                                        return (
                                            <span key={index} className="citation-wrapper">
                                                <a
                                                    href={source.link || source.url || '#'}
                                                    target="_blank"
                                                    rel="noopener noreferrer"
                                                    className="citation-link"
                                                >
                                                    {part}
                                                </a>
                                                <div className="citation-tooltip">
                                                    <div className="tooltip-title">{source.title || source.headline}</div>
                                                    <div className="tooltip-source">{source.source} - {source.date || 'Recent'}</div>
                                                    {source.link && <div className="tooltip-click">Click to read source</div>}
                                                </div>
                                            </span>
                                        );
                                    }
                                }
                                return part;
                            });
                        })()}
                    </div>

                    <style jsx="true">{`
                        .citation-wrapper {
                            position: relative;
                            display: inline-block;
                            margin: 0 2px;
                        }
                        
                        .citation-link {
                            color: #4facfe;
                            cursor: pointer;
                            font-weight: bold;
                            text-decoration: none;
                            border-bottom: 1px dotted #4facfe;
                        }
                        
                        .citation-link:hover {
                            color: #00f2fe;
                            border-bottom: 1px solid #00f2fe;
                        }

                        .citation-tooltip {
                            visibility: hidden;
                            width: 250px;
                            background-color: rgba(10, 15, 30, 0.95);
                            color: #eee;
                            text-align: left;
                            border-radius: 6px;
                            padding: 10px;
                            position: absolute;
                            z-index: 1000;
                            bottom: 125%; /* Position above reference */
                            left: 50%;
                            transform: translateX(-50%);
                            opacity: 0;
                            transition: opacity 0.3s, bottom 0.3s;
                            box-shadow: 0 4px 15px rgba(0,0,0,0.5);
                            border: 1px solid rgba(79, 172, 254, 0.3);
                            font-size: 0.85rem;
                            pointer-events: none;
                            line-height: 1.4;
                        }

                        .citation-wrapper:hover .citation-tooltip {
                            visibility: visible;
                            opacity: 1;
                            bottom: 135%;
                        }

                        .citation-tooltip::after {
                            content: "";
                            position: absolute;
                            top: 100%;
                            left: 50%;
                            margin-left: -5px;
                            border-width: 5px;
                            border-style: solid;
                            border-color: rgba(10, 15, 30, 0.95) transparent transparent transparent;
                        }

                        .tooltip-title {
                            font-weight: 600;
                            color: #fff;
                            margin-bottom: 4px;
                            display: -webkit-box;
                            -webkit-line-clamp: 2;
                            -webkit-box-orient: vertical;
                            overflow: hidden;
                        }

                        .tooltip-source {
                            color: #888;
                            font-size: 0.75rem;
                        }
                        
                        .tooltip-click {
                            color: #4facfe;
                            font-size: 0.7rem;
                            margin-top: 4px;
                            font-style: italic;
                        }
                    `}</style>

                    {/* News Sources (at bottom right) */}
                    <NewsSources context={aiData.alternative_data?.social_sentiment?.news_context || aiData.web_intelligence_summary?.key_news_items || analysis.news_context} />
                </div>

                {/* 2. Scenarios (Bull/Base/Bear) */}
                <Scenarios scenarios={analysis.scenarios} />

                {/* 3. Middle: Drivers & Risks (Side-by-Side) */}
                <div className="ai-details-grid">
                    <div className="detail-block">
                        <h5>KEY DRIVERS</h5>
                        <ul id="aiFactors" className="drivers-list">
                            {drivers.map((d, i) => <li key={i}>{typeof d === 'string' ? d : (d.description || d.title)}</li>)}
                        </ul>
                    </div>
                    <div className="detail-block">
                        <h5>RISK FACTORS</h5>
                        <ul id="aiRisks" className="risks-list">
                            {risks.map((r, i) => {
                                const cleanRiskText = (text) => {
                                    if (!text) return '';
                                    const str = String(text);
                                    const colonIndex = str.indexOf(':');
                                    return colonIndex !== -1 ? str.substring(colonIndex + 1).trim() : str;
                                };

                                return (
                                    <li key={i}>
                                        {typeof r === 'object' && r.factor ? (
                                            <>
                                                <span style={{
                                                    fontSize: '0.7rem', padding: '2px 6px', borderRadius: '4px', marginRight: '8px',
                                                    fontFamily: 'var(--font-mono)',
                                                    backgroundColor: r.impact?.toLowerCase().includes('high') ? 'rgba(239, 83, 80, 0.2)' : 'rgba(255, 255, 255, 0.1)',
                                                    color: r.impact?.toLowerCase().includes('high') ? '#ef5350' : '#ccc'
                                                }}>
                                                    {r.impact || 'RISK'}
                                                </span>
                                                {cleanRiskText(r.factor)}
                                            </>
                                        ) : cleanRiskText(typeof r === 'string' ? r : r.description)}
                                    </li>
                                );
                            })}
                        </ul>
                    </div>
                </div>

                {/* 3. Stress Test & Risk Metrics (New Sections) */}
                <StressTestWarning scenarios={analysis.scenarios} riskAnalysis={riskData} unpredictableRisks={analysis.unpredictable_risks} />
                {/* Show standalone RiskMetrics only when no stress warning is active */}
                {!(analysis.scenarios?.stress_test_warning?.triggered ||
                    analysis.unpredictable_risks?.stress_test_warning?.triggered) && (
                        <RiskMetrics riskData={riskData} unpredictableRisks={analysis.unpredictable_risks} />
                    )}
                <WhatCouldGoWrong risks={analysis.unpredictable_risks?.what_could_go_wrong || risks} />

                {/* 4. Bottom: Action/Recommendation */}
                <div className="ai-action-row">
                    <div className="recommendation-box" id="aiRecommendationBox">
                        <span className="rec-label">
                            ACTION{rec.timeframe ? ` (${rec.timeframe.toUpperCase()})` : ''}
                        </span>
                        <span className="rec-value" id="aiRecommendation">{signal}</span>
                        {rec.action && (
                            <span className="rec-action">{rec.action}</span>
                        )}
                    </div>
                </div>
            </div>
        </section>
    );
};

export default InsightPanel;
