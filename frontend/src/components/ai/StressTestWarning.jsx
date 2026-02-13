import PropTypes from 'prop-types';

const StressTestWarning = ({ scenarios, riskAnalysis, unpredictableRisks }) => {
    const getStressWarning = () => {
        // Check 0: Explicit Stress Test Warning (from unpredictable_risks)
        if (scenarios && scenarios.stress_test_warning && scenarios.stress_test_warning.triggered) {
            const st = scenarios.stress_test_warning;
            return {
                pattern: st.pattern_matched || 'UNKNOWN',
                similarity: st.similarity_pct || 0,
                drawdown: st.estimated_drawdown || 'N/A',
                action: st.defensive_action || 'Monitor closely'
            };
        }

        if (!scenarios && !riskAnalysis) return null;

        // Check 1: Bear Case Severity
        if (scenarios?.bear_case) {
            let bearText = "";
            if (typeof scenarios.bear_case === 'string') {
                bearText = scenarios.bear_case.toLowerCase();
            } else if (typeof scenarios.bear_case === 'object') {
                bearText = (scenarios.bear_case.message || scenarios.bear_case.summary || scenarios.bear_case.target || "").toLowerCase();
            }

            if (bearText.includes("crash") || bearText.includes("collapse") || bearText.includes("crisis")) {
                return {
                    pattern: 'EXTREME DOWNSIDE',
                    similarity: null,
                    drawdown: '>30%',
                    action: 'AI models detect extreme downside risk in bear scenario'
                };
            }
        }

        // Check 2: Risk Metrics
        if (riskAnalysis) {
            if (riskAnalysis.max_drawdown && (riskAnalysis.max_drawdown < -0.25 || riskAnalysis.max_drawdown > 25)) {
                return {
                    pattern: 'HIGH VOLATILITY REGIME',
                    similarity: null,
                    drawdown: `${Math.abs(riskAnalysis.max_drawdown).toFixed(1)}%`,
                    action: `Historical drawdown indicates potential for significant loss`
                };
            }
        }

        return null;
    };

    const warning = getStressWarning();
    if (!warning) return null;

    // Risk metrics data
    const eventRisk = unpredictableRisks?.event_risk_score || 0;
    const fragility = unpredictableRisks?.sentiment_fragility?.score || 0;
    const penalty = unpredictableRisks?.confidence_penalty_applied || 0;

    const getScoreColor = (score, type) => {
        if (type === 'risk') return score >= 50 ? '#ef5350' : (score >= 25 ? '#feca57' : '#4caf50');
        if (type === 'fragility') return score >= 70 ? '#ef5350' : (score >= 50 ? '#feca57' : '#4caf50');
        if (type === 'penalty') return penalty < 0 ? '#ef5350' : '#ffffff';
        return '#ffffff';
    };

    return (
        <div className="stress-warning-card">
            {/* Top accent line with glow */}
            <div className="stress-accent-line" />

            {/* Warning header */}
            <div className="stress-header-row">
                <svg className="stress-icon" viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z" />
                    <line x1="12" y1="9" x2="12" y2="13" />
                    <line x1="12" y1="17" x2="12.01" y2="17" />
                </svg>
                <span className="stress-title-label">STRESS TEST WARNING</span>
            </div>

            {/* Pattern match */}
            <div className="stress-pattern-row">
                <span className="stress-pattern-name">{warning.pattern}</span>
                {warning.similarity !== null && (
                    <span className="stress-match-badge">{warning.similarity}% MATCH</span>
                )}
            </div>

            {/* Details row */}
            <div className="stress-detail-row">
                <div className="stress-detail-item">
                    <span className="stress-detail-key">EST. DRAWDOWN</span>
                    <span className="stress-detail-value">{warning.drawdown}</span>
                </div>
                <div className="stress-detail-divider" />
                <div className="stress-detail-item">
                    <span className="stress-detail-key">ACTION</span>
                    <span className="stress-detail-value">{warning.action}</span>
                </div>
            </div>

            {/* Integrated Risk Metrics */}
            {unpredictableRisks && (
                <div className="stress-metrics-row">
                    <div className="stress-metric">
                        <span className="stress-metric-label">EVENT RISK</span>
                        <span className="stress-metric-value" style={{ color: getScoreColor(eventRisk, 'risk') }}>
                            {eventRisk}<span className="stress-metric-unit">/100</span>
                        </span>
                    </div>
                    <div className="stress-metric">
                        <span className="stress-metric-label">FRAGILITY</span>
                        <span className="stress-metric-value" style={{ color: getScoreColor(fragility, 'fragility') }}>
                            {fragility}<span className="stress-metric-unit">/100</span>
                        </span>
                    </div>
                    <div className="stress-metric">
                        <span className="stress-metric-label">CONF. PENALTY</span>
                        <span className="stress-metric-value" style={{ color: getScoreColor(penalty, 'penalty') }}>
                            {penalty}%<span className="stress-metric-unit"> applied</span>
                        </span>
                    </div>
                </div>
            )}
        </div>
    );
};

StressTestWarning.propTypes = {
    scenarios: PropTypes.object,
    riskAnalysis: PropTypes.object,
    unpredictableRisks: PropTypes.object
};

export default StressTestWarning;
