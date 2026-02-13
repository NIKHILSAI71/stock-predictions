import PropTypes from 'prop-types';

const RiskMetrics = ({ riskData, unpredictableRisks }) => {
    // Data from ai.unpredictable_risks
    const eventRisk = unpredictableRisks?.event_risk_score || 0;
    const fragility = unpredictableRisks?.sentiment_fragility?.score || 0;
    const penalty = unpredictableRisks?.confidence_penalty_applied || 0;

    // Helper for color classes - using correct CSS class names
    const getScoreColor = (score, type) => {
        if (type === 'risk') return score >= 50 ? 'text-red' : (score >= 25 ? 'text-yellow' : 'text-green');
        if (type === 'fragility') return score >= 70 ? 'text-red' : (score >= 50 ? 'text-yellow' : 'text-green');
        if (type === 'penalty') return score < 0 ? 'text-red' : '';
        return '';
    };

    if (!riskData && !unpredictableRisks) return null;

    return (
        <div id="riskMetricsSection" className="risk-metrics-section">
            <div className="risk-metrics-row">
                <div className="risk-metric-item">
                    <span className="risk-metric-label">EVENT RISK</span>
                    <div className="risk-metric-value-row">
                        <span className={`risk-metric-value ${getScoreColor(eventRisk, 'risk')}`} id="eventRiskValue">{eventRisk}</span>
                        <span className="risk-metric-max">/100</span>
                    </div>
                </div>
                <div className="risk-metric-item">
                    <span className="risk-metric-label">FRAGILITY</span>
                    <div className="risk-metric-value-row">
                        <span className={`risk-metric-value ${getScoreColor(fragility, 'fragility')}`} id="fragilityValue">{fragility}</span>
                        <span className="risk-metric-max">/100</span>
                    </div>
                </div>
                <div className="risk-metric-item">
                    <span className="risk-metric-label">CONF. PENALTY</span>
                    <div className="risk-metric-value-row">
                        <span className={`risk-metric-value ${getScoreColor(penalty, 'penalty')}`} id="confPenaltyValue">{penalty}%</span>
                        <span className="risk-metric-max">applied</span>
                    </div>
                </div>
            </div>
        </div>
    );
};

RiskMetrics.propTypes = {
    riskData: PropTypes.object,
    unpredictableRisks: PropTypes.object
};

export default RiskMetrics;

