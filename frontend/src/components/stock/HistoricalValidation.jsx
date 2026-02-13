import { useStock } from '../../context/StockContext';

const HistoricalValidation = () => {
    const { signalsData, loading } = useStock();

    // The enhanced_prediction API returns data at the ROOT level of the response
    const data = signalsData?.enhanced_prediction || {};
    const hasData = Object.keys(data).length > 0 && data.status === 'success';

    // These are at the ROOT level of the enhanced_prediction response
    const backtest = data.backtest || {};
    const validatedCount = data.validated_count;
    const sysAcc = data.system_accuracy;
    const backtestPending = backtest.status === 'pending';

    // Helper for display
    const fmt = (val, suffix = '') => (val !== undefined && val !== null && !isNaN(val)) ? (typeof val === 'number' ? val.toFixed(1) : val) + suffix : '-';
    const fmt2 = (val, suffix = '') => (val !== undefined && val !== null && !isNaN(val)) ? val.toFixed(2) + suffix : '-';

    // Model Consensus Logic - model_metrics is at ROOT level of response
    const metrics = data.model_metrics || {};
    let consensusText = '-';
    let consensusPct = 0;
    const modelCount = Object.keys(metrics).length;
    let majorityDirection = 'Neutral';

    if (modelCount > 0) {
        let bullish = 0, bearish = 0, neutral = 0;
        Object.values(metrics).forEach(m => {
            const d = (m.direction || '').toLowerCase();
            if (d === 'bullish' || d === 'up') bullish++;
            else if (d === 'bearish' || d === 'down') bearish++;
            else neutral++;
        });
        const maxAgree = Math.max(bullish, bearish, neutral);
        consensusText = `${maxAgree}/${modelCount}`;
        consensusPct = (maxAgree / modelCount) * 100;

        if (bullish >= bearish && bullish >= neutral) majorityDirection = 'Bullish';
        else if (bearish >= bullish && bearish >= neutral) majorityDirection = 'Bearish';
        else majorityDirection = 'Neutral';
    }

    // Show meaningful data or indicate no history yet
    const hasHistoryData = (validatedCount !== undefined && validatedCount > 0) ||
        (sysAcc !== undefined && sysAcc > 0);

    // Calculate dynamic bar widths based on actual data
    const validatedBarWidth = hasHistoryData && validatedCount > 0 ? Math.min(100, validatedCount * 10) : 0;
    const winRateBarWidth = backtest.win_rate ? Math.min(100, backtest.win_rate) : 0;
    const sharpeBarWidth = backtest.sharpe_ratio ? Math.min(100, (backtest.sharpe_ratio + 1) * 33) : 0;

    return (
        <section id="historicalValidation" className="history-panel">
            <div className="history-header">
                <h4>PREDICTION TRACK RECORD{!hasData && loading ? ' (LOADING...)' : ''}</h4>
                <div className="accuracy-pill">
                    SYSTEM ACCURACY: <span id="systemAccuracy">
                        {loading && !hasData ? '...' : (hasHistoryData ? fmt(sysAcc) : (backtestPending ? 'PENDING' : 'N/A'))}
                    </span>{hasHistoryData ? '%' : ''}
                </div>
            </div>

            <div className="track-record-grid">
                {/* Validated Predictions */}
                <div className="track-stat">
                    <span className="track-stat-value" id="validatedCount">
                        {loading && !hasData ? '...' : (validatedCount !== undefined ? (validatedCount > 0 ? validatedCount : (backtestPending ? 'PENDING' : '0')) : '-')}
                    </span>
                    <span className="track-stat-label">VALIDATED PREDICTIONS</span>
                    <div className="track-stat-bar">
                        <div className="track-stat-fill" style={{ width: `${validatedBarWidth}%` }}></div>
                    </div>
                </div>

                {/* Win Rate */}
                <div className="track-stat">
                    <span className={`track-stat-value ${backtest.win_rate > 50 ? 'text-green' : ''}`} id="backtestWinRate">
                        {loading && !hasData ? '...' : (backtest.win_rate ? fmt(backtest.win_rate, '%') : (backtestPending ? 'PENDING' : '-'))}
                    </span>
                    <span className="track-stat-label">BACKTEST WIN RATE</span>
                    <div className="track-stat-bar">
                        <div className="track-stat-fill fill-green" style={{ width: `${winRateBarWidth}%` }}></div>
                    </div>
                </div>

                {/* Sharpe Ratio */}
                <div className="track-stat">
                    <span className="track-stat-value" id="strategySharpe">
                        {loading && !hasData ? '...' : (backtest.sharpe_ratio ? fmt2(backtest.sharpe_ratio) : (backtestPending ? 'PENDING' : '-'))}
                    </span>
                    <span className="track-stat-label">SHARPE RATIO</span>
                    <div className="track-stat-bar">
                        <div className="track-stat-fill" style={{ width: `${sharpeBarWidth}%` }}></div>
                    </div>
                </div>

                {/* Model Consensus */}
                <div className="track-stat">
                    <span className={`track-stat-value ${majorityDirection === 'Bullish' ? 'text-green' : majorityDirection === 'Bearish' ? 'text-red' : ''}`} id="modelConsensus">
                        {loading && !hasData ? '...' : consensusText}
                    </span>
                    <span className="track-stat-label">MODELS IN AGREEMENT ({majorityDirection.toUpperCase()})</span>
                    <div className="track-stat-bar">
                        <div className={`track-stat-fill ${majorityDirection === 'Bullish' ? 'fill-green' : majorityDirection === 'Bearish' ? 'fill-red' : ''}`} style={{ width: `${consensusPct}%` }}></div>
                    </div>
                </div>
            </div>

            <p className="history-note">
                {backtestPending
                    ? '* Prediction validation in progress. Metrics will appear after predictions are verified against actual outcomes.'
                    : '* Past performance is not indicative of future results'}
            </p>
        </section>
    );
};

export default HistoricalValidation;
