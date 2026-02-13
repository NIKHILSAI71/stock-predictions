import PropTypes from 'prop-types';

const Scenarios = ({ scenarios }) => {
    if (!scenarios) return null;

    const cases = [
        { key: 'bull_case', title: 'BULL CASE', color: 'text-green' },
        { key: 'base_case', title: 'BASE CASE', color: 'text-muted' },
        { key: 'bear_case', title: 'BEAR CASE', color: 'text-red' }
    ];

    // Clean text helper (reused logic)
    const cleanText = (text) => {
        if (!text) return text;
        let cleaned = text.replace(/(\*\*|__)(.*?)\1/g, '$2').replace(/(\*|_)(.*?)\1/g, '$2');
        cleaned = cleaned.replace(/^[\s-•]+/, '');
        const colonMatch = cleaned.match(/^(.{3,60}?):\s*/);
        if (colonMatch) {
            cleaned = cleaned.substring(colonMatch[0].length).trim();
        }
        if (cleaned.length > 0) {
            cleaned = cleaned.charAt(0).toUpperCase() + cleaned.slice(1);
        }
        return cleaned;
    };

    const hasData = cases.some(c => scenarios[c.key]);

    if (!hasData) return null;

    return (
        <div className="ai-scenarios-grid" style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
            gap: '1rem',
            marginBottom: '2rem',
            borderBottom: '1px solid var(--border-light)',
            paddingBottom: '1rem'
        }}>
            {cases.map(c => {
                const data = scenarios[c.key];
                if (!data) return null;

                return (
                    <div key={c.key} className="scenario-card" style={{
                        background: '#111',
                        padding: '1rem',
                        borderRadius: '4px',
                        border: '1px solid #222'
                    }}>
                        <div style={{ fontSize: '0.7rem', color: '#666', fontFamily: 'var(--font-mono)', marginBottom: '0.5rem' }}>
                            {c.title} <span style={{ float: 'right' }}>{data.probability || ''}</span>
                        </div>
                        <div className={c.color} style={{ fontSize: '1.2rem', fontWeight: 'bold', marginBottom: '0.5rem' }}>
                            {data.target || '-'}
                        </div>
                        <div style={{ fontSize: '0.8rem', lineHeight: '1.4', color: '#aaa' }}>
                            {cleanText(data.catalyst || data.driver || data.thesis || data.risk || '')}
                        </div>
                    </div>
                );
            })}
        </div>
    );
};

Scenarios.propTypes = {
    scenarios: PropTypes.shape({
        bull_case: PropTypes.object,
        base_case: PropTypes.object,
        bear_case: PropTypes.object
    })
};

export default Scenarios;
