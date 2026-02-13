import { useStock } from '../../context/StockContext';

const PriceTargets = () => {
    const { signalsData } = useStock();

    if (!signalsData?.enhanced_prediction?.price_targets) return <section id="priceTargets" className="targets-panel hidden"></section>;

    const targets = signalsData.enhanced_prediction.price_targets;
    const periods = [
        { key: '7d', label: '7-DAY', dataKey: 'day_7' },
        { key: '30d', label: '30-DAY', dataKey: 'day_30' },
        { key: '90d', label: '90-DAY', dataKey: 'day_90' }
    ];

    return (
        <section id="priceTargets" className="targets-panel">
            <h4>PRICE FORECASTS & CONFIDENCE INTERVALS</h4>
            <div className="targets-container">
                {periods.map(({ key, label, dataKey }) => {
                    const data = targets[dataKey];
                    if (!data) return null;

                    const price = data.price;
                    const low = data.confidence_interval ? data.confidence_interval[0] : (price * 0.95);
                    const high = data.confidence_interval ? data.confidence_interval[1] : (price * 1.05);
                    const direction = data.direction || 'Neutral';
                    const dirClass = direction === 'Bullish' ? 'text-green' : direction === 'Bearish' ? 'text-red' : 'text-yellow';

                    return (
                        <div key={key} className="target-card">
                            <div className="target-period">{label}</div>
                            <div className="target-price" id={`target${key}`}>${price.toFixed(2)}</div>
                            <div className="target-range">
                                <span className="range-label">Range:</span>
                                <span id={`range${key}`}>${low.toFixed(0)} - ${high.toFixed(0)}</span>
                            </div>
                            <div className={`target-direction ${dirClass}`} id={`direction${key}`}>{direction}</div>
                        </div>
                    );
                })}
            </div>
        </section>
    );
};

export default PriceTargets;
