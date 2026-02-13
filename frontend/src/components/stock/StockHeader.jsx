import { useStock } from '../../context/StockContext';
import PriceChart from './PriceChart';

const StockHeader = () => {
    const { stockData } = useStock();

    // Safeguard
    if (!stockData) return null;

    const { symbol, name, price_info } = stockData;
    const { current_price, previous_close } = price_info;
    const change = current_price - previous_close;
    const pct = (change / previous_close) * 100;
    const isUp = change >= 0;

    return (
        <header id="stockHeader" className="stock-header">
            {/* Chart as Background */}
            <PriceChart />

            {/* Overlay: Stock Name and Price */}
            <div className="stock-overlay">
                <div className="stock-info">
                    <h1 id="stockSymbol" className="display-ticker">{symbol}</h1>
                    <h2 id="stockName" className="display-name">{name}</h2>
                </div>
                <div className="price-display">
                    <div className="current-price" id="currentPrice">${current_price.toFixed(2)}</div>
                    <div className={`price-change ${isUp ? 'text-green' : 'text-red'}`} id="priceChange">
                        {isUp ? '↑' : '↓'} {isUp ? '+' : ''}{change.toFixed(2)} ({pct.toFixed(2)}%)
                    </div>
                </div>
            </div>
        </header>
    );
};

export default StockHeader;
