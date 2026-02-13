import { createContext, useContext, useState, useEffect } from 'react';
import PropTypes from 'prop-types';

const StockContext = createContext();

export const useStock = () => {
    return useContext(StockContext);
};

export const StockProvider = ({ children }) => {
    // Initialize from URL if present (simple router)
    const initialSymbol = window.location.pathname.substring(1).split('/')[0] || '';
    const [symbol, setSymbol] = useState(initialSymbol);

    // Sync URL with symbol
    useEffect(() => {
        if (symbol) {
            window.history.pushState({}, '', `/${symbol}`);
        }
    }, [symbol]);

    // Handle browser back/forward
    useEffect(() => {
        const handlePopState = () => {
            const pathSymbol = window.location.pathname.substring(1).split('/')[0];
            setSymbol(pathSymbol || '');
        };
        window.addEventListener('popstate', handlePopState);
        return () => window.removeEventListener('popstate', handlePopState);
    }, []);
    const [market, setMarket] = useState('US');
    const [stockData, setStockData] = useState(null);
    const [aiData, setAiData] = useState(null);
    const [signalsData, setSignalsData] = useState(null); // Includes ML & Enhanced
    const [expertData, setExpertData] = useState({
        technical: null,
        fundamental: null,
        risk: null,
        patterns: null,
        valuation: null,
        alternative: null
    });

    const [loading, setLoading] = useState(false);
    const [loadingStatus, setLoadingStatus] = useState('');
    const [loadingLogs, setLoadingLogs] = useState([]);
    const [error, setError] = useState(null);
    const [lastUpdated, setLastUpdated] = useState(null);

    // Reset data when symbol changes (optional, or handle in search)
    const resetData = () => {
        setStockData(null);
        setAiData(null);
        setSignalsData(null);
        setExpertData({
            technical: null,
            fundamental: null,
            risk: null,
            patterns: null,
            valuation: null,
            alternative: null
        });
        setLoadingLogs([]);
        setError(null);
    };

    const value = {
        symbol, setSymbol,
        market, setMarket,
        stockData, setStockData,
        aiData, setAiData,
        signalsData, setSignalsData,
        expertData, setExpertData,
        loading, setLoading,
        loadingStatus, setLoadingStatus,
        loadingLogs, setLoadingLogs,
        error, setError,
        lastUpdated, setLastUpdated,
        resetData
    };

    return (
        <StockContext.Provider value={value}>
            {children}
        </StockContext.Provider>
    );
};

StockProvider.propTypes = {
    children: PropTypes.node.isRequired
};
