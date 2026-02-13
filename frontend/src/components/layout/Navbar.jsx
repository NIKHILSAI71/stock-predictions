import { useState, useEffect } from 'react';
import { useStock } from '../../context/StockContext';

const Navbar = () => {
    const { market, setMarket, symbol, setSymbol, loading, error } = useStock();
    const [dropdownOpen, setDropdownOpen] = useState(false);
    const [searchInput, setSearchInput] = useState('');
    const [timeLeft, setTimeLeft] = useState(60);

    const toggleDropdown = (e) => {
        e.stopPropagation();
        setDropdownOpen(!dropdownOpen);
    };

    const selectMarket = (value, e) => {
        e.stopPropagation();
        setMarket(value);
        setDropdownOpen(false);
    };

    const handleSearch = () => {
        if (!searchInput.trim()) return;
        setSymbol(searchInput.trim().toUpperCase());
    };

    const handleKeyPress = (e) => {
        if (e.key === 'Enter') handleSearch();
    };

    // Sync input with symbol from context (if updated elsewhere)
    useEffect(() => {
        if (symbol) setSearchInput(symbol);
    }, [symbol]);

    // Countdown Timer Logic
    useEffect(() => {
        let interval = null;
        if (symbol && !loading && !error) {
            setTimeLeft(60); // Reset on new symbol or load complete
            interval = setInterval(() => {
                setTimeLeft((prev) => {
                    if (prev <= 1) {
                        // In a real app, we might trigger a refresh here via Context
                        // setRefreshTrigger(t => t + 1);
                        return 60;
                    }
                    return prev - 1;
                });
            }, 1000);
        }
        return () => clearInterval(interval);
    }, [symbol, loading, error]);

    const getStatusText = () => {
        if (loading) return `ANALYZING ${symbol || 'MARKET'}`;
        if (error) return 'CONNECTION ERROR';
        if (symbol) return `UPDATING IN ${timeLeft}s`;
        return 'SYSTEM ONLINE';
    };

    const getStatusColor = () => {
        if (loading) return '#fff';
        if (error) return '#ef5350';
        if (symbol) return '#feca57';
        return '#00e676'; // System Online Green
    };

    return (
        <nav className="navbar">
            <div className="logo">
                <div className="logo-mark">A</div>
                <span className="logo-text">ANTIGRAVITY</span>
            </div>
            <div className="search-bar">
                <div className={`market-dropdown ${dropdownOpen ? 'open' : ''}`} id="marketDropdown">
                    <button className="market-dropdown-btn" id="marketDropdownBtn" onClick={toggleDropdown}>
                        <span id="marketSelected">{market}</span>
                        <svg className="dropdown-arrow" width="8" height="5" viewBox="0 0 8 5" fill="none">
                            <path d="M1 1L4 4L7 1" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"
                                strokeLinejoin="round" />
                        </svg>
                    </button>
                    <div className="market-dropdown-menu" id="marketDropdownMenu">
                        {['US', 'NS', 'BO', 'L'].map((m) => (
                            <div
                                key={m}
                                className={`market-option ${market === m ? 'active' : ''}`}
                                data-value={m}
                                onClick={(e) => selectMarket(m, e)}
                            >
                                {m === 'NS' ? 'NSE' : m === 'BO' ? 'BSE' : m === 'L' ? 'LSE' : m}
                            </div>
                        ))}
                    </div>
                </div>
                <div className="search-divider"></div>
                <input
                    type="text"
                    id="stockSearch"
                    placeholder="SEARCH TICKER"
                    autoComplete="off"
                    value={searchInput}
                    onChange={(e) => setSearchInput(e.target.value)}
                    onKeyPress={handleKeyPress}
                />
                <button id="analyzeBtn" className="btn-icon" onClick={handleSearch}>
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <path d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"></path>
                    </svg>
                </button>
            </div>
            <div className="nav-status">
                <span className="status-dot" style={{ backgroundColor: getStatusColor() }}></span>
                <span id="systemStatus" style={{ color: getStatusColor() }}>{getStatusText()}</span>
            </div>
        </nav>
    );
};

export default Navbar;
