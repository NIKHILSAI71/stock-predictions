import { useEffect, useRef, useCallback } from 'react';
import { useStock } from '../context/StockContext';
import { api } from '../services/api';

const useStockData = () => {
    const {
        symbol,
        setStockData,
        setAiData,
        setSignalsData,
        setExpertData,
        setLoading,
        setLoadingStatus,
        setLoadingLogs,
        setError,
        lastUpdated,
        setLastUpdated
    } = useStock();

    const abortControllerRef = useRef(null);

    const fetchData = useCallback(async (searchSymbol) => {
        if (!searchSymbol) return;

        // Cancel previous request
        if (abortControllerRef.current) {
            abortControllerRef.current.abort();
        }
        abortControllerRef.current = new AbortController();
        const signal = abortControllerRef.current.signal;

        setLoading(true);
        setError(null);
        setStockData(null); // Clear previous
        setAiData(null); // Reset AI section for manual trigger

        // Helper for safe fetching
        const safeFetch = async (promise, fallback = {}) => {
            try {
                return await promise;
            } catch (e) {
                console.warn('Sub-fetch failed:', e);
                return fallback;
            }
        };

        try {
            // 1. Basic Stock Data (CRITICAL)
            setLoadingStatus('FETCHING STOCK DATA');
            const stockRes = await api.fetchStock(searchSymbol);

            if (stockRes.status !== 'success') {
                throw new Error(stockRes.detail || 'Stock not found');
            }
            setStockData(stockRes.data);

            // 2. Expert Data (Parallel-ish or Sequential types)
            setLoadingStatus('PROCESSING TECHNICAL DATA');

            // Run independent fetches in parallel where possible or safely sequentially
            const tech = await safeFetch(api.fetchTechnical(searchSymbol), { indicators: {} });

            setLoadingStatus('ANALYZING FUNDAMENTALS');
            const fund = await safeFetch(api.fetchFundamental(searchSymbol), { fundamentals: {} });

            setLoadingStatus('CALCULATING RISK METRICS');
            const risk = await safeFetch(api.fetchRisk(searchSymbol), { risk_analysis: {} });

            setLoadingStatus('DETECTING PATTERNS');
            const patterns = await safeFetch(api.fetchPatterns(searchSymbol), { patterns: {} });

            setLoadingStatus('COMPUTING VALUATION');
            const val = await safeFetch(api.fetchValuation(searchSymbol), { valuation: {} });

            setLoadingStatus('FETCHING MARKET SENTIMENT');
            const alt = await safeFetch(api.fetchAlternative(searchSymbol), { alternative_data: {} });

            setExpertData({
                technical: tech.indicators || {},
                fundamental: fund.fundamentals || {},
                risk: risk.risk_analysis || {},
                patterns: patterns.patterns || {},
                valuation: val.valuation || {},
                alternative: alt.alternative_data || {}
            });

            // 3. Signals & Predictions
            setLoadingStatus('GENERATING TRADING SIGNALS');
            const signals = await safeFetch(api.fetchSignals(searchSymbol), {});

            setLoadingStatus('RUNNING ML PREDICTIONS');
            const uniData = await safeFetch(api.fetchUniversalSignals(searchSymbol), {});
            const mlData = (uniData.status === 'success' && uniData.data) ? (uniData.data.ml_prediction || {}) : {};

            setLoadingStatus('RUNNING ENHANCED AI PREDICTIONS');
            const enhancedData = await safeFetch(api.fetchEnhancedPrediction(searchSymbol), {});

            // Merge all signals
            const mergedSignals = {
                ...signals,
                ml_prediction: mlData,
                enhanced_prediction: enhancedData
            };
            setSignalsData(mergedSignals);

            // 4. AI Analysis (STREAMING)
            setLoadingStatus('Initializing AI Analyst...');

            // Wait a moment for UI to settle
            await new Promise(r => setTimeout(r, 100));

            const streamPromise = new Promise((resolve, reject) => {
                // EventSource doesn't support custom headers, so we pass API key as query parameter
                const apiKey = import.meta.env.VITE_API_KEY || '';
                const streamUrl = `/api/ai/analyze-stream/${searchSymbol}${apiKey ? `?api_key=${apiKey}` : ''}`;
                const eventSource = new EventSource(streamUrl);

                eventSource.onopen = () => {
                    console.log("SSE Connection Opened");
                    if (setLoadingLogs) setLoadingLogs(prev => [...prev, "Connected to AI Analysis Stream..."]);
                };

                eventSource.onmessage = (event) => {
                    try {
                        const data = JSON.parse(event.data);

                        if (data.type === 'log') {
                            setLoadingStatus(data.message);
                            if (setLoadingLogs) setLoadingLogs(prev => [...prev, data.message]);
                        } else if (data.type === 'result') {
                            // Set full data object so components can access aiData.ai_analysis, aiData.statistical_arbitrage, etc.
                            setAiData(data.data);
                        } else if (data.type === 'complete') {
                            eventSource.close();
                            resolve();
                        } else if (data.type === 'error') {
                            console.error("Stream Error:", data.message);
                            if (setLoadingLogs) setLoadingLogs(prev => [...prev, `Error: ${data.message}`]);
                            eventSource.close();
                            resolve();
                        }
                    } catch (e) {
                        console.warn("Error parsing SSE:", event.data);
                    }
                };

                eventSource.onerror = (err) => {
                    console.error("SSE Error:", err);
                    eventSource.close();
                    resolve();
                };
            });

            await streamPromise;

            setLastUpdated(Date.now());
            setLoadingStatus('ANALYSIS COMPLETE');

        } catch (e) {
            if (e.name !== 'AbortError') {
                console.error(e);
                setError(e.message);
                setStockData(null);
            }
        } finally {
            if (!signal.aborted) {
                setLoading(false);
            }
        }
    }, [setStockData, setAiData, setSignalsData, setExpertData, setLoading, setLoadingStatus, setError, setLastUpdated]);

    // Trigger fetch when symbol changes
    useEffect(() => {
        if (symbol) {
            window.history.pushState(null, '', `/${symbol}`);
            fetchData(symbol);
        }
    }, [symbol, fetchData]);

    return null; // This hook doesn't return UI types, just manages state
};

export default useStockData;
