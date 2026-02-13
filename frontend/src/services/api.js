const API_BASE = ''; // Proxy or relative path

// Get API key from environment variable (set this in frontend/.env)
const API_KEY = import.meta.env.VITE_API_KEY || '';

// Helper function to get headers with API key
const getHeaders = () => {
    const headers = {
        'Content-Type': 'application/json',
    };

    // Add API key if available
    if (API_KEY) {
        headers['X-API-Key'] = API_KEY;
    }

    return headers;
};

// Helper function for better error handling
const fetchWithErrorHandling = async (url, options = {}) => {
    try {
        // Merge default headers with any provided headers
        const fetchOptions = {
            ...options,
            headers: {
                ...getHeaders(),
                ...(options.headers || {}),
            },
        };

        const res = await fetch(url, fetchOptions);

        // Handle authentication errors
        if (res.status === 401) {
            throw new Error('Authentication required. Please set VITE_API_KEY in frontend/.env');
        } else if (res.status === 403) {
            throw new Error('Invalid API key. Please check your configuration.');
        } else if (res.status === 429) {
            throw new Error('Rate limit exceeded. Please wait a moment and try again.');
        }

        if (!res.ok) {
            if (res.status === 500) {
                throw new Error('Server error. Please try again later.');
            } else if (res.status === 404) {
                throw new Error('Symbol not found. Please check the ticker.');
            } else {
                throw new Error(`HTTP ${res.status}: ${res.statusText}`);
            }
        }
        return res.json();
    } catch (error) {
        if (error.name === 'TypeError' && error.message.includes('fetch')) {
            throw new Error('Unable to connect to server. Please ensure the backend is running on port 8000.');
        }
        throw error;
    }
};

export const api = {
    async fetchStock(symbol) {
        return fetchWithErrorHandling(`${API_BASE}/api/stock/${symbol}`);
    },

    async fetchTechnical(symbol) {
        return fetchWithErrorHandling(`${API_BASE}/api/technical/${symbol}`);
    },

    async fetchFundamental(symbol) {
        return fetchWithErrorHandling(`${API_BASE}/api/fundamental/${symbol}`);
    },

    async fetchRisk(symbol) {
        return fetchWithErrorHandling(`${API_BASE}/api/quantitative/risk/${symbol}`);
    },

    async fetchPatterns(symbol) {
        return fetchWithErrorHandling(`${API_BASE}/api/patterns/${symbol}`);
    },

    async fetchValuation(symbol) {
        return fetchWithErrorHandling(`${API_BASE}/api/valuation/${symbol}`);
    },

    async fetchAlternative(symbol) {
        return fetchWithErrorHandling(`${API_BASE}/api/alternative/${symbol}`);
    },

    async fetchSignals(symbol) {
        return fetchWithErrorHandling(`${API_BASE}/api/signals/${symbol}`);
    },

    async fetchUniversalSignals(symbol) {
        return fetchWithErrorHandling(`${API_BASE}/api/universal-signals/${symbol}`);
    },

    async fetchEnhancedPrediction(symbol) {
        return fetchWithErrorHandling(`${API_BASE}/api/enhanced-prediction/${symbol}`);
    },

    async fetchAIAnalysis(symbol, signal) {
        return fetchWithErrorHandling(`${API_BASE}/api/ai/analyze/${symbol}`, { signal });
    },

    async fetchHistory(symbol, period = 'max') {
        return fetchWithErrorHandling(`${API_BASE}/api/stock/history/${symbol}?period=${period}`);
    }
};
