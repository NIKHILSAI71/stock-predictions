
/**
 * Antigravity Finance - AI Market Intelligence
 */

const API_BASE = '';
let currentSymbol = '';
let currentPeriod = '6mo';
let activeIndicators = {
    sma: true,
    bb: false,
    volume: true
};
let updateInterval = null;

// Prediction Cache Configuration
const PREDICTION_CACHE_KEY = 'antigravity_predictions';
const PREDICTION_CACHE_DURATION = 60 * 60 * 1000; // 1 hour in milliseconds

// Cache helper functions
function getCachedPredictions(symbol) {
    try {
        const cache = JSON.parse(localStorage.getItem(PREDICTION_CACHE_KEY) || '{}');
        const data = cache[symbol.toUpperCase()];
        if (data && (Date.now() - data.timestamp) < PREDICTION_CACHE_DURATION) {
            console.log(`Using cached predictions for ${symbol}`);
            return data;
        }
    } catch (e) {
        console.warn('Error reading prediction cache:', e);
    }
    return null;
}

function cachePredictions(symbol, aiData, enhancedData, signalsData) {
    try {
        const cache = JSON.parse(localStorage.getItem(PREDICTION_CACHE_KEY) || '{}');
        cache[symbol.toUpperCase()] = {
            timestamp: Date.now(),
            ai_analysis: aiData,
            enhanced_prediction: enhancedData,
            signals: signalsData
        };
        localStorage.setItem(PREDICTION_CACHE_KEY, JSON.stringify(cache));
        console.log(`Cached predictions for ${symbol}`);
    } catch (e) {
        console.warn('Error caching predictions:', e);
    }
}

// DOM Elements
const elements = {
    input: document.getElementById('stockSearch'),
    marketDropdown: document.getElementById('marketDropdown'),
    marketDropdownBtn: document.getElementById('marketDropdownBtn'),
    marketDropdownMenu: document.getElementById('marketDropdownMenu'),
    marketSelected: document.getElementById('marketSelected'),
    btn: document.getElementById('analyzeBtn'),
    loader: document.getElementById('loadingOverlay'),

    // Sections
    header: document.getElementById('stockHeader'),
    aiSection: document.getElementById('aiSection'),
    dataGrid: document.getElementById('dataGrid'),

    // AI Elements
    aiSummary: document.getElementById('aiSummary'),
    aiRec: document.getElementById('aiRecommendation'),
    aiConfidence: document.getElementById('aiConfidence'),
    aiFactors: document.getElementById('aiFactors'),
    aiRisks: document.getElementById('aiRisks'),

    // Status - separate elements for loader overlay vs header status
    status: document.getElementById('loaderText'),  // Loading overlay text
    headerStatus: document.getElementById('systemStatus')  // Header status bar (for timer)
};

// Store selected market value
let selectedMarket = 'US';

document.addEventListener('DOMContentLoaded', () => {
    initEvents();

    // Check for symbol in URL on load
    const path = window.location.pathname;
    // Ensure we don't trigger on /error, /static, or root
    if (path && path.length > 1 && path !== '/' && !path.startsWith('/error') && !path.startsWith('/static')) {
        const symbol = path.substring(1); // Remove leading slash
        // Populate search box and trigger search
        elements.input.value = symbol;
        handleSearch(false);
    }
});

function initEvents() {
    elements.btn.addEventListener('click', () => handleSearch(false));
    elements.input.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') handleSearch(false);
    });

    // Custom Market Dropdown Logic
    if (elements.marketDropdownBtn) {
        elements.marketDropdownBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            elements.marketDropdown.classList.toggle('open');
        });

        // Handle option clicks
        document.querySelectorAll('.market-option').forEach(option => {
            option.addEventListener('click', (e) => {
                e.stopPropagation();
                const value = option.dataset.value;
                selectedMarket = value;
                elements.marketSelected.textContent = option.textContent;

                // Update active state
                document.querySelectorAll('.market-option').forEach(o => o.classList.remove('active'));
                option.classList.add('active');

                // Close dropdown
                elements.marketDropdown.classList.remove('open');
            });
        });

        // Close dropdown when clicking outside
        document.addEventListener('click', () => {
            elements.marketDropdown.classList.remove('open');
        });
    }

    // Time selectors
    document.querySelectorAll('.time-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            document.querySelectorAll('.time-btn').forEach(b => b.classList.remove('active'));
            e.target.classList.add('active');
            currentPeriod = e.target.dataset.period;
            if (currentSymbol) loadChart(currentSymbol);
        });
    });

    // Indicator buttons removed with chart controls redesign
}

async function handleSearch(isBackground = false) {
    if (countdownInterval) clearInterval(countdownInterval);
    let symbol = elements.input.value.trim().toUpperCase();
    if (!symbol) return;

    // Handle Market Selection using stored value
    const market = selectedMarket || 'US';

    if (market === 'NS' && !symbol.endsWith('.NS')) {
        symbol += '.NS';
    } else if (market === 'BO' && !symbol.endsWith('.BO')) {
        symbol += '.BO';
    } else if (market === 'L' && !symbol.endsWith('.L')) {
        symbol += '.L';
    } else if (market === 'TO' && !symbol.endsWith('.TO')) {
        symbol += '.TO';
    }

    // Ensure we don't double suffix if user typed it + selected market

    if (currentSymbol !== symbol) {
        // New search, clear interval
        if (updateInterval) clearInterval(updateInterval);
        // Set new interval for live updates (60s)
        updateInterval = setInterval(() => handleSearch(true), 60000);
    }

    currentSymbol = symbol;

    // Update browser URL without reloading
    if (!isBackground && window.location.pathname !== `/${symbol}`) {
        window.history.pushState({ symbol }, `${symbol} | Antigravity Finance`, `/${symbol}`);
    }

    // Only show loader and update status for foreground (user-initiated) searches
    // Background auto-refresh should be silent
    if (!isBackground) {
        elements.loader.classList.remove('hidden');
        elements.status.textContent = `ANALYZING ${symbol}`;
    } else {
        elements.headerStatus.textContent = `UPDATING ${symbol}`;
    }

    try {
        // Helper to update status
        const updateStatus = (msg) => {
            elements.status.textContent = msg;
        };

        // 1. Basic Stock Data
        updateStatus('FETCHING STOCK DATA');
        const stockRes = await fetch(`${API_BASE}/api/stock/${symbol}`);
        const stockData = await stockRes.json();

        if (stockData.status !== 'success') {
            // Redirect to error page
            const errorMsg = stockData.detail || 'Stock not found';
            window.location.href = `/error?message=${encodeURIComponent(errorMsg)}&code=404`;
            return;
        }

        updateHeader(stockData.data);

        // 2. Sequential Analysis Data with Status Updates
        updateStatus('PROCESSING TECHNICAL DATA');
        const techRes = await fetch(`${API_BASE}/api/technical/${symbol}`);
        const techData = await techRes.json();

        updateStatus('ANALYZING FUNDAMENTALS');
        const fundRes = await fetch(`${API_BASE}/api/fundamental/${symbol}`);
        const fundData = await fundRes.json();

        updateStatus('CALCULATING RISK METRICS');
        const riskRes = await fetch(`${API_BASE}/api/quantitative/risk/${symbol}`);
        const riskData = await riskRes.json();

        updateStatus('DETECTING PATTERNS');
        const patternRes = await fetch(`${API_BASE}/api/patterns/${symbol}`);
        const patternData = await patternRes.json();

        updateStatus('COMPUTING VALUATION');
        const valRes = await fetch(`${API_BASE}/api/valuation/${symbol}`);
        const valData = await valRes.json();

        updateStatus('FETCHING MARKET SENTIMENT');
        const altRes = await fetch(`${API_BASE}/api/alternative/${symbol}`);
        const altData = await altRes.json();

        updateStatus('GENERATING TRADING SIGNALS');
        const signalsRes = await fetch(`${API_BASE}/api/signals/${symbol}`);
        const signalsData = await signalsRes.json();

        // NEW: Fetch ML predictions from universal signals
        updateStatus('RUNNING ML PREDICTIONS');
        let mlData = {};
        try {
            const uniRes = await fetch(`${API_BASE}/api/universal-signals/${symbol}`);
            const uniData = await uniRes.json();
            if (uniData.status === 'success' && uniData.data) {
                mlData = uniData.data.ml_prediction || {};
            }
        } catch (mlError) {
            console.warn('ML predictions unavailable:', mlError);
        }

        // Merge ML predictions into signalsData
        signalsData.ml_prediction = mlData;

        // NEW: Fetch Enhanced AI Predictions (LSTM, Ensemble, Alt Data, Sector)
        updateStatus('RUNNING ENHANCED AI PREDICTIONS');
        let enhancedData = {};
        try {
            const enhancedRes = await fetch(`${API_BASE}/api/enhanced-prediction/${symbol}`);
            enhancedData = await enhancedRes.json();
        } catch (enhError) {
            console.warn('Enhanced predictions unavailable:', enhError);
        }

        // Pass enhanced data to the grid update
        signalsData.enhanced_prediction = enhancedData;

        updateDataGrid(
            techData.indicators || {},
            fundData.fundamentals || {},
            riskData.risk_analysis || {},
            patternData.patterns || {},
            valData.valuation || {},
            altData.alternative_data || {},
            signalsData  // Pass entire signalsData object (now includes ml_prediction and enhanced_prediction)
        );

        // Update Transparency, Targets, and History
        updateTransparencySection(enhancedData);

        // Only fetch AI on initial load (not background refresh) to avoid rate limits
        if (!isBackground) {
            updateStatus('AI ANALYZING NEWS & DATA');

            // Helper function to fetch with timeout and retry
            const fetchWithRetry = async (url, retries = 2, timeoutMs = 60000) => {
                for (let attempt = 0; attempt <= retries; attempt++) {
                    try {
                        const controller = new AbortController();
                        const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

                        const response = await fetch(url, { signal: controller.signal });
                        clearTimeout(timeoutId);

                        if (!response.ok) {
                            throw new Error(`HTTP ${response.status}`);
                        }
                        return await response.json();
                    } catch (err) {
                        console.warn(`AI fetch attempt ${attempt + 1} failed:`, err.message);
                        if (attempt === retries) throw err;
                        // Wait 1 second before retrying
                        await new Promise(resolve => setTimeout(resolve, 1000));
                        updateStatus(`RETRYING AI ANALYSIS (${attempt + 2}/${retries + 1})`);
                    }
                }
            };

            try {
                const aiData = await fetchWithRetry(`${API_BASE}/api/ai/analyze/${symbol}`);

                updateStatus('PROCESSING AI INSIGHTS');

                if (aiData && aiData.ai_analysis) {
                    updateAISection(aiData.ai_analysis);
                    if (aiData.macro_context) updateMacroContext(aiData.macro_context);

                    // Update Sentiment/Traffic widgets with the Real Data fetched by AI
                    if (aiData.alternative_data) {
                        const alt = aiData.alternative_data;
                        const set = (id, val, suffix = '', isGood = null) => {
                            const el = document.getElementById(id);
                            if (!el) return;
                            el.textContent = val !== undefined && val !== null ? val + suffix : '-';
                            if (isGood === true) el.className = "metric-value text-green";
                            else if (val && val.toString().includes("Loading")) el.className = "metric-value text-yellow";
                            else el.className = "metric-value";
                        };

                        set('socialSentiment', alt.social_sentiment?.sentiment_score?.toFixed(2), '', alt.social_sentiment?.sentiment_score > 0);

                        if (alt.web_traffic?.value) {
                            set('webTraffic', alt.web_traffic.value);
                        } else {
                            set('webTraffic', '-');
                        }

                        // NEWS SOURCES STACK - Use real fetched sources from news_context
                        const newsStackEl = document.getElementById('newsSourcesStack');
                        const newsExpandedEl = document.getElementById('newsSourcesExpanded');
                        const newsContext = alt.social_sentiment?.news_context || [];

                        if (newsStackEl && newsContext.length > 0) {
                            newsStackEl.innerHTML = '';
                            if (newsExpandedEl) newsExpandedEl.innerHTML = '';
                            if (newsExpandedEl) newsExpandedEl.classList.add('hidden');

                            // Add SOURCES label (restored for red markup style)
                            const label = document.createElement('span');
                            label.className = 'news-sources-label';
                            label.textContent = 'SOURCES:';
                            newsStackEl.appendChild(label);

                            const container = document.createElement('div');
                            container.className = 'news-sources-container';

                            const maxVisible = 4; // Increased to 4 matching the drawing
                            const visibleSources = newsContext.slice(0, maxVisible);
                            const overflowCount = newsContext.length - maxVisible;

                            // Extract domain from URL
                            const getDomainFromUrl = (url) => {
                                if (!url) return null;
                                try {
                                    return new URL(url).hostname;
                                } catch (e) {
                                    return null;
                                }
                            };

                            visibleSources.forEach((source, index) => {
                                const icon = document.createElement('div');
                                icon.className = 'source-icon' + (source.is_trusted ? ' is-trusted' : '');
                                icon.style.zIndex = maxVisible - index;
                                icon.setAttribute('data-source', source.source || 'Unknown');
                                icon.title = source.title || source.source || 'News Source';

                                const domain = getDomainFromUrl(source.link);
                                if (domain) {
                                    const img = document.createElement('img');
                                    img.src = `https://www.google.com/s2/favicons?domain=${domain}&sz=32`;
                                    img.alt = source.source || '';
                                    img.onerror = () => {
                                        icon.textContent = (source.source || 'UN').substring(0, 2).toUpperCase();
                                        img.remove();
                                    };
                                    icon.appendChild(img);
                                } else {
                                    icon.textContent = (source.source || 'UN').substring(0, 2).toUpperCase();
                                }

                                container.appendChild(icon);
                            });

                            // +N button with click to expand
                            if (overflowCount > 0) {
                                const moreIcon = document.createElement('div');
                                moreIcon.className = 'source-icon source-icon-more';
                                moreIcon.style.zIndex = 0;
                                moreIcon.textContent = `+${overflowCount}`;
                                moreIcon.setAttribute('data-source', `Click to see all ${newsContext.length} sources`);

                                // Click handler to expand sources
                                moreIcon.addEventListener('click', () => {
                                    if (newsExpandedEl) {
                                        newsExpandedEl.classList.toggle('hidden');
                                    }
                                });

                                container.appendChild(moreIcon);
                            }

                            newsStackEl.appendChild(container);

                            // Populate expanded sources section
                            if (newsExpandedEl && newsContext.length > 0) {
                                const header = document.createElement('div');
                                header.className = 'news-sources-expanded-header';
                                header.innerHTML = `<span>ALL SOURCES (${newsContext.length})</span><span class="news-sources-close">×</span>`;
                                header.querySelector('.news-sources-close').addEventListener('click', () => {
                                    newsExpandedEl.classList.add('hidden');
                                });
                                newsExpandedEl.appendChild(header);

                                newsContext.forEach(source => {
                                    const item = document.createElement('div');
                                    item.className = 'news-source-item';

                                    const domain = getDomainFromUrl(source.link);
                                    const faviconUrl = domain ? `https://www.google.com/s2/favicons?domain=${domain}&sz=32` : '';

                                    item.innerHTML = `
                                        ${faviconUrl ? `<img class="source-favicon" src="${faviconUrl}" alt="">` : ''}
                                        <div>
                                            <a href="${source.link || '#'}" target="_blank" rel="noopener">${source.title || 'Untitled'}</a>
                                            <div class="source-name">${source.source || 'Unknown Source'}</div>
                                        </div>
                                    `;
                                    newsExpandedEl.appendChild(item);
                                });
                            }
                        }
                    }

                    // Pass Statistical Arbitrage data to the grid
                    if (aiData.statistical_arbitrage) {
                        signalsData.statistical_arbitrage = aiData.statistical_arbitrage;
                        // Re-call updateDataGrid with the new stat arb data
                        updateDataGrid(
                            techData.indicators || {},
                            fundData.fundamentals || {},
                            riskData.risk_analysis || {},
                            patternData.patterns || {},
                            valData.valuation || {},
                            altData.alternative_data || {},
                            signalsData
                        );
                    }
                } else {
                    // AI returned empty data
                    updateAISection({
                        summary: "AI Analysis temporarily unavailable. Please refresh to try again.",
                        recommendation: "HOLD",
                        key_drivers: ["Data could not be loaded"],
                        risks: ["Unable to assess risks at this time"]
                    });
                }
            } catch (aiError) {
                console.error("AI Error after retries:", aiError);
                // Always update UI on failure so it doesn't stay stuck
                updateAISection({
                    summary: "AI Analysis failed to load. This may be due to network issues or API rate limits. Please try again in a moment.",
                    recommendation: "HOLD",
                    key_drivers: ["Analysis temporarily unavailable"],
                    risks: ["Please refresh to retry"]
                });
            }

            // Cache the predictions for this symbol (to restore on refresh)
            cachePredictions(symbol,
                document.getElementById('aiSummary')?.textContent || '',
                enhancedData,
                signalsData
            );

            updateStatus('ANALYSIS COMPLETE');
        } else {
            // Background refresh - only restore cached predictions if available
            const cached = getCachedPredictions(symbol);
            if (cached && cached.enhanced_prediction) {
                updateTransparencySection(cached.enhanced_prediction);
                signalsData.enhanced_prediction = cached.enhanced_prediction;
                signalsData.statistical_arbitrage = cached.signals?.statistical_arbitrage;
            }
        }

        // Show Sections FIRST (so chart container has proper dimensions)
        elements.header.classList.remove('hidden');
        elements.aiSection.classList.remove('hidden');
        elements.dataGrid.classList.remove('hidden');

        // 3. Load Chart (after container is visible)
        await loadChart(symbol);

        // Force Plotly to resize to container after render
        setTimeout(() => {
            const chartEl = document.getElementById('priceChart');
            if (chartEl && window.Plotly) {
                Plotly.Plots.resize(chartEl);
            }
        }, 100);

        // Reset and start countdown timer
        startUpdateTimer(60);

    } catch (e) {
        console.error(e);
        if (!isBackground) {
            // Redirect to error page
            window.location.href = `/error?message=${encodeURIComponent(e.message || 'Connection error')}&code=500`;
        }
        elements.headerStatus.textContent = 'CONNECTION ERROR';
        elements.headerStatus.style.color = '#ef5350';
    } finally {
        if (!isBackground) elements.loader.classList.add('hidden');
    }
}

let countdownInterval = null;

function startUpdateTimer(seconds) {
    if (countdownInterval) clearInterval(countdownInterval);

    let timeLeft = seconds;

    const updateDisplay = () => {
        // Update header status bar, not loader text - no stock symbol in countdown
        elements.headerStatus.textContent = `UPDATING IN ${timeLeft}s`;
        elements.headerStatus.style.color = '#feca57'; // Warning yellow
    };

    updateDisplay(); // Initial show

    countdownInterval = setInterval(() => {
        timeLeft--;
        if (timeLeft > 0) {
            updateDisplay();
        } else {
            clearInterval(countdownInterval);
            // Only show stock symbol when actually updating
            elements.headerStatus.textContent = `UPDATING ${currentSymbol}`;
        }
    }, 1000);
}

function updateHeader(data) {
    document.getElementById('stockSymbol').textContent = data.symbol;
    document.getElementById('stockName').textContent = data.name;

    const price = data.price_info.current_price;
    document.getElementById('currentPrice').textContent = `$${price.toFixed(2)}`;

    const change = data.price_info.current_price - data.price_info.previous_close;
    const pct = (change / data.price_info.previous_close) * 100;

    const changeEl = document.getElementById('priceChange');
    const arrow = change >= 0 ? '↑' : '↓';
    changeEl.textContent = `${arrow} ${change >= 0 ? '+' : ''}${change.toFixed(2)} (${pct.toFixed(2)}%)`;

    // Red/Green logic
    changeEl.className = 'price-change ' + (change >= 0 ? 'text-green' : 'text-red');
}

function updateAISection(ai) {
    if (!ai) return;

    // Helper to clean formatting (remove markdown and Title: prefixes)
    const cleanText = (text) => {
        if (!text) return text;
        // 1. Remove markdown bold/italic (**text** to text)
        let cleaned = text.replace(/(\*\*|__)(.*?)\1/g, '$2').replace(/(\*|_)(.*?)\1/g, '$2');

        // 2. Remove leading bullet points/hyphens
        cleaned = cleaned.replace(/^[\s-•]+/, '');

        // 3. Remove "Title:" prefix if present (heuristic: colon in first 50 chars)
        // Only if it looks like a title (short enough)
        const colonMatch = cleaned.match(/^(.{3,60}?):\s*/);
        if (colonMatch) {
            cleaned = cleaned.substring(colonMatch[0].length).trim();
        }

        // 4. Ensure it starts with uppercase
        if (cleaned.length > 0) {
            cleaned = cleaned.charAt(0).toUpperCase() + cleaned.slice(1);
        }
        return cleaned;
    };

    // 1. SUMMARY
    let summaryText = ai.agent_summary || ai.summary || "Analysis unavailable.";
    elements.aiSummary.textContent = cleanText(summaryText);

    // [NEW] Error Handling: Check for failure keywords
    const isError = /failed|error|unable to/i.test(summaryText);

    // Elements to toggle
    const actionRow = document.querySelector('.ai-action-row');
    const detailsGrid = document.querySelector('.ai-details-grid');
    const newsStack = document.getElementById('newsSourcesStack');

    if (isError) {
        // Hide details if error
        if (actionRow) actionRow.style.display = 'none';
        if (detailsGrid) detailsGrid.style.display = 'none';
        if (newsStack) newsStack.style.display = 'none';

        // Optional: Style the summary to indicate error
        elements.aiSummary.style.color = '#ef5350'; // Red tint
        return; // Stop further processing
    } else {
        // Restore visibility if success
        if (actionRow) actionRow.style.display = 'flex'; // or block/grid depending on css
        if (detailsGrid) detailsGrid.style.display = 'grid'; // .ai-details-grid is grid
        if (newsStack) newsStack.style.display = 'grid'; // we changed this to grid
        elements.aiSummary.style.color = ''; // Reset color
    }

    // 2. RECOMMENDATION & TIMEFRAME
    let rec = "HOLD";
    let timeframe = "";
    let action = "";

    if (typeof ai.recommendation === 'object') {
        rec = ai.recommendation.signal;
        timeframe = ai.recommendation.timeframe || "";
        action = ai.recommendation.action || "";
    } else {
        rec = ai.recommendation || "HOLD";
    }

    if (typeof rec !== 'string') rec = "HOLD";

    elements.aiRec.textContent = rec;
    document.getElementById('aiRecommendationBox').className = "recommendation-box";
    elements.aiRec.className = "rec-value";

    elements.aiConfidence.textContent = ai.confidence_score || "0";

    // Update Label to show Timeframe if available
    const recLabel = document.querySelector('.rec-label');
    if (recLabel) {
        if (timeframe) {
            recLabel.textContent = `ACTION (${timeframe})`;
            // Make it slightly smaller if text is long
            if (timeframe.length > 15) recLabel.style.fontSize = "0.55rem";
        } else {
            recLabel.textContent = "ACTION";
        }
    }

    // Insert Action Text below signal if exists
    let actionEl = document.getElementById('recActionText');
    if (!actionEl && action) {
        actionEl = document.createElement('div');
        actionEl.id = 'recActionText';
        actionEl.style.fontSize = '0.7rem';
        actionEl.style.color = '#888';
        actionEl.style.marginTop = '0.5rem';
        actionEl.style.fontFamily = 'var(--font-mono)';
        elements.aiRec.parentNode.appendChild(actionEl);
    }
    if (actionEl) actionEl.textContent = action;

    // NEWS SOURCES GRID - Display refined 2-column grid of ALL sources
    const newsStackEl = document.getElementById('newsSourcesStack');

    // We are reusing the 'newsSourcesStack' ID in HTML but changing its styling class via JS or CSS override
    if (newsStackEl) {
        newsStackEl.innerHTML = '';
        newsStackEl.className = 'news-sources-grid'; // Force new class

        let sources = [];
        if (ai.web_intelligence_summary && ai.web_intelligence_summary.key_news_items) {
            sources = ai.web_intelligence_summary.key_news_items.map(item => ({
                name: item.source || 'Unknown',
                headline: item.headline || '',
                link: item.link || '',
                isTrusted: true
            }));
        } else if (ai.news_context) {
            // Fallback to older field if web_intelligence_summary not present
            sources = ai.news_context;
        }

        if (sources.length > 0) {
            // Extract domain from URL
            const getDomainFromUrl = (url) => {
                if (!url) return null;
                try {
                    const urlObj = new URL(url);
                    return urlObj.hostname;
                } catch (e) {
                    return null;
                }
            };

            // Render ALL sources - no slicing/stacking
            sources.forEach(source => {
                const item = document.createElement('a');
                item.href = source.link || '#';
                item.target = '_blank';
                item.rel = 'noopener';
                item.className = 'news-source-item';

                const domain = getDomainFromUrl(source.link);
                const faviconUrl = domain ? `https://www.google.com/s2/favicons?domain=${domain}&sz=64` : null;

                let iconHTML = '';
                if (faviconUrl) {
                    iconHTML = `<img class="source-favicon" src="${faviconUrl}" alt="${source.name}" onerror="this.style.display='none'">`;
                }

                item.innerHTML = `
                    ${iconHTML}
                    <div class="source-content">
                        <div class="source-title" title="${source.headline || source.title}">${source.headline || source.title || 'Market News'}</div>
                        <div class="source-name">${(source.name || source.source || 'Source').toUpperCase()}</div>
                    </div>
                `;

                newsStackEl.appendChild(item);
            });
        } else {
            // Optional: Show message if no sources
            const noSources = document.createElement('div');
            noSources.style.gridColumn = "1 / -1";
            noSources.style.color = "#666";
            noSources.style.fontSize = "0.8rem";
            noSources.textContent = "No specific sources cited directly.";
            newsStackEl.appendChild(noSources);
        }
    }

    // 3. SCENARIOS (Bull/Base/Bear) - Insert before Details Grid
    let scenariosEl = document.getElementById('aiScenarios');
    if (!scenariosEl && ai.scenarios) {
        scenariosEl = document.createElement('div');
        scenariosEl.id = 'aiScenarios';
        scenariosEl.className = 'ai-scenarios-grid';

        // Styles for scenarios (inline for now, or add to CSS later)
        scenariosEl.style.display = 'grid';
        scenariosEl.style.gridTemplateColumns = 'repeat(auto-fit, minmax(200px, 1fr))';
        scenariosEl.style.gap = '1rem';
        scenariosEl.style.marginBottom = '2rem';
        scenariosEl.style.borderBottom = '1px solid var(--border-light)';
        scenariosEl.style.paddingBottom = '1rem';

        // Insert after summary, before details grid
        elements.aiSummary.parentNode.parentNode.insertBefore(scenariosEl, elements.aiSummary.parentNode.nextSibling);
    }

    if (scenariosEl && ai.scenarios) {
        scenariosEl.innerHTML = ''; // Clear
        const cases = [
            { key: 'bull_case', title: 'BULL CASE', color: 'text-green' },
            { key: 'base_case', title: 'BASE CASE', color: 'text-muted' },
            { key: 'bear_case', title: 'BEAR CASE', color: 'text-red' }
        ];

        cases.forEach(c => {
            if (ai.scenarios[c.key]) {
                const data = ai.scenarios[c.key];
                const card = document.createElement('div');
                card.className = 'scenario-card';
                card.style.background = '#111';
                card.style.padding = '1rem';
                card.style.borderRadius = '4px';
                card.style.border = '1px solid #222';

                card.innerHTML = `
                    <div style="font-size: 0.7rem; color: #666; font-family: var(--font-mono); margin-bottom: 0.5rem;">${c.title} <span style="float:right">${data.probability || ''}</span></div>
                    <div style="font-size: 1.2rem; font-weight: bold; margin-bottom: 0.5rem;" class="${c.color}">${data.target || '-'}</div>
                    <div style="font-size: 0.8rem; line-height: 1.4; color: #aaa;">${cleanText(data.catalyst || data.driver || data.thesis || data.risk || '')}</div>
                `;
                scenariosEl.appendChild(card);
            }
        });
    }


    // 4. KEY DRIVERS
    elements.aiFactors.innerHTML = '';
    const createLi = (text, badge = null) => {
        const li = document.createElement('li');
        if (badge) {
            const span = document.createElement('span');
            span.textContent = badge.text;
            span.style.fontSize = '0.7rem';
            span.style.padding = '2px 6px';
            span.style.borderRadius = '4px';
            span.style.marginRight = '8px';
            span.style.fontFamily = 'var(--font-mono)';
            span.style.backgroundColor = badge.color === 'red' ? 'rgba(239, 83, 80, 0.2)' :
                (badge.color === 'yellow' ? 'rgba(254, 202, 87, 0.2)' : 'rgba(255, 255, 255, 0.1)');
            span.style.color = badge.color === 'red' ? '#ef5350' :
                (badge.color === 'yellow' ? '#feca57' : '#ccc');
            li.appendChild(span);
            // Clean text for drivers/risks
            li.appendChild(document.createTextNode(cleanText(text)));
        } else {
            // Clean text
            li.textContent = cleanText(text);
        }
        return li;
    };

    if (ai.key_drivers && Array.isArray(ai.key_drivers)) {
        ai.key_drivers.forEach(d => {
            const text = typeof d === 'string' ? d : (d.description || d.title || JSON.stringify(d));
            elements.aiFactors.appendChild(createLi(text));
        });
    }

    // 5. RISK FACTORS (Enhanced with Badges)
    elements.aiRisks.innerHTML = '';
    if (ai.risk_assessment) {
        const risks = ai.risk_assessment.risk_factors || ai.risks;
        if (risks && Array.isArray(risks)) {
            risks.forEach(r => {
                if (typeof r === 'object' && r.factor) {
                    // New Format
                    const impactColor = (r.impact || '').toLowerCase().includes('high') ? 'red' :
                        ((r.impact || '').toLowerCase().includes('medium') ? 'yellow' : 'grey');
                    elements.aiRisks.appendChild(createLi(r.factor, {
                        text: (r.impact || 'RISK').toUpperCase(),
                        color: impactColor
                    }));
                } else {
                    // Old Format (String)
                    const text = typeof r === 'string' ? r : (r.description || r.title || JSON.stringify(r));
                    elements.aiRisks.appendChild(createLi(text));
                }
            });
        }

        // Max Drawdown Warning if High
        if (ai.risk_assessment.max_drawdown_risk && ai.risk_assessment.max_drawdown_risk.toUpperCase() === 'HIGH') {
            elements.aiRisks.appendChild(createLi("HIGH DRAWDOWN RISK DETECTED", { text: 'WARNING', color: 'red' }));
        }
    }

    // 6. UNPREDICTABLE RISKS & STRESS TEST (NEW)
    const unpredictableRisks = ai.unpredictable_risks;
    if (unpredictableRisks) {
        // STRESS TEST WARNING - Update existing HTML elements
        const stressTest = unpredictableRisks.stress_test_warning;
        const stressWarning = document.getElementById('stressTestWarning');
        const stressPattern = document.getElementById('stressPattern');
        const stressDetails = document.getElementById('stressDetails');

        if (stressTest && stressTest.triggered && stressWarning) {
            stressPattern.textContent = `Pattern: ${stressTest.pattern_matched || 'UNKNOWN'} (${stressTest.similarity_pct || 0}% match)`;
            stressDetails.textContent = `Est. Drawdown: ${stressTest.estimated_drawdown || 'N/A'} | Action: ${stressTest.defensive_action || 'Monitor'}`;
            stressWarning.classList.remove('hidden');
        } else if (stressWarning) {
            stressWarning.classList.add('hidden');
        }

        // RISK METRICS - Update existing HTML elements
        const riskMetricsSection = document.getElementById('riskMetricsSection');
        const eventRiskValue = document.getElementById('eventRiskValue');
        const fragilityValue = document.getElementById('fragilityValue');
        const confPenaltyValue = document.getElementById('confPenaltyValue');

        const fragility = unpredictableRisks.sentiment_fragility || {};
        const eventScore = unpredictableRisks.event_risk_score || 0;
        const penalty = unpredictableRisks.confidence_penalty_applied || 0;

        if (riskMetricsSection) {
            // Update values
            eventRiskValue.textContent = eventScore;
            fragilityValue.textContent = fragility.score || 0;
            confPenaltyValue.textContent = `${penalty}%`;

            // Apply color classes
            eventRiskValue.className = 'risk-metric-value' + (eventScore >= 50 ? ' text-danger' : eventScore >= 25 ? ' text-warning' : '');
            fragilityValue.className = 'risk-metric-value' + ((fragility.score || 0) >= 70 ? ' text-danger' : (fragility.score || 0) >= 50 ? ' text-warning' : '');
            confPenaltyValue.className = 'risk-metric-value' + (penalty < 0 ? ' text-danger' : '');

            riskMetricsSection.classList.remove('hidden');
        }

        // WHAT COULD GO WRONG - Update existing HTML elements
        const wcgwSection = document.getElementById('wcgwSection');
        const wcgwList = document.getElementById('wcgwList');
        const wcgw = unpredictableRisks.what_could_go_wrong;

        if (wcgwSection && wcgwList) {
            wcgwList.innerHTML = '';
            if (wcgw && wcgw.length > 0) {
                wcgw.slice(0, 3).forEach((item, index) => {
                    const li = document.createElement('li');
                    li.setAttribute('data-index', index + 1);
                    li.textContent = item;
                    wcgwList.appendChild(li);
                });
                wcgwSection.classList.remove('hidden');
            } else {
                wcgwSection.classList.add('hidden');
            }
        }
    }
}

function updateDataGrid(tech, fund, risk, patterns, val, alt, signalsData) {
    // Helper to set text and color
    const set = (id, val, suffix = '', isGood = null) => {
        const el = document.getElementById(id);
        if (!el) return;
        el.textContent = val !== undefined && val !== null ? val + suffix : '-';
        if (isGood === true) el.className = "metric-value text-green";
        else if (isGood === false) el.className = "metric-value text-red";
        else el.className = "metric-value";
    };

    // Technical
    set('rsiValue', tech.rsi);
    // MACD Display with Signal and Trending Color
    const macdVal = tech.macd;
    const macdSig = tech.macd_signal;
    const macdHist = tech.macd_histogram;

    const macdEl = document.getElementById('macdValue');
    if (macdEl) {
        if (macdVal !== undefined && macdSig !== undefined) {
            macdEl.textContent = `${macdVal.toFixed(2)} / ${macdSig.toFixed(2)}`;
            // Color based on Histogram (Bullish/Bearish)
            if (macdHist > 0) macdEl.className = "metric-value text-green";
            else if (macdHist < 0) macdEl.className = "metric-value text-red";
            else macdEl.className = "metric-value";
        } else {
            macdEl.textContent = '-';
            macdEl.className = "metric-value";
        }
    }
    set('adxValue', tech.adx);
    set('atrValue', tech.atr);

    // Fundamental
    set('peRatio', fund.pe_ratio?.toFixed(2));
    set('fwdPeRatio', fund.forward_pe ? fund.forward_pe.toFixed(2) : '-');
    set('marketCap', formatLarge(fund.market_cap));
    set('divYield', fund.dividend_yield, '%');
    set('netMargin', fund.net_margin, '%', fund.net_margin > 0);
    set('operMargin', fund.operating_margin, '%', fund.operating_margin > 0);
    set('roe', fund.roe, '%', fund.roe > 0);

    // Signals (New Enhanced Display)
    const setSignal = (id, val) => {
        const el = document.getElementById(id);
        if (!el) return;
        el.textContent = val || '-';
        const v = (val || "").toUpperCase();
        if (v.includes("BUY") || v.includes("BULLISH") || v.includes("UNDERVALUED") || v.includes("FAIR") || v.includes("ENTRY OK") || v.includes("(OK)") || v.includes("(LOW)") || v.includes("REASONABLE") || v.includes("STRONG") || v.includes("CONFIRMED")) {
            el.className = "signal-tag text-green";
        } else if (v.includes("SELL") || v.includes("BEARISH") || v.includes("OVERVALUED") || v.includes("OVERBOUGHT") || v.includes("(HIGH)") || v.includes("(ELEVATED)") || v.includes("VERY_LOW") || v.includes("VERY LOW")) {
            el.className = "signal-tag text-red";
        } else if (v.includes("WAIT") || v.includes("CAUTION") || v.includes("APPROACH") || v.includes("WEAKENING") || v.includes("SLIGHTLY") || v.includes("NEUTRAL") || v.includes("WEAK") || v.includes("NORMAL")) {
            el.className = "signal-tag text-yellow";
        } else {
            el.className = "signal-tag";
        }
    };

    // Extract signals from signalsData (handle both old and new format)
    const signals = signalsData.signals || signalsData;
    const analysis = signalsData.analysis || signals.analysis;
    const volumeConf = signalsData.volume_confirmation || {};
    const pivotLevels = signalsData.intermediate_levels || {};
    const valSnapshot = signalsData.valuation_snapshot || {};

    // Main signal from new signals endpoint
    if (signals) {
        setSignal('mainSignal', signals.signal);
        set('signalStrength', signals.signal_strength, '%', signals.signal_strength > 55);

        // Risk management
        if (signals.risk_management) {
            set('stopLoss', '$' + signals.risk_management.stop_loss_price, ` (${signals.risk_management.stop_loss_pct}%)`, false);
            set('positionSize', signals.risk_management.position_size_pct, '%');
        }

        // Analysis items from signals endpoint (PE vs Industry, Overvaluation, MACD Divergence, Pullback)
        if (analysis) {
            setSignal('peVsIndustry', analysis.pe_vs_industry);
            setSignal('overvaluationPct', analysis.overvaluation_status);
            setSignal('macdDivergence', analysis.macd_divergence);
            set('pullbackTarget', analysis.pullback_target);
        }

        // RSI and PEG status from confidence adjustments
        if (signals.confidence_adjustments) {
            setSignal('rsiStatus', signals.confidence_adjustments.rsi_status?.replace(/_/g, ' '));
            setSignal('pegStatus', signals.confidence_adjustments.peg_status?.replace(/_/g, ' '));
            setSignal('volumeConviction', signals.confidence_adjustments.volume_conviction);
        }
    }

    // NEW: Volume Confirmation Section
    if (volumeConf) {
        set('volumeRatio', volumeConf.volume_ratio?.toFixed(2), 'x', volumeConf.volume_ratio > 1);
        setSignal('obvTrend', volumeConf.obv_trend);
        set('mfiValue', volumeConf.mfi);
        setSignal('mfiSignal', volumeConf.mfi_signal);
    }

    // NEW: ML Predictions Section (from universal signals)
    const mlPrediction = signalsData.ml_prediction || {};
    if (mlPrediction) {
        // ML Direction with color coding
        setSignal('mlDirection', mlPrediction.ensemble_direction);

        // ML Confidence
        const mlConf = mlPrediction.ensemble_confidence;
        if (mlConf !== undefined) {
            set('mlConfidence', mlConf.toFixed(1), '%', mlConf > 70);
        }

        // Individual model predictions
        setSignal('rfPrediction', mlPrediction.rf_prediction);
        setSignal('svmPrediction', mlPrediction.svm_prediction);
        setSignal('momentumPrediction', mlPrediction.momentum_prediction);

        // Models agreement
        const modelsAgreeEl = document.getElementById('modelsAgree');
        if (modelsAgreeEl) {
            if (mlPrediction.models_agree === true) {
                modelsAgreeEl.textContent = 'YES';
                modelsAgreeEl.className = 'signal-tag text-green';
            } else if (mlPrediction.models_agree === false) {
                modelsAgreeEl.textContent = 'NO';
                modelsAgreeEl.className = 'signal-tag text-yellow';
            } else {
                modelsAgreeEl.textContent = '-';
                modelsAgreeEl.className = 'signal-tag';
            }
        }

        // Top feature (first from feature_importance)
        const featureImportance = mlPrediction.feature_importance || {};
        const features = Object.entries(featureImportance);
        if (features.length > 0) {
            const [topName, topVal] = features[0];
            set('topFeature', topName.replace(/_/g, ' ').toUpperCase());
        } else {
            set('topFeature', '-');
        }

        // Cluster profile
        set('clusterProfile', mlPrediction.cluster_profile || '-');
    }

    // NEW: Pivot Levels Section
    if (pivotLevels) {
        set('pivotPoint', '$' + pivotLevels.pivot_point);
        set('resistanceR1', '$' + pivotLevels.resistance_r1);
        set('resistanceR2', '$' + pivotLevels.resistance_r2);
        set('supportS1', '$' + pivotLevels.support_s1);
        set('supportS2', '$' + pivotLevels.support_s2);
    }

    // NEW: Fibonacci Levels Section
    const fibLevels = signalsData.fibonacci_levels || {};
    if (fibLevels) {
        const currentPrice = fibLevels.current_price;

        // Trend indicator
        const trend = (fibLevels.trend || '').toUpperCase();
        setSignal('fibTrend', trend);

        // Swing points
        set('fibSwingHigh', '$' + fibLevels.swing_high?.toFixed(2));
        set('fibSwingLow', '$' + fibLevels.swing_low?.toFixed(2));

        // Fibonacci levels with color coding based on current price
        const setFibLevel = (id, level) => {
            const el = document.getElementById(id);
            if (!el || level === undefined) return;
            el.textContent = '$' + level.toFixed(2);
            // Color based on whether it's support (below price) or resistance (above price)
            if (currentPrice && level < currentPrice) {
                el.className = "metric-value text-green"; // Support below
            } else if (currentPrice && level > currentPrice) {
                el.className = "metric-value text-red"; // Resistance above
            } else {
                el.className = "metric-value text-yellow"; // At level
            }
        };

        setFibLevel('fib236', fibLevels.fib_236);
        setFibLevel('fib382', fibLevels.fib_382);
        setFibLevel('fib500', fibLevels.fib_500);
        setFibLevel('fib618', fibLevels.fib_618);
    }

    // PEG ratio from valuation snapshot if available
    if (valSnapshot.peg_ratio) {
        set('pegRatio', valSnapshot.peg_ratio?.toFixed(2));
    }

    // NEW: Staged Entry Section
    const stagedEntry = signalsData.staged_entry || {};
    const marketRegimeData = signalsData.market_regime || {};

    if (stagedEntry) {
        // Position size with color coding
        const positionPct = stagedEntry.recommended_position_pct;
        const positionEl = document.getElementById('stagedPositionPct');
        if (positionEl && positionPct !== undefined) {
            positionEl.textContent = `${positionPct}%`;
            if (positionPct >= 80) {
                positionEl.className = "signal-tag text-green";  // Full position
            } else if (positionPct >= 50) {
                positionEl.className = "signal-tag text-yellow"; // Partial position
            } else {
                positionEl.className = "signal-tag text-red";    // Small position
            }
        }

        // Market Regime
        const regimeEl = document.getElementById('marketRegime');
        if (regimeEl && marketRegimeData.current_regime) {
            const regime = marketRegimeData.current_regime;
            regimeEl.textContent = regime;
            if (regime.toLowerCase().includes('bull') && regime.toLowerCase().includes('low')) {
                regimeEl.className = "signal-tag text-green";
            } else if (regime.toLowerCase().includes('bull')) {
                regimeEl.className = "signal-tag text-green";
            } else if (regime.toLowerCase().includes('bear')) {
                regimeEl.className = "signal-tag text-red";
            } else {
                regimeEl.className = "signal-tag text-yellow";
            }
        }

        // Regime Override
        const overrideEl = document.getElementById('regimeOverride');
        if (overrideEl) {
            if (stagedEntry.regime_override_applied) {
                overrideEl.textContent = "APPLIED";
                overrideEl.className = "signal-tag text-green";
            } else {
                overrideEl.textContent = "NO";
                overrideEl.className = "signal-tag";
            }
        }

        // Ideal Entry Price
        set('idealEntry', '$' + stagedEntry.ideal_entry_price);

        // Price vs Ideal
        const priceVsIdeal = stagedEntry.price_vs_ideal_pct;
        const priceVsIdealEl = document.getElementById('priceVsIdeal');
        if (priceVsIdealEl && priceVsIdeal !== undefined) {
            const symbol = priceVsIdeal > 0 ? '+' : '';
            priceVsIdealEl.textContent = `${symbol}${priceVsIdeal.toFixed(1)}%`;
            if (priceVsIdeal <= 0) {
                priceVsIdealEl.className = "metric-value text-green";  // At or below ideal
            } else if (priceVsIdeal <= 10) {
                priceVsIdealEl.className = "metric-value text-yellow"; // Slightly stretched
            } else {
                priceVsIdealEl.className = "metric-value text-red";    // Stretched
            }
        }

        // Entry Tiers (DCA levels)
        const tiers = stagedEntry.entry_tiers || [];
        if (tiers.length >= 1) {
            set('entryTier1', `$${tiers[0].price} (${tiers[0].position_pct}%)`);
        } else {
            set('entryTier1', '-');
        }
        if (tiers.length >= 2) {
            set('entryTier2', `$${tiers[1].price} (${tiers[1].position_pct}%)`);
        } else {
            set('entryTier2', '-');
        }
    }

    // DCF Valuation Display (uses valSnapshot from line 446)

    // DCF Intrinsic Value
    const dcfIntrinsic = valSnapshot.dcf_intrinsic_value;
    if (dcfIntrinsic !== undefined && dcfIntrinsic !== null) {
        const dcfVal = typeof dcfIntrinsic === 'number' ? dcfIntrinsic.toFixed(2) : dcfIntrinsic;
        set('dcfIntrinsic', '$' + dcfVal);
    } else {
        set('dcfIntrinsic', '-');
    }

    // DCF Overvaluation %
    const dcfOverval = valSnapshot.dcf_overvaluation_pct;
    const dcfOvervalEl = document.getElementById('dcfOverval');
    if (dcfOvervalEl && dcfOverval !== null && dcfOverval !== undefined) {
        const sign = dcfOverval > 0 ? '+' : '';
        const overvalVal = typeof dcfOverval === 'number' ? dcfOverval.toFixed(1) : dcfOverval;
        dcfOvervalEl.textContent = `${sign}${overvalVal}%`;

        if (dcfOverval > 50) {
            dcfOvervalEl.className = "signal-tag text-red";  // Severely overvalued
            dcfOvervalEl.textContent += " (SEVERE)";
        } else if (dcfOverval > 30) {
            dcfOvervalEl.className = "signal-tag text-red";  // Overvalued
        } else if (dcfOverval > 15) {
            dcfOvervalEl.className = "signal-tag text-yellow";  // Slightly high
        } else if (dcfOverval < -10) {
            dcfOvervalEl.className = "signal-tag text-green";  // Undervalued
        } else {
            dcfOvervalEl.className = "signal-tag";  // Fair
        }
    } else {
        set('dcfOverval', 'N/A');
    }

    // Valuation
    set('pegRatio', val.valuation_metrics?.peg_ratio);

    // Fair Value Logic: Prefer DCF, then DDM, else '-'
    if (val.dcf_valuation) {
        set('fairValue', val.dcf_valuation.fair_value?.toFixed(2), ' (DCF)', true);
    } else if (val.ddm_valuation) {
        set('fairValue', val.ddm_valuation.value_per_share?.toFixed(2), ' (DDM)', true);
    } else {
        set('fairValue', '-');
    }

    set('pbRatio', val.valuation_metrics?.pb_ratio);
    set('psRatio', val.valuation_metrics?.ps_ratio);
    set('evEbitda', val.valuation_metrics?.ev_ebitda);

    // Risk
    set('volatility', risk.volatility !== undefined ? risk.volatility.toFixed(2) : '-', '%');
    set('sharpeRatio', risk.sharpe_ratio !== undefined ? risk.sharpe_ratio.toFixed(2) : '-', '', risk.sharpe_ratio > 1);

    // Fix: Handle max_drawdown correctly (check for null/undefined explicitly)
    let maxDD = risk.max_drawdown?.max_drawdown_pct;
    if (maxDD === undefined && risk.max_drawdown?.max_drawdown !== undefined) {
        maxDD = risk.max_drawdown.max_drawdown * 100;
    }
    set('maxDrawdown', maxDD !== undefined ? maxDD.toFixed(2) : '-', '%', false);

    set('betaValue', fund.beta);
    set('debtEquity', fund.debt_to_equity);
    set('currentRatio', fund.current_ratio);

    // Patterns & Alt
    const activePatterns = Object.keys(patterns).filter(k => patterns[k] !== 'No pattern detected');
    let patternText = 'None';
    if (activePatterns.length > 0) {
        patternText = activePatterns.slice(0, 2).join(', ').replace(/_/g, ' ');
        if (activePatterns.length > 2) patternText += ` (+${activePatterns.length - 2})`;
    }
    // Capitalize first letters
    patternText = patternText.replace(/\b\w/g, l => l.toUpperCase());

    set('candlePattern', patternText);

    set('socialSentiment', alt.social_sentiment?.sentiment_score?.toFixed(2), '', alt.social_sentiment?.sentiment_score > 0);
    // Web Traffic: Handle both legacy monthly_visits and new proxy value
    if (alt.web_traffic?.value) {
        set('webTraffic', alt.web_traffic.value);
    } else {
        set('webTraffic', formatLarge(alt.web_traffic?.monthly_visits));
    }

    // NEW: Enhanced AI Predictions Section
    const enhancedPred = signalsData.enhanced_prediction || {};

    if (enhancedPred.status === 'success') {
        const prediction = enhancedPred.enhanced_prediction || {};
        const altData = enhancedPred.alternative_data || {};
        const sectorAnalysis = enhancedPred.sector_analysis || {};
        const anomalyAlerts = enhancedPred.anomaly_alerts || {};

        // LSTM Prediction
        const lstm = prediction.lstm_prediction || {};
        if (lstm.direction) {
            setSignal('lstmDirection', lstm.direction);
        }
        if (lstm.confidence) {
            set('lstmConfidence', lstm.confidence.toFixed(1) + '%', '', lstm.confidence > 60);
        }
        if (lstm.predictions?.day_5) {
            set('lstm5DayTarget', '$' + lstm.predictions.day_5.toFixed(2));
        }

        // Ensemble Confidence
        if (prediction.confidence) {
            set('ensembleScore', prediction.confidence.toFixed(1) + '%');
        }

        // Sector Strength
        const sectorStrength = sectorAnalysis.relative_strength || {};
        if (sectorStrength.rating) {
            setSignal('sectorStrength', sectorStrength.rating.replace(/_/g, ' '));
        }

        // Alternative Data Signals
        const insiderActivity = altData.insider_activity || {};
        const optionsFlow = altData.options_flow || {};
        const socialSentiment = altData.social_sentiment || {};

        // Insider Signal
        if (insiderActivity.signal) {
            setSignal('insiderSignal', insiderActivity.signal.replace(/_/g, ' '));
        }

        // Options Flow
        if (optionsFlow.signal) {
            setSignal('optionsFlow', optionsFlow.signal);
        }

        // News Sentiment (VADER)
        if (socialSentiment.signal) {
            setSignal('newsSentiment', socialSentiment.signal.replace(/_/g, ' '));
        }
        if (socialSentiment.sentiment_score !== undefined) {
            set('socialSentiment', socialSentiment.sentiment_score.toFixed(3), '', socialSentiment.sentiment_score > 0);
        }

        // Anomaly Alerts
        const totalAlerts = anomalyAlerts.total_alerts || 0;
        const anomalyEl = document.getElementById('anomalyAlerts');
        if (anomalyEl) {
            if (totalAlerts === 0) {
                anomalyEl.textContent = 'NONE';
                anomalyEl.className = 'signal-tag text-green';
            } else if (totalAlerts <= 2) {
                anomalyEl.textContent = `${totalAlerts} DETECTED`;
                anomalyEl.className = 'signal-tag text-yellow';
            } else {
                anomalyEl.textContent = `${totalAlerts} ALERTS`;
                anomalyEl.className = 'signal-tag text-red';
            }
        }

        // NEW: XGBoost Prediction
        const xgboostPred = enhancedPred.xgboost_prediction || {};
        if (xgboostPred.direction) {
            setSignal('xgboostDirection', xgboostPred.direction);
        }
        if (xgboostPred.confidence) {
            const conf = typeof xgboostPred.confidence === 'number' ? xgboostPred.confidence : 50;
            set('xgboostConfidence', conf.toFixed(1) + '%', '', conf > 65);
        }

        // NEW: GRU Prediction  
        const gruPred = enhancedPred.gru_prediction || {};
        if (gruPred.direction) {
            setSignal('gruDirection', gruPred.direction);
        }
        if (gruPred.confidence) {
            const conf = typeof gruPred.confidence === 'number' ? gruPred.confidence : 50;
            set('gruConfidence', conf.toFixed(1) + '%', '', conf > 65);
        }

        // NEW: GARCH Volatility Forecast
        const volForecast = enhancedPred.volatility_forecast || {};
        if (volForecast.model) {
            set('garchModel', volForecast.model);
        }
        if (volForecast.historical_20d) {
            const vol = typeof volForecast.historical_20d === 'number' ? volForecast.historical_20d : 0;
            const volEl = document.getElementById('garchVolatility');
            if (volEl) {
                volEl.textContent = vol.toFixed(1) + '%';
                if (vol > 40) {
                    volEl.className = 'metric-value text-red';
                } else if (vol > 25) {
                    volEl.className = 'metric-value text-yellow';
                } else {
                    volEl.className = 'metric-value text-green';
                }
            }
        }
        if (volForecast.regime && volForecast.regime.regime) {
            const regime = volForecast.regime.regime;
            const regimeEl = document.getElementById('volatilityRegime');
            if (regimeEl) {
                regimeEl.textContent = regime;
                if (regime === 'Low' || regime === 'Normal') {
                    regimeEl.className = 'signal-tag text-green';
                } else if (regime === 'High') {
                    regimeEl.className = 'signal-tag text-yellow';
                } else {
                    regimeEl.className = 'signal-tag text-red';
                }
            }
        }

        // NEW: CNN-LSTM Hybrid Prediction
        const cnnLstmPred = enhancedPred.cnn_lstm_prediction || {};
        if (cnnLstmPred.direction) {
            setSignal('cnnLstmDirection', cnnLstmPred.direction);
        }
        if (cnnLstmPred.confidence) {
            const conf = typeof cnnLstmPred.confidence === 'number' ? cnnLstmPred.confidence : 50;
            set('cnnLstmConfidence', conf.toFixed(1) + '%', '', conf > 65);
        }
        if (cnnLstmPred.predicted_change_pct !== undefined) {
            const change = cnnLstmPred.predicted_change_pct;
            const changeEl = document.getElementById('cnnLstmChange');
            if (changeEl) {
                const sign = change >= 0 ? '+' : '';
                changeEl.textContent = sign + change.toFixed(2) + '%';
                if (change > 1) {
                    changeEl.className = 'metric-value text-green';
                } else if (change < -1) {
                    changeEl.className = 'metric-value text-red';
                } else {
                    changeEl.className = 'metric-value text-yellow';
                }
            }
        }

        // NEW: Attention Transformer Prediction
        const attentionPred = enhancedPred.attention_prediction || {};
        if (attentionPred.direction) {
            setSignal('attentionDirection', attentionPred.direction);
        }
        if (attentionPred.confidence) {
            const conf = typeof attentionPred.confidence === 'number' ? attentionPred.confidence : 50;
            set('attentionConfidence', conf.toFixed(1) + '%', '', conf > 60);
        }
        if (attentionPred.attention_focus && attentionPred.attention_focus.interpretation) {
            const focusEl = document.getElementById('attentionFocus');
            if (focusEl) {
                focusEl.textContent = attentionPred.attention_focus.interpretation;
                focusEl.className = 'metric-value';
            }
        }
    }

    // NEW: Pair Trading / Statistical Arbitrage Section
    const statArb = signalsData.statistical_arbitrage || {};
    if (statArb && !statArb.error) {
        const marketPair = statArb.market_pair || {};
        const sectorPair = statArb.sector_pair || {};
        const hasOpportunity = statArb.has_opportunity;

        // Use market pair as primary (vs SPY), fallback to sector pair
        // The pair field contains "SYMBOL/SPY" format
        const activePair = (marketPair && marketPair.pair) ? marketPair :
            (sectorPair && sectorPair.pair) ? sectorPair : marketPair;
        const cointegration = activePair.cointegration || {};
        const currentSignal = activePair.current_signal || {};

        // Market Pair label - extract from pair field
        const pairName = activePair.pair || (statArb.sector_etf ? `vs ${statArb.sector_etf}` : 'vs SPY');
        set('pairMarket', pairName);

        // Cointegration status
        const cointEl = document.getElementById('pairCointegrated');
        if (cointEl) {
            if (cointegration.is_cointegrated) {
                cointEl.textContent = 'YES';
                cointEl.className = 'signal-tag text-green';
            } else {
                cointEl.textContent = 'NO';
                cointEl.className = 'signal-tag text-red';
            }
        }

        // Z-Score with color coding
        const zscore = currentSignal.current_zscore;
        const zscoreEl = document.getElementById('pairZscore');
        if (zscoreEl && zscore !== undefined) {
            zscoreEl.textContent = zscore.toFixed(2);
            if (Math.abs(zscore) >= 2) {
                zscoreEl.className = 'metric-value text-green'; // Strong signal
            } else if (Math.abs(zscore) >= 1) {
                zscoreEl.className = 'metric-value text-yellow'; // Moderate
            } else {
                zscoreEl.className = 'metric-value'; // Weak
            }
        }

        // Trading Signal
        const signalEl = document.getElementById('pairSignal');
        if (signalEl && currentSignal.signal) {
            signalEl.textContent = currentSignal.signal.replace(/_/g, ' ');
            if (currentSignal.signal === 'LONG_SPREAD' || currentSignal.signal === 'SHORT_SPREAD') {
                signalEl.className = 'signal-tag text-green';
            } else if (currentSignal.signal === 'EXIT' || currentSignal.signal === 'STOP_LOSS') {
                signalEl.className = 'signal-tag text-yellow';
            } else {
                signalEl.className = 'signal-tag';
            }
        }

        // Confidence
        if (currentSignal.confidence !== undefined) {
            set('pairConfidence', currentSignal.confidence.toFixed(1), '%', currentSignal.confidence > 60);
        }

        // Half-Life
        if (cointegration.half_life_days !== undefined) {
            const hl = cointegration.half_life_days;
            set('pairHalfLife', hl.toFixed(0), ' days', hl >= 5 && hl <= 30);
        }

        // Hedge Ratio
        if (cointegration.hedge_ratio !== undefined) {
            set('pairHedgeRatio', cointegration.hedge_ratio.toFixed(3));
        }

        // Opportunity indicator
        const oppEl = document.getElementById('pairOpportunity');
        if (oppEl) {
            if (hasOpportunity) {
                oppEl.textContent = 'ACTIVE';
                oppEl.className = 'signal-tag text-green';
            } else if (cointegration.is_cointegrated) {
                oppEl.textContent = 'WAITING';
                oppEl.className = 'signal-tag text-yellow';
            } else {
                oppEl.textContent = 'NONE';
                oppEl.className = 'signal-tag';
            }
        }
    }
}

function updateMacroContext(macro) {
    const set = (id, val, suffix = '', isGood = null) => {
        const el = document.getElementById(id);
        if (!el) return;
        el.textContent = val !== undefined && val !== null ? val + suffix : '-';
        if (isGood === true) el.className = "metric-value text-green";
        else if (isGood === false) el.className = "metric-value text-red";
        else el.className = "metric-value";
    }

    // Indices
    if (macro.market_indices) {
        const sp = macro.market_indices.sp500;
        set('sp500Val', sp?.value?.toFixed(0));
        set('vixVal', macro.market_indices.vix?.value?.toFixed(2));
    }

    // Treasury
    if (macro.treasury_yields) {
        set('10yYield', macro.treasury_yields['10_year']?.toFixed(2), '%');

        const spread = macro.treasury_yields.yield_curve_spread;
        set('yieldCurve', spread?.toFixed(2), '%', spread > 0);
    }

    // Sectors
    if (macro.sector_performance) {
        // Find top performer manually if not provided in summary
        let top = '-';
        let maxRet = -999;

        for (const [sec, data] of Object.entries(macro.sector_performance)) {
            if (data.return_pct > maxRet) {
                maxRet = data.return_pct;
                top = sec;
            }
        }
        set('topSector', top);
    }
}

async function loadChart(symbol) {
    const res = await fetch(`${API_BASE}/api/chart/${symbol}?period=max`); // Fetch ALL history
    const data = await res.json();

    if (data.status === 'success' && data.chart_data && data.chart_data.dates && data.chart_data.close) {
        renderChart(symbol, data.chart_data.dates, data.chart_data.close);
    } else {
        console.warn('Chart data not available or incomplete');
    }
}

function renderChart(symbol, dates, prices) {
    // --- Purple Gradient Area Chart ---

    if (!dates || !prices || dates.length === 0 || prices.length === 0) {
        console.warn('No chart data available');
        return;
    }

    // Purple/Magenta theme - gradient fades to transparent at bottom
    const mainColor = '#b366ff'; // Purple/Magenta line

    // Main trace with vertical gradient fill (purple at line, transparent at bottom)
    const traceWithFill = {
        x: dates,
        y: prices,
        type: 'scatter',
        mode: 'lines',
        name: symbol,
        line: {
            color: mainColor,
            width: 2.5,
            shape: 'spline'
        },
        fill: 'tozeroy',
        fillgradient: {
            type: 'vertical',
            colorscale: [
                [0, 'rgba(179, 102, 255, 0)'],
                [0.5, 'rgba(179, 102, 255, 0.3)'],
                [1, 'rgba(179, 102, 255, 0.6)']
            ]
        }
    };

    // Calculate date range for initial 6 months view
    const lastDate = new Date(dates[dates.length - 1]);
    const sixMonthsAgo = new Date(lastDate);
    sixMonthsAgo.setMonth(sixMonthsAgo.getMonth() - 6);

    const data = [traceWithFill];

    const layout = {
        showlegend: false,
        margin: { t: 0, b: 25, l: 0, r: 0 },
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        xaxis: {
            visible: true,
            showgrid: false,
            showline: false,
            zeroline: false,
            tickfont: { color: '#ffffff', size: 11, family: 'Outfit, sans-serif' },
            tickformat: '%b %Y',
            nticks: 5,
            fixedrange: false,
            range: [sixMonthsAgo.toISOString().split('T')[0], lastDate.toISOString().split('T')[0]]
        },
        yaxis: {
            visible: false,
            showgrid: false,
            showline: false,
            zeroline: false,
            fixedrange: false,
            autorange: true
        },
        hovermode: 'x unified',
        hoverlabel: {
            bgcolor: '#ffffff',
            bordercolor: '#333333',
            font: { color: '#000000', family: 'Outfit, sans-serif', size: 12 }
        },
        dragmode: 'zoom'
    };

    const config = {
        responsive: true,
        displayModeBar: false, // Hide controls for cleaner look
        displaylogo: false,
        staticPlot: false,
        scrollZoom: true
    };

    // Check if element exists
    const chartEl = document.getElementById('priceChart');
    if (!chartEl) {
        console.warn('priceChart element not found');
        return;
    }

    Plotly.newPlot('priceChart', data, layout, config);
}

function formatLarge(num) {
    if (!num && num !== 0) return '-';
    if (num >= 1e12) return (num / 1e12).toFixed(2) + 'T';
    if (num >= 1e9) return (num / 1e9).toFixed(2) + 'B';
    if (num >= 1e6) return (num / 1e6).toFixed(2) + 'M';
    if (num >= 1e3) return (num / 1e3).toFixed(2) + 'K';
    return num.toLocaleString();
}

function toggleSection(id, btnElement) { // btnElement is optional if not passed
    const el = document.getElementById(id);
    // Find the icon if not passed directly (assuming it's in the caller or nearby)
    // The onclick in HTML is: toggleSection('transparencyDetails')
    // We can find the icon by DOM traversal relative to the section ID or just query it if unique.
    // For 'transparencyDetails', the button is .panel-header-small

    // Better approach: toggle the class on the icon inside the previous sibling of the content
    // or just find .expand-icon inside the header above this content

    let icon = null;
    if (el) {
        // Assume the header is immediately before the content div
        const header = el.previousElementSibling;
        if (header) {
            icon = header.querySelector('.expand-icon');
        }
    }

    if (el.style.display === 'none' || el.classList.contains('hidden')) {
        el.style.display = 'block';
        el.classList.remove('hidden');
        if (icon) icon.classList.add('rotate-180');
    } else {
        el.style.display = 'none';
        el.classList.add('hidden');
        if (icon) icon.classList.remove('rotate-180');
    }
}

function updateTransparencySection(data) {
    if (!data || data.status !== 'success') return;

    // Show sections
    document.getElementById('modelTransparency').classList.remove('hidden');
    document.getElementById('priceTargets').classList.remove('hidden');
    document.getElementById('historicalValidation').classList.remove('hidden');

    // 1. Model Transparency
    const modelsGrid = document.getElementById('modelsGrid');
    modelsGrid.innerHTML = ''; // Clear previous

    const metrics = data.model_metrics || {};
    const modelsUsed = data.meta?.models_used || [];

    // Use only real metrics from API - no mock data
    const displayModels = metrics;

    for (const [model, stats] of Object.entries(displayModels)) {
        const div = document.createElement('div');
        div.className = 'model-card';
        div.innerHTML = `
            <div class="model-name">${model.toUpperCase().replace('_', ' ')}</div>
            <div class="model-stat-row">
                <span>Accuracy</span>
                <span class="text-green">${stats.accuracy}%</span>
            </div>
            <div class="model-stat-row">
                <span>MAPE</span>
                <span class="mape-badge">${stats.mape}%</span>
            </div>
        `;
        modelsGrid.appendChild(div);
    }

    // 2. Price Targets
    const targets = data.price_targets || {};
    const periods = ['7d', '30d', '90d'];
    const labels = { '7d': 'day_7', '30d': 'day_30', '90d': 'day_90' };

    periods.forEach(p => {
        const targetData = targets[labels[p]] || {};
        if (targetData.price) {
            document.getElementById(`target${p}`).innerText = '$' + targetData.price.toFixed(2);

            const low = targetData.confidence_interval ? targetData.confidence_interval[0] : (targetData.price * 0.95);
            const high = targetData.confidence_interval ? targetData.confidence_interval[1] : (targetData.price * 1.05);
            document.getElementById(`range${p}`).innerText = `$${low.toFixed(0)} - $${high.toFixed(0)}`;

            const dirEl = document.getElementById(`direction${p}`);
            dirEl.innerText = targetData.direction || 'Neutral';
            dirEl.className = 'target-direction ' + (
                targetData.direction === 'Bullish' ? 'text-green' :
                    (targetData.direction === 'Bearish' ? 'text-red' : 'text-yellow')
            );
        }
    });

    // 3. Historical Validation - Only use real data from API
    const sysAcc = data.system_accuracy;
    if (sysAcc !== undefined && sysAcc !== null && typeof sysAcc === 'number') {
        document.getElementById('systemAccuracy').innerText = sysAcc.toFixed(1);
    } else {
        document.getElementById('systemAccuracy').innerText = '-';
    }

    // Use real stats from API only
    const backtestData = data.backtest || {};
    const validatedCount = data.validated_count;
    document.getElementById('validatedCount').innerText = (validatedCount !== undefined && validatedCount !== null && validatedCount > 0) ? validatedCount : '-';

    const winRate = backtestData.win_rate;
    document.getElementById('backtestWinRate').innerText = (winRate !== undefined && winRate !== null && typeof winRate === 'number') ? winRate.toFixed(1) + '%' : '-';

    const sharpeRatio = backtestData.sharpe_ratio;
    document.getElementById('strategySharpe').innerText = (sharpeRatio !== undefined && sharpeRatio !== null && typeof sharpeRatio === 'number') ? sharpeRatio.toFixed(2) : '-';

    // Model Consensus - Calculate from model metrics
    const modelMetrics = data.model_metrics || {};
    const modelCount = Object.keys(modelMetrics).length;
    if (modelCount > 0) {
        // Count directions
        let bullish = 0, bearish = 0, neutral = 0;
        for (const [name, info] of Object.entries(modelMetrics)) {
            const dir = (info.direction || '').toLowerCase();
            if (dir === 'bullish') bullish++;
            else if (dir === 'bearish') bearish++;
            else neutral++;
        }
        const maxAgree = Math.max(bullish, bearish, neutral);
        const consensusEl = document.getElementById('modelConsensus');
        if (consensusEl) {
            consensusEl.innerText = `${maxAgree}/${modelCount}`;
            // Update bar width
            const barFill = consensusEl.closest('.track-stat')?.querySelector('.track-stat-fill');
            if (barFill) {
                barFill.style.width = `${(maxAgree / modelCount) * 100}%`;
            }
        }
    } else {
        const consensusEl = document.getElementById('modelConsensus');
        if (consensusEl) consensusEl.innerText = '-';
    }
}
