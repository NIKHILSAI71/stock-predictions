import { useEffect, useState } from 'react';
import Plotly from 'plotly.js-dist-min';
import createPlotlyComponentModule from 'react-plotly.js/factory';

const createPlotlyComponent = createPlotlyComponentModule.default || createPlotlyComponentModule;
const Plot = createPlotlyComponent(Plotly);
import { useStock } from '../../context/StockContext';
import { api } from '../../services/api';

const PriceChart = () => {
    const { symbol, stockData } = useStock();
    const [chartData, setChartData] = useState(null);
    const [period, setPeriod] = useState('max'); // Default to full history
    // Indicators state could be local or context if shared. Local for now.

    // Config mirroring loadChart logic
    useEffect(() => {
        if (!symbol) return;

        const fetchChart = async () => {
            try {
                // Use the API service which includes authentication
                const data = await api.fetchHistory(symbol, period);

                if (data.status === 'success') {
                    const history = data.data;

                    // Logic from static/js/app.js: renderChart
                    // Purple/Magenta theme - gradient fades to transparent at bottom
                    const mainColor = '#b366ff';

                    const trace1 = {
                        x: history.dates,
                        y: history.closes,
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
                                [0, 'rgba(179, 102, 255, 0)'],      // Transparent at bottom
                                [0.2, 'rgba(179, 102, 255, 0.1)'],  // Quick fade
                                [1, 'rgba(179, 102, 255, 0.5)']     // Semi-transparent at top
                            ]
                        }
                    };

                    // Calculate range for the last 3 months
                    const dates = history.dates;
                    const lastDateStr = dates[dates.length - 1];
                    const lastDate = new Date(lastDateStr);
                    const threeMonthsAgo = new Date(lastDate);
                    threeMonthsAgo.setMonth(threeMonthsAgo.getMonth() - 3);
                    const startDateStr = threeMonthsAgo.toISOString().split('T')[0];

                    const layout = {
                        showlegend: false,
                        margin: { t: 10, b: 40, l: 0, r: 0 }, // Increased bottom margin for dates
                        paper_bgcolor: 'rgba(0,0,0,0)',
                        plot_bgcolor: 'rgba(0,0,0,0)',
                        xaxis: {
                            visible: true,
                            showgrid: false,
                            showline: false,
                            zeroline: false,
                            type: 'date',
                            tickformat: '%b %d',
                            nticks: 6,
                            automargin: true,
                            tickfont: { color: '#ffffff', size: 11, family: 'Outfit, sans-serif' },
                            fixedrange: false,
                            range: [startDateStr, lastDateStr] // Default view to last 3 months
                        },
                        yaxis: {
                            visible: false,
                            showgrid: false,
                            showline: false,
                            zeroline: false,
                            fixedrange: true, // Lock Y-axis to prevent drifting off screen
                            autorange: true
                        },
                        hovermode: 'x unified',
                        hoverlabel: {
                            bgcolor: '#ffffff',
                            bordercolor: '#333333',
                            font: { color: '#000000', family: 'Outfit, sans-serif', size: 12 }
                        },
                        dragmode: 'zoom',
                        height: 300,
                        font: { family: 'Outfit, sans-serif' }
                    };

                    setChartData({ data: [trace1], layout });
                }
            } catch (e) {
                console.error("Chart load error", e);
            }
        };

        fetchChart();
    }, [symbol, period]);

    if (!chartData) return <div className="center-chart"></div>;

    return (
        <div id="priceChart" className="center-chart">
            <Plot
                data={chartData.data}
                layout={{
                    ...chartData.layout,
                    autosize: true
                }}
                useResizeHandler={true}
                style={{ width: '100%', height: '100%' }}
                config={{
                    responsive: true,
                    displayModeBar: false,
                    displaylogo: false,
                    staticPlot: false
                }}
            />
        </div>
    );
};

export default PriceChart;
