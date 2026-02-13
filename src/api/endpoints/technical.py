from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from src.core.utils import sanitize_for_json
from src.data import get_stock_data, validate_symbol
from src.analysis.technical import (
    sma, ema, rsi, macd, bollinger_bands, stochastic_oscillator,
    adx, cci, atr, parabolic_sar, supertrend, fibonacci_retracement,
    true_strength_index
)
from src.analysis.technical.chart_patterns import detect_all_chart_patterns

router = APIRouter()


@router.get("/technical/{symbol}")
async def get_technical_analysis(
    symbol: str,
    period: str = "1y",
    interval: str = "1d"
):
    """Get all technical indicators for a stock."""
    try:
        if not validate_symbol(symbol):
            return JSONResponse(
                status_code=404,
                content={"status": "error",
                         "detail": f"Symbol '{symbol}' not found."}
            )

        data = get_stock_data(symbol, period, interval)

        # Calculate all indicators
        indicators = {}

        # Moving Averages
        indicators['sma_20'] = round(sma(data, 20).iloc[-1], 2)
        indicators['sma_50'] = round(sma(data, 50).iloc[-1], 2)
        indicators['sma_200'] = round(
            sma(data, 200).iloc[-1], 2) if len(data) >= 200 else None
        indicators['ema_12'] = round(ema(data, 12).iloc[-1], 2)
        indicators['ema_26'] = round(ema(data, 26).iloc[-1], 2)

        # RSI
        rsi_values = rsi(data)
        indicators['rsi'] = round(rsi_values.iloc[-1], 2)
        indicators['rsi_signal'] = 'Overbought' if rsi_values.iloc[-1] > 70 else 'Oversold' if rsi_values.iloc[-1] < 30 else 'Neutral'

        # MACD
        macd_data = macd(data)
        indicators['macd'] = round(macd_data['macd'].iloc[-1], 2)
        indicators['macd_signal'] = round(macd_data['signal'].iloc[-1], 2)
        indicators['macd_histogram'] = round(
            macd_data['histogram'].iloc[-1], 2)
        indicators['macd_trend'] = 'Bullish' if macd_data['histogram'].iloc[-1] > 0 else 'Bearish'

        # Bollinger Bands
        bb = bollinger_bands(data)
        indicators['bb_upper'] = round(bb['upper'].iloc[-1], 2)
        indicators['bb_middle'] = round(bb['middle'].iloc[-1], 2)
        indicators['bb_lower'] = round(bb['lower'].iloc[-1], 2)
        indicators['bb_percent_b'] = round(bb['percent_b'].iloc[-1], 2)

        # Stochastic
        stoch = stochastic_oscillator(data)
        indicators['stoch_k'] = round(stoch['k'].iloc[-1], 2)
        indicators['stoch_d'] = round(stoch['d'].iloc[-1], 2)

        # ADX
        adx_data = adx(data)
        indicators['adx'] = round(adx_data['adx'].iloc[-1], 2)
        indicators['plus_di'] = round(adx_data['plus_di'].iloc[-1], 2)
        indicators['minus_di'] = round(adx_data['minus_di'].iloc[-1], 2)
        indicators['trend_strength'] = 'Strong' if adx_data['adx'].iloc[-1] > 25 else 'Weak'

        # CCI
        cci_values = cci(data)
        indicators['cci'] = round(cci_values.iloc[-1], 2)

        # ATR
        atr_values = atr(data)
        indicators['atr'] = round(atr_values.iloc[-1], 2)

        # Parabolic SAR
        psar = parabolic_sar(data)
        indicators['psar'] = round(psar['sar'].iloc[-1], 2)
        indicators['psar_trend'] = 'Bullish' if psar['trend'].iloc[-1] == 1 else 'Bearish'

        # Supertrend
        st = supertrend(data)
        indicators['supertrend'] = round(st['supertrend'].iloc[-1], 2)
        indicators['supertrend_signal'] = 'Buy' if st['trend'].iloc[-1] == 1 else 'Sell'

        # Fibonacci
        fib = fibonacci_retracement(data)
        indicators['fibonacci'] = fib

        # Support/Resistance - Placeholder
        indicators['support_levels'] = []
        indicators['resistance_levels'] = []

        # Pivot Points - Placeholder
        indicators['pivot_points'] = {}

        # Current price
        indicators['current_price'] = round(data['Close'].iloc[-1], 2)

        # Price change
        if len(data) >= 2:
            price_change = data['Close'].iloc[-1] - data['Close'].iloc[-2]
            price_change_pct = (price_change / data['Close'].iloc[-2]) * 100
            indicators['price_change'] = round(price_change, 2)
            indicators['price_change_pct'] = round(price_change_pct, 2)

        return {
            "status": "success",
            "symbol": symbol.upper(),
            "period": period,
            "interval": interval,
            "indicators": sanitize_for_json(indicators)
        }
    except HTTPException:
        raise
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "detail": str(e)}
        )


@router.get("/volume/{symbol}")
async def get_volume_analysis(
    symbol: str,
    period: str = "6mo"
):
    """Get comprehensive volume analysis."""
    try:
        data = get_stock_data(symbol, period, "1d")

        # Volume analysis module removed, using basic metrics
        vol_analysis = {
            "current_volume": int(data['Volume'].iloc[-1]),
            "avg_volume": int(data['Volume'].mean())
        }

        # Add TSI (Kept)
        tsi_data = true_strength_index(data)
        vol_analysis['tsi'] = round(tsi_data['tsi'].iloc[-1], 2)
        vol_analysis['tsi_signal'] = round(tsi_data['signal'].iloc[-1], 2)

        return {
            "status": "success",
            "symbol": symbol.upper(),
            "volume_analysis": sanitize_for_json(vol_analysis)
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "detail": str(e)}
        )


@router.get("/patterns/{symbol}")
async def get_chart_patterns(
    symbol: str,
    period: str = "6mo",
    interval: str = "1d"
):
    """Detect chart patterns in stock price data."""
    try:
        if not validate_symbol(symbol):
            return JSONResponse(
                status_code=404,
                content={"status": "error",
                         "detail": f"Symbol '{symbol}' not found."}
            )

        data = get_stock_data(symbol, period, interval)

        # Detect all chart patterns
        pattern_results = detect_all_chart_patterns(data)

        return {
            "status": "success",
            "symbol": symbol.upper(),
            "period": period,
            "interval": interval,
            "patterns": sanitize_for_json(pattern_results)
        }
    except HTTPException:
        raise
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "detail": str(e)}
        )
