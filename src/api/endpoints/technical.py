from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from src.core.utils import sanitize_for_json
from src.data import get_stock_data, validate_symbol
from src.analysis.technical import (
    sma, ema, rsi, macd, bollinger_bands, stochastic_oscillator,
    adx, cci, atr, parabolic_sar, supertrend, fibonacci_retracement,
    find_support_resistance, pivot_points, detect_patterns,
    ichimoku_cloud, ichimoku_signals, true_strength_index,
    volume_analysis
)

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
                content={"status": "error", "detail": f"Symbol '{symbol}' not found."}
            )
        
        data = get_stock_data(symbol, period, interval)
        
        # Calculate all indicators
        indicators = {}
        
        # Moving Averages
        indicators['sma_20'] = round(sma(data, 20).iloc[-1], 2)
        indicators['sma_50'] = round(sma(data, 50).iloc[-1], 2)
        indicators['sma_200'] = round(sma(data, 200).iloc[-1], 2) if len(data) >= 200 else None
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
        indicators['macd_histogram'] = round(macd_data['histogram'].iloc[-1], 2)
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
        
        # Support/Resistance
        sr = find_support_resistance(data)
        indicators['support_levels'] = sr['support'][:3]
        indicators['resistance_levels'] = sr['resistance'][:3]
        
        # Pivot Points
        pivots = pivot_points(data)
        indicators['pivot_points'] = pivots
        
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

@router.get("/patterns/{symbol}")
async def get_candlestick_patterns(
    symbol: str,
    period: str = "3mo"
):
    """Detect candlestick patterns in recent price data."""
    try:
        data = get_stock_data(symbol, period, "1d")
        patterns = detect_patterns(data)
        
        return {
            "status": "success",
            "symbol": symbol.upper(),
            "patterns": patterns
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/ichimoku/{symbol}")
async def get_ichimoku_analysis(
    symbol: str,
    period: str = "1y"
):
    """Get Ichimoku Cloud analysis for a stock."""
    try:
        if not validate_symbol(symbol):
            return JSONResponse(
                status_code=404,
                content={"status": "error", "detail": f"Symbol '{symbol}' not found."}
            )
        
        data = get_stock_data(symbol, period, "1d")
        signals = ichimoku_signals(data)
        
        return {
            "status": "success",
            "symbol": symbol.upper(),
            "ichimoku": sanitize_for_json(signals)
        }
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
    """Get comprehensive volume analysis (OBV, MFI, CMF)."""
    try:
        if not validate_symbol(symbol):
            return JSONResponse(
                status_code=404,
                content={"status": "error", "detail": f"Symbol '{symbol}' not found."}
            )
        
        data = get_stock_data(symbol, period, "1d")
        vol_analysis = volume_analysis(data)
        
        # Add TSI
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
