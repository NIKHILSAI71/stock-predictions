"""
Technical Signals Multi-Indicator Model

Aggregates multiple technical indicators with weighted voting:
- Trend Indicators (40%): MA crossovers, MACD, ADX
- Momentum Indicators (30%): RSI, Stochastic, CCI
- Volume Indicators (20%): OBV, Volume momentum
- Volatility Indicators (10%): Bollinger Bands, ATR
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


def calculate_adx(high, low, close, period=14):
    """Calculate Average Directional Index."""
    try:
        # True Range
        hl = high - low
        hc = np.abs(high - close.shift(1))
        lc = np.abs(low - close.shift(1))
        tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)

        # Directional Movement
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low

        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) &
                            (down_move > 0), down_move, 0)

        # Smoothed
        atr = tr.rolling(period).mean()
        plus_di = 100 * pd.Series(plus_dm).rolling(period).mean() / atr
        minus_di = 100 * pd.Series(minus_dm).rolling(period).mean() / atr

        # ADX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(period).mean()

        return float(adx.iloc[-1]) if not adx.empty else 25.0
    except:
        return 25.0


def get_technical_signals(stock_data: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate technical signal prediction from multiple indicators.

    Args:
        stock_data: DataFrame with OHLCV data

    Returns:
        Technical signals prediction
    """
    try:
        if len(stock_data) < 50:
            return {
                "error": "Insufficient data",
                "direction": "Neutral",
                "confidence": 50,
                "status": "failed"
            }

        close = stock_data['Close']
        high = stock_data.get('High', close)
        low = stock_data.get('Low', close)
        volume = stock_data.get('Volume', pd.Series(
            [1]*len(close), index=close.index))

        signals = {}
        scores = []

        # === TREND INDICATORS (40% weight) ===

        # MA Crossovers
        sma_20 = close.rolling(20).mean()
        sma_50 = close.rolling(50).mean()
        if len(close) >= 50:
            ma_signal = 1 if sma_20.iloc[-1] > sma_50.iloc[-1] else -1
            ma_strength = abs(
                sma_20.iloc[-1] - sma_50.iloc[-1]) / sma_50.iloc[-1] * 100
            signals['ma_crossover'] = ma_signal
            scores.append(('trend', ma_signal * min(ma_strength, 5) * 20))

        # MACD
        ema_12 = close.ewm(span=12).mean()
        ema_26 = close.ewm(span=26).mean()
        macd = ema_12 - ema_26
        signal_line = macd.ewm(span=9).mean()
        macd_hist = macd - signal_line
        if len(macd_hist) > 0:
            macd_signal = 1 if macd_hist.iloc[-1] > 0 else -1
            signals['macd'] = macd_signal
            scores.append(('trend', macd_signal * 80))

        # ADX (trend strength)
        adx_value = calculate_adx(high, low, close)
        if adx_value > 25:  # Strong trend
            trend_direction = 1 if close.iloc[-1] > close.iloc[-20] else -1
            signals['adx'] = trend_direction
            scores.append(('trend', trend_direction * min(adx_value, 50)))

        # === MOMENTUM INDICATORS (30% weight) ===

        # RSI
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        if len(rsi) > 0:
            rsi_val = rsi.iloc[-1]
            if rsi_val > 70:
                rsi_signal = -1  # Overbought
            elif rsi_val < 30:
                rsi_signal = 1  # Oversold
            else:
                rsi_signal = 0
            signals['rsi'] = rsi_signal
            if rsi_signal != 0:
                scores.append(('momentum', rsi_signal * 70))

        # Stochastic Oscillator
        if len(high) >= 14:
            low_14 = low.rolling(14).min()
            high_14 = high.rolling(14).max()
            stoch_k = 100 * (close - low_14) / (high_14 - low_14 + 1e-10)
            if len(stoch_k) > 0:
                stoch_val = stoch_k.iloc[-1]
                if stoch_val > 80:
                    stoch_signal = -1
                elif stoch_val < 20:
                    stoch_signal = 1
                else:
                    stoch_signal = 0
                signals['stochastic'] = stoch_signal
                if stoch_signal != 0:
                    scores.append(('momentum', stoch_signal * 60))

        # === VOLUME INDICATORS (20% weight) ===

        # Volume trend
        vol_sma = volume.rolling(20).mean()
        if len(volume) >= 20:
            recent_vol = volume.tail(5).mean()
            vol_signal = 1 if recent_vol > vol_sma.iloc[-1] * 1.2 else 0
            signals['volume_trend'] = vol_signal
            if vol_signal > 0:
                scores.append(('volume', vol_signal * 50))

        # OBV (On-Balance Volume)
        obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
        obv_sma = obv.rolling(20).mean()
        if len(obv) >= 20:
            obv_signal = 1 if obv.iloc[-1] > obv_sma.iloc[-1] else -1
            signals['obv'] = obv_signal
            scores.append(('volume', obv_signal * 50))

        # === VOLATILITY INDICATORS (10% weight) ===

        # Bollinger Bands
        bb_sma = close.rolling(20).mean()
        bb_std = close.rolling(20).std()
        upper_band = bb_sma + 2 * bb_std
        lower_band = bb_sma - 2 * bb_std
        if len(close) >= 20:
            if close.iloc[-1] < lower_band.iloc[-1]:
                bb_signal = 1  # Oversold
            elif close.iloc[-1] > upper_band.iloc[-1]:
                bb_signal = -1  # Overbought
            else:
                bb_signal = 0
            signals['bollinger'] = bb_signal
            if bb_signal != 0:
                scores.append(('volatility', bb_signal * 40))

        # === AGGREGATE SCORES ===

        # Weight by category
        category_weights = {'trend': 0.4, 'momentum': 0.3,
                            'volume': 0.2, 'volatility': 0.1}

        weighted_score = 0
        for category, score in scores:
            weighted_score += score * category_weights[category]

        # Normalize to 0-100
        final_score = 50 + (weighted_score / 10)  # Scale to range
        final_score = max(0, min(100, final_score))

        # Direction
        if final_score > 60:
            direction = "Bullish"
        elif final_score < 40:
            direction = "Bearish"
        else:
            direction = "Neutral"

        # Confidence
        confidence = abs(final_score - 50) * 2  # 0-100 scale
        confidence = max(35, min(85, confidence))

        # Expected change
        market_volatility = close.pct_change().tail(20).std()
        expected_change = (final_score - 50) / 50 * market_volatility * 100

        return {
            "direction": direction,
            "confidence": round(confidence, 1),
            "predicted_change_pct": round(expected_change, 2),
            "current_price": float(close.iloc[-1]),
            "technical_score": round(final_score, 1),
            "signals": signals,
            "signal_breakdown": {
                "trend_indicators": sum(1 for cat, _ in scores if cat == 'trend'),
                "momentum_indicators": sum(1 for cat, _ in scores if cat == 'momentum'),
                "volume_indicators": sum(1 for cat, _ in scores if cat == 'volume'),
                "volatility_indicators": sum(1 for cat, _ in scores if cat == 'volatility')
            },
            "timestamp": datetime.now().isoformat(),
            "method": "technical_signals_aggregator",
            "status": "success"
        }

    except Exception as e:
        logger.error(f"Error in technical signals: {e}", exc_info=True)
        return {
            "error": str(e),
            "direction": "Neutral",
            "confidence": 50,
            "status": "failed"
        }
