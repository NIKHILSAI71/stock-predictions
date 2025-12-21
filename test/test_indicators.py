
import unittest
import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from technical.indicators import (
    rsi, macd, bollinger_bands, stochastic_oscillator, adx, cci, atr,
    williams_r, momentum, roc, true_strength_index
)

class TestIndicators(unittest.TestCase):
    def setUp(self):
        # Create sample data - 50 periods
        np.random.seed(42)
        dates = pd.date_range(start='2020-01-01', periods=50)
        self.close = pd.Series(np.cumsum(np.random.randn(50)) + 100, index=dates, name="Close")
        self.high = self.close + np.random.rand(50)
        self.low = self.close - np.random.rand(50)
        self.df = pd.DataFrame({
            "Close": self.close,
            "High": self.high,
            "Low": self.low
        })

    def test_rsi(self):
        # RSI range 0-100
        val = rsi(self.close, period=14)
        self.assertFalse(val.isna().all())
        # First 14 should be NaN (or handled differently depending on implementation)
        # Implementation says: avg_gains.iloc[period - 1] = first_avg_gain.
        # Then loop from period to end.
        # So first period-1 might be NaN?
        # Let's check tail
        self.assertTrue((val.dropna() >= 0).all() and (val.dropna() <= 100).all())

    def test_macd(self):
        res = macd(self.close)
        self.assertIn('macd', res)
        self.assertIn('signal', res)
        self.assertIn('histogram', res)
        self.assertEqual(len(res['macd']), 50)

    def test_bollinger_bands(self):
        res = bollinger_bands(self.close)
        self.assertIn('upper', res)
        self.assertIn('lower', res)
        self.assertIn('middle', res)
        # Check logic: Upper > Middle > Lower (after stabilization)
        # Provide enough data points
        valid_indices = res['middle'].notna()
        self.assertTrue((res['upper'][valid_indices] >= res['middle'][valid_indices]).all())
        self.assertTrue((res['middle'][valid_indices] >= res['lower'][valid_indices]).all())

    def test_stochastic_oscillator(self):
        res = stochastic_oscillator(self.df)
        self.assertIn('k', res)
        self.assertIn('d', res)
        # K should be approx 0-100
        valid_k = res['k'].dropna()
        # It's possible to exceed slightly with some definitions or float errors, but bounded usually
        # Formula: %K = ((Close - Lowest Low) / (Highest High - Lowest Low)) * 100
        # Since Close is within [Lowest Low, Highest High], it must be 0-100.
        self.assertTrue((valid_k >= 0).all() and (valid_k <= 100).all())

    def test_adx(self):
        res = adx(self.df)
        self.assertIn('adx', res)
        self.assertIn('plus_di', res)
        self.assertIn('minus_di', res)
        # ADX >= 0
        valid_adx = res['adx'].dropna()
        self.assertTrue((valid_adx >= 0).all())

    def test_cci(self):
        val = cci(self.df)
        # CCI can be negative or positive large numbers
        self.assertFalse(val.isna().all())

    def test_atr(self):
        val = atr(self.df)
        self.assertFalse(val.isna().all())
        # ATR should be positive
        self.assertTrue((val.dropna() > 0).all())

    def test_williams_r(self):
        val = williams_r(self.df)
        # Range -100 to 0
        valid = val.dropna()
        self.assertTrue((valid >= -100).all() and (valid <= 0).all())

    def test_momentum(self):
        val = momentum(self.close)
        self.assertEqual(len(val), 50)

    def test_roc(self):
        val = roc(self.close)
        self.assertEqual(len(val), 50)

    def test_true_strength_index(self):
        res = true_strength_index(self.close)
        self.assertIn('tsi', res)
        self.assertIn('signal', res)

if __name__ == '__main__':
    unittest.main()
