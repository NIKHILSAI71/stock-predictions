
import unittest
import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from technical.moving_averages import (
    sma, ema, wma, dema, tema, vwap
)

class TestMovingAverages(unittest.TestCase):
    def setUp(self):
        # Create sample data
        self.prices = pd.Series([10, 11, 12, 13, 14, 15, 16, 17, 18, 19], name="Close")
        self.df = pd.DataFrame({
            "Close": self.prices,
            "High": self.prices + 1,
            "Low": self.prices - 1,
            "Volume": [100] * 10
        })

    def test_sma(self):
        # SMA of [10, 11, 12] over 3 periods is 11
        result = sma(self.prices, period=3)
        self.assertEqual(result.iloc[2], 11.0)
        self.assertTrue(pd.isna(result.iloc[0])) # First 2 should be NaN
        self.assertTrue(pd.isna(result.iloc[1]))
        
        # Test DataFrame input
        result_df = sma(self.df, period=3)
        self.assertEqual(result.iloc[2], 11.0)

    def test_ema(self):
        # EMA calculation
        # Initial EMA is usually SMA of first period? Or price itself?
        # Pandas ewm(adjust=False) starts with the first value.
        # P0=10. EMA0=10.
        # P1=11. Multiplier=2/(3+1)=0.5. EMA1 = (11-10)*0.5 + 10 = 10.5.
        # P2=12. EMA2 = (12-10.5)*0.5 + 10.5 = 0.75 + 10.5 = 11.25.
        
        result = ema(self.prices, period=3)
        self.assertEqual(result.iloc[0], 10.0)
        self.assertEqual(result.iloc[1], 10.5)
        self.assertEqual(result.iloc[2], 11.25)

    def test_wma(self):
        # WMA of [10, 11, 12], period=3
        # (10*1 + 11*2 + 12*3) / 6 = (10 + 22 + 36) / 6 = 68 / 6 = 11.3333...
        result = wma(self.prices, period=3)
        self.assertAlmostEqual(result.iloc[2], 11.3333, places=4)
        
    def test_dema(self):
        # Just check it runs and produces values
        result = dema(self.prices, period=3)
        self.assertFalse(result.isna().all())
        
    def test_tema(self):
        result = tema(self.prices, period=3)
        self.assertFalse(result.isna().all())

    def test_vwap(self):
        # Typical Price = (H+L+C)/3
        # Here H=C+1, L=C-1. TP = (C+1 + C-1 + C)/3 = 3C/3 = C.
        # So TP is same as Close.
        # Volume is constant 100.
        # VWAP = Cumulative(C * 100) / Cumulative(100)
        # = 100 * Cumulative(C) / (100 * N)
        # = Sum(C) / N
        # This is exactly SMA if we look at the last point of Cumulative average.
        
        # Wait, VWAP is usually intraday but here it's cumulative over the whole series provided.
        # For index 2 (prices 10, 11, 12):
        # Sum(C) = 33. N=3. VWAP = 11.
        result = vwap(self.df)
        self.assertEqual(result.iloc[2], 11.0)
        self.assertEqual(result.iloc[0], 10.0)

if __name__ == '__main__':
    unittest.main()
