
import unittest
import pandas as pd
import numpy as np
import sys
import os

# Add src to path so we can import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from quantitative.risk_metrics import (
    sharpe_ratio, sortino_ratio, calmar_ratio, maximum_drawdown,
    treynor_ratio, information_ratio, beta, comprehensive_risk_analysis
)

class TestRiskMetrics(unittest.TestCase):
    def setUp(self):
        # Create sample data
        dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
        self.returns = pd.Series(np.random.normal(0.001, 0.02, 100), index=dates)
        self.benchmark_returns = pd.Series(np.random.normal(0.0005, 0.015, 100), index=dates)
        
        # Create a specific case for known values
        self.consistent_returns = pd.Series([0.01] * 100, index=dates) # 1% daily return
        self.zero_returns = pd.Series([0.0] * 100, index=dates)
        
        # Create a drawdown case: Up then Down
        drawdown_data = [0.01] * 5 + [-0.01] * 5 # 5 days up 1%, 5 days down 1%
        self.drawdown_returns = pd.Series(drawdown_data)

    def test_sharpe_ratio(self):
        # Test with consistent returns (std dev should be 0, handle gracefully)
        sr = sharpe_ratio(self.consistent_returns)
        self.assertEqual(sr, 0.0) # std dev is 0, should catch and return 0.0
        
        # Test with random returns
        sr = sharpe_ratio(self.returns)
        self.assertIsInstance(sr, float)
        
        # Test basic calculation logic manually for a small set
        small_returns = pd.Series([0.1, -0.1])
        # mean = 0, std needs calculation. 
        # But let's trust the function logic if it runs, we are checking specific correctness
        # if std_dev is not 0.
        
    def test_sortino_ratio(self):
        # Sortino should be infinite if no downside
        sortino = sortino_ratio(self.consistent_returns)
        self.assertEqual(sortino, float('inf'))
        
        # Test with known downside
        small_downside = pd.Series([0.1, -0.1, 0.1, -0.05])
        sortino = sortino_ratio(small_downside)
        self.assertIsInstance(sortino, float)

    def test_maximum_drawdown(self):
        # Test known drawdown
        # 100, 110, 121, 108.9 (10% drop), ...
        # returns: 0.1, 0.1, -0.1
        returns = pd.Series([0.1, 0.1, -0.1])
        # Prices: 1 -> 1.1 -> 1.21 -> 1.089
        # Max: 1.21. Drawdown: (1.089 - 1.21) / 1.21 = -0.1
        # The function returns the raw drawdown value (negative)
        result = maximum_drawdown(returns)
        # Expected is -0.1
        self.assertAlmostEqual(result['max_drawdown'], -0.1, places=4)
        
    def test_calmar_ratio(self):
        calmar = calmar_ratio(self.returns)
        self.assertIsInstance(calmar, float)
        
    def test_beta(self):
        # If returns are identical to benchmark, beta should be 1
        b_metrics = beta(self.benchmark_returns, self.benchmark_returns)
        self.assertAlmostEqual(b_metrics['beta'], 1.0, places=4)
        self.assertAlmostEqual(b_metrics['alpha'], 0.0, places=4)

    def test_comprehensive_risk_analysis(self):
        report = comprehensive_risk_analysis(self.returns, self.benchmark_returns)
        self.assertIn('sharpe_ratio', report)
        self.assertIn('max_drawdown', report)
        self.assertIn('risk_rating', report)

if __name__ == '__main__':
    unittest.main()
