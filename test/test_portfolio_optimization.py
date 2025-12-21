
import unittest
import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from quantitative.portfolio_optimization import (
    mean_variance_optimization, min_variance_portfolio, risk_parity,
    kelly_criterion, black_litterman_returns, portfolio_rebalance_signals,
    calculate_portfolio_returns, calculate_portfolio_volatility
)

class TestPortfolioOptimization(unittest.TestCase):
    def setUp(self):
        # Create sample returns for 3 assets
        np.random.seed(42)
        dates = pd.date_range(start='2020-01-01', periods=252)
        self.returns = pd.DataFrame({
            'AAPL': np.random.normal(0.001, 0.02, 252),
            'GOOGL': np.random.normal(0.0008, 0.015, 252),
            'MSFT': np.random.normal(0.0009, 0.018, 252)
        }, index=dates)

    def test_calculate_portfolio_returns(self):
        weights = np.array([0.5, 0.5])
        mean_returns = np.array([0.1, 0.2])
        # Expected: 0.5*0.1 + 0.5*0.2 = 0.05 + 0.1 = 0.15
        ret = calculate_portfolio_returns(weights, mean_returns)
        self.assertAlmostEqual(ret, 0.15)

    def test_calculate_portfolio_volatility(self):
        weights = np.array([0.5, 0.5])
        # Covariance matrix: [[0.04, 0], [0, 0.04]] (uncorrelated 20% vol)
        cov_matrix = np.array([[0.04, 0], [0, 0.04]])
        # Variance = w'Cw. 
        # [0.5, 0.5] @ [[0.04, 0], [0, 0.04]] = [0.02, 0.02]
        # [0.02, 0.02] @ [0.5, 0.5] = 0.01 + 0.01 = 0.02
        # Vol = sqrt(0.02) = 0.1414...
        vol = calculate_portfolio_volatility(weights, cov_matrix)
        self.assertAlmostEqual(vol, np.sqrt(0.02))

    def test_mean_variance_optimization(self):
        result = mean_variance_optimization(self.returns, num_portfolios=10)
        self.assertIn('max_sharpe_portfolio', result)
        self.assertIn('min_volatility_portfolio', result)
        self.assertIn('efficient_frontier', result)
        
        # Check that weights sum to roughly 1
        weights = result['max_sharpe_portfolio']['weights']
        self.assertAlmostEqual(sum(weights.values()), 1.0, places=2)

    def test_min_variance_portfolio(self):
        result = min_variance_portfolio(self.returns)
        self.assertIn('weights', result)
        self.assertIn('expected_return', result)
        
        weights = result['weights']
        self.assertAlmostEqual(sum(weights.values()), 1.0, places=2)

    def test_risk_parity(self):
        result = risk_parity(self.returns)
        self.assertIn('weights', result)
        self.assertIn('risk_contribution', result)
        
        # Risk contribution should be roughly equal (since it's inverse vol, only approx true risk parity)
        # The implementation is simplified (Inverse Volatility), so contributions won't be exactly equal unless correlation is 0.
        # But let's check weights sum to 1.
        weights = result['weights']
        self.assertAlmostEqual(sum(weights.values()), 1.0, places=2)

    def test_kelly_criterion(self):
        # 55% win rate, 1:1 payout
        res = kelly_criterion(0.55, 100, 100)
        # Kelly = 0.55 - (0.45/1) = 0.10 -> 10%
        self.assertAlmostEqual(res['kelly_percentage'], 10.0)
        
        # Loss cannot be zero
        res_err = kelly_criterion(0.55, 100, 0)
        self.assertIn('error', res_err)

    def test_black_litterman_returns(self):
        market_caps = {'AAPL': 1000, 'GOOGL': 800, 'MSFT': 900}
        # Normalize weights
        total_mc = sum(market_caps.values())
        weights = {k: v/total_mc for k, v in market_caps.items()}
        
        cov = self.returns.cov() * 252
        
        views = [{'asset': 'AAPL', 'view': 10, 'confidence': 0.8}] # View: AAPL return 10%
        
        res = black_litterman_returns(weights, views, cov)
        self.assertIn('AAPL', res)
        self.assertIn('GOOGL', res)

    def test_portfolio_rebalance_signals(self):
        current = {'AAPL': 0.6, 'GOOGL': 0.4}
        target = {'AAPL': 0.5, 'GOOGL': 0.5}
        
        # Drift: AAPL +0.1 (10%), GOOGL -0.1 (-10%). Threshold 0.05.
        # Should rebalance.
        res = portfolio_rebalance_signals(current, target, threshold=0.05)
        self.assertTrue(res['needs_rebalance'])
        self.assertIn('SELL', res['actions']['AAPL'])
        self.assertIn('BUY', res['actions']['GOOGL'])

if __name__ == '__main__':
    unittest.main()
