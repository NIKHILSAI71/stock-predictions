
import unittest
import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from quantitative.backtesting import (
    Trade, backtest_strategy, calculate_performance_metrics,
    sma_crossover_strategy
)

class TestBacktesting(unittest.TestCase):
    def setUp(self):
        # Create sample data suitable for SMA crossover
        # A sine wave or trend to generate crossings
        dates = pd.date_range(start='2020-01-01', periods=100)
        x = np.linspace(0, 10, 100)
        
        # Fast MA (short period) tracks price closely
        # Slow MA (long period) tracks loosely
        # Create price that oscillates around a trend
        self.price = pd.Series(100 + 10*np.sin(x) + x, index=dates, name="Close")
        self.data = pd.DataFrame({'Close': self.price}, index=dates)

        # Create a simple signal generator
        self.simple_signal = pd.Series(0, index=dates)
        # Buy at index 10, Sell at index 20
        self.simple_signal.iloc[10] = 1
        self.simple_signal.iloc[20] = -1

    def test_trade_class(self):
        t = Trade(
            entry_date=pd.Timestamp('2020-01-01'),
            entry_price=100.0,
            shares=10
        )
        t.exit_date = pd.Timestamp('2020-01-02')
        t.exit_price = 110.0
        
        self.assertEqual(t.pnl, (110 - 100) * 10) # 100 profit
        self.assertAlmostEqual(t.return_pct, 10.0) # 10%

    def test_backtest_strategy(self):
        # Mock signal generator
        def mock_signals(data):
            return self.simple_signal
            
        res = backtest_strategy(
            self.data, 
            mock_signals,
            initial_capital=10000,
            commission=0.0,
            slippage=0.0
        )
        
        self.assertIn('total_trades', res['metrics'])
        self.assertIn('equity_curve', res)
        # We made 1 trade: Buy @ 10, Sell @ 20.
        # Price at 10 and 20.
        p_entry = self.price.iloc[10]
        p_exit = self.price.iloc[20]
        # PnL roughly (p_exit - p_entry) * shares
        # p_exit > p_entry?
        # Check correctness of logic execution, not exact values unless hardcoded.
        self.assertEqual(res['metrics']['total_trades'], 1)

    def test_calculate_performance_metrics(self):
        # Create outcomes
        t1 = Trade(pd.Timestamp('2020-01-01'), 100, pd.Timestamp('2020-01-02'), 110, shares=1) # +10
        t2 = Trade(pd.Timestamp('2020-01-03'), 100, pd.Timestamp('2020-01-04'), 95, shares=1) # -5
        
        trades = [t1, t2]
        returns = pd.Series([0.1, -0.05]) # Simple returns
        
        metrics = calculate_performance_metrics(returns, 100, 105, trades)
        self.assertEqual(metrics['winning_trades'], 1)
        self.assertEqual(metrics['losing_trades'], 1)
        self.assertGreater(metrics['profit_factor'], 1.0) # 10 / 5 = 2.0

    def test_sma_crossover_strategy(self):
        # fast=5, slow=10
        signals = sma_crossover_strategy(self.data, fast=5, slow=10)
        # Should have values 0, 1, -1
        unique_vals = signals.unique()
        for v in unique_vals:
            self.assertIn(v, [0, 1, -1])

if __name__ == '__main__':
    unittest.main()
