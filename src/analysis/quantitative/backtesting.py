"""
Backtesting Engine Module
Strategy backtesting and performance metrics
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Callable, List, Optional
from dataclasses import dataclass


@dataclass
class Trade:
    """Represents a single trade."""
    entry_date: pd.Timestamp
    entry_price: float
    exit_date: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    position_type: str = 'long'  # 'long' or 'short'
    shares: float = 1.0
    
    @property
    def pnl(self) -> Optional[float]:
        if self.exit_price is None:
            return None
        if self.position_type == 'long':
            return (self.exit_price - self.entry_price) * self.shares
        else:
            return (self.entry_price - self.exit_price) * self.shares
    
    @property
    def return_pct(self) -> Optional[float]:
        if self.exit_price is None:
            return None
        if self.position_type == 'long':
            return (self.exit_price - self.entry_price) / self.entry_price * 100
        else:
            return (self.entry_price - self.exit_price) / self.entry_price * 100


def backtest_strategy(
    data: pd.DataFrame,
    signal_generator: Callable[[pd.DataFrame], pd.Series],
    initial_capital: float = 10000,
    position_size: float = 1.0,
    commission: float = 0.001,
    slippage: float = 0.0005
) -> Dict[str, Any]:
    """
    Backtest a trading strategy.
    
    Args:
        data: DataFrame with OHLCV data
        signal_generator: Function that generates buy/sell signals
            Returns Series with 1 for buy, -1 for sell, 0 for hold
        initial_capital: Starting capital
        position_size: Fraction of capital per trade
        commission: Commission per trade (decimal)
        slippage: Slippage per trade (decimal)
    
    Returns:
        Dictionary with backtest results
    """
    # Generate signals
    signals = signal_generator(data)
    
    # Initialize tracking variables
    capital = initial_capital
    position = 0
    shares = 0
    entry_price = 0
    trades: List[Trade] = []
    equity_curve = [initial_capital]
    
    for i in range(1, len(data)):
        date = data.index[i]
        price = data['Close'].iloc[i]
        signal = signals.iloc[i] if i < len(signals) else 0
        
        # Apply slippage
        buy_price = price * (1 + slippage)
        sell_price = price * (1 - slippage)
        
        if signal == 1 and position == 0:  # Buy signal, no position
            # Enter long
            trade_capital = capital * position_size
            shares = trade_capital / buy_price
            commission_cost = trade_capital * commission
            capital -= (trade_capital + commission_cost)
            position = 1
            entry_price = buy_price
            trades.append(Trade(
                entry_date=date,
                entry_price=entry_price,
                position_type='long',
                shares=shares
            ))
            
        elif signal == -1 and position == 1:  # Sell signal, long position
            # Exit long
            trade_value = shares * sell_price
            commission_cost = trade_value * commission
            capital += (trade_value - commission_cost)
            trades[-1].exit_date = date
            trades[-1].exit_price = sell_price
            position = 0
            shares = 0
        
        # Calculate current equity
        current_equity = capital + (shares * price if position else 0)
        equity_curve.append(current_equity)
    
    # Close any open position
    if position == 1:
        final_price = data['Close'].iloc[-1]
        trade_value = shares * final_price
        capital += trade_value
        trades[-1].exit_date = data.index[-1]
        trades[-1].exit_price = final_price
    
    # Calculate metrics
    equity_series = pd.Series(equity_curve, index=data.index[:len(equity_curve)])
    returns = equity_series.pct_change().dropna()
    
    metrics = calculate_performance_metrics(
        returns, 
        initial_capital, 
        capital,
        trades
    )
    
    return {
        'initial_capital': initial_capital,
        'final_capital': round(capital, 2),
        'equity_curve': equity_series.tolist(),
        'dates': [d.strftime('%Y-%m-%d') for d in equity_series.index],
        'trades': len(trades),
        'metrics': metrics
    }


def calculate_performance_metrics(
    returns: pd.Series,
    initial_capital: float,
    final_capital: float,
    trades: List[Trade]
) -> Dict[str, Any]:
    """Calculate comprehensive performance metrics."""
    
    total_return = (final_capital - initial_capital) / initial_capital * 100
    
    # Sharpe Ratio (assuming 252 trading days, 0% risk-free rate)
    if len(returns) > 0 and returns.std() != 0:
        sharpe = (returns.mean() / returns.std()) * np.sqrt(252)
    else:
        sharpe = 0
    
    # Sortino Ratio (downside deviation)
    downside_returns = returns[returns < 0]
    if len(downside_returns) > 0 and downside_returns.std() != 0:
        sortino = (returns.mean() / downside_returns.std()) * np.sqrt(252)
    else:
        sortino = 0
    
    # Maximum Drawdown
    cumulative = (1 + returns).cumprod()
    rolling_max = cumulative.expanding().max()
    drawdowns = (cumulative - rolling_max) / rolling_max
    max_drawdown = drawdowns.min() * 100
    
    # Calmar Ratio
    years = len(returns) / 252
    annual_return = (1 + total_return/100) ** (1/years) - 1 if years > 0 else 0
    calmar = abs(annual_return / max_drawdown * 100) if max_drawdown != 0 else 0
    
    # Win Rate
    winning_trades = [t for t in trades if t.pnl and t.pnl > 0]
    losing_trades = [t for t in trades if t.pnl and t.pnl <= 0]
    win_rate = len(winning_trades) / len(trades) * 100 if trades else 0
    
    # Average Win/Loss
    avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
    avg_loss = abs(np.mean([t.pnl for t in losing_trades])) if losing_trades else 0
    
    # Profit Factor
    gross_profit = sum(t.pnl for t in winning_trades) if winning_trades else 0
    gross_loss = abs(sum(t.pnl for t in losing_trades)) if losing_trades else 1
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
    
    return {
        'total_return_pct': round(total_return, 2),
        'annual_return_pct': round(annual_return * 100, 2),
        'sharpe_ratio': round(sharpe, 2),
        'sortino_ratio': round(sortino, 2),
        'max_drawdown_pct': round(max_drawdown, 2),
        'calmar_ratio': round(calmar, 2),
        'win_rate_pct': round(win_rate, 2),
        'profit_factor': round(profit_factor, 2),
        'avg_win': round(avg_win, 2),
        'avg_loss': round(avg_loss, 2),
        'total_trades': len(trades),
        'winning_trades': len(winning_trades),
        'losing_trades': len(losing_trades)
    }


# Pre-built strategy signal generators
def sma_crossover_strategy(data: pd.DataFrame, fast: int = 20, slow: int = 50) -> pd.Series:
    """SMA crossover strategy."""
    fast_sma = data['Close'].rolling(fast).mean()
    slow_sma = data['Close'].rolling(slow).mean()
    
    signals = pd.Series(0, index=data.index)
    signals[fast_sma > slow_sma] = 1  # Bullish
    signals[fast_sma < slow_sma] = -1  # Bearish
    
    # Only signal on crossovers
    signal_diff = signals.diff()
    result = pd.Series(0, index=data.index)
    result[signal_diff == 2] = 1  # Buy
    result[signal_diff == -2] = -1  # Sell
    
    return result


def rsi_strategy(data: pd.DataFrame, period: int = 14, oversold: int = 30, overbought: int = 70) -> pd.Series:
    """RSI overbought/oversold strategy."""
    from src.analysis.technical import rsi
    
    rsi_values = rsi(data, period)
    
    signals = pd.Series(0, index=data.index)
    signals[rsi_values < oversold] = 1  # Buy when oversold
    signals[rsi_values > overbought] = -1  # Sell when overbought
    
    return signals


def macd_strategy(data: pd.DataFrame) -> pd.Series:
    """MACD crossover strategy."""
    from src.analysis.technical import macd
    
    macd_data = macd(data)
    
    signals = pd.Series(0, index=data.index)
    signals[macd_data['histogram'] > 0] = 1
    signals[macd_data['histogram'] < 0] = -1
    
    # Signal on crossover
    signal_diff = signals.diff()
    result = pd.Series(0, index=data.index)
    result[signal_diff == 2] = 1
    result[signal_diff == -2] = -1
    
    return result
