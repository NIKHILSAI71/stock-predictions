"""
Backtesting Module
Comprehensive backtesting engine for trading strategies with performance metrics.
"""

from dataclasses import dataclass, field
from typing import List, Callable, Dict, Any, Optional
import pandas as pd
import numpy as np
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """Represents a single trade with entry, exit, and performance metrics."""

    entry_date: pd.Timestamp
    entry_price: float
    shares: int
    signal_type: str = "LONG"  # LONG or SHORT
    exit_date: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None

    @property
    def pnl(self) -> float:
        """Calculate profit/loss in dollars."""
        if not self.exit_price:
            return 0.0

        if self.signal_type == "LONG":
            return (self.exit_price - self.entry_price) * self.shares
        else:  # SHORT
            return (self.entry_price - self.exit_price) * self.shares

    @property
    def return_pct(self) -> float:
        """Calculate return as percentage."""
        if not self.exit_price or self.entry_price == 0:
            return 0.0
        return (self.pnl / (self.entry_price * self.shares)) * 100

    @property
    def duration_days(self) -> int:
        """Calculate trade duration in days."""
        if not self.exit_date:
            return 0
        return (self.exit_date - self.entry_date).days


def backtest_strategy(
    data: pd.DataFrame,
    signal_generator: Callable[[pd.DataFrame], pd.Series],
    initial_capital: float = 10000.0,
    commission: float = 0.001,  # 0.1%
    slippage: float = 0.0005,  # 0.05%
    position_size: float = 0.95,  # Use 95% of capital
) -> Dict[str, Any]:
    """
    Backtest a trading strategy with realistic costs.

    Args:
        data: DataFrame with OHLCV data (must have DatetimeIndex and 'Close' column)
        signal_generator: Function that returns signals (1=BUY, -1=SELL, 0=HOLD)
        initial_capital: Starting capital in dollars
        commission: Commission as fraction (0.001 = 0.1%)
        slippage: Slippage as fraction (0.0005 = 0.05%)
        position_size: Fraction of capital to use per trade (0-1)

    Returns:
        Dictionary with performance metrics, trade history, and equity curve
    """
    logger.info(
        f"Starting backtest: ${initial_capital:.2f} initial capital, "
        f"commission={commission*100:.2f}%, slippage={slippage*100:.3f}%"
    )

    # Generate signals
    signals = signal_generator(data)

    # Initialize portfolio state
    cash = initial_capital
    position = 0  # Number of shares held
    trades: List[Trade] = []
    equity_curve = []
    daily_returns = []

    for i in range(len(data)):
        date = data.index[i]
        price = data['Close'].iloc[i]
        signal = signals.iloc[i]

        # Calculate portfolio value
        portfolio_value = cash + (position * price)
        equity_curve.append(portfolio_value)

        # Calculate daily return
        if i > 0:
            prev_value = equity_curve[i - 1]
            daily_return = (portfolio_value - prev_value) / prev_value
            daily_returns.append(daily_return)

        # Entry signal (BUY) - only if no position
        if signal == 1 and position == 0:
            shares_to_buy = int(
                (cash * position_size) / (price * (1 + slippage + commission))
            )

            if shares_to_buy > 0:
                cost = shares_to_buy * price * (1 + slippage + commission)

                if cost <= cash:
                    cash -= cost
                    position = shares_to_buy

                    trades.append(
                        Trade(
                            entry_date=date,
                            entry_price=price,
                            shares=shares_to_buy,
                            signal_type="LONG",
                        )
                    )
                    logger.debug(
                        f"BUY: {shares_to_buy} shares @ ${price:.2f} on {date.date()}"
                    )

        # Exit signal (SELL) - only if holding position
        elif signal == -1 and position > 0:
            proceeds = position * price * (1 - slippage - commission)
            cash += proceeds

            # Update last trade
            trades[-1].exit_date = date
            trades[-1].exit_price = price

            logger.debug(
                f"SELL: {position} shares @ ${price:.2f} on {date.date()}, "
                f"P&L: ${trades[-1].pnl:.2f}"
            )

            position = 0

    # Close any open position at end
    if position > 0:
        final_price = data['Close'].iloc[-1]
        proceeds = position * final_price * (1 - commission)
        cash += proceeds

        if trades and not trades[-1].exit_price:
            trades[-1].exit_date = data.index[-1]
            trades[-1].exit_price = final_price
            logger.debug(
                f"CLOSE: {position} shares @ ${final_price:.2f}, "
                f"P&L: ${trades[-1].pnl:.2f}"
            )

        position = 0

    final_value = cash
    total_return = ((final_value - initial_capital) / initial_capital) * 100

    # Calculate performance metrics
    metrics = calculate_performance_metrics(
        trades=trades,
        equity_curve=equity_curve,
        daily_returns=daily_returns,
        initial_capital=initial_capital,
        final_value=final_value,
    )

    logger.info(
        f"Backtest complete: {len(trades)} trades, "
        f"{metrics['win_rate']:.1f}% win rate, "
        f"{total_return:.2f}% total return"
    )

    return {
        "initial_capital": initial_capital,
        "final_value": final_value,
        "total_return_pct": total_return,
        "metrics": metrics,
        "trades": trades,
        "equity_curve": equity_curve,
        "num_trades": len(trades),
    }


def calculate_performance_metrics(
    trades: List[Trade],
    equity_curve: List[float],
    daily_returns: List[float],
    initial_capital: float,
    final_value: float,
) -> Dict[str, Any]:
    """
    Calculate comprehensive performance metrics.

    Args:
        trades: List of Trade objects
        equity_curve: List of portfolio values over time
        daily_returns: List of daily returns
        initial_capital: Starting capital
        final_value: Final portfolio value

    Returns:
        Dictionary of performance metrics
    """
    if len(trades) == 0:
        return {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "win_rate": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "profit_factor": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown_pct": 0.0,
            "avg_trade_duration_days": 0,
        }

    # Categorize trades
    winning_trades = [t for t in trades if t.pnl > 0]
    losing_trades = [t for t in trades if t.pnl < 0]

    # Profit metrics
    total_profit = sum(t.pnl for t in winning_trades)
    total_loss = abs(sum(t.pnl for t in losing_trades))

    profit_factor = (
        total_profit / total_loss if total_loss > 0 else float('inf')
    )

    # Sharpe ratio (annualized, risk-free rate = 0)
    if len(daily_returns) > 1:
        mean_return = np.mean(daily_returns)
        std_return = np.std(daily_returns)
        sharpe = (
            (mean_return / std_return) * np.sqrt(252)
            if std_return > 0
            else 0
        )
    else:
        sharpe = 0.0

    # Max drawdown
    equity_series = pd.Series(equity_curve)
    cummax = equity_series.cummax()
    drawdown = (equity_series - cummax) / cummax
    max_drawdown = drawdown.min() * 100 if len(drawdown) > 0 else 0.0

    # Average trade metrics
    avg_win = total_profit / len(winning_trades) if winning_trades else 0
    avg_loss = total_loss / len(losing_trades) if losing_trades else 0

    return {
        "total_trades": len(trades),
        "winning_trades": len(winning_trades),
        "losing_trades": len(losing_trades),
        "win_rate": (len(winning_trades) / len(trades)) * 100,
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "profit_factor": round(profit_factor, 2) if profit_factor != float('inf') else 999.99,
        "sharpe_ratio": round(sharpe, 2),
        "max_drawdown_pct": round(max_drawdown, 2),
        "avg_trade_duration_days": round(
            sum(t.duration_days for t in trades) / len(trades), 1
        ),
    }


def sma_crossover_strategy(
    data: pd.DataFrame, fast: int = 20, slow: int = 50
) -> pd.Series:
    """
    Simple Moving Average crossover strategy.

    Generates BUY signal when fast SMA crosses above slow SMA.
    Generates SELL signal when fast SMA crosses below slow SMA.

    Args:
        data: DataFrame with 'Close' column
        fast: Fast SMA period (default: 20)
        slow: Slow SMA period (default: 50)

    Returns:
        Series with signals: 1 (BUY), -1 (SELL), 0 (HOLD)
    """
    sma_fast = data['Close'].rolling(fast).mean()
    sma_slow = data['Close'].rolling(slow).mean()

    # Initialize signals as 0 (HOLD)
    signals = pd.Series(0, index=data.index)

    # Set signals based on SMA positions
    signals[sma_fast > sma_slow] = 1  # LONG when fast > slow
    signals[sma_fast < sma_slow] = -1  # EXIT when fast < slow

    # Only signal on crossovers (changes)
    signals = signals.diff().fillna(0)
    signals[signals > 0] = 1  # BUY signal
    signals[signals < 0] = -1  # SELL signal

    return signals


def rsi_strategy(data: pd.DataFrame, period: int = 14, oversold: float = 30, overbought: float = 70) -> pd.Series:
    """
    RSI-based mean reversion strategy.

    Args:
        data: DataFrame with 'Close' column
        period: RSI calculation period
        oversold: Oversold threshold (BUY signal)
        overbought: Overbought threshold (SELL signal)

    Returns:
        Series with signals
    """
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))

    signals = pd.Series(0, index=data.index)
    signals[rsi < oversold] = 1  # BUY when oversold
    signals[rsi > overbought] = -1  # SELL when overbought

    # Only signal on transitions
    signals = signals.diff().fillna(0)
    signals[signals > 0] = 1
    signals[signals < 0] = -1

    return signals
