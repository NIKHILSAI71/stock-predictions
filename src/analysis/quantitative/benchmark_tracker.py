"""
Benchmark Tracker Module

Compares model predictions against buy-and-hold benchmark to calculate alpha.
Helps evaluate if the prediction system adds value over passive investing.

Data Storage Structure:
    data/
    └── benchmarks/
        ├── trade_history.json    # All recorded trades
        └── performance/
            ├── summary.json      # Overall performance summary
            └── by_symbol/        # Performance by symbol
                ├── AAPL.json
                └── ...
"""

from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import json
import os


class BenchmarkTracker:
    """
    Track model performance vs buy-and-hold benchmark.
    
    Features:
    - Human-readable JSON with metadata
    - Organized folder structure
    - Per-symbol performance tracking
    
    Calculates:
    - Model return vs benchmark return
    - Alpha (excess return)
    - Win rate vs benchmark
    - Risk-adjusted metrics
    """
    
    def __init__(self, storage_dir: str = "data/benchmarks"):
        self.storage_dir = storage_dir
        self.history_file = os.path.join(storage_dir, "trade_history.json")
        self.performance_dir = os.path.join(storage_dir, "performance")
        self.by_symbol_dir = os.path.join(self.performance_dir, "by_symbol")
        self.history: List[Dict] = []
        
        # Ensure directories exist
        os.makedirs(self.storage_dir, exist_ok=True)
        os.makedirs(self.performance_dir, exist_ok=True)
        os.makedirs(self.by_symbol_dir, exist_ok=True)
        
        self._load()
    
    def _load(self):
        """Load benchmark history from file."""
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Handle both old format (list) and new format (dict with metadata)
                    if isinstance(data, dict) and 'data' in data:
                        self.history = data['data']
                    else:
                        self.history = data
            except Exception as e:
                print(f"[BenchmarkTracker] Error loading {self.history_file}: {e}")
                self.history = []
    
    def _save(self):
        """Save benchmark history to file with human-readable formatting."""
        try:
            # Calculate quick stats for metadata
            total_trades = len(self.history)
            wins = sum(1 for h in self.history if h.get('was_correct'))
            beats = sum(1 for h in self.history if h.get('beat_benchmark'))
            
            output = {
                "_metadata": {
                    "description": "Benchmark comparison trade history",
                    "last_updated": datetime.now().isoformat(),
                    "total_trades": total_trades,
                    "win_rate": f"{(wins/total_trades*100):.1f}%" if total_trades > 0 else "N/A",
                    "beat_benchmark_rate": f"{(beats/total_trades*100):.1f}%" if total_trades > 0 else "N/A",
                    "version": "2.0"
                },
                "data": self.history[-500:]  # Keep last 500 entries
            }
            
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(output, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            print(f"[BenchmarkTracker] Error saving {self.history_file}: {e}")
    
    def record_trade(
        self,
        symbol: str,
        entry_price: float,
        exit_price: float,
        model_signal: str,  # "BUY", "SELL", "HOLD"
        confidence: float,
        holding_days: int
    ):
        """
        Record a completed trade for benchmark comparison.
        
        Args:
            symbol: Stock ticker
            entry_price: Price when signal was generated
            exit_price: Price at evaluation time
            model_signal: What the model recommended
            confidence: Model confidence (0-100)
            holding_days: Days between entry and exit
        """
        # Calculate actual return
        actual_return_pct = ((exit_price - entry_price) / entry_price) * 100
        
        # Determine if model was correct
        if model_signal == "BUY":
            model_return = actual_return_pct
            was_correct = actual_return_pct > 0
        elif model_signal == "SELL":
            model_return = -actual_return_pct  # Inverse for shorts
            was_correct = actual_return_pct < 0
        else:  # HOLD
            model_return = 0  # No action taken
            was_correct = abs(actual_return_pct) < 2  # Correct if price stable
        
        # Buy-and-hold return (always long)
        benchmark_return = actual_return_pct
        
        # Calculate alpha
        alpha = model_return - benchmark_return
        
        record = {
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "model_signal": model_signal,
            "confidence": confidence,
            "holding_days": holding_days,
            "actual_return_pct": round(actual_return_pct, 2),
            "model_return_pct": round(model_return, 2),
            "benchmark_return_pct": round(benchmark_return, 2),
            "alpha": round(alpha, 2),
            "was_correct": was_correct,
            "beat_benchmark": model_return > benchmark_return
        }
        
        self.history.append(record)
        self._save()
        
        return record
    
    def get_performance_summary(
        self,
        lookback_days: int = 30,
        symbol: str = None
    ) -> Dict[str, Any]:
        """
        Get performance summary comparing model vs benchmark.
        
        Args:
            lookback_days: How far back to analyze
            symbol: Optional filter by symbol
            
        Returns:
            Performance metrics and comparison
        """
        cutoff = datetime.now() - timedelta(days=lookback_days)
        
        relevant = [
            h for h in self.history
            if datetime.fromisoformat(h["timestamp"]) > cutoff
            and (symbol is None or h["symbol"] == symbol)
        ]
        
        if not relevant:
            return {
                "total_trades": 0,
                "message": "No trades recorded in this period",
                "model_total_return": 0,
                "benchmark_total_return": 0,
                "alpha": 0
            }
        
        # Calculate aggregate metrics
        total_model_return = sum(h["model_return_pct"] for h in relevant)
        total_benchmark_return = sum(h["benchmark_return_pct"] for h in relevant)
        total_alpha = sum(h["alpha"] for h in relevant)
        
        correct_count = sum(1 for h in relevant if h["was_correct"])
        beat_benchmark_count = sum(1 for h in relevant if h["beat_benchmark"])
        
        # Calculate averages
        avg_model_return = total_model_return / len(relevant)
        avg_benchmark_return = total_benchmark_return / len(relevant)
        avg_alpha = total_alpha / len(relevant)
        
        # Win rates
        win_rate = (correct_count / len(relevant)) * 100
        beat_benchmark_rate = (beat_benchmark_count / len(relevant)) * 100
        
        # Calculate by confidence tier
        high_conf = [h for h in relevant if h["confidence"] >= 70]
        low_conf = [h for h in relevant if h["confidence"] < 70]
        
        high_conf_win_rate = (sum(1 for h in high_conf if h["was_correct"]) / len(high_conf) * 100) if high_conf else 0
        low_conf_win_rate = (sum(1 for h in low_conf if h["was_correct"]) / len(low_conf) * 100) if low_conf else 0
        
        return {
            "period": f"Last {lookback_days} days",
            "total_trades": len(relevant),
            
            # Aggregate returns
            "model_total_return_pct": round(total_model_return, 2),
            "benchmark_total_return_pct": round(total_benchmark_return, 2),
            "total_alpha_pct": round(total_alpha, 2),
            
            # Averages per trade
            "avg_model_return_pct": round(avg_model_return, 2),
            "avg_benchmark_return_pct": round(avg_benchmark_return, 2),
            "avg_alpha_per_trade": round(avg_alpha, 2),
            
            # Win rates
            "win_rate_pct": round(win_rate, 1),
            "beat_benchmark_rate_pct": round(beat_benchmark_rate, 1),
            
            # Confidence analysis
            "high_confidence_win_rate": round(high_conf_win_rate, 1),
            "low_confidence_win_rate": round(low_conf_win_rate, 1),
            "high_confidence_trades": len(high_conf),
            "low_confidence_trades": len(low_conf),
            
            # Assessment
            "outperforming_benchmark": total_alpha > 0,
            "assessment": self._generate_assessment(total_alpha, win_rate, beat_benchmark_rate)
        }
    
    def _generate_assessment(self, alpha: float, win_rate: float, beat_rate: float) -> str:
        """Generate human-readable assessment."""
        if alpha > 5 and win_rate > 65:
            return "EXCELLENT - Model significantly outperforming benchmark"
        elif alpha > 0 and win_rate > 55:
            return "GOOD - Model adding value over buy-and-hold"
        elif alpha > -2 and win_rate > 50:
            return "MARGINAL - Model performing near benchmark"
        elif alpha > -5:
            return "UNDERPERFORMING - Consider adjusting model parameters"
        else:
            return "POOR - Model underperforming benchmark significantly"


def get_benchmark_comparison(symbol: str = None, days: int = 30) -> Dict[str, Any]:
    """
    Quick function to get benchmark comparison.
    
    Returns performance metrics comparing model to buy-and-hold.
    """
    tracker = BenchmarkTracker()
    return tracker.get_performance_summary(lookback_days=days, symbol=symbol)
