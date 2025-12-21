"""
Adaptive Learning System
Tracks prediction accuracy and adjusts signal weights dynamically.

Data Storage Structure:
    data/
    ├── predictions/
    │   ├── signals/           # Individual prediction logs by symbol
    │   │   ├── AAPL.json
    │   │   ├── GOOGL.json
    │   │   └── ...
    │   └── all_predictions.json  # Master log (last 500)
    │
    └── models/
        └── performance/
            ├── by_sector.json    # Performance by sector
            └── summary.json      # Overall model performance
"""

import json
import os
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np


class AdaptiveLearningSystem:
    """
    Adaptive Learning System for tracking and improving prediction accuracy.
    
    Stores data in organized, human-readable folders:
    - Predictions are saved per-symbol for easy tracking
    - Model performance is tracked by sector
    - All JSON files use pretty printing for readability
    """
    
    def __init__(self, data_dir: str = "data"):
        # Organized folder structure
        self.data_dir = data_dir
        self.predictions_dir = os.path.join(data_dir, "predictions", "signals")
        self.models_dir = os.path.join(data_dir, "models", "performance")
        
        # File paths
        self.master_predictions_file = os.path.join(data_dir, "predictions", "all_predictions.json")
        self.sector_performance_file = os.path.join(self.models_dir, "by_sector.json")
        self.summary_file = os.path.join(self.models_dir, "summary.json")
        
        self.min_confidence_threshold = 0.6
        
        # Ensure directories exist
        os.makedirs(self.predictions_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Load existing data
        self.predictions = self._load_json(self.master_predictions_file, default=[])
        self.performance = self._load_json(self.sector_performance_file, default={})

    def _load_json(self, filepath: str, default: Any) -> Any:
        """Load JSON file with error handling."""
        try:
            if os.path.exists(filepath):
                with open(filepath, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return default
        except Exception as e:
            print(f"[AdaptiveLearning] Error loading {filepath}: {e}")
            return default

    def _save_json(self, filepath: str, data: Any, description: str = ""):
        """
        Save data to JSON file with human-readable formatting.
        
        Features:
        - Pretty printing with 2-space indent
        - Sorted keys for consistency
        - UTF-8 encoding for special characters
        - Metadata header for context
        """
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Wrap data with metadata for readability
            output = {
                "_metadata": {
                    "description": description or "Adaptive Learning System Data",
                    "last_updated": datetime.now().isoformat(),
                    "version": "2.0"
                },
                "data": data
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(output, f, indent=2, sort_keys=False, ensure_ascii=False)
        except Exception as e:
            print(f"[AdaptiveLearning] Error saving {filepath}: {e}")

    def record_prediction(self, symbol: str, signal: Dict[str, Any], current_price: float, classification: Dict[str, Any]):
        """
        Log a prediction for future verification.
        """
        prediction_entry = {
            "id": f"{symbol}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "date": datetime.now().isoformat(),
            "symbol": symbol,
            "entry_price": current_price,
            "signal_type": signal.get("signal_type", "HOLD"),
            "confidence": signal.get("confidence", 0),
            "predicted_direction": signal.get("recommendation", "HOLD"),
            "sector": classification.get("sector", "Unknown"),
            "stock_type": classification.get("stock_type", "Unknown"),
            "status": "PENDING", # PENDING, VERIFIED, EXPIRED
            "verification_date": None,
            "outcome": None
        }
        
        # Only record significant signals
        if prediction_entry["signal_type"] != "HOLD":
            self.predictions.append(prediction_entry)
            
            # Save to master predictions file (keep last 500)
            self._save_json(
                self.master_predictions_file, 
                self.predictions[-500:],
                description=f"Master prediction log - All symbols"
            )
            
            # Also save to symbol-specific file for easy tracking
            symbol_file = os.path.join(self.predictions_dir, f"{symbol}.json")
            symbol_predictions = [p for p in self.predictions if p['symbol'] == symbol][-100:]
            self._save_json(
                symbol_file,
                symbol_predictions,
                description=f"Prediction history for {symbol}"
            )

    def verify_predictions(self, get_price_func):
        """
        Check past predictions against current prices.
        Args:
            get_price_func: Function(symbol) -> current_price
        """
        updates_made = False
        now = datetime.now()
        
        for pred in self.predictions:
            if pred["status"] != "PENDING":
                continue
                
            pred_date = datetime.fromisoformat(pred["date"])
            
            # Verify after 5 days (trading week)
            # For demo purposes, we might check sooner or simulates
            if (now - pred_date).days >= 5:
                try:
                    current_price = get_price_func(pred["symbol"])
                    if not current_price:
                        continue
                        
                    start_price = pred["entry_price"]
                    change_pct = (current_price - start_price) / start_price
                    
                    is_correct = False
                    if pred["predicted_direction"] == "BUY" and change_pct > 0.02: # +2% gain
                        is_correct = True
                    elif pred["predicted_direction"] == "SELL" and change_pct < -0.02: # -2% drop (profit)
                        is_correct = True
                        
                    pred["status"] = "VERIFIED"
                    pred["verification_date"] = now.isoformat()
                    pred["outcome_return"] = change_pct
                    pred["is_correct"] = is_correct
                    
                    self._update_performance_stats(pred)
                    updates_made = True
                    
                except Exception as e:
                    print(f"Error verifying prediction for {pred['symbol']}: {e}")
        
        if updates_made:
            # Save master predictions
            self._save_json(
                self.master_predictions_file, 
                self.predictions[-500:],
                description="Master prediction log - All symbols"
            )
            
            # Save sector performance
            self._save_json(
                self.sector_performance_file, 
                self.performance,
                description="Model performance statistics by sector"
            )
            
            # Update symbol-specific files for verified predictions
            updated_symbols = set(p['symbol'] for p in self.predictions if p.get('status') == 'VERIFIED')
            for sym in updated_symbols:
                symbol_file = os.path.join(self.predictions_dir, f"{sym}.json")
                symbol_predictions = [p for p in self.predictions if p['symbol'] == sym][-100:]
                self._save_json(
                    symbol_file,
                    symbol_predictions,
                    description=f"Prediction history for {sym}"
                )

    def _update_performance_stats(self, prediction):
        """Update aggregate stats logic."""
        sector = prediction["sector"]
        if sector not in self.performance:
            self.performance[sector] = {"total": 0, "correct": 0, "accuracy": 0.5, "weight": 1.0}
            
        stats = self.performance[sector]
        stats["total"] += 1
        if prediction["is_correct"]:
            stats["correct"] += 1
            
        # Update accuracy
        stats["accuracy"] = stats["correct"] / stats["total"]
        
        # Adjust weight: Base 1.0. 
        # If accuracy > 60%, boost weight. 
        # If accuracy < 40%, lower weight.
        # Simple sigmoid-like scaling
        acc = stats["accuracy"]
        if acc > 0.6:
            stats["weight"] = min(1.5, 1.0 + (acc - 0.6) * 2) # Max 1.5x boost
        elif acc < 0.4:
            stats["weight"] = max(0.5, 1.0 - (0.4 - acc) * 2) # Min 0.5x penalty
        else:
            stats["weight"] = 1.0

    def get_sector_weight(self, sector: str) -> float:
        """Get the adaptive weight for a sector based on past performance."""
        return self.performance.get(sector, {}).get("weight", 1.0)

    def get_performance_summary(self) -> Dict[str, Any]:
        return self.performance
