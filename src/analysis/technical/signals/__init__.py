"""
Signals Module
Trading signal generation with filters and risk management.
"""

from .entry_signals import (
    generate_entry_signal,
    calculate_valuation_signal,
    calculate_rsi_signal,
    detect_macd_divergence,
    calculate_entry_recommendation,
    calculate_risk_management,
    calculate_volume_signal,
    calculate_peg_signal,
    INDUSTRY_PE_AVERAGE,
    OVERVALUATION_THRESHOLD,
    UNDERVALUATION_THRESHOLD
)

from .universal_signal import generate_universal_signal

__all__ = [
    'generate_entry_signal',
    'calculate_valuation_signal',
    'calculate_rsi_signal',
    'detect_macd_divergence',
    'calculate_entry_recommendation',
    'calculate_risk_management',
    'calculate_volume_signal',
    'calculate_peg_signal',
    'INDUSTRY_PE_AVERAGE',
    'OVERVALUATION_THRESHOLD',
    'UNDERVALUATION_THRESHOLD',
    # NEW: Universal signal generator
    'generate_universal_signal'
]

