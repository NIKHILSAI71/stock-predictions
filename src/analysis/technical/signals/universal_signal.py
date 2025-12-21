"""
Universal Signal Generator
Refactored to use Class-Based Strategy Pattern.
Delegates logic to src.quantitative.EnsembleScorer and specialized signals.

Now integrated with:
- Adaptive Learning System (adjusts weights based on historical accuracy)
- ML Aggregator (combines RF, SVM, PCA, momentum predictions)
"""

import pandas as pd
from typing import Dict, Any, Optional

from src.analysis.quantitative.classification import (
    classify_stock, 
    SECTOR_BENCHMARKS,
    DEFAULT_SECTOR_PROFILE
)
from src.analysis.quantitative.ml_models import regime_detection
from src.analysis.quantitative.ml_aggregator import get_ml_ensemble_prediction
from src.analysis.quantitative.ensemble_scorer import get_ensemble_prediction
from src.analysis.quantitative.ml_aggregator import get_ml_ensemble_prediction
from src.adaptive.adaptive_learning import AdaptiveLearningSystem

# Initialize global systems
adaptive_system = AdaptiveLearningSystem()

def generate_universal_signal(
    symbol: str,
    stock_data: pd.DataFrame,
    company_info: Dict[str, Any],
    technical_data: Dict[str, Any],
    risk_tolerance: str = "medium",
    benchmark_data: Optional[pd.DataFrame] = None
) -> Dict[str, Any]:
    """
    Master signal generator.
    Now acts as a Facade/Controller for the Ensemble Architecture.
    """
    
    # Step 1: Classify the stock
    classification = classify_stock(symbol)
    
    # Step 2: Detect market regime
    returns = stock_data['Close'].pct_change().dropna()
    regime = regime_detection(returns, window=min(60, len(returns) - 1))
    
    # Handle regime detection error (fallback)
    if 'error' in regime:
        regime = {
            'current_regime': 'Bull Low Volatility',
            'momentum_weight': 0.5,
            'fundamental_weight': 0.5,
            'position_size_multiplier': 1.0,
            'confidence_adjustment': 1.0,
            'regime_stability': 50.0
        }
    
    # Step 3: Get sector-specific benchmarks (for display/context)
    sector = classification.get('sector', 'Unknown')
    sector_profile = SECTOR_BENCHMARKS.get(sector, DEFAULT_SECTOR_PROFILE)
    
    # Step 4: Calculate sector-adjusted valuation (Helper for context)
    pe_ratio = company_info.get('trailing_pe')
    valuation_analysis = _analyze_valuation_sector_adjusted(
        pe_ratio=pe_ratio,
        sector_profile=sector_profile,
        classification=classification
    )
    
    # Step 5: Get ML ensemble predictions (Aggregator)
    ml_prediction = get_ml_ensemble_prediction(
        stock_data=stock_data,
        lookback_days=252,
        prediction_horizon=5
    )
    
    # Step 6: Get adaptive learning weights
    adaptive_weights = adaptive_system.get_performance_summary()
    sector_adaptive_weight = adaptive_system.get_sector_weight(sector)
    
    # Step 7: DELEGATE TO ENSEMBLE SCORER (Replaced EnsembleCombiner)
    # This uses the new EnsembleScorer to unify ML and Technical/Fundamental signals
    
    # Extract fundamental data for scorer
    fund_data = {
        "pe_ratio": pe_ratio,
        "peg_ratio": company_info.get('peg_ratio'),
        "valuation_status": valuation_analysis.get('status')
    }
    
    # Get unified ensemble prediction
    ensemble_result = get_ensemble_prediction(
        symbol=symbol,
        stock_data=stock_data,
        ml_prediction=ml_prediction,
        technical_signals={"signal": "HOLD", "signal_strength": 50}, # Base baseline, will be enriched
        fundamental_data=fund_data
    )
    
    # Map ensemble result to legacy signal structure for frontend compatibility
    signal = {
        "recommendation": ensemble_result.get("direction", "NEUTRAL").upper(),
        "confidence": ensemble_result.get("confidence", 50),
        "factors": [ensemble_result.get("explanation", "")]
    }
    
    # Step 8: Combine ML prediction with signal (Refined logic)
    if ml_prediction.get('ensemble_direction') != 'NEUTRAL':
        ml_direction = ml_prediction['ensemble_direction']
        signal_direction = signal['recommendation']
        
        # Normalize signal direction
        if "BUY" in signal_direction or "BULLISH" in signal_direction:
            sig_dir_norm = "Bullish"
        elif "SELL" in signal_direction or "BEARISH" in signal_direction:
            sig_dir_norm = "Bearish"
        else:
            sig_dir_norm = "Neutral"
            
        if ml_direction == sig_dir_norm:
            # ML agrees with signal - boost confidence
            signal['confidence'] = min(95, signal.get('confidence', 50) * 1.1)
            signal['factors'].append(f'ML ensemble agrees ({ml_prediction.get("ensemble_confidence", 0):.0f}% conf)')
        elif ml_direction != sig_dir_norm and sig_dir_norm != "Neutral":
            # ML disagrees - reduce confidence
            signal['confidence'] = max(25, signal.get('confidence', 50) * 0.85)
            signal['factors'].append(f'ML ensemble caution: predicts {ml_direction}')
    
    # Step 9: Apply final regime adjustments (Layer 2 Risk Management)
    regime_multiplier = regime.get('confidence_adjustment', 1.0)
    position_multiplier = regime.get('position_size_multiplier', 1.0)
    
    # Apply adaptive sector weight
    signal['confidence'] = round(signal.get('confidence', 50) * regime_multiplier * sector_adaptive_weight, 1)
    
    # Step 10: Generate risk-adjusted position sizing
    risk_adjustments = _calculate_risk_adjusted_sizing(
        classification=classification,
        regime=regime,
        risk_tolerance=risk_tolerance,
        volatility=classification.get('volatility_30d_pct', 30)
    )
    
    # Apply model specific multiplier
    curr_pos_mult = signal.get('position_size_multiplier', 1.0)
    risk_adjustments['recommended_position_pct'] = round(risk_adjustments['recommended_position_pct'] * curr_pos_mult, 2)

    
    # Step 11: Compile final result
    current_price = stock_data['Close'].iloc[-1]
    
    return {
        # Classification
        'classification': {
            'market_cap_tier': classification.get('market_cap_tier'),
            'sector': classification.get('sector'),
            'industry': classification.get('industry'),
            'stock_type': classification.get('stock_type'), # From classifier
            'volatility_profile': classification.get('volatility_profile'),
            'liquidity_profile': classification.get('liquidity_profile'),
            'is_tradeable': classification.get('liquidity_profile') != 'illiquid',
            'warnings': classification.get('warnings', [])
        },
        
        # Regime context
        'market_regime': {
            'current_regime': regime.get('current_regime'),
            'regime_stability': regime.get('regime_stability'),
            'recommended_action': regime.get('recommended_action'),
            'preferred_sectors': regime.get('preferred_sectors', []),
            'entry_strategy': regime.get('entry_strategy')
        },
        
        # Sector-adjusted valuation
        'valuation': valuation_analysis,
        
        # Signal (From Ensemble)
        'signal': signal,
        
        # ML Prediction (NEW)
        'ml_prediction': {
            'ensemble_direction': ml_prediction.get('ensemble_direction'),
            'ensemble_confidence': ml_prediction.get('ensemble_confidence'),
            'rf_prediction': ml_prediction.get('rf_prediction'),
            'svm_prediction': ml_prediction.get('svm_prediction'),
            'momentum_prediction': ml_prediction.get('momentum_prediction'),
            'regime': ml_prediction.get('regime'),
            'models_agree': ml_prediction.get('models_agree'),
            'feature_importance': ml_prediction.get('feature_importance', {}),
            'cluster_profile': ml_prediction.get('cluster_profile')
        },
        
        # Risk management
        'risk_management': {
            **risk_adjustments,
            'stop_loss_multiplier': regime.get('stop_loss_multiplier', 1.0)
        },
        
        # Analysis weights used
        'analysis_weights': {
            'momentum': regime.get('momentum_weight', 0.5),
            'fundamental': regime.get('fundamental_weight', 0.5),
            'model_used': signal.get('model_used'),
            'adaptive_sector_weight': sector_adaptive_weight
        },
        
        # Current state
        'current_price': round(float(current_price), 2)
    }


def _analyze_valuation_sector_adjusted(
    pe_ratio: Optional[float],
    sector_profile: Dict,
    classification: Dict
) -> Dict[str, Any]:
    """Analyze valuation relative to sector, not a fixed 17.8x benchmark."""
    
    sector_avg_pe = sector_profile['avg_pe']
    pe_range = sector_profile['pe_range']
    
    if pe_ratio is None or pe_ratio <= 0:
        # Handle biotech/loss-making companies
        if classification.get('stock_type') == 'biotech':
            return {
                'pe_ratio': None,
                'sector_avg_pe': sector_avg_pe,
                'pe_applicable': False,
                'verdict': 'P/E not applicable - evaluate by pipeline value',
                'status': 'NOT_APPLICABLE'
            }
        return {
            'pe_ratio': None,
            'sector_avg_pe': sector_avg_pe,
            'pe_applicable': False,
            'verdict': 'No P/E data available',
            'status': 'UNKNOWN'
        }
    
    # Calculate deviation from sector average
    pe_deviation = ((pe_ratio - sector_avg_pe) / sector_avg_pe) * 100
    
    # Determine status based on sector-specific ranges
    min_pe, max_pe = pe_range
    
    if pe_ratio < min_pe:
        status = 'CHEAP'
        verdict = f'P/E of {pe_ratio:.1f}x is below sector range ({min_pe}-{max_pe}x) - potentially undervalued'
    elif pe_ratio <= sector_avg_pe * 1.1:  # Within 10% of average
        status = 'FAIR'
        verdict = f'P/E of {pe_ratio:.1f}x is near sector average of {sector_avg_pe}x - fairly valued'
    elif pe_ratio <= max_pe:
        status = 'ELEVATED'
        verdict = f'P/E of {pe_ratio:.1f}x is {pe_deviation:.1f}% above sector average - slightly expensive'
    else:
        status = 'EXPENSIVE'
        verdict = f'P/E of {pe_ratio:.1f}x is above sector range ({min_pe}-{max_pe}x) - overvalued'
    
    return {
        'pe_ratio': round(pe_ratio, 2),
        'sector_avg_pe': sector_avg_pe,
        'sector_pe_range': pe_range,
        'pe_deviation_pct': round(pe_deviation, 1),
        'pe_applicable': True,
        'status': status,
        'verdict': verdict
    }


def _calculate_risk_adjusted_sizing(
    classification: Dict,
    regime: Dict,
    risk_tolerance: str,
    volatility: float
) -> Dict[str, Any]:
    """
    Calculate risk-adjusted position sizing.
    """
    
    # Base position size by risk tolerance
    base_sizes = {
        'low': 0.02,      # 2% of portfolio
        'medium': 0.05,   # 5% of portfolio
        'high': 0.08      # 8% of portfolio
    }
    base = base_sizes.get(risk_tolerance, 0.05)
    
    # Adjust for volatility
    if volatility > 50:
        vol_mult = 0.5
    elif volatility > 35:
        vol_mult = 0.7
    elif volatility > 25:
        vol_mult = 0.85
    else:
        vol_mult = 1.0
    
    # Adjust for classification
    class_mult = classification.get('confidence_adjustments', {}).get('overall_multiplier', 1.0)
    
    # Adjust for regime
    regime_mult = regime.get('position_size_multiplier', 1.0)
    
    # Final position size
    final_size = base * vol_mult * class_mult * regime_mult
    max_size = 0.10  # Never more than 10%
    
    recommended_size = min(final_size, max_size)
    
    return {
        'base_position_pct': base * 100,
        'volatility_adjustment': vol_mult,
        'classification_adjustment': class_mult,
        'regime_adjustment': regime_mult,
        'recommended_position_pct': round(recommended_size * 100, 2),
        'max_position_pct': max_size * 100,
        'risk_tolerance': risk_tolerance
    }
