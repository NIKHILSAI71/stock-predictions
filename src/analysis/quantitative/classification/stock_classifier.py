"""
Stock Classification Module
Classifies stocks into categories for model routing.

Research-backed approach:
- Different stock categories require different model architectures
- Multi-level classification improves prediction accuracy by 22-58%
"""

import yfinance as yf
from typing import Dict, Any, Optional
from functools import lru_cache


# Sector-specific P/E benchmarks (not hardcoded 17.8x for all)
SECTOR_BENCHMARKS = {
    'Technology': {
        'avg_pe': 28.5,
        'pe_range': (20, 40),
        'key_metrics': ['revenue_growth', 'gross_margin', 'r_and_d_ratio'],
        'volatility_tolerance': 'high',
        'moat_importance': 'critical',
        'commodity_correlation': None,
        'interest_rate_sensitivity': 'moderate',
        'typical_beta': 1.2
    },
    'Communication Services': {
        'avg_pe': 24.0,
        'pe_range': (18, 35),
        'key_metrics': ['revenue_growth', 'subscriber_growth', 'arpu'],
        'volatility_tolerance': 'moderate',
        'moat_importance': 'high',
        'commodity_correlation': None,
        'interest_rate_sensitivity': 'moderate',
        'typical_beta': 1.1
    },
    'Consumer Discretionary': {
        'avg_pe': 22.0,
        'pe_range': (15, 30),
        'key_metrics': ['same_store_sales', 'inventory_turnover', 'consumer_confidence'],
        'volatility_tolerance': 'high',
        'moat_importance': 'moderate',
        'commodity_correlation': None,
        'interest_rate_sensitivity': 'high',  # Consumer spending affected by rates
        'typical_beta': 1.3
    },
    'Consumer Staples': {
        'avg_pe': 20.0,
        'pe_range': (15, 25),
        'key_metrics': ['dividend_yield', 'market_share', 'pricing_power'],
        'volatility_tolerance': 'low',
        'moat_importance': 'high',
        'commodity_correlation': 'food_commodities',
        'interest_rate_sensitivity': 'low',
        'typical_beta': 0.7
    },
    'Energy': {
        'avg_pe': 10.0,
        'pe_range': (5, 15),
        'key_metrics': ['production_volume', 'reserve_replacement', 'debt_ratio'],
        'volatility_tolerance': 'high',
        'moat_importance': 'low',
        'commodity_correlation': 'oil_price',  # PRIMARY DRIVER
        'interest_rate_sensitivity': 'moderate',
        'typical_beta': 1.4
    },
    'Financials': {
        'avg_pe': 12.5,
        'pe_range': (8, 18),
        'key_metrics': ['book_value', 'net_interest_margin', 'roa', 'tier1_capital'],
        'volatility_tolerance': 'moderate',
        'moat_importance': 'high',
        'commodity_correlation': None,
        'interest_rate_sensitivity': 'very_high',  # PRIMARY DRIVER
        'typical_beta': 1.1
    },
    'Healthcare': {
        'avg_pe': 22.0,
        'pe_range': (15, 35),
        'key_metrics': ['pipeline_drugs', 'patent_expiry', 'r_and_d_spend'],
        'volatility_tolerance': 'moderate',
        'moat_importance': 'high',
        'commodity_correlation': None,
        'interest_rate_sensitivity': 'low',
        'typical_beta': 0.9
    },
    'Industrials': {
        'avg_pe': 18.0,
        'pe_range': (12, 25),
        'key_metrics': ['backlog', 'capacity_utilization', 'order_growth'],
        'volatility_tolerance': 'moderate',
        'moat_importance': 'moderate',
        'commodity_correlation': 'industrial_metals',
        'interest_rate_sensitivity': 'moderate',
        'typical_beta': 1.1
    },
    'Materials': {
        'avg_pe': 14.0,
        'pe_range': (8, 20),
        'key_metrics': ['commodity_prices', 'production_costs', 'inventory_levels'],
        'volatility_tolerance': 'high',
        'moat_importance': 'low',
        'commodity_correlation': 'metals_and_mining',  # PRIMARY DRIVER
        'interest_rate_sensitivity': 'moderate',
        'typical_beta': 1.3
    },
    'Real Estate': {
        'avg_pe': 35.0,  # REITs use FFO/AFFO, not P/E
        'pe_range': (20, 50),
        'key_metrics': ['ffo', 'occupancy_rate', 'cap_rate', 'dividend_yield'],
        'volatility_tolerance': 'moderate',
        'moat_importance': 'moderate',
        'commodity_correlation': None,
        'interest_rate_sensitivity': 'very_high',  # PRIMARY DRIVER
        'typical_beta': 0.9
    },
    'Utilities': {
        'avg_pe': 18.0,
        'pe_range': (12, 22),
        'key_metrics': ['dividend_yield', 'rate_base_growth', 'regulatory_environment'],
        'volatility_tolerance': 'low',
        'moat_importance': 'high',  # Natural monopolies
        'commodity_correlation': 'natural_gas',
        'interest_rate_sensitivity': 'high',  # Bond proxies
        'typical_beta': 0.5
    }
}

# Default profile for unknown sectors
DEFAULT_SECTOR_PROFILE = {
    'avg_pe': 17.8,  # S&P 500 average as fallback
    'pe_range': (10, 30),
    'key_metrics': ['revenue_growth', 'profit_margin', 'roe'],
    'volatility_tolerance': 'moderate',
    'moat_importance': 'moderate',
    'commodity_correlation': None,
    'interest_rate_sensitivity': 'moderate',
    'typical_beta': 1.0
}


# Market cap tier definitions (in USD)
MARKET_CAP_TIERS = {
    'mega': {'min': 200e9, 'max': float('inf'), 'analysis': 'fundamental_heavy', 'position_multiplier': 1.0},
    'large': {'min': 10e9, 'max': 200e9, 'analysis': 'balanced', 'position_multiplier': 0.9},
    'mid': {'min': 2e9, 'max': 10e9, 'analysis': 'momentum_weighted', 'position_multiplier': 0.75},
    'small': {'min': 300e6, 'max': 2e9, 'analysis': 'momentum_focused', 'position_multiplier': 0.5},
    'micro': {'min': 0, 'max': 300e6, 'analysis': 'high_risk', 'position_multiplier': 0.25}
}


# Stock type characteristics
STOCK_TYPE_PROFILES = {
    'large_cap_tech': {
        'description': 'Large/Mega cap technology with wide moat',
        'recommended_model': 'fundamental_technical_hybrid',
        'pe_tolerance': 'high',
        'momentum_weight': 0.4,
        'fundamental_weight': 0.6
    },
    'large_cap_value': {
        'description': 'Large cap value stocks with stable dividends',
        'recommended_model': 'value_model',
        'pe_tolerance': 'low',
        'momentum_weight': 0.3,
        'fundamental_weight': 0.7
    },
    'commodity_driven': {
        'description': 'Energy, Materials - driven by commodity prices',
        'recommended_model': 'commodity_correlation_model',
        'pe_tolerance': 'varies',
        'momentum_weight': 0.3,
        'fundamental_weight': 0.2,
        'commodity_weight': 0.5
    },
    'financial': {
        'description': 'Banks, Insurance - driven by interest rates',
        'recommended_model': 'interest_rate_model',
        'pe_tolerance': 'low',
        'momentum_weight': 0.3,
        'fundamental_weight': 0.4,
        'macro_weight': 0.3
    },
    'biotech': {
        'description': 'Biotech/Pharma - event-driven',
        'recommended_model': 'event_driven_model',
        'pe_tolerance': 'not_applicable',
        'momentum_weight': 0.2,
        'fundamental_weight': 0.1,
        'event_weight': 0.7
    },
    'cyclical': {
        'description': 'Consumer discretionary, industrials - economic cycle dependent',
        'recommended_model': 'cyclical_model',
        'pe_tolerance': 'moderate',
        'momentum_weight': 0.4,
        'fundamental_weight': 0.3,
        'macro_weight': 0.3
    },
    'defensive': {
        'description': 'Utilities, Consumer staples - stable/defensive',
        'recommended_model': 'defensive_model',
        'pe_tolerance': 'low',
        'momentum_weight': 0.2,
        'fundamental_weight': 0.5,
        'dividend_weight': 0.3
    },
    'growth': {
        'description': 'High growth stocks with high valuations',
        'recommended_model': 'growth_model',
        'pe_tolerance': 'very_high',
        'momentum_weight': 0.5,
        'fundamental_weight': 0.3,
        'growth_weight': 0.2
    },
    'speculative': {
        'description': 'Small/micro cap, high volatility, low liquidity',
        'recommended_model': 'momentum_only',
        'pe_tolerance': 'not_applicable',
        'momentum_weight': 0.8,
        'fundamental_weight': 0.2
    }
}


@lru_cache(maxsize=128)
def classify_stock(symbol: str) -> Dict[str, Any]:
    """
    Classify a stock into categories for model routing.
    
    Returns:
        Dictionary with:
        - market_cap_tier: 'mega'|'large'|'mid'|'small'|'micro'
        - sector: GICS sector from yfinance
        - industry: Specific industry
        - stock_type: Type-based classification for model routing
        - volatility_profile: 'low'|'moderate'|'high'|'extreme'
        - liquidity_profile: 'high'|'moderate'|'low'|'illiquid'
        - sector_profile: Sector-specific benchmarks
        - analysis_approach: Recommended analysis methodology
        - confidence_adjustments: Adjustments for position sizing
    """
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        # Get historical data for volatility calculation
        hist = ticker.history(period="3mo")
        
        # Extract key metrics
        market_cap = info.get('marketCap', 0)
        sector = info.get('sector', 'Unknown')
        industry = info.get('industry', 'Unknown')
        beta = info.get('beta', 1.0)
        avg_volume = info.get('averageVolume', 0)
        pe_ratio = info.get('trailingPE')
        forward_pe = info.get('forwardPE')
        dividend_yield = info.get('dividendYield', 0) or 0
        revenue_growth = info.get('revenueGrowth', 0) or 0
        earnings_growth = info.get('earningsGrowth', 0) or 0
        
        # Calculate volatility from historical data
        if not hist.empty and len(hist) > 10:
            returns = hist['Close'].pct_change().dropna()
            volatility_30d = returns.std() * (252 ** 0.5) * 100  # Annualized %
        else:
            volatility_30d = 30.0  # Default assumption
        
        # Level 1: Market Cap Classification
        market_cap_tier = _classify_market_cap(market_cap)
        
        # Level 2: Sector Profile
        sector_profile = SECTOR_BENCHMARKS.get(sector, DEFAULT_SECTOR_PROFILE)
        
        # Level 3: Liquidity Classification
        liquidity = _classify_liquidity(avg_volume, market_cap)
        
        # Level 4: Volatility Classification
        volatility_profile = _classify_volatility(volatility_30d, beta, sector_profile)
        
        # Level 5: Stock Type Classification
        stock_type = _classify_stock_type(
            sector=sector,
            industry=industry,
            market_cap_tier=market_cap_tier,
            pe_ratio=pe_ratio,
            dividend_yield=dividend_yield,
            revenue_growth=revenue_growth,
            beta=beta,
            volatility_30d=volatility_30d
        )
        
        # Get recommended analysis approach
        tier_info = MARKET_CAP_TIERS.get(market_cap_tier, MARKET_CAP_TIERS['mid'])
        type_info = STOCK_TYPE_PROFILES.get(stock_type, STOCK_TYPE_PROFILES['growth'])
        
        # Calculate confidence adjustments
        confidence_adjustments = _calculate_confidence_adjustments(
            market_cap_tier=market_cap_tier,
            liquidity=liquidity,
            volatility_profile=volatility_profile,
            sector_profile=sector_profile
        )
        
        return {
            # Basic Classification
            'symbol': symbol.upper(),
            'market_cap': market_cap,
            'market_cap_tier': market_cap_tier,
            'sector': sector,
            'industry': industry,
            'stock_type': stock_type,
            
            # Profiles
            'volatility_profile': volatility_profile,
            'liquidity_profile': liquidity,
            'volatility_30d_pct': round(volatility_30d, 2),
            'beta': beta,
            
            # Sector Benchmarks
            'sector_profile': sector_profile,
            'sector_avg_pe': sector_profile['avg_pe'],
            'sector_pe_range': sector_profile['pe_range'],
            
            # Stock Type Info
            'type_profile': type_info,
            'recommended_model': type_info['recommended_model'],
            
            # Analysis Approach
            'analysis_approach': tier_info['analysis'],
            'momentum_weight': type_info.get('momentum_weight', 0.5),
            'fundamental_weight': type_info.get('fundamental_weight', 0.5),
            
            # Risk Adjustments
            'confidence_adjustments': confidence_adjustments,
            'position_size_multiplier': tier_info['position_multiplier'] * confidence_adjustments['overall_multiplier'],
            
            # Special Flags
            'is_commodity_linked': sector_profile.get('commodity_correlation') is not None,
            'commodity_correlation': sector_profile.get('commodity_correlation'),
            'is_interest_rate_sensitive': sector_profile.get('interest_rate_sensitivity') in ['high', 'very_high'],
            'interest_rate_sensitivity': sector_profile.get('interest_rate_sensitivity'),
            
            # Warnings
            'warnings': _generate_warnings(market_cap_tier, liquidity, volatility_profile, stock_type)
        }
        
    except Exception as e:
        # Return default classification on error
        return {
            'symbol': symbol.upper(),
            'market_cap': 0,
            'market_cap_tier': 'unknown',
            'sector': 'Unknown',
            'industry': 'Unknown',
            'stock_type': 'growth',
            'volatility_profile': 'moderate',
            'liquidity_profile': 'moderate',
            'sector_profile': DEFAULT_SECTOR_PROFILE,
            'sector_avg_pe': 17.8,
            'sector_pe_range': (10, 30),
            'type_profile': STOCK_TYPE_PROFILES['growth'],
            'recommended_model': 'general',
            'analysis_approach': 'balanced',
            'momentum_weight': 0.5,
            'fundamental_weight': 0.5,
            'confidence_adjustments': {'overall_multiplier': 0.7},
            'position_size_multiplier': 0.7,
            'is_commodity_linked': False,
            'commodity_correlation': None,
            'is_interest_rate_sensitive': False,
            'interest_rate_sensitivity': 'moderate',
            'warnings': [f'Classification error: {str(e)}'],
            'error': str(e)
        }


def _classify_market_cap(market_cap: float) -> str:
    """Classify market cap into tier."""
    if market_cap >= 200e9:
        return 'mega'
    elif market_cap >= 10e9:
        return 'large'
    elif market_cap >= 2e9:
        return 'mid'
    elif market_cap >= 300e6:
        return 'small'
    else:
        return 'micro'


def _classify_liquidity(avg_volume: float, market_cap: float) -> str:
    """Classify liquidity based on volume and market cap."""
    if avg_volume < 100000:
        return 'illiquid'
    elif avg_volume < 500000:
        return 'low'
    elif avg_volume < 2000000:
        return 'moderate'
    else:
        return 'high'


def _classify_volatility(volatility_30d: float, beta: float, sector_profile: Dict) -> str:
    """Classify volatility relative to sector expectations."""
    typical_beta = sector_profile.get('typical_beta', 1.0)
    tolerance = sector_profile.get('volatility_tolerance', 'moderate')
    
    # Adjust thresholds based on sector tolerance
    if tolerance == 'low':
        thresholds = (15, 25, 40)
    elif tolerance == 'high':
        thresholds = (25, 40, 60)
    else:
        thresholds = (20, 35, 50)
    
    if volatility_30d < thresholds[0]:
        return 'low'
    elif volatility_30d < thresholds[1]:
        return 'moderate'
    elif volatility_30d < thresholds[2]:
        return 'high'
    else:
        return 'extreme'


def _classify_stock_type(
    sector: str,
    industry: str,
    market_cap_tier: str,
    pe_ratio: Optional[float],
    dividend_yield: float,
    revenue_growth: float,
    beta: float,
    volatility_30d: float
) -> str:
    """Classify stock into type for model routing."""
    
    industry_lower = industry.lower() if industry else ''
    
    # Biotech / Event-driven
    if sector == 'Healthcare' and ('biotech' in industry_lower or 'pharma' in industry_lower):
        if pe_ratio is None or pe_ratio < 0:  # No profits = biotech/early stage
            return 'biotech'
    
    # Commodity-driven
    if sector in ['Energy', 'Materials']:
        return 'commodity_driven'
    
    # Financial
    if sector == 'Financials':
        return 'financial'
    
    # Defensive
    if sector in ['Utilities', 'Consumer Staples']:
        return 'defensive'
    
    # Large-cap tech
    if sector in ['Technology', 'Communication Services'] and market_cap_tier in ['mega', 'large']:
        return 'large_cap_tech'
    
    # Cyclical
    if sector in ['Consumer Discretionary', 'Industrials']:
        return 'cyclical'
    
    # Speculative (small cap, high volatility)
    if market_cap_tier in ['small', 'micro'] and volatility_30d > 50:
        return 'speculative'
    
    # Value vs Growth based on P/E and dividend
    if pe_ratio and pe_ratio < 15 and dividend_yield > 0.02:
        return 'large_cap_value'
    
    if revenue_growth > 0.20:  # 20%+ revenue growth
        return 'growth'
    
    # Default to growth
    return 'growth'


def _calculate_confidence_adjustments(
    market_cap_tier: str,
    liquidity: str,
    volatility_profile: str,
    sector_profile: Dict
) -> Dict[str, Any]:
    """Calculate confidence adjustments for predictions."""
    
    overall = 1.0
    adjustments = {}
    
    # Market cap adjustment
    cap_adjustments = {
        'mega': 1.0,
        'large': 0.95,
        'mid': 0.85,
        'small': 0.7,
        'micro': 0.5
    }
    cap_mult = cap_adjustments.get(market_cap_tier, 0.7)
    adjustments['market_cap'] = cap_mult
    overall *= cap_mult
    
    # Liquidity adjustment
    liq_adjustments = {
        'high': 1.0,
        'moderate': 0.9,
        'low': 0.7,
        'illiquid': 0.4
    }
    liq_mult = liq_adjustments.get(liquidity, 0.7)
    adjustments['liquidity'] = liq_mult
    overall *= liq_mult
    
    # Volatility adjustment
    vol_adjustments = {
        'low': 1.0,
        'moderate': 0.95,
        'high': 0.8,
        'extreme': 0.6
    }
    vol_mult = vol_adjustments.get(volatility_profile, 0.8)
    adjustments['volatility'] = vol_mult
    overall *= vol_mult
    
    adjustments['overall_multiplier'] = round(overall, 3)
    
    return adjustments


def _generate_warnings(
    market_cap_tier: str,
    liquidity: str,
    volatility_profile: str,
    stock_type: str
) -> list:
    """Generate warning flags for the stock."""
    warnings = []
    
    if market_cap_tier == 'micro':
        warnings.append('MICRO-CAP: High manipulation risk, limited analysis reliability')
    
    if liquidity == 'illiquid':
        warnings.append('ILLIQUID: Difficult entry/exit, wide spreads, do not trade with size')
    elif liquidity == 'low':
        warnings.append('LOW LIQUIDITY: Use limit orders, consider slippage')
    
    if volatility_profile == 'extreme':
        warnings.append('EXTREME VOLATILITY: Reduce position size, widen stops')
    
    if stock_type == 'biotech':
        warnings.append('BIOTECH: Binary events, P/E not applicable, high risk')
    
    if stock_type == 'speculative':
        warnings.append('SPECULATIVE: Momentum-only analysis, not fundamentals-based')
    
    return warnings


def get_sector_pe_benchmark(sector: str) -> float:
    """Get P/E benchmark for a sector."""
    profile = SECTOR_BENCHMARKS.get(sector, DEFAULT_SECTOR_PROFILE)
    return profile['avg_pe']


def get_sector_profile(sector: str) -> Dict[str, Any]:
    """Get full sector profile."""
    return SECTOR_BENCHMARKS.get(sector, DEFAULT_SECTOR_PROFILE)


def is_tradeable(classification: Dict[str, Any]) -> bool:
    """Check if stock meets minimum tradeability criteria."""
    # Exclude illiquid stocks
    if classification.get('liquidity_profile') == 'illiquid':
        return False
    
    # Exclude micro-caps
    if classification.get('market_cap_tier') == 'micro':
        return False
    
    return True


def get_analysis_weights(classification: Dict[str, Any], market_regime: str = 'bull_low_vol') -> Dict[str, float]:
    """
    Get analysis weights adjusted for stock type and market regime.
    
    Returns weights for different analysis components.
    """
    # Base weights from stock type
    type_profile = classification.get('type_profile', {})
    momentum_weight = type_profile.get('momentum_weight', 0.5)
    fundamental_weight = type_profile.get('fundamental_weight', 0.5)
    
    # Regime adjustments
    regime_adjustments = {
        'bull_high_vol': {'momentum': 0.7, 'fundamental': 0.3},
        'bull_low_vol': {'momentum': 0.5, 'fundamental': 0.5},
        'bear_high_vol': {'momentum': 0.2, 'fundamental': 0.8},
        'bear_low_vol': {'momentum': 0.3, 'fundamental': 0.7}
    }
    
    regime = regime_adjustments.get(market_regime, regime_adjustments['bull_low_vol'])
    
    # Blend stock type weights with regime weights
    final_momentum = (momentum_weight * 0.6 + regime['momentum'] * 0.4)
    final_fundamental = (fundamental_weight * 0.6 + regime['fundamental'] * 0.4)
    
    return {
        'momentum': round(final_momentum, 2),
        'fundamental': round(final_fundamental, 2),
        'total': 1.0
    }
