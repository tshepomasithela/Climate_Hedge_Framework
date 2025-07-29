# Climate Transition Risk Dynamic Hedging Framework
__version__ = "1.0.0"
__author__ = "Quantitative Finance Team"

from .risk_factors import ClimateRiskFactorEngine
from .risk_scoring import ClimateTransitionRiskScorer
from .portfolio_analysis import PortfolioClimateRiskAnalyzer
from .hedging_optimizer import DynamicHedgingOptimizer
from .data_pipeline import ClimateDataPipeline

__all__ = [
    'ClimateRiskFactorEngine',
    'ClimateTransitionRiskScorer', 
    'PortfolioClimateRiskAnalyzer',
    'DynamicHedgingOptimizer',
    'ClimateDataPipeline'
]
