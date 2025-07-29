"""
Climate Data Pipeline
Handles ingestion, cleaning, and preprocessing of climate-related financial data
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
import requests
import logging
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore')

class ClimateDataPipeline:
    """
    Comprehensive data pipeline for climate transition risk analysis
    Integrates ESG data, policy trackers, tech innovation, sentiment, and market data
    """
    
    def __init__(self, config: Dict[str, str] = None):
        self.config = config or {}
        self.logger = self._setup_logging()
        self.data_cache = {}
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logger = logging.getLogger('ClimateDataPipeline')
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger
    
    def ingest_esg_data(self, tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        """
        Simulate ESG data ingestion (in production, connect to providers like MSCI, Sustainalytics)
        """
        self.logger.info(f"Ingesting ESG data for {len(tickers)} tickers")
        
        # Simulate ESG metrics for demonstration
        np.random.seed(42)  # For reproducible results
        dates = pd.date_range(start_date, end_date, freq='M')
        
        esg_data = []
        for ticker in tickers:
            for date in dates:
                # Simulate realistic ESG metrics
                carbon_intensity = np.random.lognormal(3, 1)  # Tons CO2e per $M revenue
                green_revenue_pct = np.random.beta(2, 8) * 100  # % of revenue from green activities
                climate_targets = np.random.choice([0, 1], p=[0.3, 0.7])  # Has climate targets
                stranded_asset_exposure = np.random.beta(3, 7) * 100  # % assets at risk
                
                esg_data.append({
                    'ticker': ticker,
                    'date': date,
                    'carbon_intensity': carbon_intensity,
                    'green_revenue_pct': green_revenue_pct,
                    'climate_targets': climate_targets,
                    'stranded_asset_exposure': stranded_asset_exposure,
                    'esg_score': np.random.normal(50, 15)  # ESG score 0-100
                })
        
        return pd.DataFrame(esg_data)
    
    def ingest_policy_data(self, regions: List[str] = None) -> pd.DataFrame:
        """
        Simulate policy tracker data (carbon pricing, regulations, etc.)
        """
        regions = regions or ['US', 'EU', 'China', 'Global']
        self.logger.info(f"Ingesting policy data for regions: {regions}")
        
        # Simulate policy developments
        np.random.seed(123)
        dates = pd.date_range('2020-01-01', datetime.now(), freq='W')
        
        policy_data = []
        for region in regions:
            for date in dates:
                # Simulate carbon price trajectory with volatility
                base_price = {'US': 15, 'EU': 80, 'China': 8, 'Global': 25}[region]
                carbon_price = base_price * (1 + np.random.normal(0, 0.02))
                
                # Policy stringency index (0-10)
                policy_stringency = np.random.gamma(4, 1.5)
                
                # Regulatory announcements (binary events)
                reg_announcement = np.random.choice([0, 1], p=[0.95, 0.05])
                
                policy_data.append({
                    'region': region,
                    'date': date,
                    'carbon_price_usd': max(0, carbon_price),
                    'policy_stringency': min(10, policy_stringency),
                    'regulatory_announcement': reg_announcement
                })
        
        return pd.DataFrame(policy_data)
    
    def ingest_tech_innovation_data(self, sectors: List[str] = None) -> pd.DataFrame:
        """
        Simulate technology innovation indicators (patents, investment flows)
        """
        sectors = sectors or ['renewable_energy', 'battery_storage', 'carbon_capture', 'ev_tech']
        self.logger.info(f"Ingesting tech innovation data for sectors: {sectors}")
        
        # Simulate innovation metrics
        np.random.seed(456)
        dates = pd.date_range('2020-01-01', datetime.now(), freq='M')
        
        tech_data = []
        for sector in sectors:
            trend = np.linspace(1, 2, len(dates))  # Growing trend
            for i, date in enumerate(dates):
                # Patent filings with trend
                patents = np.random.poisson(100 * trend[i])
                
                # Investment flows ($ millions)
                investment = np.random.lognormal(8, 1) * trend[i]
                
                # Innovation momentum index
                momentum = trend[i] + np.random.normal(0, 0.1)
                
                tech_data.append({
                    'sector': sector,
                    'date': date,
                    'patent_filings': patents,
                    'investment_millions_usd': investment,
                    'innovation_momentum': momentum
                })
        
        return pd.DataFrame(tech_data)
    
    def ingest_sentiment_data(self, tickers: List[str], lookback_days: int = 30) -> pd.DataFrame:
        """
        Simulate climate-related news sentiment analysis
        """
        self.logger.info(f"Ingesting sentiment data for {len(tickers)} tickers")
        
        # Simulate sentiment scores
        np.random.seed(789)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)
        dates = pd.date_range(start_date, end_date, freq='D')
        
        sentiment_data = []
        for ticker in tickers:
            for date in dates:
                # Climate sentiment (-1 to 1, where 1 is very positive for climate transition)
                climate_sentiment = np.random.normal(0, 0.3)
                climate_sentiment = np.clip(climate_sentiment, -1, 1)
                
                # News volume (number of climate-related articles)
                news_volume = np.random.poisson(5)
                
                sentiment_data.append({
                    'ticker': ticker,
                    'date': date,
                    'climate_sentiment': climate_sentiment,
                    'news_volume': news_volume
                })
        
        return pd.DataFrame(sentiment_data)
    
    def ingest_market_data(self, tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        """
        Simulate market data ingestion (prices, returns, volatility)
        """
        self.logger.info(f"Ingesting market data for {len(tickers)} tickers")
        
        # Simulate price data
        np.random.seed(999)
        dates = pd.date_range(start_date, end_date, freq='D')
        
        market_data = []
        for ticker in tickers:
            price = 100  # Starting price
            for date in dates:
                # Random walk with drift
                daily_return = np.random.normal(0.0005, 0.02)  # ~12.5% annual return, 32% vol
                price *= (1 + daily_return)
                
                volume = np.random.lognormal(10, 1)  # Trading volume
                
                market_data.append({
                    'ticker': ticker,
                    'date': date,
                    'price': price,
                    'daily_return': daily_return,
                    'volume': volume
                })
        
        return pd.DataFrame(market_data)
    
    def create_integrated_dataset(self, tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        """
        Create comprehensive integrated dataset combining all data sources
        """
        self.logger.info("Creating integrated climate risk dataset")
        
        # Ingest all data sources in parallel
        with ThreadPoolExecutor(max_workers=4) as executor:
            esg_future = executor.submit(self.ingest_esg_data, tickers, start_date, end_date)
            policy_future = executor.submit(self.ingest_policy_data)
            tech_future = executor.submit(self.ingest_tech_innovation_data)
            sentiment_future = executor.submit(self.ingest_sentiment_data, tickers)
            market_future = executor.submit(self.ingest_market_data, tickers, start_date, end_date)
            
            # Collect results
            esg_data = esg_future.result()
            policy_data = policy_future.result()
            tech_data = tech_future.result()
            sentiment_data = sentiment_future.result()
            market_data = market_future.result()
        
        # Merge datasets
        # Start with market data as base
        integrated_data = market_data.copy()
        
        # Add ESG data
        integrated_data = integrated_data.merge(
            esg_data, on=['ticker', 'date'], how='left'
        )
        
        # Add sentiment data
        integrated_data = integrated_data.merge(
            sentiment_data, on=['ticker', 'date'], how='left'
        )
        
        # Aggregate policy data by date (take mean across regions)
        policy_agg = policy_data.groupby('date').agg({
            'carbon_price_usd': 'mean',
            'policy_stringency': 'mean',
            'regulatory_announcement': 'max'  # Any announcement triggers
        }).reset_index()
        
        integrated_data = integrated_data.merge(
            policy_agg, on='date', how='left'
        )
        
        # Aggregate tech data by date
        tech_agg = tech_data.groupby('date').agg({
            'patent_filings': 'sum',
            'investment_millions_usd': 'sum',
            'innovation_momentum': 'mean'
        }).reset_index()
        
        integrated_data = integrated_data.merge(
            tech_agg, on='date', how='left'
        )
        
        # Forward fill missing values
        integrated_data = integrated_data.sort_values(['ticker', 'date'])
        integrated_data = integrated_data.groupby('ticker').fillna(method='ffill')
        
        # Calculate additional derived features
        integrated_data = self._calculate_derived_features(integrated_data)
        
        self.logger.info(f"Integrated dataset created with {len(integrated_data)} records")
        return integrated_data
    
    def _calculate_derived_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate derived features for climate risk analysis
        """
        data = data.copy()
        
        # Rolling statistics
        data['return_volatility_30d'] = data.groupby('ticker')['daily_return'].rolling(30).std().reset_index(0, drop=True)
        data['sentiment_ma_7d'] = data.groupby('ticker')['climate_sentiment'].rolling(7).mean().reset_index(0, drop=True)
        
        # Climate transition momentum
        data['policy_momentum'] = data['policy_stringency'] * data['carbon_price_usd'] / 100
        
        # Green transition score (composite)
        data['green_transition_score'] = (
            data['green_revenue_pct'] / 100 * 0.3 +
            data['climate_targets'] * 0.2 +
            (100 - data['stranded_asset_exposure']) / 100 * 0.3 +
            (data['climate_sentiment'] + 1) / 2 * 0.2  # Normalize sentiment to 0-1
        )
        
        return data
    
    def validate_data_quality(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        Validate data quality and return quality metrics
        """
        quality_metrics = {}
        
        # Missing data percentage
        quality_metrics['missing_data_pct'] = (data.isnull().sum().sum() / 
                                             (len(data) * len(data.columns))) * 100
        
        # Data freshness (days since last update)
        latest_date = data['date'].max()
        quality_metrics['data_age_days'] = (datetime.now() - latest_date).days
        
        # Outlier detection (using IQR method on numerical columns)
        numerical_cols = data.select_dtypes(include=[np.number]).columns
        outlier_count = 0
        for col in numerical_cols:
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((data[col] < (Q1 - 1.5 * IQR)) | 
                       (data[col] > (Q3 + 1.5 * IQR))).sum()
            outlier_count += outliers
        
        quality_metrics['outlier_pct'] = (outlier_count / len(data)) * 100
        
        self.logger.info(f"Data quality metrics: {quality_metrics}")
        return quality_metrics


# Example usage and testing
if __name__ == "__main__":
    # Initialize pipeline
    pipeline = ClimateDataPipeline()
    
    # Sample tickers representing different climate risk profiles
    tickers = ['TSLA', 'XOM', 'AAPL', 'NEE', 'COAL', 'ENPH']
    
    # Create integrated dataset
    integrated_data = pipeline.create_integrated_dataset(
        tickers=tickers,
        start_date='2023-01-01', 
        end_date='2024-01-01'
    )
    
    # Validate data quality
    quality_metrics = pipeline.validate_data_quality(integrated_data)
    
    print(f"Dataset shape: {integrated_data.shape}")
    print(f"Quality metrics: {quality_metrics}")
    print("\nSample data:")
    print(integrated_data.head())
