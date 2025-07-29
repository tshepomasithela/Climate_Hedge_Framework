"""
Climate Risk Factor Engine
Identifies and constructs climate risk factors using PCA, ICA, and ML techniques
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.decomposition import PCA, FastICA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class ClimateRiskFactorEngine:
    """
    Advanced climate risk factor identification and construction engine
    """
    
    def __init__(self, n_factors: int = 5, method: str = 'pca'):
        self.n_factors = n_factors
        self.method = method  # 'pca', 'ica', 'ml_hybrid'
        self.scaler = StandardScaler()
        self.factor_model = None
        self.factor_loadings_ = None
        self.factor_scores_ = None
        self.explained_variance_ratio_ = None
        self.factor_names_ = []
        
    def extract_climate_factors(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Extract climate risk factors from integrated dataset
        """
        # Select climate-relevant features
        climate_features = [
            'carbon_intensity', 'green_revenue_pct', 'stranded_asset_exposure',
            'esg_score', 'carbon_price_usd', 'policy_stringency', 
            'innovation_momentum', 'climate_sentiment', 'green_transition_score',
            'policy_momentum', 'return_volatility_30d'
        ]
        
        # Prepare feature matrix
        feature_data = data[climate_features].dropna()
        
        # Standardize features
        scaled_features = self.scaler.fit_transform(feature_data)
        scaled_df = pd.DataFrame(scaled_features, columns=climate_features, 
                               index=feature_data.index)
        
        if self.method == 'pca':
            factors, loadings = self._extract_pca_factors(scaled_df)
        elif self.method == 'ica':
            factors, loadings = self._extract_ica_factors(scaled_df)
        elif self.method == 'ml_hybrid':
            factors, loadings = self._extract_ml_hybrid_factors(scaled_df, data)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        # Create factor DataFrames
        factor_df = pd.DataFrame(factors, index=feature_data.index, 
                               columns=self.factor_names_)
        loading_df = pd.DataFrame(loadings, index=climate_features, 
                                columns=self.factor_names_)
        
        return factor_df, loading_df
    
    def _extract_pca_factors(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Extract factors using Principal Component Analysis"""
        self.factor_model = PCA(n_components=self.n_factors, random_state=42)
        factors = self.factor_model.fit_transform(data)
        loadings = self.factor_model.components_.T
        
        self.explained_variance_ratio_ = self.factor_model.explained_variance_ratio_
        
        # Name factors based on dominant loadings
        self.factor_names_ = []
        for i in range(self.n_factors):
            dominant_feature_idx = np.argmax(np.abs(loadings[:, i]))
            dominant_feature = data.columns[dominant_feature_idx]
            self.factor_names_.append(f"Climate_Factor_{i+1}_{dominant_feature}")
        
        return factors, loadings
    
    def _extract_ica_factors(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Extract factors using Independent Component Analysis"""
        self.factor_model = FastICA(n_components=self.n_factors, random_state=42, 
                                  max_iter=1000)
        factors = self.factor_model.fit_transform(data)
        loadings = self.factor_model.mixing_.T
        
        # Calculate pseudo explained variance for ICA
        factor_vars = np.var(factors, axis=0)
        self.explained_variance_ratio_ = factor_vars / np.sum(factor_vars)
        
        # Name ICA factors
        self.factor_names_ = []
        for i in range(self.n_factors):
            dominant_feature_idx = np.argmax(np.abs(loadings[:, i]))
            dominant_feature = data.columns[dominant_feature_idx]
            self.factor_names_.append(f"ICA_Factor_{i+1}_{dominant_feature}")
        
        return factors, loadings
    
    def _extract_ml_hybrid_factors(self, data: pd.DataFrame, 
                                 full_data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Extract factors using ML-guided hybrid approach"""
        # First, use PCA for initial factor extraction
        pca = PCA(n_components=min(self.n_factors * 2, len(data.columns)), random_state=42)
        pca_factors = pca.fit_transform(data)
        
        # Use clustering to identify factor regimes
        kmeans = KMeans(n_clusters=3, random_state=42)  # Bull, Bear, Neutral regimes
        regimes = kmeans.fit_predict(pca_factors[:, :3])
        
        # Train random forest to predict returns using factors
        if 'daily_return' in full_data.columns:
            returns = full_data.loc[data.index, 'daily_return'].fillna(0)
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            rf.fit(pca_factors, returns)
            
            # Select most important factors based on feature importance
            feature_importance = rf.feature_importances_
            important_factors_idx = np.argsort(feature_importance)[-self.n_factors:]
            
            factors = pca_factors[:, important_factors_idx]
            loadings = pca.components_[important_factors_idx, :].T
        else:
            # Fallback to standard PCA
            factors = pca_factors[:, :self.n_factors]
            loadings = pca.components_[:self.n_factors, :].T
        
        self.explained_variance_ratio_ = pca.explained_variance_ratio_[:self.n_factors]
        
        # Enhanced factor naming for ML hybrid
        self.factor_names_ = [
            "Carbon_Risk_Factor",
            "Policy_Transition_Factor", 
            "Green_Innovation_Factor",
            "Market_Sentiment_Factor",
            "ESG_Momentum_Factor"
        ][:self.n_factors]
        
        return factors, loadings
    
    def calculate_factor_exposures(self, data: pd.DataFrame, 
                                 factor_scores: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate factor exposures (betas) for each asset
        """
        exposures = {}
        
        # Get unique tickers
        tickers = data['ticker'].unique()
        
        for ticker in tickers:
            ticker_data = data[data['ticker'] == ticker].copy()
            
            if len(ticker_data) < 30:  # Need minimum observations
                continue
                
            # Align dates
            common_dates = ticker_data.index.intersection(factor_scores.index)
            if len(common_dates) < 20:
                continue
                
            ticker_returns = ticker_data.loc[common_dates, 'daily_return']
            ticker_factors = factor_scores.loc[common_dates]
            
            # Calculate factor exposures using regression
            exposures[ticker] = self._calculate_factor_betas(ticker_returns, ticker_factors)
        
        return pd.DataFrame(exposures).T
    
    def _calculate_factor_betas(self, returns: pd.Series, 
                              factors: pd.DataFrame) -> Dict[str, float]:
        """Calculate factor betas using linear regression"""
        betas = {}
        
        for factor_name in factors.columns:
            # Remove NaN values
            valid_idx = ~(returns.isna() | factors[factor_name].isna())
            
            if valid_idx.sum() < 10:  # Need minimum observations
                betas[factor_name] = 0.0
                continue
            
            clean_returns = returns[valid_idx]
            clean_factor = factors.loc[valid_idx, factor_name]
            
            # Calculate beta using linear regression
            try:
                slope, intercept, r_value, p_value, std_err = stats.linregress(
                    clean_factor, clean_returns
                )
                betas[factor_name] = slope
            except:
                betas[factor_name] = 0.0
        
        return betas
    
    def construct_risk_model(self, data: pd.DataFrame) -> Dict:
        """
        Construct comprehensive climate risk model
        """
        # Extract factors
        factor_scores, factor_loadings = self.extract_climate_factors(data)
        
        # Calculate factor exposures
        factor_exposures = self.calculate_factor_exposures(data, factor_scores)
        
        # Calculate factor covariance matrix
        factor_cov = factor_scores.cov()
        
        # Calculate idiosyncratic risk
        idiosyncratic_risk = self._calculate_idiosyncratic_risk(data, 
                                                               factor_scores, 
                                                               factor_exposures)
        
        # Factor risk attribution
        risk_attribution = self._calculate_risk_attribution(factor_exposures, 
                                                           factor_cov)
        
        risk_model = {
            'factor_scores': factor_scores,
            'factor_loadings': factor_loadings,
            'factor_exposures': factor_exposures,
            'factor_covariance': factor_cov,
            'idiosyncratic_risk': idiosyncratic_risk,
            'risk_attribution': risk_attribution,
            'explained_variance': self.explained_variance_ratio_,
            'factor_names': self.factor_names_
        }
        
        return risk_model
    
    def _calculate_idiosyncratic_risk(self, data: pd.DataFrame,
                                    factor_scores: pd.DataFrame,
                                    factor_exposures: pd.DataFrame) -> pd.Series:
        """Calculate idiosyncratic (stock-specific) risk"""
        idiosyncratic_risk = {}
        
        for ticker in factor_exposures.index:
            ticker_data = data[data['ticker'] == ticker].copy()
            
            if len(ticker_data) < 30:
                continue
            
            returns = ticker_data['daily_return'].dropna()
            
            # Calculate systematic component
            common_dates = returns.index.intersection(factor_scores.index)
            if len(common_dates) < 20:
                continue
            
            systematic_returns = pd.Series(0, index=common_dates)
            for factor_name in factor_scores.columns:
                if factor_name in factor_exposures.columns:
                    beta = factor_exposures.loc[ticker, factor_name]
                    systematic_returns += beta * factor_scores.loc[common_dates, factor_name]
            
            # Idiosyncratic returns = total returns - systematic returns
            idiosyncratic_returns = returns.loc[common_dates] - systematic_returns
            
            # Idiosyncratic risk = volatility of idiosyncratic returns
            idiosyncratic_risk[ticker] = idiosyncratic_returns.std() * np.sqrt(252)  # Annualized
        
        return pd.Series(idiosyncratic_risk)
    
    def _calculate_risk_attribution(self, factor_exposures: pd.DataFrame,
                                  factor_cov: pd.DataFrame) -> pd.DataFrame:
        """Calculate risk attribution by factor"""
        risk_attribution = {}
        
        for ticker in factor_exposures.index:
            exposures = factor_exposures.loc[ticker]
            
            # Factor risk contribution = beta^T * Cov * beta
            factor_risk_contributions = {}
            for factor in factor_cov.columns:
                if factor in exposures.index:
                    beta = exposures[factor]
                    factor_var = factor_cov.loc[factor, factor]
                    factor_risk_contributions[factor] = (beta ** 2) * factor_var
            
            risk_attribution[ticker] = factor_risk_contributions
        
        return pd.DataFrame(risk_attribution).T.fillna(0)
    
    def predict_factor_movements(self, historical_factors: pd.DataFrame, 
                               horizon_days: int = 30) -> pd.DataFrame:
        """
        Predict future factor movements using time series models
        """
        from statsmodels.tsa.arima.model import ARIMA
        
        predictions = {}
        
        for factor in historical_factors.columns:
            factor_series = historical_factors[factor].dropna()
            
            if len(factor_series) < 50:
                # Not enough data, use simple random walk
                last_value = factor_series.iloc[-1]
                predictions[factor] = [last_value] * horizon_days
                continue
            
            try:
                # Fit ARIMA model
                model = ARIMA(factor_series, order=(1, 1, 1))
                fitted_model = model.fit()
                
                # Forecast
                forecast = fitted_model.forecast(steps=horizon_days)
                predictions[factor] = forecast.tolist()
                
            except:
                # Fallback to random walk
                last_value = factor_series.iloc[-1]
                predictions[factor] = [last_value] * horizon_days
        
        # Create prediction DataFrame
        future_dates = pd.date_range(
            start=historical_factors.index[-1] + pd.Timedelta(days=1),
            periods=horizon_days,
            freq='D'
        )
        
        return pd.DataFrame(predictions, index=future_dates)


# Example usage and testing
if __name__ == "__main__":
    from data_pipeline import ClimateDataPipeline
    
    # Create sample data
    pipeline = ClimateDataPipeline()
    tickers = ['TSLA', 'XOM', 'AAPL', 'NEE']
    data = pipeline.create_integrated_dataset(tickers, '2023-01-01', '2024-01-01')
    
    # Initialize risk factor engine
    factor_engine = ClimateRiskFactorEngine(n_factors=5, method='ml_hybrid')
    
    # Build risk model
    risk_model = factor_engine.construct_risk_model(data)
    
    print("Climate Risk Model Summary:")
    print(f"Factors: {risk_model['factor_names']}")
    print(f"Explained Variance: {risk_model['explained_variance']}")
    print(f"Factor Loadings Shape: {risk_model['factor_loadings'].shape}")
    print(f"Factor Exposures Shape: {risk_model['factor_exposures'].shape}")
    
    # Predict factor movements
    factor_predictions = factor_engine.predict_factor_movements(
        risk_model['factor_scores'], horizon_days=10
    )
    print(f"\nFactor Predictions Shape: {factor_predictions.shape}")
    print(factor_predictions.head())
