"""
Advanced analysis module for comprehensive feature analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import shap
from scipy import stats
import statsmodels.api as sm
from statsmodels.tsa.stattools import grangercausalitytests
import networkx as nx

class AdvancedAnalyzer:
    def __init__(self, X: pd.DataFrame, y: pd.Series):
        """
        Initialize the advanced analyzer.
        
        Args:
            X (pd.DataFrame): Feature matrix
            y (pd.Series): Target variable
        """
        self.X = X
        self.y = y
        self.feature_names = X.columns.tolist()
        
    def analyze_temporal_patterns(self) -> Dict:
        """
        Analyze temporal patterns in features and target.
        
        Returns:
            Dict: Dictionary containing temporal analysis results
        """
        results = {}
        
        for col in self.feature_names + ['target']:
            series = self.X[col] if col in self.X else self.y
            
            # Seasonal decomposition
            decomposition = sm.tsa.seasonal_decompose(
                series, 
                period=12,  # Assuming monthly data
                extrapolate_trend='freq'
            )
            
            # Calculate temporal statistics
            results[col] = {
                'trend': decomposition.trend.values,
                'seasonal': decomposition.seasonal.values,
                'resid': decomposition.resid.values,
                'acf': sm.tsa.acf(series, nlags=12),  # 12 months of autocorrelation
                'pacf': sm.tsa.pacf(series, nlags=12),
                'seasonal_strength': np.var(decomposition.seasonal) / np.var(series)
            }
            
        return results
    
    def analyze_granger_causality(self, max_lag: int = 6) -> Dict:
        """
        Perform Granger causality tests between features and target.
        
        Args:
            max_lag (int): Maximum number of lags to test
            
        Returns:
            Dict: Dictionary containing Granger causality results
        """
        results = {}
        
        # Prepare data for Granger causality test
        data = pd.concat([self.X, self.y], axis=1)
        
        for feature in self.feature_names:
            # Test if feature Granger-causes target
            gc_test = grangercausalitytests(
                data[[feature, 'target']], 
                maxlag=max_lag,
                verbose=False
            )
            
            # Extract p-values for each lag
            p_values = {
                lag: round(test[0]['ssr_chi2test'][1], 4)
                for lag, test in gc_test.items()
            }
            
            results[feature] = {
                'p_values': p_values,
                'min_p_value': min(p_values.values()),
                'optimal_lag': min(
                    p_values.items(), 
                    key=lambda x: x[1]
                )[0]
            }
            
        return results
    
    def analyze_nonlinear_relationships(self) -> Dict:
        """
        Analyze nonlinear relationships between features and target.
        
        Returns:
            Dict: Dictionary containing nonlinearity analysis results
        """
        results = {}
        
        for feature in self.feature_names:
            # Calculate mutual information
            mi_score = self._calculate_mutual_information(
                self.X[feature], 
                self.y
            )
            
            # Test for monotonicity
            spearman_corr = stats.spearmanr(self.X[feature], self.y)
            
            # Test for quadratic relationship
            quad_fit = np.polyfit(self.X[feature], self.y, deg=2)
            quad_r2 = self._calculate_polynomial_r2(
                self.X[feature], 
                self.y, 
                quad_fit
            )
            
            results[feature] = {
                'mutual_information': mi_score,
                'spearman_correlation': spearman_corr.correlation,
                'spearman_pvalue': spearman_corr.pvalue,
                'quadratic_r2': quad_r2,
                'is_nonlinear': quad_r2 > spearman_corr.correlation ** 2
            }
            
        return results
    
    def _calculate_mutual_information(self, x: pd.Series, y: pd.Series) -> float:
        """
        Calculate mutual information between two variables.
        
        Args:
            x (pd.Series): First variable
            y (pd.Series): Second variable
            
        Returns:
            float: Mutual information score
        """
        from sklearn.feature_selection import mutual_info_regression
        return mutual_info_regression(
            x.values.reshape(-1, 1),
            y
        )[0]
    
    def _calculate_polynomial_r2(self, x: pd.Series, y: pd.Series, coeffs: np.ndarray) -> float:
        """
        Calculate R² for polynomial fit.
        
        Args:
            x (pd.Series): Input variable
            y (pd.Series): Target variable
            coeffs (np.ndarray): Polynomial coefficients
            
        Returns:
            float: R² score
        """
        y_pred = np.polyval(coeffs, x)
        return 1 - (np.sum((y - y_pred) ** 2) / np.sum((y - y.mean()) ** 2))
    
    def analyze_feature_interactions_network(self, threshold: float = 0.1) -> nx.Graph:
        """
        Create a network representation of feature interactions.
        
        Args:
            threshold (float): Minimum interaction strength to include
            
        Returns:
            nx.Graph: NetworkX graph of feature interactions
        """
        # Create graph
        G = nx.Graph()
        
        # Add nodes (features)
        for feature in self.feature_names:
            G.add_node(feature)
        
        # Add edges (interactions)
        for i, f1 in enumerate(self.feature_names):
            for j, f2 in enumerate(self.feature_names[i+1:], i+1):
                # Calculate interaction strength
                interaction = self._calculate_interaction_strength(f1, f2)
                
                if abs(interaction) > threshold:
                    G.add_edge(f1, f2, weight=abs(interaction))
        
        return G
    
    def _calculate_interaction_strength(self, feature1: str, feature2: str) -> float:
        """
        Calculate interaction strength between two features.
        
        Args:
            feature1 (str): First feature name
            feature2 (str): Second feature name
            
        Returns:
            float: Interaction strength
        """
        # Create interaction term
        interaction_term = self.X[feature1] * self.X[feature2]
        
        # Fit linear model with individual features and interaction
        X_interaction = pd.DataFrame({
            'f1': self.X[feature1],
            'f2': self.X[feature2],
            'interaction': interaction_term
        })
        
        model = sm.OLS(self.y, sm.add_constant(X_interaction))
        results = model.fit()
        
        # Return the t-statistic for the interaction term
        return results.tvalues['interaction'] 