"""
Module for advanced temporal analysis of time series data.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, acf, pacf, grangercausalitytests
from scipy import stats
import plotly.graph_objects as go
import plotly.express as px

class TemporalAnalyzer:
    def __init__(self, data: pd.DataFrame, date_column: str = None):
        """
        Initialize the temporal analyzer.
        
        Args:
            data (pd.DataFrame): Time series data
            date_column (str, optional): Name of date column if not index
        """
        self.data = data.copy()
        if date_column and date_column in self.data.columns:
            self.data.set_index(date_column, inplace=True)
        self.features = self.data.columns.tolist()
        self.seasonal_results = {}
        self.stationarity_results = {}
        self.causality_results = {}
        self.nonlinearity_results = {}
        
    def analyze_seasonality(self, period: int = 12) -> Dict:
        """
        Analyze seasonal patterns in time series data.
        
        Args:
            period (int): Number of time steps in seasonal cycle
            
        Returns:
            Dict: Seasonality analysis results
        """
        results = {}
        
        for feature in self.features:
            # Perform seasonal decomposition
            decomposition = seasonal_decompose(
                self.data[feature],
                period=period,
                extrapolate_trend='freq'
            )
            
            # Calculate seasonal strength
            seasonal_strength = np.var(decomposition.seasonal) / np.var(decomposition.seasonal + decomposition.resid)
            
            # Store results
            results[feature] = {
                'trend': decomposition.trend,
                'seasonal': decomposition.seasonal,
                'residual': decomposition.resid,
                'seasonal_strength': seasonal_strength,
                'period': period
            }
            
        self.seasonal_results = results
        return results
    
    def analyze_stationarity(self) -> Dict:
        """
        Test for stationarity using augmented Dickey-Fuller test.
        
        Returns:
            Dict: Stationarity test results
        """
        results = {}
        
        for feature in self.features:
            # Perform ADF test
            adf_test = adfuller(self.data[feature].dropna())
            
            # Calculate rolling statistics
            rolling_mean = self.data[feature].rolling(window=12).mean()
            rolling_std = self.data[feature].rolling(window=12).std()
            
            results[feature] = {
                'adf_statistic': adf_test[0],
                'p_value': adf_test[1],
                'critical_values': adf_test[4],
                'is_stationary': adf_test[1] < 0.05,
                'rolling_mean': rolling_mean,
                'rolling_std': rolling_std
            }
            
        self.stationarity_results = results
        return results
    
    def analyze_granger_causality(self, max_lag: int = 12) -> Dict:
        """
        Analyze Granger causality between features.
        
        Args:
            max_lag (int): Maximum number of lags to test
            
        Returns:
            Dict: Granger causality test results
        """
        results = {}
        
        for feature1 in self.features:
            results[feature1] = {}
            for feature2 in self.features:
                if feature1 != feature2:
                    # Prepare data for test
                    data = pd.concat([
                        self.data[feature1],
                        self.data[feature2]
                    ], axis=1)
                    
                    # Perform Granger causality test
                    gc_test = grangercausalitytests(
                        data,
                        maxlag=max_lag,
                        verbose=False
                    )
                    
                    # Extract p-values for each lag
                    p_values = {
                        lag: round(test[0]['ssr_chi2test'][1], 4)
                        for lag, test in gc_test.items()
                    }
                    
                    # Find optimal lag
                    optimal_lag = min(p_values.items(), key=lambda x: x[1])[0]
                    
                    results[feature1][feature2] = {
                        'p_values': p_values,
                        'optimal_lag': optimal_lag,
                        'min_p_value': p_values[optimal_lag],
                        'causes_at_optimal_lag': p_values[optimal_lag] < 0.05
                    }
                    
        self.causality_results = results
        return results
    
    def analyze_nonlinearity(self) -> Dict:
        """
        Test for nonlinear temporal patterns using multiple approaches.
        
        Returns:
            Dict: Nonlinearity test results
        """
        results = {}
        
        for feature in self.features:
            series = self.data[feature].dropna()
            
            # Calculate ACF and PACF
            acf_values = acf(series, nlags=20)
            pacf_values = pacf(series, nlags=20)
            
            # Test for asymmetry in distribution
            skewness = stats.skew(series)
            kurtosis = stats.kurtosis(series)
            
            # Calculate Terasvirta test (simplified version)
            # Based on testing quadratic terms in AR model
            X = pd.DataFrame({
                'y': series[1:],
                'x': series[:-1],
                'x2': series[:-1]**2
            }).dropna()
            
            if len(X) > 3:  # Need at least 4 points for the test
                # Fit linear model
                from sklearn.linear_model import LinearRegression
                linear_model = LinearRegression()
                linear_score = linear_model.fit(
                    X[['x']], X['y']
                ).score(X[['x']], X['y'])
                
                # Fit nonlinear model
                nonlinear_model = LinearRegression()
                nonlinear_score = nonlinear_model.fit(
                    X[['x', 'x2']], X['y']
                ).score(X[['x', 'x2']], X['y'])
                
                # If nonlinear model fits significantly better, likely nonlinear
                nonlinearity_score = nonlinear_score - linear_score
                is_nonlinear = nonlinearity_score > 0.1
            else:
                nonlinearity_score = 0
                is_nonlinear = None
            
            # Check for volatility clustering
            returns = np.diff(series) / series[:-1]
            volatility = np.abs(returns)
            vol_acf = acf(volatility, nlags=5)
            has_volatility_clustering = np.any(vol_acf[1:] > 0.2)
            
            results[feature] = {
                'is_nonlinear': is_nonlinear,
                'nonlinearity_score': nonlinearity_score,
                'acf_values': acf_values,
                'pacf_values': pacf_values,
                'skewness': skewness,
                'kurtosis': kurtosis,
                'distribution_asymmetry': abs(skewness) > 1,
                'volatility_clustering': has_volatility_clustering,
                'volatility_acf': vol_acf.tolist()
            }
            
        self.nonlinearity_results = results
        return results
    
    def create_seasonal_plot(self, feature: str, output_path: str = None) -> go.Figure:
        """
        Create interactive plot of seasonal decomposition.
        
        Args:
            feature (str): Feature to plot
            output_path (str): Path to save the plot
            
        Returns:
            go.Figure: Plotly figure object
        """
        if feature not in self.seasonal_results:
            raise ValueError(f"No seasonal analysis results for {feature}")
            
        results = self.seasonal_results[feature]
        
        # Create subplots
        fig = go.Figure()
        
        # Original data
        fig.add_trace(go.Scatter(
            x=self.data.index,
            y=self.data[feature],
            name='Original',
            line=dict(color='blue')
        ))
        
        # Trend
        fig.add_trace(go.Scatter(
            x=self.data.index,
            y=results['trend'],
            name='Trend',
            line=dict(color='red')
        ))
        
        # Seasonal
        fig.add_trace(go.Scatter(
            x=self.data.index,
            y=results['seasonal'],
            name='Seasonal',
            line=dict(color='green')
        ))
        
        # Residual
        fig.add_trace(go.Scatter(
            x=self.data.index,
            y=results['residual'],
            name='Residual',
            line=dict(color='gray')
        ))
        
        fig.update_layout(
            title=f'Seasonal Decomposition - {feature}',
            xaxis_title='Date',
            yaxis_title='Value',
            height=800,
            showlegend=True
        )
        
        if output_path:
            fig.write_html(output_path)
            
        return fig
    
    def create_causality_heatmap(self, output_path: str = None) -> go.Figure:
        """
        Create heatmap of Granger causality relationships.
        
        Args:
            output_path (str): Path to save the plot
            
        Returns:
            go.Figure: Plotly figure object
        """
        if not self.causality_results:
            raise ValueError("No causality analysis results available")
            
        # Prepare data for heatmap
        causality_matrix = np.zeros((len(self.features), len(self.features)))
        
        for i, feature1 in enumerate(self.features):
            for j, feature2 in enumerate(self.features):
                if feature1 != feature2:
                    result = self.causality_results[feature1][feature2]
                    causality_matrix[i, j] = -np.log10(result['min_p_value'])
                    
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=causality_matrix,
            x=self.features,
            y=self.features,
            colorscale='RdBu',
            text=np.round(causality_matrix, 2),
            texttemplate='%{text}',
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title='Granger Causality Relationships (-log10 p-value)',
            xaxis_title='Potential Effect',
            yaxis_title='Potential Cause',
            width=1000,
            height=1000
        )
        
        if output_path:
            fig.write_html(output_path)
            
        return fig
    
    def create_temporal_network(self, p_value_threshold: float = 0.05,
                              output_path: str = None) -> go.Figure:
        """
        Create network visualization of temporal relationships.
        
        Args:
            p_value_threshold (float): Threshold for significant relationships
            output_path (str): Path to save the plot
            
        Returns:
            go.Figure: Plotly figure object
        """
        if not self.causality_results or not self.nonlinearity_results:
            raise ValueError("Missing analysis results")
            
        # Create network layout
        pos = {}
        for i, feature in enumerate(self.features):
            angle = 2 * np.pi * i / len(self.features)
            pos[feature] = [np.cos(angle), np.sin(angle)]
            
        # Create node trace
        node_x = []
        node_y = []
        node_text = []
        node_size = []
        
        for feature in self.features:
            x, y = pos[feature]
            node_x.append(x)
            node_y.append(y)
            
            # Create node text with nonlinearity info
            nonlin_info = self.nonlinearity_results[feature]
            node_text.append(
                f"{feature}<br>"
                f"Nonlinear: {nonlin_info['is_nonlinear']}<br>"
                f"Skewness: {nonlin_info['skewness']:.2f}"
            )
            
            # Node size based on nonlinearity
            node_size.append(30 if nonlin_info['is_nonlinear'] else 20)
            
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            marker=dict(
                size=node_size,
                color='lightblue',
                line=dict(color='black', width=1)
            ),
            text=self.features,
            textposition='bottom center',
            hovertext=node_text,
            hoverinfo='text'
        )
        
        # Create edge traces
        edge_traces = []
        
        for feature1 in self.features:
            for feature2 in self.features:
                if feature1 != feature2:
                    result = self.causality_results[feature1][feature2]
                    if result['min_p_value'] < p_value_threshold:
                        x0, y0 = pos[feature1]
                        x1, y1 = pos[feature2]
                        
                        edge_trace = go.Scatter(
                            x=[x0, x1],
                            y=[y0, y1],
                            mode='lines',
                            line=dict(
                                width=2,
                                color=f'rgba(100,100,100,{1-result["min_p_value"]})'
                            ),
                            hovertext=f"{feature1} → {feature2}<br>"
                                     f"p-value: {result['min_p_value']:.4f}<br>"
                                     f"Lag: {result['optimal_lag']}",
                            hoverinfo='text'
                        )
                        edge_traces.append(edge_trace)
                        
        # Create figure
        fig = go.Figure(data=edge_traces + [node_trace])
        
        fig.update_layout(
            title='Temporal Relationship Network',
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            width=1000,
            height=1000
        )
        
        if output_path:
            fig.write_html(output_path)
            
        return fig 