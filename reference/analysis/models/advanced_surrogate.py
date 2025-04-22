"""
Module for advanced surrogate analysis to discover complex feature relationships.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_squared_error
import shap
from scipy import stats
import networkx as nx
from itertools import combinations
import plotly.graph_objects as go
import plotly.express as px

class AdvancedSurrogateAnalyzer:
    def __init__(self, random_state: int = 42):
        """
        Initialize the advanced surrogate analyzer.
        
        Args:
            random_state (int): Random state for reproducibility
        """
        self.random_state = random_state
        self.complex_model = None
        self.surrogate_models = {}
        self.feature_names = None
        self.polynomial_features = None
        self.interaction_scores = None
        self.nonlinear_relationships = None
        self.poly_transformer = None
        
    def fit_complex_model(self, X: pd.DataFrame, y: pd.Series):
        """
        Fit a complex model (ensemble of different models) to capture intricate patterns.
        
        Args:
            X (pd.DataFrame): Feature matrix
            y (pd.Series): Target variable
        """
        self.feature_names = X.columns.tolist()
        
        # Create polynomial features for interaction detection
        self.poly_transformer = PolynomialFeatures(degree=2, include_bias=False)
        X_poly = self.poly_transformer.fit_transform(X)
        self.polynomial_features = self.poly_transformer.get_feature_names_out(self.feature_names)
        
        # Fit complex model (Gradient Boosting)
        self.complex_model = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            random_state=self.random_state
        )
        self.complex_model.fit(X_poly, y)
        
    def fit_surrogate_models(self, X: pd.DataFrame, y: pd.Series):
        """
        Fit multiple surrogate models to capture different aspects of the complex model.
        
        Args:
            X (pd.DataFrame): Feature matrix
            y (pd.Series): Target variable
        """
        # Get complex model predictions
        X_poly = self.poly_transformer.transform(X)
        complex_predictions = self.complex_model.predict(X_poly)
        
        # Fit different surrogate models
        self.surrogate_models['decision_tree'] = DecisionTreeRegressor(
            max_depth=5,
            random_state=self.random_state
        ).fit(X, complex_predictions)
        
        self.surrogate_models['random_forest'] = RandomForestRegressor(
            n_estimators=50,
            max_depth=5,
            random_state=self.random_state
        ).fit(X, complex_predictions)
        
    def analyze_feature_interactions(self, X: pd.DataFrame) -> Dict:
        """
        Analyze complex feature interactions using multiple methods.
        
        Args:
            X (pd.DataFrame): Feature matrix
            
        Returns:
            Dict: Feature interaction analysis results
        """
        interaction_scores = {}
        
        # Transform data for complex model
        X_poly = self.poly_transformer.transform(X)
        
        # Create a wrapper model for SHAP analysis
        class ModelWrapper:
            def __init__(self, model, poly_transformer):
                self.model = model
                self.poly_transformer = poly_transformer
                
            def predict(self, X):
                X_poly = self.poly_transformer.transform(X)
                return self.model.predict(X_poly)
        
        wrapped_model = ModelWrapper(self.complex_model, self.poly_transformer)
        
        # Calculate SHAP values with feature perturbation
        explainer = shap.KernelExplainer(
            wrapped_model.predict,
            shap.sample(X, 100),  # Use background dataset
            feature_names=self.feature_names
        )
        shap_values = explainer.shap_values(X.iloc[:100])  # Analyze subset for efficiency
        
        # Analyze pairwise interactions
        for i, j in combinations(range(len(self.feature_names)), 2):
            feature1, feature2 = self.feature_names[i], self.feature_names[j]
            
            # Calculate various interaction metrics
            interaction_scores[(feature1, feature2)] = {
                'shap_interaction': self._calculate_shap_interaction(
                    X[feature1], X[feature2], shap_values
                ),
                'mutual_info': self._calculate_mutual_information(
                    X[feature1], X[feature2]
                ),
                'correlation': stats.spearmanr(X[feature1], X[feature2]).correlation,
                'polynomial_importance': self._get_polynomial_importance(
                    feature1, feature2
                )
            }
            
        self.interaction_scores = interaction_scores
        return interaction_scores
    
    def analyze_nonlinear_relationships(self, X: pd.DataFrame) -> Dict:
        """
        Analyze nonlinear relationships between features and target.
        
        Args:
            X (pd.DataFrame): Feature matrix
            
        Returns:
            Dict: Nonlinear relationship analysis results
        """
        nonlinear_scores = {}
        
        # Transform data for predictions
        X_poly = self.poly_transformer.transform(X)
        predictions = self.complex_model.predict(X_poly)
        
        for feature in self.feature_names:
            # Get feature importance from polynomial terms
            poly_importance = self._get_polynomial_importance(feature)
            
            # Calculate various nonlinearity metrics
            nonlinear_scores[feature] = {
                'polynomial_importance': poly_importance,
                'shap_nonlinearity': self._calculate_nonlinearity_score(
                    X[feature], predictions
                ),
                'mutual_info': self._calculate_mutual_information(
                    X[feature], pd.Series(predictions)
                )
            }
            
        self.nonlinear_relationships = nonlinear_scores
        return nonlinear_scores
    
    def _calculate_shap_interaction(self, 
                                  feature1: pd.Series, 
                                  feature2: pd.Series,
                                  shap_values: np.ndarray) -> float:
        """Calculate SHAP interaction score between two features."""
        # Use correlation between feature product and SHAP values
        feature_product = feature1 * feature2
        shap_sum = np.sum(shap_values, axis=1)
        return abs(np.corrcoef(feature_product, shap_sum)[0, 1])
    
    def _calculate_mutual_information(self,
                                    x: pd.Series,
                                    y: pd.Series) -> float:
        """Calculate mutual information between two variables."""
        from sklearn.feature_selection import mutual_info_regression
        return mutual_info_regression(
            x.values.reshape(-1, 1),
            y
        )[0]
    
    def _get_polynomial_importance(self,
                                 feature1: str,
                                 feature2: Optional[str] = None) -> float:
        """Get importance of polynomial terms for feature(s)."""
        if feature2 is None:
            # Get quadratic term importance
            term = f"{feature1}^2"
            idx = np.where(self.polynomial_features == term)[0]
            if len(idx) > 0:
                return abs(self.complex_model.feature_importances_[idx[0]])
            return 0
        else:
            # Get interaction term importance
            term = f"{feature1} {feature2}"
            idx = np.where(self.polynomial_features == term)[0]
            if len(idx) > 0:
                return abs(self.complex_model.feature_importances_[idx[0]])
            return 0
    
    def _calculate_nonlinearity_score(self, feature: pd.Series, predictions: np.ndarray) -> float:
        """
        Calculate nonlinearity score using various metrics.
        
        Args:
            feature (pd.Series): Feature values
            predictions (np.ndarray): Model predictions
            
        Returns:
            float: Nonlinearity score
        """
        # Fit linear and quadratic models
        X = feature.values.reshape(-1, 1)
        linear_fit = np.polyfit(X.ravel(), predictions, 1)
        quad_fit = np.polyfit(X.ravel(), predictions, 2)
        
        # Calculate R² for both models
        linear_pred = np.polyval(linear_fit, X.ravel())
        quad_pred = np.polyval(quad_fit, X.ravel())
        
        linear_r2 = r2_score(predictions, linear_pred)
        quad_r2 = r2_score(predictions, quad_pred)
        
        # Return improvement in R² from quadratic fit
        return max(0, quad_r2 - linear_r2)
    
    def create_interaction_network(self, 
                                 min_score: float = 0.1) -> nx.Graph:
        """
        Create network representation of feature interactions.
        
        Args:
            min_score (float): Minimum interaction score to include
            
        Returns:
            nx.Graph: NetworkX graph of interactions
        """
        G = nx.Graph()
        
        # Add nodes for features
        for feature in self.feature_names:
            nonlinear_score = self.nonlinear_relationships[feature]['polynomial_importance']
            G.add_node(feature, nonlinearity=nonlinear_score)
        
        # Add edges for interactions
        for (f1, f2), scores in self.interaction_scores.items():
            interaction_strength = scores['polynomial_importance']
            if interaction_strength > min_score:
                G.add_edge(f1, f2, 
                          weight=interaction_strength,
                          mutual_info=scores['mutual_info'],
                          correlation=scores['correlation'])
                
        return G
    
    def visualize_interaction_network(self, 
                                    G: nx.Graph,
                                    output_path: str = None) -> go.Figure:
        """
        Create interactive visualization of feature interaction network.
        
        Args:
            G (nx.Graph): Feature interaction network
            output_path (str): Path to save the plot
            
        Returns:
            go.Figure: Plotly figure object
        """
        pos = nx.spring_layout(G, dim=3)
        
        # Create node trace
        node_x, node_y, node_z = [], [], []
        node_text = []
        node_size = []
        node_names = list(G.nodes())  # Convert nodes to list
        
        for node in node_names:
            x, y, z = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_z.append(z)
            node_size.append(G.nodes[node]['nonlinearity'] * 50)
            node_text.append(
                f"{node}<br>"
                f"Nonlinearity: {G.nodes[node]['nonlinearity']:.3f}"
            )
            
        node_trace = go.Scatter3d(
            x=node_x, y=node_y, z=node_z,
            mode='markers+text',
            marker=dict(
                size=node_size,
                color=node_size,
                colorscale='Viridis',
                showscale=True
            ),
            text=node_names,  # Use node names for text
            hovertext=node_text,  # Use detailed text for hover
            hoverinfo='text'
        )
        
        # Create edge traces
        edge_x, edge_y, edge_z = [], [], []
        edge_text = []
        
        for edge in G.edges(data=True):
            x0, y0, z0 = pos[edge[0]]
            x1, y1, z1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_z.extend([z0, z1, None])
            edge_text.append(
                f"Interaction: {edge[2]['weight']:.3f}<br>"
                f"Mutual Info: {edge[2]['mutual_info']:.3f}<br>"
                f"Correlation: {edge[2]['correlation']:.3f}"
            )
            
        edge_trace = go.Scatter3d(
            x=edge_x, y=edge_y, z=edge_z,
            mode='lines',
            line=dict(
                width=2,
                color='rgba(100,100,100,0.5)'
            ),
            hovertext=edge_text,
            hoverinfo='text'
        )
        
        # Create figure
        fig = go.Figure(data=[edge_trace, node_trace])
        
        fig.update_layout(
            title='Feature Interaction Network (3D)',
            showlegend=False,
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z'
            ),
            width=1000,
            height=1000
        )
        
        if output_path:
            fig.write_html(output_path)
            
        return fig
    
    def create_relationship_heatmap(self, output_path: str = None) -> go.Figure:
        """
        Create heatmap of feature relationships.
        
        Args:
            output_path (str): Path to save the plot
            
        Returns:
            go.Figure: Plotly figure object
        """
        # Prepare data for heatmap
        n_features = len(self.feature_names)
        relationship_matrix = np.zeros((n_features, n_features))
        
        # Fill diagonal with nonlinearity scores
        for i, feature in enumerate(self.feature_names):
            relationship_matrix[i, i] = self.nonlinear_relationships[feature]['polynomial_importance']
        
        # Fill off-diagonal with interaction scores
        for (f1, f2), scores in self.interaction_scores.items():
            i = self.feature_names.index(f1)
            j = self.feature_names.index(f2)
            relationship_matrix[i, j] = scores['polynomial_importance']
            relationship_matrix[j, i] = scores['polynomial_importance']
            
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=relationship_matrix,
            x=self.feature_names,
            y=self.feature_names,
            colorscale='RdBu',
            zmid=np.mean(relationship_matrix),
            text=np.round(relationship_matrix, 3),
            texttemplate='%{text}',
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title='Feature Relationships Heatmap',
            xaxis_title='Feature',
            yaxis_title='Feature',
            width=1000,
            height=1000
        )
        
        if output_path:
            fig.write_html(output_path)
            
        return fig
    
    def get_model_performance(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """
        Calculate performance metrics for all models.
        
        Args:
            X (pd.DataFrame): Feature matrix
            y (pd.Series): Target variable
            
        Returns:
            Dict: Performance metrics for each model
        """
        X_poly = self.poly_transformer.transform(X)
        complex_pred = self.complex_model.predict(X_poly)
        
        performance = {
            'complex_model': {
                'r2': r2_score(y, complex_pred),
                'mse': mean_squared_error(y, complex_pred)
            }
        }
        
        for name, model in self.surrogate_models.items():
            pred = model.predict(X)
            performance[name] = {
                'r2': r2_score(y, pred),
                'mse': mean_squared_error(y, pred),
                'fidelity': r2_score(complex_pred, pred)
            }
            
        return performance 