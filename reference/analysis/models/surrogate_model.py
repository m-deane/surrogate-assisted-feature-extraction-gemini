"""
Surrogate model module implementing SAFE (Surrogate Assisted Feature Extraction) methodology.
This module handles the creation and analysis of surrogate models for feature importance
and interaction analysis.
"""

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import shap
from typing import Tuple, List, Dict
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt

class SAFEAnalyzer:
    def __init__(self, random_state: int = 42):
        """
        Initialize the SAFE analyzer.
        
        Args:
            random_state (int): Random state for reproducibility
        """
        self.random_state = random_state
        self.surrogate_model = None
        self.original_model = None
        self.feature_names = None
        self.shap_values = None
        self.feature_interactions = None
        self.X = None
        
    def fit_surrogate_model(self, X: pd.DataFrame, y: pd.Series):
        """
        Fit the surrogate decision tree model and the original complex model.
        
        Args:
            X (pd.DataFrame): Feature matrix
            y (pd.Series): Target variable
        """
        self.feature_names = X.columns.tolist()
        self.X = X
        
        # Fit original complex model (Random Forest)
        self.original_model = RandomForestRegressor(
            n_estimators=100,
            random_state=self.random_state
        )
        self.original_model.fit(X, y)
        
        # Fit surrogate model (Decision Tree)
        self.surrogate_model = DecisionTreeRegressor(
            random_state=self.random_state,
            max_depth=5  # Adjustable parameter
        )
        
        # Get predictions from complex model
        complex_predictions = self.original_model.predict(X)
        
        # Fit surrogate model on complex model's predictions
        self.surrogate_model.fit(X, complex_predictions)
        
    def analyze_thresholds(self) -> Dict:
        """
        Extract and analyze thresholds from the surrogate decision tree model.
        
        Returns:
            Dict: Dictionary containing threshold analysis results
        """
        tree = self.surrogate_model.tree_
        threshold_analysis = {}
        
        for feature_idx in range(len(self.feature_names)):
            # Get all thresholds for this feature
            feature_thresholds = tree.threshold[tree.feature == feature_idx]
            if len(feature_thresholds) > 0:
                threshold_analysis[self.feature_names[feature_idx]] = {
                    'thresholds': sorted(feature_thresholds),
                    'importance': self.surrogate_model.feature_importances_[feature_idx]
                }
        
        return threshold_analysis
    
    def analyze_feature_interactions(self) -> Dict:
        """
        Analyze feature interactions using SHAP values and Friedman's H-statistic.
        
        Returns:
            Dict: Dictionary containing feature interaction analysis results
        """
        if self.X is None:
            raise ValueError("Must call fit_surrogate_model before analyzing feature interactions")
            
        # Calculate SHAP values
        explainer = shap.TreeExplainer(self.surrogate_model)
        self.shap_values = explainer.shap_values(self.X)
        
        # Calculate feature interactions
        interaction_values = {}
        n_features = len(self.feature_names)
        
        for i in range(n_features):
            for j in range(i+1, n_features):
                interaction_score = self._calculate_h_statistic(i, j)
                interaction_values[(self.feature_names[i], self.feature_names[j])] = interaction_score
        
        self.feature_interactions = interaction_values
        return interaction_values
    
    def _calculate_h_statistic(self, feature_i: int, feature_j: int) -> float:
        """
        Calculate Friedman's H-statistic for feature interaction strength.
        
        Args:
            feature_i (int): Index of first feature
            feature_j (int): Index of second feature
            
        Returns:
            float: H-statistic value
        """
        # This is a simplified version of the H-statistic calculation
        # For a complete implementation, refer to the original paper
        shap_ij = np.abs(self.shap_values[:, feature_i] * self.shap_values[:, feature_j])
        return np.mean(shap_ij)
    
    def plot_interaction_heatmap(self, output_path: str = None):
        """
        Create and save a heatmap of feature interactions.
        
        Args:
            output_path (str): Path to save the plot
        """
        if self.feature_interactions is None:
            raise ValueError("Run analyze_feature_interactions first")
            
        n_features = len(self.feature_names)
        interaction_matrix = np.zeros((n_features, n_features))
        
        for (f1, f2), score in self.feature_interactions.items():
            i = self.feature_names.index(f1)
            j = self.feature_names.index(f2)
            interaction_matrix[i, j] = score
            interaction_matrix[j, i] = score
            
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            interaction_matrix,
            xticklabels=self.feature_names,
            yticklabels=self.feature_names,
            cmap='YlOrRd',
            annot=True
        )
        plt.title('Feature Interaction Strength')
        
        if output_path:
            plt.savefig(output_path)
        plt.close()
        
    def get_model_performance(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """
        Calculate performance metrics for both original and surrogate models.
        
        Args:
            X (pd.DataFrame): Feature matrix
            y (pd.Series): True target values
            
        Returns:
            Dict: Dictionary containing performance metrics
        """
        original_pred = self.original_model.predict(X)
        surrogate_pred = self.surrogate_model.predict(X)
        
        return {
            'original_r2': r2_score(y, original_pred),
            'surrogate_r2': r2_score(y, surrogate_pred),
            'original_mse': mean_squared_error(y, original_pred),
            'surrogate_mse': mean_squared_error(y, surrogate_pred)
        } 