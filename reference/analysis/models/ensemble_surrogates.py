"""
Ensemble surrogate modeling module for combining multiple interpretable models.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import shap
from interpret.glassbox import ExplainableBoostingRegressor
from pygam import GAM, s, f
import joblib
import os
import pickle
from functools import partial
import networkx as nx
import plotly.graph_objects as go

class EnsembleSurrogateAnalyzer:
    def __init__(self, random_state: int = 42):
        """
        Initialize the ensemble surrogate analyzer.
        
        Args:
            random_state (int): Random seed for reproducibility
        """
        self.random_state = random_state
        self.surrogate_models = {}
        self.surrogate_weights = {}
        self.feature_importance = {}
        self.ensemble_importance = {}
        self.shap_values = {}
        np.random.seed(random_state)
    
    def fit_surrogate_ensemble(self, X: pd.DataFrame, y: pd.Series, 
                             model_types: Optional[List[str]] = None) -> Dict:
        """
        Fit an ensemble of surrogate models to the data.
        
        Args:
            X (pd.DataFrame): Feature matrix
            y (pd.Series): Target values
            model_types (List[str], optional): List of model types to include in ensemble
                Options: 'dt' (decision tree), 'linear', 'ridge', 'lasso', 'ebm' (explainable boosting machine),
                'gam' (generalized additive model)
                
        Returns:
            Dict: Dictionary with model metrics
        """
        self.feature_names = X.columns.tolist()
        
        # If no model types provided, use all
        if model_types is None:
            model_types = ['dt', 'linear', 'ridge', 'lasso', 'ebm', 'gam']
        
        # Initialize models
        models = {
            'dt': DecisionTreeRegressor(random_state=self.random_state, max_depth=5),
            'linear': LinearRegression(),
            'ridge': Ridge(alpha=1.0, random_state=self.random_state),
            'lasso': Lasso(alpha=0.1, random_state=self.random_state),
            'ebm': ExplainableBoostingRegressor(random_state=self.random_state),
            'gam': GAM(s(0) + s(1) + s(2) + s(3) + s(4) + s(5), 
                      distribution='normal', link='identity')
        }
        
        # Filter models based on model_types
        models = {k: v for k, v in models.items() if k in model_types}
        
        # Fit models and calculate performance
        performance = {}
        
        for model_name, model in models.items():
            try:
                if model_name == 'gam':
                    # GAM requires a different fitting API
                    # Limit to first 6 features for simplicity
                    X_gam = X.iloc[:, :6].values
                    model.fit(X_gam, y)
                    y_pred = model.predict(X_gam)
                else:
                    model.fit(X, y)
                    y_pred = model.predict(X)
                
                r2 = r2_score(y, y_pred)
                mse = mean_squared_error(y, y_pred)
                
                self.surrogate_models[model_name] = model
                performance[model_name] = {
                    'r2': r2,
                    'mse': mse
                }
                
                # Calculate SHAP values for feature importance
                if model_name != 'gam':  # Skip GAM for SHAP due to complexity
                    explainer = shap.Explainer(model, X)
                    shap_values = explainer(X)
                    self.shap_values[model_name] = shap_values
                    
                    # Feature importance based on SHAP
                    feature_importance = np.abs(shap_values.values).mean(0)
                    self.feature_importance[model_name] = dict(zip(X.columns, feature_importance))
            except Exception as e:
                print(f"Error fitting model {model_name}: {e}")
                continue
        
        # Calculate ensemble weights based on R² score
        total_r2 = sum(info['r2'] for info in performance.values())
        if total_r2 > 0:
            self.surrogate_weights = {
                model_name: info['r2'] / total_r2
                for model_name, info in performance.items()
            }
        else:
            # Equal weights if all models perform poorly
            n_models = len(performance)
            self.surrogate_weights = {
                model_name: 1.0 / n_models
                for model_name in performance.keys()
            }
        
        # Calculate ensemble feature importance
        ensemble_importance = {}
        for feature in X.columns:
            ensemble_importance[feature] = sum(
                self.surrogate_weights.get(model_name, 0) * 
                self.feature_importance.get(model_name, {}).get(feature, 0)
                for model_name in self.surrogate_models.keys()
            )
        
        self.ensemble_importance = ensemble_importance
        
        return {
            'model_performance': performance,
            'ensemble_weights': self.surrogate_weights,
            'ensemble_importance': self.ensemble_importance
        }
    
    def predict_ensemble(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate ensemble predictions by weighted averaging of surrogate models.
        
        Args:
            X (pd.DataFrame): Feature matrix
            
        Returns:
            np.ndarray: Ensemble predictions
        """
        if not self.surrogate_models:
            raise ValueError("No surrogate models fitted yet. Run fit_surrogate_ensemble first.")
        
        predictions = {}
        
        for model_name, model in self.surrogate_models.items():
            try:
                if model_name == 'gam':
                    # Handle GAM differently
                    X_gam = X.iloc[:, :6].values
                    predictions[model_name] = model.predict(X_gam)
                else:
                    predictions[model_name] = model.predict(X)
            except Exception as e:
                print(f"Error predicting with model {model_name}: {e}")
                continue
        
        # Weighted ensemble prediction
        ensemble_pred = np.zeros(len(X))
        
        for model_name, pred in predictions.items():
            weight = self.surrogate_weights.get(model_name, 0)
            ensemble_pred += weight * pred
        
        return ensemble_pred
    
    def analyze_feature_thresholds(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """
        Analyze feature thresholds across all surrogate models.
        
        Args:
            X (pd.DataFrame): Feature matrix
            y (pd.Series): Target values
            
        Returns:
            Dict: Dictionary with threshold analysis results
        """
        if not self.surrogate_models:
            raise ValueError("No surrogate models fitted yet. Run fit_surrogate_ensemble first.")
        
        threshold_results = {}
        
        # Extract thresholds from tree-based models
        if 'dt' in self.surrogate_models:
            dt_model = self.surrogate_models['dt']
            
            # Get feature thresholds from decision tree
            for feature_idx, feature_name in enumerate(X.columns):
                thresholds = []
                tree = dt_model.tree_
                
                # Extract all thresholds for this feature
                for node_id in range(tree.node_count):
                    if tree.feature[node_id] == feature_idx and tree.threshold[node_id] != -2:
                        thresholds.append(tree.threshold[node_id])
                
                if thresholds:
                    if feature_name not in threshold_results:
                        threshold_results[feature_name] = {
                            'thresholds': [],
                            'importance': self.ensemble_importance.get(feature_name, 0)
                        }
                    
                    threshold_results[feature_name]['thresholds'].extend(thresholds)
        
        # Extract "virtual thresholds" from linear models using coefficient signs
        for model_name in ['linear', 'ridge', 'lasso']:
            if model_name in self.surrogate_models:
                model = self.surrogate_models[model_name]
                
                for feature_idx, feature_name in enumerate(X.columns):
                    if hasattr(model, 'coef_'):
                        coef = model.coef_[feature_idx]
                        
                        # If coefficient is significant, estimate a threshold
                        if abs(coef) > 0.01:
                            # For linear models, we use mean as a rough "threshold"
                            mean_val = X[feature_name].mean()
                            
                            if feature_name not in threshold_results:
                                threshold_results[feature_name] = {
                                    'thresholds': [],
                                    'importance': self.ensemble_importance.get(feature_name, 0)
                                }
                            
                            threshold_results[feature_name]['thresholds'].append(mean_val)
        
        # Extract partial dependence thresholds from EBM
        if 'ebm' in self.surrogate_models:
            ebm_model = self.surrogate_models['ebm']
            
            for feature_idx, feature_name in enumerate(X.columns):
                # Get the feature's global effect
                try:
                    feature_effect = ebm_model.explain_global().data(feature_name)
                    if feature_effect is not None:
                        scores = feature_effect['scores']
                        bin_edges = feature_effect['names']
                        
                        # Look for sign changes in the effect
                        sign_changes = []
                        for i in range(1, len(scores)):
                            if scores[i-1] * scores[i] < 0:  # Sign change
                                # Estimate threshold as midpoint between bin edges
                                threshold = (float(bin_edges[i-1]) + float(bin_edges[i])) / 2
                                sign_changes.append(threshold)
                        
                        if sign_changes:
                            if feature_name not in threshold_results:
                                threshold_results[feature_name] = {
                                    'thresholds': [],
                                    'importance': self.ensemble_importance.get(feature_name, 0)
                                }
                            
                            threshold_results[feature_name]['thresholds'].extend(sign_changes)
                except:
                    # Skip if feature explanation fails
                    continue
        
        # Consolidate similar thresholds
        for feature_name, info in threshold_results.items():
            thresholds = info['thresholds']
            if thresholds:
                # Sort thresholds
                thresholds = sorted(thresholds)
                
                # Consolidate similar thresholds
                consolidated = [thresholds[0]]
                for thresh in thresholds[1:]:
                    # If this threshold is similar to the previous one, skip it
                    if abs(thresh - consolidated[-1]) < 0.05 * (X[feature_name].max() - X[feature_name].min()):
                        continue
                    consolidated.append(thresh)
                
                info['thresholds'] = consolidated
        
        return threshold_results
    
    def analyze_model_disagreement(self, X: pd.DataFrame) -> Dict:
        """
        Analyze disagreement between surrogate models to identify uncertain predictions.
        
        Args:
            X (pd.DataFrame): Feature matrix
            
        Returns:
            Dict: Dictionary with disagreement analysis results
        """
        if not self.surrogate_models:
            raise ValueError("No surrogate models fitted yet. Run fit_surrogate_ensemble first.")
        
        model_predictions = {}
        
        # Get predictions from all models
        for model_name, model in self.surrogate_models.items():
            try:
                if model_name == 'gam':
                    # Handle GAM differently
                    X_gam = X.iloc[:, :6].values
                    model_predictions[model_name] = model.predict(X_gam)
                else:
                    model_predictions[model_name] = model.predict(X)
            except Exception as e:
                print(f"Error predicting with model {model_name}: {e}")
                continue
        
        # Calculate disagreement metrics
        n_samples = len(X)
        disagreement_scores = np.zeros(n_samples)
        
        # Compute the variance of predictions for each sample
        all_preds = np.array([preds for preds in model_predictions.values()])
        disagreement_scores = np.var(all_preds, axis=0)
        
        # Find points with highest disagreement
        top_disagreement_indices = np.argsort(disagreement_scores)[-10:]
        top_disagreement_samples = {
            i: {
                'sample': X.iloc[i].to_dict(),
                'predictions': {model: float(preds[i]) for model, preds in model_predictions.items()},
                'disagreement_score': float(disagreement_scores[i])
            } for i in top_disagreement_indices
        }
        
        # Calculate feature-specific disagreement
        feature_disagreement = {}
        
        for feature in X.columns:
            # Sort data by feature value
            sorted_indices = X[feature].argsort()
            sorted_disagreement = disagreement_scores[sorted_indices]
            
            # Calculate rolling average of disagreement
            window_size = min(10, len(sorted_disagreement))
            if window_size > 0:
                rolling_disagreement = np.convolve(sorted_disagreement, 
                                                np.ones(window_size)/window_size, 
                                                mode='valid')
                
                # Find peaks in rolling disagreement
                peak_indices = []
                for i in range(1, len(rolling_disagreement)-1):
                    if (rolling_disagreement[i] > rolling_disagreement[i-1] and 
                        rolling_disagreement[i] > rolling_disagreement[i+1] and
                        rolling_disagreement[i] > np.mean(rolling_disagreement)):
                        peak_indices.append(i)
                
                # Map peak indices back to feature values
                feature_values = X[feature].iloc[sorted_indices[window_size//2:-window_size//2+1]]
                peak_values = [feature_values.iloc[i] for i in peak_indices]
                
                if peak_values:
                    feature_disagreement[feature] = {
                        'peak_values': peak_values,
                        'mean_disagreement': float(np.mean(rolling_disagreement)),
                        'max_disagreement': float(np.max(rolling_disagreement))
                    }
        
        return {
            'overall_disagreement': float(np.mean(disagreement_scores)),
            'top_disagreement_samples': top_disagreement_samples,
            'feature_disagreement': feature_disagreement
        }
    
    def compare_model_explanations(self, X: pd.DataFrame) -> Dict:
        """
        Compare explanations from different surrogate models.
        
        Args:
            X (pd.DataFrame): Feature matrix
            
        Returns:
            Dict: Dictionary with explanation comparison results
        """
        if not self.surrogate_models or not self.shap_values:
            raise ValueError("No surrogate models or SHAP values available.")
        
        explanation_comparison = {}
        
        # Get feature rankings from each model
        feature_rankings = {}
        
        for model_name, importances in self.feature_importance.items():
            # Skip GAM which doesn't have SHAP values
            if model_name == 'gam':
                continue
                
            # Rank features by importance
            ranked_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)
            feature_rankings[model_name] = [feature for feature, _ in ranked_features]
        
        # Calculate Spearman rank correlation between model explanations
        rank_correlation = {}
        
        for model1 in feature_rankings:
            for model2 in feature_rankings:
                if model1 != model2:
                    if model1 not in rank_correlation:
                        rank_correlation[model1] = {}
                    
                    # Calculate correlation between rankings
                    # Use relative positions instead of scipy's spearmanr for simplicity
                    ranks1 = {feature: i for i, feature in enumerate(feature_rankings[model1])}
                    ranks2 = {feature: i for i, feature in enumerate(feature_rankings[model2])}
                    
                    common_features = set(ranks1.keys()) & set(ranks2.keys())
                    
                    if common_features:
                        # Calculate sum of squared rank differences
                        sum_sq_diff = sum((ranks1[f] - ranks2[f])**2 for f in common_features)
                        n = len(common_features)
                        
                        # Spearman correlation formula
                        correlation = 1 - (6 * sum_sq_diff) / (n * (n**2 - 1))
                        rank_correlation[model1][model2] = correlation
        
        explanation_comparison['feature_rankings'] = feature_rankings
        explanation_comparison['rank_correlation'] = rank_correlation
        
        # Find consistent and inconsistent features across models
        consistent_features = []
        inconsistent_features = []
        
        all_features = list(X.columns)
        for feature in all_features:
            # Get rankings of this feature across models
            rankings = []
            
            for model_name, ranked_features in feature_rankings.items():
                if feature in ranked_features:
                    # Get relative rank (normalized by feature count)
                    rel_rank = ranked_features.index(feature) / len(ranked_features)
                    rankings.append(rel_rank)
            
            if rankings:
                # If the standard deviation of rankings is low, feature is consistent
                if np.std(rankings) < 0.2:  # Threshold for consistency
                    consistent_features.append({
                        'feature': feature,
                        'avg_rank': float(np.mean(rankings)),
                        'std_rank': float(np.std(rankings))
                    })
                else:
                    inconsistent_features.append({
                        'feature': feature,
                        'avg_rank': float(np.mean(rankings)),
                        'std_rank': float(np.std(rankings)),
                        'model_ranks': {model: ranked_features.index(feature) 
                                      for model, ranked_features in feature_rankings.items()
                                      if feature in ranked_features}
                    })
        
        # Sort by average rank
        consistent_features.sort(key=lambda x: x['avg_rank'])
        inconsistent_features.sort(key=lambda x: x['std_rank'], reverse=True)
        
        explanation_comparison['consistent_features'] = consistent_features
        explanation_comparison['inconsistent_features'] = inconsistent_features
        
        return explanation_comparison
    
    def plot_model_comparison(self, output_path: str = None) -> plt.Figure:
        """
        Plot comparison of surrogate models.
        
        Args:
            output_path (str, optional): Path to save the plot
            
        Returns:
            plt.Figure: Matplotlib figure
        """
        if not self.surrogate_models:
            raise ValueError("No surrogate models fitted yet. Run fit_surrogate_ensemble first.")
        
        # Prepare data for plotting
        model_names = list(self.surrogate_weights.keys())
        weights = [self.surrogate_weights[model] for model in model_names]
        
        # Create figure with three subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot model weights
        ax1.bar(model_names, weights)
        ax1.set_title('Surrogate Model Weights')
        ax1.set_ylabel('Weight')
        ax1.set_ylim(0, max(weights) * 1.2)
        
        # Plot top features by importance
        top_n = 10
        top_features = sorted(self.ensemble_importance.items(), 
                            key=lambda x: x[1], reverse=True)[:top_n]
        
        feature_names = [f[0] for f in top_features]
        importances = [f[1] for f in top_features]
        
        ax2.barh(feature_names[::-1], importances[::-1])
        ax2.set_title(f'Top {top_n} Features by Ensemble Importance')
        ax2.set_xlabel('Importance')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path)
            plt.close()
            return None
        
        return fig
    
    def plot_model_disagreement(self, X: pd.DataFrame, disagreement_results: Dict, 
                              output_path: str = None) -> plt.Figure:
        """
        Plot model disagreement analysis.
        
        Args:
            X (pd.DataFrame): Feature matrix
            disagreement_results (Dict): Results from analyze_model_disagreement
            output_path (str, optional): Path to save the plot
            
        Returns:
            plt.Figure: Matplotlib figure
        """
        feature_disagreement = disagreement_results.get('feature_disagreement', {})
        
        if not feature_disagreement:
            raise ValueError("No feature disagreement data available.")
        
        # Sort features by max disagreement
        sorted_features = sorted(
            feature_disagreement.items(),
            key=lambda x: x[1]['max_disagreement'],
            reverse=True
        )
        
        # Limit to top 6 features
        top_features = sorted_features[:6]
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, (feature, info) in enumerate(top_features):
            if i < len(axes):
                # Sort data by feature value
                sorted_indices = X[feature].argsort()
                feature_values = X[feature].iloc[sorted_indices]
                
                # Plot feature distribution
                sns.histplot(feature_values, ax=axes[i], kde=True, color='skyblue')
                
                # Mark disagreement peak values
                for peak in info['peak_values']:
                    axes[i].axvline(peak, color='red', linestyle='--', alpha=0.7)
                
                axes[i].set_title(f"{feature}\nMax Disagreement: {info['max_disagreement']:.4f}")
                axes[i].set_xlabel(feature)
                axes[i].set_ylabel('Count')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path)
            plt.close()
            return None
        
        return fig
    
    def create_explanation_consistency_network(self, explanation_comparison: Dict, 
                                           output_path: str = None) -> go.Figure:
        """
        Create an interactive network visualization of explanation consistency.
        
        Args:
            explanation_comparison (Dict): Results from compare_model_explanations
            output_path (str, optional): Path to save the plot
            
        Returns:
            go.Figure: Plotly figure
        """
        rank_correlation = explanation_comparison.get('rank_correlation', {})
        
        if not rank_correlation:
            raise ValueError("No rank correlation data available.")
        
        # Create graph
        G = nx.Graph()
        
        # Add model nodes
        model_names = list(self.surrogate_weights.keys())
        for model in model_names:
            G.add_node(model, size=self.surrogate_weights.get(model, 0), type='model')
        
        # Add correlation edges
        for model1, correlations in rank_correlation.items():
            for model2, corr in correlations.items():
                if corr > 0.1:  # Only add edges with meaningful correlation
                    G.add_edge(model1, model2, weight=corr)
        
        # Add feature nodes for consistent features
        consistent_features = explanation_comparison.get('consistent_features', [])
        for feature_info in consistent_features[:10]:  # Top 10 consistent features
            feature = feature_info['feature']
            G.add_node(feature, size=self.ensemble_importance.get(feature, 0), 
                     type='consistent_feature')
            
            # Connect to models
            for model in model_names:
                if model != 'gam':  # Skip GAM
                    G.add_edge(feature, model, weight=0.5)
        
        # Add feature nodes for inconsistent features
        inconsistent_features = explanation_comparison.get('inconsistent_features', [])
        for feature_info in inconsistent_features[:5]:  # Top 5 inconsistent features
            feature = feature_info['feature']
            G.add_node(feature, size=self.ensemble_importance.get(feature, 0), 
                     type='inconsistent_feature')
            
            # Connect to models with varying weights based on ranks
            model_ranks = feature_info.get('model_ranks', {})
            for model, rank in model_ranks.items():
                # Normalize rank to [0, 1] (invert so lower rank = higher weight)
                if len(self.feature_names) > 1:
                    normalized_rank = 1 - (rank / (len(self.feature_names) - 1))
                else:
                    normalized_rank = 1
                G.add_edge(feature, model, weight=normalized_rank)
        
        # Calculate node positions using force-directed layout
        pos = nx.spring_layout(G, k=0.5, iterations=50)
        
        # Create node traces
        node_traces = {}
        
        # Model nodes
        model_x = []
        model_y = []
        model_text = []
        model_size = []
        
        # Consistent feature nodes
        con_feature_x = []
        con_feature_y = []
        con_feature_text = []
        con_feature_size = []
        
        # Inconsistent feature nodes
        incon_feature_x = []
        incon_feature_y = []
        incon_feature_text = []
        incon_feature_size = []
        
        for node in G.nodes():
            x, y = pos[node]
            node_type = G.nodes[node].get('type', '')
            size = G.nodes[node].get('size', 0.1)
            
            if node_type == 'model':
                model_x.append(x)
                model_y.append(y)
                model_text.append(f"Model: {node}<br>Weight: {size:.4f}")
                model_size.append(size * 50 + 20)
            elif node_type == 'consistent_feature':
                con_feature_x.append(x)
                con_feature_y.append(y)
                con_feature_text.append(f"Feature: {node}<br>Importance: {size:.4f}<br>Type: Consistent")
                con_feature_size.append(size * 40 + 15)
            elif node_type == 'inconsistent_feature':
                incon_feature_x.append(x)
                incon_feature_y.append(y)
                incon_feature_text.append(f"Feature: {node}<br>Importance: {size:.4f}<br>Type: Inconsistent")
                incon_feature_size.append(size * 40 + 15)
        
        # Create node traces
        model_trace = go.Scatter(
            x=model_x, y=model_y,
            mode='markers+text',
            marker=dict(
                color='blue',
                size=model_size,
                line=dict(width=1)
            ),
            text=[node for node in G.nodes() if G.nodes[node].get('type', '') == 'model'],
            textposition="top center",
            hovertext=model_text,
            name='Models'
        )
        
        con_feature_trace = go.Scatter(
            x=con_feature_x, y=con_feature_y,
            mode='markers+text',
            marker=dict(
                color='green',
                size=con_feature_size,
                line=dict(width=1)
            ),
            text=[node for node in G.nodes() if G.nodes[node].get('type', '') == 'consistent_feature'],
            textposition="bottom center",
            hovertext=con_feature_text,
            name='Consistent Features'
        )
        
        incon_feature_trace = go.Scatter(
            x=incon_feature_x, y=incon_feature_y,
            mode='markers+text',
            marker=dict(
                color='red',
                size=incon_feature_size,
                line=dict(width=1)
            ),
            text=[node for node in G.nodes() if G.nodes[node].get('type', '') == 'inconsistent_feature'],
            textposition="bottom center",
            hovertext=incon_feature_text,
            name='Inconsistent Features'
        )
        
        # Create edge trace
        edge_x = []
        edge_y = []
        edge_text = []
        edge_color = []
        
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.append(x0)
            edge_x.append(x1)
            edge_x.append(None)
            edge_y.append(y0)
            edge_y.append(y1)
            edge_y.append(None)
            
            weight = G.edges[edge].get('weight', 0)
            edge_text.append(f"{edge[0]} - {edge[1]}<br>Weight: {weight:.4f}")
            
            # Color based on nodes connected
            node1_type = G.nodes[edge[0]].get('type', '')
            node2_type = G.nodes[edge[1]].get('type', '')
            
            if 'inconsistent_feature' in [node1_type, node2_type]:
                edge_color.append('rgba(255,0,0,0.5)')
            elif 'consistent_feature' in [node1_type, node2_type]:
                edge_color.append('rgba(0,255,0,0.5)')
            else:
                edge_color.append('rgba(0,0,255,0.5)')
        
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1, color=edge_color),
            hoverinfo='text',
            mode='lines',
            hovertext=edge_text
        )
        
        # Create figure
        fig = go.Figure(data=[edge_trace, model_trace, con_feature_trace, incon_feature_trace],
                      layout=go.Layout(
                          title="Model Explanation Consistency Network",
                          titlefont_size=16,
                          showlegend=True,
                          hovermode='closest',
                          margin=dict(b=20, l=5, r=5, t=40),
                          xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                          yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                      ))
        
        if output_path:
            fig.write_html(output_path)
        
        return fig
    
    def get_model_performance(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """
        Calculate performance metrics for all surrogate models and ensemble.
        
        Args:
            X (pd.DataFrame): Feature matrix
            y (pd.Series): True target values
            
        Returns:
            Dict: Dictionary containing performance metrics for each model
        """
        if not self.surrogate_models:
            raise ValueError("No surrogate models fitted yet. Run fit_surrogate_ensemble first.")
        
        performance_metrics = {}
        
        # Calculate metrics for each model
        for model_name, model in self.surrogate_models.items():
            try:
                if model_name == 'gam':
                    # Handle GAM differently
                    X_gam = X.iloc[:, :6].values
                    predictions = model.predict(X_gam)
                else:
                    predictions = model.predict(X)
                
                performance_metrics[model_name] = {
                    'r2': r2_score(y, predictions),
                    'mse': mean_squared_error(y, predictions),
                    'fidelity': self.surrogate_weights.get(model_name, 0)
                }
            except Exception as e:
                print(f"Error calculating metrics for model {model_name}: {e}")
                continue
        
        # Calculate ensemble performance
        ensemble_predictions = self.predict_ensemble(X)
        performance_metrics['ensemble'] = {
            'r2': r2_score(y, ensemble_predictions),
            'mse': mean_squared_error(y, ensemble_predictions),
            'fidelity': 1.0  # Ensemble has perfect fidelity to itself
        }
        
        return performance_metrics 