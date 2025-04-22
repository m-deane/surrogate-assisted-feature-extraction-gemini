"""
Module for creating interactive visualizations of ensemble model analysis results.
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from typing import Dict, List
import networkx as nx

class EnsembleVisualizer:
    def __init__(self):
        """Initialize the ensemble visualizer."""
        self.color_palette = px.colors.qualitative.Set3
        
    def create_model_comparison_plot(self,
                                   performance_metrics: Dict,
                                   output_path: str = None) -> go.Figure:
        """
        Create an interactive plot comparing model performances.
        
        Args:
            performance_metrics (Dict): Dictionary of model performance metrics
            output_path (str): Path to save the plot
            
        Returns:
            go.Figure: Plotly figure object
        """
        # Prepare data for plotting
        models = []
        r2_scores = []
        mse_scores = []
        fidelity_scores = []
        
        for model_name, metrics in performance_metrics.items():
            models.append(model_name)
            r2_scores.append(metrics['r2'])
            mse_scores.append(metrics['mse'])
            if 'fidelity' in metrics:
                fidelity_scores.append(metrics['fidelity'])
            else:
                fidelity_scores.append(None)
        
        # Create figure with secondary y-axis
        fig = go.Figure()
        
        # Add R² bars
        fig.add_trace(go.Bar(
            name='R² Score',
            x=models,
            y=r2_scores,
            marker_color='blue',
            opacity=0.7
        ))
        
        # Add MSE line
        fig.add_trace(go.Scatter(
            name='MSE',
            x=models,
            y=mse_scores,
            yaxis='y2',
            mode='lines+markers',
            line=dict(color='red', width=2),
            marker=dict(size=10)
        ))
        
        # Add fidelity points if available
        if any(score is not None for score in fidelity_scores):
            fig.add_trace(go.Scatter(
                name='Fidelity',
                x=models,
                y=fidelity_scores,
                mode='markers',
                marker=dict(
                    size=15,
                    symbol='star',
                    color='green'
                )
            ))
        
        fig.update_layout(
            title='Model Performance Comparison',
            xaxis_title='Model',
            yaxis_title='R² Score',
            yaxis2=dict(
                title='MSE',
                overlaying='y',
                side='right'
            ),
            height=600,
            template='plotly_white',
            showlegend=True,
            hovermode='x unified',
            barmode='group'
        )
        
        if output_path:
            fig.write_html(output_path)
            
        return fig
    
    def create_disagreement_heatmap(self,
                                  disagreement_results: Dict,
                                  features: List[str],
                                  output_path: str = None) -> go.Figure:
        """
        Create a heatmap showing model disagreement patterns.
        
        Args:
            disagreement_results (Dict): Results from disagreement analysis
            features (List[str]): List of feature names
            output_path (str): Path to save the plot
            
        Returns:
            go.Figure: Plotly figure object
        """
        # Prepare disagreement matrix
        disagreement_matrix = np.zeros((len(features), len(features)))
        
        for i, feature1 in enumerate(features):
            for j, feature2 in enumerate(features):
                if feature1 in disagreement_results and feature2 in disagreement_results[feature1]:
                    disagreement_matrix[i, j] = disagreement_results[feature1][feature2]
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=disagreement_matrix,
            x=features,
            y=features,
            colorscale='RdBu',
            zmid=np.mean(disagreement_matrix),
            text=np.round(disagreement_matrix, 3),
            texttemplate='%{text}',
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title='Model Disagreement Patterns',
            xaxis_title='Feature',
            yaxis_title='Feature',
            width=1000,
            height=1000,
            template='plotly_white'
        )
        
        if output_path:
            fig.write_html(output_path)
            
        return fig
    
    def create_explanation_consistency_network(self,
                                            explanation_comparison: Dict,
                                            threshold: float = 0.5,
                                            output_path: str = None) -> go.Figure:
        """
        Create a network visualization of explanation consistency across models.
        
        Args:
            explanation_comparison (Dict): Results from explanation comparison
            threshold (float): Minimum consistency score to show connection
            output_path (str): Path to save the plot
            
        Returns:
            go.Figure: Plotly figure object
        """
        # Create network
        G = nx.Graph()
        
        # Add nodes for models and features
        models = set()
        features = set()
        
        def get_score_from_value(value):
            """Helper function to extract score from various value types."""
            if isinstance(value, (int, float)):
                return float(value)
            elif isinstance(value, list):
                # Try to find a numeric value in the list
                for item in value:
                    if isinstance(item, (int, float)):
                        return float(item)
                    elif isinstance(item, dict) and 'score' in item:
                        return float(item['score'])
                return 0.0
            elif isinstance(value, dict):
                if 'score' in value:
                    return float(value['score'])
                # Try to find any numeric value in the dict
                for v in value.values():
                    if isinstance(v, (int, float)):
                        return float(v)
                return 0.0
            elif isinstance(value, str):
                try:
                    return float(value)
                except ValueError:
                    return 0.0
            return 0.0
        
        # Handle both dictionary and list inputs
        if isinstance(explanation_comparison, dict):
            for model, explanations in explanation_comparison.items():
                models.add(model)
                if isinstance(explanations, dict):
                    for feature, score in explanations.items():
                        features.add(feature)
                elif isinstance(explanations, list):
                    for feature_data in explanations:
                        if isinstance(feature_data, dict):
                            features.add(feature_data.get('feature'))
                        else:
                            features.add(str(feature_data))
        elif isinstance(explanation_comparison, list):
            for item in explanation_comparison:
                if isinstance(item, dict):
                    models.add(item.get('model', 'Unknown'))
                    features.add(item.get('feature', 'Unknown'))
        
        # Add nodes with different colors for models and features
        for model in models:
            G.add_node(model, node_type='model')
        for feature in features:
            if feature:  # Skip None values
                G.add_node(feature, node_type='feature')
        
        # Add edges for consistent explanations
        edge_weights = []
        if isinstance(explanation_comparison, dict):
            for model, explanations in explanation_comparison.items():
                if isinstance(explanations, dict):
                    for feature, value in explanations.items():
                        score = get_score_from_value(value)
                        if score > threshold:
                            G.add_edge(model, feature, weight=score)
                            edge_weights.append(score)
                elif isinstance(explanations, list):
                    for feature_data in explanations:
                        if isinstance(feature_data, dict):
                            score = get_score_from_value(feature_data.get('score', 0))
                            feature = feature_data.get('feature')
                            if score > threshold and feature:
                                G.add_edge(model, feature, weight=score)
                                edge_weights.append(score)
        elif isinstance(explanation_comparison, list):
            for item in explanation_comparison:
                if isinstance(item, dict):
                    model = item.get('model', 'Unknown')
                    feature = item.get('feature', 'Unknown')
                    score = get_score_from_value(item.get('score', 0))
                    if score > threshold:
                        G.add_edge(model, feature, weight=score)
                        edge_weights.append(score)
        
        # If no edges were created, return an empty figure
        if not edge_weights:
            fig = go.Figure()
            fig.update_layout(
                title='No significant explanation consistencies found',
                showlegend=False,
                width=1000,
                height=800
            )
            if output_path:
                fig.write_html(output_path)
            return fig
        
        # Create layout
        pos = nx.spring_layout(G)
        
        # Create edge trace
        edge_x, edge_y = [], []
        edge_text = []
        
        for edge in G.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_text.append(f"Consistency: {edge[2]['weight']:.3f}")
        
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            mode='lines',
            line=dict(
                width=1,
                color=edge_weights,
                colorscale='Viridis',
                colorbar=dict(title='Consistency Score')
            ),
            hovertext=edge_text,
            hoverinfo='text'
        )
        
        # Create node traces for models and features
        model_x, model_y = [], []
        feature_x, feature_y = [], []
        model_text, feature_text = [], []
        
        for node in G.nodes(data=True):
            x, y = pos[node[0]]
            if node[1]['node_type'] == 'model':
                model_x.append(x)
                model_y.append(y)
                model_text.append(node[0])
            else:
                feature_x.append(x)
                feature_y.append(y)
                feature_text.append(node[0])
        
        model_trace = go.Scatter(
            x=model_x, y=model_y,
            mode='markers+text',
            name='Models',
            marker=dict(
                size=20,
                color='lightblue',
                line=dict(color='black', width=1)
            ),
            text=model_text,
            textposition='bottom center',
            hoverinfo='text'
        )
        
        feature_trace = go.Scatter(
            x=feature_x, y=feature_y,
            mode='markers+text',
            name='Features',
            marker=dict(
                size=15,
                color='lightgreen',
                symbol='diamond',
                line=dict(color='black', width=1)
            ),
            text=feature_text,
            textposition='bottom center',
            hoverinfo='text'
        )
        
        # Create figure
        fig = go.Figure(data=[edge_trace, model_trace, feature_trace])
        
        fig.update_layout(
            title='Explanation Consistency Network',
            showlegend=True,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            width=1000,
            height=800
        )
        
        if output_path:
            fig.write_html(output_path)
            
        return fig
    
    def create_threshold_comparison_plot(self,
                                      threshold_results: Dict,
                                      feature: str,
                                      output_path: str = None) -> go.Figure:
        """
        Create a plot comparing feature thresholds across models.
        
        Args:
            threshold_results (Dict): Results from threshold analysis
            feature (str): Feature to plot thresholds for
            output_path (str): Path to save the plot
            
        Returns:
            go.Figure: Plotly figure object
        """
        fig = go.Figure()
        
        # Add threshold points for each model
        for model, thresholds in threshold_results.items():
            if feature in thresholds:
                threshold_values = thresholds[feature]['thresholds']
                importance = thresholds[feature]['importance']
                
                fig.add_trace(go.Scatter(
                    x=threshold_values,
                    y=[importance] * len(threshold_values),
                    name=model,
                    mode='markers',
                    marker=dict(
                        size=15,
                        symbol='circle',
                        opacity=0.7
                    ),
                    hovertext=[f"{model}<br>Threshold: {val:.3f}<br>Importance: {importance:.3f}"
                              for val in threshold_values]
                ))
        
        fig.update_layout(
            title=f'Threshold Comparison - {feature}',
            xaxis_title='Threshold Value',
            yaxis_title='Feature Importance',
            height=600,
            template='plotly_white',
            showlegend=True,
            hovermode='closest'
        )
        
        if output_path:
            fig.write_html(output_path)
            
        return fig 