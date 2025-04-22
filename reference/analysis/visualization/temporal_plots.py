"""
Module for creating interactive visualizations of temporal analysis results.
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from typing import Dict, List
import networkx as nx

class TemporalVisualizer:
    def __init__(self):
        """Initialize the temporal visualizer."""
        self.color_palette = px.colors.qualitative.Set3
        
    def create_temporal_decomposition_plot(self,
                                         data: pd.DataFrame,
                                         feature: str,
                                         decomposition_results: Dict,
                                         output_path: str = None) -> go.Figure:
        """
        Create an interactive plot showing temporal decomposition components.
        
        Args:
            data (pd.DataFrame): Original time series data
            feature (str): Feature name to plot
            decomposition_results (Dict): Results from seasonal decomposition
            output_path (str): Path to save the plot
            
        Returns:
            go.Figure: Plotly figure object
        """
        fig = go.Figure()
        
        # Add original data
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data[feature],
            name='Original',
            line=dict(color='blue', width=2)
        ))
        
        # Add trend
        fig.add_trace(go.Scatter(
            x=data.index,
            y=decomposition_results['trend'],
            name='Trend',
            line=dict(color='red', width=2)
        ))
        
        # Add seasonal component
        fig.add_trace(go.Scatter(
            x=data.index,
            y=decomposition_results['seasonal'],
            name='Seasonal',
            line=dict(color='green', width=2)
        ))
        
        # Add residual
        fig.add_trace(go.Scatter(
            x=data.index,
            y=decomposition_results['residual'],
            name='Residual',
            line=dict(color='gray', width=1, dash='dash')
        ))
        
        fig.update_layout(
            title=f'Temporal Decomposition - {feature}',
            xaxis_title='Time',
            yaxis_title='Value',
            height=800,
            template='plotly_white',
            showlegend=True,
            hovermode='x unified'
        )
        
        if output_path:
            fig.write_html(output_path)
            
        return fig
    
    def create_stationarity_plot(self,
                                data: pd.DataFrame,
                                feature: str,
                                stationarity_results: Dict,
                                output_path: str = None) -> go.Figure:
        """
        Create an interactive plot showing stationarity analysis.
        
        Args:
            data (pd.DataFrame): Time series data
            feature (str): Feature name to plot
            stationarity_results (Dict): Results from stationarity analysis
            output_path (str): Path to save the plot
            
        Returns:
            go.Figure: Plotly figure object
        """
        fig = go.Figure()
        
        # Add original data
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data[feature],
            name='Original',
            line=dict(color='blue', width=2)
        ))
        
        # Add rolling mean
        fig.add_trace(go.Scatter(
            x=data.index,
            y=stationarity_results['rolling_mean'],
            name='Rolling Mean',
            line=dict(color='red', width=2)
        ))
        
        # Add rolling std bands
        rolling_std = stationarity_results['rolling_std']
        rolling_mean = stationarity_results['rolling_mean']
        
        fig.add_trace(go.Scatter(
            x=data.index,
            y=rolling_mean + 2*rolling_std,
            name='Upper Band',
            line=dict(color='gray', width=1, dash='dash')
        ))
        
        fig.add_trace(go.Scatter(
            x=data.index,
            y=rolling_mean - 2*rolling_std,
            name='Lower Band',
            line=dict(color='gray', width=1, dash='dash'),
            fill='tonexty'
        ))
        
        # Add test results annotation
        adf_result = f"ADF Statistic: {stationarity_results['adf_statistic']:.2f}<br>"
        adf_result += f"p-value: {stationarity_results['p_value']:.4f}<br>"
        adf_result += f"Is Stationary: {stationarity_results['is_stationary']}"
        
        fig.add_annotation(
            x=0.02,
            y=0.98,
            xref='paper',
            yref='paper',
            text=adf_result,
            showarrow=False,
            bgcolor='white',
            bordercolor='black',
            borderwidth=1
        )
        
        fig.update_layout(
            title=f'Stationarity Analysis - {feature}',
            xaxis_title='Time',
            yaxis_title='Value',
            height=600,
            template='plotly_white',
            showlegend=True,
            hovermode='x unified'
        )
        
        if output_path:
            fig.write_html(output_path)
            
        return fig
    
    def create_causality_network(self,
                               causality_results: Dict,
                               features: List[str],
                               p_value_threshold: float = 0.05,
                               output_path: str = None) -> go.Figure:
        """
        Create an interactive network visualization of Granger causality relationships.
        
        Args:
            causality_results (Dict): Results from Granger causality analysis
            features (List[str]): List of feature names
            p_value_threshold (float): Threshold for significant relationships
            output_path (str): Path to save the plot
            
        Returns:
            go.Figure: Plotly figure object
        """
        # Create network layout
        G = nx.DiGraph()
        
        # Add nodes
        for feature in features:
            G.add_node(feature)
            
        # Add edges for significant causal relationships
        edge_weights = []
        for feature1 in features:
            for feature2 in features:
                if feature1 != feature2:
                    result = causality_results[feature1][feature2]
                    if result['min_p_value'] < p_value_threshold:
                        G.add_edge(feature1, feature2, 
                                 weight=-np.log10(result['min_p_value']),
                                 lag=result['optimal_lag'])
                        edge_weights.append(-np.log10(result['min_p_value']))
        
        # Create layout
        pos = nx.spring_layout(G, k=1/np.sqrt(G.number_of_nodes()), dim=3)
        
        # Create edge trace
        edge_x, edge_y, edge_z = [], [], []
        edge_text = []
        
        for edge in G.edges(data=True):
            x0, y0, z0 = pos[edge[0]]
            x1, y1, z1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_z.extend([z0, z1, None])
            
            edge_text.append(
                f"{edge[0]} → {edge[1]}<br>"
                f"Lag: {edge[2]['lag']}<br>"
                f"Strength: {edge[2]['weight']:.2f}"
            )
        
        edge_trace = go.Scatter3d(
            x=edge_x, y=edge_y, z=edge_z,
            mode='lines',
            line=dict(
                width=2,
                color=edge_weights,
                colorscale='Viridis',
                colorbar=dict(title='Causality Strength<br>(-log10 p-value)')
            ),
            hovertext=edge_text,
            hoverinfo='text'
        )
        
        # Create node trace
        node_x, node_y, node_z = [], [], []
        node_text = []
        
        for node in G.nodes():
            x, y, z = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_z.append(z)
            node_text.append(node)
        
        node_trace = go.Scatter3d(
            x=node_x, y=node_y, z=node_z,
            mode='markers+text',
            marker=dict(
                size=10,
                color='lightblue',
                line=dict(color='black', width=1)
            ),
            text=node_text,
            hoverinfo='text'
        )
        
        # Create figure
        fig = go.Figure(data=[edge_trace, node_trace])
        
        fig.update_layout(
            title='Granger Causality Network',
            showlegend=False,
            scene=dict(
                xaxis=dict(showticklabels=False),
                yaxis=dict(showticklabels=False),
                zaxis=dict(showticklabels=False)
            ),
            width=1000,
            height=1000
        )
        
        if output_path:
            fig.write_html(output_path)
            
        return fig
    
    def create_nonlinearity_plot(self,
                                data: pd.DataFrame,
                                feature: str,
                                nonlinearity_results: Dict,
                                output_path: str = None) -> go.Figure:
        """
        Create an interactive plot showing nonlinearity analysis results.
        
        Args:
            data (pd.DataFrame): Time series data
            feature (str): Feature name to plot
            nonlinearity_results (Dict): Results from nonlinearity analysis
            output_path (str): Path to save the plot
            
        Returns:
            go.Figure: Plotly figure object
        """
        # Create subplots
        fig = go.Figure()
        
        # Plot ACF
        fig.add_trace(go.Scatter(
            x=list(range(len(nonlinearity_results['acf_values']))),
            y=nonlinearity_results['acf_values'],
            name='ACF',
            line=dict(color='blue')
        ))
        
        # Plot PACF
        fig.add_trace(go.Scatter(
            x=list(range(len(nonlinearity_results['pacf_values']))),
            y=nonlinearity_results['pacf_values'],
            name='PACF',
            line=dict(color='red')
        ))
        
        # Add confidence bands
        conf_int = 1.96/np.sqrt(len(data))
        fig.add_hline(y=conf_int, line_dash="dash", line_color="gray")
        fig.add_hline(y=-conf_int, line_dash="dash", line_color="gray")
        
        # Add nonlinearity metrics annotation
        metrics = f"Nonlinearity Score: {nonlinearity_results['nonlinearity_score']:.3f}<br>"
        metrics += f"Skewness: {nonlinearity_results['skewness']:.3f}<br>"
        metrics += f"Kurtosis: {nonlinearity_results['kurtosis']:.3f}<br>"
        metrics += f"Has Volatility Clustering: {nonlinearity_results['volatility_clustering']}"
        
        fig.add_annotation(
            x=0.02,
            y=0.98,
            xref='paper',
            yref='paper',
            text=metrics,
            showarrow=False,
            bgcolor='white',
            bordercolor='black',
            borderwidth=1
        )
        
        fig.update_layout(
            title=f'Nonlinearity Analysis - {feature}',
            xaxis_title='Lag',
            yaxis_title='Correlation',
            height=600,
            template='plotly_white',
            showlegend=True,
            hovermode='x unified'
        )
        
        if output_path:
            fig.write_html(output_path)
            
        return fig 