"""
Module for advanced visualizations and interpretations of analysis results.
"""

import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
import networkx as nx
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

class AdvancedVisualizer:
    def __init__(self):
        """Initialize the advanced visualizer."""
        self.color_palette = px.colors.qualitative.Set3
        
    def create_feature_importance_sunburst(self,
                                         threshold_analysis: Dict,
                                         output_path: str = None) -> go.Figure:
        """
        Create an interactive sunburst plot of feature importances and thresholds.
        
        Args:
            threshold_analysis (Dict): Threshold analysis results
            output_path (str): Path to save the plot
            
        Returns:
            go.Figure: Plotly figure object
        """
        # Prepare data for sunburst plot
        labels = []
        parents = []
        values = []
        
        # Sort features by importance
        sorted_features = sorted(
            threshold_analysis.items(),
            key=lambda x: x[1]['importance'],
            reverse=True
        )
        
        # Add root
        labels.append('Features')
        parents.append('')
        values.append(1)
        
        # Add features and thresholds
        for feature, data in sorted_features:
            # Add feature
            labels.append(feature)
            parents.append('Features')
            values.append(data['importance'])
            
            # Add thresholds
            if data['thresholds']:
                for threshold in sorted(data['thresholds']):
                    labels.append(f'{threshold:.2f}')
                    parents.append(feature)
                    values.append(data['importance'] / len(data['thresholds']))
        
        # Create sunburst plot
        fig = go.Figure(go.Sunburst(
            labels=labels,
            parents=parents,
            values=values,
            branchvalues='total'
        ))
        
        fig.update_layout(
            title='Feature Importance Hierarchy',
            width=800,
            height=800
        )
        
        if output_path:
            fig.write_html(output_path)
            
        return fig
    
    def create_interaction_network_3d(self,
                                    interaction_analysis: Dict,
                                    threshold_analysis: Dict,
                                    output_path: str = None) -> go.Figure:
        """
        Create an interactive 3D network visualization of feature interactions.
        
        Args:
            interaction_analysis (Dict): Feature interaction results
            threshold_analysis (Dict): Threshold analysis results
            output_path (str): Path to save the plot
            
        Returns:
            go.Figure: Plotly figure object
        """
        # Create network layout
        G = nx.Graph()
        
        # Add nodes with importance as size
        for feature, data in threshold_analysis.items():
            G.add_node(feature, importance=data['importance'])
            
        # Add edges with interaction strength
        for (f1, f2), strength in interaction_analysis.items():
            G.add_edge(f1, f2, weight=strength)
            
        # Get 3D layout
        pos = nx.spring_layout(G, dim=3, k=1/np.sqrt(G.number_of_nodes()))
        
        # Create node trace
        node_x = []
        node_y = []
        node_z = []
        node_size = []
        node_text = []
        
        for node in G.nodes():
            x, y, z = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_z.append(z)
            node_size.append(G.nodes[node]['importance'] * 100)
            node_text.append(f"{node}<br>Importance: {G.nodes[node]['importance']:.4f}")
            
        node_trace = go.Scatter3d(
            x=node_x, y=node_y, z=node_z,
            mode='markers',
            marker=dict(
                size=node_size,
                color=list(range(len(node_x))),
                colorscale='Viridis',
                opacity=0.8
            ),
            text=node_text,
            hoverinfo='text'
        )
        
        # Create edge traces
        edge_x = []
        edge_y = []
        edge_z = []
        edge_text = []
        
        for edge in G.edges():
            x0, y0, z0 = pos[edge[0]]
            x1, y1, z1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_z.extend([z0, z1, None])
            edge_text.append(f"Interaction: {G.edges[edge]['weight']:.4f}")
            
        edge_trace = go.Scatter3d(
            x=edge_x, y=edge_y, z=edge_z,
            mode='lines',
            line=dict(
                width=2,
                color='rgba(100,100,100,0.5)'
            ),
            text=edge_text,
            hoverinfo='text'
        )
        
        # Create figure
        fig = go.Figure(data=[edge_trace, node_trace])
        
        fig.update_layout(
            title='3D Feature Interaction Network',
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
    
    def create_interaction_heatmap(self,
                                 interaction_analysis: Dict,
                                 output_path: str = None) -> go.Figure:
        """
        Create an interactive heatmap of feature interactions.
        
        Args:
            interaction_analysis (Dict): Feature interaction results
            output_path (str): Path to save the plot
            
        Returns:
            go.Figure: Plotly figure object
        """
        # Get unique features
        features = set()
        for f1, f2 in interaction_analysis.keys():
            features.add(f1)
            features.add(f2)
        features = sorted(list(features))
        
        # Create interaction matrix
        matrix = np.zeros((len(features), len(features)))
        for (f1, f2), strength in interaction_analysis.items():
            i = features.index(f1)
            j = features.index(f2)
            matrix[i, j] = strength
            matrix[j, i] = strength
            
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=matrix,
            x=features,
            y=features,
            colorscale='RdBu',
            zmid=0,
            text=matrix.round(4),
            texttemplate='%{text}',
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title='Feature Interaction Heatmap',
            xaxis_title='Feature',
            yaxis_title='Feature',
            width=1000,
            height=1000
        )
        
        if output_path:
            fig.write_html(output_path)
            
        return fig
    
    def generate_interpretation_report(self,
                                    threshold_analysis: Dict,
                                    interaction_analysis: Dict,
                                    X: pd.DataFrame,
                                    output_path: str = None) -> str:
        """
        Generate a detailed interpretation report of the analysis results.
        
        Args:
            threshold_analysis (Dict): Threshold analysis results
            interaction_analysis (Dict): Feature interaction results
            X (pd.DataFrame): Feature matrix
            output_path (str): Path to save the report
            
        Returns:
            str: Markdown formatted report
        """
        report = """# Detailed Analysis Interpretation

## Feature Importance Analysis

"""
        # Sort features by importance
        sorted_features = sorted(
            threshold_analysis.items(),
            key=lambda x: x[1]['importance'],
            reverse=True
        )
        
        # Analyze feature importance distribution
        importances = [data['importance'] for _, data in sorted_features]
        total_importance = sum(importances)
        cumulative_importance = np.cumsum(importances) / total_importance
        
        report += "### Key Features\n\n"
        for i, (feature, data) in enumerate(sorted_features):
            # Calculate percentile statistics
            feature_values = X[feature]
            percentiles = np.percentile(feature_values, [25, 50, 75])
            
            report += f"#### {i+1}. {feature}\n"
            report += f"- Relative Importance: {data['importance']/total_importance:.2%}\n"
            report += f"- Cumulative Importance: {cumulative_importance[i]:.2%}\n"
            report += f"- Distribution Statistics:\n"
            report += f"  * Median: {percentiles[1]:.2f}\n"
            report += f"  * IQR: [{percentiles[0]:.2f}, {percentiles[2]:.2f}]\n"
            
            if data['thresholds']:
                report += "- Decision Thresholds:\n"
                for threshold in sorted(data['thresholds']):
                    # Calculate data split at threshold
                    below_pct = (feature_values < threshold).mean() * 100
                    report += f"  * {threshold:.2f} (splits data {below_pct:.1f}% / {100-below_pct:.1f}%)\n"
            report += "\n"
            
        report += "\n## Feature Interactions\n\n"
        
        # Sort interactions by absolute strength
        sorted_interactions = sorted(
            interaction_analysis.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        
        # Analyze interaction patterns
        report += "### Strong Interaction Patterns\n\n"
        for (f1, f2), strength in sorted_interactions[:5]:
            # Calculate correlation
            correlation = stats.spearmanr(X[f1], X[f2]).correlation
            
            report += f"#### {f1} × {f2}\n"
            report += f"- Interaction Strength: {strength:.4f}\n"
            report += f"- Correlation: {correlation:.4f}\n"
            report += f"- Interpretation: "
            
            if abs(correlation) > 0.7:
                report += "Strong correlation suggests redundant information\n"
            elif abs(correlation) < 0.3:
                report += "Low correlation suggests complementary information\n"
            else:
                report += "Moderate correlation suggests partial information overlap\n"
                
            report += "\n"
            
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report)
                
        return report 