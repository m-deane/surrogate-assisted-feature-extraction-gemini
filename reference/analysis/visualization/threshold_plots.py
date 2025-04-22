"""
Module for creating visualizations of threshold analysis results.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List
import plotly.graph_objects as go

class ThresholdVisualizer:
    def __init__(self):
        """Initialize the threshold visualizer."""
        sns.set_style("whitegrid")  # Use seaborn's whitegrid style
        
    def plot_feature_thresholds(self, 
                              threshold_analysis: Dict,
                              output_path: str = None):
        """
        Create a visualization of feature thresholds.
        
        Args:
            threshold_analysis (Dict): Dictionary containing threshold analysis results
            output_path (str): Path to save the plot
        """
        # Extract features and their importance scores
        features = list(threshold_analysis.keys())
        importance_scores = [data['importance'] for data in threshold_analysis.values()]
        
        # Sort by importance
        sorted_indices = np.argsort(importance_scores)[::-1]
        features = [features[i] for i in sorted_indices]
        importance_scores = [importance_scores[i] for i in sorted_indices]
        
        # Create the plot
        plt.figure(figsize=(12, 6))
        bars = plt.barh(range(len(features)), importance_scores)
        plt.yticks(range(len(features)), features)
        plt.xlabel('Feature Importance')
        plt.title('Feature Importance with Threshold Analysis')
        
        # Add threshold annotations
        for idx, feature in enumerate(features):
            thresholds = threshold_analysis[feature]['thresholds']
            if thresholds:
                threshold_text = f"Thresholds: {', '.join([f'{t:.2f}' for t in thresholds])}"
                plt.text(bars[idx].get_width(), idx, 
                        f'  {threshold_text}',
                        va='center')
        
        plt.tight_layout()
        if output_path:
            plt.savefig(output_path)
        plt.close()
        
    def plot_threshold_distribution(self,
                                  X: pd.DataFrame,
                                  feature_name: str,
                                  thresholds: List[float],
                                  output_path: str = None):
        """
        Create a distribution plot showing where thresholds fall on feature values.
        
        Args:
            X (pd.DataFrame): Feature matrix
            feature_name (str): Name of the feature to plot
            thresholds (List[float]): List of thresholds for this feature
            output_path (str): Path to save the plot
        """
        plt.figure(figsize=(10, 6))
        
        # Plot feature distribution
        sns.histplot(X[feature_name], kde=True)
        
        # Add threshold lines
        for threshold in thresholds:
            plt.axvline(x=threshold, color='r', linestyle='--', alpha=0.5)
            plt.text(threshold, plt.ylim()[1], f'{threshold:.2f}',
                    rotation=90, va='top')
        
        plt.title(f'Distribution of {feature_name} with Decision Thresholds')
        plt.xlabel(feature_name)
        plt.ylabel('Count')
        
        if output_path:
            plt.savefig(output_path)
        plt.close()
        
    def create_interactive_threshold_plot(self,
                                       X: pd.DataFrame,
                                       feature_name: str,
                                       thresholds: List[float],
                                       output_path: str = None):
        """
        Create an interactive plot using plotly showing thresholds and distributions.
        
        Args:
            X (pd.DataFrame): Feature matrix
            feature_name (str): Name of the feature to plot
            thresholds (List[float]): List of thresholds for this feature
            output_path (str): Path to save the plot as HTML
        """
        # Create histogram trace
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=X[feature_name],
            name='Distribution',
            nbinsx=30,
            opacity=0.75
        ))
        
        # Add threshold lines
        for threshold in thresholds:
            fig.add_vline(
                x=threshold,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Threshold: {threshold:.2f}",
                annotation_position="top right"
            )
        
        fig.update_layout(
            title=f'Interactive Distribution of {feature_name} with Decision Thresholds',
            xaxis_title=feature_name,
            yaxis_title='Count',
            showlegend=True
        )
        
        if output_path:
            fig.write_html(output_path)
            
        return fig 