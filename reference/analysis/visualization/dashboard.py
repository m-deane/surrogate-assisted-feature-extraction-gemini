"""
Interactive dashboard for visualization and exploration of analysis results.
"""

import dash
from dash import dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import json
import os
from typing import Dict, List
import networkx as nx
from dash.exceptions import PreventUpdate

class AnalysisDashboard:
    def __init__(self, results_path: str, plots_dir: str):
        """
        Initialize the dashboard with analysis results.
        
        Args:
            results_path (str): Path to analysis JSON results
            plots_dir (str): Directory containing plots
        """
        self.results_path = results_path
        self.plots_dir = plots_dir
        self.results = self._load_results()
        self.app = self._create_app()
    
    def _load_results(self) -> Dict:
        """Load analysis results from JSON file."""
        if not os.path.exists(self.results_path):
            raise FileNotFoundError(f"Results file not found: {self.results_path}")
        
        with open(self.results_path, 'r') as f:
            return json.load(f)
    
    def _create_app(self) -> dash.Dash:
        """Create and configure the Dash app."""
        app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        
        # Create layout with tabs
        app.layout = html.Div([
            html.H1("PREEM Analysis Dashboard", className="dashboard-title"),
            
            dbc.Tabs([
                # Overview Tab
                dbc.Tab(label="Overview", children=[
                    html.Div([
                        html.H3("Model Performance", className="mt-4"),
                        dbc.Row([
                            dbc.Col(dcc.Graph(id='performance-gauge', figure=self._create_performance_gauge()), width=6),
                            dbc.Col(html.Div([
                                html.H4("Analysis Summary"),
                                html.Div(id="summary-stats")
                            ]), width=6)
                        ]),
                        
                        html.H3("Feature Importance", className="mt-4"),
                        dbc.Row([
                            dbc.Col(dcc.Graph(id='feature-importance-bar'), width=12)
                        ]),
                        
                        html.H3("Analysis Timeline", className="mt-4"),
                        dcc.Graph(id='analysis-timeline')
                    ])
                ]),
                
                # Feature Analysis Tab
                dbc.Tab(label="Feature Analysis", children=[
                    html.Div([
                        html.H3("Feature Selector", className="mt-2"),
                        dcc.Dropdown(
                            id='feature-selector',
                            options=[{'label': feature, 'value': feature} 
                                    for feature in self._get_feature_list()],
                            value=self._get_feature_list()[0] if self._get_feature_list() else None
                        ),
                        
                        dbc.Tabs([
                            dbc.Tab(label="Threshold Analysis", children=[
                                dcc.Graph(id='threshold-plot')
                            ]),
                            dbc.Tab(label="Distribution Analysis", children=[
                                dcc.Graph(id='distribution-plot')
                            ]),
                            dbc.Tab(label="Temporal Patterns", children=[
                                dcc.Graph(id='temporal-plot')
                            ])
                        ]),
                        
                        html.H4("Feature Details", className="mt-4"),
                        html.Div(id='feature-details', className="feature-details-box")
                    ])
                ]),
                
                # Interaction Analysis Tab
                dbc.Tab(label="Interaction Analysis", children=[
                    html.Div([
                        html.H3("Interaction Strength", className="mt-2"),
                        dcc.Graph(id='interaction-heatmap'),
                        
                        html.H3("Feature Network", className="mt-4"),
                        dcc.Graph(id='feature-network'),
                        
                        html.H4("Interaction Filter", className="mt-2"),
                        dcc.Slider(
                            id='interaction-threshold-slider',
                            min=0,
                            max=1,
                            step=0.05,
                            value=0.2,
                            marks={i/10: str(i/10) for i in range(0, 11, 2)}
                        )
                    ])
                ]),
                
                # Rule Analysis Tab
                dbc.Tab(label="Rule Analysis", children=[
                    html.Div([
                        html.H3("Decision Rules", className="mt-2"),
                        dcc.Graph(id='rule-heatmap'),
                        
                        html.H3("Decision Paths", className="mt-4"),
                        dcc.Graph(id='decision-sankey'),
                        
                        html.H4("Rule Selection", className="mt-2"),
                        dcc.Dropdown(
                            id='rule-selector',
                            options=[{'label': f"Rule {i+1}", 'value': i} 
                                    for i in range(self._get_rule_count())],
                            value=0
                        ),
                        
                        html.Div(id='rule-details', className="rule-details-box")
                    ])
                ]),
                
                # Advanced Analysis Tab
                dbc.Tab(label="Advanced Analysis", children=[
                    html.Div([
                        html.H3("Nonlinear Relationships", className="mt-2"),
                        dcc.Graph(id='nonlinear-plot'),
                        
                        html.H3("Causal Analysis", className="mt-4"),
                        dcc.Graph(id='causal-plot'),
                        
                        html.H3("Interaction Analysis", className="mt-4"),
                        dbc.Row([
                            dbc.Col(html.Div([
                                html.Label("Feature 1"),
                                dcc.Dropdown(
                                    id='feature1-selector',
                                    options=[{'label': feature, 'value': feature} 
                                            for feature in self._get_feature_list()],
                                    value=self._get_feature_list()[0] if self._get_feature_list() else None
                                )
                            ]), width=6),
                            dbc.Col(html.Div([
                                html.Label("Feature 2"),
                                dcc.Dropdown(
                                    id='feature2-selector',
                                    options=[{'label': feature, 'value': feature} 
                                            for feature in self._get_feature_list()],
                                    value=self._get_feature_list()[1] if len(self._get_feature_list()) > 1 else None
                                )
                            ]), width=6)
                        ]),
                        dcc.Graph(id='interaction-detail-plot')
                    ])
                ])
            ], className="main-tabs")
        ], className="dashboard-container")
        
        # Register callbacks
        self._register_callbacks(app)
        
        return app
    
    def _register_callbacks(self, app):
        """Register all dashboard callbacks."""
        
        @app.callback(
            Output('feature-importance-bar', 'figure'),
            Input('feature-importance-bar', 'id')
        )
        def update_feature_importance(_):
            return self._create_feature_importance_bar()
        
        @app.callback(
            Output('interaction-heatmap', 'figure'),
            Input('interaction-heatmap', 'id')
        )
        def update_interaction_heatmap(_):
            return self._create_interaction_heatmap()
        
        @app.callback(
            Output('feature-network', 'figure'),
            Input('interaction-threshold-slider', 'value')
        )
        def update_feature_network(threshold):
            return self._create_feature_network(threshold)
        
        @app.callback(
            Output('threshold-plot', 'figure'),
            Input('feature-selector', 'value')
        )
        def update_threshold_plot(feature):
            if not feature:
                raise PreventUpdate
            return self._create_threshold_plot(feature)
        
        @app.callback(
            Output('feature-details', 'children'),
            Input('feature-selector', 'value')
        )
        def update_feature_details(feature):
            if not feature:
                return "Select a feature to view details"
            
            threshold_info = self.results.get('threshold_analysis', {}).get(feature, {})
            
            if not threshold_info:
                return f"No detailed analysis available for {feature}"
            
            importance = threshold_info.get('importance', 0)
            thresholds = threshold_info.get('thresholds', [])
            
            details = [
                html.H5(f"Feature: {feature}"),
                html.P(f"Importance: {importance:.4f}"),
                html.P(f"Number of thresholds: {len(thresholds)}"),
            ]
            
            if thresholds:
                details.append(html.P("Threshold values:"))
                details.append(html.Ul([
                    html.Li(f"{threshold:.4f}") for threshold in thresholds
                ]))
            
            # Add additional details if available
            if 'nonlinear_analysis' in self.results and feature in self.results['nonlinear_analysis']:
                nonlinear_info = self.results['nonlinear_analysis'][feature]
                details.append(html.P(f"Nonlinearity score: {nonlinear_info.get('nonlinearity_score', 0):.4f}"))
            
            return details
        
        @app.callback(
            Output('rule-details', 'children'),
            Input('rule-selector', 'value')
        )
        def update_rule_details(rule_idx):
            if rule_idx is None:
                return "Select a rule to view details"
            
            rules = self._get_rules()
            if not rules or rule_idx >= len(rules):
                return "Rule information not available"
            
            rule = rules[rule_idx]
            
            return [
                html.H5(f"Rule {rule_idx + 1}"),
                html.P(f"Confidence: {rule.get('confidence', 0):.4f}"),
                html.P(f"Coverage: {rule.get('coverage', 0):.4f}"),
                html.P("Conditions:"),
                html.Ul([
                    html.Li(f"{cond}") for cond in rule.get('conditions', [])
                ]),
                html.P(f"Prediction: {rule.get('prediction', 0):.4f}")
            ]
        
        @app.callback(
            Output('interaction-detail-plot', 'figure'),
            [Input('feature1-selector', 'value'),
             Input('feature2-selector', 'value')]
        )
        def update_interaction_detail(feature1, feature2):
            if not feature1 or not feature2:
                raise PreventUpdate
            return self._create_interaction_detail(feature1, feature2)
        
        @app.callback(
            Output('summary-stats', 'children'),
            Input('summary-stats', 'id')
        )
        def update_summary_stats(_):
            # Get performance metrics
            performance = self.results.get('performance_metrics', {})
            threshold_analysis = self.results.get('threshold_analysis', {})
            
            # Calculate summary statistics
            feature_count = len(threshold_analysis)
            total_thresholds = sum(len(info.get('thresholds', [])) for info in threshold_analysis.values())
            avg_thresholds = total_thresholds / feature_count if feature_count > 0 else 0
            
            # Get top features
            top_features = sorted(
                [(feature, info.get('importance', 0)) for feature, info in threshold_analysis.items()],
                key=lambda x: x[1], reverse=True
            )[:5]
            
            return [
                html.P(f"Total features analyzed: {feature_count}"),
                html.P(f"Total identified thresholds: {total_thresholds}"),
                html.P(f"Average thresholds per feature: {avg_thresholds:.2f}"),
                html.P("Top 5 important features:"),
                html.Ul([
                    html.Li(f"{feature}: {importance:.4f}") 
                    for feature, importance in top_features
                ])
            ]
    
    def _create_performance_gauge(self) -> go.Figure:
        """Create a gauge chart of model performance."""
        performance = self.results.get('performance_metrics', {})
        r2_score = performance.get('surrogate_r2', 0)
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=r2_score,
            title={'text': "Surrogate Model R² Score"},
            gauge={
                'axis': {'range': [0, 1]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 0.3], 'color': "red"},
                    {'range': [0.3, 0.7], 'color': "orange"},
                    {'range': [0.7, 1], 'color': "green"}
                ]
            }
        ))
        
        return fig
    
    def _create_feature_importance_bar(self) -> go.Figure:
        """Create a bar chart of feature importance."""
        threshold_analysis = self.results.get('threshold_analysis', {})
        
        features = []
        importance = []
        
        for feature, info in threshold_analysis.items():
            features.append(feature)
            importance.append(info.get('importance', 0))
        
        # Sort by importance
        sorted_indices = np.argsort(importance)[::-1]
        features = [features[i] for i in sorted_indices]
        importance = [importance[i] for i in sorted_indices]
        
        # Limit to top 15 features
        if len(features) > 15:
            features = features[:15]
            importance = importance[:15]
        
        fig = go.Figure(go.Bar(
            x=features,
            y=importance,
            marker_color='darkblue'
        ))
        
        fig.update_layout(
            title="Feature Importance",
            xaxis_title="Feature",
            yaxis_title="Importance Score",
            height=500
        )
        
        return fig
    
    def _create_interaction_heatmap(self) -> go.Figure:
        """Create a heatmap of feature interactions."""
        interaction_analysis = self.results.get('interaction_analysis', {})
        
        if not interaction_analysis:
            # Create empty figure with message
            fig = go.Figure()
            fig.update_layout(
                title="No interaction data available",
                annotations=[{
                    'text': "Interaction analysis results not found",
                    'showarrow': False,
                    'font': {'size': 20}
                }]
            )
            return fig
        
        # Convert interaction dictionary to matrix
        features = set()
        for f1, f2 in interaction_analysis.keys():
            features.add(f1)
            features.add(f2)
        
        features = sorted(list(features))
        n_features = len(features)
        
        # Create interaction matrix
        matrix = np.zeros((n_features, n_features))
        for (f1, f2), strength in interaction_analysis.items():
            i = features.index(f1)
            j = features.index(f2)
            matrix[i, j] = strength
            matrix[j, i] = strength  # Mirror for symmetry
        
        fig = go.Figure(data=go.Heatmap(
            z=matrix,
            x=features,
            y=features,
            colorscale='Viridis',
            zmin=0,
            zmax=max(0.0001, np.max(matrix)),
            colorbar={"title": "Interaction Strength"}
        ))
        
        fig.update_layout(
            title="Feature Interaction Heatmap",
            xaxis={"title": "Feature"},
            yaxis={"title": "Feature"},
            height=800,
            width=800
        )
        
        return fig
    
    def _create_feature_network(self, threshold: float = 0.2) -> go.Figure:
        """Create a network visualization of feature interactions."""
        interaction_analysis = self.results.get('interaction_analysis', {})
        threshold_analysis = self.results.get('threshold_analysis', {})
        
        if not interaction_analysis or not threshold_analysis:
            # Create empty figure with message
            fig = go.Figure()
            fig.update_layout(
                title="No interaction data available",
                annotations=[{
                    'text': "Interaction analysis results not found",
                    'showarrow': False,
                    'font': {'size': 20}
                }]
            )
            return fig
        
        # Create networkx graph
        G = nx.Graph()
        
        # Add nodes with importance as size
        for feature, info in threshold_analysis.items():
            importance = info.get('importance', 0)
            G.add_node(feature, importance=importance, 
                     threshold_count=len(info.get('thresholds', [])))
        
        # Add edges with interaction strength as weight
        for (f1, f2), strength in interaction_analysis.items():
            if strength >= threshold:  # Filter by threshold
                G.add_edge(f1, f2, weight=strength)
        
        # Calculate node positions using force-directed layout
        pos = nx.spring_layout(G, k=0.5, iterations=50)
        
        # Create node trace
        node_x = []
        node_y = []
        node_text = []
        node_size = []
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            importance = G.nodes[node].get('importance', 0)
            threshold_count = G.nodes[node].get('threshold_count', 0)
            
            node_text.append(
                f"Feature: {node}<br>"
                f"Importance: {importance:.4f}<br>"
                f"Thresholds: {threshold_count}"
            )
            
            # Scale node size by importance
            node_size.append(importance * 100 + 10)
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            marker=dict(
                showscale=True,
                colorscale='YlGnBu',
                color=[G.nodes[node].get('importance', 0) for node in G.nodes()],
                size=node_size,
                colorbar=dict(
                    thickness=15,
                    title='Feature Importance',
                    xanchor='left',
                    titleside='right'
                ),
                line_width=2
            ),
            text=[node for node in G.nodes()],
            textposition="top center",
            hovertext=node_text
        )
        
        # Create edge trace
        edge_x = []
        edge_y = []
        edge_text = []
        edge_width = []
        
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
            edge_text.append(
                f"{edge[0]} - {edge[1]}<br>"
                f"Interaction: {weight:.4f}"
            )
            edge_width.append(weight * 10)
        
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=edge_width, color='rgba(150,150,150,0.7)'),
            hoverinfo='text',
            mode='lines',
            hovertext=edge_text
        )
        
        # Create figure
        fig = go.Figure(data=[edge_trace, node_trace],
                      layout=go.Layout(
                          title=f"Feature Interaction Network (threshold: {threshold})",
                          titlefont_size=16,
                          showlegend=False,
                          hovermode='closest',
                          margin=dict(b=20, l=5, r=5, t=40),
                          xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                          yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                      ))
        
        return fig
    
    def _create_threshold_plot(self, feature: str) -> go.Figure:
        """Create a plot of thresholds for a specific feature."""
        threshold_info = self.results.get('threshold_analysis', {}).get(feature, {})
        
        if not threshold_info:
            # Create empty figure with message
            fig = go.Figure()
            fig.update_layout(
                title=f"No threshold data available for {feature}",
                annotations=[{
                    'text': "Threshold analysis results not found",
                    'showarrow': False,
                    'font': {'size': 20}
                }]
            )
            return fig
        
        thresholds = threshold_info.get('thresholds', [])
        
        fig = go.Figure()
        
        # Add histogram or distribution of feature (if we had the data)
        # Here we're just showing the thresholds
        
        # Add threshold lines
        for i, threshold in enumerate(thresholds):
            fig.add_shape(
                type="line",
                x0=threshold,
                y0=0,
                x1=threshold,
                y1=1,
                line=dict(
                    color="red",
                    width=2,
                    dash="dash",
                ),
                xref="x",
                yref="paper"
            )
            
            # Add annotations
            fig.add_annotation(
                x=threshold,
                y=0.9 - (i * 0.1),  # Stagger annotations
                text=f"Threshold: {threshold:.4f}",
                showarrow=False,
                font=dict(color="red")
            )
        
        fig.update_layout(
            title=f"Threshold Analysis for {feature}",
            xaxis_title=feature,
            yaxis_title="Density",
            height=500
        )
        
        return fig
    
    def _create_interaction_detail(self, feature1: str, feature2: str) -> go.Figure:
        """Create a detailed interaction plot between two features."""
        interaction_key = (feature1, feature2)
        alt_key = (feature2, feature1)
        
        interaction_analysis = self.results.get('interaction_analysis', {})
        
        interaction_strength = interaction_analysis.get(interaction_key, 
                                                      interaction_analysis.get(alt_key, 0))
        
        # Create a simple scatter plot (would be better with actual data)
        fig = go.Figure()
        
        fig.add_annotation(
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            text=f"Interaction Strength: {interaction_strength:.4f}",
            showarrow=False,
            font=dict(size=16)
        )
        
        fig.update_layout(
            title=f"Interaction between {feature1} and {feature2}",
            xaxis_title=feature1,
            yaxis_title=feature2,
            height=500
        )
        
        return fig
    
    def _get_feature_list(self) -> List[str]:
        """Get list of all features in the analysis."""
        threshold_analysis = self.results.get('threshold_analysis', {})
        return sorted(list(threshold_analysis.keys()))
    
    def _get_rules(self) -> List[Dict]:
        """Get list of decision rules."""
        # Check if we have rule analysis results
        if 'rule_analysis' not in self.results:
            return []
        
        return self.results['rule_analysis'].get('rules', [])
    
    def _get_rule_count(self) -> int:
        """Get the number of rules in the analysis."""
        rules = self._get_rules()
        return len(rules)
    
    def run_server(self, debug: bool = True, port: int = 8050):
        """Run the dashboard server."""
        self.app.run_server(debug=debug, port=port)


def create_dashboard(results_path: str, plots_dir: str) -> AnalysisDashboard:
    """
    Create and initialize a dashboard instance.
    
    Args:
        results_path (str): Path to analysis JSON results
        plots_dir (str): Directory containing plots
    
    Returns:
        AnalysisDashboard: Configured dashboard instance
    """
    return AnalysisDashboard(results_path, plots_dir)


if __name__ == "__main__":
    # Example usage
    results_path = "results/analysis_results.json"
    plots_dir = "plots/advanced"
    
    dashboard = create_dashboard(results_path, plots_dir)
    dashboard.run_server(debug=True) 