"""
Module for analyzing and visualizing decision rules from the surrogate model.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import plotly.graph_objects as go
import plotly.express as px
from sklearn.tree import DecisionTreeRegressor, _tree
import networkx as nx
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

class RuleAnalyzer:
    def __init__(self, surrogate_model: DecisionTreeRegressor, X: pd.DataFrame, y: pd.Series):
        """
        Initialize the rule analyzer.
        
        Args:
            surrogate_model (DecisionTreeRegressor): Fitted surrogate decision tree model
            X (pd.DataFrame): Feature matrix
            y (pd.Series): Target variable
        """
        self.model = surrogate_model
        self.X = X
        self.y = y
        self.feature_names = X.columns.tolist()
        
    def extract_decision_rules(self) -> List[Dict]:
        """
        Extract all decision rules from the tree.
        
        Returns:
            List[Dict]: List of decision rules with their properties
        """
        tree = self.model.tree_
        
        def recurse(node: int, path: List[Tuple]) -> List[Dict]:
            """Recursively extract rules from tree."""
            rules = []
            
            if tree.feature[node] != _tree.TREE_UNDEFINED:
                # Internal node
                feature = self.feature_names[tree.feature[node]]
                threshold = tree.threshold[node]
                
                # Left path (<=)
                left_path = path + [(feature, "≤", threshold)]
                rules.extend(recurse(tree.children_left[node], left_path))
                
                # Right path (>)
                right_path = path + [(feature, ">", threshold)]
                rules.extend(recurse(tree.children_right[node], right_path))
            else:
                # Leaf node
                if path:  # Skip empty root
                    rules.append({
                        'conditions': path,
                        'prediction': tree.value[node][0][0],
                        'samples': tree.n_node_samples[node],
                        'impurity': tree.impurity[node]
                    })
            
            return rules
            
        return recurse(0, [])
    
    def analyze_rule_impacts(self, rules: List[Dict]) -> List[Dict]:
        """
        Analyze the impact of each rule on the target variable.
        
        Args:
            rules (List[Dict]): List of decision rules
            
        Returns:
            List[Dict]: Rules with impact analysis
        """
        for rule in rules:
            # Create mask for samples matching the rule
            mask = np.ones(len(self.X), dtype=bool)
            for feature, op, threshold in rule['conditions']:
                if op == "≤":
                    mask &= (self.X[feature] <= threshold)
                else:
                    mask &= (self.X[feature] > threshold)
            
            # Calculate statistics for matching samples
            matching_y = self.y[mask]
            rule.update({
                'matching_samples': sum(mask),
                'coverage': sum(mask) / len(self.X),
                'target_mean': matching_y.mean(),
                'target_std': matching_y.std(),
                'target_range': [matching_y.min(), matching_y.max()],
                'confidence': 1 - (rule['impurity'] / self.y.var())
            })
            
        return rules
    
    def create_rule_network(self, rules: List[Dict]) -> nx.DiGraph:
        """
        Create a network representation of rule relationships.
        
        Args:
            rules (List[Dict]): List of decision rules
            
        Returns:
            nx.DiGraph: NetworkX directed graph of rules
        """
        G = nx.DiGraph()
        
        # Add nodes for features and rules
        feature_nodes = set()
        for rule in rules:
            rule_id = f"Rule_{rules.index(rule)}"
            G.add_node(rule_id, 
                      type='rule',
                      samples=rule['samples'],
                      prediction=rule['prediction'],
                      confidence=rule['confidence'])
            
            # Add feature nodes and edges
            for feature, op, threshold in rule['conditions']:
                if feature not in feature_nodes:
                    G.add_node(feature, type='feature')
                    feature_nodes.add(feature)
                G.add_edge(feature, rule_id, 
                          operation=op,
                          threshold=threshold)
                
        return G
    
    def visualize_rule_network(self, G: nx.DiGraph, output_path: str = None) -> go.Figure:
        """
        Create an interactive visualization of the rule network.
        
        Args:
            G (nx.DiGraph): Rule network graph
            output_path (str): Path to save the plot
            
        Returns:
            go.Figure: Plotly figure object
        """
        pos = nx.spring_layout(G)
        
        # Create node traces
        feature_nodes = [n for n, d in G.nodes(data=True) if d['type'] == 'feature']
        rule_nodes = [n for n, d in G.nodes(data=True) if d['type'] == 'rule']
        
        feature_trace = go.Scatter(
            x=[pos[n][0] for n in feature_nodes],
            y=[pos[n][1] for n in feature_nodes],
            mode='markers+text',
            name='Features',
            marker=dict(size=30, color='lightblue'),
            text=feature_nodes,
            textposition='bottom center'
        )
        
        rule_trace = go.Scatter(
            x=[pos[n][0] for n in rule_nodes],
            y=[pos[n][1] for n in rule_nodes],
            mode='markers+text',
            name='Rules',
            marker=dict(
                size=[G.nodes[n]['samples']/10 for n in rule_nodes],
                color=[G.nodes[n]['confidence'] for n in rule_nodes],
                colorscale='Viridis',
                showscale=True
            ),
            text=[f"{n}<br>Pred: {G.nodes[n]['prediction']:.2f}" for n in rule_nodes],
            textposition='bottom center'
        )
        
        # Create edge traces
        edge_x = []
        edge_y = []
        edge_text = []
        
        for edge in G.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_text.append(
                f"{edge[2]['operation']} {edge[2]['threshold']:.2f}"
                if 'threshold' in edge[2] else ""
            )
            
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            mode='lines+text',
            line=dict(width=1, color='gray'),
            text=edge_text,
            textposition='middle center',
            hoverinfo='text'
        )
        
        # Create figure
        fig = go.Figure(data=[edge_trace, feature_trace, rule_trace])
        
        fig.update_layout(
            title='Decision Rule Network',
            showlegend=True,
            hovermode='closest',
            width=1200,
            height=800
        )
        
        if output_path:
            fig.write_html(output_path)
            
        return fig
    
    def create_rule_impact_heatmap(self, rules: List[Dict], output_path: str = None) -> go.Figure:
        """
        Create a heatmap showing rule impacts on different metrics.
        
        Args:
            rules (List[Dict]): List of analyzed rules
            output_path (str): Path to save the plot
            
        Returns:
            go.Figure: Plotly figure object
        """
        # Prepare data for heatmap
        metrics = ['prediction', 'coverage', 'confidence', 'target_mean', 'target_std']
        data = np.zeros((len(rules), len(metrics)))
        
        for i, rule in enumerate(rules):
            data[i] = [
                rule['prediction'],
                rule['coverage'],
                rule['confidence'],
                rule['target_mean'],
                rule['target_std']
            ]
            
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=data,
            x=metrics,
            y=[f"Rule {i}" for i in range(len(rules))],
            colorscale='RdBu',
            zmid=np.mean(data),
            text=np.round(data, 3),
            texttemplate='%{text}',
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title='Rule Impact Analysis',
            xaxis_title='Metric',
            yaxis_title='Rule',
            width=1000,
            height=len(rules) * 50 + 200
        )
        
        if output_path:
            fig.write_html(output_path)
            
        return fig
    
    def create_rule_path_sankey(self, rules: List[Dict], output_path: str = None) -> go.Figure:
        """
        Create a Sankey diagram showing decision paths.
        
        Args:
            rules (List[Dict]): List of decision rules
            output_path (str): Path to save the plot
            
        Returns:
            go.Figure: Plotly figure object
        """
        # Prepare nodes and links
        nodes = []
        node_dict = {}
        links = []
        
        def get_node_id(name: str) -> int:
            if name not in node_dict:
                node_dict[name] = len(nodes)
                nodes.append(name)
            return node_dict[name]
        
        # Process each rule
        for rule_idx, rule in enumerate(rules):
            prev_node = "Start"
            for feature, op, threshold in rule['conditions']:
                # Add feature node
                curr_node = f"{feature} {op} {threshold:.2f}"
                source = get_node_id(prev_node)
                target = get_node_id(curr_node)
                links.append({
                    'source': source,
                    'target': target,
                    'value': rule['samples']
                })
                prev_node = curr_node
            
            # Add final prediction node
            pred_node = f"Prediction: {rule['prediction']:.2f}"
            source = get_node_id(prev_node)
            target = get_node_id(pred_node)
            links.append({
                'source': source,
                'target': target,
                'value': rule['samples']
            })
        
        # Create Sankey diagram
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=nodes,
                color="blue"
            ),
            link=dict(
                source=[link['source'] for link in links],
                target=[link['target'] for link in links],
                value=[link['value'] for link in links]
            )
        )])
        
        fig.update_layout(
            title='Decision Paths Flow',
            font_size=10,
            width=1500,
            height=800
        )
        
        if output_path:
            fig.write_html(output_path)
            
        return fig
    
    def analyze_rule_conditions(self, rules: List[Dict]) -> pd.DataFrame:
        """
        Analyze the conditions used in rules.
        
        Args:
            rules (List[Dict]): List of decision rules
            
        Returns:
            pd.DataFrame: Analysis of rule conditions
        """
        condition_stats = []
        
        for feature in self.feature_names:
            # Collect all thresholds for this feature
            thresholds = []
            operations = []
            impacts = []
            
            for rule in rules:
                for cond_feature, op, threshold in rule['conditions']:
                    if cond_feature == feature:
                        thresholds.append(threshold)
                        operations.append(op)
                        impacts.append(rule['samples'] / len(self.X))
            
            if thresholds:
                condition_stats.append({
                    'feature': feature,
                    'num_thresholds': len(thresholds),
                    'min_threshold': min(thresholds),
                    'max_threshold': max(thresholds),
                    'mean_threshold': np.mean(thresholds),
                    'std_threshold': np.std(thresholds),
                    'most_common_op': max(set(operations), key=operations.count),
                    'mean_impact': np.mean(impacts)
                })
                
        return pd.DataFrame(condition_stats)
    
    def create_condition_summary_plot(self, 
                                    condition_stats: pd.DataFrame,
                                    output_path: str = None) -> go.Figure:
        """
        Create a summary plot of condition statistics.
        
        Args:
            condition_stats (pd.DataFrame): Condition statistics
            output_path (str): Path to save the plot
            
        Returns:
            go.Figure: Plotly figure object
        """
        # Sort features by mean impact
        condition_stats = condition_stats.sort_values('mean_impact', ascending=False)
        
        # Create figure with secondary y-axis
        fig = go.Figure()
        
        # Add threshold range bars
        fig.add_trace(go.Bar(
            name='Threshold Range',
            x=condition_stats['feature'],
            y=condition_stats['max_threshold'] - condition_stats['min_threshold'],
            error_y=dict(
                type='data',
                array=condition_stats['std_threshold'],
                visible=True
            ),
            yaxis='y'
        ))
        
        # Add impact line
        fig.add_trace(go.Scatter(
            name='Mean Impact',
            x=condition_stats['feature'],
            y=condition_stats['mean_impact'],
            yaxis='y2',
            line=dict(color='red', width=2)
        ))
        
        # Update layout
        fig.update_layout(
            title='Feature Condition Summary',
            xaxis_title='Feature',
            yaxis_title='Threshold Range',
            yaxis2=dict(
                title='Mean Impact',
                overlaying='y',
                side='right'
            ),
            barmode='group',
            width=1200,
            height=600
        )
        
        if output_path:
            fig.write_html(output_path)
            
        return fig 