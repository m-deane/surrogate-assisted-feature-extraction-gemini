"""
Main script for running the SAFE (Surrogate Assisted Feature Extraction) analysis pipeline.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from data.data_loader import PreemDataLoader
from models.surrogate_model import SAFEAnalyzer
from models.advanced_analysis import AdvancedAnalyzer
from models.rule_analysis import RuleAnalyzer
from models.advanced_surrogate import AdvancedSurrogateAnalyzer
from models.temporal_analysis import TemporalAnalyzer
from models.ensemble_surrogates import EnsembleSurrogateAnalyzer
from visualization.threshold_plots import ThresholdVisualizer
from visualization.advanced_plots import AdvancedVisualizer
from visualization.temporal_plots import TemporalVisualizer
from visualization.ensemble_plots import EnsembleVisualizer
from utils.report_generator import ReportGenerator
import networkx as nx

def create_feature_network_plot(G: nx.Graph, output_path: str):
    """Create and save network visualization of feature interactions."""
    plt.figure(figsize=(12, 12))
    pos = nx.spring_layout(G)
    
    # Draw the network
    nx.draw_networkx_nodes(G, pos, node_size=2000, node_color='lightblue')
    nx.draw_networkx_labels(G, pos)
    
    # Draw edges with varying thickness based on weight
    edges = G.edges(data=True)
    weights = [d['weight'] * 2 for (u, v, d) in edges]
    nx.draw_networkx_edges(G, pos, width=weights, alpha=0.5)
    
    plt.title('Feature Interaction Network')
    plt.axis('off')
    plt.savefig(output_path)
    plt.close()

def plot_temporal_patterns(temporal_results: dict, output_dir: str):
    """Create and save temporal pattern visualizations."""
    seasonal_results = temporal_results.get('seasonal_analysis', {})
    stationarity_results = temporal_results.get('stationarity_analysis', {})
    nonlinearity_results = temporal_results.get('nonlinearity_analysis', {})
    
    for feature, results in seasonal_results.items():
        plt.figure(figsize=(15, 10))
        
        # Plot seasonal decomposition
        plt.subplot(411)
        plt.plot(results['trend'], label='Trend')
        plt.title(f'Temporal Decomposition - {feature}')
        plt.legend()
        
        plt.subplot(412)
        plt.plot(results['seasonal'], label='Seasonal')
        plt.legend()
        
        plt.subplot(413)
        plt.plot(results['residual'], label='Residual')
        plt.legend()
        
        # Plot ACF/PACF from nonlinearity analysis
        if feature in nonlinearity_results:
            plt.subplot(414)
            nonlin_results = nonlinearity_results[feature]
            plt.plot(nonlin_results['acf_values'], label='ACF')
            plt.plot(nonlin_results['pacf_values'], label='PACF')
            plt.legend()
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/temporal_{feature}.png')
        plt.close()

def main():
    # Create output directories
    os.makedirs('plots', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    os.makedirs('plots/advanced', exist_ok=True)
    os.makedirs('plots/rules', exist_ok=True)
    os.makedirs('plots/surrogate', exist_ok=True)
    os.makedirs('plots/temporal', exist_ok=True)
    os.makedirs('plots/ensemble', exist_ok=True)
    
    # Initialize data loader and load data
    data_loader = PreemDataLoader('_data/preem.csv')
    data = data_loader.load_data()
    
    # Get feature statistics
    feature_stats = data_loader.get_feature_statistics()
    
    # Prepare features with outlier and missing value handling
    X, y = data_loader.prepare_features(handle_outliers=True, handle_missing=True)
    
    # Get train-validation-test split
    X_train, X_val, X_test, y_train, y_val, y_test = data_loader.get_train_test_split(
        test_months=6,
        validation_months=3
    )
    
    print("Data loaded and preprocessed successfully.")
    
    # Initialize visualizers
    temporal_viz = TemporalVisualizer()
    ensemble_viz = EnsembleVisualizer()
    
    # Initialize and run temporal analysis
    print("\nPerforming temporal analysis...")
    temporal_analyzer = TemporalAnalyzer(data)
    
    # Analyze seasonality
    seasonal_results = temporal_analyzer.analyze_seasonality()
    print("Seasonal analysis completed.")
    
    # Create seasonal decomposition plots
    for feature in data.columns:
        if feature != 'date':
            temporal_viz.create_temporal_decomposition_plot(
                data,
                feature,
                seasonal_results[feature],
                output_path=f'plots/temporal/decomposition_{feature}.html'
            )
    
    # Analyze stationarity
    stationarity_results = temporal_analyzer.analyze_stationarity()
    print("Stationarity analysis completed.")
    
    # Create stationarity plots
    for feature in data.columns:
        if feature != 'date':
            temporal_viz.create_stationarity_plot(
                data,
                feature,
                stationarity_results[feature],
                output_path=f'plots/temporal/stationarity_{feature}.html'
            )
    
    # Analyze Granger causality
    causality_results = temporal_analyzer.analyze_granger_causality()
    print("Granger causality analysis completed.")
    
    # Create causality network visualization
    temporal_viz.create_causality_network(
        causality_results,
        [col for col in data.columns if col != 'date'],
        output_path='plots/temporal/causality_network.html'
    )
    
    # Analyze nonlinearity
    nonlinearity_results = temporal_analyzer.analyze_nonlinearity()
    print("Nonlinearity analysis completed.")
    
    # Create nonlinearity plots
    for feature in data.columns:
        if feature != 'date':
            temporal_viz.create_nonlinearity_plot(
                data,
                feature,
                nonlinearity_results[feature],
                output_path=f'plots/temporal/nonlinearity_{feature}.html'
            )
    
    # Initialize and run ensemble analysis
    print("\nPerforming ensemble analysis...")
    ensemble_analyzer = EnsembleSurrogateAnalyzer()
    
    # Fit surrogate ensemble
    ensemble_results = ensemble_analyzer.fit_surrogate_ensemble(X_train, y_train)
    print("Ensemble models fitted successfully.")
    
    # Analyze feature thresholds
    threshold_results = ensemble_analyzer.analyze_feature_thresholds(X_train, y_train)
    print("Threshold analysis completed.")
    
    # Create threshold comparison plots
    for feature in X.columns:
        ensemble_viz.create_threshold_comparison_plot(
            threshold_results,
            feature,
            output_path=f'plots/ensemble/threshold_comparison_{feature}.html'
        )
    
    # Analyze model disagreement
    disagreement_results = ensemble_analyzer.analyze_model_disagreement(X_val)
    print("Model disagreement analysis completed.")
    
    # Create disagreement heatmap
    ensemble_viz.create_disagreement_heatmap(
        disagreement_results,
        X.columns.tolist(),
        output_path='plots/ensemble/disagreement_heatmap.html'
    )
    
    # Compare model explanations
    explanation_comparison = ensemble_analyzer.compare_model_explanations(X_val)
    print("Model explanation comparison completed.")
    
    # Create explanation consistency network
    ensemble_viz.create_explanation_consistency_network(
        explanation_comparison,
        output_path='plots/ensemble/explanation_network.html'
    )
    
    # Create model performance comparison plot
    performance_metrics = ensemble_analyzer.get_model_performance(X_test, y_test)
    ensemble_viz.create_model_comparison_plot(
        performance_metrics,
        output_path='plots/ensemble/model_comparison.html'
    )
    
    # Initialize report generator
    report_generator = ReportGenerator()
    
    # Add analysis results
    report_generator.add_model_performance(performance_metrics)
    report_generator.add_threshold_analysis(threshold_results)
    
    # Get interaction analysis from explanation comparison
    interaction_analysis = explanation_comparison.get('rank_correlation', {})
    report_generator.add_feature_interactions(interaction_analysis)
    
    # Store temporal results
    temporal_analysis_results = {
        'seasonal_analysis': seasonal_results,
        'stationarity_analysis': stationarity_results,
        'causality_analysis': causality_results,
        'nonlinearity_analysis': nonlinearity_results
    }
    
    # Store ensemble results
    ensemble_analysis_results = {
        'threshold_analysis': threshold_results,
        'disagreement_analysis': disagreement_results,
        'explanation_comparison': explanation_comparison,
        'performance_metrics': performance_metrics
    }
    
    # Save analysis results
    report_generator.save_analysis_results(
        'results/analysis_report.md',
        temporal_analysis=temporal_analysis_results,
        ensemble_analysis=ensemble_analysis_results
    )
    
    print("\nAnalysis report and results saved successfully.")
    print("\nAnalysis pipeline completed!")

if __name__ == "__main__":
    main() 