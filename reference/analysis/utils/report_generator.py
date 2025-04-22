"""
Module for generating comprehensive analysis reports.
"""

import pandas as pd
import numpy as np
from typing import Dict, List
from datetime import datetime
import json
import matplotlib.pyplot as plt
import seaborn as sns

class ReportGenerator:
    def __init__(self):
        """Initialize the report generator."""
        self.report_sections = []
        
    def add_section(self, title: str, content: str):
        """
        Add a section to the report.
        
        Args:
            title (str): Section title
            content (str): Section content
        """
        self.report_sections.append({
            'title': title,
            'content': content
        })
        
    def add_model_performance(self, performance_metrics: Dict):
        """
        Add detailed model performance metrics to the report.
        
        Args:
            performance_metrics (Dict): Dictionary containing model performance metrics
        """
        content = """
## Model Performance Analysis

### Ensemble Model Performance
- R² Score: {:.4f}
  * Interpretation: {:.1%} of variance in target variable explained
  * Quality: {}
- Mean Squared Error: {:.4f}
  * Root MSE: {:.4f}
  * Indicates average prediction error magnitude

### Individual Model Performance
""".format(
            performance_metrics['ensemble']['r2'],
            performance_metrics['ensemble']['r2'],
            "Excellent" if performance_metrics['ensemble']['r2'] > 0.8 else 
            "Good" if performance_metrics['ensemble']['r2'] > 0.6 else "Fair",
            performance_metrics['ensemble']['mse'],
            np.sqrt(performance_metrics['ensemble']['mse'])
        )
        
        # Add performance metrics for each individual model
        for model_name, metrics in performance_metrics.items():
            if model_name != 'ensemble':
                content += f"""
#### {model_name.upper()}
- R² Score: {metrics['r2']:.4f}
  * Interpretation: {metrics['r2']:.1%} of variance explained
  * Quality: {"Excellent" if metrics['r2'] > 0.8 else "Good" if metrics['r2'] > 0.6 else "Fair"}
- Mean Squared Error: {metrics['mse']:.4f}
  * Root MSE: {np.sqrt(metrics['mse']):.4f}
- Fidelity Score: {metrics['fidelity']:.4f}
  * Contribution to ensemble: {metrics['fidelity']*100:.1f}%
"""
        
        # Add model comparison summary
        best_model = max(performance_metrics.items(), key=lambda x: x[1]['r2'])
        worst_model = min(performance_metrics.items(), key=lambda x: x[1]['r2'])
        
        content += f"""
### Model Comparison Summary
- Best performing model: {best_model[0].upper()} (R² = {best_model[1]['r2']:.4f})
- Most challenging model: {worst_model[0].upper()} (R² = {worst_model[1]['r2']:.4f})
- Performance spread: {(best_model[1]['r2'] - worst_model[1]['r2'])*100:.1f}% difference in R²

### Recommendations
- {"Use ensemble predictions for best overall performance" if performance_metrics['ensemble']['r2'] >= best_model[1]['r2']
   else f"Consider using {best_model[0].upper()} model for best performance"}
- {"Individual models show consistent performance" if (best_model[1]['r2'] - worst_model[1]['r2']) < 0.2
   else "Significant variation between models suggests careful model selection is important"}
"""
        
        self.add_section("Model Performance", content)
        
    def add_threshold_analysis(self, threshold_analysis: Dict):
        """
        Add comprehensive threshold analysis results to the report.
        
        Args:
            threshold_analysis (Dict): Dictionary containing threshold analysis results
        """
        content = "## Feature Thresholds and Decision Boundaries\n\n"
        
        # Sort features by importance
        sorted_features = sorted(
            threshold_analysis.items(),
            key=lambda x: x[1]['importance'],
            reverse=True
        )
        
        # Add summary statistics
        total_importance = sum(data['importance'] for _, data in sorted_features)
        num_features_with_thresholds = sum(1 for _, data in sorted_features if data['thresholds'])
        
        content += f"""### Summary Statistics
- Total number of features analyzed: {len(sorted_features)}
- Features with identified thresholds: {num_features_with_thresholds}
- Average thresholds per feature: {sum(len(data['thresholds']) for _, data in sorted_features) / num_features_with_thresholds:.1f}

### Key Decision Boundaries\n\n"""
        
        # Add detailed feature analysis
        cumulative_importance = 0
        for feature, data in sorted_features:
            cumulative_importance += data['importance'] / total_importance
            
            content += f"### {feature}\n"
            content += f"- **Importance Score**: {data['importance']:.4f} ({data['importance']/total_importance:.1%} of total)\n"
            content += f"- **Cumulative Importance**: {cumulative_importance:.1%}\n"
            
            if data['thresholds']:
                content += "- **Decision Thresholds**:\n"
                sorted_thresholds = sorted(data['thresholds'])
                for i, threshold in enumerate(sorted_thresholds):
                    if i == 0:
                        content += f"  * Primary split at {threshold:.4f}\n"
                    else:
                        content += f"  * Secondary split at {threshold:.4f}\n"
                        
                # Add threshold interpretation
                if len(sorted_thresholds) == 1:
                    content += "  * Binary decision boundary\n"
                else:
                    content += f"  * Creates {len(sorted_thresholds) + 1} distinct regions\n"
                    
                # Add threshold spacing analysis
                if len(sorted_thresholds) > 1:
                    spacing = np.diff(sorted_thresholds)
                    content += f"  * Average gap between thresholds: {np.mean(spacing):.4f}\n"
                    
            content += "\n"
            
            # Stop after explaining 80% of cumulative importance
            if cumulative_importance > 0.8:
                remaining_features = len(sorted_features) - sorted_features.index((feature, data)) - 1
                if remaining_features > 0:
                    content += f"\n*Note: Remaining {remaining_features} features account for {(1-cumulative_importance):.1%} of total importance*\n"
                break
            
        self.add_section("Threshold Analysis", content)
        
    def add_feature_interactions(self, interaction_analysis: Dict):
        """
        Add feature interaction analysis to the report.
        
        Args:
            interaction_analysis (Dict): Dictionary containing feature interaction results
        """
        content = """## Feature Interaction Analysis

The following interactions were identified between features:

"""
        
        # Handle different formats of interaction analysis results
        if isinstance(interaction_analysis, dict):
            # Sort interactions by absolute strength
            sorted_interactions = []
            for feature1, interactions in interaction_analysis.items():
                if isinstance(interactions, dict):
                    for feature2, strength in interactions.items():
                        if isinstance(strength, (int, float)):
                            sorted_interactions.append((feature1, feature2, strength))
            
            sorted_interactions.sort(key=lambda x: abs(x[2]), reverse=True)
            
            # Add top interactions to report
            for feature1, feature2, strength in sorted_interactions[:5]:
                content += f"- {feature1} ↔ {feature2}: Interaction Strength = {strength:.4f}\n"
        
        content += "\n### Interpretation\n\n"
        content += "Strong interactions indicate features that work together to influence the target variable. "
        content += "These relationships should be considered when making predictions or analyzing feature importance."
        
        self.add_section("Feature Interactions", content)
        
    def add_rule_analysis(self, rules: List[Dict], condition_stats: pd.DataFrame):
        """
        Add comprehensive rule analysis to the report.
        
        Args:
            rules (List[Dict]): List of analyzed rules
            condition_stats (pd.DataFrame): Statistics about rule conditions
        """
        content = """## Decision Rule Analysis

### Overview
Decision rules represent the logical paths through the surrogate model. Each rule combines multiple conditions
to make predictions, providing interpretable decision logic.

"""
        # Sort rules by confidence
        sorted_rules = sorted(rules, key=lambda x: x['confidence'], reverse=True)
        
        # Add rule statistics
        content += f"""### Rule Statistics
- Total number of rules: {len(rules)}
- Average rule confidence: {np.mean([r['confidence'] for r in rules]):.2%}
- Average rule coverage: {np.mean([r['coverage'] for r in rules]):.2%}
- Average conditions per rule: {np.mean([len(r['conditions']) for r in rules]):.1f}

### High-Confidence Rules (Top 5)\n\n"""

        # Add detailed rule analysis
        for i, rule in enumerate(sorted_rules[:5], 1):
            content += f"#### Rule {i}\n"
            content += f"- **Confidence**: {rule['confidence']:.2%}\n"
            content += f"- **Coverage**: {rule['coverage']:.2%} of samples\n"
            content += f"- **Prediction**: {rule['prediction']:.2f}\n"
            content += "- **Conditions**:\n"
            
            for feature, op, threshold in rule['conditions']:
                content += f"  * {feature} {op} {threshold:.4f}\n"
                
            content += f"- **Impact Range**: [{rule['target_range'][0]:.2f}, {rule['target_range'][1]:.2f}]\n"
            content += f"- **Stability**: ±{rule['target_std']:.2f} (std. dev.)\n\n"
            
        content += """### Feature Usage in Rules

The following analysis shows how different features are used in decision rules:

"""
        # Add feature usage statistics
        sorted_stats = condition_stats.sort_values('mean_impact', ascending=False)
        for _, row in sorted_stats.iterrows():
            content += f"#### {row['feature']}\n"
            content += f"- Used in {row['num_thresholds']} conditions\n"
            content += f"- Threshold range: [{row['min_threshold']:.2f}, {row['max_threshold']:.2f}]\n"
            content += f"- Most common operation: {row['most_common_op']}\n"
            content += f"- Average impact: {row['mean_impact']:.2%} of samples affected\n\n"
            
        self.add_section("Rule Analysis", content)
        
    def add_temporal_analysis(self, temporal_results: Dict):
        """
        Add temporal analysis results to the report.
        
        Args:
            temporal_results: Dictionary containing temporal analysis results
        """
        self.report_sections.append("\n## Temporal Analysis\n")
        
        # Add seasonal analysis results
        self.report_sections.append("\n### Seasonal Analysis\n")
        seasonal_results = temporal_results.get('seasonal_analysis', {})
        for feature, results in seasonal_results.items():
            seasonal_strength = results.get('seasonal_strength', 0)
            self.report_sections.append(f"\n#### {feature}\n")
            self.report_sections.append(f"- Seasonal strength: {seasonal_strength:.4f}")
            self.report_sections.append(f"- Period: {results.get('period', 'N/A')}")
            
            if seasonal_strength > 0.3:
                self.report_sections.append("- Strong seasonal pattern detected")
            elif seasonal_strength > 0.1:
                self.report_sections.append("- Moderate seasonal pattern detected")
            else:
                self.report_sections.append("- Weak or no seasonal pattern detected")
        
        # Add stationarity analysis results
        self.report_sections.append("\n### Stationarity Analysis\n")
        stationarity_results = temporal_results.get('stationarity_analysis', {})
        for feature, results in stationarity_results.items():
            self.report_sections.append(f"\n#### {feature}\n")
            self.report_sections.append(f"- ADF statistic: {results.get('adf_statistic', 0):.4f}")
            self.report_sections.append(f"- p-value: {results.get('p_value', 1):.4f}")
            
            if results.get('is_stationary', False):
                self.report_sections.append("- Series is stationary")
            else:
                self.report_sections.append("- Series is non-stationary")
        
        # Add Granger causality results
        self.report_sections.append("\n### Granger Causality Analysis\n")
        causality_results = temporal_results.get('causality_analysis', {})
        
        # Find significant causal relationships
        significant_relationships = []
        for cause, effects in causality_results.items():
            for effect, result in effects.items():
                if result.get('causes_at_optimal_lag', False):
                    significant_relationships.append({
                        'cause': cause,
                        'effect': effect,
                        'lag': result.get('optimal_lag', 0),
                        'p_value': result.get('min_p_value', 1)
                    })
        
        # Sort by p-value and report top relationships
        significant_relationships.sort(key=lambda x: x['p_value'])
        
        if significant_relationships:
            self.report_sections.append("\nSignificant causal relationships found:\n")
            for rel in significant_relationships[:10]:  # Show top 10
                self.report_sections.append(
                    f"- {rel['cause']} → {rel['effect']} "
                    f"(lag: {rel['lag']}, p-value: {rel['p_value']:.4f})"
                )
        else:
            self.report_sections.append("\nNo significant causal relationships found.")
        
        # Add nonlinearity analysis results
        self.report_sections.append("\n### Nonlinearity Analysis\n")
        nonlinearity_results = temporal_results.get('nonlinearity_analysis', {})
        
        for feature, results in nonlinearity_results.items():
            self.report_sections.append(f"\n#### {feature}\n")
            
            # Report nonlinearity test results
            if results.get('is_nonlinear') is not None:
                self.report_sections.append(
                    f"- Nonlinear patterns: "
                    f"{'Detected' if results['is_nonlinear'] else 'Not detected'}"
                )
            
            # Report distribution characteristics
            skewness = results.get('skewness', 0)
            kurtosis = results.get('kurtosis', 0)
            self.report_sections.append(f"- Skewness: {skewness:.4f}")
            self.report_sections.append(f"- Kurtosis: {kurtosis:.4f}")
            
            if abs(skewness) > 1:
                self.report_sections.append("- Significant asymmetry in distribution")
            if abs(kurtosis) > 3:
                self.report_sections.append("- Heavy-tailed distribution")
        
    def generate_report(self, output_path: str = None) -> str:
        """
        Generate the final report.
        
        Args:
            output_path (str): Path to save the report
            
        Returns:
            str: The complete report in markdown format
        """
        report = f"""# Enhanced SAFE Analysis Report
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary
This report provides a comprehensive analysis of the model's decision-making process,
feature interactions, and key thresholds. Use this information to understand:
- How the model makes predictions
- Which features are most important
- How features interact with each other
- Where critical decision boundaries lie

"""
        
        for section in self.report_sections:
            report += f"# {section['title']}\n\n"
            report += f"{section['content']}\n\n"
            
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report)
                
        return report
        
    def _convert_to_serializable(self, obj):
        """Convert objects to JSON serializable format.
        
        Args:
            obj: Object to convert (can be dict, list, pandas Series, etc.)
            
        Returns:
            JSON serializable object
        """
        if isinstance(obj, dict):
            return {
                str(k) if isinstance(k, (np.int64, np.int32)) else k: 
                self._convert_to_serializable(v) 
                for k, v in obj.items()
            }
        elif isinstance(obj, list):
            return [self._convert_to_serializable(item) for item in obj]
        elif isinstance(obj, pd.Series):
            return obj.tolist()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict('records')
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32, np.float64, np.float32)):
            return obj.item()
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, (pd.Timestamp, datetime)):
            return obj.isoformat()
        else:
            return obj

    def save_analysis_results(self, 
                            output_path: str,
                            temporal_analysis: Dict = None,
                            ensemble_analysis: Dict = None):
        """
        Save all analysis results to a JSON file.
        
        Args:
            output_path (str): Path to save the JSON file
            temporal_analysis (Dict): Temporal pattern analysis results
            ensemble_analysis (Dict): Ensemble analysis results including thresholds,
                                    disagreement, explanations, and performance metrics
        """
        # Convert interaction analysis to a serializable format
        serializable_interactions = {}
        if ensemble_analysis and 'explanation_comparison' in ensemble_analysis:
            explanation_comparison = ensemble_analysis['explanation_comparison']
            if isinstance(explanation_comparison, dict):
                for key, value in explanation_comparison.items():
                    if isinstance(key, tuple):
                        serializable_interactions[f"{key[0]}__X__{key[1]}"] = value
                    else:
                        serializable_interactions[key] = value
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'temporal_analysis': temporal_analysis,
            'ensemble_analysis': {
                'threshold_analysis': ensemble_analysis.get('threshold_analysis') if ensemble_analysis else None,
                'disagreement_analysis': ensemble_analysis.get('disagreement_analysis') if ensemble_analysis else None,
                'explanation_comparison': serializable_interactions,
                'performance_metrics': ensemble_analysis.get('performance_metrics') if ensemble_analysis else None
            } if ensemble_analysis else None
        }
        
        # Convert all results to JSON serializable format
        serializable_results = self._convert_to_serializable(results)
        
        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=4) 