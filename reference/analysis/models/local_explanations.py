"""
Local explanations module for instance-level model interpretations.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Callable
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import BaseEstimator
import lime
import lime.lime_tabular
import shap
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import json

class LocalExplainer:
    def __init__(self, model: BaseEstimator, X: pd.DataFrame, 
               model_type: str = 'regression', 
               feature_names: Optional[List[str]] = None):
        """
        Initialize the local explainer.
        
        Args:
            model: Fitted model to explain
            X: Feature matrix used for training
            model_type: Type of model ('regression' or 'classification')
            feature_names: Optional list of feature names
        """
        self.model = model
        self.X = X
        self.model_type = model_type
        self.feature_names = feature_names if feature_names else X.columns.tolist()
        
        # Initialize explainers
        self._initialize_explainers()
        
    def _initialize_explainers(self):
        """Initialize LIME and SHAP explainers."""
        # LIME explainer
        self.lime_explainer = lime.lime_tabular.LimeTabularExplainer(
            self.X.values,
            feature_names=self.feature_names,
            mode=self.model_type,
            training_labels=None,
            categorical_features=None,
            categorical_names=None,
            kernel_width=3
        )
        
        # SHAP explainer
        try:
            # Try Tree explainer first for tree-based models
            if hasattr(self.model, 'predict_proba'):
                self.shap_explainer = shap.TreeExplainer(self.model)
            else:
                self.shap_explainer = shap.Explainer(self.model, self.X)
        except:
            # Fallback to Kernel explainer
            def predict_fn(x):
                return self.model.predict(x)
            self.shap_explainer = shap.KernelExplainer(predict_fn, shap.sample(self.X, 100))
    
    def explain_instance_lime(self, instance: Union[pd.Series, np.ndarray, List], 
                           num_features: int = 10) -> Dict:
        """
        Generate LIME explanation for a single instance.
        
        Args:
            instance: Feature vector to explain
            num_features: Number of features to include in explanation
            
        Returns:
            Dict: LIME explanation results
        """
        # Convert instance to proper format
        if isinstance(instance, pd.Series):
            instance_array = instance.values
        elif isinstance(instance, list):
            instance_array = np.array(instance)
        else:
            instance_array = instance
            
        # Generate explanation
        if self.model_type == 'regression':
            explanation = self.lime_explainer.explain_instance(
                instance_array, 
                self.model.predict,
                num_features=num_features
            )
        else:
            explanation = self.lime_explainer.explain_instance(
                instance_array, 
                self.model.predict_proba,
                num_features=num_features,
                top_labels=1
            )
            
        # Extract explanation details
        if self.model_type == 'regression':
            features_importance = explanation.as_list()
            prediction = float(self.model.predict([instance_array])[0])
        else:
            # For classification, get explanation for predicted class
            label_idx = int(self.model.predict([instance_array])[0])
            features_importance = explanation.as_list(label=label_idx)
            prediction = float(self.model.predict_proba([instance_array])[0][label_idx])
            
        # Format results
        lime_explanation = {
            'prediction': prediction,
            'feature_importance': {feature: float(importance) 
                                 for feature, importance in features_importance},
            'intercept': float(explanation.intercept[0] if self.model_type == 'regression' 
                             else explanation.intercept[label_idx]),
            'local_model_score': float(explanation.score)
        }
        
        return lime_explanation
    
    def explain_instance_shap(self, instance: Union[pd.Series, np.ndarray, List]) -> Dict:
        """
        Generate SHAP explanation for a single instance.
        
        Args:
            instance: Feature vector to explain
            
        Returns:
            Dict: SHAP explanation results
        """
        # Convert instance to proper format
        if isinstance(instance, pd.Series):
            instance_df = pd.DataFrame([instance.values], columns=self.feature_names)
        elif isinstance(instance, list):
            instance_df = pd.DataFrame([instance], columns=self.feature_names)
        elif isinstance(instance, np.ndarray) and instance.ndim == 1:
            instance_df = pd.DataFrame([instance], columns=self.feature_names)
        else:
            instance_df = pd.DataFrame(instance, columns=self.feature_names)
            
        # Generate explanation
        shap_values = self.shap_explainer(instance_df)
        
        # Get base value and SHAP values
        if hasattr(shap_values, 'base_values'):
            base_value = float(shap_values.base_values[0])
            values = shap_values.values[0]
        else:
            base_value = float(shap_values.expected_value)
            values = shap_values.values[0]
            
        # Format results
        shap_explanation = {
            'base_value': base_value,
            'feature_values': instance_df.iloc[0].to_dict(),
            'shap_values': {feature: float(value) 
                          for feature, value in zip(self.feature_names, values)},
            'prediction': float(self.model.predict(instance_df)[0])
        }
        
        return shap_explanation
    
    def explain_batch(self, instances: pd.DataFrame, 
                    method: str = 'both', 
                    num_features: int = 10) -> List[Dict]:
        """
        Generate explanations for multiple instances.
        
        Args:
            instances: DataFrame of instances to explain
            method: Explanation method ('lime', 'shap', or 'both')
            num_features: Number of features for LIME explanations
            
        Returns:
            List[Dict]: List of explanation results
        """
        explanations = []
        
        for i, row in instances.iterrows():
            explanation = {'instance_index': i}
            
            if method in ['lime', 'both']:
                lime_result = self.explain_instance_lime(row, num_features)
                explanation['lime'] = lime_result
                
            if method in ['shap', 'both']:
                shap_result = self.explain_instance_shap(row)
                explanation['shap'] = shap_result
                
            explanations.append(explanation)
            
        return explanations
    
    def explain_counterfactual(self, instance: Union[pd.Series, np.ndarray, List],
                             target_outcome: float = None,
                             feature_constraints: Dict = None,
                             max_iterations: int = 1000) -> Dict:
        """
        Generate counterfactual explanation for a single instance.
        
        Args:
            instance: Feature vector to explain
            target_outcome: Target prediction value (for regression) or class (for classification)
            feature_constraints: Dictionary of feature constraints {feature_name: (min, max)}
            max_iterations: Maximum number of iterations for optimization
            
        Returns:
            Dict: Counterfactual explanation
        """
        # Convert instance to proper format
        if isinstance(instance, pd.Series):
            instance_array = instance.values
            instance_dict = instance.to_dict()
        elif isinstance(instance, list):
            instance_array = np.array(instance)
            instance_dict = {name: val for name, val in zip(self.feature_names, instance)}
        else:
            instance_array = instance
            instance_dict = {name: val for name, val in zip(self.feature_names, instance)}
            
        # Get original prediction
        orig_prediction = self.model.predict([instance_array])[0]
        
        # If no target outcome is provided, use opposite prediction
        if target_outcome is None:
            if self.model_type == 'regression':
                # For regression, move prediction by 20% of the output range
                y_range = np.max(self.model.predict(self.X)) - np.min(self.model.predict(self.X))
                target_outcome = orig_prediction + (0.2 * y_range)
            else:
                # For classification, predict the other class
                target_outcome = 1 - int(orig_prediction)
        
        # Setup constraints
        if feature_constraints is None:
            feature_constraints = {}
            
        # Set default constraints based on dataset range
        for feature in self.feature_names:
            if feature not in feature_constraints:
                min_val = self.X[feature].min()
                max_val = self.X[feature].max()
                feature_constraints[feature] = (min_val, max_val)
        
        # Simple gradient-based approach to find counterfactual
        counterfactual = instance_array.copy()
        step_size = 0.01
        
        # Function to compute loss (difference between prediction and target)
        def compute_loss(x):
            pred = self.model.predict([x])[0]
            return (pred - target_outcome) ** 2
        
        # Function to compute gradient (numerical approximation)
        def compute_gradient(x):
            gradient = np.zeros_like(x)
            loss = compute_loss(x)
            
            for i in range(len(x)):
                # Compute partial derivative
                x_plus = x.copy()
                x_plus[i] += 0.01
                
                # Respect constraints
                feature_name = self.feature_names[i]
                min_val, max_val = feature_constraints.get(feature_name, (None, None))
                if max_val is not None and x_plus[i] > max_val:
                    x_plus[i] = max_val
                    
                loss_plus = compute_loss(x_plus)
                gradient[i] = (loss_plus - loss) / 0.01
                
            return gradient
        
        # Optimization loop
        current_loss = compute_loss(counterfactual)
        
        for _ in range(max_iterations):
            if current_loss < 0.001:  # Convergence criterion
                break
                
            # Compute gradient
            gradient = compute_gradient(counterfactual)
            
            # Update counterfactual
            counterfactual = counterfactual - step_size * gradient
            
            # Apply constraints
            for i, feature in enumerate(self.feature_names):
                min_val, max_val = feature_constraints.get(feature, (None, None))
                if min_val is not None and counterfactual[i] < min_val:
                    counterfactual[i] = min_val
                if max_val is not None and counterfactual[i] > max_val:
                    counterfactual[i] = max_val
            
            # Compute new loss
            new_loss = compute_loss(counterfactual)
            
            # Check for improvement
            if new_loss >= current_loss:
                step_size *= 0.5  # Reduce step size if no improvement
            
            current_loss = new_loss
            
            if step_size < 1e-6:  # Minimum step size
                break
        
        # Create result
        counterfactual_prediction = self.model.predict([counterfactual])[0]
        counterfactual_dict = {name: float(val) for name, val in zip(self.feature_names, counterfactual)}
        
        # Calculate differences
        differences = {}
        for feature in self.feature_names:
            orig_val = instance_dict[feature]
            new_val = counterfactual_dict[feature]
            abs_diff = abs(new_val - orig_val)
            pct_diff = abs_diff / abs(orig_val) if abs(orig_val) > 1e-10 else float('inf')
            
            differences[feature] = {
                'original': float(orig_val),
                'counterfactual': float(new_val),
                'absolute_diff': float(abs_diff),
                'percentage_diff': float(pct_diff) if not np.isinf(pct_diff) else None
            }
        
        # Sort differences by absolute change
        sorted_diffs = sorted(differences.items(), 
                            key=lambda x: x[1]['absolute_diff'], 
                            reverse=True)
        
        # Keep only the top features that changed
        top_diffs = {k: v for k, v in sorted_diffs[:10]}
        
        return {
            'original': {
                'values': instance_dict,
                'prediction': float(orig_prediction)
            },
            'counterfactual': {
                'values': counterfactual_dict,
                'prediction': float(counterfactual_prediction)
            },
            'target_outcome': float(target_outcome),
            'differences': top_diffs,
            'loss': float(current_loss)
        }
    
    def visualize_lime_explanation(self, lime_explanation: Dict, 
                                output_path: str = None) -> go.Figure:
        """
        Create visualization for LIME explanation.
        
        Args:
            lime_explanation: LIME explanation from explain_instance_lime
            output_path: Path to save the plot
            
        Returns:
            go.Figure: Plotly figure
        """
        # Extract data
        features = list(lime_explanation['feature_importance'].keys())
        importances = list(lime_explanation['feature_importance'].values())
        
        # Sort by absolute importance
        sorted_indices = np.argsort(np.abs(importances))[::-1]
        features = [features[i] for i in sorted_indices]
        importances = [importances[i] for i in sorted_indices]
        
        # Create colors based on positive/negative contributions
        colors = ['green' if imp > 0 else 'red' for imp in importances]
        
        # Create figure
        fig = go.Figure(go.Bar(
            x=importances,
            y=features,
            orientation='h',
            marker_color=colors
        ))
        
        fig.update_layout(
            title=f"LIME Explanation (Prediction: {lime_explanation['prediction']:.4f})",
            xaxis_title="Feature Contribution",
            yaxis_title="Feature",
            height=max(400, len(features) * 30),
            width=800
        )
        
        if output_path:
            fig.write_html(output_path)
            
        return fig
    
    def visualize_shap_explanation(self, shap_explanation: Dict,
                                output_path: str = None) -> go.Figure:
        """
        Create visualization for SHAP explanation.
        
        Args:
            shap_explanation: SHAP explanation from explain_instance_shap
            output_path: Path to save the plot
            
        Returns:
            go.Figure: Plotly figure
        """
        # Extract data
        features = list(shap_explanation['shap_values'].keys())
        values = list(shap_explanation['shap_values'].values())
        base_value = shap_explanation['base_value']
        
        # Sort by absolute SHAP value
        sorted_indices = np.argsort(np.abs(values))[::-1]
        features = [features[i] for i in sorted_indices]
        values = [values[i] for i in sorted_indices]
        
        # Create colors based on positive/negative contributions
        colors = ['green' if val > 0 else 'red' for val in values]
        
        # Create figure
        fig = go.Figure(go.Waterfall(
            name="SHAP",
            orientation="h",
            y=features,
            x=values,
            connector={"mode": "spanning", "line": {"width": 1, "color": "rgb(102, 102, 102)", "dash": "dot"}},
            decreasing={"marker": {"color": "red"}},
            increasing={"marker": {"color": "green"}},
            base=base_value,
            text=[f"{val:.4f}" for val in values],
            textposition="outside"
        ))
        
        # Add base value and final prediction
        prediction = shap_explanation['prediction']
        fig.add_annotation(
            x=base_value,
            y=-1,
            text=f"Base value: {base_value:.4f}",
            showarrow=False
        )
        
        fig.add_annotation(
            x=prediction,
            y=len(features),
            text=f"Prediction: {prediction:.4f}",
            showarrow=False
        )
        
        fig.update_layout(
            title="SHAP Feature Contributions",
            xaxis_title="SHAP Value",
            yaxis_title="Feature",
            height=max(400, len(features) * 30),
            width=800,
            showlegend=False
        )
        
        if output_path:
            fig.write_html(output_path)
            
        return fig
    
    def visualize_counterfactual(self, counterfactual_explanation: Dict,
                              output_path: str = None) -> go.Figure:
        """
        Create visualization for counterfactual explanation.
        
        Args:
            counterfactual_explanation: Result from explain_counterfactual
            output_path: Path to save the plot
            
        Returns:
            go.Figure: Plotly figure
        """
        # Extract data
        differences = counterfactual_explanation['differences']
        features = list(differences.keys())
        orig_values = [differences[f]['original'] for f in features]
        cf_values = [differences[f]['counterfactual'] for f in features]
        
        # Calculate percentage changes for display
        pct_changes = []
        for f in features:
            pct = differences[f]['percentage_diff']
            if pct is None:
                pct_changes.append("N/A")
            else:
                pct_changes.append(f"{pct*100:.1f}%")
        
        # Create subplot with two charts
        fig = make_subplots(rows=1, cols=2, 
                          subplot_titles=["Feature Values", "Original vs Counterfactual"])
        
        # Add bar chart comparing original vs counterfactual for each feature
        fig.add_trace(
            go.Bar(
                x=features,
                y=orig_values,
                name="Original",
                marker_color='blue'
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(
                x=features,
                y=cf_values,
                name="Counterfactual",
                marker_color='orange'
            ),
            row=1, col=1
        )
        
        # Add scatter plot showing the change
        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[counterfactual_explanation['original']['prediction'], 
                   counterfactual_explanation['counterfactual']['prediction']],
                mode='markers+lines+text',
                marker=dict(size=12),
                line=dict(dash='dash'),
                text=["Original", "Counterfactual"],
                textposition="top center"
            ),
            row=1, col=2
        )
        
        # Add target line
        fig.add_shape(
            type="line",
            x0=0,
            x1=1,
            y0=counterfactual_explanation['target_outcome'],
            y1=counterfactual_explanation['target_outcome'],
            line=dict(color="red", dash="dot"),
            row=1, col=2
        )
        
        fig.add_annotation(
            x=0.5,
            y=counterfactual_explanation['target_outcome'],
            text=f"Target: {counterfactual_explanation['target_outcome']:.4f}",
            showarrow=False,
            yshift=10,
            row=1, col=2
        )
        
        # Update layout
        fig.update_layout(
            title="Counterfactual Explanation",
            height=500,
            width=1000,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            annotations=[
                dict(
                    x=0.1, 
                    y=1.05,
                    text="Orig: {:.4f} → CF: {:.4f}".format(
                        counterfactual_explanation['original']['prediction'],
                        counterfactual_explanation['counterfactual']['prediction']
                    ),
                    showarrow=False,
                    xref="paper",
                    yref="paper"
                )
            ]
        )
        
        # Update axes
        fig.update_xaxes(title_text="Feature", row=1, col=1)
        fig.update_yaxes(title_text="Value", row=1, col=1)
        fig.update_xaxes(title_text="", showticklabels=False, row=1, col=2)
        fig.update_yaxes(title_text="Prediction", row=1, col=2)
        
        if output_path:
            fig.write_html(output_path)
            
        return fig
    
    def create_interactive_report(self, 
                               explanations: List[Dict],
                               output_dir: str = "explanations"):
        """
        Create interactive HTML report with various explanations.
        
        Args:
            explanations: List of explanation results
            output_dir: Directory to save report files
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Create index.html
        index_html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Local Explanations Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 0; padding: 20px; }
                .container { max-width: 1200px; margin: 0 auto; }
                h1, h2, h3 { color: #333; }
                .card { border: 1px solid #ddd; border-radius: 5px; padding: 15px; margin-bottom: 20px; }
                .explanation-links { display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 20px; }
                .explanation-link { 
                    display: inline-block; padding: 8px 15px; background-color: #f0f0f0; 
                    border-radius: 5px; text-decoration: none; color: #333; 
                }
                .explanation-link:hover { background-color: #e0e0e0; }
                table { border-collapse: collapse; width: 100%; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
                tr:nth-child(even) { background-color: #f9f9f9; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Local Explanations Report</h1>
                <p>This report contains instance-level explanations for model predictions.</p>
                
                <h2>Explanation Instances</h2>
                <div class="explanation-links">
        """
        
        # Add links to each explanation
        for i, explanation in enumerate(explanations):
            index_html += f'<a class="explanation-link" href="instance_{i}.html">Instance {i}</a>\n'
        
        index_html += """
                </div>
                
                <h2>Summary Statistics</h2>
                <div class="card">
                    <table>
                        <tr>
                            <th>Instance</th>
                            <th>Prediction</th>
                            <th>Top Positive Feature</th>
                            <th>Top Negative Feature</th>
                        </tr>
        """
        
        # Add summary rows
        for i, explanation in enumerate(explanations):
            # Get prediction
            if 'shap' in explanation:
                prediction = explanation['shap']['prediction']
            elif 'lime' in explanation:
                prediction = explanation['lime']['prediction']
            else:
                prediction = "N/A"
                
            # Get top features
            top_pos_feature = "N/A"
            top_neg_feature = "N/A"
            
            if 'shap' in explanation:
                shap_values = explanation['shap']['shap_values']
                if shap_values:
                    sorted_features = sorted(shap_values.items(), key=lambda x: x[1], reverse=True)
                    top_pos_feature = f"{sorted_features[0][0]} ({sorted_features[0][1]:.4f})" if sorted_features[0][1] > 0 else "None"
                    
                    sorted_features_neg = sorted(shap_values.items(), key=lambda x: x[1])
                    top_neg_feature = f"{sorted_features_neg[0][0]} ({sorted_features_neg[0][1]:.4f})" if sorted_features_neg[0][1] < 0 else "None"
            
            index_html += f"""
                <tr>
                    <td>{i}</td>
                    <td>{prediction:.4f}</td>
                    <td>{top_pos_feature}</td>
                    <td>{top_neg_feature}</td>
                </tr>
            """
        
        index_html += """
                    </table>
                </div>
            </div>
        </body>
        </html>
        """
        
        # Write index.html
        with open(os.path.join(output_dir, "index.html"), "w") as f:
            f.write(index_html)
        
        # Create individual instance pages
        for i, explanation in enumerate(explanations):
            instance_html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Explanation for Instance {i}</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; }}
                    .container {{ max-width: 1200px; margin: 0 auto; }}
                    h1, h2, h3 {{ color: #333; }}
                    .card {{ border: 1px solid #ddd; border-radius: 5px; padding: 15px; margin-bottom: 20px; }}
                    .back-link {{ display: inline-block; margin-bottom: 20px; }}
                    iframe {{ border: none; width: 100%; height: 500px; }}
                </style>
            </head>
            <body>
                <div class="container">
                    <a href="index.html" class="back-link">← Back to Index</a>
                    <h1>Explanation for Instance {i}</h1>
            """
            
            # Generate LIME visualization if available
            if 'lime' in explanation:
                lime_path = os.path.join(output_dir, f"lime_{i}.html")
                self.visualize_lime_explanation(explanation['lime'], lime_path)
                instance_html += f"""
                    <h2>LIME Explanation</h2>
                    <div class="card">
                        <iframe src="lime_{i}.html"></iframe>
                    </div>
                """
            
            # Generate SHAP visualization if available
            if 'shap' in explanation:
                shap_path = os.path.join(output_dir, f"shap_{i}.html")
                self.visualize_shap_explanation(explanation['shap'], shap_path)
                instance_html += f"""
                    <h2>SHAP Explanation</h2>
                    <div class="card">
                        <iframe src="shap_{i}.html"></iframe>
                    </div>
                """
                
                # Generate counterfactual
                try:
                    if 'shap' in explanation:
                        instance = explanation['shap']['feature_values']
                        instance_values = [instance[f] for f in self.feature_names]
                        counterfactual = self.explain_counterfactual(instance_values)
                        
                        cf_path = os.path.join(output_dir, f"counterfactual_{i}.html")
                        self.visualize_counterfactual(counterfactual, cf_path)
                        
                        instance_html += f"""
                            <h2>Counterfactual Explanation</h2>
                            <div class="card">
                                <iframe src="counterfactual_{i}.html"></iframe>
                            </div>
                        """
                except Exception as e:
                    print(f"Error generating counterfactual for instance {i}: {e}")
            
            # Add feature values table
            if 'shap' in explanation and 'feature_values' in explanation['shap']:
                instance_html += """
                    <h2>Feature Values</h2>
                    <div class="card">
                        <table>
                            <tr>
                                <th>Feature</th>
                                <th>Value</th>
                            </tr>
                """
                
                for feature, value in explanation['shap']['feature_values'].items():
                    instance_html += f"""
                        <tr>
                            <td>{feature}</td>
                            <td>{value:.4f}</td>
                        </tr>
                    """
                
                instance_html += """
                        </table>
                    </div>
                """
            
            instance_html += """
                </div>
            </body>
            </html>
            """
            
            # Write instance.html
            with open(os.path.join(output_dir, f"instance_{i}.html"), "w") as f:
                f.write(instance_html)
        
        print(f"Interactive explanation report generated in {output_dir}")
        
def generate_local_explanations(model, X_train, X_explain, output_dir="explanations"):
    """
    Generate local explanations for a model and create an interactive report.
    
    Args:
        model: Fitted model to explain
        X_train: Training data
        X_explain: Instances to explain
        output_dir: Directory to save explanations
    """
    # Limit explanation set to a maximum of 20 instances
    if len(X_explain) > 20:
        X_explain = X_explain.sample(20)
        
    # Create explainer
    explainer = LocalExplainer(model, X_train)
    
    # Generate explanations
    explanations = explainer.explain_batch(X_explain)
    
    # Create report
    explainer.create_interactive_report(explanations, output_dir)
    
    return explanations 