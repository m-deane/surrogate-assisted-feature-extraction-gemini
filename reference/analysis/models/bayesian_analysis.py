"""
Bayesian analysis module for feature relationship uncertainty quantification.
"""

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

class BayesianAnalyzer:
    def __init__(self, random_state: int = 42):
        """
        Initialize the Bayesian analyzer.
        
        Args:
            random_state (int): Random seed for reproducibility
        """
        self.random_state = random_state
        self.models = {}
        self.traces = {}
        self.feature_importance = {}
        self.threshold_distributions = {}
        np.random.seed(random_state)
    
    def fit_bayesian_models(self, X: pd.DataFrame, y: pd.Series, 
                           features_to_analyze: Optional[List[str]] = None,
                           n_samples: int = 1000) -> Dict:
        """
        Fit Bayesian models to quantify uncertainty in feature relationships.
        
        Args:
            X (pd.DataFrame): Feature matrix
            y (pd.Series): Target values
            features_to_analyze (List[str], optional): Specific features to analyze
            n_samples (int): Number of posterior samples
            
        Returns:
            Dict: Dictionary with Bayesian analysis results
        """
        # If no specific features provided, use all
        if features_to_analyze is None:
            features_to_analyze = X.columns.tolist()
        
        # Standardize for better sampling
        X_std = (X - X.mean()) / X.std()
        y_std = (y - y.mean()) / y.std()
        
        results = {}
        
        # Fit individual models for each feature to analyze nonlinear relationships
        for feature in features_to_analyze:
            with pm.Model() as model:
                # Priors
                alpha = pm.Normal('alpha', mu=0, sigma=1)
                beta = pm.Normal('beta', mu=0, sigma=1)
                beta_sq = pm.Normal('beta_sq', mu=0, sigma=1)  # For quadratic term
                threshold = pm.Normal('threshold', mu=0, sigma=1)  # For threshold effect
                sigma = pm.HalfNormal('sigma', sigma=1)
                
                # Expected value with threshold and nonlinear effects
                x_val = X_std[feature].values
                
                # Threshold indicator (sigmoid approximation for smoothness)
                threshold_effect = pm.math.sigmoid((x_val - threshold) * 10) 
                
                # Combine linear, quadratic, and threshold effects
                mu = alpha + beta * x_val + beta_sq * (x_val**2) + threshold_effect
                
                # Likelihood
                y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y_std.values)
                
                # Sample posterior
                trace = pm.sample(n_samples, tune=500, cores=2, random_seed=self.random_state)
                
            # Store models and traces
            self.models[feature] = model
            self.traces[feature] = trace
            
            # Calculate feature importance as variance explained
            posterior_samples = az.extract(trace)
            beta_samples = posterior_samples['beta'].values
            beta_sq_samples = posterior_samples['beta_sq'].values
            
            # Importance is the variance of the effect
            importance = np.var(beta_samples * x_val + beta_sq_samples * (x_val**2))
            self.feature_importance[feature] = float(importance)
            
            # Extract threshold distribution
            threshold_samples = posterior_samples['threshold'].values
            # Convert back to original scale
            threshold_orig_scale = threshold_samples * X[feature].std() + X[feature].mean()
            self.threshold_distributions[feature] = threshold_orig_scale
            
            # Store summary statistics
            results[feature] = {
                'importance': float(importance),
                'threshold_mean': float(np.mean(threshold_orig_scale)),
                'threshold_std': float(np.std(threshold_orig_scale)),
                'threshold_95_ci': (float(np.percentile(threshold_orig_scale, 2.5)),
                                  float(np.percentile(threshold_orig_scale, 97.5))),
                'linear_effect': float(np.mean(beta_samples)),
                'nonlinear_effect': float(np.mean(beta_sq_samples))
            }
        
        return results
    
    def plot_threshold_distributions(self, output_dir: str = None):
        """
        Plot the posterior distributions of thresholds with credible intervals.
        
        Args:
            output_dir (str, optional): Directory to save plots
        """
        if not self.threshold_distributions:
            raise ValueError("No Bayesian models fitted yet. Run fit_bayesian_models first.")
        
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Sort features by importance
        sorted_features = sorted(self.feature_importance.items(), 
                               key=lambda x: x[1], reverse=True)
        
        for feature, _ in sorted_features:
            plt.figure(figsize=(10, 6))
            threshold_samples = self.threshold_distributions[feature]
            
            # Plot distribution
            sns.histplot(threshold_samples, kde=True)
            
            # Add mean and credible intervals
            mean_val = np.mean(threshold_samples)
            ci_low = np.percentile(threshold_samples, 2.5)
            ci_high = np.percentile(threshold_samples, 97.5)
            
            plt.axvline(mean_val, color='red', linestyle='--', 
                      label=f'Mean: {mean_val:.2f}')
            plt.axvline(ci_low, color='green', linestyle=':', 
                      label=f'2.5%: {ci_low:.2f}')
            plt.axvline(ci_high, color='green', linestyle=':', 
                      label=f'97.5%: {ci_high:.2f}')
            
            plt.title(f'Threshold Posterior Distribution - {feature}')
            plt.xlabel('Threshold Value')
            plt.ylabel('Density')
            plt.legend()
            
            if output_dir:
                plt.savefig(os.path.join(output_dir, f'bayesian_threshold_{feature}.png'))
                plt.close()
            else:
                plt.show()
    
    def plot_feature_effects(self, X: pd.DataFrame, output_dir: str = None):
        """
        Plot the posterior distribution of feature effects.
        
        Args:
            X (pd.DataFrame): Feature matrix
            output_dir (str, optional): Directory to save plots
        """
        if not self.traces:
            raise ValueError("No Bayesian models fitted yet. Run fit_bayesian_models first.")
        
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Sort features by importance
        sorted_features = sorted(self.feature_importance.items(), 
                               key=lambda x: x[1], reverse=True)
        
        for feature, _ in sorted_features:
            plt.figure(figsize=(12, 8))
            
            # Create grid of feature values
            x_min, x_max = X[feature].min(), X[feature].max()
            x_grid = np.linspace(x_min, x_max, 100)
            x_grid_std = (x_grid - X[feature].mean()) / X[feature].std()
            
            # Extract posterior samples
            posterior = az.extract(self.traces[feature])
            alpha_samples = posterior['alpha'].values
            beta_samples = posterior['beta'].values
            beta_sq_samples = posterior['beta_sq'].values
            threshold_samples = posterior['threshold'].values
            
            # Calculate predicted values for a subset of posterior samples
            n_curves = 100
            sample_indices = np.random.choice(len(alpha_samples), n_curves)
            
            # Calculate effect curves
            effect_curves = np.zeros((n_curves, len(x_grid_std)))
            for i, idx in enumerate(sample_indices):
                alpha = alpha_samples[idx]
                beta = beta_samples[idx]
                beta_sq = beta_sq_samples[idx]
                threshold = threshold_samples[idx]
                
                # Threshold effect (sigmoid approximation)
                threshold_effect = 1 / (1 + np.exp(-10 * (x_grid_std - threshold)))
                
                # Total effect
                effect = alpha + beta * x_grid_std + beta_sq * (x_grid_std**2) + threshold_effect
                effect_curves[i, :] = effect
            
            # Convert effects back to original scale of target
            # (simplified approach)
            
            # Plot posterior effect curves
            for i in range(n_curves):
                plt.plot(x_grid, effect_curves[i], color='blue', alpha=0.05)
            
            # Calculate and plot mean effect
            mean_effect = np.mean(effect_curves, axis=0)
            plt.plot(x_grid, mean_effect, color='red', label='Mean Effect')
            
            # Calculate and plot credible interval
            lower_ci = np.percentile(effect_curves, 2.5, axis=0)
            upper_ci = np.percentile(effect_curves, 97.5, axis=0)
            plt.fill_between(x_grid, lower_ci, upper_ci, color='red', alpha=0.2, 
                          label='95% Credible Interval')
            
            # Add threshold range
            mean_threshold = np.mean(threshold_samples) * X[feature].std() + X[feature].mean()
            plt.axvline(mean_threshold, color='green', linestyle='--', 
                      label=f'Mean Threshold: {mean_threshold:.2f}')
            
            plt.title(f'Posterior Feature Effect - {feature}')
            plt.xlabel(feature)
            plt.ylabel('Effect on Target')
            plt.legend()
            
            if output_dir:
                plt.savefig(os.path.join(output_dir, f'bayesian_effect_{feature}.png'))
                plt.close()
            else:
                plt.show()
    
    def get_feature_credible_intervals(self) -> Dict:
        """
        Get credible intervals for feature importance.
        
        Returns:
            Dict: Dictionary with feature importance credible intervals
        """
        results = {}
        
        for feature, trace in self.traces.items():
            posterior = az.extract(trace)
            beta_samples = posterior['beta'].values
            beta_sq_samples = posterior['beta_sq'].values
            
            # Compute combined effect size (simplified)
            effect_size = np.abs(beta_samples) + np.abs(beta_sq_samples)
            
            results[feature] = {
                'mean_importance': float(np.mean(effect_size)),
                'median_importance': float(np.median(effect_size)),
                'ci_low': float(np.percentile(effect_size, 2.5)),
                'ci_high': float(np.percentile(effect_size, 97.5))
            }
            
        return results 