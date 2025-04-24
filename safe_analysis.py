import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import xgboost as xgb
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import plot_tree, export_text
from sklearn.inspection import permutation_importance, partial_dependence, PartialDependenceDisplay
# Import internal function to potentially bypass check_is_fitted issue
from sklearn.inspection import _partial_dependence
from sklearn.utils import resample # For bootstrapping
from mpl_toolkits.mplot3d import Axes3D # Import for 3D plots
import shap
from lofo import LOFOImportance, Dataset, plot_importance
from itertools import combinations
from collections import Counter
import warnings
import os
import json # For saving results
from datetime import datetime # For report timestamp
# Import for Granger Causality
from statsmodels.tsa.stattools import grangercausalitytests, adfuller
import statsmodels.api as sm # Optional for stationarity checks
# Import for RuleFit
from imodels import RuleFitRegressor

warnings.filterwarnings('ignore')

# --- Report Generator Class (Copied from reference) ---
# NOTE: Minor adaptations might be needed based on actual script variables
class ReportGenerator:
    def __init__(self):
        """Initialize the report generator."""
        self.report_sections = []
        self.results_dict = {} # Store results for JSON saving

    def add_section(self, title: str, content: str):
        """
        Add a section to the report.
        """
        self.report_sections.append({
            'title': title,
            'content': content
        })

    def add_dataset_overview(self, df_shape, features_list, date_range):
        """Adds dataset overview section."""
        content = f"""- **Dataset Size**: {df_shape[0]} samples, {df_shape[1]} columns (including target/date)
- **Features Analyzed**: {len(features_list)} numeric features
- **Date Range**: {date_range[0]} to {date_range[1]}
"""
        self.add_section("1. Dataset Overview", content)
        self.results_dict['dataset_overview'] = {
            'shape': df_shape,
            'num_features': len(features_list),
            'date_start': str(date_range[0]),
            'date_end': str(date_range[1])
        }

    def add_surrogate_model_performance(self, model_type, train_rmse, train_r2, test_rmse, test_r2):
        """Adds performance metrics for the single surrogate model."""
        content = f"""### Surrogate Model ({model_type}) Performance
- **Training RMSE**: {train_rmse:.4f}
- **Training R²**: {train_r2:.4f}
- **Test RMSE**: {test_rmse:.4f}
- **Test R²**: {test_r2:.4f}

*Interpretation Guidance:*
* *R²*: Proportion of variance explained (closer to 1 is better). Negative R² indicates the model performs worse than a horizontal line.
* *RMSE*: Root Mean Squared Error (lower is better), in the units of the target variable.
* *Train vs. Test Gap*: A large difference often indicates overfitting.
"""
        self.add_section("5. Model Performance", content) # Use section number from example md
        self.results_dict['surrogate_performance'] = {
            'model_type': model_type,
            'train_rmse': train_rmse,
            'train_r2': train_r2,
            'test_rmse': test_rmse,
            'test_r2': test_r2
        }

    def add_feature_importance_summary(self, shap_df, combined_imp_df):
        """Adds summary of top features based on SHAP and combined rank."""
        content = "### Top Features by SHAP Analysis (Train Set):\n"
        if shap_df is not None and not shap_df.empty:
            for i, row in shap_df.head(5).iterrows():
                content += f"- `{row['feature']}` (Mean Abs SHAP: {row['shap_importance']:.3f})\n"
            self.results_dict['top_features_shap'] = shap_df.head(5).to_dict('records')
        else:
            content += "- (SHAP analysis not available)\n"

        content += "\n### Top Features by Combined Importance Rank (Train Set):\n"
        if combined_imp_df is not None and not combined_imp_df.empty and 'mean_rank' in combined_imp_df.columns:
            for i, (idx, row) in enumerate(combined_imp_df.head(5).iterrows()):
                content += f"- `{idx}` (Mean Rank: {row['mean_rank']:.2f})\n"
            self.results_dict['top_features_combined'] = combined_imp_df.head(5).to_dict('index')
        else:
            content += "- (Combined ranking not available)\n"

        self.add_section("2. Key Features and Their Importance", content)

    def add_feature_interactions_summary(self, h_values, interaction_stability, h_3way_stats=None):
        """Adds feature interaction summary."""
        content = "### Significant Pairwise Interactions (Approx. H-statistic):\n"
        if h_values is not None and not h_values.empty:
            # Display top N interactions
            top_h_interactions_dict = {}
            count = 0
            for (f1, f2), h_stat in h_values.head(10).items():
                content += f"- `{f1}` × `{f2}` (H ≈ {h_stat:.3f})"
                top_h_interactions_dict[f'{f1}_x_{f2}'] = h_stat # Store for JSON
                # Add stability info if available
                pair_key = tuple(sorted((f1, f2)))
                if pair_key in interaction_stability:
                    stab = interaction_stability[pair_key]
                    content += f" (Stable: Mean={stab['mean']:.3f} ± {stab['std']:.3f})\n"
                else:
                    content += "\n"
                count += 1
                if count >= 5: # Limit report summary to top 5
                    break
            self.results_dict['pairwise_interactions_h_top5'] = top_h_interactions_dict
            self.results_dict['interaction_stability'] = interaction_stability
        else:
            content += "- (No significant pairwise interactions found or calculated)\n"

        content += "\n### Three-way Interactions (Approx. H-statistic for Top Feature Triplets):\n"
        if h_3way_stats is not None and not h_3way_stats.empty:
            # Sort by H-statistic descending
            h_3way_stats_sorted = h_3way_stats.sort_values('H_statistic', ascending=False)
            for idx, row in h_3way_stats_sorted.iterrows():
                triplet_str = ' × '.join(row['Triplet'])
                content += f"- `{triplet_str}` (H ≈ {row['H_statistic']:.4f})\n"
            self.results_dict['threeway_interactions_h'] = h_3way_stats_sorted.to_dict('records')
        else:
            content += "- (Not calculated or available for the selected triplets)\n"

        self.add_section("3. Feature Interactions", content)

    def add_condition_importance_summary(self, condition_importance_df):
        """Adds summary of top rule conditions based on importance."""
        content = "### Top Impactful Conditions/Thresholds (by Permutation Importance):\n"
        if condition_importance_df is not None and not condition_importance_df.empty:
            for i, row in condition_importance_df.head(5).iterrows():
                feat, op_thresh = row['Original_Condition']
                content += f"- `{feat} {op_thresh}` (Importance: {row['Importance']:.4f})\n"
            self.results_dict['top_conditions_by_importance'] = condition_importance_df.head(5).to_dict('records')
        else:
            content += "- (Condition importance not available or calculated)\n"

        self.add_section("4. Critical Thresholds / Conditions", content)

    def add_limitations_section(self):
        content = """- Dataset size might limit generalizability.
- Training-test performance gap may indicate potential overfitting or concept drift.
- H-statistic values are approximations.
- Condition importance is a proxy derived from a secondary model.
- Temporal dynamics beyond simple train/test split are not explicitly modeled in this script (e.g., complex seasonality, autocorrelation in residuals)."""
        self.add_section("7. Limitations", content)

    def generate_report(self, output_path: str = None) -> str:
        """
        Generate the final report in markdown format.
        """
        report = f"""# SAFE Analysis Summary Report
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 

## Executive Summary
This report summarizes findings from the Surrogate-Assisted Feature Extraction (SAFE) workflow.
It highlights key features, interactions, and decision boundaries identified by analyzing a
surrogate model trained on the data.

"""
        for section in self.report_sections:
            report += f"## {section['title']}\n"
            report += f"{section['content']}\n\n"

        if output_path:
            try:
                with open(output_path, 'w') as f:
                    f.write(report)
                print(f"Successfully generated report: {output_path}")
            except Exception as e:
                print(f"Error writing report file: {e}")

        return report

    def _convert_to_serializable(self, obj):
        # ... (Remains the same) ...
        if isinstance(obj, dict):
            return {
                str(k) if isinstance(k, (np.int64, np.int32)) else str(k): # Ensure keys are strings
                self._convert_to_serializable(v)
                for k, v in obj.items()
            }
        elif isinstance(obj, list):
            return [self._convert_to_serializable(item) for item in obj]
        elif isinstance(obj, tuple): # Convert tuples (e.g., from dict keys) to strings
            return '__'.join(map(str, obj))
        elif isinstance(obj, pd.Series):
            # Convert Series index if necessary, then values
            return {str(k): self._convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, pd.DataFrame):
            # Make a copy to avoid modifying original
            df_copy = obj.copy()
            # Convert index to string if it's not numeric
            if not pd.api.types.is_numeric_dtype(df_copy.index.dtype):
                 df_copy.index = df_copy.index.map(str)
            # Convert columns to string
            df_copy.columns = df_copy.columns.map(str)
            return df_copy.to_dict(orient='split') # 'split' orientation is often json friendly
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, np.uint8,
            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float16, np.float32, np.float64)):
            return float(obj)
        # --- Update Check for NumPy Complex Types ---
        elif isinstance(obj, (np.complex64, np.complex128)):
            return {'real': obj.real, 'imag': obj.imag}
        # --- End Update ---
        elif isinstance(obj, (np.bool_)):
            return bool(obj)
        elif isinstance(obj, (np.void)):
            return None # Or other representation
        elif isinstance(obj, (pd.Timestamp, datetime)):
            return obj.isoformat()
        elif isinstance(obj, Counter):
            return dict(obj)
        else:
            # Fallback for other types - attempt string conversion
            try:
                # Check if it's basic type first
                if isinstance(obj, (str, int, float, bool, type(None))):
                    return obj
                # Attempt json dump for complex objects? Risky.
                # Safest fallback is string representation
                return str(obj)
            except Exception:
                return f"<unserializable type: {type(obj).__name__}>"

    def save_analysis_results(self, output_path: str):
        """
        Save key analysis results stored in self.results_dict to a JSON file.
        """
        print(f"Attempting to save analysis results to: {output_path}")
        try:
            # Ensure all data added via methods is in results_dict
            # Add other relevant data if not already added by specific methods
            if 'combined_feature_importance' not in self.results_dict and 'combined_imp' in globals() and combined_imp is not None:
                self.results_dict['combined_feature_importance'] = combined_imp
            if 'mdi_importance' not in self.results_dict and 'mdi_df' in globals() and mdi_df is not None:
                self.results_dict['mdi_importance'] = mdi_df
            if 'permutation_importance' not in self.results_dict and 'perm_df' in globals() and perm_df is not None:
                self.results_dict['permutation_importance'] = perm_df
            if 'shap_importance' not in self.results_dict and 'shap_df' in globals() and shap_df is not None:
                self.results_dict['shap_importance'] = shap_df
            if 'rule_conditions' not in self.results_dict and 'rule_conditions' in globals() and rule_conditions:
                self.results_dict['rule_conditions_frequency'] = rule_conditions
            if 'rulefit_rules' not in self.results_dict and 'rulefit_rules_df' in globals() and rulefit_rules_df is not None:
                 self.results_dict['rulefit_rules'] = rulefit_rules_df

            serializable_results = self._convert_to_serializable(self.results_dict)
            with open(output_path, 'w') as f:
                json.dump(serializable_results, f, indent=4)
            print(f"Successfully saved analysis results to: {output_path}")
        except Exception as e:
            print(f"Error saving analysis results to JSON: {e}")
            # Optionally print the problematic dictionary structure
            # import pprint
            # try:
            #     pprint.pprint(self.results_dict)
            # except Exception as pe:
            #     print(f"Could not pretty-print results_dict: {pe}")

    def add_granger_causality_summary(self, p_value_matrix, max_lag):
        """Adds Granger causality heatmap info to the report."""
        content = f"Pairwise Granger causality tests performed up to lag {max_lag}. "
        content += "Lower p-values suggest one variable Granger-causes another (helps predict its future values).\n\n"
        content += "See the heatmap plot (`causality_granger_heatmap.png/.html`) for details.\n"
        content += "Note: Granger causality checks for predictive power, not necessarily true causation, and assumes stationarity.\n"
        self.add_section("8. Causality Analysis (Granger)", content)
        # Store the matrix itself in results
        if p_value_matrix is not None:
            self.results_dict['granger_causality_p_values'] = p_value_matrix

    def add_stationarity_summary(self, stationarity_results):
        """Adds summary of ADF stationarity tests to the report."""
        content = "ADF Test performed on training features to check for stationarity (required for Granger causality)."\
                  " A p-value <= 0.05 suggests stationarity.\n\n"
        num_stationary = 0
        for feature, result in stationarity_results.items():
            p_val = result['p_value']
            is_stationary = p_val <= 0.05
            if is_stationary:
                 num_stationary += 1
            content += f"- **`{feature}`**: ADF Stat={result['adf_stat']:.3f}, p-value={p_val:.3f} -> {'Stationary' if is_stationary else 'Non-Stationary'}\n"

        content += f"\nSummary: {num_stationary} out of {len(stationarity_results)} features appear stationary (p<=0.05).\n"
        content += "Non-stationary features may violate Granger assumptions.\n"
        self.add_section("8a. Stationarity Tests (ADF)", content)
        self.results_dict['stationarity_test_adf'] = stationarity_results

    def add_shap_interaction_summary(self, shap_interaction_df):
        """Adds summary of top SHAP interactions."""
        content = "Top pairwise feature interactions based on Mean Absolute SHAP Interaction values.\n\n"
        if shap_interaction_df is not None and not shap_interaction_df.empty:
            # Extract top off-diagonal values
            inter_values = shap_interaction_df.mask(np.equal(*np.indices(shap_interaction_df.shape))).stack().sort_values(ascending=False)
            for (f1, f2), val in inter_values.head(10).items():
                 content += f"- `{f1}` <> `{f2}`: {val:.4f}\n"
            self.results_dict['top_shap_interactions'] = inter_values.head(10).to_dict()
        else:
             content += "- (SHAP interaction values not calculated or available).\n"
        # Add this as part of section 3
        # Find section 3 index
        sec3_idx = -1
        for i, sec in enumerate(self.report_sections):
            if sec['title'].startswith("3."):
                 sec3_idx = i
                 break
        if sec3_idx != -1:
             self.report_sections[sec3_idx]['content'] += "\n\n### Top SHAP Interactions (Mean Abs Value):\n" + content
        else: # Add as new section if 3 doesn't exist
             self.add_section("3b. SHAP Interaction Analysis", content)

    # Modify save_analysis_results to include new data
    def save_analysis_results(self, output_path: str):
        """ Save key analysis results stored in self.results_dict to a JSON file. """
        print(f"Attempting to save analysis results to: {output_path}")
        try:
            # Add any remaining relevant dataframes/results if not added by specific methods
            # (Ensure these variables exist in the scope where save_analysis_results is called)
            global_vars = globals()
            if 'combined_feature_importance' not in self.results_dict and 'combined_imp' in global_vars and global_vars['combined_imp'] is not None:
                self.results_dict['combined_feature_importance'] = global_vars['combined_imp']
            if 'mdi_importance' not in self.results_dict and 'mdi_df' in global_vars and global_vars['mdi_df'] is not None:
                 self.results_dict['mdi_importance'] = global_vars['mdi_df']
            if 'permutation_importance' not in self.results_dict and 'perm_df' in global_vars and global_vars['perm_df'] is not None:
                 self.results_dict['permutation_importance'] = global_vars['perm_df']
            if 'shap_importance' not in self.results_dict and 'shap_df' in global_vars and global_vars['shap_df'] is not None:
                 self.results_dict['shap_importance'] = global_vars['shap_df']
            if 'rule_conditions_frequency' not in self.results_dict and 'rule_conditions' in global_vars and global_vars['rule_conditions']:
                 self.results_dict['rule_conditions_frequency'] = global_vars['rule_conditions']
            if 'shap_interaction_df' not in self.results_dict and 'shap_interaction_df' in global_vars and shap_interaction_df is not None:
                 self.results_dict['shap_interaction_values_mean_abs'] = shap_interaction_df
            # Note: Granger p_values and stationarity added by their specific methods

            serializable_results = self._convert_to_serializable(self.results_dict)
            with open(output_path, 'w') as f:
                json.dump(serializable_results, f, indent=4)
            print(f"Successfully saved analysis results to: {output_path}")
        except Exception as e:
            print(f"Error saving analysis results to JSON: {e}")

    def add_granger_causality_summary(self, p_value_series, max_lag):
        """Adds Granger causality (feature -> target) summary to the report."""
        content = f"Pairwise Granger causality tests performed for each feature predicting the target variable, up to lag {max_lag}. "
        content += "Lower p-values suggest a feature Granger-causes the target (helps predict its future values).\n\n"
        content += "See the bar chart plot (`causality_granger_bar.png/.html`) for details.\n"
        content += "Note: Granger causality checks for predictive power, not necessarily true causation, and assumes stationarity.\n"
        self.add_section("8b. Causality Analysis (Granger Feature -> Target)", content)
        # Store the series itself in results
        if p_value_series is not None:
            self.results_dict['granger_causality_feature_to_target_p_values'] = p_value_series

    def add_dynamic_summary(self, combined_imp, condition_importance_df, h_values, interaction_stability,
                             shap_interaction_df, granger_p_values, stationarity_results,
                             surrogate_perf, linear_perf, engineered_feature_names):
        """Generates a dynamic narrative summary of key findings."""
        summary_text = """
This analysis utilized a RandomForest surrogate model to explore feature relationships and importance.
Key findings are summarized below:

"""
        key_findings_dict = {}

        # Surrogate Model Performance
        summary_text += "**Model Performance:**\n"
        train_r2 = surrogate_perf.get('train_r2', -999)
        test_r2 = surrogate_perf.get('test_r2', -999)
        summary_text += f"- The surrogate model achieved a training R² of {train_r2:.3f}.\n"
        if test_r2 < 0 or (train_r2 - test_r2 > 0.5 and train_r2 > 0): # Check for negative R2 or large gap
            summary_text += f"- However, test R² ({test_r2:.3f}) indicates poor generalization or significant overfitting, warranting caution.\n"
        else:
            summary_text += f"- Test R² was {test_r2:.3f}, suggesting reasonable generalization.\n"
        key_findings_dict['surrogate_performance_summary'] = {'train_r2': train_r2, 'test_r2': test_r2}

        # Key Drivers (Features)
        summary_text += "\n**Key Features:**\n"
        if combined_imp is not None and not combined_imp.empty and 'mean_rank' in combined_imp.columns:
            top_features = combined_imp.index[:3].tolist()
            summary_text += f"- Consistently high importance across methods was observed for: `{top_features[0]}`, `{top_features[1]}`, and `{top_features[2]}`.\n"
            key_findings_dict['top_ranked_features'] = top_features
        else:
            summary_text += "- Combined feature importance ranking was not available.\n"

        # Key Conditions/Thresholds
        summary_text += "\n**Key Conditions/Thresholds:**\n"
        if condition_importance_df is not None and not condition_importance_df.empty:
            top_conditions = condition_importance_df['Original_Condition'].head(3).tolist()
            summary_text += "- The analysis of rule conditions suggests specific thresholds are particularly influential (based on permutation importance of condition-derived features):\n"
            for cond in top_conditions:
                summary_text += f"  - Condition: `{cond[0]} {cond[1]}`\n"
            key_findings_dict['top_important_conditions'] = top_conditions
        else:
            summary_text += "- Analysis of specific condition importance was not performed or yielded no results.\n"

        # Key Interactions
        summary_text += "\n**Feature Interactions:**\n"
        top_stable_interactions = []
        if h_values is not None and not h_values.empty:
            summary_text += "- Pairwise interactions were assessed using an approximate H-statistic:\n"
            count = 0
            for (f1, f2), h_stat in h_values.head(5).items(): # Check top 5 H-stats for stability
                stability_info = " (Stability not assessed)"
                is_stable = False
                pair_key = tuple(sorted((f1, f2)))
                if pair_key in interaction_stability:
                    stab = interaction_stability[pair_key]
                    # Define stability (e.g., low std dev relative to mean)
                    if stab['std'] < 0.1 * abs(stab['mean']): # Example threshold
                         stability_info = f" (Stable: Mean={stab['mean']:.3f} ± {stab['std']:.3f})"
                         is_stable = True
                    else:
                         stability_info = f" (Less Stable: Mean={stab['mean']:.3f} ± {stab['std']:.3f})"
                
                if count < 2 or is_stable: # Report top 2, plus any other stable ones in top 5
                     summary_text += f"  - `{f1}` × `{f2}` (H ≈ {h_stat:.3f}){stability_info}\n"
                     if is_stable:
                          top_stable_interactions.append((f1,f2))
                count += 1
            key_findings_dict['top_h_interactions'] = h_values.head(5).to_dict()
            key_findings_dict['top_stable_interactions'] = top_stable_interactions
        elif shap_interaction_df is not None: # Fallback to SHAP if H failed
             summary_text += "- Pairwise interactions were assessed using SHAP interaction values:\n"
             inter_values = shap_interaction_df.mask(np.equal(*np.indices(shap_interaction_df.shape))).stack().sort_values(ascending=False)
             for (f1, f2), val in inter_values.head(3).items():
                 summary_text += f"  - `{f1}` <> `{f2}` (Mean Abs SHAP Inter: {val:.4f})\n"
             key_findings_dict['top_shap_interactions'] = inter_values.head(3).to_dict()
        else:
            summary_text += "- Interaction analysis was not performed or yielded no results.\n"

        # Feature Engineering Impact
        summary_text += "\n**Feature Engineering Impact:**\n"
        if linear_perf.get('engineered_features_created', False):
            r2_orig = linear_perf.get('r2_orig', -999)
            r2_eng = linear_perf.get('r2_eng', -999)
            improvement = r2_eng > r2_orig
            summary_text += f"- Features engineered from surrogate insights ({len(engineered_feature_names)} created: threshold={linear_perf.get('threshold_features', 0)}, interaction={linear_perf.get('interaction_features', 0)}) "
            if improvement:
                summary_text += f"led to an *improvement* in the linear model's test R² (from {r2_orig:.3f} to {r2_eng:.3f}).\n"
            elif r2_eng == r2_orig:
                 summary_text += f"did *not change* the linear model's test R² ({r2_orig:.3f}).\n"
            else:
                summary_text += f"*did not improve* the linear model's test R² (from {r2_orig:.3f} to {r2_eng:.3f}).\n"
            key_findings_dict['feature_engineering_result'] = {'improved': improvement, 'r2_orig': r2_orig, 'r2_eng': r2_eng}
        else:
            summary_text += "- Feature engineering based on surrogate insights was not performed or did not yield usable features.\n"
            key_findings_dict['feature_engineering_result'] = {'improved': None}

        # Causality Insights
        summary_text += "\n**Potential Predictive Relationships (Granger Causality):**\n"
        significant_granger = []
        if granger_p_values is not None and not granger_p_values.isnull().all():
            significant_granger = granger_p_values[granger_p_values <= 0.05].index.tolist()
            if significant_granger:
                summary_text += "- The following features showed significant Granger causality towards the target (p<=0.05):\n"
                for feat in significant_granger:
                    stationarity = stationarity_results.get(feat, {})
                    non_stationary_warning = "" if stationarity.get('is_stationary', True) else " (Warning: Non-Stationary)"
                    summary_text += f"  - `{feat}`{non_stationary_warning}\n"
                summary_text += "- Note: Granger causality suggests predictive power but not true causation; non-stationarity can affect validity.\n"
            else:
                summary_text += "- No features showed significant Granger causality for the target at the tested lag.\n"
            key_findings_dict['significant_granger_features'] = significant_granger
        else:
            summary_text += "- Granger causality analysis was not performed or yielded no results.\n"

        # Add this summary as Section 0
        # Check if section 0 already exists, if so, replace it, otherwise insert at beginning
        section_0_exists = any(sec['title'].startswith("0.") for sec in self.report_sections)
        if section_0_exists:
             for i, sec in enumerate(self.report_sections):
                  if sec['title'].startswith("0."):
                       self.report_sections[i] = {'title': "0. Analysis Highlights", 'content': summary_text}
                       break
        else:
             self.report_sections.insert(0, {'title': "0. Analysis Highlights", 'content': summary_text})

        # Add summary findings to main results dict
        self.results_dict['analysis_highlights'] = key_findings_dict

    def add_rulefit_summary(self, rules_df):
        """Adds summary of top rules from RuleFitRegressor."""
        content = "RuleFitRegressor was trained to extract interpretable rules. "
        content += "Rules combine feature thresholds; coefficients indicate importance.\n\n"
        content += "**Top Rules by Coefficient Magnitude:**\n"
        if rules_df is not None and not rules_df.empty:
            # Sort by absolute coefficient for importance
            rules_df_sorted = rules_df.iloc[rules_df['coef'].abs().sort_values(ascending=False).index]
            for i, row in rules_df_sorted.head(10).iterrows(): # Show top 10
                content += f"- **Rule:** `{row['rule']}` (Coef: {row['coef']:.4f}, Support: {row['support']:.3f})\n"
            # Add to results dict for JSON
            self.results_dict['rulefit_top_rules'] = rules_df_sorted.head(10).to_dict('records')
        else:
            content += "- (RuleFit analysis not performed or yielded no rules).\n"
            self.results_dict['rulefit_top_rules'] = []

        # Add after section 4 (Thresholds/Conditions)
        sec4_idx = -1
        for i, sec in enumerate(self.report_sections):
            if sec['title'].startswith("4."):
                sec4_idx = i
                break
        if sec4_idx != -1:
            self.report_sections.insert(sec4_idx + 1, {'title': "4b. RuleFit Extracted Rules", 'content': content})
        else: # Fallback add at end
             self.add_section("4b. RuleFit Extracted Rules", content)

    # Modify save_analysis_results to potentially include RuleFit df
    def save_analysis_results(self, output_path: str):
        """ Save key analysis results stored in self.results_dict to a JSON file. """
        print(f"Attempting to save analysis results to: {output_path}")
        try:
            # Ensure all data added via methods is in results_dict
            # Add other relevant data if not already added by specific methods
            global_vars = globals()
            if 'combined_feature_importance' not in self.results_dict and 'combined_imp' in global_vars and global_vars['combined_imp'] is not None:
                self.results_dict['combined_feature_importance'] = global_vars['combined_imp']
            if 'mdi_importance' not in self.results_dict and 'mdi_df' in global_vars and global_vars['mdi_df'] is not None:
                 self.results_dict['mdi_importance'] = global_vars['mdi_df']
            if 'permutation_importance' not in self.results_dict and 'perm_df' in global_vars and global_vars['perm_df'] is not None:
                 self.results_dict['permutation_importance'] = global_vars['perm_df']
            if 'shap_importance' not in self.results_dict and 'shap_df' in global_vars and global_vars['shap_df'] is not None:
                 self.results_dict['shap_importance'] = global_vars['shap_df']
            if 'rule_conditions_frequency' not in self.results_dict and 'rule_conditions' in global_vars and global_vars['rule_conditions']:
                 self.results_dict['rule_conditions_frequency'] = global_vars['rule_conditions']
            if 'shap_interaction_df' not in self.results_dict and 'shap_interaction_df' in global_vars and shap_interaction_df is not None:
                 self.results_dict['shap_interaction_values_mean_abs'] = shap_interaction_df
            # Add RuleFit rules if calculated
            if 'rulefit_rules' not in self.results_dict and 'rulefit_rules_df' in global_vars and rulefit_rules_df is not None:
                 self.results_dict['rulefit_rules'] = rulefit_rules_df
            # Note: Granger p_values and stationarity added by their specific methods

            serializable_results = self._convert_to_serializable(self.results_dict)
            with open(output_path, 'w') as f:
                json.dump(serializable_results, f, indent=4)
            print(f"Successfully saved analysis results to: {output_path}")
        except Exception as e:
            print(f"Error saving analysis results to JSON: {e}")

    # ... (Rest of ReportGenerator methods) ...

# --- End Report Generator Class ---

# --- Configuration ---
DATA_PATH = '_data/preem.csv'
TARGET_VARIABLE = 'target'
DATE_COLUMN = 'date' # Assuming a 'date' column exists for time series
GROUPING_COLUMN = 'country' # Optional: Set to None to disable filtering
FILTER_VALUE = 'USA' # Optional: Value to filter by if GROUPING_COLUMN is set
TEST_MONTHS = 12 # Number of months for the test set
SURROGATE_MODEL_TYPE = 'random_forest' # 'xgboost' or 'random_forest'
N_TREES_TO_VISUALIZE = 1
N_TOP_FEATURES = 10 # For summaries
N_BOOTSTRAP_SAMPLES = 10 # For interaction stability (can be slow)
N_TOP_FEATURES_FOR_PDP_3D = 5 # Limit Plotly 3D PDPs to top N features
OUTPUT_DIR = 'safe_analysis' # Directory to save plots

# Create output directory if it doesn't exist
# Use exist_ok=True for robustness
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Ensured output directory exists: {OUTPUT_DIR}")

# --- Instantiate Report Generator ---
report_generator = ReportGenerator()

# --- Initialize result placeholders ---
# This ensures variables exist even if sections fail/are skipped
rulefit_rules_df = None
condition_importance_df = pd.DataFrame() # From Sec 6b
condition_mdi_df = pd.DataFrame()        # From Sec 6b
condition_shap_df = pd.DataFrame()       # From Sec 6b
condition_linear_results = None          # From Sec 6b.1
combined_imp = pd.DataFrame()            # From importance combination step
shap_df = None                           # From Sec 6 SHAP
h_values = None                          # From Sec 7 H-stats
interaction_stability = {}               # From Sec 7 H-stats stability
h_3way = None                            # From Sec 7 3-way H-stat
shap_interaction_df = None               # From Sec 6 SHAP Interaction
granger_p_values = None                  # From Sec 8 Granger
stationarity_results = {}                # From Sec 8 Stationarity
# Variables from optional linear comparison (Sec 12)
linear_perf = {}
engineered_feature_names = []


# --- Helper Functions ---

def save_plot(fig, filename, output_dir=OUTPUT_DIR):
    """Saves plotly figure as HTML and static image, or matplotlib fig as png."""
    if fig is None:
        print(f"Skipping save for {filename}, figure is None.")
        return
    try:
        # Ensure filename has .png for static matplotlib, handle plotly separately
        if isinstance(fig, go.Figure):
            filepath_html = os.path.join(output_dir, f"{filename}.html")
            filepath_png = os.path.join(output_dir, f"{filename}.png")
            fig.write_html(filepath_html)
            try:
                fig.write_image(filepath_png)
                print(f"Saved plot: {filepath_html} and {filepath_png}")
            except ValueError as e:
                print(f"Could not save static Plotly image {filepath_png}. Ensure kaleido or orca is installed. Error: {e}")
                print(f"Saved plot (HTML only): {filepath_html}")
        elif isinstance(fig, plt.Figure):
             filepath_png = os.path.join(output_dir, f"{filename}.png")
             fig.savefig(filepath_png, bbox_inches='tight')
             plt.close(fig) # Close plot after saving
             print(f"Saved plot: {filepath_png}")
        elif hasattr(fig, 'figure') and isinstance(fig.figure, plt.Figure):
             # Handle cases like seaborn plots where the figure is an attribute
             filepath_png = os.path.join(output_dir, f"{filename}.png")
             fig.figure.savefig(filepath_png, bbox_inches='tight')
             plt.close(fig.figure)
             print(f"Saved plot: {filepath_png}")
        else:
            print(f"Warning: Unsupported figure type for saving: {type(fig)}")

    except Exception as e:
        print(f"Error saving plot {filename}: {e}")

def time_series_split_by_month(df, date_col, test_months):
    """Splits time series data based on the last N months."""
    df = df.sort_index() # Ensure data is sorted by time index
    split_date = df.index.max() - pd.DateOffset(months=test_months)
    train_df = df[df.index <= split_date]
    test_df = df[df.index > split_date]
    print(f"Time Series Split:")
    print(f"  Train range: {train_df.index.min()} to {train_df.index.max()} ({len(train_df)} rows)")
    print(f"  Test range:  {test_df.index.min()} to {test_df.index.max()} ({len(test_df)} rows)")
    return train_df, test_df

def get_tree_rules_and_conditions(tree_model, feature_names):
    """Extracts conditions (feature-threshold pairs) from all trees in a RandomForest."""
    conditions = Counter()
    all_rules_text = []

    if not hasattr(tree_model, 'estimators_'):
        print("Rule/condition extraction only implemented for RandomForest.")
        return conditions, all_rules_text

    print(f"Extracting conditions from {len(tree_model.estimators_)} trees...")
    for i, tree in enumerate(tree_model.estimators_):
        if i == 0: # Export text for first tree only
            try:
                all_rules_text.append(export_text(tree, feature_names=feature_names))
            except Exception as e:
                print(f"Could not export text for tree {i}: {e}")

        tree_structure = tree.tree_
        node_count = tree_structure.node_count
        for node_idx in range(node_count):
            # If it's a split node
            if tree_structure.children_left[node_idx] != tree_structure.children_right[node_idx]:
                feature_index = tree_structure.feature[node_idx]
                threshold = tree_structure.threshold[node_idx]
                if feature_index >= 0 and feature_index < len(feature_names): # Valid feature index
                    feature = feature_names[feature_index]
                    # Store condition as a tuple (feature, operator, threshold)
                    condition_left = (feature, f"<={threshold:.3f}") # Representing the left split
                    conditions[condition_left] += 1
                    # Also count the inverse condition implicitly used for the right split
                    condition_right = (feature, f"> {threshold:.3f}")
                    conditions[condition_right] += 1

    return conditions, all_rules_text

def friedman_h_statistic(model, X, feature1, feature2):
    """(Approximation) Calculates Friedman's H-statistic for pairwise interaction."""
    feature_names = X.columns.tolist()
    try:
        f1_idx = feature_names.index(feature1)
        f2_idx = feature_names.index(feature2)
    except ValueError as e:
        print(f"Error finding feature index for H-stat: {e}")
        return np.nan

    try:
        # Create grid points for each feature
        grid_resolution = 30  # Keeping the same resolution as before
        features = [feature1, feature2]
        
        grid_points = {}
        for feature in features:
            x_min, x_max = X[feature].min(), X[feature].max()
            grid_points[feature] = np.linspace(x_min, x_max, grid_resolution)
        
        # Calculate partial dependence for individual features
        individual_pd = {}
        for feature in features:
            grid = grid_points[feature]
            pd_values = np.zeros_like(grid, dtype=float)
            
            for i, value in enumerate(grid):
                X_temp = X.copy()
                X_temp[feature] = value
                pd_values[i] = model.predict(X_temp).mean()
                
            individual_pd[feature] = pd_values
        
        # Calculate the sum of individual partial dependence effects
        sum_individual_effects = np.zeros((grid_resolution, grid_resolution))
        
        # Add first feature effect (broadcast across columns)
        for i in range(grid_resolution):
            sum_individual_effects[i, :] += individual_pd[feature1][i]
        
        # Add second feature effect (broadcast across rows)
        for j in range(grid_resolution):
            sum_individual_effects[:, j] += individual_pd[feature2][j]
        
        # Calculate mean prediction (constant effect)
        mean_prediction = model.predict(X).mean()
        
        # Adjust for double-counting of the mean in the sum of individual effects
        sum_individual_effects -= mean_prediction
        
        # Calculate joint partial dependence
        joint_pd = np.zeros((grid_resolution, grid_resolution))
        
        for i, val1 in enumerate(grid_points[feature1]):
            for j, val2 in enumerate(grid_points[feature2]):
                X_temp = X.copy()
                X_temp[feature1] = val1
                X_temp[feature2] = val2
                joint_pd[i, j] = model.predict(X_temp).mean()
        
        # Calculate interaction effect
        interaction_effect = joint_pd - sum_individual_effects
        
        # Variance of the interaction effect
        var_interaction = np.var(interaction_effect)
        
        # Variance of the joint partial dependence
        var_joint = np.var(joint_pd)
        
        # Calculate H-statistic (normalized measure of interaction strength)
        h_stat = var_interaction / (var_joint + 1e-8)  # Add small constant to avoid division by zero
        
        # Keep as-is but ensure it's between 0 and 1
        return max(0, min(h_stat, 1.0))

    except Exception as e:
        # print(f"Error calculating H-statistic for ({feature1}, {feature2}): {e}")
        return np.nan # Return NaN if calculation fails

def friedman_h_3way(model, X, feature1, feature2, feature3):
    """(Approximation) Calculates Friedman's H-statistic for three-way interaction."""
    feature_names = X.columns.tolist()
    try:
        f1_idx = feature_names.index(feature1)
        f2_idx = feature_names.index(feature2)
        f3_idx = feature_names.index(feature3)
    except ValueError as e:
        print(f"Error finding feature index for 3-way H-stat: {e}")
        return np.nan

    try:
        # Use lower grid resolution for three-way interactions to improve performance
        grid_resolution = 10
        features = [feature1, feature2, feature3]
        
        # Create grid points for each feature
        grid_points = {}
        for feature in features:
            x_min, x_max = X[feature].min(), X[feature].max()
            grid_points[feature] = np.linspace(x_min, x_max, grid_resolution)
        
        # Calculate partial dependence for individual features
        individual_pd = {}
        for feature in features:
            grid = grid_points[feature]
            pd_values = np.zeros_like(grid, dtype=float)
            
            for i, value in enumerate(grid):
                X_temp = X.copy()
                X_temp[feature] = value
                pd_values[i] = model.predict(X_temp).mean()
                
            individual_pd[feature] = pd_values
        
        # Calculate the sum of individual partial dependence effects
        sum_individual_effects = np.zeros((grid_resolution, grid_resolution, grid_resolution))
        
        # Add individual feature effects (broadcast across dimensions)
        for i in range(grid_resolution):
            for j in range(grid_resolution):
                for k in range(grid_resolution):
                    sum_individual_effects[i, j, k] = (
                        individual_pd[feature1][i] + 
                        individual_pd[feature2][j] + 
                        individual_pd[feature3][k]
                    )
        
        # Calculate mean prediction (constant effect)
        mean_prediction = model.predict(X).mean()
        
        # Adjust for double-counting of the mean in the sum of individual effects
        # For 3 individual features, we added the mean 3 times, so subtract 2 times
        sum_individual_effects -= 2 * mean_prediction
        
        # Calculate joint partial dependence
        joint_pd = np.zeros((grid_resolution, grid_resolution, grid_resolution))
        
        for i, val1 in enumerate(grid_points[feature1]):
            for j, val2 in enumerate(grid_points[feature2]):
                for k, val3 in enumerate(grid_points[feature3]):
                    X_temp = X.copy()
                    X_temp[feature1] = val1
                    X_temp[feature2] = val2
                    X_temp[feature3] = val3
                    joint_pd[i, j, k] = model.predict(X_temp).mean()
        
        # Calculate interaction effect
        interaction_effect = joint_pd - sum_individual_effects
        
        # Variance of the interaction effect
        var_interaction = np.var(interaction_effect)
        
        # Variance of the joint partial dependence
        var_joint = np.var(joint_pd)
        
        # Calculate H-statistic (normalized measure of interaction strength)
        h_stat = var_interaction / (var_joint + 1e-8)  # Add small constant to avoid division by zero
        
        return max(0, min(h_stat, 1.0))  # Ensure it's between 0 and 1
    except Exception as e:
        print(f"Error calculating 3-way H-statistic for ({feature1}, {feature2}, {feature3}): {e}")
        return np.nan

# --- Added: Custom Interaction PDP Function ---
def create_interaction_pdp(surrogate_model, X_train, features, feature_names=None, grid_resolution=20):
    """Create detailed partial dependence interaction plots"""
    if feature_names is None:
        feature_names = features

    # Calculate partial dependence manually
    try:
        # Create feature grid
        feature_values = []
        for feature in features:
            unique_vals = np.unique(X_train[feature])
            if len(unique_vals) > grid_resolution:
                feature_vals = np.linspace(
                    np.min(unique_vals),
                    np.max(unique_vals),
                    grid_resolution
                )
            else:
                feature_vals = unique_vals
            feature_values.append(feature_vals)

        # Create meshgrid
        x_values, y_values = np.meshgrid(feature_values[0], feature_values[1])
        grid_points = np.column_stack([x_values.ravel(), y_values.ravel()])

        # Calculate PDP values
        z_values = np.zeros(len(grid_points))
        # Use predict on samples for efficiency
        X_temp_base = X_train.copy()
        for i, point in enumerate(grid_points):
            X_temp = X_temp_base.copy()
            X_temp[features[0]] = point[0]
            X_temp[features[1]] = point[1]
            z_values[i] = np.mean(surrogate_model.predict(X_temp))

        z_values = z_values.reshape(x_values.shape)

    except Exception as e:
        print(f"Error in PDP calculation: {e}")
        return None, None

    # Create 2x2 grid of visualizations using Matplotlib
    fig, axs = plt.subplots(2, 2, figsize=(18, 16))
    # Title will be set later after calculating H-stat

    # 1. Contour plot
    contour = axs[0, 0].contourf(
        x_values, y_values, z_values,
        cmap='viridis', levels=15
    )
    axs[0, 0].set_xlabel(feature_names[0])
    axs[0, 0].set_ylabel(feature_names[1])
    axs[0, 0].set_title('PDP Interaction Contour')
    fig.colorbar(contour, ax=axs[0, 0], label='Predicted Value')

    # 2. 3D Surface plot
    ax_3d = fig.add_subplot(2, 2, 2, projection='3d')
    surface = ax_3d.plot_surface(
        x_values, y_values, z_values,
        cmap='viridis',
        edgecolor='none',
        alpha=0.8
    )
    ax_3d.set_xlabel(feature_names[0])
    ax_3d.set_ylabel(feature_names[1])
    ax_3d.set_zlabel('Predicted Value')
    ax_3d.set_title('PDP Interaction 3D Surface')
    fig.colorbar(surface, ax=ax_3d, shrink=0.5, label='Predicted Value')

    # 3. Heatmap with annotations
    sns.heatmap(
        z_values,
        ax=axs[1, 0],
        cmap='viridis',
        annot=True,
        fmt='.1f',
        cbar=False, # Avoid duplicate colorbar
        xticklabels=np.round(feature_values[0], 2),
        yticklabels=np.round(feature_values[1], 2)
    )
    axs[1, 0].set_xlabel(feature_names[0])
    axs[1, 0].set_ylabel(feature_names[1])
    axs[1, 0].set_title('PDP Interaction Heatmap')
    axs[1, 0].tick_params(axis='x', rotation=45)
    axs[1, 0].tick_params(axis='y', rotation=0)

    # 4. Overlay with actual data points
    contour_overlay = axs[1, 1].contourf(
        x_values, y_values, z_values,
        cmap='viridis',
        levels=15,
        alpha=0.7
    )
    scatter = axs[1, 1].scatter(
        X_train[features[0]],
        X_train[features[1]],
        c='white',
        edgecolor='black',
        alpha=0.6,
        s=30
    )
    axs[1, 1].set_xlabel(feature_names[0])
    axs[1, 1].set_ylabel(feature_names[1])
    axs[1, 1].set_title('PDP with Data Distribution')
    fig.colorbar(contour_overlay, ax=axs[1, 1], label='Predicted Value')

    # Calculate and display interaction strength (Simplified H-statistic approximation)
    interaction_strength = np.nan # Default
    # --- FIX: Wrap H-stat calculation in try-except --- 
    try:
        pdp1_values = np.zeros(len(feature_values[0]))
        pdp2_values = np.zeros(len(feature_values[1]))

        X_temp_base_h = X_train.copy()
        for i, val in enumerate(feature_values[0]):
            X_temp = X_temp_base_h.copy()
            X_temp[features[0]] = val
            pdp1_values[i] = np.mean(surrogate_model.predict(X_temp))

        for i, val in enumerate(feature_values[1]):
            X_temp = X_temp_base_h.copy()
            X_temp[features[1]] = val
            pdp2_values[i] = np.mean(surrogate_model.predict(X_temp))

        # Simplified H-stat calculation
        pdp1_mean = np.mean(pdp1_values)
        pdp2_mean = np.mean(pdp2_values)
        z_mean = np.mean(z_values)
        # Ensure pdp1_values and pdp2_values are broadcastable to z_values shape
        pdp1_broadcast = np.outer(pdp1_values, np.ones(len(pdp2_values)))
        pdp2_broadcast = np.outer(np.ones(len(pdp1_values)), pdp2_values)
        numerator = np.mean((z_values - pdp1_broadcast - pdp2_broadcast + pdp1_mean + pdp2_mean - z_mean)**2)
        denominator = np.mean((z_values - z_mean)**2)
        interaction_strength = np.sqrt(max(0, numerator / denominator)) if denominator > 1e-8 else 0

    except Exception as e:
        print(f"Error calculating interaction strength for ({features[0]}, {features[1]}): {e}")
        interaction_strength = np.nan # Ensure it's NaN on error
    # --- END FIX --- 

    # Set the main title including the H-statistic
    if not np.isnan(interaction_strength):
        title_h_stat = f" (H-stat Approx: {interaction_strength:.4f})"
    else:
        title_h_stat = " (H-stat Error)"
    fig.suptitle(f'PDP Interaction Analysis: {feature_names[0]} vs {feature_names[1]}{title_h_stat}', fontsize=16)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout
    # plt.subplots_adjust(wspace=0.3, hspace=0.4) # Fine-tune spacing if needed

    return fig, interaction_strength
# --- End Helper Function ---

# --- Added: Granger Causality Helper ---
def perform_granger_causality(data_features, data_target, feature_names, max_lag, test='ssr_ftest'): # Removed target_name
    """Performs Granger causality tests for each feature predicting the target."""
    p_values = pd.Series(index=feature_names, dtype=float)
    target_name = data_target.name # Get target name from the Series
    print(f"Performing Granger Causality tests (Feature -> {target_name}) up to lag {max_lag}...")
    
    # Combine for testing
    df_test = pd.concat([data_target, data_features], axis=1).dropna()

    for feature in feature_names:
        try:
            # Check for constant series
            if df_test[feature].nunique() == 1 or df_test[target_name].nunique() == 1:
                p_values.loc[feature] = np.nan
                continue

            # Test if feature Granger-causes target
            test_result = grangercausalitytests(df_test[[target_name, feature]], maxlag=max_lag, verbose=False)
            
            # Get p-value for the specified test at the max lag
            if max_lag in test_result and test in test_result[max_lag][0]:
                 p_val = test_result[max_lag][0][test][1]
                 p_values.loc[feature] = p_val
            else:
                 print(f"Warning: Could not find result for lag {max_lag} and test '{test}' for ({feature} -> {target_name})")
                 p_values.loc[feature] = np.nan
        except Exception as e:
            # print(f"Error running Granger test for ({feature} -> {target_name}): {e}")
            p_values.loc[feature] = np.nan # Error occurred

    print("Granger Causality tests complete.")
    return p_values

# ... (End of Helper Functions section) ...

# --- 1. Load Data ---
print("\n--- 1. Loading Data ---")
try:
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded data: {df.shape[0]} rows, {df.shape[1]} columns")
    print("Initial columns:", df.columns.tolist())
except FileNotFoundError:
    print(f"Error: Data file not found at {DATA_PATH}")
    exit()
except Exception as e:
    print(f"Error loading data: {e}")
    exit()

# --- 2. Prepare Data ---
print("\n--- 2. Preparing Data ---")
# a. Time Index
try:
    if DATE_COLUMN not in df.columns:
        raise ValueError(f"Date column '{DATE_COLUMN}' not found in the dataset.")
    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN])
    df = df.set_index(DATE_COLUMN)
    df = df.sort_index()
    print(f"Set '{DATE_COLUMN}' as time index.")
except KeyError:
     print(f"Error: Date column '{DATE_COLUMN}' not found.")
     exit()
except Exception as e:
     print(f"Error processing date column: {e}")
     exit()

# b. Optional Filtering
if GROUPING_COLUMN and FILTER_VALUE:
    if GROUPING_COLUMN in df.columns:
        print(f"Filtering data by '{GROUPING_COLUMN}' == '{FILTER_VALUE}'")
        df_filtered = df[df[GROUPING_COLUMN] == FILTER_VALUE].copy()
        if df_filtered.empty:
            print(f"Warning: Filtering resulted in an empty DataFrame. Proceeding with unfiltered data.")
        else:
            df = df_filtered
            print(f"Data filtered: {df.shape[0]} rows remaining.")
            if GROUPING_COLUMN in df.columns: # Check if column still exists before dropping
                 df = df.drop(columns=[GROUPING_COLUMN])
                 print(f"Dropped constant column '{GROUPING_COLUMN}' after filtering.")
    else:
        print(f"Warning: Grouping column '{GROUPING_COLUMN}' not found. Skipping filtering.")

# Drop rows with NaN in target variable BEFORE splitting
df.dropna(subset=[TARGET_VARIABLE], inplace=True)
print(f"Dropped rows with NaN target: {df.shape[0]} rows remaining.")

# Identify features (numeric only for most models here)
features = df.select_dtypes(include=np.number).columns.tolist()
if TARGET_VARIABLE in features:
    features.remove(TARGET_VARIABLE)
else:
    print(f"Warning: Target variable '{TARGET_VARIABLE}' not found in numeric columns.")
    if TARGET_VARIABLE not in df.columns:
        print(f"Error: Target variable '{TARGET_VARIABLE}' not found in the DataFrame.")
        exit()

print(f"Identified Features ({len(features)}): {features}")
print(f"Target Variable: {TARGET_VARIABLE}")

# c. Time Series Split
train_df, test_df = time_series_split_by_month(df, DATE_COLUMN, TEST_MONTHS)

# Define X_train, X_test, y_train, y_test first
X_train, y_train = train_df[features], train_df[TARGET_VARIABLE]
X_test, y_test = test_df[features], test_df[TARGET_VARIABLE]

# Handle potential NaNs introduced by feature selection or lagging (impute or drop)
# Simple mean imputation for this example - applied AFTER splitting
for col in features:
    if X_train[col].isnull().any():
        mean_val = X_train[col].mean()
        X_train.loc[:, col] = X_train[col].fillna(mean_val)
        # Use train mean for test set and handle potential non-existent columns in X_test if features differ
        if col in X_test.columns:
             X_test.loc[:, col] = X_test[col].fillna(mean_val)
        print(f"Imputed NaNs in feature: {col}")

print(f"Train shapes: X={X_train.shape}, y={y_train.shape}")
print(f"Test shapes:  X={X_test.shape}, y={y_test.shape}")

# Add Dataset Overview to Report
report_generator.add_dataset_overview(
    df_shape=df.shape,
    features_list=features,
    date_range=(df.index.min(), df.index.max())
)

# --- 4. Surrogate Model Training ---
print("\n--- 4. Surrogate Model Training ---")
print(f"Training {SURROGATE_MODEL_TYPE} model...")
if SURROGATE_MODEL_TYPE == 'random_forest':
    surrogate_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, max_depth=10, min_samples_leaf=5)
elif SURROGATE_MODEL_TYPE == 'xgboost':
    surrogate_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42, n_jobs=-1, max_depth=5, learning_rate=0.1)
else:
    raise ValueError("Invalid SURROGATE_MODEL_TYPE. Choose 'xgboost' or 'random_forest'.")
surrogate_model.fit(X_train, y_train)
y_pred_train_surrogate = surrogate_model.predict(X_train)
y_pred_test_surrogate = surrogate_model.predict(X_test)
train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train_surrogate))
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test_surrogate))
train_r2 = r2_score(y_train, y_pred_train_surrogate)
test_r2 = r2_score(y_test, y_pred_test_surrogate)
print(f"Surrogate Model Performance:")
print(f"  Train RMSE: {train_rmse:.4f}, R2: {train_r2:.4f}")
print(f"  Test RMSE:  {test_rmse:.4f}, R2: {test_r2:.4f}")

# Add Model Performance to Report
report_generator.add_surrogate_model_performance(
    model_type=SURROGATE_MODEL_TYPE,
    train_rmse=train_rmse,
    train_r2=train_r2,
    test_rmse=test_rmse,
    test_r2=test_r2
)

# --- 5. Surrogate Model Interpretation - Rules & Thresholds ---
print("\n--- 5. Surrogate Model Interpretation - Rules & Thresholds ---")

# a. Decision Tree Extraction & Visualization
rule_conditions = Counter()
cond_df = pd.DataFrame() # Initialize dataframe
if SURROGATE_MODEL_TYPE == 'random_forest':
    print("Visualizing sample Decision Trees & Extracting Rule Conditions...")
    # Visualize first tree
    try:
        plt.figure(figsize=(20, 10))
        plot_tree(surrogate_model.estimators_[0],
                    feature_names=features,
                    filled=True, rounded=True, precision=2,
                    max_depth=3, fontsize=10)
        plt.title(f'Decision Tree 0 from Random Forest (Max Depth 3)')
        tree_plot_fig = plt.gcf() # Get current figure
        save_plot(tree_plot_fig, f"surrogate_tree_0", OUTPUT_DIR)
    except Exception as e:
        print(f"Could not visualize tree 0: {e}")

    # Extract rules and conditions from all trees
    rule_conditions, example_rules_text = get_tree_rules_and_conditions(surrogate_model, features)

    if example_rules_text:
        print("\nExample Tree Rules (Tree 0):")
        print(example_rules_text[0][:1000]) # Print first 1000 chars
        rules_filename = os.path.join(OUTPUT_DIR, "surrogate_tree_0_rules.txt")
        with open(rules_filename, "w") as f:
            f.write(example_rules_text[0])
        print(f"Saved full rules for tree 0: {rules_filename}")

    # Plotting and filtering moved to end of Section 6 after importance is calculated
    # if rule_conditions:
    #    ... (Old filtering/plotting code removed from here) ...

elif SURROGATE_MODEL_TYPE == 'xgboost':
    print("Visualizing XGBoost Tree (requires graphviz)...")
    try:
        import graphviz
        for i in range(min(N_TREES_TO_VISUALIZE, surrogate_model.n_estimators)):
             booster = surrogate_model.get_booster()
             dot_data = xgb.to_graphviz(booster, num_trees=i)
             graph_filename = os.path.join(OUTPUT_DIR, f"surrogate_xgb_tree_{i}")
             dot_data.render(graph_filename, view=False, format='png')
             print(f"Saved XGBoost tree plot: {graph_filename}.png")
    except ImportError:
        print("Graphviz library not found. Skipping XGBoost tree visualization.")
        print("Install it via pip install graphviz and ensure the graphviz binaries are in your system PATH.")
    except Exception as e:
        print(f"Could not plot XGBoost tree {i}: {e}")
    print("Rule/condition extraction currently implemented only for RandomForest.")

# b. Threshold Analysis (from Condition Analysis)
if SURROGATE_MODEL_TYPE == 'random_forest' and rule_conditions:
    print("\nAnalyzing Feature Thresholds (Unique per feature from Condition Analysis)...")
    feature_thresholds = {}
    for (feature, condition_str), count in rule_conditions.items():
        try:
            # Extract threshold robustly
            parts = condition_str.replace('<=', ' ').replace('>', ' ').split()
            if len(parts) > 0:
                threshold = float(parts[-1])
                if feature not in feature_thresholds:
                    feature_thresholds[feature] = set()
                feature_thresholds[feature].add(threshold)
        except:
            continue # Skip if parsing fails

    threshold_counts = {f: len(t) for f, t in feature_thresholds.items() if t}
    if threshold_counts:
        threshold_df = pd.DataFrame.from_dict(threshold_counts, orient='index', columns=['Count'])
        threshold_df = threshold_df.sort_values('Count', ascending=False)
        fig_thresh = px.bar(threshold_df, x=threshold_df.index, y='Count',
                            title='Number of Unique Decision Thresholds per Feature (Random Forest)',
                            labels={'x': 'Feature', 'Count': 'Number of Unique Thresholds'})
        save_plot(fig_thresh, "surrogate_threshold_counts", OUTPUT_DIR)
        print("Unique Threshold Counts per Feature:")
        print(threshold_df)
    else:
        print("No thresholds extracted for counting.")

# --- 6. Feature Importance Analysis ---
print("\n--- 6. Feature Importance Analysis ---")
# a. Permutation Feature Importance (on Training Data)
print("Calculating Permutation Importance (Train Set)...")
perm_df = None
try:
    # Calculate on TRAINING data
    perm_importance = permutation_importance(surrogate_model, X_train, y_train, n_repeats=10, random_state=42, n_jobs=-1)
    perm_sorted_idx = perm_importance.importances_mean.argsort()
    perm_df = pd.DataFrame({
        'feature': X_train.columns[perm_sorted_idx], # Use X_train columns
        'importance_mean': perm_importance.importances_mean[perm_sorted_idx],
        'importance_std': perm_importance.importances_std[perm_sorted_idx]
    }).sort_values('importance_mean', ascending=False)

    fig_perm = px.bar(perm_df, x='importance_mean', y='feature', orientation='h',
                      error_x='importance_std', title='Permutation Feature Importance (Train Set)',
                      labels={'importance_mean': 'Importance Mean', 'feature': 'Feature'})
    save_plot(fig_perm, "featimp_permutation_train", OUTPUT_DIR) # Update filename
    print("Permutation Importance (Train Set - Top 10):")
    print(perm_df.head(N_TOP_FEATURES))
except Exception as e:
    print(f"Error calculating Permutation Importance: {e}")

# b. LOFO Feature Importance
print("\nCalculating LOFO Importance...")
importance_df = None # Initialize
try:
    # LOFO needs a defined CV scheme. Using TimeSeriesSplit for temporal data.
    cv = TimeSeriesSplit(n_splits=5) # Adjust n_splits as needed
    lofo_dataset = Dataset(df=pd.concat([X_train, y_train], axis=1), target=TARGET_VARIABLE, features=features)

    # Note: LOFO may fail with certain model/sklearn versions due to internal checks (e.g., __sklearn_tags__)
    print("Running LOFOImportance...")
    lofo_imp = LOFOImportance(lofo_dataset, model=surrogate_model, cv=cv, scoring='neg_root_mean_squared_error')
    importance_df = lofo_imp.get_importance()
    print(f"LOFO Raw Results:\n{importance_df}")
    print(f"LOFO Columns: {importance_df.columns.tolist()}")

    # Try to find the correct importance column
    imp_col_name = None
    if 'importance' in importance_df.columns:
        imp_col_name = 'importance'
    elif 'val_imp_mean' in importance_df.columns:
        imp_col_name = 'val_imp_mean' # Common in some versions
    elif 'importance_mean' in importance_df.columns:
        imp_col_name = 'importance_mean'
    # Add other potential names if needed

    if imp_col_name:
        print(f"Using LOFO importance column: '{imp_col_name}'")
        importance_df['importance_mean'] = importance_df[imp_col_name]
        importance_df = importance_df.sort_values('importance_mean', ascending=False)

        # Fix LOFO Plotting Call
        try:
            print("Generating LOFO plot...")
            # Call plot_importance without ax
            plot_importance(importance_df)
            fig_lofo = plt.gcf()
            fig_lofo.suptitle('LOFO Feature Importance')
            save_plot(fig_lofo, "featimp_lofo", OUTPUT_DIR)
            print("LOFO Importance (Top 10):")
            print(importance_df.head(N_TOP_FEATURES))
        except Exception as plot_e:
             print(f"Error generating LOFO plot: {plot_e}")
    else:
        print("Could not identify the correct importance column in LOFO results. Skipping plots.")
        importance_df = None # Reset if processing failed

except ImportError:
    print("LOFO library not found (pip install lofo-importance). Skipping LOFO.")
    importance_df = None
except Exception as e:
    print(f"Error during LOFO calculation/processing: {e}")
    importance_df = None

# c/e. SHAP Feature Importance & Summary Plot (on Training Data)
print("\nCalculating SHAP Importance (Train Set)...")
shap_values = None
shap_df = None
shap_interaction_values = None # Initialize
shap_interaction_df = None # Initialize
try:
    explainer = shap.TreeExplainer(surrogate_model, X_train)
    shap_values = explainer(X_train)
    shap_mean_abs = np.abs(shap_values.values).mean(axis=0)
    shap_df = pd.DataFrame({'feature': X_train.columns, 'shap_importance': shap_mean_abs}) # Use X_train cols
    shap_df = shap_df.sort_values('shap_importance', ascending=False)
    fig_shap_bar = px.bar(shap_df, x='shap_importance', y='feature', orientation='h',
                           title='SHAP Feature Importance (Mean Abs Value - Train Set)',
                           labels={'shap_importance': 'Mean |SHAP Value|', 'feature': 'Feature'})
    save_plot(fig_shap_bar, "featimp_shap_bar_train", OUTPUT_DIR) # Update filename
    print("SHAP Importance (Train Set - Top 10):")
    print(shap_df.head(N_TOP_FEATURES))
    print("Generating SHAP Summary Plot (Train Set)...")
    fig_shap_summary, ax_shap_summary = plt.subplots()
    shap.summary_plot(shap_values, X_train, plot_type="dot", show=False, sort=True)
    plt.title('SHAP Summary Plot (Beeswarm - Train Set)')
    save_plot(fig_shap_summary, "featimp_shap_summary_train", OUTPUT_DIR) # Update filename

    # Calculate SHAP Interaction Values
    print("Calculating SHAP Interaction Values (can be slow)...")
    shap_interaction_values = explainer.shap_interaction_values(X_train)
    mean_abs_shap_inter = np.abs(shap_interaction_values).mean(0)
    shap_interaction_df = pd.DataFrame(mean_abs_shap_inter, index=features, columns=features)

    # Add SHAP Interaction summary to report
    report_generator.add_shap_interaction_summary(shap_interaction_df)

except Exception as e:
    print(f"Error calculating or plotting SHAP values/interactions: {e}")

# d. MDI Feature Importance (if applicable)
mdi_df = None
if hasattr(surrogate_model, 'feature_importances_'):
    print("\nCalculating MDI Importance...")
    try:
        mdi_importances = surrogate_model.feature_importances_
        mdi_df = pd.DataFrame({'feature': features, 'mdi_importance': mdi_importances})
        mdi_df = mdi_df.sort_values('mdi_importance', ascending=False)
        fig_mdi = px.bar(mdi_df, x='mdi_importance', y='feature', orientation='h',
                         title='Mean Decrease in Impurity (MDI) Importance',
                         labels={'mdi_importance': 'Importance', 'feature': 'Feature'})
        save_plot(fig_mdi, "featimp_mdi", OUTPUT_DIR)
        print("MDI Importance (Top 10):")
        print(mdi_df.head(N_TOP_FEATURES))
    except Exception as e:
         print(f"Error calculating MDI importance: {e}")
         mdi_df = None # Ensure mdi_df is None if error occurs
else:
    print("\nMDI importance not available for this model type.")

# --- Added: Analyze Importance of Top Conditions ---
print("\n--- 6b. Analyzing Importance of Top Conditions ---")
condition_importance_df = pd.DataFrame()
if SURROGATE_MODEL_TYPE == 'random_forest' and rule_conditions:
    # 1. Identify Top Frequent Conditions
    N_TOP_CONDITIONS = 20 # How many top conditions to analyze
    top_conditions = rule_conditions.most_common(N_TOP_CONDITIONS)
    print(f"Identified Top {len(top_conditions)} most frequent conditions.")

    # 2. Create Binary Features from Conditions
    X_train_cond_feats = pd.DataFrame(index=X_train.index)
    X_test_cond_feats = pd.DataFrame(index=X_test.index)
    condition_feature_map = {}
    valid_conditions_created = 0

    for condition_tuple, freq in top_conditions:
        feature, condition_str = condition_tuple
        try:
            op = None
            threshold = None
            if '<=' in condition_str:
                parts = condition_str.split('<=')
                if len(parts) == 2: op, threshold = 'le', float(parts[1].strip())
            elif '>' in condition_str:
                parts = condition_str.split('>')
                if len(parts) == 2: op, threshold = 'gt', float(parts[1].strip())

            if op and threshold is not None:
                thresh_str = f"{abs(threshold):.2f}".replace('.','p')
                neg_prefix = 'neg' if threshold < 0 else ''
                new_feat_name = f"cond_{feature}_{op}_{neg_prefix}{thresh_str}"

                if new_feat_name in X_train_cond_feats.columns:
                    continue

                if op == 'le':
                     X_train_cond_feats[new_feat_name] = (X_train[feature] <= threshold).astype(int)
                     X_test_cond_feats[new_feat_name] = (X_test[feature] <= threshold).astype(int)
                elif op == 'gt':
                     X_train_cond_feats[new_feat_name] = (X_train[feature] > threshold).astype(int)
                     X_test_cond_feats[new_feat_name] = (X_test[feature] > threshold).astype(int)

                condition_feature_map[new_feat_name] = condition_tuple
                valid_conditions_created += 1
            else:
                print(f"Skipping condition due to parsing issue: {condition_tuple}")
        except Exception as e:
            print(f"Error creating binary feature for condition {condition_tuple}: {e}")

    print(f"Created {valid_conditions_created} binary features based on top conditions.")

    if not X_train_cond_feats.empty:
        # 3. Train a Secondary Model
        print("Training secondary model with original + condition features...")
        X_train_plus_cond = pd.concat([X_train, X_train_cond_feats], axis=1)
        # Use a simpler model, e.g., fewer estimators
        secondary_model = RandomForestRegressor(n_estimators=50, random_state=123, n_jobs=-1, max_depth=8, min_samples_leaf=5)
        secondary_model.fit(X_train_plus_cond, y_train)

        # 4. Calculate Importance of Condition Features (using Permutation Importance on Train set)
        print("Calculating permutation importance for condition features (Train Set)...")
        try:
            cond_perm_importance = permutation_importance(
                secondary_model, X_train_plus_cond, y_train, n_repeats=5, random_state=123, n_jobs=-1 # Fewer repeats for speed
            )
            cond_importances = pd.Series(cond_perm_importance.importances_mean, index=X_train_plus_cond.columns)

            # 5. Analyze & Plot
            # Filter for only the condition features
            condition_feature_names = list(X_train_cond_feats.columns)
            condition_importance_scores = cond_importances[condition_feature_names]

            condition_importance_df = pd.DataFrame({
                'Condition_Feature': condition_feature_names,
                'Importance': condition_importance_scores
            }).sort_values('Importance', ascending=False)

            # Map back to original condition tuple for clarity
            condition_importance_df['Original_Condition'] = condition_importance_df['Condition_Feature'].map(condition_feature_map)

            if not condition_importance_df.empty:
                fig_cond_imp = px.bar(condition_importance_df.head(N_TOP_CONDITIONS).sort_values('Importance', ascending=True),
                                      x='Importance', y='Condition_Feature', orientation='h',
                                      title=f'Top {N_TOP_CONDITIONS} Condition Feature Importances (Permutation on Train)',
                                      hover_data=['Original_Condition'])
                save_plot(fig_cond_imp, "featimp_condition_importance", OUTPUT_DIR)
                print("Top 10 Condition Feature Importances:")
                print(condition_importance_df[['Original_Condition', 'Importance']].head(10))
            else:
                print("No importance scores calculated for condition features.")
        except Exception as e:
            print(f"Error during condition importance calculation: {e}")

        # --- Added: Calculate MDI for Condition Features ---
        print("Calculating MDI importance for condition features...")
        condition_mdi_df = pd.DataFrame() # Initialize
        if hasattr(secondary_model, 'feature_importances_'):
            try:
                mdi_importances_secondary = secondary_model.feature_importances_
                cond_mdi_scores = pd.Series(mdi_importances_secondary, index=X_train_plus_cond.columns)
                condition_mdi_filtered = cond_mdi_scores[condition_feature_names] # Filter for condition features

                condition_mdi_df = pd.DataFrame({
                    'Condition_Feature': condition_feature_names,
                    'MDI_Importance': condition_mdi_filtered
                }).sort_values('MDI_Importance', ascending=False)
                condition_mdi_df['Original_Condition'] = condition_mdi_df['Condition_Feature'].map(condition_feature_map)

                if not condition_mdi_df.empty:
                    fig_cond_mdi = px.bar(condition_mdi_df.head(N_TOP_CONDITIONS).sort_values('MDI_Importance', ascending=True),
                                          x='MDI_Importance', y='Condition_Feature', orientation='h',
                                          title=f'Top {N_TOP_CONDITIONS} Condition Feature Importances (MDI on Secondary Model)',
                                          hover_data=['Original_Condition'])
                    save_plot(fig_cond_mdi, "featimp_condition_importance_mdi", OUTPUT_DIR)
                    print("Top 10 Condition Feature Importances (MDI):")
                    print(condition_mdi_df[['Original_Condition', 'MDI_Importance']].head(10))
                else:
                     print("No MDI importance scores calculated for condition features.")

            except Exception as e:
                print(f"Error calculating MDI importance for conditions: {e}")
        else:
            print("Secondary model does not have feature_importances_ attribute for MDI.")
        # --- End MDI Calculation ---

        # --- Added: Calculate SHAP for Condition Features ---
        print("Calculating SHAP importance for condition features (can be slow)...")
        condition_shap_df = pd.DataFrame() # Initialize
        try:
            # --- FIX: Add check_additivity=False --- 
            explainer_secondary = shap.TreeExplainer(secondary_model, X_train_plus_cond, check_additivity=False)
            # --- END FIX --- 
            shap_values_secondary = explainer_secondary(X_train_plus_cond)
            shap_mean_abs_secondary = np.abs(shap_values_secondary.values).mean(axis=0)

            cond_shap_scores = pd.Series(shap_mean_abs_secondary, index=X_train_plus_cond.columns)
            condition_shap_filtered = cond_shap_scores[condition_feature_names] # Filter for condition features

            condition_shap_df = pd.DataFrame({
                'Condition_Feature': condition_feature_names,
                'SHAP_Importance': condition_shap_filtered
            }).sort_values('SHAP_Importance', ascending=False)
            condition_shap_df['Original_Condition'] = condition_shap_df['Condition_Feature'].map(condition_feature_map)

            if not condition_shap_df.empty:
                 fig_cond_shap = px.bar(condition_shap_df.head(N_TOP_CONDITIONS).sort_values('SHAP_Importance', ascending=True),
                                        x='SHAP_Importance', y='Condition_Feature', orientation='h',
                                        title=f'Top {N_TOP_CONDITIONS} Condition Feature Importances (Mean Abs SHAP on Secondary Model)',
                                        hover_data=['Original_Condition'])
                 save_plot(fig_cond_shap, "featimp_condition_importance_shap", OUTPUT_DIR)
                 print("Top 10 Condition Feature Importances (SHAP):")
                 print(condition_shap_df[['Original_Condition', 'SHAP_Importance']].head(10))
            else:
                 print("No SHAP importance scores calculated for condition features.")

        except Exception as e:
            print(f"Error calculating SHAP importance for conditions: {e}")
        # --- End SHAP Calculation ---

        # --- Added: Linear Models using ONLY Condition Features ---
        print("\n--- 6b.1 Linear Models using Condition Features ---")
        try:
            # Prepare data (using only condition features)
            X_train_cond_only = X_train_cond_feats
            X_test_cond_only = X_test_cond_feats

            # 1. Standard Linear Regression
            lr_cond = LinearRegression()
            lr_cond.fit(X_train_cond_only, y_train)
            y_pred_test_lr_cond = lr_cond.predict(X_test_cond_only)
            rmse_lr_cond = np.sqrt(mean_squared_error(y_test, y_pred_test_lr_cond))
            r2_lr_cond = r2_score(y_test, y_pred_test_lr_cond)
            print(f"  Linear Regression (Conditions Only): Test RMSE={rmse_lr_cond:.4f}, R2={r2_lr_cond:.4f}")

            # 2. Ridge Regression (with Cross-Validation for alpha)
            ridge_alphas = np.logspace(-4, 2, 100) # Alpha range
            ridge_cond = RidgeCV(alphas=ridge_alphas, store_cv_values=True)
            ridge_cond.fit(X_train_cond_only, y_train)
            y_pred_test_ridge_cond = ridge_cond.predict(X_test_cond_only)
            rmse_ridge_cond = np.sqrt(mean_squared_error(y_test, y_pred_test_ridge_cond))
            r2_ridge_cond = r2_score(y_test, y_pred_test_ridge_cond)
            print(f"  RidgeCV Regression (Conditions Only):  Test RMSE={rmse_ridge_cond:.4f}, R2={r2_ridge_cond:.4f}, Best Alpha={ridge_cond.alpha_:.4f}")

            # 3. LASSO Regression (with Cross-Validation for alpha)
            lasso_cond = LassoCV(cv=5, random_state=42, max_iter=10000) # 5-fold CV
            lasso_cond.fit(X_train_cond_only, y_train)
            y_pred_test_lasso_cond = lasso_cond.predict(X_test_cond_only)
            rmse_lasso_cond = np.sqrt(mean_squared_error(y_test, y_pred_test_lasso_cond))
            r2_lasso_cond = r2_score(y_test, y_pred_test_lasso_cond)
            print(f"  LassoCV Regression (Conditions Only):  Test RMSE={rmse_lasso_cond:.4f}, R2={r2_lasso_cond:.4f}, Best Alpha={lasso_cond.alpha_:.4f}")

            # 4. Analyze Coefficients
            ridge_coefs = pd.DataFrame({
                'Condition_Feature': condition_feature_names,
                'Ridge_Coefficient': ridge_cond.coef_
            })
            ridge_coefs['Original_Condition'] = ridge_coefs['Condition_Feature'].map(condition_feature_map)
            ridge_coefs = ridge_coefs.iloc[np.abs(ridge_coefs['Ridge_Coefficient']).argsort()[::-1]] # Sort by abs value

            lasso_coefs = pd.DataFrame({
                'Condition_Feature': condition_feature_names,
                'Lasso_Coefficient': lasso_cond.coef_
            })
            lasso_coefs['Original_Condition'] = lasso_coefs['Condition_Feature'].map(condition_feature_map)
            lasso_coefs['Is_Zero'] = np.isclose(lasso_coefs['Lasso_Coefficient'], 0)
            lasso_coefs = lasso_coefs.iloc[np.abs(lasso_coefs['Lasso_Coefficient']).argsort()[::-1]] # Sort by abs value

            print("\nTop Ridge Coefficients (Conditions Only):")
            print(ridge_coefs[['Original_Condition', 'Ridge_Coefficient']].head(10))

            print("\nTop Lasso Coefficients (Conditions Only):")
            print(lasso_coefs[['Original_Condition', 'Lasso_Coefficient', 'Is_Zero']].head(15))
            print(f"Number of Lasso coefficients shrunk to zero: {lasso_coefs['Is_Zero'].sum()}")

            # Store results for summary file
            condition_linear_results = {
                'linear': {'rmse': rmse_lr_cond, 'r2': r2_lr_cond},
                'ridge': {'rmse': rmse_ridge_cond, 'r2': r2_ridge_cond, 'alpha': ridge_cond.alpha_,
                           'coefficients': ridge_coefs.to_dict('records')},
                'lasso': {'rmse': rmse_lasso_cond, 'r2': r2_lasso_cond, 'alpha': lasso_cond.alpha_,
                           'coefficients': lasso_coefs.to_dict('records')}
            }

        except Exception as e:
            print(f"Error during Linear Model training on condition features: {e}")
            condition_linear_results = None # Ensure it exists but is None
        # --- End Linear Models on Conditions ---

    else:
        print("No valid binary condition features created, skipping importance analysis and linear models.")
        # Ensure these are defined even if skipped, for the summary file logic
        condition_importance_df = pd.DataFrame()
        condition_mdi_df = pd.DataFrame()
        condition_shap_df = pd.DataFrame()
        condition_linear_results = None

# --- Calculate Combined Importance (Moved from Section 10) ---
print("\nCalculating combined importance rankings...")
all_imp = {}
if perm_df is not None: all_imp['Permutation_Train'] = perm_df.set_index('feature')['importance_mean']
if shap_df is not None: all_imp['SHAP_Train'] = shap_df.set_index('feature')['shap_importance']
if mdi_df is not None: all_imp['MDI'] = mdi_df.set_index('feature')['mdi_importance']

combined_imp = pd.DataFrame() # Initialize
if all_imp:
    combined_imp = pd.DataFrame(all_imp)
    rank_cols = []
    for col in combined_imp.columns:
         rank_col_name = f'{col}_rank'
         if (combined_imp[col] < 0).any() and col == 'Permutation_Train':
              combined_imp[rank_col_name] = combined_imp[col].abs().rank(ascending=False)
         else:
              combined_imp[rank_col_name] = combined_imp[col].rank(ascending=False)
         rank_cols.append(rank_col_name)

    if rank_cols:
        combined_imp['mean_rank'] = combined_imp[rank_cols].mean(axis=1)
        combined_imp = combined_imp.sort_values('mean_rank')
        print("Top Features (Mean Rank - based on Train Set calculations):")
        print(combined_imp[['mean_rank']].head(N_TOP_FEATURES))
    else:
         print("Could not calculate combined ranks.")
else:
     print("No importance scores available to combine.")
# --- End Combined Importance Calculation ---

# --- 7. Feature Interaction Analysis ---
print("\n--- 7. Feature Interaction Analysis ---")

# a. Pairwise Friedman H-statistics
print("Calculating pairwise Friedman H-statistics (approximation)...")
print("Note: H-statistic values are approximations based on PDP variances and may differ from other methods.")
h_matrix = pd.DataFrame(index=features, columns=features, dtype=float)
h_values = None # Initialize h_values
np.fill_diagonal(h_matrix.values, 1.0)
interaction_pairs = list(combinations(features, 2))

if interaction_pairs: # Only proceed if there are pairs to calculate
    X_sample_h = X_train.sample(min(100, len(X_train)), random_state=42) if len(X_train) > 100 else X_train
    h_calc_count = 0
    for i, (f1, f2) in enumerate(interaction_pairs):
        print(f"  Calculating H for ({f1}, {f2}) - {i+1}/{len(interaction_pairs)}", end='\r')
        h = friedman_h_statistic(surrogate_model, X_sample_h, f1, f2)
        if not np.isnan(h):
            h_matrix.loc[f1, f2] = h
            h_matrix.loc[f2, f1] = h # Symmetric
            h_calc_count += 1
    print(f"\nPairwise H-statistic calculation complete ({h_calc_count} calculated).")

    if not h_matrix.isnull().all().all():
        fig_h = px.imshow(h_matrix.fillna(0), text_auto=".2f", aspect="auto", # Fill NaN for plot
                      title='Pairwise Friedman H-statistic (Approximation)',
                      color_continuous_scale='Viridis', range_color=[0,1])
        save_plot(fig_h, "interaction_h_statistic_heatmap", OUTPUT_DIR)
        print("Saved H-statistic heatmap.")
        # Extract top interactions from the original matrix with NaNs potentially
        h_values = h_matrix.unstack().dropna().sort_values(ascending=False)
        h_values = h_values[h_values.index.get_level_values(0) != h_values.index.get_level_values(1)]
        if not h_values.empty:
            h_values = h_values.drop_duplicates()
            print("\nTop Pairwise Interactions (H-statistic):")
            print(h_values.head(N_TOP_FEATURES))
        else:
            print("No valid H-statistic values found after filtering.")
            h_values = None
    else:
        print("H-statistic calculation yielded no valid results.")
else:
    print("No feature pairs found for H-statistic calculation.")

# b. Three-way Friedman H-statistic
# --- MODIFIED: Calculate for top N feature triplets --- 
print("\nCalculating Three-way H-statistic (approximation, for top feature triplets)...")
h_3way_results = []
N_TOP_FEATURES_3WAY = 4 # Calculate for triplets from top 4 features

if len(features) >= 3 and not combined_imp.empty:
    # Select top N features based on combined importance rank
    top_n_features = combined_imp.head(N_TOP_FEATURES_3WAY).index.tolist()
    print(f"Calculating 3-way H for triplets from top {N_TOP_FEATURES_3WAY} features: {top_n_features}")

    triplet_combinations = list(combinations(top_n_features, 3))
    print(f"Number of triplets to calculate: {len(triplet_combinations)}")

    # Use a smaller sample for 3-way PDP (can be slow)
    X_sample_3way = X_train.sample(min(50, len(X_train)), random_state=42) if len(X_train) > 50 else X_train

    for i, triplet in enumerate(triplet_combinations):
        f1, f2, f3 = triplet
        print(f"  Calculating for triplet {i+1}/{len(triplet_combinations)}: ({f1}, {f2}, {f3})", end='\r')
        h_3way = friedman_h_3way(surrogate_model, X_sample_3way, f1, f2, f3)

        if h_3way is not None and not np.isnan(h_3way):
            h_3way_results.append({'Triplet': triplet, 'H_statistic': h_3way})
            print(f"  Triplet {i+1}: ({f1}, {f2}, {f3}) -> H ≈ {h_3way:.4f}       ") # Overwrite progress line
        else:
            print(f"  Could not calculate 3-way H-statistic proxy for ({f1}, {f2}, {f3}). Skipping.")

    print("\nThree-way H-statistic calculation complete.")

    if h_3way_results:
        h_3way_df = pd.DataFrame(h_3way_results)
        h_3way_df['Triplet_Str'] = h_3way_df['Triplet'].apply(lambda x: ' x '.join(x))
        h_3way_df = h_3way_df.sort_values('H_statistic', ascending=True)

        # Plot results
        fig_h3 = px.bar(h_3way_df,
                        x='H_statistic', y='Triplet_Str', orientation='h',
                        title=f'Approximate 3-Way H-Statistic (Top {N_TOP_FEATURES_3WAY} Feature Triplets)',
                        labels={'H_statistic': 'H-statistic (Approximation)', 'Triplet_Str': 'Feature Triplet'})
        # --- ADDED: Adjust layout for readability --- 
        fig_h3.update_layout(
            yaxis={'categoryorder':'total ascending'}, # Ensure y-axis is sorted by value
            margin=dict(l=350) # Increase left margin significantly (pixels)
        )
        # --- END ADDED --- 
        save_plot(fig_h3, "interaction_h_statistic_3way_bar", OUTPUT_DIR)
        print("Saved 3-way H-statistic bar chart.")

        # Update report
        report_generator.add_feature_interactions_summary(h_values, interaction_stability, h_3way_df)
    else:
        print("No valid 3-way H-statistics calculated.")
        report_generator.add_feature_interactions_summary(h_values, interaction_stability, None) # Update report anyway

elif len(features) < 3:
    print("Not enough features (need 3+) to calculate 3-way interactions.")
    report_generator.add_feature_interactions_summary(h_values, interaction_stability, None)
elif combined_imp.empty:
     print("Combined importance not available, cannot determine top features for 3-way interactions.")
     report_generator.add_feature_interactions_summary(h_values, interaction_stability, None)
else:
    print("Skipping 3-way H-statistic calculation due to other reasons.")
    report_generator.add_feature_interactions_summary(h_values, interaction_stability, None)

# c. Interaction Stability
print("\nCalculating Interaction Stability (Bootstrapped H-statistic, can be slow)...")
interaction_stability = {}
if h_values is not None and not h_values.empty and N_BOOTSTRAP_SAMPLES > 0:
    # Focus on top 2-3 interactions from initial calculation
    top_pairs_for_stability = h_values.head(min(3, len(h_values))).index.tolist()
    print(f"Assessing stability for pairs: {top_pairs_for_stability}")

    bootstrap_h_stats = {pair: [] for pair in top_pairs_for_stability}

    for i in range(N_BOOTSTRAP_SAMPLES):
        print(f"  Bootstrap sample {i+1}/{N_BOOTSTRAP_SAMPLES}", end='\r')
        X_boot, y_boot = resample(X_train, y_train, random_state=42 + i)

        # --- Option 1: Use original model on bootstrap sample (faster approximation) --- 
        boot_model = surrogate_model
        X_sample_boot = X_boot.sample(min(100, len(X_boot)), random_state=123 + i) if len(X_boot) > 100 else X_boot

        # --- Option 2: Retrain model on bootstrap sample (much slower, more accurate stability) ---
        # try:
        #     boot_model = clone(surrogate_model).fit(X_boot, y_boot)
        #     X_sample_boot = X_boot # Use full bootstrap sample if retraining
        # except Exception as fit_err:
        #     print(f"\nError fitting model on bootstrap sample {i+1}: {fit_err}. Skipping sample.")
        #     continue
        # --------------------------------------------------------------------------------------

        for f1, f2 in top_pairs_for_stability:
             h_boot = friedman_h_statistic(boot_model, X_sample_boot, f1, f2)
             if not np.isnan(h_boot):
                 bootstrap_h_stats[(f1, f2)].append(h_boot)

    print("\nInteraction Stability Results (Mean +/- Std Dev H-statistic across samples):")
    for pair, h_list in bootstrap_h_stats.items():
        if h_list:
            mean_h = np.mean(h_list)
            std_h = np.std(h_list)
            interaction_stability[pair] = {'mean': mean_h, 'std': std_h}
            print(f"  {pair}: {mean_h:.3f} +/- {std_h:.3f}")
        else:
            print(f"  {pair}: No valid H-statistics calculated in bootstrap.")
else:
    print("Skipping interaction stability (no initial H-stats or N_BOOTSTRAP_SAMPLES is 0).")

# d/e/g/i. Visualize Non-linear Effects & Interactions
print("\nGenerating Partial Dependence Plots (PDP/ICE)...")
top_shap_features = shap_df['feature'].head(5).tolist() if shap_df is not None else features[:5]

for feature in top_shap_features:
    try:
        print(f"  Generating PDP/ICE for {feature}")
        fig_pdp, ax_pdp = plt.subplots(figsize=(8, 6))
        display = PartialDependenceDisplay.from_estimator(
            surrogate_model,
            X_train, # Use train data for PDP calculation
            features=[feature],
            kind='both',
            ax=ax_pdp
        )
        ax_pdp.set_title(f'PDP and ICE for {feature}')
        plt.tight_layout()
        save_plot(fig_pdp, f"pdp_ice_{feature}", OUTPUT_DIR)
    except Exception as e:
        print(f"Could not generate PDP/ICE for {feature}: {e}")

print("\nGenerating Custom Interaction PDP plots (Matplotlib)...")
interaction_strengths = {} # Store H-stats calculated by custom function

# Determine the list of pairs to plot
if interaction_pairs: # interaction_pairs holds all combinations from earlier
    pairs_to_plot = interaction_pairs
    print(f"Generating custom interaction plots for all {len(pairs_to_plot)} feature pairs...")
    print("WARNING: This may take a significant amount of time and generate many files.")
else:
    print("No interaction pairs identified to plot.")
    pairs_to_plot = []

# Generate custom interaction plots for ALL pairs
for pair in pairs_to_plot: # Iterate through all pairs
    if isinstance(pair, tuple) and len(pair) == 2:
        f1, f2 = pair
        print(f"\nCreating custom PDP interaction plot for {f1} and {f2}")
        try:
            # Call the custom plotting function
            fig_custom_pdp, h_stat_custom = create_interaction_pdp(
                surrogate_model, X_train,
                [f1, f2],
                [f1, f2] # Pass names explicitly
            )
            if fig_custom_pdp is not None:
                # Use save_plot helper for the matplotlib figure
                save_plot(fig_custom_pdp, f"custom_pdp_interaction_{f1}_vs_{f2}", OUTPUT_DIR)
                # Only store strength if calculated successfully
                if h_stat_custom is not None and not np.isnan(h_stat_custom):
                     interaction_strengths[(f1, f2)] = h_stat_custom
                print(f"Custom PDP interaction plot saved. Calculated H-stat Approx: {h_stat_custom if h_stat_custom is not None else 'Error'}")
            else:
                print(f"Custom PDP interaction plot generation failed for ({f1}, {f2}).")
        except Exception as e:
            print(f"Error creating custom PDP plot for ({f1}, {f2}): {e}")
    else:
         print(f"Skipping custom interaction plot for invalid pair: {pair}")

# --- Add Feature Importance (after calculations in Sec 6/6b) ---
# Consolidate importance results before adding to report
report_generator.add_feature_importance_summary(shap_df, combined_imp)

# --- 8. Causality Analysis (Granger Causality) ---
print("\n--- 8. Causality Analysis (Granger) ---")
# Warning: Granger causality assumes stationarity. Data should ideally be checked/transformed first.
# For simplicity, we run it directly on X_train here.
GRANGER_MAX_LAG = 3 # Define max lag for tests

# Add Stationarity Check before Granger
print("Performing Stationarity Tests (ADF) on training data...")
stationarity_results = {}
for feature in features:
    try:
        result = adfuller(X_train[feature].dropna()) # Drop NaNs just in case
        stationarity_results[feature] = {
            'adf_stat': result[0],
            'p_value': result[1],
            'is_stationary': result[1] <= 0.05
        }
        print(f"  {feature}: p-value={result[1]:.3f} ({'Stationary' if result[1] <= 0.05 else 'Non-Stationary'})")
    except Exception as e:
        print(f"  Could not perform ADF test for {feature}: {e}")
        stationarity_results[feature] = {'error': str(e)}

# Add stationarity results to report
report_generator.add_stationarity_summary(stationarity_results)

print("\nRunning Granger Causality Tests...")
# ... (Existing Granger causality code: perform_granger_causality call, plotting, reporting) ...

# --- 9. Interpretability Summary (Reiteration) ---
print("\n--- 9. Interpretability Summary (Reiteration) ---")
if shap_values is not None:
    print("Generating SHAP Force Plots for sample predictions...")
    try:
        # Ensure explainer and shap_values correspond to the same model type if switching
        if SURROGATE_MODEL_TYPE == 'random_forest' or SURROGATE_MODEL_TYPE == 'xgboost':
             force_plot_html = shap.force_plot(explainer.expected_value, shap_values.values[0,:], X_test.iloc[0,:], show=False)
             shap.save_html(os.path.join(OUTPUT_DIR, "interpret_shap_force_plot_first.html"), force_plot_html)
             print("Saved sample SHAP force plot (HTML).")
        else:
             print("SHAP Force plot generation not configured for this model type.")
    except Exception as e:
        print(f"Could not generate SHAP force plot: {e}")
else:
    print("SHAP values not available, skipping force plots.")

# --- 10. Summarise Key Findings ---
print("\n--- 10. Summarizing Key Findings (Detailed Text File) ---")
summary_filename = os.path.join(OUTPUT_DIR, "summary_findings_detailed.txt") # Rename old file
with open(summary_filename, "w") as f:
    f.write("--- Summary of Key Findings ---\n\n")
    if not combined_imp.empty:
        f.write("--- Feature Importance Rankings (Full Lists) ---\n")
        f.write("\nImportance Scores:\n")
        f.write(combined_imp.to_string())
        f.write("\n\nImportance Ranks (Lower is better):\n")
        rank_cols = [col for col in combined_imp.columns if '_rank' in col]
        if rank_cols and 'mean_rank' in combined_imp.columns:
            f.write(combined_imp[['mean_rank'] + rank_cols].to_string())
        else:
            f.write("(Ranking data incomplete)")
        f.write("\n\n")
    else:
        f.write("No combined importance scores available.\n\n")

    if SURROGATE_MODEL_TYPE == 'random_forest' and not condition_importance_df.empty:
        f.write(f"--- Top {len(condition_importance_df)} Rule Conditions by Importance (Permutation) ---\n")
        f.write(condition_importance_df[['Original_Condition', 'Importance']].to_string(index=False))
        f.write("\n\n")
    elif SURROGATE_MODEL_TYPE == 'random_forest':
        f.write("--- Rule Conditions ---\nCondition importance could not be calculated or not applicable.\n\n")

    if h_values is not None and not h_values.empty:
        f.write(f"--- Top Feature Interactions (Approx. H-statistic) ---\n")
        f.write(h_values.to_string())
        f.write("\n\n")
    else:
        f.write("--- Feature Interactions ---\nNo valid H-statistic interactions calculated.\n\n")

    if interaction_stability:
        f.write("--- Interaction Stability (Mean +/- Std Dev H-statistic across Bootstraps) ---\n")
        for pair, stats in interaction_stability.items():
            pair_str = '__'.join(map(str, pair)) # Convert tuple key
            f.write(f"  {pair_str}: {stats['mean']:.3f} +/- {stats['std']:.3f}\n")
        f.write("\n")

    if h_3way is not None and not np.isnan(h_3way) and 'top_3_features' in locals():
        f.write("--- Approx. 3-Way Interaction (Std Dev of 3D PDP) ---\n")
        f.write(f"  Triplet {top_3_features}: {h_3way:.4f}\n\n")

    # Add Linear Model Results if they exist
    if 'rmse_orig' in locals():
        f.write("--- Linear Model Comparison ---\n")
        if 'actual_eng_features' in locals():
            f.write(f"Engineered Features Added: {actual_eng_features}\n")
        f.write(f"Linear Model (Original): Test RMSE={rmse_orig:.4f}, R2={r2_orig:.4f}\n")
        if 'rmse_eng' in locals():
            f.write(f"Linear Model (Enginrd.): Test RMSE={rmse_eng:.4f}, R2={r2_eng:.4f}\n")
        f.write("\n")

    # Add RuleFit Rules to detailed summary
    if rulefit_rules_df is not None and not rulefit_rules_df.empty:
         f.write(f"--- Top RuleFit Rules by Importance ---\n")
         f.write(rulefit_rules_df[['rule', 'coef', 'support', 'importance']].to_string(index=False))
         f.write("\n\n")

    # Add Condition-Only Linear Model Results
    if 'condition_linear_results' in locals() and condition_linear_results is not None:
        f.write("--- Linear Models Using ONLY Condition Features ---\n")
        f.write(f"Linear Regression: Test RMSE={condition_linear_results['linear']['rmse']:.4f}, R2={condition_linear_results['linear']['r2']:.4f}\n")
        f.write(f"RidgeCV Regression: Test RMSE={condition_linear_results['ridge']['rmse']:.4f}, R2={condition_linear_results['ridge']['r2']:.4f}, Alpha={condition_linear_results['ridge']['alpha']:.4f}\n")
        f.write(f"LassoCV Regression: Test RMSE={condition_linear_results['lasso']['rmse']:.4f}, R2={condition_linear_results['lasso']['r2']:.4f}, Alpha={condition_linear_results['lasso']['alpha']:.4f}\n\n")

        # Write Ridge Coefficients
        f.write("Top Ridge Coefficients (Conditions Only):\n")
        ridge_df_sum = pd.DataFrame(condition_linear_results['ridge']['coefficients'])
        f.write(ridge_df_sum[['Original_Condition', 'Ridge_Coefficient']].head(15).to_string(index=False, float_format="%.4f"))
        f.write("\n\n")

        # Write Lasso Coefficients
        f.write("Top Lasso Coefficients (Conditions Only):\n")
        lasso_df_sum = pd.DataFrame(condition_linear_results['lasso']['coefficients'])
        lasso_zeros = lasso_df_sum['Is_Zero'].sum()
        f.write(lasso_df_sum[['Original_Condition', 'Lasso_Coefficient', 'Is_Zero']].head(20).to_string(index=False, float_format="%.4f"))
        f.write(f"\n(Number of Lasso coefficients shrunk to zero: {lasso_zeros})\n\n")
    else:
        f.write("--- Linear Models Using ONLY Condition Features ---\n")
        f.write("(Not calculated, likely no condition features were generated)\n\n")

print(f"Saved detailed findings to: {summary_filename}")

# --- 11. Summary Visualisations ---
print("\n--- 11. Summary Visualizations ---")
# a. Feature Importance Comparison
if not combined_imp.empty:
    print("Generating Feature Importance Comparison plot...")
    # Ensure 'mean_rank' exists before plotting
    if 'mean_rank' in combined_imp.columns:
        plot_df = combined_imp.head(N_TOP_FEATURES).sort_values('mean_rank', ascending=True)
        fig_imp_comp = px.bar(plot_df,
                              x='mean_rank', y=plot_df.index, orientation='h',
                              title=f'Top {N_TOP_FEATURES} Features by Mean Importance Rank',
                              labels={'mean_rank': 'Mean Rank (Lower is Better)', 'y': 'Feature'})
        fig_imp_comp.update_layout(yaxis={'categoryorder':'total ascending'})
        save_plot(fig_imp_comp, "summary_featimp_comparison_rank", OUTPUT_DIR)
    else:
        print("'mean_rank' column not found, skipping importance comparison plot.")
else:
    print("Skipping feature importance comparison plot (no combined data).")

# b/c. Complex/Surrogate Features Plots (Engineered Feature Distributions)
# Moved plotting logic to Step 12 after features are created
# print("Generating plots for engineered features...")
# engineered_features_plotted = pd.DataFrame() # Initialize
# ... (rest of old plotting block removed)


# --- 12. Feature Engineering & Linear Model Comparison (Optional) ---
print("\n--- 12. Feature Engineering & Linear Model Comparison (Optional) ---")
engineered_features_train_df = pd.DataFrame(index=X_train.index)
engineered_features_test_df = pd.DataFrame(index=X_test.index)
new_feature_names = []
created_threshold_features = 0
created_interaction_features = 0

# Create Threshold features based on CONDITION IMPORTANCE
# Use the condition_importance_df created at the end of Section 6b
if SURROGATE_MODEL_TYPE == 'random_forest' and not condition_importance_df.empty:
    print("Creating engineered features based on MOST IMPORTANT conditions...")
    # Use conditions ranked by importance
    top_conditions_for_eng = condition_importance_df['Original_Condition'].head(min(5, len(condition_importance_df))).tolist()

    for condition_tuple in top_conditions_for_eng:
        feature, condition_str = condition_tuple
        try:
            op = None
            threshold = None
            # Robust parsing for operators and negative numbers
            if '<=' in condition_str:
                parts = condition_str.split('<=')
                if len(parts) == 2:
                    op = 'le'
                    threshold = float(parts[1].strip())
            elif '>' in condition_str:
                parts = condition_str.split('>')
                if len(parts) == 2:
                    op = 'gt'
                    threshold = float(parts[1].strip())

            if op and threshold is not None:
                # Format threshold for filename/colname, handle negative sign
                thresh_str = f"{abs(threshold):.2f}".replace('.','p') # use p for decimal
                neg_prefix = 'neg' if threshold < 0 else ''
                new_feat_name = f"{feature}_{op}_{neg_prefix}{thresh_str}"

                if new_feat_name in engineered_features_train_df.columns:
                     continue

                if op == 'le':
                     engineered_features_train_df[new_feat_name] = (X_train[feature] <= threshold).astype(int)
                     engineered_features_test_df[new_feat_name] = (X_test[feature] <= threshold).astype(int)
                elif op == 'gt':
                     engineered_features_train_df[new_feat_name] = (X_train[feature] > threshold).astype(int)
                     engineered_features_test_df[new_feat_name] = (X_test[feature] > threshold).astype(int)

                new_feature_names.append(new_feat_name)
                created_threshold_features += 1
                if created_threshold_features >= 4: # Limit to ~4 threshold features
                    break
            else:
                 print(f"Could not parse condition string: {condition_str}")
        except ValueError as ve:
             print(f"Could not convert threshold to float in condition {condition_tuple}: {ve}")
             continue
        except Exception as e:
            print(f"Could not create feature from condition {condition_tuple}: {e}")
            continue
    print(f"Created {created_threshold_features} threshold-based features.")
elif SURROGATE_MODEL_TYPE == 'random_forest':
     # This case handles when condition_importance_df IS empty
     print("Condition importance data not available or empty, skipping threshold feature engineering based on importance.")

# Create Interaction features
if h_values is not None and not h_values.empty:
    print("Creating engineered features based on interactions...")
    top_pairs = h_values.head(min(2, len(h_values))).index.tolist() # Top 2 pairs
    for pair in top_pairs:
         if isinstance(pair, tuple) and len(pair) == 2:
            f1, f2 = pair
            new_feat_name = f"{f1}_x_{f2}"
            if new_feat_name not in engineered_features_train_df.columns:
                 engineered_features_train_df[new_feat_name] = X_train[f1] * X_train[f2]
                 engineered_features_test_df[new_feat_name] = X_test[f1] * X_test[f2]
                 new_feature_names.append(new_feat_name)
                 created_interaction_features +=1
         else:
             print(f"Skipping interaction feature creation for invalid pair: {pair}")
    print(f"Created {created_interaction_features} interaction-based features.")

# --- Plot Engineered Features (Moved from Step 11) ---
print("Generating plots for engineered features...")
if not engineered_features_train_df.empty:
    print(f"Plotting distributions for {len(engineered_features_train_df.columns)} engineered features...")
    for eng_feat in engineered_features_train_df.columns:
        try:
            if engineered_features_train_df[eng_feat].nunique() <= 2:
                 counts = engineered_features_train_df[eng_feat].value_counts()
                 fig_eng_dist = px.bar(counts, x=counts.index, y=counts.values,
                                        title=f'Distribution of Engineered Feature: {eng_feat} (Train Set)',
                                        labels={'x': 'Value', 'y': 'Count'})
            else:
                 fig_eng_dist = px.histogram(engineered_features_train_df, x=eng_feat,
                                            title=f'Distribution of Engineered Feature: {eng_feat} (Train Set)')
            save_plot(fig_eng_dist, f"engineered_dist_{eng_feat}", OUTPUT_DIR)
        except Exception as e:
            print(f"Could not plot distribution for engineered feature {eng_feat}: {e}")
else:
    print("No engineered features were created to plot.")
# --- End Plotting Block ---


# Compare Linear Models
if not new_feature_names:
    print("No engineered features created based on thresholds or interactions. Skipping linear model comparison.")
    X_train_eng = X_train.copy() # Define for potential downstream use even if empty
    X_test_eng = X_test.copy()
else:
    # Combine original and engineered features
    X_train_eng = pd.concat([X_train, engineered_features_train_df], axis=1)
    X_test_eng = pd.concat([X_test, engineered_features_test_df], axis=1)
    # Ensure columns match, handling potential NaNs if test set alignment fails
    try:
        X_test_eng = X_test_eng.reindex(columns=X_train_eng.columns).fillna(0) # Fill potential new NaNs with 0
    except Exception as e:
        print(f"Error aligning test set columns for engineered features: {e}. Skipping linear model.")
        X_train_eng = X_train.copy()
        X_test_eng = X_test.copy()
        new_feature_names = [] # Reset features if alignment failed

# Only proceed with linear models if engineered features were successfully added or not attempted
if 'X_train_eng' in locals() and 'X_test_eng' in locals():
    print("Training Linear Models...")
    try:
        # Model 1: Original Features
        lr_orig = LinearRegression()
        lr_orig.fit(X_train, y_train)
        y_pred_test_orig = lr_orig.predict(X_test)
        rmse_orig = np.sqrt(mean_squared_error(y_test, y_pred_test_orig))
        r2_orig = r2_score(y_test, y_pred_test_orig)
        print(f"  Linear Model (Original): Test RMSE={rmse_orig:.4f}, R2={r2_orig:.4f}")

        # Model 2: Engineered Features (only if new features were added)
        if new_feature_names:
            lr_eng = LinearRegression()
            lr_eng.fit(X_train_eng, y_train)
            y_pred_test_eng = lr_eng.predict(X_test_eng)
            rmse_eng = np.sqrt(mean_squared_error(y_test, y_pred_test_eng))
            r2_eng = r2_score(y_test, y_pred_test_eng)
            print(f"  Linear Model (Enginrd.): Test RMSE={rmse_eng:.4f}, R2={r2_eng:.4f}")

            # Save results to summary
            with open(summary_filename, "a") as f:
                f.write("\n\n--- Linear Model Comparison ---\n")
                actual_eng_features = list(engineered_features_train_df.columns)
                f.write(f"Engineered Features Added: {actual_eng_features}\n")
                f.write(f"Linear Model (Original): Test RMSE={rmse_orig:.4f}, R2={r2_orig:.4f}\n")
                f.write(f"Linear Model (Enginrd.): Test RMSE={rmse_eng:.4f}, R2={r2_eng:.4f}\n")
        else:
             # Save only original model results if no engineered features used
             with open(summary_filename, "a") as f:
                f.write("\n\n--- Linear Model Comparison ---\n")
                f.write("(No engineered features added or used)\n")
                f.write(f"Linear Model (Original): Test RMSE={rmse_orig:.4f}, R2={r2_orig:.4f}\n")

    except Exception as e:
        print(f"Error during Linear Model training or evaluation: {e}")

print("\n--- Generating Summary Report & Saving Results ---")
report_generator.add_limitations_section()
report_markdown = report_generator.generate_report(output_path=os.path.join(OUTPUT_DIR, "analysis_summary_report.md"))
report_generator.save_analysis_results(output_path=os.path.join(OUTPUT_DIR, "analysis_results.json"))

print("\n--- Workflow Complete ---")
print(f"All outputs saved in directory: {OUTPUT_DIR}")

# --- Re-enabled: Generate Plotly 3D PDP plots ---
print("\nAttempting to generate Plotly 3D PDP plots...")

# --- MODIFIED: Select pairs based on top N features --- 
pairs_for_plotly_pdp = []
if not combined_imp.empty and len(features) >= 2:
    top_n_pdp_features = combined_imp.head(N_TOP_FEATURES_FOR_PDP_3D).index.tolist()
    if len(top_n_pdp_features) >= 2:
        pairs_for_plotly_pdp = list(combinations(top_n_pdp_features, 2))
        print(f"Generating Plotly 3D PDPs for top {N_TOP_FEATURES_FOR_PDP_3D} feature pairs ({len(pairs_for_plotly_pdp)} pairs): {top_n_pdp_features}")
    else:
         print(f"Need at least 2 top features for Plotly PDP plots, found {len(top_n_pdp_features)}.")
elif combined_imp.empty:
    print("Combined importance is empty, cannot determine top features for Plotly PDP plots.")
elif len(features) < 2:
     print("Need at least 2 features in the dataset for Plotly PDP plots.")

if pairs_for_plotly_pdp: # Only proceed if pairs were generated
    # Plot for the selected pairs
    for pair in pairs_for_plotly_pdp: # Iterate through defined pairs
        if isinstance(pair, tuple) and len(pair) == 2:
            f1, f2 = pair
            print(f"\nCalculating PDP for Plotly 3D plot: ({f1}, {f2})")
            try:
                feature_names_pdp = X_train.columns.tolist()
                # Find indices safely
                if f1 not in feature_names_pdp or f2 not in feature_names_pdp:
                     print(f"  Skipping pair ({f1}, {f2}), feature not found in X_train columns.")
                     continue
                f1_idx = feature_names_pdp.index(f1)
                f2_idx = feature_names_pdp.index(f2)

                pdp_results = partial_dependence(
                    surrogate_model,
                    X_train,
                    features=[(f1_idx, f2_idx)], # Use indices
                    kind='average',
                    grid_resolution=20
                )

                # --- Access results using correct keys --- 
                print("Accessing PDP results...")
                # Ensure results are structured as expected
                if 'average' in pdp_results and len(pdp_results['average']) > 0 and \
                   'grid_values' in pdp_results and len(pdp_results['grid_values']) == 2:
                    pdp_values = pdp_results['average'][0]
                    grid_f1 = pdp_results['grid_values'][0]
                    grid_f2 = pdp_results['grid_values'][1]
                else:
                     print(f"  Error: Unexpected structure in partial_dependence results for ({f1}, {f2}). Skipping plot.")
                     continue
                # --- End Access --- 

                # --- Create Plotly 3D Plot --- 
                print("Creating Plotly 3D Surface plot...")
                fig_plotly_pdp3d = go.Figure(data=[go.Surface(z=pdp_values, x=grid_f1, y=grid_f2, colorscale='Viridis')])
                fig_plotly_pdp3d.update_layout(title=f'Plotly 3D PDP: {f1} vs {f2}',
                                      scene = dict(xaxis_title=f1, yaxis_title=f2, zaxis_title='Partial Dependence'),
                                      autosize=True, margin=dict(l=65, r=50, b=65, t=90))
                save_plot(fig_plotly_pdp3d, f"plotly_pdp_3d_{f1}_{f2}", OUTPUT_DIR)
                # --- End Plotting --- 

            except Exception as e:
                print(f"Error generating Plotly 3D PDP plot for ({f1}, {f2}): {e}")
        else:
             print(f"Skipping Plotly 3D PDP for invalid pair: {pair}")
else:
    print("No feature pairs selected for Plotly 3D PDP plots based on top features.")
# --- END MODIFICATION --- 


# --- 8. Causality Analysis (Granger Feature -> Target) ---
print("\n--- 8. Causality Analysis (Granger Feature -> Target) ---")
# ... (Stationarity Check code remains the same) ...

print("\nRunning Granger Causality Tests (Feature -> Target)...")
GRANGER_MAX_LAG = 3 # Define max lag for tests

# Call should now match definition (no target_name arg)
granger_p_values = perform_granger_causality(X_train, y_train, features, GRANGER_MAX_LAG)

if granger_p_values is not None and not granger_p_values.isnull().all():
    # Visualize p-values with a bar chart
    granger_plot_data = granger_p_values.reset_index()
    granger_plot_data.columns = ['Feature', 'P_Value']
    # Sort for better visualization
    granger_plot_data = granger_plot_data.sort_values('P_Value')

    fig_granger = px.bar(granger_plot_data, x='Feature', y='P_Value',
                         title=f'Granger Causality (Feature -> {TARGET_VARIABLE}) P-Values (Lag {GRANGER_MAX_LAG})',
                         labels={'P_Value': 'P-Value (Lower suggests causality)'})
    # Add significance line
    fig_granger.add_hline(y=0.05, line_dash="dash", line_color="red", annotation_text="p=0.05")
    fig_granger.update_layout(yaxis_range=[0, max(0.1, granger_plot_data['P_Value'].max()*1.1)]) # Adjust y-axis focus
    save_plot(fig_granger, "causality_granger_bar", OUTPUT_DIR) # Changed filename
    print("Saved Granger Causality bar chart.")

    # Add to report
    report_generator.add_granger_causality_summary(granger_p_values, GRANGER_MAX_LAG)
else:
    print("Granger Causality analysis failed or yielded no results. Skipping chart and report section.")
    report_generator.add_section("8b. Causality Analysis (Granger Feature -> Target)", "Granger causality tests could not be completed.")

# --- 6c. Rule Extraction with imodels (RuleFit) ---
print("\n--- 6c. Rule Extraction with imodels (RuleFit) ---")
# rulefit_rules_df = None # Initialized earlier
try:
    print("Training RuleFitRegressor model...")
    # Initialize and fit RuleFitRegressor
    # Consider adjusting hyperparameters like max_rules, tree_size, etc.
    rulefit = RuleFitRegressor(random_state=42)
    rulefit.fit(X_train.values, y_train, feature_names=features) # RuleFit often prefers numpy arrays

    print("Extracting rules from RuleFit...")
    # --- FIX: Use .rules_ attribute instead of .get_rules() ---
    rules_ = rulefit.rules_
    # --- END FIX ---
    # --- ADDED: Print structure for debugging ---
    print(f"RuleFit raw rules_ structure (first 5): {rules_[:5]}") 
    # --- END ADDED ---
    # --- FIX: Check list emptiness correctly and convert to DataFrame ---
    if rules_ is None or not rules_: # Check if list is None or empty
         print("RuleFit model did not generate any rules.")
         rulefit_rules_df = None
    else:
        rulefit_rules_df = pd.DataFrame(rules_) # Convert list of dicts to DataFrame
        # --- END FIX ---
        # Filter out rules with zero coefficient (linear terms are handled separately if needed)
        # Ensure DataFrame is not empty after conversion before filtering
        if not rulefit_rules_df.empty:
            # --- Temporarily Commented Out for Debugging --- 
            # # --- FIX: Check if 'type' column exists before filtering --- 
            # if 'type' in rulefit_rules_df.columns:
            #     rulefit_rules_df = rulefit_rules_df[rulefit_rules_df['type'] == 'rule'].copy() # Ensure we copy
            # else:
            #     print("Warning: 'type' column not found in RuleFit rules. Skipping filtering by type.")
            # # --- END FIX ---
            # # Filter by non-zero coefficient
            # if 'coef' in rulefit_rules_df.columns:
            #     rulefit_rules_df = rulefit_rules_df[rulefit_rules_df['coef'] != 0].copy()
            # else:
            #      print("Warning: 'coef' column not found in RuleFit rules. Cannot filter by coefficient.")
            #      rulefit_rules_df = pd.DataFrame() # Make empty if coef missing
            pass # Keep indentation correct
            # --- End Temporarily Commented Out --- 
        else:
             print("RuleFit generated rules but DataFrame conversion or initial rules were empty.")
             rulefit_rules_df = pd.DataFrame() # Ensure it's an empty DF

    # --- Temporarily Commented Out for Debugging ---
    # if rulefit_rules_df is not None and not rulefit_rules_df.empty:
    #     # Sort rules by absolute coefficient magnitude
    #     # --- Need to confirm 'coef' column name based on debug output --- 
    #     # if 'coef' in rulefit_rules_df.columns: 
    #     #     rulefit_rules_df['importance'] = rulefit_rules_df['coef'].abs()
    #     #     rulefit_rules_df = rulefit_rules_df.sort_values('importance', ascending=False)
    # 
    #     #     print("\nTop 10 Rules from RuleFitRegressor:")
    #     #     print(rulefit_rules_df[['rule', 'coef', 'support']].head(10))
    #     # 
    #     #     # Visualize top rule importances
    #     #     plot_df = rulefit_rules_df.head(20).sort_values('importance', ascending=True)
    #     #     fig_rulefit = px.bar(plot_df,
    #     #                          x='coef', y='rule', orientation='h',
    #     #                          title='Top 20 RuleFit Rule Importances (Coefficient Magnitude)',
    #     #                          labels={'coef': 'Coefficient (Importance)', 'rule': 'Rule'},
    #     #                          color='coef',
    #     #                          color_continuous_scale=px.colors.diverging.Picnic)
    #     #     fig_rulefit.update_layout(yaxis={'tickmode': 'array', 'tickvals': list(range(len(plot_df))), 'ticktext': plot_df['rule'].tolist()})
    #     #     save_plot(fig_rulefit, "rulefit_rule_importance", OUTPUT_DIR)
    #     # else:
    #     #     print("Cannot sort/plot RuleFit importance, 'coef' column missing.")
    # 
    #     # Add summary to report
    #     report_generator.add_rulefit_summary(rulefit_rules_df)
    # else:
    #     print("RuleFitRegressor did not generate any rules with non-zero coefficients or analysis failed.")
    #     report_generator.add_rulefit_summary(None) # Add section indicating no rules
    # --- End Temporarily Commented Out --- 

except ImportError:
    print("imodels library not found (pip install imodels). Skipping RuleFit analysis.")
    report_generator.add_rulefit_summary(None)
except Exception as e:
    print(f"Error during RuleFit analysis: {e}")
    report_generator.add_rulefit_summary(None)

# ... (Rest of script) ...