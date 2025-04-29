# %% [markdown]
# # Surrogate-Assisted Feature Extraction (SAFE) Analysis Workflow
#
# This script replicates the analysis from `safe_analysis.py` but is structured
# like a Jupyter Notebook using special cell comments (`# %%`).
# It aims to analyze feature importance, interactions, and generate
# interpretable insights from complex machine learning models using a surrogate.

# %% [markdown]
# ## 1. Imports and Setup

# %%
# Standard library imports
import os
import json
import warnings
from datetime import datetime
from itertools import combinations
from collections import Counter

# Third-party imports
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, TimeSeriesSplit, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.inspection import (
    partial_dependence,
    permutation_importance,
    PartialDependenceDisplay
)
from sklearn.tree import plot_tree, export_text
from sklearn.utils import resample
from imodels import RuleFitRegressor
import shap
from lofo import LOFOImportance, Dataset, plot_importance
import xgboost as xgb
import statsmodels.api as sm
from statsmodels.tsa.stattools import grangercausalitytests, adfuller
from sklearn.base import clone

# Suppress warnings
warnings.filterwarnings('ignore')

# %% [markdown]
# ## 2. Configuration Constants

# %%
# --- Configuration ---
DATA_PATH = '_data/refinery_margins.csv'
TARGET_VARIABLE = 'refinery_kbd'
DATE_COLUMN = 'date'
GROUPING_COLUMN = 'country'
FILTER_VALUE = 'United Kingdom'
TEST_MONTHS = 12
SURROGATE_MODEL_TYPE = 'random_forest' # 'random_forest' or 'xgboost'
N_TREES_TO_VISUALIZE = 2
N_TOP_FEATURES = 10
N_BOOTSTRAP_SAMPLES = 3 # Reduced for speed
OUTPUT_DIR = 'safe_analysis_notebook_output' # New output dir for this version
EXOGENOUS_VARIABLES = 'all'  # Can be 'all' or a list of variable names
RANDOM_STATE = 42

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Ensured output directory exists: {OUTPUT_DIR}")

# %% [markdown]
# ## 3. Report Generator Class Definition

# %%
class ReportGenerator:
    """
    A class for generating comprehensive analysis reports using surrogate models.
    (Includes all methods from the original script)
    """

    def __init__(
        self,
        output_dir=OUTPUT_DIR, # Use the globally defined OUTPUT_DIR
        surrogate_model_type=SURROGATE_MODEL_TYPE,
        exogenous_variables=EXOGENOUS_VARIABLES
    ):
        """Initialize the ReportGenerator."""
        self.output_dir = output_dir
        self.surrogate_model_type = surrogate_model_type
        self.exogenous_variables = exogenous_variables
        self.feature_names = None
        self.target_name = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.surrogate_performance = {}
        self.engineered_performance = {}
        self.analysis_results = {
            'top_features': [],
            'feature_importance': {},
            'interactions': {},
            'surrogate_performance': {},
            'engineered_performance': {}
        }
        self.report_sections = []
        self.ensure_output_dir()

        # Initialize performance metrics
        self.train_rmse = None
        self.train_r2 = None
        self.test_rmse = None
        self.test_r2 = None

        # Initialize feature importance results
        self.top_features = []
        self.engineered_features = []
        self.actual_eng_features = []
        self.top_3_features = []

        # Initialize model performance metrics
        self.rmse_orig = None
        self.r2_orig = None
        self.rmse_eng = None
        self.r2_eng = None

        # Initialize analysis results (detailed)
        self.analysis_results = {
            'top_features': [],
            'feature_importance': {},
            'interactions': {},
            'performance': {
                'original': {'rmse': None, 'r2': None},
                'engineered': {'rmse': None, 'r2': None}
            }
        }

        # Initialize results dictionary for JSON dump
        self.results_dict = {}

        # Initialize report sections list
        self.report_sections = [] # Stores {'title': ..., 'content': ...} dicts

    def ensure_output_dir(self):
        """Create output directory if it doesn't exist."""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print(f"Ensured output directory exists: {self.output_dir}")

    def add_section(
        self,
        title,
        content,
        level=1,
        add_newlines=True,
        code_block=False
    ):
        """Add a section to the report sections list."""
        # This method now stores the sections instead of writing directly
        self.report_sections.append({'title': title, 'content': content, 'level': level, 'code_block': code_block})
        print(f"Added report section: {title}")

    def generate_markdown_report(self, filename="analysis_report.md"):
        """Generates the full markdown report from stored sections."""
        full_report = f"# SAFE Analysis Report ({datetime.now().strftime('%Y-%m-%d %H:%M')})\n\n"
        full_report += "## Table of Contents\n"
        for i, section_data in enumerate(self.report_sections):
             title = section_data['title']
             # Create a simple anchor link (replace spaces, lower case)
             anchor = title.lower().replace(' ', '-').replace('/', '').replace(':', '').replace('.', '')
             full_report += f"  {i+1}. [{title}](#{anchor})\n"
        full_report += "\n---\n\n"

        for section_data in self.report_sections:
            title = section_data['title']
            content = section_data['content']
            level = section_data.get('level', 1)
            code_block = section_data.get('code_block', False)
            anchor = title.lower().replace(' ', '-').replace('/', '').replace(':', '').replace('.', '') # Recreate anchor

            full_report += f"\n<a name='{anchor}'></a>\n" # Add anchor tag
            full_report += f"{'#' * level} {title}\n\n"

            if code_block:
                content = f"```\n{content}\n```"

            full_report += f"{content}\n\n"

        filepath = os.path.join(self.output_dir, filename)
        try:
            with open(filepath, 'w') as f:
                f.write(full_report)
            print(f"Successfully generated Markdown report: {filepath}")
        except Exception as e:
            print(f"Error writing Markdown report: {e}")

    def update_analysis_results(self):
        """Update the analysis results dictionary with current metrics."""
        # This seems redundant with how results are added to results_dict,
        # but keeping it for potential future use or structure.
        self.analysis_results['performance']['original'] = {
            'rmse': self.results_dict.get('rmse_orig'),
            'r2': self.results_dict.get('r2_orig')
        }
        self.analysis_results['performance']['engineered'] = {
            'rmse': self.results_dict.get('rmse_eng'),
            'r2': self.results_dict.get('r2_eng')
        }
        self.analysis_results['top_features'] = self.results_dict.get('top_3_features', [])
        self.analysis_results['engineered_features'] = self.results_dict.get('actual_eng_features', [])

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
            self.results_dict['top_features_shap'] = []

        content += "\n### Top Features by Combined Importance Rank (Train Set):\n"
        if combined_imp_df is not None and not combined_imp_df.empty and 'mean_rank' in combined_imp_df.columns:
            top_combined = combined_imp_df.head(5)
            for i, (idx, row) in enumerate(top_combined.iterrows()):
                content += f"- `{idx}` (Mean Rank: {row['mean_rank']:.2f})\n"
            self.results_dict['top_features_combined'] = top_combined.to_dict('index')
        else:
            content += "- (Combined ranking not available)\n"
            self.results_dict['top_features_combined'] = {}

        self.add_section("2. Key Features and Their Importance", content)

    def add_feature_interactions_summary(self, h_values, interaction_stability, h_3way_stats=None):
        """Adds feature interaction summary."""
        content = "### Significant Pairwise Interactions (Approx. H-statistic):\n"
        top_h_interactions_dict = {}
        if h_values is not None and not h_values.empty:
            count = 0
            for (f1, f2), h_stat in h_values.head(10).items():
                content += f"- `{f1}` × `{f2}` (H ≈ {h_stat:.3f})"
                pair_key = tuple(sorted((f1, f2))) # Ensure consistent tuple order for lookup
                if f'{f1}_x_{f2}' not in top_h_interactions_dict: # Avoid duplicates due to order
                    top_h_interactions_dict[f'{f1}_x_{f2}'] = h_stat # Store for JSON

                if pair_key in interaction_stability:
                    stab = interaction_stability[pair_key]
                    content += f" (Stable: Mean={stab['mean']:.3f} ± {stab['std']:.3f})\n"
                else:
                    content += "\n"
                count += 1
                if count >= 5: # Limit report summary to top 5
                    break
            self.results_dict['pairwise_interactions_h_top5'] = top_h_interactions_dict
            self.results_dict['interaction_stability'] = interaction_stability # Store full stability dict
        else:
            content += "- (No significant pairwise interactions found or calculated)\n"
            self.results_dict['pairwise_interactions_h_top5'] = {}
            self.results_dict['interaction_stability'] = {}

        content += "\n### Three-way Interactions (Approx. H-statistic for Top Feature Triplets):\n"
        if h_3way_stats is not None and not h_3way_stats.empty:
            h_3way_stats_sorted = h_3way_stats.sort_values('H_statistic', ascending=False)
            for idx, row in h_3way_stats_sorted.head(5).iterrows(): # Show top 5 3-way
                triplet_str = ' × '.join(row['Triplet'])
                content += f"- `{triplet_str}` (H ≈ {row['H_statistic']:.4f})\n"
            self.results_dict['threeway_interactions_h'] = h_3way_stats_sorted.to_dict('records')
        else:
            content += "- (Not calculated or available for the selected triplets)\n"
            self.results_dict['threeway_interactions_h'] = []

        self.add_section("3. Feature Interactions", content)

    def add_condition_importance_summary(self, condition_importance_df):
        """Adds summary of top rule conditions based on importance."""
        content = "### Top Impactful Conditions/Thresholds (by Permutation Importance):\n"
        if condition_importance_df is not None and not condition_importance_df.empty:
            top_conditions_rec = []
            for i, row in condition_importance_df.head(5).iterrows():
                # Ensure 'Original_Condition' exists and is a tuple/list
                orig_cond = row.get('Original_Condition')
                if isinstance(orig_cond, (list, tuple)) and len(orig_cond) == 2:
                    feat, op_thresh = orig_cond
                    content += f"- `{feat} {op_thresh}` (Importance: {row['Importance']:.4f})\n"
                    top_conditions_rec.append(row.to_dict())
                else:
                    content += f"- `Condition Feature: {row.get('Condition_Feature', 'N/A')}` (Importance: {row['Importance']:.4f}) - Original condition missing\n"
            self.results_dict['top_conditions_by_importance'] = top_conditions_rec
        else:
            content += "- (Condition importance not available or calculated)\n"
            self.results_dict['top_conditions_by_importance'] = []

        self.add_section("4. Critical Thresholds / Conditions", content)

    def add_limitations_section(self):
        content = """- Dataset size might limit generalizability.
- Training-test performance gap may indicate potential overfitting or concept drift.
- H-statistic values are approximations based on PDP variances.
- Condition importance is a proxy derived from a secondary model's permutation importance.
- RuleFit rule extraction depends on its internal algorithm and may not capture all nuances.
- Temporal dynamics beyond simple train/test split are not explicitly modeled in this script (e.g., complex seasonality, autocorrelation in residuals require dedicated time series models).
- Granger causality checks for predictive power, not true causation, and assumes stationarity (which may be violated)."""
        self.add_section("7. Limitations", content)

    def _convert_to_serializable(self, obj):
        """Converts various data types to JSON-serializable formats."""
        if isinstance(obj, dict):
            return {
                str(k) if isinstance(k, (np.int64, np.int32)) else str(k):
                self._convert_to_serializable(v)
                for k, v in obj.items()
            }
        elif isinstance(obj, list):
            return [self._convert_to_serializable(item) for item in obj]
        elif isinstance(obj, tuple):
            # Convert tuple keys (like from H-stats) into strings
            return '__x__'.join(map(str, obj))
        elif isinstance(obj, pd.Series):
            # Convert Series to a dictionary {index: value}
            # Ensure index and values are serializable
            return {str(k): self._convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, pd.DataFrame):
            # Convert DataFrame to a dictionary format (e.g., 'split')
            df_copy = obj.copy()
            # Convert index and columns to string if necessary
            if not pd.api.types.is_numeric_dtype(df_copy.index.dtype):
                 df_copy.index = df_copy.index.map(str)
            df_copy.columns = df_copy.columns.map(str)
            # Handle potential mixed types within columns if necessary before to_dict
            # For simplicity, using 'split' which is often robust
            return df_copy.to_dict(orient='split')
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, np.uint8,
            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float16, np.float32, np.float64)):
            # Handle potential NaN/inf
            if np.isnan(obj): return 'NaN'
            if np.isinf(obj): return 'Infinity' if obj > 0 else '-Infinity'
            return float(obj)
        elif isinstance(obj, (np.complex64, np.complex128)):
             # Handle complex numbers
             real = obj.real
             imag = obj.imag
             if np.isnan(real): real = 'NaN'
             if np.isinf(real): real = 'Infinity' if real > 0 else '-Infinity'
             if np.isnan(imag): imag = 'NaN'
             if np.isinf(imag): imag = 'Infinity' if imag > 0 else '-Infinity'
             return {'real': real, 'imag': imag}
        elif isinstance(obj, (np.bool_)):
            return bool(obj)
        elif isinstance(obj, (np.void)):
            return None # Represent void as None
        elif isinstance(obj, (pd.Timestamp, datetime)):
            return obj.isoformat()
        elif isinstance(obj, Counter):
            # Convert Counter keys/values just in case
            return {self._convert_to_serializable(k): self._convert_to_serializable(v) for k, v in obj.items()}
        elif obj is None:
            return None
        else:
            # Fallback for other types
            try:
                # Check if it's already a basic serializable type
                if isinstance(obj, (str, int, float, bool)):
                    return obj
                # Last resort: string representation, warn about it
                # print(f"Warning: Converting type {type(obj).__name__} to string for JSON serialization.")
                return str(obj)
            except Exception:
                return f"<unserializable type: {type(obj).__name__}>"

    def save_analysis_results(self, output_path="analysis_results.json"):
        """Save the results_dict to a JSON file after converting types."""
        filepath = os.path.join(self.output_dir, output_path)
        try:
            # Ensure all values in results_dict are serializable
            serializable_results = {k: self._convert_to_serializable(v) for k, v in self.results_dict.items()}

            with open(filepath, 'w') as f:
                json.dump(serializable_results, f, indent=4)
            print(f"Successfully saved analysis results (JSON): {filepath}")

        except TypeError as te:
             print(f"JSON Serialization Error: {te}")
             print("Attempting to save problematic keys/values for debugging:")
             for k, v in self.results_dict.items():
                 try:
                     json.dumps({k: self._convert_to_serializable(v)})
                 except Exception as e:
                     print(f"  - Failed key: '{k}', type: {type(v)}, Error: {e}")
             print("Please review the types above and adjust the '_convert_to_serializable' method if needed.")
        except Exception as e:
            print(f"Error saving analysis results to JSON: {str(e)}")
            # Consider saving a partial dump or logging the error differently

    def add_granger_causality_summary(self, p_value_series, max_lag):
        """Adds Granger causality (feature -> target) summary to the report."""
        content = f"Pairwise Granger causality tests performed for each feature predicting the target variable, up to lag {max_lag}. "
        content += "Lower p-values suggest a feature Granger-causes the target (helps predict its future values).\n\n"
        if p_value_series is not None and not p_value_series.isnull().all():
             significant = p_value_series[p_value_series <= 0.05]
             if not significant.empty:
                  content += "**Significant Features (p <= 0.05):**\n"
                  for feat, pval in significant.items():
                       content += f"- `{feat}` (p={pval:.3f})\n"
             else:
                  content += "**No features found significant (p <= 0.05).**\n"
        else:
             content += "(Results not available or all NaN).\n"

        content += "\nSee the bar chart plot (`causality_granger_bar.png/.html`) for details.\n"
        content += "Note: Granger causality checks for predictive power, not necessarily true causation, and assumes stationarity (check Section 8a).\n"
        self.add_section("8b. Causality Analysis (Granger Feature -> Target)", content)
        # Store the series itself in results
        if p_value_series is not None:
            self.results_dict['granger_causality_feature_to_target_p_values'] = p_value_series
        else:
             self.results_dict['granger_causality_feature_to_target_p_values'] = None

    def add_stationarity_summary(self, stationarity_results):
        """Adds summary of ADF stationarity tests to the report."""
        content = "Augmented Dickey-Fuller (ADF) Test performed on training features to check for stationarity."\
                  " A p-value <= 0.05 typically suggests stationarity (rejecting the null hypothesis of a unit root).\n\n"
        num_stationary = 0
        num_tested = 0
        if stationarity_results:
            for feature, result in stationarity_results.items():
                 if 'p_value' in result:
                      p_val = result['p_value']
                      is_stationary = p_val <= 0.05
                      if is_stationary:
                           num_stationary += 1
                      content += f"- **`{feature}`**: ADF Stat={result.get('adf_stat', 'N/A'):.3f}, p-value={p_val:.3f} -> {'Stationary' if is_stationary else 'Non-Stationary'}\n"
                      num_tested += 1
                 elif 'error' in result:
                      content += f"- **`{feature}`**: Error during test - {result['error']}\n"
                 else:
                      content += f"- **`{feature}`**: Test result format unknown.\n"

            content += f"\nSummary: {num_stationary} out of {num_tested} tested features appear stationary (p<=0.05).\n"
            content += "Non-stationary features may violate assumptions for standard Granger causality tests.\n"
        else:
             content += "No stationarity test results available.\n"
        self.add_section("8a. Stationarity Tests (ADF)", content)
        self.results_dict['stationarity_test_adf'] = stationarity_results

    def add_shap_interaction_summary(self, shap_interaction_df):
        """Adds summary of top SHAP interactions."""
        content = "Top pairwise feature interactions based on Mean Absolute SHAP Interaction values.\n"
        content += "These values represent the average magnitude of how the SHAP value of one feature changes based on the value of another feature.\n\n"
        if shap_interaction_df is not None and not shap_interaction_df.empty:
            # Extract top off-diagonal values
            try:
                # Ensure diagonal is ignored
                np.fill_diagonal(shap_interaction_df.values, -np.inf) # Set diagonal to negative infinity
                inter_values = shap_interaction_df.stack().sort_values(ascending=False)
                inter_values = inter_values[inter_values != -np.inf] # Remove diagonal entries

                content += "**Top Interactions:**\n"
                top_shap_inters = {}
                for (f1, f2), val in inter_values.head(10).items():
                     content += f"- `{f1}` <> `{f2}`: {val:.4f}\n"
                     # Store consistently ordered pair
                     pair_key = tuple(sorted((f1,f2)))
                     str_key = f'{pair_key[0]}_vs_{pair_key[1]}'
                     if str_key not in top_shap_inters:
                          top_shap_inters[str_key] = val
                self.results_dict['top_shap_interactions'] = top_shap_inters
            except Exception as e:
                content += f"- (Error processing SHAP interaction values: {e}).\n"
                self.results_dict['top_shap_interactions'] = {}
        else:
             content += "- (SHAP interaction values not calculated or available).\n"
             self.results_dict['top_shap_interactions'] = {}

        # Add this as part of section 3 (Interactions)
        sec3_idx = -1
        for i, sec in enumerate(self.report_sections):
            if sec['title'].startswith("3."):
                 sec3_idx = i
                 break
        if sec3_idx != -1:
             # Append to existing Section 3 content
             self.report_sections[sec3_idx]['content'] += "\n\n### Top SHAP Interactions (Mean Abs Value):\n" + content
        else: # Add as new section if 3 doesn't exist (shouldn't happen often)
             self.add_section("3b. SHAP Interaction Analysis", content)

    def add_rulefit_summary(self, rules_df):
        """Adds summary of top rules from RuleFitRegressor."""
        content = "RuleFitRegressor was trained to extract interpretable rules. "
        content += "Rules are linear combinations of the original features and automatically generated decision rules (feature thresholds/interactions).\n"
        content += "Coefficients indicate the importance and direction of the effect for each rule/feature in the linear model.\n\n"
        content += "**Top Rules/Features by Coefficient Magnitude:**\n"
        if rules_df is not None and not rules_df.empty and 'coef' in rules_df.columns:
            # Sort by absolute coefficient for importance
            rules_df_sorted = rules_df.iloc[rules_df['coef'].abs().sort_values(ascending=False).index]
            for i, row in rules_df_sorted.head(15).iterrows(): # Show top 15 rules/features
                rule_type = row.get('type', 'unknown')
                content += f"- **Type:** `{rule_type}` | **Rule/Feature:** `{row['rule']}` | **Coef:** {row['coef']:.4f} | **Support:** {row.get('support', 'N/A'):.3f}\n"
            # Add to results dict for JSON
            self.results_dict['rulefit_top_rules'] = rules_df_sorted.head(15).to_dict('records')
        else:
            content += "- (RuleFit analysis not performed or yielded no rules/features with coefficients).\n"
            self.results_dict['rulefit_top_rules'] = []

        # Add after section 4 (Thresholds/Conditions)
        sec4_idx = -1
        for i, sec in enumerate(self.report_sections):
            if sec['title'].startswith("4."):
                sec4_idx = i
                break
        if sec4_idx != -1:
            # Insert as a new dictionary in the list
            self.report_sections.insert(sec4_idx + 1, {'title': "4b. RuleFit Extracted Rules/Features", 'content': content})
        else: # Fallback add at end
             self.add_section("4b. RuleFit Extracted Rules/Features", content)


    def add_dynamic_summary(self):
        """Generates a dynamic narrative summary of key findings based on self.results_dict."""
        summary_text = """
This analysis utilized a surrogate model ({model_type}) to explore feature relationships and importance for predicting '{target}'.
Key findings are summarized below:

""".format(model_type=self.surrogate_model_type, target=self.target_name or TARGET_VARIABLE)

        key_findings_dict = {} # To store summary points for JSON

        # --- Surrogate Model Performance ---
        summary_text += "**Model Performance:**\n"
        surrogate_perf = self.results_dict.get('surrogate_performance', {})
        train_r2 = surrogate_perf.get('train_r2', -999)
        test_r2 = surrogate_perf.get('test_r2', -999)
        train_rmse = surrogate_perf.get('train_rmse', -999)
        test_rmse = surrogate_perf.get('test_rmse', -999)

        summary_text += f"- The surrogate model achieved a training R² of {train_r2:.3f} (RMSE: {train_rmse:.3f}).\n"
        if test_r2 < 0 or test_r2 < 0.1: # Poor generalization threshold
            summary_text += f"- However, the test R² was very low ({test_r2:.3f}, RMSE: {test_rmse:.3f}), indicating **poor generalization** to unseen data. The insights derived should be treated with caution.\n"
        elif train_r2 > 0.2 and (train_r2 - test_r2 > 0.3): # Potential overfitting threshold
            summary_text += f"- The test R² was {test_r2:.3f} (RMSE: {test_rmse:.3f}). The significant drop from training R² suggests potential **overfitting**.\n"
        elif test_r2 >= 0.1 :
            summary_text += f"- Test R² was {test_r2:.3f} (RMSE: {test_rmse:.3f}), suggesting some level of generalization.\n"
        key_findings_dict['surrogate_performance_summary'] = {'train_r2': train_r2, 'test_r2': test_r2, 'train_rmse': train_rmse, 'test_rmse': test_rmse}

        # --- Key Drivers (Features) ---
        summary_text += "\n**Key Features (Based on Combined Rank):**\n"
        combined_imp = self.results_dict.get('top_features_combined', {}) # Get from results_dict
        if combined_imp:
            # combined_imp is likely a dict {feature: {col: val}}, extract feature names
            top_features = list(combined_imp.keys())[:3] # Assume it's sorted, take top 3 keys
            if top_features:
                 summary_text += f"- Consistently high importance across methods was observed for: `{top_features[0]}`, `{top_features[1]}`, and `{top_features[2]}` (based on mean rank).\n"
                 key_findings_dict['top_ranked_features'] = top_features
            else:
                 summary_text += "- Unable to determine top 3 features from combined rank data.\n"
                 key_findings_dict['top_ranked_features'] = []
        else:
            summary_text += "- Combined feature importance ranking was not available or empty.\n"
            key_findings_dict['top_ranked_features'] = []


        # --- Key Conditions/Thresholds ---
        summary_text += "\n**Key Conditions/Thresholds (Based on Condition Importance):**\n"
        top_conditions_list = self.results_dict.get('top_conditions_by_importance', [])
        if top_conditions_list:
             summary_text += "- Analysis of rule conditions suggests specific thresholds are particularly influential (based on permutation importance of condition-derived features):\n"
             key_conditions = []
             for cond_data in top_conditions_list[:3]: # Top 3 conditions
                  orig_cond = cond_data.get('Original_Condition')
                  importance = cond_data.get('Importance', 'N/A')
                  if isinstance(orig_cond, (list, tuple)) and len(orig_cond) == 2:
                       summary_text += f"  - Condition: `{orig_cond[0]} {orig_cond[1]}` (Importance ≈ {importance:.3f})\n"
                       key_conditions.append(f"{orig_cond[0]} {orig_cond[1]}")
                  else:
                       summary_text += f"  - Condition Feature: `{cond_data.get('Condition_Feature', 'N/A')}` (Importance ≈ {importance:.3f}) - Original format issue\n"
             key_findings_dict['top_important_conditions'] = key_conditions
        else:
             summary_text += "- Analysis of specific condition importance was not performed or yielded no results.\n"
             key_findings_dict['top_important_conditions'] = []

        # --- Key Interactions ---
        summary_text += "\n**Feature Interactions:**\n"
        top_h_interactions = self.results_dict.get('pairwise_interactions_h_top5', {})
        interaction_stability = self.results_dict.get('interaction_stability', {})
        top_shap_interactions = self.results_dict.get('top_shap_interactions', {})
        key_stable_interactions = []

        if top_h_interactions:
             summary_text += "- Pairwise interactions assessed using approximate H-statistic:\n"
             count = 0
             for pair_str, h_stat in top_h_interactions.items():
                 f1, f2 = pair_str.split('_x_') # Reconstruct pair from string key
                 pair_key = tuple(sorted((f1, f2))) # Key for stability dict
                 stability_info = " (Stability not assessed)"
                 is_stable = False
                 if pair_key in interaction_stability:
                      stab = interaction_stability[pair_key]
                      # Define stability (e.g., low std dev relative to mean, ensure mean isn't near zero)
                      if stab.get('mean') is not None and stab.get('std') is not None and abs(stab['mean']) > 1e-6 and stab['std'] < 0.2 * abs(stab['mean']):
                           stability_info = f" (Stable: Mean={stab['mean']:.3f} ± {stab['std']:.3f})"
                           is_stable = True
                      elif stab.get('mean') is not None and stab.get('std') is not None:
                            stability_info = f" (Less Stable: Mean={stab['mean']:.3f} ± {stab['std']:.3f})"

                 summary_text += f"  - `{f1}` × `{f2}` (H ≈ {h_stat:.3f}){stability_info}\n"
                 if is_stable:
                      key_stable_interactions.append(f"{f1} x {f2}")
                 count += 1
             key_findings_dict['top_h_interactions'] = top_h_interactions
             key_findings_dict['stable_h_interactions'] = key_stable_interactions

        elif top_shap_interactions: # Fallback to SHAP if H failed/not available
             summary_text += "- Pairwise interactions assessed using Mean Absolute SHAP interaction values:\n"
             for pair_str, val in list(top_shap_interactions.items())[:3]: # Top 3 SHAP
                 f1, f2 = pair_str.split('_vs_')
                 summary_text += f"  - `{f1}` <> `{f2}` (Mean Abs SHAP Inter: {val:.4f})\n"
             key_findings_dict['top_shap_interactions'] = top_shap_interactions
        else:
             summary_text += "- Interaction analysis (H-statistic or SHAP) was not performed or yielded no results.\n"

        # --- Feature Engineering Impact ---
        summary_text += "\n**Feature Engineering Impact (Linear Model Test R²):**\n"
        linear_perf = self.results_dict.get('linear_model_comparison', {}) # Get from results_dict
        engineered_feature_names = self.results_dict.get('actual_eng_features', [])

        if linear_perf and engineered_feature_names: # Check if comparison was done and features were created
            r2_orig = linear_perf.get('r2_orig', -999)
            r2_eng = linear_perf.get('r2_eng', -999)
            improvement = r2_eng > r2_orig
            num_eng = len(engineered_feature_names)
            num_thresh = linear_perf.get('threshold_features_created', 0)
            num_inter = linear_perf.get('interaction_features_created', 0)

            summary_text += f"- {num_eng} features engineered from surrogate insights (threshold={num_thresh}, interaction={num_inter}).\n"
            if r2_eng == -999 or r2_orig == -999:
                 summary_text += "- Performance metrics for linear models missing, cannot assess impact.\n"
                 key_findings_dict['feature_engineering_result'] = {'improved': None}
            elif improvement:
                 summary_text += f"- Adding these features **improved** the linear model's test R² (from {r2_orig:.3f} to {r2_eng:.3f}).\n"
                 key_findings_dict['feature_engineering_result'] = {'improved': True, 'r2_orig': r2_orig, 'r2_eng': r2_eng}
            elif np.isclose(r2_eng, r2_orig):
                 summary_text += f"- Adding these features **did not change** the linear model's test R² ({r2_orig:.3f}).\n"
                 key_findings_dict['feature_engineering_result'] = {'improved': False, 'r2_orig': r2_orig, 'r2_eng': r2_eng}
            else: # r2_eng < r2_orig
                 summary_text += f"- Adding these features **decreased** the linear model's test R² (from {r2_orig:.3f} to {r2_eng:.3f}).\n"
                 key_findings_dict['feature_engineering_result'] = {'improved': False, 'r2_orig': r2_orig, 'r2_eng': r2_eng}
        else:
            summary_text += "- Feature engineering or linear model comparison was not performed / did not yield usable features.\n"
            key_findings_dict['feature_engineering_result'] = {'improved': None}


        # --- Causality Insights ---
        summary_text += "\n**Potential Predictive Relationships (Granger Causality - Feature -> Target):**\n"
        granger_p_values = self.results_dict.get('granger_causality_feature_to_target_p_values') # Get from dict
        stationarity_results = self.results_dict.get('stationarity_test_adf', {})
        significant_granger = []

        if granger_p_values is not None:
            # Filter Series for p <= 0.05, excluding NaN
            significant_granger_series = granger_p_values.dropna()[granger_p_values <= 0.05]
            if not significant_granger_series.empty:
                summary_text += "- The following features showed significant Granger causality towards the target (p<=0.05):\n"
                for feat, pval in significant_granger_series.items():
                    stationarity_info = stationarity_results.get(feat, {})
                    non_stationary_warning = ""
                    if 'is_stationary' in stationarity_info and not stationarity_info['is_stationary']:
                         non_stationary_warning = " (Warning: Feature Non-Stationary)"
                    elif 'error' in stationarity_info:
                         non_stationary_warning = " (Warning: Stationarity Test Error)"

                    summary_text += f"  - `{feat}` (p={pval:.3f}){non_stationary_warning}\n"
                    significant_granger.append(feat)
                summary_text += "- Note: Granger causality suggests predictive power, not true causation; non-stationarity can affect validity.\n"
            else:
                summary_text += "- No features showed significant Granger causality for the target (p<=0.05) at the tested lag.\n"
            key_findings_dict['significant_granger_features'] = significant_granger
        else:
            summary_text += "- Granger causality analysis was not performed or yielded no results.\n"
            key_findings_dict['significant_granger_features'] = []


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

        # Add summary findings dict to main results dict
        self.results_dict['analysis_highlights'] = key_findings_dict


    def write_summary_file(self, summary_filename="summary_findings_detailed.txt"):
        """Write a detailed summary file accessing data from self.results_dict."""
        filepath = os.path.join(self.output_dir, summary_filename)
        print(f"Attempting to write detailed summary to: {filepath}")

        # Retrieve potentially large DataFrames/Series from results_dict
        combined_imp = self.results_dict.get('combined_imp')
        condition_importance_df = self.results_dict.get('condition_importance_df')
        h_values = self.results_dict.get('h_values')
        interaction_stability = self.results_dict.get('interaction_stability', {})
        h_3way_df = self.results_dict.get('threeway_interactions_h') # This should be the DataFrame now
        linear_perf = self.results_dict.get('linear_model_comparison', {})
        rulefit_rules_df = self.results_dict.get('rulefit_top_rules') # Get the top rules list/dict
        condition_linear_results = self.results_dict.get('condition_linear_results')
        condition_lofo_df = self.results_dict.get('condition_lofo_importance')

        with open(filepath, 'w') as f:
            f.write(f"--- Summary of Key Findings ({datetime.now().strftime('%Y-%m-%d %H:%M')}) ---\n")
            f.write(f"--- Output Directory: {self.output_dir} ---\n\n")

            # --- Feature Importance ---
            f.write("--- Feature Importance Rankings (Combined) ---\n")
            if combined_imp is not None and isinstance(combined_imp, pd.DataFrame) and not combined_imp.empty:
                 # Ensure combined_imp is DataFrame before using .to_string()
                 rank_cols = [col for col in combined_imp.columns if '_rank' in col]
                 cols_to_show = ['mean_rank'] + rank_cols if 'mean_rank' in combined_imp.columns and rank_cols else list(combined_imp.columns)
                 try:
                      f.write(combined_imp[cols_to_show].to_string())
                      f.write("\n(Lower rank is better)\n\n")
                 except Exception as e:
                      f.write(f"(Error converting combined importance to string: {e})\n\n")
            else:
                f.write("No combined importance scores available or not a DataFrame.\n\n")

            # --- Condition Importance (Permutation) ---
            f.write("--- Top Rule Conditions by Importance (Permutation) ---\n")
            if condition_importance_df is not None and isinstance(condition_importance_df, pd.DataFrame) and not condition_importance_df.empty:
                 try:
                      cols_to_show = ['Original_Condition', 'Importance']
                      existing_cols = [c for c in cols_to_show if c in condition_importance_df.columns]
                      f.write(condition_importance_df[existing_cols].head(20).to_string(index=False))
                      f.write("\n\n")
                 except Exception as e:
                      f.write(f"(Error converting condition importance to string: {e})\n\n")
            else:
                f.write("No condition importance (permutation) calculated or not a DataFrame.\n\n")

            # --- Pairwise Interactions (H-statistic) ---
            f.write("--- Top Feature Interactions (Approx. H-statistic) ---\n")
            if h_values is not None and isinstance(h_values, pd.Series) and not h_values.empty:
                 try:
                      f.write(h_values.head(20).to_string())
                      f.write("\n\n")
                 except Exception as e:
                      f.write(f"(Error converting H-values to string: {e})\n\n")
            else:
                f.write("No valid H-statistic interactions calculated or not a Series.\n\n")

            # --- Interaction Stability ---
            f.write("--- Interaction Stability (Mean +/- Std Dev H-statistic) ---\n")
            if interaction_stability:
                for pair, stats in interaction_stability.items():
                     # Ensure pair is tuple for string conversion
                     pair_str = '__x__'.join(map(str, pair)) if isinstance(pair, tuple) else str(pair)
                     mean_val = stats.get('mean', np.nan)
                     std_val = stats.get('std', np.nan)
                     f.write(f"  {pair_str}: {mean_val:.3f} +/- {std_val:.3f}\n")
                f.write("\n")
            else:
                 f.write("Interaction stability not calculated or empty.\n\n")

            # --- 3-Way Interactions ---
            f.write("--- Approx. 3-Way Interaction (H-statistic for Top Triplets) ---\n")
            if h_3way_df is not None: # h_3way_df is now list of dicts from results
                if isinstance(h_3way_df, list) and h_3way_df:
                    # Sort by H_statistic if possible
                    try:
                         h_3way_df_sorted = sorted(h_3way_df, key=lambda x: x.get('H_statistic', -np.inf), reverse=True)
                    except:
                         h_3way_df_sorted = h_3way_df # Keep original if sort fails

                    for item in h_3way_df_sorted:
                        triplet = item.get('Triplet', ('N/A','N/A','N/A'))
                        h_stat = item.get('H_statistic', np.nan)
                        f.write(f"  Triplet {' x '.join(map(str, triplet))}: H_statistic = {h_stat:.4f}\n")
                    f.write("\n")
                elif isinstance(h_3way_df, pd.DataFrame) and not h_3way_df.empty:
                     # If it's still a DataFrame somehow
                     try:
                          f.write(h_3way_df[['Triplet', 'H_statistic']].to_string(index=False))
                          f.write("\n\n")
                     except Exception as e:
                          f.write(f"(Error converting 3-way H DataFrame to string: {e})\n\n")
                else:
                     f.write("No valid 3-way interaction results calculated.\n\n")
            else:
                f.write("No 3-way interaction results available.\n\n")

            # --- Linear Model Comparison ---
            f.write("--- Linear Model Comparison ---\n")
            rmse_orig = linear_perf.get('rmse_orig')
            r2_orig = linear_perf.get('r2_orig')
            rmse_eng = linear_perf.get('rmse_eng')
            r2_eng = linear_perf.get('r2_eng')
            actual_eng_features = self.results_dict.get('actual_eng_features', []) # Get from main dict

            if actual_eng_features:
                 f.write(f"Engineered Features Added ({len(actual_eng_features)}): {actual_eng_features}\n")
            else:
                 f.write("(No engineered features added or used)\n")

            if rmse_orig is not None and r2_orig is not None:
                 f.write(f"Linear Model (Original): Test RMSE={rmse_orig:.4f}, R2={r2_orig:.4f}\n")
            else:
                 f.write("Linear Model (Original): Results not available.\n")

            if actual_eng_features and rmse_eng is not None and r2_eng is not None:
                 f.write(f"Linear Model (Enginrd.): Test RMSE={rmse_eng:.4f}, R2={r2_eng:.4f}\n")
            elif actual_eng_features:
                 f.write("Linear Model (Enginrd.): Results not available (check steps).\n")
            f.write("\n")

            # --- RuleFit Rules ---
            f.write("--- Top RuleFit Rules/Features by Importance ---\n")
            if rulefit_rules_df: # This is likely a list of dicts now
                if isinstance(rulefit_rules_df, list) and rulefit_rules_df:
                     for rule_data in rulefit_rules_df: # Iterate through list
                          rule = rule_data.get('rule', 'N/A')
                          coef = rule_data.get('coef', np.nan)
                          support = rule_data.get('support', np.nan)
                          importance = rule_data.get('importance', np.nan)
                          rule_type = rule_data.get('type', 'N/A')
                          f.write(f"Type: {rule_type}, Rule: {rule}, Coef: {coef:.4f}, Support: {support:.3f}, Importance: {importance:.4f}\n")
                     f.write("\n")
                elif isinstance(rulefit_rules_df, pd.DataFrame) and not rulefit_rules_df.empty:
                      # Fallback if it's still a DataFrame
                     try:
                          f.write(rulefit_rules_df[['rule', 'coef', 'support', 'importance']].to_string(index=False))
                          f.write("\n\n")
                     except Exception as e:
                          f.write(f"(Error converting RuleFit DataFrame to string: {e})\n\n")
                else:
                     f.write("(RuleFit analysis not performed or yielded no rules).\n\n")
            else:
                f.write("(RuleFit analysis not performed or yielded no rules).\n\n")

            # --- Linear Models on Condition Features ---
            f.write("--- Linear Models Using ONLY Condition Features ---\n")
            if condition_linear_results is not None and isinstance(condition_linear_results, dict):
                 lr_res = condition_linear_results.get('linear', {})
                 ridge_res = condition_linear_results.get('ridge', {})
                 lasso_res = condition_linear_results.get('lasso', {})

                 f.write(f"Linear Regression: Test RMSE={lr_res.get('rmse', np.nan):.4f}, R2={lr_res.get('r2', np.nan):.4f}\n")
                 f.write(f"RidgeCV Regression: Test RMSE={ridge_res.get('rmse', np.nan):.4f}, R2={ridge_res.get('r2', np.nan):.4f}, Alpha={ridge_res.get('alpha', np.nan):.4f}\n")
                 f.write(f"LassoCV Regression: Test RMSE={lasso_res.get('rmse', np.nan):.4f}, R2={lasso_res.get('r2', np.nan):.4f}, Alpha={lasso_res.get('alpha', np.nan):.4f}\n\n")

                 # Write Coefficients (using stored dicts)
                 ridge_coefs = ridge_res.get('coefficients')
                 if ridge_coefs and isinstance(ridge_coefs, list):
                      f.write("Top Ridge Coefficients (Conditions Only):\n")
                      # Sort list of dicts by abs coefficient
                      ridge_coefs_sorted = sorted(ridge_coefs, key=lambda x: abs(x.get('Ridge_Coefficient', 0)), reverse=True)
                      for coef_data in ridge_coefs_sorted[:15]:
                           orig_cond = coef_data.get('Original_Condition', ('N/A','N/A'))
                           cond_str = f"{orig_cond[0]} {orig_cond[1]}" if isinstance(orig_cond, (list,tuple)) else "N/A"
                           f.write(f"  Condition: {cond_str}, Coefficient: {coef_data.get('Ridge_Coefficient', np.nan):.4f}\n")
                      f.write("\n")

                 lasso_coefs = lasso_res.get('coefficients')
                 if lasso_coefs and isinstance(lasso_coefs, list):
                      f.write("Top Lasso Coefficients (Conditions Only):\n")
                      lasso_coefs_sorted = sorted(lasso_coefs, key=lambda x: abs(x.get('Lasso_Coefficient', 0)), reverse=True)
                      lasso_zeros = sum(1 for item in lasso_coefs if np.isclose(item.get('Lasso_Coefficient', 1), 0))
                      for coef_data in lasso_coefs_sorted[:20]:
                           orig_cond = coef_data.get('Original_Condition', ('N/A','N/A'))
                           cond_str = f"{orig_cond[0]} {orig_cond[1]}" if isinstance(orig_cond, (list,tuple)) else "N/A"
                           is_zero = coef_data.get('Is_Zero', False)
                           f.write(f"  Condition: {cond_str}, Coefficient: {coef_data.get('Lasso_Coefficient', np.nan):.4f}, IsZero: {is_zero}\n")
                      f.write(f"(Number of Lasso coefficients shrunk to zero: {lasso_zeros})\n\n")
            else:
                f.write("(Not calculated, likely no condition features were generated or error occurred).\n\n")

            # --- Condition LOFO Importance ---
            f.write("--- Condition Feature Importance (LOFO) ---\n")
            if condition_lofo_df is not None and isinstance(condition_lofo_df, pd.DataFrame) and not condition_lofo_df.empty:
                 try:
                      # Select relevant columns if they exist
                      cols_to_write = ['Original_Condition', 'importance_mean', 'importance_std', 'feature']
                      existing_cols = [col for col in cols_to_write if col in condition_lofo_df.columns]
                      f.write(condition_lofo_df[existing_cols].to_string(index=False))
                      f.write("\n\n")
                 except Exception as e:
                      f.write(f"(Error converting Condition LOFO DataFrame to string: {e})\n\n")
            else:
                f.write("(LOFO analysis for conditions yielded no results or failed).\n\n")


        print(f"Successfully saved detailed findings to: {filepath}")


# %% [markdown]
# ## 4. Helper Function Definitions

# %%
def save_plot(fig, filename, output_dir=OUTPUT_DIR):
    """Saves plotly figure as HTML and static image, or matplotlib fig as png."""
    if fig is None:
        print(f"Skipping save for {filename}, figure is None.")
        return
    try:
        # Ensure filename has extension for identification, add default if missing
        base, ext = os.path.splitext(filename)
        if not ext:
            ext = '.png' # Default to png
            filename = base + ext

        filepath_base = os.path.join(output_dir, base)

        if isinstance(fig, go.Figure):
            filepath_html = filepath_base + ".html"
            filepath_png = filepath_base + ".png"
            fig.write_html(filepath_html)
            try:
                # Requires kaleido: pip install -U kaleido
                fig.write_image(filepath_png, scale=2) # Increase scale for better resolution
                print(f"Saved plot: {filepath_html} and {filepath_png}")
            except ValueError as e:
                print(f"Could not save static Plotly image {filepath_png}. Ensure kaleido is installed ('pip install -U kaleido'). Error: {e}")
                print(f"Saved plot (HTML only): {filepath_html}")
            except Exception as img_e:
                 print(f"Unexpected error saving static Plotly image {filepath_png}: {img_e}")
                 print(f"Saved plot (HTML only): {filepath_html}")

        elif isinstance(fig, plt.Figure):
             filepath_png = filepath_base + ".png"
             fig.savefig(filepath_png, bbox_inches='tight', dpi=150) # Increase dpi
             plt.close(fig) # Close plot after saving
             print(f"Saved plot: {filepath_png}")

        elif hasattr(fig, 'figure') and isinstance(fig.figure, plt.Figure):
             # Handle cases like seaborn plots where the figure is an attribute
             filepath_png = filepath_base + ".png"
             fig.figure.savefig(filepath_png, bbox_inches='tight', dpi=150)
             plt.close(fig.figure)
             print(f"Saved plot: {filepath_png}")

        elif isinstance(fig, PartialDependenceDisplay):
             # Handle Scikit-learn display objects
             filepath_png = filepath_base + ".png"
             fig.figure_.savefig(filepath_png, bbox_inches='tight', dpi=150)
             plt.close(fig.figure_)
             print(f"Saved plot: {filepath_png}")

        else:
            print(f"Warning: Unsupported figure type for saving: {type(fig)}. Attempting generic save.")
            try:
                 filepath_png = filepath_base + ".png"
                 plt.savefig(filepath_png, bbox_inches='tight', dpi=150)
                 plt.close() # Close the current figure context if possible
                 print(f"Saved plot (generic attempt): {filepath_png}")
            except Exception as generic_e:
                 print(f"Generic save failed for {filename}: {generic_e}")

    except Exception as e:
        print(f"Error saving plot {filename}: {e}")

# %%
def time_series_split_by_month(df, date_col_name, test_months):
    """Splits time series data based on the last N months using the DataFrame index."""
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a DatetimeIndex for time series split.")

    df = df.sort_index() # Ensure data is sorted by time index
    split_date = df.index.max() - pd.DateOffset(months=test_months)

    # Handle cases where the split date might be before the start date
    if split_date < df.index.min():
        print(f"Warning: Split date ({split_date}) is before the earliest data point ({df.index.min()}). Adjusting split.")
        # Option: Split roughly in the middle or use a fixed number of test points
        split_idx = int(len(df) * (1 - (test_months / (12*5)))) # Fallback: split based on proportion
        if split_idx <= 0: split_idx = len(df)//2 # Ensure some split
        split_date = df.index[split_idx]
        print(f"Adjusted split date to: {split_date}")


    train_df = df[df.index <= split_date]
    test_df = df[df.index > split_date]

    # Check for empty splits
    if train_df.empty:
        print("Warning: Training set is empty after time series split. Check test_months and data range.")
    if test_df.empty:
        print("Warning: Test set is empty after time series split. Check test_months and data range.")


    print(f"Time Series Split (using index):")
    if not train_df.empty:
        print(f"  Train range: {train_df.index.min()} to {train_df.index.max()} ({len(train_df)} rows)")
    else:
        print(f"  Train range: EMPTY")
    if not test_df.empty:
        print(f"  Test range:  {test_df.index.min()} to {test_df.index.max()} ({len(test_df)} rows)")
    else:
        print(f"  Test range: EMPTY")

    return train_df, test_df

# %%
def get_tree_rules_and_conditions(tree_model, feature_names):
    """Extracts conditions (feature-threshold pairs) from all trees in a RandomForest."""
    conditions = Counter()
    all_rules_text = []

    if not isinstance(tree_model, RandomForestRegressor) or not hasattr(tree_model, 'estimators_'):
        print("Rule/condition extraction currently implemented only for RandomForestRegressor.")
        return conditions, all_rules_text

    print(f"Extracting conditions from {len(tree_model.estimators_)} trees...")
    extracted_count = 0
    for i, tree in enumerate(tree_model.estimators_):
        if i == 0: # Export text for first tree only for example
            try:
                all_rules_text.append(export_text(tree, feature_names=feature_names))
            except Exception as e:
                print(f"Could not export text for tree {i}: {e}")

        try:
            tree_structure = tree.tree_
            if not tree_structure: continue # Skip if tree structure is invalid

            node_count = tree_structure.node_count
            for node_idx in range(node_count):
                # If it's a split node (not a leaf)
                if tree_structure.children_left[node_idx] != tree_structure.children_right[node_idx]:
                    feature_index = tree_structure.feature[node_idx]
                    threshold = tree_structure.threshold[node_idx]

                    # Validate feature index
                    if feature_index >= 0 and feature_index < len(feature_names):
                        feature = feature_names[feature_index]
                        # Store condition as a tuple (feature, condition_string)
                        # Use consistent formatting
                        condition_left = (feature, f"<={threshold:.4f}")
                        conditions[condition_left] += 1
                        condition_right = (feature, f"> {threshold:.4f}")
                        conditions[condition_right] += 1
                        extracted_count += 2
                    else:
                        # Handle invalid feature index if necessary, e.g., print a warning
                        # print(f"Warning: Invalid feature index {feature_index} in tree {i}, node {node_idx}")
                        pass
        except Exception as tree_e:
            print(f"Error processing tree {i}: {tree_e}")
            continue # Move to the next tree

    print(f"Finished extracting conditions. Total conditions counted: {extracted_count}")
    return conditions, all_rules_text

# %%
def friedman_h_statistic(model, X, feature1, feature2, grid_resolution=15, sample_size=None):
    """(Approximation) Calculates Friedman's H-statistic for pairwise interaction."""
    feature_names = X.columns.tolist()
    try:
        f1_idx = feature_names.index(feature1)
        f2_idx = feature_names.index(feature2)
    except ValueError as e:
        print(f"Error finding feature index for H-stat ({feature1}, {feature2}): {e}")
        return np.nan

    X_sample = X
    if sample_size is not None and sample_size < len(X):
        X_sample = X.sample(sample_size, random_state=RANDOM_STATE)

    try:
        # Create grid points for each feature
        features = [feature1, feature2]
        grid_points = {}
        for feature in features:
            x_vals = X_sample[feature].unique()
            if len(x_vals) > grid_resolution:
                # Use percentiles for grid points to handle skewed distributions better
                grid_points[feature] = np.percentile(X_sample[feature].dropna(), np.linspace(0, 100, grid_resolution))
            else:
                grid_points[feature] = np.sort(x_vals)
        
        # Calculate partial dependence for individual features (f1, f2)
        individual_pd = {}
        for feature, idx in [(feature1, f1_idx), (feature2, f2_idx)]:
            grid = grid_points[feature]
            pd_values = np.zeros_like(grid, dtype=float)
            X_temp_base = X_sample.copy()

            for i, value in enumerate(grid):
                X_temp = X_temp_base.copy()
                X_temp[feature] = value
                pd_values[i] = model.predict(X_temp).mean()
            individual_pd[feature] = pd_values

        # Calculate the sum of individual partial dependence effects (main effects centered)
        pd1_centered = individual_pd[feature1] - np.mean(individual_pd[feature1])
        pd2_centered = individual_pd[feature2] - np.mean(individual_pd[feature2])
        # Broadcasting pd1 across columns and pd2 across rows
        main_effects_grid = pd1_centered[:, np.newaxis] + pd2_centered

        # Calculate joint partial dependence (PDP over the 2D grid)
        grid_res1 = len(grid_points[feature1])
        grid_res2 = len(grid_points[feature2])
        joint_pd = np.zeros((grid_res1, grid_res2))
        X_temp_base_joint = X_sample.copy()

        for i, val1 in enumerate(grid_points[feature1]):
            for j, val2 in enumerate(grid_points[feature2]):
                X_temp = X_temp_base_joint.copy()
                X_temp[feature1] = val1
                X_temp[feature2] = val2
                joint_pd[i, j] = model.predict(X_temp).mean()

        # Calculate interaction effect (joint PDP minus centered main effects)
        joint_pd_centered = joint_pd - np.mean(joint_pd)
        interaction_effect = joint_pd_centered - main_effects_grid

        # Calculate H-statistic: Variance(Interaction) / Variance(Joint Centered PDP)
        var_interaction = np.var(interaction_effect)
        var_joint_centered = np.var(joint_pd_centered)

        if var_joint_centered < 1e-10: # Avoid division by zero or near-zero
             h_stat = 0.0
        else:
             h_stat = var_interaction / var_joint_centered

        # H should be between 0 and 1, but numerical issues can occur
        return max(0.0, min(h_stat, 1.0))

    except Exception as e:
        # import traceback
        # print(f"Error calculating H-statistic for ({feature1}, {feature2}): {e}\n{traceback.format_exc()}")
        return np.nan # Return NaN if calculation fails

# %%
def friedman_h_3way(model, X, feature1, feature2, feature3, grid_resolution=8, sample_size=None):
    """(Approximation) Calculates Friedman's H-statistic for three-way interaction."""
    feature_names = X.columns.tolist()
    try:
        f1_idx = feature_names.index(feature1)
        f2_idx = feature_names.index(feature2)
        f3_idx = feature_names.index(feature3)
    except ValueError as e:
        print(f"Error finding feature index for 3-way H-stat ({feature1}, {feature2}, {feature3}): {e}")
        return np.nan

    X_sample = X
    if sample_size is not None and sample_size < len(X):
        X_sample = X.sample(sample_size, random_state=RANDOM_STATE)

    try:
        features = [feature1, feature2, feature3]
        grid_points = {}
        for feature in features:
             x_vals = X_sample[feature].unique()
             if len(x_vals) > grid_resolution:
                 grid_points[feature] = np.percentile(X_sample[feature].dropna(), np.linspace(0, 100, grid_resolution))
             else:
                 grid_points[feature] = np.sort(x_vals)

        # Calculate individual PDPs and center them
        individual_pd_centered = {}
        for feature in features:
            grid = grid_points[feature]
            pd_values = np.zeros_like(grid, dtype=float)
            X_temp_base = X_sample.copy()
            for i, value in enumerate(grid):
                X_temp = X_temp_base.copy()
                X_temp[feature] = value
                pd_values[i] = model.predict(X_temp).mean()
            individual_pd_centered[feature] = pd_values - np.mean(pd_values)

        # Calculate pairwise PDPs and center them
        pairwise_pd_centered = {}
        for pair in combinations(features, 2):
            f_a, f_b = pair
            grid_a = grid_points[f_a]
            grid_b = grid_points[f_b]
            pd_values = np.zeros((len(grid_a), len(grid_b)))
            X_temp_base_pair = X_sample.copy()
            for i, val_a in enumerate(grid_a):
                for j, val_b in enumerate(grid_b):
                    X_temp = X_temp_base_pair.copy()
                    X_temp[f_a] = val_a
                    X_temp[f_b] = val_b
                    pd_values[i, j] = model.predict(X_temp).mean()
            pairwise_pd_centered[pair] = pd_values - np.mean(pd_values)

        # Calculate three-way joint PDP and center it
        grid_res1 = len(grid_points[feature1])
        grid_res2 = len(grid_points[feature2])
        grid_res3 = len(grid_points[feature3])
        joint_pd_3way = np.zeros((grid_res1, grid_res2, grid_res3))
        X_temp_base_3way = X_sample.copy()
        for i, val1 in enumerate(grid_points[feature1]):
            for j, val2 in enumerate(grid_points[feature2]):
                for k, val3 in enumerate(grid_points[feature3]):
                    X_temp = X_temp_base_3way.copy()
                    X_temp[feature1] = val1
                    X_temp[feature2] = val2
                    X_temp[feature3] = val3
                    joint_pd_3way[i, j, k] = model.predict(X_temp).mean()
        joint_pd_3way_centered = joint_pd_3way - np.mean(joint_pd_3way)

        # Calculate sum of centered main effects and centered pairwise effects
        main_effects_sum = (individual_pd_centered[feature1][:, np.newaxis, np.newaxis] +
                            individual_pd_centered[feature2][np.newaxis, :, np.newaxis] +
                            individual_pd_centered[feature3][np.newaxis, np.newaxis, :])

        pairwise_effects_sum = (pairwise_pd_centered[(feature1, feature2)][:, :, np.newaxis] +
                                pairwise_pd_centered[(feature1, feature3)][:, np.newaxis, :] +
                                pairwise_pd_centered[(feature2, feature3)][np.newaxis, :, :])

        # Isolate pure three-way interaction effect
        # Interaction = Total - MainEffects - PairwiseInteractions (where PairwiseInteractions = PairwisePDP - MainEffects)
        # Interaction = TotalCentered - (Pairwise12Centered + Pairwise13Centered + Pairwise23Centered) - (Main1Centered + Main2Centered + Main3Centered) # Incorrect - leads to double counting
        # Correct Hiller approach: F_123 = F_1 + F_2 + F_3 + F_12 + F_13 + F_23 + F_123_interaction
        # => F_123_interaction = F_123 - (F_1+F_2+F_3) - (F_12_interaction + F_13_interaction + F_23_interaction)
        # where F_12_interaction = F_12 - (F_1 + F_2)
        # Substitute: F_123_interaction = F_123 - (F_1+F_2+F_3) - (F_12 - F_1 - F_2) - (F_13 - F_1 - F_3) - (F_23 - F_2 - F_3)
        # => F_123_interaction = F_123_centered - (F_12_centered - F_1_c - F_2_c) - (F_13_centered - F_1_c - F_3_c) - (F_23_centered - F_2_c - F_3_c) - (F_1_c + F_2_c + F_3_c)
        # => F_123_interaction = F_123_c - F_12_c - F_13_c - F_23_c + F_1_c + F_2_c + F_3_c # Simplified
        interaction_12 = pairwise_pd_centered[(feature1, feature2)] - (individual_pd_centered[feature1][:, np.newaxis] + individual_pd_centered[feature2])
        interaction_13 = pairwise_pd_centered[(feature1, feature3)] - (individual_pd_centered[feature1][:, np.newaxis] + individual_pd_centered[feature3])
        interaction_23 = pairwise_pd_centered[(feature2, feature3)] - (individual_pd_centered[feature2][:, np.newaxis] + individual_pd_centered[feature3])

        three_way_interaction_effect = (joint_pd_3way_centered -
                                        interaction_12[:, :, np.newaxis] -
                                        interaction_13[:, np.newaxis, :] -
                                        interaction_23[np.newaxis, :, :] -
                                        main_effects_sum)

        # Variance of the interaction effect
        var_interaction = np.var(three_way_interaction_effect)

        # Variance of the centered joint partial dependence
        var_joint_centered = np.var(joint_pd_3way_centered)

        if var_joint_centered < 1e-10:
             h_stat = 0.0
        else:
             h_stat = var_interaction / var_joint_centered

        return max(0.0, min(h_stat, 1.0))

    except Exception as e:
        # import traceback
        # print(f"Error calculating 3-way H-statistic for ({feature1}, {feature2}, {feature3}): {e}\n{traceback.format_exc()}")
        return np.nan

# %%
def create_interaction_pdp(surrogate_model, X_train, features, feature_names=None, grid_resolution=20):
    """Create detailed partial dependence interaction plots using Matplotlib."""
    if feature_names is None:
        feature_names = features
    if len(features) != 2:
        print("Error: create_interaction_pdp requires exactly two features.")
        return None, np.nan

    f1, f2 = features
    fn1, fn2 = feature_names # Feature names for labels

    # Calculate partial dependence manually
    try:
        # Create feature grid using percentiles for robustness
        feature_values = []
        for feature in features:
            unique_vals = X_train[feature].dropna().unique()
            if len(unique_vals) > grid_resolution:
                feature_vals = np.percentile(
                    X_train[feature].dropna(),
                    np.linspace(0, 100, grid_resolution)
                )
            else:
                feature_vals = np.sort(unique_vals)
            feature_values.append(feature_vals)

        x_values, y_values = np.meshgrid(feature_values[0], feature_values[1])
        grid_points = np.column_stack([x_values.ravel(), y_values.ravel()])

        # Calculate PDP values using predict on samples
        z_values = np.zeros(len(grid_points))
        # Use a sample of X_train for prediction if it's large, for efficiency
        sample_size_pdp = min(len(X_train), 1000)
        X_temp_base = X_train.sample(sample_size_pdp, random_state=RANDOM_STATE) if sample_size_pdp < len(X_train) else X_train.copy()

        for i, point in enumerate(grid_points):
            X_temp = X_temp_base.copy()
            X_temp[f1] = point[0]
            X_temp[f2] = point[1]
            z_values[i] = np.mean(surrogate_model.predict(X_temp))

        z_values = z_values.reshape(x_values.shape)

    except Exception as e:
        print(f"Error in PDP calculation for ({f1}, {f2}): {e}")
        return None, np.nan

    # Create 2x2 grid of visualizations using Matplotlib
    fig = plt.figure(figsize=(18, 16)) # Use plt.figure to manage the overall figure

    # 1. Contour plot
    ax1 = fig.add_subplot(2, 2, 1)
    contour = ax1.contourf(
        x_values, y_values, z_values,
        cmap='viridis', levels=15
    )
    ax1.set_xlabel(fn1)
    ax1.set_ylabel(fn2)
    ax1.set_title('PDP Interaction Contour')
    fig.colorbar(contour, ax=ax1, label='Predicted Value')

    # 2. 3D Surface plot
    ax_3d = fig.add_subplot(2, 2, 2, projection='3d')
    surface = ax_3d.plot_surface(
        x_values, y_values, z_values,
        cmap='viridis',
        edgecolor='none',
        alpha=0.9
    )
    ax_3d.set_xlabel(fn1)
    ax_3d.set_ylabel(fn2)
    ax_3d.set_zlabel('Predicted Value')
    ax_3d.set_title('PDP Interaction 3D Surface')
    fig.colorbar(surface, ax=ax_3d, shrink=0.5, label='Predicted Value', pad=0.1)

    # 3. Heatmap with annotations
    ax3 = fig.add_subplot(2, 2, 3)
    sns.heatmap(
        z_values,
        ax=ax3,
        cmap='viridis',
        annot=True, # Annotate cells
        fmt='.2f', # Format annotations
        linewidths=.5, # Add lines between cells
        annot_kws={"size": 8}, # Adjust annotation font size
        cbar=False, # Contour plot has colorbar
        xticklabels=np.round(feature_values[0], 2),
        yticklabels=np.round(feature_values[1], 2)
    )
    ax3.set_xlabel(fn1)
    ax3.set_ylabel(fn2)
    ax3.set_title('PDP Interaction Heatmap')
    ax3.tick_params(axis='x', rotation=45)
    ax3.tick_params(axis='y', rotation=0)

    # 4. Overlay with actual data points
    ax4 = fig.add_subplot(2, 2, 4)
    contour_overlay = ax4.contourf(
        x_values, y_values, z_values,
        cmap='viridis',
        levels=15,
        alpha=0.7
    )
    # Use a smaller sample for scatter plot if X_train is large
    scatter_sample_size = min(len(X_train), 500)
    scatter_data = X_train.sample(scatter_sample_size, random_state=RANDOM_STATE) if scatter_sample_size < len(X_train) else X_train
    scatter = ax4.scatter(
        scatter_data[f1],
        scatter_data[f2],
        c='white',
        edgecolor='black',
        alpha=0.5,
        s=20,
        label=f'Data Points (Sample N={scatter_sample_size})'
    )
    ax4.set_xlabel(fn1)
    ax4.set_ylabel(fn2)
    ax4.set_title('PDP with Data Distribution')
    fig.colorbar(contour_overlay, ax=ax4, label='Predicted Value')
    ax4.legend()

    # Calculate H-statistic approximation using the calculated PDP grids
    interaction_strength = np.nan
    try:
         # Re-use the friedman_h_statistic function logic on the calculated grids
         # This requires calculating individual PDPs on the same grid points
         individual_pd = {}
         for feature, grid_vals in zip(features, feature_values):
             pd_values = np.zeros_like(grid_vals, dtype=float)
             X_temp_base_h = X_train.sample(sample_size_pdp, random_state=RANDOM_STATE) if sample_size_pdp < len(X_train) else X_train.copy() # Use same sample as PDP
             for i, value in enumerate(grid_vals):
                 X_temp = X_temp_base_h.copy()
                 X_temp[feature] = value
                 pd_values[i] = surrogate_model.predict(X_temp).mean()
             individual_pd[feature] = pd_values

         pd1_centered = individual_pd[f1] - np.mean(individual_pd[f1])
         pd2_centered = individual_pd[f2] - np.mean(individual_pd[f2])
         main_effects_grid = pd1_centered[:, np.newaxis] + pd2_centered

         joint_pd_centered = z_values - np.mean(z_values) # z_values is the joint PDP grid
         interaction_effect = joint_pd_centered - main_effects_grid

         var_interaction = np.var(interaction_effect)
         var_joint_centered = np.var(joint_pd_centered)

         if var_joint_centered < 1e-10:
              interaction_strength = 0.0
         else:
              interaction_strength = var_interaction / var_joint_centered
         interaction_strength = max(0.0, min(interaction_strength, 1.0))

    except Exception as e:
        print(f"Error calculating interaction strength within PDP plot for ({f1}, {f2}): {e}")
        interaction_strength = np.nan

    # Set the main title including the H-statistic
    if not np.isnan(interaction_strength):
        title_h_stat = f" (H-stat Approx: {interaction_strength:.4f})"
    else:
        title_h_stat = " (H-stat Error)"
    fig.suptitle(f'PDP Interaction Analysis: {fn1} vs {fn2}{title_h_stat}', fontsize=16, y=1.02) # Adjust title position

    plt.tight_layout(rect=[0, 0.03, 1, 0.98]) # Adjust layout further if needed

    return fig, interaction_strength

# %%
def perform_granger_causality(data_features, data_target, feature_names, max_lag, test='ssr_ftest'):
    """Performs Granger causality tests for each feature predicting the target."""
    if not isinstance(data_target, pd.Series):
        raise TypeError("data_target must be a pandas Series.")
    if not isinstance(data_features, pd.DataFrame):
        raise TypeError("data_features must be a pandas DataFrame.")

    p_values = pd.Series(index=feature_names, dtype=float)
    target_name = data_target.name
    if target_name is None:
         target_name = 'target' # Assign default name if Series has no name
         data_target = data_target.rename(target_name)

    print(f"Performing Granger Causality tests (Feature -> {target_name}) up to lag {max_lag}...")

    # Combine for testing, ensuring index alignment
    df_test = pd.concat([data_target, data_features[feature_names]], axis=1).dropna()

    if df_test.empty:
        print("Warning: DataFrame is empty after combining features and target for Granger test. Returning NaNs.")
        p_values[:] = np.nan
        return p_values

    if len(df_test) <= max_lag * 2: # Heuristic check for sufficient data length
        print(f"Warning: Data length ({len(df_test)}) may be insufficient for Granger test with lag {max_lag}. Results might be unreliable.")

    for feature in feature_names:
        if feature == target_name: continue # Skip testing target against itself
        if feature not in df_test.columns:
            print(f"Warning: Feature '{feature}' not found in combined data for Granger test. Skipping.")
            p_values.loc[feature] = np.nan
            continue

        test_cols = [target_name, feature]
        try:
            # Check for constant series within the test subset
            if df_test[feature].nunique() <= 1 or df_test[target_name].nunique() <= 1:
                print(f"Warning: Constant series detected for ({feature} -> {target_name}). Setting p-value to NaN.")
                p_values.loc[feature] = np.nan
                continue

            # Perform the test
            # Ensure data length is sufficient for the model order (number of lags * number of variables^2)
            # This is a check within statsmodels, but good to be aware of.
            test_result = grangercausalitytests(df_test[test_cols], maxlag=[max_lag], verbose=False) # Test only max lag

            # Extract p-value for the specified test at the max lag
            # test_result is now a dict keyed by lag
            if max_lag in test_result and test in test_result[max_lag][0]:
                 p_val = test_result[max_lag][0][test][1]
                 p_values.loc[feature] = p_val
            else:
                 print(f"Warning: Could not find result for lag {max_lag} and test '{test}' for ({feature} -> {target_name})")
                 p_values.loc[feature] = np.nan
        except ValueError as ve:
             # Catch specific errors like insufficient data length for VAR
             print(f"ValueError during Granger test for ({feature} -> {target_name}): {ve}. Setting p-value to NaN.")
             p_values.loc[feature] = np.nan
        except Exception as e:
            print(f"Unexpected error running Granger test for ({feature} -> {target_name}): {e}")
            p_values.loc[feature] = np.nan

    print("Granger Causality tests complete.")
    return p_values

# %% [markdown]
# ## 5. Initialization

# %%
# Instantiate Report Generator
report_generator = ReportGenerator(output_dir=OUTPUT_DIR, surrogate_model_type=SURROGATE_MODEL_TYPE)

# Initialize result placeholders
# These will be populated during the analysis steps
rulefit_rules_df = None
condition_importance_df = pd.DataFrame()
condition_mdi_df = pd.DataFrame()
condition_shap_df = pd.DataFrame()
condition_lofo_df = pd.DataFrame() # Added for LOFO on conditions
condition_linear_results = None
combined_imp = pd.DataFrame()
perm_df = None # Permutation importance df
lofo_df = None # LOFO importance df (renamed from importance_df for clarity)
shap_df = None
mdi_df = None
shap_interaction_df = None
h_values = None # Pairwise H-stats Series
interaction_stability = {}
h_3way_df = None # 3-way H-stats DataFrame
granger_p_values = None
stationarity_results = {}
linear_perf = {} # For original vs engineered linear model comparison
engineered_feature_names = []
shap_values = None # To store raw SHAP values object
explainer = None # To store SHAP explainer

# %% [markdown]
# ## 6. Data Loading and Preparation

# %% [markdown]
# ### 6.1 Load Data

# %%
print("\n--- 1. Loading Data ---")
try:
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded data: {df.shape[0]} rows, {df.shape[1]} columns")
    print("Initial columns:", df.columns.tolist())
    print(df.head())
except FileNotFoundError:
    print(f"Error: Data file not found at {DATA_PATH}")
    exit()
except Exception as e:
    print(f"Error loading data: {e}")
    exit()

# %% [markdown]
# ### 6.2 Prepare Data

# %%
print("\n--- 2. Preparing Data ---")
# a. Time Index
try:
    if DATE_COLUMN not in df.columns:
        raise ValueError(f"Date column '{DATE_COLUMN}' not found in the dataset.")
    # Attempt to infer format, provide specific format if needed
    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN], errors='coerce') # Coerce errors to NaT
    df.dropna(subset=[DATE_COLUMN], inplace=True) # Drop rows where date conversion failed
    df = df.set_index(DATE_COLUMN)
    df = df.sort_index()
    print(f"Set '{DATE_COLUMN}' as time index.")
    print(f"Index type: {type(df.index)}")
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
        df_original_size = df.shape[0]
        df_filtered = df[df[GROUPING_COLUMN] == FILTER_VALUE].copy()
        if df_filtered.empty:
            print(f"Warning: Filtering resulted in an empty DataFrame. Proceeding with unfiltered data.")
            # Optionally exit or proceed with the full dataset
            # df = df # Keep original df (no filtering applied)
        else:
            df = df_filtered
            print(f"Data filtered: {df.shape[0]} rows remaining (from {df_original_size}).")
            # Drop the grouping column ONLY if it's now constant
            if df[GROUPING_COLUMN].nunique() == 1:
                 df = df.drop(columns=[GROUPING_COLUMN])
                 print(f"Dropped constant column '{GROUPING_COLUMN}' after filtering.")
            else:
                 print(f"Keeping column '{GROUPING_COLUMN}' as it's not constant after filtering.")
    else:
        print(f"Warning: Grouping column '{GROUPING_COLUMN}' not found. Skipping filtering.")

# c. Define Target and Features
if TARGET_VARIABLE not in df.columns:
    print(f"Error: Target variable '{TARGET_VARIABLE}' not found in the DataFrame columns: {df.columns.tolist()}")
    exit()

# Drop rows with NaN in target variable BEFORE splitting
initial_rows = df.shape[0]
df.dropna(subset=[TARGET_VARIABLE], inplace=True)
print(f"Dropped {initial_rows - df.shape[0]} rows with NaN target: {df.shape[0]} rows remaining.")

# Identify features (numeric only for most models here)
features = df.select_dtypes(include=np.number).columns.tolist()
if TARGET_VARIABLE in features:
    features.remove(TARGET_VARIABLE)
else:
    print(f"Warning: Target variable '{TARGET_VARIABLE}' not found in *numeric* columns. Ensure it's the correct type or handle non-numeric features appropriately.")
    # Check if it exists at all
    if TARGET_VARIABLE not in df.columns:
        print(f"CRITICAL Error: Target variable '{TARGET_VARIABLE}' does not exist in the DataFrame.")
        exit()

print(f"Identified Features ({len(features)}): {features}")
print(f"Target Variable: {TARGET_VARIABLE}")
# Set feature_names and target_name in the report generator instance
report_generator.feature_names = features
report_generator.target_name = TARGET_VARIABLE


# d. Time Series Split
try:
    train_df, test_df = time_series_split_by_month(df, DATE_COLUMN, TEST_MONTHS)
except ValueError as ve:
     print(f"Error during time series split: {ve}")
     exit()
except Exception as e:
     print(f"Unexpected error during time series split: {e}")
     exit()

# Check for empty dataframes after split
if train_df.empty or test_df.empty:
    print("Error: Train or Test DataFrame is empty after split. Cannot proceed.")
    exit()

# Define X_train, X_test, y_train, y_test
# Ensure only selected features are used
X_train, y_train = train_df[features], train_df[TARGET_VARIABLE]
X_test, y_test = test_df[features], test_df[TARGET_VARIABLE]

# e. Handle potential NaNs in features (impute AFTER splitting)
print("Handling NaNs in features (using mean imputation)...")
imputation_values = {}
for col in features:
    if X_train[col].isnull().any():
        mean_val = X_train[col].mean()
        imputation_values[col] = mean_val # Store for test set
        X_train.loc[:, col] = X_train[col].fillna(mean_val)
        print(f"  Imputed NaNs in TRAIN feature: {col} with mean {mean_val:.4f}")
    else:
         imputation_values[col] = X_train[col].mean() # Store mean even if no NaNs in train

    # Impute test set using TRAINING data mean
    if col in X_test.columns and X_test[col].isnull().any():
        mean_val_train = imputation_values[col]
        X_test.loc[:, col] = X_test[col].fillna(mean_val_train)
        print(f"  Imputed NaNs in TEST feature: {col} with TRAIN mean {mean_val_train:.4f}")

# Final check for NaNs
if X_train.isnull().any().any() or X_test.isnull().any().any():
    print("Warning: NaNs still present in features after imputation.")
    print("X_train NaNs:\n", X_train.isnull().sum()[X_train.isnull().sum() > 0])
    print("X_test NaNs:\n", X_test.isnull().sum()[X_test.isnull().sum() > 0])

# Store final data splits in report generator (optional, for potential reuse in methods)
report_generator.X_train = X_train
report_generator.y_train = y_train
report_generator.X_test = X_test
report_generator.y_test = y_test

print(f"Train shapes: X={X_train.shape}, y={y_train.shape}")
print(f"Test shapes:  X={X_test.shape}, y={y_test.shape}")

# %% [markdown]
# ### 6.3 Add Dataset Overview to Report

# %%
report_generator.add_dataset_overview(
    df_shape=df.shape, # Use shape of df AFTER filtering and NaN drop
    features_list=features,
    date_range=(df.index.min(), df.index.max())
)

# %% [markdown]
# ## 7. Surrogate Model Training & Evaluation

# %%
print("\n--- 4. Surrogate Model Training ---")
print(f"Training {SURROGATE_MODEL_TYPE} model...")

if SURROGATE_MODEL_TYPE == 'random_forest':
    # Adjusted RF hyperparameters to reduce potential overfitting
    surrogate_model = RandomForestRegressor(
        n_estimators=100,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        max_depth=6,          # Slightly deeper than previous attempt
        min_samples_leaf=10,  # Minimum samples at a leaf node
        max_features='sqrt',  # Consider sqrt or log2 of features at each split
        oob_score=True        # Enable Out-of-Bag score for generalization estimate
    )
elif SURROGATE_MODEL_TYPE == 'xgboost':
    surrogate_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=100,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        max_depth=5,          # Max depth of trees
        learning_rate=0.1,    # Step size shrinkage
        subsample=0.8,        # Fraction of samples used per tree
        colsample_bytree=0.8  # Fraction of features used per tree
    )
else:
    raise ValueError("Invalid SURROGATE_MODEL_TYPE. Choose 'xgboost' or 'random_forest'.")

try:
    surrogate_model.fit(X_train, y_train)
except Exception as fit_e:
    print(f"Error fitting surrogate model: {fit_e}")
    exit()

print("Evaluating surrogate model...")
y_pred_train_surrogate = surrogate_model.predict(X_train)
y_pred_test_surrogate = surrogate_model.predict(X_test)

train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train_surrogate))
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test_surrogate))
train_r2 = r2_score(y_train, y_pred_train_surrogate)
test_r2 = r2_score(y_test, y_pred_test_surrogate)

print(f"Surrogate Model Performance:")
print(f"  Train RMSE: {train_rmse:.4f}, R2: {train_r2:.4f}")
print(f"  Test RMSE:  {test_rmse:.4f}, R2: {test_r2:.4f}")
if hasattr(surrogate_model, 'oob_score_') and surrogate_model.oob_score_:
     print(f"  OOB R2 Score: {surrogate_model.oob_score_:.4f}") # OOB score for RF

# Add Model Performance to Report
report_generator.add_surrogate_model_performance(
    model_type=SURROGATE_MODEL_TYPE,
    train_rmse=train_rmse,
    train_r2=train_r2,
    test_rmse=test_rmse,
    test_r2=test_r2
)

# %% [markdown]
# ## 8. Surrogate Model Interpretation - Rules & Thresholds

# %% [markdown]
# ### 8.1 Decision Tree Visualization & Condition Extraction (Random Forest)

# %%
print("\n--- 5. Surrogate Model Interpretation - Rules & Thresholds ---")
rule_conditions = Counter() # Initialize
example_rules_text = [] # Initialize

if SURROGATE_MODEL_TYPE == 'random_forest':
    print("Visualizing sample Decision Trees & Extracting Rule Conditions...")
    if hasattr(surrogate_model, 'estimators_') and len(surrogate_model.estimators_) > 0:
        # Visualize first few trees (controlled by N_TREES_TO_VISUALIZE)
        for i in range(min(N_TREES_TO_VISUALIZE, len(surrogate_model.estimators_))):
            try:
                plt.figure(figsize=(25, 15)) # Larger figure size
                plot_tree(surrogate_model.estimators_[i],
                            feature_names=features,
                            filled=True, rounded=True, precision=2,
                            max_depth=3, fontsize=8) # Limit depth for viz, smaller font
                plt.title(f'Decision Tree {i} from Random Forest (Max Depth 3)')
                tree_plot_fig = plt.gcf()
                save_plot(tree_plot_fig, f"surrogate_tree_{i}_viz", output_dir=OUTPUT_DIR)
            except Exception as e:
                print(f"Could not visualize tree {i}: {e}")

        # Extract rules and conditions from all trees
        rule_conditions, example_rules_text = get_tree_rules_and_conditions(surrogate_model, features)

        if example_rules_text:
            print("\nExample Tree Rules (Tree 0 - First 1000 chars):")
            print(example_rules_text[0][:1000] + "...")
            rules_filename = os.path.join(OUTPUT_DIR, "surrogate_tree_0_rules.txt")
            try:
                with open(rules_filename, "w") as f:
                    f.write(example_rules_text[0])
                print(f"Saved full rules for tree 0: {rules_filename}")
            except Exception as e:
                print(f"Error saving tree 0 rules: {e}")
    else:
         print("Random Forest model has no estimators to visualize or extract rules from.")

elif SURROGATE_MODEL_TYPE == 'xgboost':
    print("Visualizing XGBoost Tree (requires graphviz)...")
    try:
        import graphviz # Check import here
        booster = surrogate_model.get_booster()
        for i in range(min(N_TREES_TO_VISUALIZE, booster.num_boosted_rounds())):
             try:
                  dot_data = xgb.to_graphviz(booster, num_trees=i)
                  graph_filename = os.path.join(OUTPUT_DIR, f"surrogate_xgb_tree_{i}")
                  dot_data.render(graph_filename, view=False, format='png', cleanup=True) # Add cleanup
                  print(f"Saved XGBoost tree plot: {graph_filename}.png")
             except Exception as plot_e:
                  print(f"Could not plot XGBoost tree {i}: {plot_e}")
    except ImportError:
        print("Graphviz library not found. Skipping XGBoost tree visualization.")
        print("Install it via pip install python-graphviz and ensure the graphviz binaries are in your system PATH.")
    except Exception as e:
        print(f"Error preparing for XGBoost tree visualization: {e}")
    print("Rule/condition extraction (counting thresholds) currently implemented only for RandomForest.")

# %% [markdown]
# ### 8.2 Feature Threshold Analysis (Random Forest)

# %%
if SURROGATE_MODEL_TYPE == 'random_forest' and rule_conditions:
    print("\nAnalyzing Feature Thresholds (Unique per feature from Condition Analysis)...")
    feature_thresholds = {}
    parsed_conditions = 0
    for condition_tuple, count in rule_conditions.items():
        # Ensure it's a tuple of format (feature_name, condition_string)
        if isinstance(condition_tuple, tuple) and len(condition_tuple) == 2:
            feature, condition_str = condition_tuple
            try:
                # Extract threshold robustly (handle <= and >)
                parts = condition_str.replace('<=', ' ').replace('>', ' ').split()
                if len(parts) > 0:
                    threshold = float(parts[-1])
                    if feature not in feature_thresholds:
                        feature_thresholds[feature] = set()
                    feature_thresholds[feature].add(threshold)
                    parsed_conditions += 1
            except ValueError:
                # print(f"Warning: Could not parse threshold from condition string: {condition_str}")
                continue # Skip if threshold parsing fails
            except Exception as parse_e:
                 print(f"Warning: Error parsing condition {condition_tuple}: {parse_e}")
                 continue
        # else:
            # print(f"Warning: Skipping condition with unexpected format: {condition_tuple}")

    print(f"Successfully parsed {parsed_conditions} conditions to extract thresholds.")

    threshold_counts = {f: len(t_set) for f, t_set in feature_thresholds.items() if t_set}
    if threshold_counts:
        threshold_df = pd.DataFrame.from_dict(threshold_counts, orient='index', columns=['Count'])
        threshold_df = threshold_df.sort_values('Count', ascending=False)
        fig_thresh = px.bar(threshold_df, x=threshold_df.index, y='Count',
                            title='Number of Unique Decision Thresholds per Feature (Random Forest)',
                            labels={'x': 'Feature', 'Count': 'Number of Unique Thresholds'})
        fig_thresh.update_layout(xaxis_tickangle=-45)
        save_plot(fig_thresh, "surrogate_threshold_counts", output_dir=OUTPUT_DIR)
        print("Unique Threshold Counts per Feature (Top 15):")
        print(threshold_df.head(15))
        # Store in results
        report_generator.results_dict['threshold_counts'] = threshold_df.to_dict()['Count']
    else:
        print("No valid thresholds extracted for counting.")
        report_generator.results_dict['threshold_counts'] = {}
elif SURROGATE_MODEL_TYPE == 'random_forest':
     print("Rule conditions not available, skipping threshold analysis.")

# %% [markdown]
# ## 9. Feature Importance Analysis

# %% [markdown]
# ### 9.1 Permutation Importance (Train Set)

# %%
print("\n--- 6a. Calculating Permutation Importance (Train Set) ---")
perm_df = None # Initialize
try:
    print("Calculating permutation importance (this may take a while)...")
    # Calculate on TRAINING data
    perm_importance = permutation_importance(
        surrogate_model,
        X_train, y_train,
        n_repeats=10,        # Number of times to permute each feature
        random_state=RANDOM_STATE,
        n_jobs=-1,           # Use all available cores
        scoring='neg_root_mean_squared_error' # Use RMSE drop as score (higher is better)
    )
    # Importance score is the drop in score, so higher means more important
    # We negate it if using neg_rmse so positive score drop = positive importance
    perm_imp_means = perm_importance.importances_mean
    perm_imp_std = perm_importance.importances_std

    perm_sorted_idx = perm_imp_means.argsort()[::-1] # Sort descending

    perm_df = pd.DataFrame({
        'feature': X_train.columns[perm_sorted_idx],
        'importance_mean': perm_imp_means[perm_sorted_idx],
        'importance_std': perm_imp_std[perm_sorted_idx]
    })

    # Plotting
    plot_df_perm = perm_df.head(N_TOP_FEATURES).sort_values('importance_mean', ascending=True) # Ascending for horizontal bar
    fig_perm = px.bar(plot_df_perm,
                      x='importance_mean', y='feature', orientation='h',
                      error_x='importance_std',
                      title=f'Top {N_TOP_FEATURES} Permutation Feature Importance (Train Set, Higher is Better)',
                      labels={'importance_mean': 'Mean Importance (Score Drop)', 'feature': 'Feature'})
    save_plot(fig_perm, "featimp_permutation_train", output_dir=OUTPUT_DIR)
    print("Permutation Importance (Train Set - Top 10):")
    print(perm_df.head(N_TOP_FEATURES))

except Exception as e:
    print(f"Error calculating Permutation Importance: {e}")
    perm_df = pd.DataFrame() # Ensure it's an empty DF on error

# Store results
report_generator.results_dict['permutation_importance_train'] = perm_df

# %% [markdown]
# ### 9.2 LOFO Importance (Train Set)

# %%
print("\n--- 6b. Calculating LOFO Importance (Train Set) ---")
lofo_df = None # Initialize
try:
    print("Setting up LOFO dataset...")
    # LOFO needs a defined CV scheme. Using TimeSeriesSplit for temporal data.
    # Ensure enough splits are possible given train data size
    n_splits_lofo = min(5, len(X_train) // 2) if len(X_train) >= 4 else 2 # Basic check
    if n_splits_lofo < 2:
        print("Warning: Train data too small for reliable TimeSeriesSplit in LOFO. Skipping.")
        raise ValueError("Insufficient data for LOFO CV.")

    cv_lofo = TimeSeriesSplit(n_splits=n_splits_lofo)
    # Combine X_train and y_train for LOFO Dataset input
    lofo_input_df = pd.concat([X_train, y_train], axis=1)
    lofo_dataset = Dataset(df=lofo_input_df, target=TARGET_VARIABLE, features=features)

    print(f"Running LOFOImportance with {n_splits_lofo} time series splits...")
    # Use a cloned model to avoid modifying the original
    lofo_model = clone(surrogate_model)
    lofo_imp = LOFOImportance(lofo_dataset, model=lofo_model, cv=cv_lofo, scoring='r2') # Use R2 score
    lofo_df_raw = lofo_imp.get_importance()
    print(f"LOFO Raw Results:\n{lofo_df_raw}")

    # Process LOFO results (find the right importance column, 'importance_mean' preferred)
    imp_col_name = None
    expected_cols = ['importance_mean', 'importance', 'val_imp_mean']
    for col in expected_cols:
        if col in lofo_df_raw.columns:
            imp_col_name = col
            break

    if imp_col_name:
        print(f"Using LOFO importance column: '{imp_col_name}'")
        # Rename the column to 'importance_mean' for consistency
        lofo_df = lofo_df_raw[['feature', imp_col_name]].rename(columns={imp_col_name: 'importance_mean'})
        lofo_df = lofo_df.sort_values('importance_mean', ascending=False)

        # Plotting using LOFO's built-in plotter
        try:
            print("Generating LOFO plot...")
            plot_importance(lofo_df_raw, figsize=(10, max(6, len(features)//2))) # Use raw df for plot function
            fig_lofo = plt.gcf()
            fig_lofo.suptitle('LOFO Feature Importance (Train Set, Higher is Better)', y=1.02)
            plt.tight_layout(rect=[0, 0.03, 1, 0.98])
            save_plot(fig_lofo, "featimp_lofo", output_dir=OUTPUT_DIR)
            print("LOFO Importance (Train Set - Top 10):")
            print(lofo_df.head(N_TOP_FEATURES))
        except Exception as plot_e:
             print(f"Error generating LOFO plot: {plot_e}")
    else:
        print("Could not identify a suitable importance column in LOFO results. Skipping plots.")
        lofo_df = pd.DataFrame() # Empty DF if processing failed

except ImportError:
    print("LOFO library not found (pip install lofo-importance). Skipping LOFO.")
    lofo_df = pd.DataFrame()
except ValueError as ve: # Catch specific value errors like insufficient data
     print(f"ValueError during LOFO setup/calculation: {ve}")
     lofo_df = pd.DataFrame()
except Exception as e:
    import traceback
    print(f"Error during LOFO calculation/processing: {e}")
    # print(traceback.format_exc()) # Uncomment for detailed traceback
    lofo_df = pd.DataFrame()

# Store results
report_generator.results_dict['lofo_importance_train'] = lofo_df

# %% [markdown]
# ### 9.3 RuleFit Analysis (Train Set)

# %%
print("\n--- 6c. Rule Extraction with imodels (RuleFit) ---")
rulefit_rules_df = None # Initialize
try:
    print("Training RuleFitRegressor model...")
    # Initialize and fit RuleFitRegressor
    # Consider adjusting hyperparameters like max_rules, tree_size, tree_depth etc.
    # Using default hyperparameters first. Ensure data is numpy array.
    rulefit = RuleFitRegressor(random_state=RANDOM_STATE, max_rules=50) # Limit max rules
    # Convert to numpy, handle potential errors
    try:
         X_train_np = X_train.values
         y_train_np = y_train.values
    except AttributeError as ae:
         print(f"Error converting data to numpy array for RuleFit: {ae}")
         raise # Re-raise error to be caught by outer try-except

    rulefit.fit(X_train_np, y_train_np, feature_names=features)

    print("Extracting rules from RuleFit...")
    rules_list_from_rulefit = rulefit.rules_ # This attribute holds rules and linear terms

    if not rules_list_from_rulefit:
         print("RuleFit model did not generate any rules or linear terms.")
         rulefit_rules_df = pd.DataFrame() # Use empty DF
    else:
        print("Converting RuleFit rules list to DataFrame...")
        try:
            # Extract relevant attributes, handle potential missing ones
            rules_data = []
            for r in rules_list_from_rulefit:
                 rule_dict = {
                      'rule': getattr(r, 'rule', 'N/A'),
                      'coef': getattr(r, 'coef', np.nan),
                      'support': getattr(r, 'support', np.nan),
                      'type': getattr(r, 'type', 'unknown'),
                      'feature': getattr(r, 'feature', 'N/A') # Added for linear terms
                 }
                 rules_data.append(rule_dict)

            rulefit_rules_df = pd.DataFrame(rules_data)
            # Filter out rules/terms with zero or NaN coefficients
            rulefit_rules_df = rulefit_rules_df.dropna(subset=['coef'])
            rulefit_rules_df = rulefit_rules_df[rulefit_rules_df['coef'] != 0]

            if not rulefit_rules_df.empty:
                print(f"Successfully converted {len(rulefit_rules_df)} rules/linear terms with non-zero coefficients.")
                # Calculate importance (absolute coefficient)
                rulefit_rules_df['importance'] = rulefit_rules_df['coef'].abs()
                rulefit_rules_df = rulefit_rules_df.sort_values('importance', ascending=False)

                print("\nTop 15 Rules/Features from RuleFitRegressor (Sorted by Importance):")
                print(rulefit_rules_df[['type', 'rule', 'coef', 'support', 'importance']].head(15))

                # Visualize top rule/feature importances
                plot_df_rf = rulefit_rules_df.head(25).sort_values('importance', ascending=True) # Plot top 25, sort asc for plot
                fig_rulefit = px.bar(plot_df_rf,
                                     x='coef', y='rule', orientation='h',
                                     title='Top 25 RuleFit Rule/Feature Importances (Coefficient Magnitude)',
                                     labels={'coef': 'Coefficient (Importance)', 'rule': 'Rule / Feature'},
                                     color='coef', # Color by coefficient value
                                     color_continuous_scale=px.colors.diverging.Picnic, # Use a diverging colorscale
                                     hover_data=['type', 'support', 'importance'])
                fig_rulefit.update_layout(yaxis={'tickfont': {'size': 8}}) # Adjust font size if needed
                save_plot(fig_rulefit, "rulefit_rule_importance", output_dir=OUTPUT_DIR)

            else:
                print("No rules/linear terms with non-zero coefficients found after conversion.")
                rulefit_rules_df = pd.DataFrame() # Ensure empty DF
        except Exception as ex:
            print(f"Error converting RuleFit rules list to DataFrame: {ex}")
            rulefit_rules_df = pd.DataFrame()

    # Add summary to report (even if empty)
    report_generator.add_rulefit_summary(rulefit_rules_df)

except ImportError:
    print("imodels library not found (pip install imodels). Skipping RuleFit analysis.")
    rulefit_rules_df = pd.DataFrame()
    report_generator.add_rulefit_summary(None) # Add section indicating skipped
except Exception as e:
    import traceback
    print(f"Error during RuleFit analysis: {e}")
    # print(traceback.format_exc()) # Uncomment for detailed traceback
    rulefit_rules_df = pd.DataFrame()
    report_generator.add_rulefit_summary(None) # Add section indicating error


# Store results (potentially empty DF)
report_generator.results_dict['rulefit_rules_df'] = rulefit_rules_df

# %% [markdown]
# ### 9.4 SHAP Importance & Interaction Analysis (Train Set)

# %%
print("\n--- 6d/e. Calculating SHAP Importance & Interactions (Train Set) ---")
shap_values = None # Raw SHAP values object
shap_df = None # Importance DataFrame
shap_interaction_values = None # Raw interaction values object
shap_interaction_df = None # Interaction matrix DataFrame
explainer = None # SHAP explainer object

try:
    print("Initializing SHAP explainer...")
    # Use TreeExplainer for tree-based models, check model type
    if SURROGATE_MODEL_TYPE in ['random_forest', 'xgboost']:
         # Using feature_perturbation='interventional' with background data (X_train) is generally preferred
         # Using masker=X_train provides the background dataset directly
         # Using check_additivity=False can prevent some errors with certain model versions/outputs
         explainer = shap.TreeExplainer(surrogate_model, X_train, feature_perturbation='interventional', model_output='raw') # Add check_additivity=False? Test first.
    else:
         print(f"SHAP TreeExplainer not suitable for model type {SURROGATE_MODEL_TYPE}. Skipping SHAP.")
         raise NotImplementedError("SHAP Explainer not configured for this model type.")

    print("Calculating SHAP values (this may take time)...")
    shap_values = explainer(X_train) # Pass X_train again for calculation with background data

    # --- SHAP Importance ---
    print("Processing SHAP importance...")
    shap_mean_abs = np.abs(shap_values.values).mean(axis=0)
    shap_df = pd.DataFrame({'feature': X_train.columns, 'shap_importance': shap_mean_abs})
    shap_df = shap_df.sort_values('shap_importance', ascending=False)

    # Bar plot
    plot_df_shap = shap_df.head(N_TOP_FEATURES).sort_values('shap_importance', ascending=True)
    fig_shap_bar = px.bar(plot_df_shap, x='shap_importance', y='feature', orientation='h',
                           title=f'Top {N_TOP_FEATURES} SHAP Feature Importance (Mean Abs Value - Train Set)',
                           labels={'shap_importance': 'Mean |SHAP Value|', 'feature': 'Feature'})
    save_plot(fig_shap_bar, "featimp_shap_bar_train", output_dir=OUTPUT_DIR)
    print("SHAP Importance (Train Set - Top 10):")
    print(shap_df.head(N_TOP_FEATURES))

    # Summary plot (beeswarm)
    print("Generating SHAP Summary Plot (Beeswarm - Train Set)...")
    fig_shap_summary, ax_shap_summary = plt.subplots()
    shap.summary_plot(shap_values, X_train, plot_type="dot", show=False, sort=True)
    ax_shap_summary.set_title('SHAP Summary Plot (Beeswarm - Train Set)') # Set title via ax
    plt.tight_layout()
    save_plot(fig_shap_summary, "featimp_shap_summary_train", output_dir=OUTPUT_DIR)

    # --- SHAP Interaction Values ---
    print("Calculating SHAP Interaction Values (this can be very slow)...")
    shap_interaction_values = explainer.shap_interaction_values(X_train) # Calculate interactions

    if shap_interaction_values is not None:
         # Average absolute interaction values across samples
         # Shape is typically (N, F, F), average over N (axis=0)
         mean_abs_shap_inter = np.abs(shap_interaction_values).mean(axis=0)
         shap_interaction_df = pd.DataFrame(mean_abs_shap_inter, index=features, columns=features)
         print("SHAP Interaction Matrix (Mean Absolute Values):")
         print(shap_interaction_df.round(3).head()) # Print head of matrix

         # Plot heatmap of interactions
         plt.figure(figsize=(max(10, len(features)*0.6), max(8, len(features)*0.5)))
         sns.heatmap(shap_interaction_df, cmap="viridis", annot=True, fmt=".3f", linewidths=.5, annot_kws={"size": 8})
         plt.title('Mean Absolute SHAP Interaction Values')
         plt.xticks(rotation=45, ha='right')
         plt.yticks(rotation=0)
         plt.tight_layout()
         interaction_heatmap_fig = plt.gcf()
         save_plot(interaction_heatmap_fig, "featimp_shap_interaction_heatmap", output_dir=OUTPUT_DIR)

         # Add summary to report (uses shap_interaction_df)
         report_generator.add_shap_interaction_summary(shap_interaction_df)
    else:
         print("SHAP interaction values could not be calculated.")
         report_generator.add_shap_interaction_summary(None) # Indicate missing

except NotImplementedError as nie:
     print(nie)
except ImportError:
     print("SHAP library not found (pip install shap). Skipping SHAP analysis.")
except Exception as e:
    import traceback
    print(f"Error calculating or plotting SHAP values/interactions: {e}")
    # print(traceback.format_exc()) # Uncomment for detailed traceback
    # Ensure DataFrames are empty/None on error
    shap_df = pd.DataFrame()
    shap_interaction_df = pd.DataFrame()
    report_generator.add_shap_interaction_summary(None) # Indicate missing in report

# Store results
report_generator.results_dict['shap_importance_train'] = shap_df
report_generator.results_dict['shap_interaction_values_train'] = shap_interaction_df # Store the matrix DF

# %% [markdown]
# ### 9.5 MDI Importance (if applicable)

# %%
mdi_df = None # Initialize
if hasattr(surrogate_model, 'feature_importances_'):
    print("\n--- 6f. Calculating MDI Importance ---")
    try:
        mdi_importances = surrogate_model.feature_importances_
        mdi_df = pd.DataFrame({'feature': features, 'mdi_importance': mdi_importances})
        mdi_df = mdi_df.sort_values('mdi_importance', ascending=False)

        # Plotting
        plot_df_mdi = mdi_df.head(N_TOP_FEATURES).sort_values('mdi_importance', ascending=True)
        fig_mdi = px.bar(plot_df_mdi, x='mdi_importance', y='feature', orientation='h',
                         title=f'Top {N_TOP_FEATURES} Mean Decrease in Impurity (MDI) Importance',
                         labels={'mdi_importance': 'Importance (MDI)', 'feature': 'Feature'})
        save_plot(fig_mdi, "featimp_mdi", output_dir=OUTPUT_DIR)
        print("MDI Importance (Top 10):")
        print(mdi_df.head(N_TOP_FEATURES))

    except Exception as e:
         print(f"Error calculating MDI importance: {e}")
         mdi_df = pd.DataFrame() # Ensure empty DF on error
else:
    print("\nMDI importance not available for this model type.")
    mdi_df = pd.DataFrame() # Ensure empty DF if not applicable

# Store results
report_generator.results_dict['mdi_importance'] = mdi_df

# %% [markdown]
# ### 9.6 Combined Importance Ranking

# %%
print("\n--- Calculating Combined Importance Rankings ---")
all_imp = {}
# Add results if they exist and are DataFrames with correct structure
if perm_df is not None and isinstance(perm_df, pd.DataFrame) and 'feature' in perm_df.columns and 'importance_mean' in perm_df.columns and not perm_df.empty:
     all_imp['Permutation_Train'] = perm_df.set_index('feature')['importance_mean']
if shap_df is not None and isinstance(shap_df, pd.DataFrame) and 'feature' in shap_df.columns and 'shap_importance' in shap_df.columns and not shap_df.empty:
     all_imp['SHAP_Train'] = shap_df.set_index('feature')['shap_importance']
if mdi_df is not None and isinstance(mdi_df, pd.DataFrame) and 'feature' in mdi_df.columns and 'mdi_importance' in mdi_df.columns and not mdi_df.empty:
     all_imp['MDI'] = mdi_df.set_index('feature')['mdi_importance']
if lofo_df is not None and isinstance(lofo_df, pd.DataFrame) and 'feature' in lofo_df.columns and 'importance_mean' in lofo_df.columns and not lofo_df.empty:
     # LOFO lower score is better if using error drop, higher if using R2. Assuming higher is better for now.
     all_imp['LOFO_Train'] = lofo_df.set_index('feature')['importance_mean']

combined_imp = pd.DataFrame() # Initialize
if all_imp:
    combined_imp = pd.DataFrame(all_imp)
    # Fill NaNs with a value that ranks last (e.g., 0 if higher is better)
    combined_imp = combined_imp.fillna(0)

    rank_cols = []
    print("Calculating ranks for each importance method (higher score = better rank)...")
    for col in combined_imp.columns:
         rank_col_name = f'{col}_rank'
         # Rank descending (higher score gets rank 1)
         combined_imp[rank_col_name] = combined_imp[col].rank(ascending=False, method='min')
         rank_cols.append(rank_col_name)
         print(f"  - Ranks calculated for: {col}")

    if rank_cols:
        combined_imp['mean_rank'] = combined_imp[rank_cols].mean(axis=1)
        combined_imp = combined_imp.sort_values('mean_rank') # Sort by mean rank ascending
        print("\nTop Features (Mean Rank - based on available Train Set calculations):")
        print(combined_imp[['mean_rank'] + rank_cols].head(N_TOP_FEATURES))

        # Plot combined rank
        plot_df_comb = combined_imp.head(N_TOP_FEATURES).sort_values('mean_rank', ascending=False) # False for horiz bar
        fig_imp_comp = px.bar(plot_df_comb,
                              x='mean_rank', y=plot_df_comb.index, orientation='h',
                              title=f'Top {N_TOP_FEATURES} Features by Mean Importance Rank (Lower is Better)',
                              labels={'mean_rank': 'Mean Rank', 'y': 'Feature'},
                              text='mean_rank') # Show rank value on bars
        fig_imp_comp.update_traces(texttemplate='%{text:.2f}', textposition='outside')
        fig_imp_comp.update_layout(yaxis={'categoryorder':'total ascending'})
        save_plot(fig_imp_comp, "summary_featimp_comparison_rank", output_dir=OUTPUT_DIR)
    else:
         print("Could not calculate combined ranks (no valid rank columns generated).")
         combined_imp = pd.DataFrame() # Reset if failed
else:
     print("No valid importance scores available to combine.")
     combined_imp = pd.DataFrame() # Ensure empty

# Store combined results
report_generator.results_dict['combined_imp'] = combined_imp

# Add summary of top features to report (uses SHAP and Combined)
report_generator.add_feature_importance_summary(shap_df, combined_imp)

# %% [markdown]
# ## 10. Analysis of Top Conditions (Random Forest Only)

# %% [markdown]
# ### 10.1 Condition Feature Creation

# %%
print("\n--- 6b. Analyzing Importance of Top Conditions (if RF) ---")
condition_feature_map = {} # Map generated condition feature name -> original tuple
X_train_cond_feats = pd.DataFrame(index=X_train.index) # Empty DF to store new features
X_test_cond_feats = pd.DataFrame(index=X_test.index)  # Empty DF for test set
secondary_model = None # Initialize secondary model variable

if SURROGATE_MODEL_TYPE == 'random_forest' and rule_conditions:
    # 1. Identify Top Frequent Conditions
    N_TOP_CONDITIONS_TO_ANALYZE = 25 # How many top conditions to potentially turn into features
    if not rule_conditions:
         print("Rule conditions counter is empty. Skipping condition analysis.")
    else:
        top_conditions = rule_conditions.most_common(N_TOP_CONDITIONS_TO_ANALYZE)
        print(f"Identified Top {len(top_conditions)} most frequent conditions to potentially analyze.")

        # 2. Create Binary Features from Conditions
        valid_conditions_created = 0
        created_feature_names = set() # Track names to avoid duplicates

        for condition_tuple, freq in top_conditions:
            # Basic validation of the tuple structure
            if not (isinstance(condition_tuple, tuple) and len(condition_tuple) == 2):
                # print(f"Skipping invalid condition format: {condition_tuple}")
                continue

            feature, condition_str = condition_tuple
            try:
                op = None
                threshold = None
                # Parse condition string robustly
                if '<=' in condition_str:
                    parts = condition_str.split('<=')
                    if len(parts) == 2: op, threshold = 'le', float(parts[1].strip())
                elif '>' in condition_str:
                    parts = condition_str.split('>')
                    if len(parts) == 2: op, threshold = 'gt', float(parts[1].strip())

                if op and threshold is not None and feature in X_train.columns: # Check feature exists
                    # Create a unique, descriptive feature name
                    # Handle potential float precision issues in name
                    thresh_str = f"{threshold:.3f}".replace('.', 'p').replace('-', 'neg')
                    new_feat_name = f"cond_{feature}_{op}_{thresh_str}"
                    # Clean name further (remove potential invalid characters?) - Optional

                    if new_feat_name in created_feature_names:
                        continue # Skip if already created

                    # Create the binary feature
                    if op == 'le':
                         X_train_cond_feats[new_feat_name] = (X_train[feature] <= threshold).astype(int)
                         if feature in X_test.columns:
                              X_test_cond_feats[new_feat_name] = (X_test[feature] <= threshold).astype(int)
                         else: X_test_cond_feats[new_feat_name] = 0 # Default if feature missing in test
                    elif op == 'gt':
                         X_train_cond_feats[new_feat_name] = (X_train[feature] > threshold).astype(int)
                         if feature in X_test.columns:
                              X_test_cond_feats[new_feat_name] = (X_test[feature] > threshold).astype(int)
                         else: X_test_cond_feats[new_feat_name] = 0

                    condition_feature_map[new_feat_name] = condition_tuple
                    created_feature_names.add(new_feat_name)
                    valid_conditions_created += 1
                # else:
                    # print(f"Skipping condition due to parsing issue or missing feature: {condition_tuple}")
            except ValueError as ve:
                 # print(f"Skipping condition due to threshold conversion error: {condition_tuple} - {ve}")
                 continue
            except Exception as e:
                 print(f"Error creating binary feature for condition {condition_tuple}: {e}")

        print(f"Created {valid_conditions_created} binary features based on top conditions.")
        # Ensure test set has same columns as train set, fill missing with 0
        missing_cols_test = set(X_train_cond_feats.columns) - set(X_test_cond_feats.columns)
        for col in missing_cols_test:
            X_test_cond_feats[col] = 0
        X_test_cond_feats = X_test_cond_feats[X_train_cond_feats.columns] # Align order and columns


elif SURROGATE_MODEL_TYPE == 'random_forest':
     print("Rule conditions counter not available/empty. Skipping condition feature creation.")
else:
     print("Condition feature analysis only performed for RandomForest surrogate. Skipping.")

# Display head of created features
if not X_train_cond_feats.empty:
     print("Example Condition Features (Train Head):")
     print(X_train_cond_feats.head())


# %% [markdown]
# ### 10.2 Secondary Model Training (Original + Condition Features)

# %%
X_train_plus_cond = pd.DataFrame() # Initialize

if not X_train_cond_feats.empty:
    # 3. Train a Secondary Model
    print("\nTraining secondary model with original + condition features...")
    X_train_plus_cond = pd.concat([X_train, X_train_cond_feats], axis=1)

    # Use a slightly simpler model than the main surrogate? Or same? Using simpler here.
    secondary_model = RandomForestRegressor(
        n_estimators=50,       # Fewer estimators
        random_state=123,      # Different random state?
        n_jobs=-1,
        max_depth=8,           # Limit depth
        min_samples_leaf=10     # Increase min samples leaf
    )
    try:
         secondary_model.fit(X_train_plus_cond, y_train)
         print("Secondary model trained successfully.")
    except Exception as e:
         print(f"Error training secondary model: {e}")
         secondary_model = None # Ensure model is None if fitting failed
else:
    print("No condition features created, skipping secondary model training.")

# %% [markdown]
# ### 10.3 Importance of Condition Features (Permutation, MDI, SHAP, LOFO)

# %%
# Initialize result dataframes for this section
condition_importance_df = pd.DataFrame()
condition_mdi_df = pd.DataFrame()
condition_shap_df = pd.DataFrame()
condition_lofo_df = pd.DataFrame()

if secondary_model is not None and not X_train_plus_cond.empty:
    condition_feature_names = list(X_train_cond_feats.columns) # Get names of added features

    # --- Permutation Importance for Conditions ---
    print("\nCalculating permutation importance for condition features (Train Set)...")
    try:
        cond_perm_importance = permutation_importance(
            secondary_model, X_train_plus_cond, y_train,
            n_repeats=5, random_state=123, n_jobs=-1, # Fewer repeats for speed
            scoring='neg_root_mean_squared_error' # Higher drop = more important
        )
        cond_importances = pd.Series(cond_perm_importance.importances_mean, index=X_train_plus_cond.columns)
        condition_importance_scores = cond_importances[condition_feature_names].sort_values(ascending=False) # Filter and sort

        condition_importance_df = pd.DataFrame({
            'Condition_Feature': condition_importance_scores.index,
            'Importance': condition_importance_scores.values
        })
        # Map back to original condition tuple
        condition_importance_df['Original_Condition'] = condition_importance_df['Condition_Feature'].map(condition_feature_map)

        if not condition_importance_df.empty:
            plot_df_cond_perm = condition_importance_df.head(N_TOP_FEATURES).sort_values('Importance', ascending=True)
            fig_cond_imp = px.bar(plot_df_cond_perm,
                                  x='Importance', y='Condition_Feature', orientation='h',
                                  title=f'Top {N_TOP_FEATURES} Condition Feature Importances (Permutation on Train)',
                                  hover_data=['Original_Condition'])
            save_plot(fig_cond_imp, "featimp_condition_importance", output_dir=OUTPUT_DIR)
            print("Top 10 Condition Feature Importances (Permutation):")
            print(condition_importance_df[['Original_Condition', 'Importance']].head(10))
        else:
            print("No permutation importance scores calculated for condition features.")
    except Exception as e:
        print(f"Error during condition permutation importance calculation: {e}")
        condition_importance_df = pd.DataFrame()

    # --- MDI Importance for Conditions ---
    print("\nCalculating MDI importance for condition features...")
    if hasattr(secondary_model, 'feature_importances_'):
        try:
            mdi_importances_secondary = secondary_model.feature_importances_
            cond_mdi_scores = pd.Series(mdi_importances_secondary, index=X_train_plus_cond.columns)
            condition_mdi_filtered = cond_mdi_scores[condition_feature_names].sort_values(ascending=False) # Filter & sort

            condition_mdi_df = pd.DataFrame({
                'Condition_Feature': condition_mdi_filtered.index,
                'MDI_Importance': condition_mdi_filtered.values
            })
            condition_mdi_df['Original_Condition'] = condition_mdi_df['Condition_Feature'].map(condition_feature_map)

            if not condition_mdi_df.empty:
                plot_df_cond_mdi = condition_mdi_df.head(N_TOP_FEATURES).sort_values('MDI_Importance', ascending=True)
                fig_cond_mdi = px.bar(plot_df_cond_mdi,
                                      x='MDI_Importance', y='Condition_Feature', orientation='h',
                                      title=f'Top {N_TOP_FEATURES} Condition Feature Importances (MDI on Secondary Model)',
                                      hover_data=['Original_Condition'])
                save_plot(fig_cond_mdi, "featimp_condition_importance_mdi", output_dir=OUTPUT_DIR)
                print("Top 10 Condition Feature Importances (MDI):")
                print(condition_mdi_df[['Original_Condition', 'MDI_Importance']].head(10))
            else:
                 print("No MDI importance scores calculated for condition features.")
        except Exception as e:
            print(f"Error calculating MDI importance for conditions: {e}")
            condition_mdi_df = pd.DataFrame()
    else:
        print("Secondary model does not have feature_importances_ attribute for MDI.")
        condition_mdi_df = pd.DataFrame()

    # --- SHAP Importance for Conditions ---
    print("\nCalculating SHAP importance for condition features (can be slow)...")
    try:
        # Re-initialize explainer for the secondary model
        explainer_secondary = shap.TreeExplainer(secondary_model, X_train_plus_cond, feature_perturbation='interventional') # Use combined data as background
        shap_values_secondary = explainer_secondary(X_train_plus_cond) # Calculate SHAP for combined data
        shap_mean_abs_secondary = np.abs(shap_values_secondary.values).mean(axis=0)

        cond_shap_scores = pd.Series(shap_mean_abs_secondary, index=X_train_plus_cond.columns)
        condition_shap_filtered = cond_shap_scores[condition_feature_names].sort_values(ascending=False) # Filter & sort

        condition_shap_df = pd.DataFrame({
            'Condition_Feature': condition_shap_filtered.index,
            'SHAP_Importance': condition_shap_filtered.values
        })
        condition_shap_df['Original_Condition'] = condition_shap_df['Condition_Feature'].map(condition_feature_map)

        if not condition_shap_df.empty:
             plot_df_cond_shap = condition_shap_df.head(N_TOP_FEATURES).sort_values('SHAP_Importance', ascending=True)
             fig_cond_shap = px.bar(plot_df_cond_shap,
                                    x='SHAP_Importance', y='Condition_Feature', orientation='h',
                                    title=f'Top {N_TOP_FEATURES} Condition Feature Importances (Mean Abs SHAP on Secondary Model)',
                                    hover_data=['Original_Condition'])
             save_plot(fig_cond_shap, "featimp_condition_importance_shap", output_dir=OUTPUT_DIR)
             print("Top 10 Condition Feature Importances (SHAP):")
             print(condition_shap_df[['Original_Condition', 'SHAP_Importance']].head(10))
        else:
             print("No SHAP importance scores calculated for condition features.")
    except Exception as e:
        print(f"Error calculating SHAP importance for conditions: {e}")
        condition_shap_df = pd.DataFrame()

    # --- LOFO Importance for Conditions ---
    print("\nCalculating LOFO importance for condition features...")
    try:
        # Use KFold for cross-validation as condition features might not be time-dependent in isolation
        # relative to original features, but evaluate importance *within* the secondary model
        n_splits_cond_lofo = min(3, len(X_train_plus_cond) // 2) if len(X_train_plus_cond) >= 4 else 2
        if n_splits_cond_lofo < 2:
             print("Warning: Not enough samples for KFold CV in Condition LOFO. Skipping.")
             raise ValueError("Insufficient samples for LOFO KFold CV.")

        cv_lofo_cond = KFold(n_splits=n_splits_cond_lofo, shuffle=True, random_state=RANDOM_STATE + 1)
        # Create Dataset with original + condition features
        lofo_dataset_cond = Dataset(df=pd.concat([X_train_plus_cond, y_train], axis=1),
                                    target=TARGET_VARIABLE,
                                    features=condition_feature_names) # Only test leaving out condition features

        # Use the secondary_model (trained on original+condition)
        lofo_model_cond = clone(secondary_model) # Clone secondary model
        lofo_imp_cond = LOFOImportance(lofo_dataset_cond,
                                    model=lofo_model_cond,
                                    cv=cv_lofo_cond,
                                    scoring='r2') # Use R2 score

        importance_df_cond_lofo_raw = lofo_imp_cond.get_importance()

        # Process results
        imp_col_name_cond_lofo = None
        expected_cols = ['importance_mean', 'importance', 'val_imp_mean']
        for col in expected_cols:
             if col in importance_df_cond_lofo_raw.columns:
                  imp_col_name_cond_lofo = col
                  break

        if imp_col_name_cond_lofo:
            print(f"Using Condition LOFO importance column: '{imp_col_name_cond_lofo}'")
            condition_lofo_df = importance_df_cond_lofo_raw[['feature', imp_col_name_cond_lofo]].rename(columns={imp_col_name_cond_lofo: 'importance_mean'})
            condition_lofo_df = condition_lofo_df.sort_values('importance_mean', ascending=False)
            condition_lofo_df['Original_Condition'] = condition_lofo_df['feature'].map(condition_feature_map) # Map feature name back

            # Plotting
            try:
                plot_importance(importance_df_cond_lofo_raw, figsize=(10, max(6, len(condition_feature_names)//2)))
                fig_cond_lofo = plt.gcf()
                fig_cond_lofo.suptitle(f'Top Condition Feature Importances (LOFO R2 on Secondary Model)', y=1.02)
                plt.tight_layout(rect=[0, 0.03, 1, 0.98])
                save_plot(fig_cond_lofo, "featimp_condition_importance_lofo", output_dir=OUTPUT_DIR)
                print("Top 10 Condition Feature Importances (LOFO):")
                print(condition_lofo_df[['Original_Condition', 'importance_mean']].head(10))
            except Exception as plot_e:
                print(f"Error generating Condition LOFO plot: {plot_e}")
        else:
            print("Could not identify the correct importance column in Condition LOFO results.")
            condition_lofo_df = pd.DataFrame() # Reset if failed

    except ImportError:
         print("LOFO library not found. Skipping LOFO for conditions.")
    except ValueError as ve:
         print(f"ValueError during Condition LOFO setup/run: {ve}")
    except Exception as e:
        print(f"Error calculating LOFO importance for conditions: {e}")
        condition_lofo_df = pd.DataFrame()

else:
     print("Secondary model not trained or no condition features available. Skipping condition importance analysis.")


# Store results
report_generator.results_dict['condition_importance_perm'] = condition_importance_df
report_generator.results_dict['condition_importance_mdi'] = condition_mdi_df
report_generator.results_dict['condition_importance_shap'] = condition_shap_df
report_generator.results_dict['condition_importance_lofo'] = condition_lofo_df

# Add summary to report (uses permutation importance df)
report_generator.add_condition_importance_summary(condition_importance_df)

# %% [markdown]
# ### 10.4 Linear Models using ONLY Condition Features

# %%
print("\n--- 6b.1 Linear Models using ONLY Condition Features ---")
condition_linear_results = None # Initialize local variable

if not X_train_cond_feats.empty:
    try:
        # Prepare data (using only condition features)
        X_train_cond_only = X_train_cond_feats
        X_test_cond_only = X_test_cond_feats

        # 1. Standard Linear Regression
        print("Training Linear Regression (Conditions Only)...")
        lr_cond = LinearRegression()
        lr_cond.fit(X_train_cond_only, y_train)
        y_pred_test_lr_cond = lr_cond.predict(X_test_cond_only)
        rmse_lr_cond = np.sqrt(mean_squared_error(y_test, y_pred_test_lr_cond))
        r2_lr_cond = r2_score(y_test, y_pred_test_lr_cond)
        print(f"  Linear Regression: Test RMSE={rmse_lr_cond:.4f}, R2={r2_lr_cond:.4f}")

        # 2. Ridge Regression (with Cross-Validation for alpha)
        print("Training RidgeCV Regression (Conditions Only)...")
        ridge_alphas = np.logspace(-4, 2, 100) # Alpha range
        # Use KFold for CV as condition features might not be time-ordered in the same way
        cv_ridge_lasso = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE + 2)
        ridge_cond = RidgeCV(alphas=ridge_alphas, cv=cv_ridge_lasso) # Removed store_cv_values=True (can use lots of memory)
        ridge_cond.fit(X_train_cond_only, y_train)
        y_pred_test_ridge_cond = ridge_cond.predict(X_test_cond_only)
        rmse_ridge_cond = np.sqrt(mean_squared_error(y_test, y_pred_test_ridge_cond))
        r2_ridge_cond = r2_score(y_test, y_pred_test_ridge_cond)
        print(f"  RidgeCV Regression:  Test RMSE={rmse_ridge_cond:.4f}, R2={r2_ridge_cond:.4f}, Best Alpha={ridge_cond.alpha_:.4f}")

        # 3. LASSO Regression (with Cross-Validation for alpha)
        print("Training LassoCV Regression (Conditions Only)...")
        lasso_cond = LassoCV(cv=cv_ridge_lasso, random_state=RANDOM_STATE + 3, max_iter=10000, n_jobs=-1)
        lasso_cond.fit(X_train_cond_only, y_train)
        y_pred_test_lasso_cond = lasso_cond.predict(X_test_cond_only)
        rmse_lasso_cond = np.sqrt(mean_squared_error(y_test, y_pred_test_lasso_cond))
        r2_lasso_cond = r2_score(y_test, y_pred_test_lasso_cond)
        print(f"  LassoCV Regression:  Test RMSE={rmse_lasso_cond:.4f}, R2={r2_lasso_cond:.4f}, Best Alpha={lasso_cond.alpha_:.4f}")

        # 4. Analyze Coefficients
        ridge_coefs = pd.DataFrame({
            'Condition_Feature': X_train_cond_only.columns,
            'Ridge_Coefficient': ridge_cond.coef_
        })
        ridge_coefs['Original_Condition'] = ridge_coefs['Condition_Feature'].map(condition_feature_map)
        ridge_coefs = ridge_coefs.iloc[np.abs(ridge_coefs['Ridge_Coefficient']).argsort()[::-1]] # Sort by abs value

        lasso_coefs = pd.DataFrame({
            'Condition_Feature': X_train_cond_only.columns,
            'Lasso_Coefficient': lasso_cond.coef_
        })
        lasso_coefs['Original_Condition'] = lasso_coefs['Condition_Feature'].map(condition_feature_map)
        lasso_coefs['Is_Zero'] = np.isclose(lasso_coefs['Lasso_Coefficient'], 0)
        lasso_coefs = lasso_coefs.iloc[np.abs(lasso_coefs['Lasso_Coefficient']).argsort()[::-1]] # Sort by abs value

        print("\nTop Ridge Coefficients (Conditions Only):")
        print(ridge_coefs[['Original_Condition', 'Ridge_Coefficient']].head(10))

        print("\nTop Lasso Coefficients (Conditions Only):")
        print(lasso_coefs[['Original_Condition', 'Lasso_Coefficient', 'Is_Zero']].head(15))
        num_lasso_zeros = lasso_coefs['Is_Zero'].sum()
        print(f"Number of Lasso coefficients shrunk to zero: {num_lasso_zeros} / {len(lasso_coefs)}")

        # Store results for summary file/JSON
        condition_linear_results = {
            'linear': {'rmse': rmse_lr_cond, 'r2': r2_lr_cond},
            'ridge': {'rmse': rmse_ridge_cond, 'r2': r2_ridge_cond, 'alpha': ridge_cond.alpha_,
                       'coefficients': ridge_coefs.to_dict('records')}, # Store as list of dicts
            'lasso': {'rmse': rmse_lasso_cond, 'r2': r2_lasso_cond, 'alpha': lasso_cond.alpha_,
                       'coefficients': lasso_coefs.to_dict('records')} # Store as list of dicts
        }
        # Add to main results dict
        report_generator.results_dict['condition_linear_results'] = condition_linear_results

    except Exception as e:
        import traceback
        print(f"Error during Linear Model training on condition features: {e}")
        # print(traceback.format_exc()) # Uncomment for detail
        report_generator.results_dict['condition_linear_results'] = None # Ensure None on error
else:
    print("No condition features were generated. Skipping linear models based on conditions.")
    report_generator.results_dict['condition_linear_results'] = None

# %% [markdown]
# ## 11. Feature Interaction Analysis

# %% [markdown]
# ### 11.1 Pairwise H-Statistic & Stability

# %%
print("\n--- 7a. Calculating Pairwise Friedman H-statistics (Approximation) ---")
print("Note: H-statistic values are approximations based on PDP variances.")
h_values = None # Initialize H-statistic Series
interaction_stability = {} # Initialize stability dict

# Check if features exist
if not features:
     print("No features found to calculate interactions. Skipping.")
else:
    h_matrix = pd.DataFrame(index=features, columns=features, dtype=float)
    np.fill_diagonal(h_matrix.values, 1.0) # Fill diagonal with 1 (or 0?) - H is for pairs
    interaction_pairs = list(combinations(features, 2))

    if interaction_pairs:
        # Use a sample of training data for H-stat calculation for speed
        h_stat_sample_size = min(100, len(X_train)) # Sample size for H-stat
        X_sample_h = X_train.sample(h_stat_sample_size, random_state=RANDOM_STATE) if h_stat_sample_size < len(X_train) else X_train
        print(f"Calculating H-statistic on a sample of {len(X_sample_h)} training rows.")

        h_calc_count = 0
        for i, (f1, f2) in enumerate(interaction_pairs):
            print(f"  Calculating H for ({f1}, {f2}) - Pair {i+1}/{len(interaction_pairs)}", end='\r')
            # Pass reduced grid resolution and sample size
            h = friedman_h_statistic(surrogate_model, X_sample_h, f1, f2, grid_resolution=15) # Use defined function
            if not np.isnan(h):
                h_matrix.loc[f1, f2] = h
                h_matrix.loc[f2, f1] = h # Symmetric
                h_calc_count += 1
        print(f"\nPairwise H-statistic calculation complete ({h_calc_count} valid pairs calculated).")

        if h_calc_count > 0:
            # Fill NaNs with 0 for plotting, but keep original matrix for extraction
            fig_h = px.imshow(h_matrix.fillna(0), text_auto=".2f", aspect="auto",
                          title='Pairwise Friedman H-statistic (Approximation)',
                          color_continuous_scale='Viridis', range_color=[0,1],
                          labels=dict(color="H-statistic"))
            save_plot(fig_h, "interaction_h_statistic_heatmap", output_dir=OUTPUT_DIR)
            print("Saved H-statistic heatmap.")

            # Extract top interactions (excluding diagonal, handling duplicates)
            h_values_unstacked = h_matrix.unstack()
            # Filter out diagonal before dropping NaN and sorting
            h_values_unstacked = h_values_unstacked[h_values_unstacked.index.get_level_values(0) != h_values_unstacked.index.get_level_values(1)]
            h_values = h_values_unstacked.dropna().sort_values(ascending=False)

            # Keep only one entry for each pair (e.g., (A,B) but not (B,A))
            h_values = h_values[~h_values.index.map(lambda x: tuple(sorted(x))).duplicated()]

            if not h_values.empty:
                print("\nTop Pairwise Interactions (H-statistic Approx):")
                print(h_values.head(N_TOP_FEATURES))
            else:
                print("No valid H-statistic values found after filtering.")
                h_values = pd.Series(dtype=float) # Ensure empty Series if no results
        else:
            print("H-statistic calculation yielded no valid results.")
            h_values = pd.Series(dtype=float) # Ensure empty Series
    else:
        print("No feature pairs found for H-statistic calculation.")
        h_values = pd.Series(dtype=float) # Ensure empty Series

    # --- Interaction Stability ---
    print("\nCalculating Interaction Stability (Bootstrapped H-statistic)...")
    if h_values is not None and not h_values.empty and N_BOOTSTRAP_SAMPLES > 0:
        # Focus on top interactions from initial calculation
        num_pairs_stability = min(5, len(h_values)) # Assess top 5 pairs for stability
        top_pairs_for_stability = h_values.head(num_pairs_stability).index.tolist()
        print(f"Assessing stability for top {num_pairs_stability} pairs: {top_pairs_for_stability}")

        bootstrap_h_stats = {pair: [] for pair in top_pairs_for_stability}

        for i in range(N_BOOTSTRAP_SAMPLES):
            print(f"  Bootstrap sample {i+1}/{N_BOOTSTRAP_SAMPLES}", end='\r')
            X_boot, y_boot = resample(X_train, y_train, random_state=RANDOM_STATE + i) # Resample with replacement

            # Option 1: Use original model on bootstrap sample (faster approximation)
            # boot_model_stable = surrogate_model
            # X_sample_boot = X_boot.sample(h_stat_sample_size, random_state=123 + i) if h_stat_sample_size < len(X_boot) else X_boot

            # Option 2: Retrain model on bootstrap sample (slower, more accurate stability)
            try:
                 boot_model_stable = clone(surrogate_model).fit(X_boot, y_boot)
                 X_sample_boot = X_boot # Use full bootstrap sample for H-stat if retraining
            except Exception as fit_err:
                 print(f"\nError fitting model on bootstrap sample {i+1}: {fit_err}. Skipping sample.")
                 continue
            # --------------------------------------------------------------------------------------

            for pair in top_pairs_for_stability:
                 if isinstance(pair, tuple) and len(pair)==2:
                      f1_s, f2_s = pair
                      # Use the same settings as the initial H-stat calc
                      h_boot = friedman_h_statistic(boot_model_stable, X_sample_boot, f1_s, f2_s, grid_resolution=15)
                      if not np.isnan(h_boot):
                           bootstrap_h_stats[pair].append(h_boot)

        print("\nInteraction Stability Results (Mean +/- Std Dev H-statistic across samples):")
        interaction_stability = {} # Reset and populate
        for pair, h_list in bootstrap_h_stats.items():
            if h_list: # Ensure list is not empty
                mean_h = np.mean(h_list)
                std_h = np.std(h_list)
                interaction_stability[pair] = {'mean': mean_h, 'std': std_h}
                print(f"  {pair}: {mean_h:.3f} +/- {std_h:.3f} (n={len(h_list)})")
            else:
                print(f"  {pair}: No valid H-statistics calculated in bootstrap.")
                interaction_stability[pair] = {'mean': np.nan, 'std': np.nan} # Store NaN if no results
    else:
        print("Skipping interaction stability (no initial H-stats, N_BOOTSTRAP_SAMPLES=0, or error).")

# Store results
report_generator.results_dict['h_values'] = h_values
report_generator.results_dict['interaction_stability'] = interaction_stability

# %% [markdown]
# ### 11.2 Three-Way H-Statistic (Top Feature Triplets)

# %%
print("\n--- 7b. Calculating Three-way H-statistic (Approximation, for Top Feature Triplets) ---")
h_3way_df = None # Initialize DataFrame for results

# Need combined importance to select top features
if combined_imp is None or combined_imp.empty:
     print("Combined importance not available, cannot determine top features for 3-way H-statistic. Skipping.")
elif len(features) < 3:
    print("Not enough features (need 3+) to calculate 3-way interactions. Skipping.")
else:
    N_TOP_FEATURES_3WAY = 4 # Calculate for triplets from top 4 features
    top_n_features = combined_imp.head(N_TOP_FEATURES_3WAY).index.tolist()
    print(f"Calculating 3-way H for triplets from top {N_TOP_FEATURES_3WAY} features: {top_n_features}")

    triplet_combinations = list(combinations(top_n_features, 3))
    print(f"Number of triplets to calculate: {len(triplet_combinations)}")

    if not triplet_combinations:
         print("No triplets formed from top features. Skipping.")
    else:
        h_3way_results_list = []
        # Use a smaller sample for 3-way calculation
        h_3way_sample_size = min(50, len(X_train)) # Sample size for 3-way H
        X_sample_3way = X_train.sample(h_3way_sample_size, random_state=RANDOM_STATE + 10) if h_3way_sample_size < len(X_train) else X_train
        print(f"Calculating 3-way H on sample size: {len(X_sample_3way)}")

        for i, triplet in enumerate(triplet_combinations):
            f1, f2, f3 = triplet
            print(f"  Calculating for triplet {i+1}/{len(triplet_combinations)}: ({f1}, {f2}, {f3})", end='\r')
            # Pass reduced grid resolution and sample
            h_3way = friedman_h_3way(surrogate_model, X_sample_3way, f1, f2, f3, grid_resolution=8) # Use defined function

            if h_3way is not None and not np.isnan(h_3way):
                h_3way_results_list.append({'Triplet': triplet, 'H_statistic': h_3way})
                print(f"  Triplet {i+1}: ({f1}, {f2}, {f3}) -> H ≈ {h_3way:.4f}      ") # Overwrite progress line with result
            else:
                print(f"  Could not calculate 3-way H-statistic proxy for ({f1}, {f2}, {f3}). Skipping triplet. ")

        print("\nThree-way H-statistic calculation complete.")

        if h_3way_results_list:
            h_3way_df = pd.DataFrame(h_3way_results_list)
            # Format triplet for plotting/readability
            h_3way_df['Triplet_Str'] = h_3way_df['Triplet'].apply(lambda x: ' x '.join(map(str,x)))
            h_3way_df = h_3way_df.sort_values('H_statistic', ascending=True) # Sort asc for plot

            # Plot results
            fig_h3 = px.bar(h_3way_df,
                            x='H_statistic', y='Triplet_Str', orientation='h',
                            title=f'Approximate 3-Way H-Statistic (Top {N_TOP_FEATURES_3WAY} Feature Triplets)',
                            labels={'H_statistic': 'H-statistic (Approximation)', 'Triplet_Str': 'Feature Triplet'},
                            text='H_statistic')
            fig_h3.update_traces(texttemplate='%{text:.3f}', textposition='outside')
            fig_h3.update_layout(
                yaxis={'categoryorder':'total ascending'}, # Ensure y-axis reflects sorted values
                xaxis_range=[0, max(0.1, h_3way_df['H_statistic'].max() * 1.1)], # Adjust x-axis range
                margin=dict(l=max(150, h_3way_df['Triplet_Str'].str.len().max() * 6)) # Adjust left margin based on label length
            )
            save_plot(fig_h3, "interaction_h_statistic_3way_bar", output_dir=OUTPUT_DIR)
            print("Saved 3-way H-statistic bar chart.")
            print("Top 3-way Interactions:")
            print(h_3way_df.sort_values('H_statistic', ascending=False).head()) # Print top descending
        else:
            print("No valid 3-way H-statistics calculated.")
            h_3way_df = pd.DataFrame() # Ensure empty df

# Store results (potentially empty DF)
report_generator.results_dict['threeway_interactions_h'] = h_3way_df

# Add feature interaction summary to report (uses h_values, stability, h_3way_df)
report_generator.add_feature_interactions_summary(h_values, interaction_stability, h_3way_df)

# %% [markdown]
# ### 11.3 PDP / ICE Plots

# %%
print("\n--- 7d. Generating Partial Dependence Plots (PDP/ICE) ---")
# Use top features from combined importance if available, otherwise SHAP, otherwise first N features
if combined_imp is not None and not combined_imp.empty:
     top_features_pdp = combined_imp.head(5).index.tolist()
elif shap_df is not None and not shap_df.empty:
     top_features_pdp = shap_df['feature'].head(5).tolist()
else:
     top_features_pdp = features[:5]

print(f"Generating PDP/ICE for top features: {top_features_pdp}")
for feature in top_features_pdp:
    if feature not in X_train.columns:
        print(f"Skipping PDP/ICE for feature '{feature}' as it's not in X_train columns.")
        continue
    try:
        print(f"  Generating PDP/ICE for {feature}...")
        fig_pdp_ice, ax_pdp_ice = plt.subplots(figsize=(8, 6))
        # Use from_estimator for convenience
        pdp_display = PartialDependenceDisplay.from_estimator(
            surrogate_model,
            X_train, # Use train data for PDP calculation
            features=[feature], # Feature to plot
            kind='both',    # 'average' for PDP, 'individual' for ICE, 'both'
            ice_lines_kw={"color": "tab:blue", "alpha": 0.2, "linewidth": 0.5},
            ax=ax_pdp_ice # Pass the created axes
        )
        ax_pdp_ice.set_title(f'PDP and ICE for {feature}')
        ax_pdp_ice.grid(True)
        plt.tight_layout()
        # Save using the helper function, which handles the display object
        save_plot(pdp_display, f"pdp_ice_{feature}", output_dir=OUTPUT_DIR)
    except Exception as e:
        import traceback
        print(f"Could not generate PDP/ICE for {feature}: {e}")
        # print(traceback.format_exc()) # Uncomment for detailed traceback
        # Ensure plot is closed if error occurred after figure creation
        plt.close(fig_pdp_ice)


# %% [markdown]
# ### 11.4 Custom Interaction PDP Plots

# %%
print("\n--- 7e. Generating Custom Interaction PDP plots (Matplotlib) ---")
interaction_strengths_custom = {} # Store H-stats calculated by custom function

# Determine the list of pairs to plot based on H-statistic or SHAP interactions
pairs_to_plot = []
if h_values is not None and not h_values.empty:
    pairs_to_plot = h_values.head(min(5, len(h_values))).index.tolist() # Top 5 H-stat pairs
    print(f"Generating custom interaction plots for top {len(pairs_to_plot)} feature pairs (based on H-statistic)...")
elif shap_interaction_df is not None and not shap_interaction_df.empty:
    print("H-statistic values not available/empty. Falling back to top SHAP interactions...")
    # Extract top off-diagonal SHAP interactions
    try:
         inter_vals_shap = shap_interaction_df.copy()
         np.fill_diagonal(inter_vals_shap.values, -np.inf) # Ignore diagonal
         inter_vals_shap_stacked = inter_vals_shap.stack().sort_values(ascending=False)
         inter_vals_shap_stacked = inter_vals_shap_stacked[inter_vals_shap_stacked != -np.inf]
         pairs_to_plot = inter_vals_shap_stacked.head(min(5, len(inter_vals_shap_stacked))).index.tolist()
         print(f"Generating custom interaction plots for top {len(pairs_to_plot)} feature pairs (based on SHAP)...")
    except Exception as shap_pair_e:
         print(f"Error processing SHAP interactions to find pairs: {shap_pair_e}")
else:
    print("No interaction ranking (H-statistic or SHAP) available to select top pairs. Skipping custom interaction plots.")

# Generate custom interaction plots for the selected top pairs
plotted_count = 0
for pair in pairs_to_plot:
    if isinstance(pair, tuple) and len(pair) == 2:
        f1, f2 = pair
        # Check if features exist in training data
        if f1 not in X_train.columns or f2 not in X_train.columns:
            print(f"Skipping custom PDP for pair ({f1}, {f2}): Feature not found in X_train.")
            continue

        print(f"\nCreating custom PDP interaction plot for {f1} and {f2}")
        try:
            # Call the custom plotting function
            fig_custom_pdp, h_stat_custom = create_interaction_pdp(
                surrogate_model, X_train,
                [f1, f2],
                [f1, f2], # Pass names explicitly
                grid_resolution=20 # Use default or adjust
            )
            if fig_custom_pdp is not None:
                # Use save_plot helper for the matplotlib figure
                save_plot(fig_custom_pdp, f"custom_pdp_interaction_{f1}_vs_{f2}", output_dir=OUTPUT_DIR)
                # Only store strength if calculated successfully
                if h_stat_custom is not None and not np.isnan(h_stat_custom):
                     interaction_strengths_custom[pair] = h_stat_custom
                     print(f"Custom PDP interaction plot saved. Calculated H-stat Approx: {h_stat_custom:.4f}")
                else:
                     print(f"Custom PDP interaction plot saved. H-stat Approx calculation failed.")
                plotted_count += 1
            else:
                print(f"Custom PDP interaction plot generation failed for ({f1}, {f2}).")
        except Exception as e:
            import traceback
            print(f"Error creating custom PDP plot for ({f1}, {f2}): {e}")
            # print(traceback.format_exc()) # Uncomment for detailed traceback
            # Ensure plot is closed if error happened after figure creation
            if 'fig_custom_pdp' in locals() and fig_custom_pdp is not None:
                 plt.close(fig_custom_pdp)
    else:
         print(f"Skipping custom interaction plot for invalid pair format: {pair}")

print(f"\nFinished generating {plotted_count} custom interaction PDP plots.")
report_generator.results_dict['custom_pdp_h_stats'] = interaction_strengths_custom

# %% [markdown]
# ## 12. Causality Analysis (Granger)

# %% [markdown]
# ### 12.1 Stationarity Tests (ADF)

# %%
print("\n--- 8a. Performing Stationarity Tests (ADF) on Training Data ---")
stationarity_results = {} # Re-initialize just in case
if not features:
     print("No features available for stationarity testing.")
else:
    for feature in features:
        if feature not in X_train.columns: continue # Should not happen, but check
        try:
            # Drop NaNs for the specific feature series before testing
            feature_series = X_train[feature].dropna()
            if feature_series.nunique() <= 1: # Check for constant series
                 print(f"  {feature}: Constant series, skipping ADF test.")
                 stationarity_results[feature] = {'adf_stat': np.nan, 'p_value': np.nan, 'is_stationary': False, 'notes': 'Constant series'}
                 continue
            if len(feature_series) < 10: # Check for very short series
                 print(f"  {feature}: Series too short ({len(feature_series)}), skipping ADF test.")
                 stationarity_results[feature] = {'adf_stat': np.nan, 'p_value': np.nan, 'is_stationary': False, 'notes': 'Series too short'}
                 continue

            result = adfuller(feature_series)
            p_value = result[1]
            is_stationary = p_value <= 0.05
            stationarity_results[feature] = {
                'adf_stat': result[0],
                'p_value': p_value,
                'is_stationary': is_stationary
            }
            print(f"  {feature}: ADF Stat={result[0]:.3f}, p-value={p_value:.3f} -> {'Stationary' if is_stationary else 'Non-Stationary'}")
        except Exception as e:
            print(f"  Could not perform ADF test for {feature}: {e}")
            stationarity_results[feature] = {'error': str(e), 'is_stationary': False} # Assume non-stationary on error

# Add stationarity results to report
report_generator.add_stationarity_summary(stationarity_results)
# Store in main dict
report_generator.results_dict['stationarity_test_adf'] = stationarity_results

# %% [markdown]
# ### 12.2 Granger Causality (Feature -> Target)

# %%
print("\n--- 8b. Running Granger Causality Tests (Feature -> Target) ---")
GRANGER_MAX_LAG = 3 # Define max lag for tests
granger_p_values = None # Initialize

if not features:
     print("No features available for Granger Causality testing.")
else:
    try:
        # Call the helper function
        granger_p_values = perform_granger_causality(
            data_features=X_train[features], # Ensure only feature columns are passed
            data_target=y_train,
            feature_names=features,
            max_lag=GRANGER_MAX_LAG
        )

        if granger_p_values is not None and not granger_p_values.isnull().all():
            # Visualize p-values with a bar chart
            granger_plot_data = granger_p_values.dropna().reset_index() # Drop NaNs for plotting
            granger_plot_data.columns = ['Feature', 'P_Value']
            granger_plot_data = granger_plot_data.sort_values('P_Value') # Sort by p-value

            fig_granger = px.bar(granger_plot_data, x='Feature', y='P_Value',
                                 title=f'Granger Causality (Feature -> {TARGET_VARIABLE}) P-Values (Lag {GRANGER_MAX_LAG})',
                                 labels={'P_Value': 'P-Value (Lower suggests causality)'})
            # Add significance line
            fig_granger.add_hline(y=0.05, line_dash="dash", line_color="red", annotation_text="p=0.05")
            # Adjust y-axis focus, ensure it starts at 0
            max_p_val = granger_plot_data['P_Value'].max() if not granger_plot_data.empty else 0.1
            fig_granger.update_layout(yaxis_range=[0, max(0.1, max_p_val * 1.1)])
            fig_granger.update_layout(xaxis_tickangle=-45)
            save_plot(fig_granger, "causality_granger_bar", output_dir=OUTPUT_DIR)
            print("Saved Granger Causality bar chart.")
            print("\nGranger Causality Results (Sorted by p-value):")
            print(granger_plot_data)

        else:
            print("Granger Causality analysis yielded no valid results (all NaN or None).")
            if granger_p_values is None: granger_p_values = pd.Series(dtype=float) # Ensure empty Series

    except Exception as e:
        import traceback
        print(f"Error during Granger Causality analysis: {e}")
        # print(traceback.format_exc()) # Uncomment for detail
        granger_p_values = pd.Series(dtype=float) # Ensure empty Series on error

# Add to report (handles None case)
report_generator.add_granger_causality_summary(granger_p_values, GRANGER_MAX_LAG)
# Store in main dict
report_generator.results_dict['granger_causality_feature_to_target_p_values'] = granger_p_values

# %% [markdown]
# ## 13. Interpretability Summary Plots (SHAP Force Plot)

# %%
print("\n--- 9. Interpretability Summary Plots ---")

if explainer is not None and shap_values is not None and not X_test.empty:
    print("Generating SHAP Force Plots for sample predictions (Test Set)...")
    # Generate for first and maybe a 'median' or 'high prediction' sample? Just first for now.
    try:
        sample_index = 0 # First sample in test set
        # Ensure sample_index is valid
        if sample_index < len(X_test):
            # Force plot requires expected value, SHAP values for the instance, and feature values
            # Note: explainer.expected_value might be a single value or array depending on model output
            expected_value = explainer.expected_value
            if isinstance(expected_value, (np.ndarray, list)):
                expected_value = expected_value[0] # Use first element if it's array-like

            force_plot_html = shap.force_plot(
                expected_value,
                shap_values.values[sample_index,:], # SHAP values for the sample
                X_test.iloc[sample_index,:],       # Feature values for the sample
                show=False, matplotlib=False # Ensure it generates HTML version
            )
            force_plot_path = os.path.join(OUTPUT_DIR, f"interpret_shap_force_plot_test_sample_{sample_index}.html")
            shap.save_html(force_plot_path, force_plot_html)
            print(f"Saved sample SHAP force plot (HTML): {force_plot_path}")
        else:
            print("Test set has fewer samples than requested index for SHAP force plot.")

    except Exception as e:
        print(f"Could not generate SHAP force plot: {e}")
else:
    print("SHAP explainer, SHAP values, or Test set not available. Skipping force plots.")

# %% [markdown]
# ## 14. Feature Engineering & Linear Model Comparison

# %% [markdown]
# ### 14.1 Engineered Feature Creation

# %%
print("\n--- 12. Feature Engineering & Linear Model Comparison ---")
engineered_features_train_df = pd.DataFrame(index=X_train.index)
engineered_features_test_df = pd.DataFrame(index=X_test.index)
engineered_feature_names = [] # List to store names of successfully created features
created_threshold_features = 0
created_interaction_features = 0

# --- Create Threshold features based on MOST IMPORTANT conditions ---
# Use the condition_importance_df calculated earlier (based on permutation importance)
if SURROGATE_MODEL_TYPE == 'random_forest' and condition_importance_df is not None and not condition_importance_df.empty:
    print("Creating engineered features based on MOST IMPORTANT conditions...")
    # Sort by importance and select top N conditions
    N_THRESHOLD_FEATURES = 5 # Max number of threshold features to create
    top_conditions_for_eng_df = condition_importance_df.sort_values('Importance', ascending=False).head(N_THRESHOLD_FEATURES)
    print(f"Using top {len(top_conditions_for_eng_df)} conditions based on permutation importance:")
    print(top_conditions_for_eng_df[['Original_Condition', 'Importance']])

    eng_feature_names_thresh = set() # Track created names
    for idx, row in top_conditions_for_eng_df.iterrows():
        condition_tuple = row.get('Original_Condition')
        condition_feature_name = row.get('Condition_Feature') # Use generated name if tuple missing

        if not isinstance(condition_tuple, (list, tuple)) or len(condition_tuple) != 2:
             print(f"Skipping condition feature due to missing/invalid Original_Condition format: {condition_feature_name}")
             continue

        feature, condition_str = condition_tuple
        try:
            op = None
            threshold = None
            # Robust parsing
            if '<=' in condition_str:
                parts = condition_str.split('<=')
                if len(parts) == 2: op, threshold = 'le', float(parts[1].strip())
            elif '>' in condition_str:
                parts = condition_str.split('>')
                if len(parts) == 2: op, threshold = 'gt', float(parts[1].strip())

            if op and threshold is not None and feature in X_train.columns:
                 # Use the already generated unique name if available, otherwise create
                 new_feat_name = condition_feature_name or f"eng_cond_{feature}_{op}_{str(threshold).replace('.','p').replace('-','neg')}"

                 if new_feat_name in eng_feature_names_thresh: continue # Avoid duplicates

                 print(f"  Creating threshold feature: {new_feat_name} from ({feature} {condition_str})")
                 if op == 'le':
                      engineered_features_train_df[new_feat_name] = (X_train[feature] <= threshold).astype(int)
                      if feature in X_test.columns: engineered_features_test_df[new_feat_name] = (X_test[feature] <= threshold).astype(int)
                      else: engineered_features_test_df[new_feat_name] = 0
                 elif op == 'gt':
                      engineered_features_train_df[new_feat_name] = (X_train[feature] > threshold).astype(int)
                      if feature in X_test.columns: engineered_features_test_df[new_feat_name] = (X_test[feature] > threshold).astype(int)
                      else: engineered_features_test_df[new_feat_name] = 0

                 engineered_feature_names.append(new_feat_name)
                 eng_feature_names_thresh.add(new_feat_name)
                 created_threshold_features += 1
            # else:
                 # print(f"Could not parse condition string or feature '{feature}' missing: {condition_str}")
        except ValueError as ve:
             print(f"Could not convert threshold to float in condition {condition_tuple}: {ve}")
             continue
        except Exception as e:
            print(f"Could not create feature from condition {condition_tuple}: {e}")
            continue
    print(f"Created {created_threshold_features} threshold-based features.")
elif SURROGATE_MODEL_TYPE == 'random_forest':
     print("Condition importance data not available or empty, skipping threshold feature engineering.")

# --- Create Interaction features based on H-VALUES or SHAP ---
print("\nCreating engineered interaction features...")
N_INTERACTION_FEATURES = 3 # Max interaction features to create
top_pairs = []
interaction_source = "None"

if h_values is not None and not h_values.empty:
     top_pairs = h_values.head(N_INTERACTION_FEATURES).index.tolist()
     interaction_source = "H-statistic"
elif shap_interaction_df is not None and not shap_interaction_df.empty:
     # Extract top pairs from SHAP interaction matrix
     try:
          inter_vals_shap_eng = shap_interaction_df.copy()
          np.fill_diagonal(inter_vals_shap_eng.values, -np.inf)
          inter_vals_shap_eng_stacked = inter_vals_shap_eng.stack().sort_values(ascending=False)
          inter_vals_shap_eng_stacked = inter_vals_shap_eng_stacked[inter_vals_shap_eng_stacked != -np.inf]
          top_pairs = inter_vals_shap_eng_stacked.head(N_INTERACTION_FEATURES).index.tolist()
          interaction_source = "SHAP"
     except Exception as e:
          print(f"Error processing SHAP interactions for feature engineering: {e}")

if top_pairs:
    print(f"Using top {len(top_pairs)} interactions based on {interaction_source}: {top_pairs}")
    eng_feature_names_inter = set()
    for pair in top_pairs:
         if isinstance(pair, tuple) and len(pair) == 2:
            f1, f2 = pair
            # Ensure features exist
            if f1 not in X_train.columns or f2 not in X_train.columns: continue

            new_feat_name = f"eng_{f1}_x_{f2}"
            if new_feat_name in eng_feature_names_inter or new_feat_name in engineered_features_train_df.columns: continue

            print(f"  Creating interaction feature: {new_feat_name}")
            engineered_features_train_df[new_feat_name] = X_train[f1] * X_train[f2]
            # Check if features exist in test before multiplying
            if f1 in X_test.columns and f2 in X_test.columns:
                 engineered_features_test_df[new_feat_name] = X_test[f1] * X_test[f2]
            else:
                 engineered_features_test_df[new_feat_name] = 0 # Or impute differently? Default 0

            engineered_feature_names.append(new_feat_name)
            eng_feature_names_inter.add(new_feat_name)
            created_interaction_features +=1
         else:
             print(f"Skipping interaction feature creation for invalid pair format: {pair}")
    print(f"Created {created_interaction_features} interaction-based features.")
else:
    print("No top interactions identified from H-statistic or SHAP. Skipping interaction feature engineering.")

# --- Align columns in test set ---
if not engineered_features_train_df.empty:
     missing_cols_test_eng = set(engineered_features_train_df.columns) - set(engineered_features_test_df.columns)
     for col in missing_cols_test_eng:
          engineered_features_test_df[col] = 0 # Add missing columns with default value
     # Ensure same column order
     engineered_features_test_df = engineered_features_test_df[engineered_features_train_df.columns]


print(f"\nTotal engineered features created: {len(engineered_feature_names)}")
report_generator.results_dict['actual_eng_features'] = engineered_feature_names # Store names

# %% [markdown]
# ### 14.2 Plot Engineered Feature Distributions

# %%
print("\nGenerating plots for engineered features...")
if not engineered_features_train_df.empty:
    print(f"Plotting distributions for {len(engineered_features_train_df.columns)} engineered features...")
    for eng_feat in engineered_features_train_df.columns:
        try:
            plt.figure(figsize=(10, 4))
            # Use distplot for continuous, countplot for binary/few categories
            if engineered_features_train_df[eng_feat].nunique() <= 10: # Heuristic for categorical/binary
                 sns.countplot(x=engineered_features_train_df[eng_feat])
                 plt.title(f'Count Plot: {eng_feat} (Train Set)')
            else:
                 sns.histplot(engineered_features_train_df[eng_feat], kde=True)
                 plt.title(f'Distribution Plot: {eng_feat} (Train Set)')
            plt.xlabel(eng_feat)
            plt.ylabel('Frequency / Count')
            plt.tight_layout()
            dist_fig = plt.gcf()
            save_plot(dist_fig, f"engineered_dist_{eng_feat}", output_dir=OUTPUT_DIR)
        except Exception as e:
            print(f"Could not plot distribution for engineered feature {eng_feat}: {e}")
            plt.close() # Close plot if error occurred
else:
    print("No engineered features were created to plot.")

# %% [markdown]
# ### 14.3 Linear Model Comparison

# %%
# --- Compare Linear Models ---
rmse_orig, r2_orig, rmse_eng, r2_eng = None, None, None, None # Initialize metrics

# Define combined datasets
X_train_eng = X_train # Default to original if no engineered features
X_test_eng = X_test
if not engineered_features_train_df.empty:
    # Combine original and engineered features
    X_train_eng = pd.concat([X_train, engineered_features_train_df], axis=1)
    X_test_eng = pd.concat([X_test, engineered_features_test_df], axis=1)
    # Sanity check column alignment (should be aligned already)
    X_test_eng = X_test_eng.reindex(columns=X_train_eng.columns, fill_value=0)

print("\nTraining Linear Models for Comparison...")
try:
    # --- Model 1: Original Features ---
    print("Training Linear Model (Original Features)...")
    # Scale features for linear models
    scaler_orig = StandardScaler()
    X_train_scaled_orig = scaler_orig.fit_transform(X_train)
    X_test_scaled_orig = scaler_orig.transform(X_test)

    lr_orig = LinearRegression()
    lr_orig.fit(X_train_scaled_orig, y_train)
    y_pred_test_orig = lr_orig.predict(X_test_scaled_orig)
    rmse_orig = np.sqrt(mean_squared_error(y_test, y_pred_test_orig))
    r2_orig = r2_score(y_test, y_pred_test_orig)
    print(f"  Linear Model (Original): Test RMSE={rmse_orig:.4f}, R2={r2_orig:.4f}")

    # --- Model 2: Original + Engineered Features ---
    if not engineered_features_train_df.empty:
        print("Training Linear Model (Original + Engineered Features)...")
        # Scale combined features
        scaler_eng = StandardScaler()
        X_train_scaled_eng = scaler_eng.fit_transform(X_train_eng)
        X_test_scaled_eng = scaler_eng.transform(X_test_eng)

        lr_eng = LinearRegression()
        lr_eng.fit(X_train_scaled_eng, y_train)
        y_pred_test_eng = lr_eng.predict(X_test_scaled_eng)
        rmse_eng = np.sqrt(mean_squared_error(y_test, y_pred_test_eng))
        r2_eng = r2_score(y_test, y_pred_test_eng)
        print(f"  Linear Model (Enginrd.): Test RMSE={rmse_eng:.4f}, R2={r2_eng:.4f}")
    else:
         print("Skipping Linear Model (Engineered) as no engineered features were created.")
         rmse_eng, r2_eng = np.nan, np.nan # Set to NaN if not run

except Exception as e:
    import traceback
    print(f"Error during Linear Model training or evaluation: {e}")
    # print(traceback.format_exc()) # Uncomment for detail
    # Set results to NaN on error
    rmse_orig, r2_orig = (np.nan, np.nan) if rmse_orig is None else (rmse_orig, r2_orig)
    rmse_eng, r2_eng = np.nan, np.nan


# Store results for summary file and JSON
linear_perf = {
    'rmse_orig': rmse_orig, 'r2_orig': r2_orig,
    'rmse_eng': rmse_eng, 'r2_eng': r2_eng,
    'engineered_features_created': len(engineered_feature_names) > 0,
    'threshold_features_created': created_threshold_features,
    'interaction_features_created': created_interaction_features
}
report_generator.results_dict['linear_model_comparison'] = linear_perf
# Also store individual metrics directly in results_dict for easier access in summary functions
report_generator.results_dict['rmse_orig'] = rmse_orig
report_generator.results_dict['r2_orig'] = r2_orig
report_generator.results_dict['rmse_eng'] = rmse_eng
report_generator.results_dict['r2_eng'] = r2_eng

# %% [markdown]
# ## 15. Final Report Generation

# %% [markdown]
# ### 15.1 Add Dynamic Summary & Limitations

# %%
print("\n--- Generating Dynamic Summary and Adding Limitations ---")
# Add dynamic summary (Section 0) based on collected results
report_generator.add_dynamic_summary() # This reads from results_dict now

# Add limitations section
report_generator.add_limitations_section()

# %% [markdown]
# ### 15.2 Write Detailed Summary File

# %%
print("\n--- 10. Summarizing Key Findings (Detailed Text File) ---")
summary_filename = "summary_findings_detailed.txt" # Name for the text summary
# Call write_summary_file - it will access results from report_generator.results_dict
report_generator.write_summary_file(summary_filename=summary_filename)

# %% [markdown]
# ### 15.3 Save JSON Results

# %%
print("\n--- Saving Analysis Results to JSON ---")
report_generator.save_analysis_results(output_path="analysis_results.json")

# %% [markdown]
# ### 15.4 Generate Markdown Report

# %%
print("\n--- Generating Final Markdown Report ---")
report_generator.generate_markdown_report(filename="SAFE_Analysis_Report.md")

# %% [markdown]
# ## 16. Workflow Complete

# %%
print("\n--- Workflow Complete ---")
print(f"All outputs saved in directory: {OUTPUT_DIR}")
