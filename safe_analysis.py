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
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import plot_tree, export_text
from sklearn.inspection import permutation_importance, partial_dependence, PartialDependenceDisplay
# Import internal function to potentially bypass check_is_fitted issue
from sklearn.inspection import _partial_dependence
from sklearn.utils import resample # For bootstrapping
import shap
from lofo import LOFOImportance, Dataset, plot_importance
from itertools import combinations
from collections import Counter
import warnings
import os

warnings.filterwarnings('ignore')

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
OUTPUT_DIR = 'safe_analysis' # Directory to save plots

# Create output directory if it doesn't exist
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print(f"Created output directory: {OUTPUT_DIR}")

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
        # PDP for feature 1
        pdp_f1 = _partial_dependence.partial_dependence(model, X, features=[f1_idx],
                                                        grid_resolution=30, # Reduced grid for speed
                                                        kind='average').average[0]
        # PDP for feature 2
        pdp_f2 = _partial_dependence.partial_dependence(model, X, features=[f2_idx],
                                                        grid_resolution=30,
                                                        kind='average').average[0]
        # PDP for interaction
        pdp_f1f2 = _partial_dependence.partial_dependence(model, X, features=[(f1_idx, f2_idx)],
                                                        grid_resolution=30,
                                                        kind='average').average[0]

        # H-statistic calculation based on Friedman 2008 "Predictive Learning via Rule Ensembles"
        # Formula: H^2 = sum[(PD_12(x1, x2) - PD_1(x1) - PD_2(x2))^2] / sum[PD_12(x1, x2)^2]
        # where PD are centered partial dependence functions.
        # The calculation below uses the uncentered PDP from sklearn and approximates.

        # Center the 1D PDPs
        pd_f1_centered = pdp_f1 - np.mean(pdp_f1)
        pd_f2_centered = pdp_f2 - np.mean(pdp_f2)

        # Center the 2D PDP
        pd_f1f2_centered = pdp_f1f2 - np.mean(pdp_f1) - np.mean(pdp_f2) + np.mean(pdp_f1f2.ravel())
        # Correction: The above centering might be incorrect. Friedman's paper implies centering
        # the PD_12 term itself before squaring.
        # Let's use a simpler variance-based approximation as before, acknowledging its limits.

        uncentered_pd_f1f2_flat = pdp_f1f2.ravel()
        interaction_effect = uncentered_pd_f1f2_flat - np.repeat(pdp_f1, len(pdp_f2)) - np.tile(pdp_f2, len(pdp_f1))

        numerator = np.mean(interaction_effect**2) # Approximation of numerator
        denominator = np.mean(uncentered_pd_f1f2_flat**2) # Approximation of denominator

        # Original simpler approximation (may be more robust despite less theoretical grounding)
        # joint_effect_variance = np.var(pdp_f1f2.ravel())
        # main_effect_variance = np.var(pdp_f1) + np.var(pdp_f2)
        # h_squared = max(0, joint_effect_variance - main_effect_variance) / joint_effect_variance if joint_effect_variance > 0 else 0

        # Using numerator/denominator approach (can be sensitive)
        h_squared = numerator / denominator if denominator > 1e-8 else 0
        return np.sqrt(max(0, min(h_squared, 1.0))) # Cap H at 1

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
        # Very simplified placeholder: calculate 3D PDP and return its standard deviation
        # A full calculation requires subtracting all lower-order effects (1D, 2D PDPs)
        pdp_f1f2f3 = _partial_dependence.partial_dependence(model, X, features=[(f1_idx, f2_idx, f3_idx)],
                                                          grid_resolution=10, # Very low grid for speed
                                                          kind='average').average[0]
        h_approx = np.std(pdp_f1f2f3) # Use std dev as proxy for interaction strength
        return h_approx
    except Exception as e:
        print(f"Error calculating 3-way H-statistic for ({feature1}, {feature2}, {feature3}): {e}")
        return np.nan

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

# --- 3. EDA Plots ---
print("\n--- 3. Exploratory Data Analysis (EDA) ---")
# a. Time Series Plot
print("Generating Time Series plot...")
fig_ts = px.line(df, y=TARGET_VARIABLE, title=f'{TARGET_VARIABLE} Over Time')
train_end_date = X_train.index.max()
test_start_date = X_test.index.min()
fig_ts.add_vline(x=train_end_date, line_dash="dash", line_color="red")
# Add annotation using appropriate y-coordinate based on data range
y_range = df[TARGET_VARIABLE].max() - df[TARGET_VARIABLE].min()
y_pos = df[TARGET_VARIABLE].min() + y_range * 0.9 # Position annotation near the top
fig_ts.add_annotation(x=train_end_date, y=y_pos, text="Train/Test Split", showarrow=False, yshift=10)
save_plot(fig_ts, "eda_timeseries_target", OUTPUT_DIR)

# b. Feature Distributions
print("Generating Feature Distribution plots...")
for feature in features:
    fig_dist = px.histogram(X_train, x=feature, marginal="rug", title=f'Distribution of {feature} (Train Set)')
    save_plot(fig_dist, f"eda_dist_{feature}", OUTPUT_DIR)

# c. Correlations
print("Generating Correlation heatmap...")
corr_matrix = pd.concat([X_train, y_train], axis=1).corr()
fig_corr = px.imshow(corr_matrix, text_auto=".1f", aspect="auto", # Reduced precision for text
                     title='Correlation Matrix (Train Set)', color_continuous_scale='RdBu_r')
fig_corr.update_xaxes(tickangle=45) # Rotate labels
fig_corr.update_layout(font=dict(size=8)) # Smaller font
save_plot(fig_corr, "eda_correlation_matrix", OUTPUT_DIR)

# d. Scatter Plots (All features vs Target)
print("Generating Scatter plots (each feature vs target)...")
for feature in features:
    try:
        fig_scatter = px.scatter(train_df, x=feature, y=TARGET_VARIABLE,
                                 title=f'Scatter Plot: {feature} vs {TARGET_VARIABLE}',
                                 trendline="ols", trendline_color_override="red") # Add trendline
        save_plot(fig_scatter, f"eda_scatter_{feature}_vs_target", OUTPUT_DIR)
    except Exception as e:
        print(f"Could not generate scatter plot for {feature} vs target: {e}")

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
# Note: LOFO errored out in previous runs (KeyError: 'importance').
# Skipping LOFO calculation.
importance_df = None
print("Skipping LOFO calculation due to previous errors. Check library versions if needed.")

# c/e. SHAP Feature Importance & Summary Plot (on Training Data)
print("\nCalculating SHAP Importance (Train Set)...")
shap_values = None
shap_df = None
try:
    # Use X_train for background AND for calculating SHAP values
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
except Exception as e:
    print(f"Error calculating or plotting SHAP values: {e}")

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
    else:
        print("No valid binary condition features created, skipping importance analysis.")
else:
    print("Skipping condition importance analysis (Not RandomForest or no conditions found).")
# --- End Added Subsection ---

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
print("\nCalculating Three-way H-statistic (approximation, computationally expensive)...")
h_3way = None
if len(features) >= 3:
    # Select top 3 features based on MDI or SHAP (fallback to first 3)
    top_3_features = features[:3] # Default to first 3
    if mdi_df is not None and len(mdi_df) >= 3:
        top_3_features = mdi_df['feature'].head(3).tolist()
    elif shap_df is not None and len(shap_df) >= 3:
        top_3_features = shap_df['feature'].head(3).tolist()

    f1, f2, f3 = top_3_features
    print(f"Calculating for triplet: ({f1}, {f2}, {f3})")
    # Use a smaller sample for 3-way PDP
    X_sample_3way = X_train.sample(min(50, len(X_train)), random_state=42) if len(X_train) > 50 else X_train
    h_3way = friedman_h_3way(surrogate_model, X_sample_3way, f1, f2, f3)
    if h_3way is not None and not np.isnan(h_3way):
        print(f"  Approximate 3-way H-statistic (Std Dev of 3D PDP) for ({f1}, {f2}, {f3}): {h_3way:.4f}")
    else:
        print(f"  Could not calculate 3-way H-statistic for ({f1}, {f2}, {f3})")
else:
    print("Not enough features (need 3+) to calculate 3-way interaction example.")

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

print("\nGenerating SHAP Dependence and PDP Interaction plots...")
if shap_values is not None:
    # Determine top interaction pairs based on H-statistic if available
    top_interaction_pairs = []
    if h_values is not None and not h_values.empty:
        top_interaction_pairs = h_values.head(min(3, len(h_values))).index.tolist()
    else:
        # Fallback to combinations of top SHAP features if H-stats failed
        if len(top_shap_features) >= 2:
            top_interaction_pairs = list(combinations(top_shap_features, 2))[:3]
        print("Using fallback interaction pairs based on SHAP importance.")

    for feature in top_shap_features:
         try:
             print(f"  Generating SHAP Dependence plot for {feature}")
             fig_shap_dep, ax_shap_dep = plt.subplots()
             shap.dependence_plot(feature, shap_values.values, X_train, interaction_index="auto", ax=ax_shap_dep, show=False)
             plt.tight_layout()
             save_plot(fig_shap_dep, f"interaction_shap_dependence_{feature}", OUTPUT_DIR)
         except Exception as e:
             print(f"Could not generate SHAP dependence plot for {feature}: {e}")

    for pair in top_interaction_pairs:
        # Ensure pair is a tuple of two elements
        if isinstance(pair, tuple) and len(pair) == 2:
            f1, f2 = pair
            try:
                print(f"  Generating 2D PDP for interaction: ({f1}, {f2})")
                fig_pdp2d, ax_pdp2d = plt.subplots(figsize=(8, 7))
                display = PartialDependenceDisplay.from_estimator(
                    surrogate_model,
                    X_train,
                    features=[(f1, f2)],
                    kind='average',
                    ax=ax_pdp2d
                )
                ax_pdp2d.set_title(f'2D Partial Dependence: {f1} vs {f2}')
                plt.tight_layout()
                save_plot(fig_pdp2d, f"interaction_pdp_2d_{f1}_{f2}", OUTPUT_DIR)

                print(f"  Generating SHAP interaction plot for: ({f1}, {f2})")
                fig_shap_int, ax_shap_int = plt.subplots()
                shap.dependence_plot(f1, shap_values.values, X_train, interaction_index=f2, ax=ax_shap_int, show=False)
                plt.tight_layout()
                save_plot(fig_shap_int, f"interaction_shap_{f1}_vs_{f2}", OUTPUT_DIR)

                # Create 3D Surface Plot for PDP
                try:
                    print(f"  Generating 3D PDP Surface plot for interaction: ({f1}, {f2})")
                    # Get feature indices
                    feature_names_pdp = X_train.columns.tolist()
                    f1_idx = feature_names_pdp.index(f1)
                    f2_idx = feature_names_pdp.index(f2)

                    # Use sklearn's partial_dependence directly for calculation with indices
                    pdp_results = partial_dependence(
                        surrogate_model,
                        X_train, # Use training data
                        features=[(f1_idx, f2_idx)], # Pass indices
                        kind='average',
                        grid_resolution=20 # Lower resolution for 3D plot speed
                    )
                    # Correctly access results: average is the 2D array, values is list of grids
                    pdp_values = pdp_results.average
                    grid_f1 = pdp_results.values[0]
                    grid_f2 = pdp_results.values[1]

                    # Create the 3D surface plot using Plotly
                    fig_pdp3d = go.Figure(data=[go.Surface(z=pdp_values, x=grid_f1, y=grid_f2,
                                                           colorscale='Viridis')])
                    fig_pdp3d.update_layout(title=f'3D Partial Dependence: {f1} vs {f2}',
                                          scene = dict(
                                              xaxis_title=f1,
                                              yaxis_title=f2,
                                              zaxis_title='Partial Dependence'),
                                          autosize=True, margin=dict(l=65, r=50, b=65, t=90))
                    save_plot(fig_pdp3d, f"interaction_pdp_3d_{f1}_{f2}", OUTPUT_DIR)

                except Exception as e:
                    print(f"Could not generate 3D PDP surface plot for ({f1}, {f2}): {e}")

            except Exception as e:
                 print(f"Could not generate interaction plot for ({f1}, {f2}): {e}")
        else:
             print(f"Skipping interaction plot for invalid pair: {pair}")
else:
    print("SHAP values not available, skipping SHAP dependence/interaction plots.")

# --- 8. Causality (Optional Placeholder - Enhanced Description) ---
print("\n--- 8. Causality Analysis (Placeholder) ---")
print("Causality analysis aims to understand cause-and-effect relationships, going beyond correlation.")
print("This requires domain expertise and specific assumptions about the data generating process.")
print("Libraries like DoWhy and CausalNex can help structure this analysis:")
print("  - DoWhy: Follows a 4-step process: Model (define graph), Identify (find estimand), Estimate (compute effect), Refute (validate).")
print("    Example: model = CausalModel(data=df, treatment='feature_X', outcome='target', graph=your_defined_graph) # graph uses DOT format string or networkx object")
print("  - CausalNex: Helps learn structure (Bayesian Networks) from data and visualize graphs.")
print("    Example: sm = StructureModel(); sm.add_edges_from([(f1, f2), ...]); plot(sm)")
print("Implementation requires careful definition of a causal graph based on domain knowledge or discovery algorithms, and potentially installing these libraries.")

# --- 9. Interpretability Summary (Focus on SHAP) ---
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
print("\n--- 10. Summarizing Key Findings ---")
summary_filename = os.path.join(OUTPUT_DIR, "summary_findings.txt")
with open(summary_filename, "w") as f:
    f.write("--- Summary of Key Findings ---\n\n")

    # Combine importance scores
    f.write(f"--- Feature Importance Rankings (Full Lists) ---\n")
    all_imp = {}
    # Use updated DataFrames calculated on train set
    if perm_df is not None: all_imp['Permutation_Train'] = perm_df.set_index('feature')['importance_mean']
    if shap_df is not None: all_imp['SHAP_Train'] = shap_df.set_index('feature')['shap_importance']
    if mdi_df is not None: all_imp['MDI'] = mdi_df.set_index('feature')['mdi_importance']

    combined_imp = pd.DataFrame() # Initialize
    if all_imp:
        combined_imp = pd.DataFrame(all_imp)
        rank_cols = []
        f.write("\nImportance Scores:\n")
        f.write(combined_imp.to_string())
        f.write("\n\nImportance Ranks (Lower is better):\n")
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
            # Write full ranked table
            f.write(combined_imp[['mean_rank'] + rank_cols].to_string())
            f.write("\n\n")
            print("Top Features (Mean Rank - based on Train Set calculations):")
            print(combined_imp[['mean_rank']].head(N_TOP_FEATURES))
        else:
             f.write("No ranks calculated.\n\n")
    else:
         f.write("No importance scores available to combine.\n\n")

    # Add Top Rule Conditions ranked by IMPORTANCE
    if SURROGATE_MODEL_TYPE == 'random_forest' and not condition_importance_df.empty:
         f.write(f"--- Top {len(condition_importance_df)} Rule Conditions by Importance (Permutation) ---\n")
         # Use the DataFrame with importance scores
         f.write(condition_importance_df[['Original_Condition', 'Importance']].to_string(index=False))
         f.write("\n\n")
    elif SURROGATE_MODEL_TYPE == 'random_forest':
         f.write("--- Rule Conditions ---\nCondition importance could not be calculated.\n\n")

    # Add Top Interactions (H-statistic)
    if h_values is not None and not h_values.empty:
        f.write(f"--- Top Feature Interactions (Approx. H-statistic) ---\n")
        # Write full list of interactions
        f.write(h_values.to_string())
        f.write("\n\n")
    else:
        f.write("--- Feature Interactions ---\nNo valid H-statistic interactions calculated.\n\n")

    # Add Interaction Stability
    if interaction_stability:
        f.write("--- Interaction Stability (Mean +/- Std Dev H-statistic across Bootstraps) ---\n")
        for pair, stats in interaction_stability.items():
            f.write(f"  {pair}: {stats['mean']:.3f} +/- {stats['std']:.3f}\n")
        f.write("\n")

    # Add 3-way H-stat result
    if h_3way is not None and not np.isnan(h_3way) and 'top_3_features' in locals():
         f.write("--- Approx. 3-Way Interaction (Std Dev of 3D PDP) ---\n")
         f.write(f"  Triplet {top_3_features}: {h_3way:.4f}\n\n")

print(f"Saved summary findings to: {summary_filename}")

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
                     continue # Avoid duplicates

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

print("\n--- Workflow Complete ---")
print(f"All outputs saved in directory: {OUTPUT_DIR}") 