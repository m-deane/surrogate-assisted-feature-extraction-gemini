#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Surrogate-Assisted Feature Extraction (SAFE) Workflow
for Petroleum Price Analysis with Feature Importance and Interactions

This script implements a workflow for analyzing the preem.csv dataset using
surrogate-assisted feature extraction, with special focus on feature importance
and interactions between variables. The target variable is "target" and "totaltar"
is guaranteed to be included in the model.
"""

# %%
# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, IsolationForest
from sklearn.tree import DecisionTreeRegressor, plot_tree, export_text
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.inspection import permutation_importance, partial_dependence
from sklearn.linear_model import LinearRegression
from sklearn.covariance import EllipticEnvelope
import ruptures as rpt
from scipy.cluster.hierarchy import ward, dendrogram, cut_tree
from scipy import stats
from kneed import KneeLocator
import shap
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
import warnings
import ruptures as rpt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.tree import export_graphviz
import graphviz
from sklearn import tree as _tree
import itertools
from tqdm import tqdm

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# %%
# Load and inspect the dataset
print("Loading and inspecting the dataset...")

data = pd.read_csv("_data/preem.csv")
print(f"Dataset shape: {data.shape}")
print("\nFirst few rows:")
print(data.head())

print("\nData information:")
print(data.info())

print("\nSummary statistics:")
print(data.describe())

# Check for missing values
missing_values = data.isnull().sum()
print("\nMissing values:")
print(missing_values)

# %%
# Enhanced data preprocessing with anomaly detection
print("\nPreprocessing the data with anomaly detection...")

def detect_anomalies(df, numeric_cols, methods=['iqr', 'isolation_forest', 'zscore'], 
                    contamination=0.1, zscore_threshold=3):
    """
    Detect anomalies using multiple methods
    
    Parameters:
    -----------
    df : pandas DataFrame
        Input data
    numeric_cols : list
        List of numeric columns to check for anomalies
    methods : list
        List of methods to use ['iqr', 'isolation_forest', 'zscore']
    contamination : float
        Expected proportion of outliers (for Isolation Forest)
    zscore_threshold : float
        Z-score threshold for outlier detection
        
    Returns:
    --------
    dict : Dictionary containing anomaly indices for each method
    """
    anomalies = {}
    
    if 'iqr' in methods:
        # IQR method
        anomalies['iqr'] = set()
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)].index
            anomalies['iqr'].update(outliers)
    
    if 'isolation_forest' in methods:
        # Isolation Forest for multivariate outlier detection
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        yhat = iso_forest.fit_predict(df[numeric_cols])
        anomalies['isolation_forest'] = set(df[yhat == -1].index)
    
    if 'zscore' in methods:
        # Z-score method
        anomalies['zscore'] = set()
        for col in numeric_cols:
            z_scores = np.abs(stats.zscore(df[col]))
            outliers = df[z_scores > zscore_threshold].index
            anomalies['zscore'].update(outliers)
    
    return anomalies

def visualize_anomalies(df, numeric_cols, anomalies, max_cols=3):
    """Visualize detected anomalies for each feature"""
    n_cols = min(max_cols, len(numeric_cols))
    n_rows = int(np.ceil(len(numeric_cols) / n_cols))
    
    plt.figure(figsize=(15, 5 * n_rows))
    
    for i, col in enumerate(numeric_cols):
        plt.subplot(n_rows, n_cols, i + 1)
        
        # Plot normal points
        normal_idx = set(df.index) - set().union(*anomalies.values())
        plt.scatter(df.index[list(normal_idx)], df[col].iloc[list(normal_idx)], 
                   c='blue', label='Normal', alpha=0.5)
        
        # Plot anomalies from each method with different colors
        colors = ['red', 'green', 'orange']
        for (method, indices), color in zip(anomalies.items(), colors):
            if indices:
                plt.scatter(df.index[list(indices)], df[col].iloc[list(indices)], 
                          c=color, label=f'{method} anomalies', alpha=0.7)
        
        plt.title(f'Anomalies in {col}')
        plt.xlabel('Index')
        plt.ylabel('Value')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig('anomalies_detection.png')
    print("Anomalies visualization saved as 'anomalies_detection.png'")

def clean_anomalies(df, numeric_cols, anomalies, method='cap'):
    """
    Clean detected anomalies using specified method
    
    Parameters:
    -----------
    df : pandas DataFrame
        Input data
    numeric_cols : list
        List of numeric columns to clean
    anomalies : dict
        Dictionary of anomaly indices for each method
    method : str
        Cleaning method ('cap' or 'remove')
    
    Returns:
    --------
    pandas DataFrame : Cleaned data
    """
    df_cleaned = df.copy()
    
    # Combine all anomaly indices
    all_anomalies = set().union(*anomalies.values())
    
    if method == 'remove':
        # Remove all detected anomalies
        df_cleaned = df_cleaned.drop(index=list(all_anomalies))
    
    elif method == 'cap':
        # Cap anomalies at percentile values
        for col in numeric_cols:
            lower_bound = df[col].quantile(0.01)
            upper_bound = df[col].quantile(0.99)
            df_cleaned.loc[list(all_anomalies), col] = df_cleaned.loc[list(all_anomalies), col].clip(lower_bound, upper_bound)
    
    return df_cleaned

# Function to handle missing values
def handle_missing_values(df):
    """Handle missing values in the dataset"""
    # Forward fill for time series data
    df_filled = df.fillna(method='ffill')
    # If any values remain missing, use backward fill
    df_filled = df_filled.fillna(method='bfill')
    # If STILL any values remain missing (e.g., if entire column is NA), fill with 0
    df_filled = df_filled.fillna(0)
    return df_filled

# Apply preprocessing pipeline
print("\nApplying preprocessing pipeline...")

# Handle missing values first
data_processed = handle_missing_values(data)

# Convert date to datetime 
data_processed['date'] = pd.to_datetime(data_processed['date'])
data_processed = data_processed.sort_values('date')

# Get numeric columns for anomaly detection (excluding date and target)
numeric_cols = data_processed.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols = [col for col in numeric_cols if col not in ['target']]

# Detect anomalies
print("\nDetecting anomalies...")
anomalies = detect_anomalies(data_processed, numeric_cols, 
                           methods=['iqr', 'isolation_forest', 'zscore'],
                           contamination=0.1, zscore_threshold=3)

# Visualize anomalies
print("\nVisualizing detected anomalies...")
visualize_anomalies(data_processed, numeric_cols, anomalies)

# Print anomaly statistics
print("\nAnomaly detection statistics:")
for method, indices in anomalies.items():
    print(f"{method}: {len(indices)} anomalies detected ({len(indices)/len(data_processed)*100:.1f}% of data)")

# Clean anomalies
print("\nCleaning anomalies...")
data_processed = clean_anomalies(data_processed, numeric_cols, anomalies, method='cap')

# Visualize after cleaning
print("\nVisualizing data after anomaly cleaning...")
plt.figure(figsize=(15, 5))
for i, col in enumerate(numeric_cols[:3]):  # Show first 3 columns as example
    plt.subplot(1, 3, i + 1)
    plt.plot(data_processed['date'], data_processed[col], label='Cleaned')
    plt.title(f'Cleaned {col}')
    plt.xticks(rotation=45)
    plt.legend()
plt.tight_layout()
plt.savefig('anomalies_cleaned.png')
print("Cleaned data visualization saved as 'anomalies_cleaned.png'")

# Ensure totaltar is included
if 'totaltar' not in data_processed.columns:
    print("Warning: 'totaltar' not found in the dataset. Please check the data.")
else:
    print("'totaltar' found in the dataset and will be preserved in the model.")

# Split the data chronologically for time series
print("\nSplitting the data chronologically...")
train_size = int(len(data_processed) * 0.8)
X_train = data_processed.iloc[:train_size].drop(['date', 'target'], axis=1)
y_train = data_processed.iloc[:train_size]['target']
X_test = data_processed.iloc[train_size:].drop(['date', 'target'], axis=1)
y_test = data_processed.iloc[train_size:]['target']

print(f"Training set size: {X_train.shape[0]} samples")
print(f"Testing set size: {X_test.shape[0]} samples")

# %%
# Initial feature importance assessment
print("\nPerforming initial feature importance assessment...")

# Train the surrogate model
print("Training the surrogate model (Gradient Boosting Regressor)...")
surrogate_model = GradientBoostingRegressor(
    n_estimators=200,
    max_depth=3,
    learning_rate=0.1,
    loss='squared_error',
    random_state=42
)
surrogate_model.fit(X_train, y_train)

# Enhance the feature importance calculation with LOFO (Leave One Feature Out)
def calculate_feature_importance(model, X_train, X_test, y_test):
    """Calculate multiple feature importance metrics for the model"""
    # Built-in feature importance (MDI)
    mdi_importance = pd.Series(model.feature_importances_, index=X_train.columns)
    
    # Permutation importance
    perm_importance = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
    perm_importance_df = pd.Series(perm_importance.importances_mean, index=X_train.columns)
    
    # LOFO importance (Leave One Feature Out)
    lofo_importance = {}
    # Compute baseline score with all features
    baseline_score = r2_score(y_test, model.predict(X_test))
    
    # For each feature, measure performance drop when it's removed
    for feature in X_train.columns:
        # Skip totaltar if it's meant to be preserved
        if feature == 'totaltar' and 'totaltar' in X_train.columns:
            # Still calculate but mark as protected
            X_train_subset = X_train.drop(columns=[feature])
            X_test_subset = X_test.drop(columns=[feature])
            
            # Train model without this feature
            model_subset = GradientBoostingRegressor(
                n_estimators=200, max_depth=3, learning_rate=0.1, 
                loss='squared_error', random_state=42
            ).fit(X_train_subset, y_train)
            
            # Measure performance drop
            subset_score = r2_score(y_test, model_subset.predict(X_test_subset))
            lofo_importance[feature] = baseline_score - subset_score
            continue
            
        X_train_subset = X_train.drop(columns=[feature])
        X_test_subset = X_test.drop(columns=[feature])
        
        # Train model without this feature
        model_subset = GradientBoostingRegressor(
            n_estimators=200, max_depth=3, learning_rate=0.1, 
            loss='squared_error', random_state=42
        ).fit(X_train_subset, y_train)
        
        # Measure performance drop
        subset_score = r2_score(y_test, model_subset.predict(X_test_subset))
        lofo_importance[feature] = baseline_score - subset_score
        
    lofo_importance_df = pd.Series(lofo_importance)
    
    # Combine importances
    importance_df = pd.DataFrame({
        'MDI': mdi_importance,
        'Permutation': perm_importance_df,
        'LOFO': lofo_importance_df
    }).sort_values('LOFO', ascending=False)
    
    # Ensure totaltar is always included
    if 'totaltar' in importance_df.index:
        importance_df.loc['totaltar', 'Protected'] = True
    
    return importance_df

# Calculate and display feature importances
importance_df = calculate_feature_importance(surrogate_model, X_train, X_test, y_test)
print("\nFeature importance (top 10):")
print(importance_df.head(10))

# Create a more comprehensive feature importance plot showing all metrics
plt.figure(figsize=(14, 10))
importance_long = importance_df.reset_index().melt(
    id_vars='index', 
    value_vars=['MDI', 'Permutation', 'LOFO'],
    var_name='Method', 
    value_name='Importance'
)
sns.barplot(x='index', y='Importance', hue='Method', data=importance_long)
plt.title('Feature Importance - Multiple Metrics Comparison')
plt.xlabel('Feature')
plt.ylabel('Importance Score')
plt.xticks(rotation=45, ha='right')

# Add explanation text
plt.figtext(0.5, 0.02, 
            "HOW TO INTERPRET THIS PLOT:\n"
            "• MDI (Mean Decrease in Impurity): Shows how much each feature reduces weighted impurity across all trees.\n"
            "  Higher values indicate features that better split the data.\n"
            "• Permutation: Measures decrease in model performance when feature values are randomly shuffled.\n"
            "  Negative values suggest the feature might be noisy or redundant.\n"
            "• LOFO (Leave One Feature Out): Shows performance drop when each feature is removed.\n"
            "  Higher positive values indicate more important features; negative values suggest potential overfitting.\n"
            "Note: Protected features (like 'totaltar') are always included regardless of importance.",
            ha='center', fontsize=10, bbox={"facecolor":"lightgrey", "alpha":0.5, "pad":5})

plt.tight_layout()
plt.subplots_adjust(bottom=0.25)  # Make room for the explanation text
plt.savefig('feature_importance_all_methods.png')
print("All feature importance methods comparison saved as 'feature_importance_all_methods.png'")

# %%
# SHAP-based deep feature analysis
print("\nPerforming SHAP-based feature importance analysis...")

# Create SHAP explainer for the surrogate model
explainer = shap.TreeExplainer(surrogate_model)
shap_values = explainer.shap_values(X_test)

# Calculate mean absolute SHAP values for feature importance
shap_importance = pd.DataFrame({
    'mean_abs_shap': np.abs(shap_values).mean(0)
}, index=X_test.columns).sort_values('mean_abs_shap', ascending=False)

print("\nSHAP feature importance (top 10):")
print(shap_importance.head(10))

# SHAP summary plot
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
plt.tight_layout()
plt.savefig('shap_summary_plot.png')
print("SHAP summary plot saved as 'shap_summary_plot.png'")

plt.figure(figsize=(14, 10))
shap.summary_plot(shap_values, X_test, show=False)
plt.tight_layout()
plt.savefig('shap_detailed_summary_plot.png')
print("Detailed SHAP summary plot saved as 'shap_detailed_summary_plot.png'")

# %%
# Feature interaction strength quantification
print("\nAnalyzing feature interactions...")

# Calculate SHAP interaction values
print("Calculating SHAP interaction values (this may take some time)...")
interaction_values = explainer.shap_interaction_values(X_test.iloc[:100])  # Using subset for speed

# Create interaction strength matrix
interaction_strength = np.zeros((X_test.shape[1], X_test.shape[1]))
for i in range(X_test.shape[1]):
    for j in range(X_test.shape[1]):
        interaction_strength[i, j] = np.abs(interaction_values[:, i, j]).mean()

interaction_df = pd.DataFrame(
    interaction_strength, 
    index=X_test.columns, 
    columns=X_test.columns
)

# Display top interactions
print("\nTop 10 feature interactions:")
interaction_flat = interaction_df.unstack()
interaction_flat = interaction_flat[interaction_flat > 0]  # Remove zero interactions
top_interactions = interaction_flat.sort_values(ascending=False)
print(top_interactions.head(10))

# Plot interaction heatmap
plt.figure(figsize=(14, 12))
mask = np.zeros_like(interaction_df)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(interaction_df, cmap="YlOrRd", mask=mask, 
            linewidths=0.5, annot=True, fmt=".2f", square=True)
plt.title('Feature Interaction Strength (SHAP Interaction Values)')
plt.tight_layout()
plt.savefig('interaction_heatmap.png')
print("Interaction heatmap saved as 'interaction_heatmap.png'")

# %%
# Interaction Network Graph
print("\nCreating feature interaction network graph...")

def plot_interaction_network(interaction_df, feature_importance, threshold=0.01):
    """Plot network graph of feature interactions"""
    # Create graph
    G = nx.Graph()
    
    # Normalize importances for node sizing
    max_importance = feature_importance['mean_abs_shap'].max()
    normalized_importance = feature_importance['mean_abs_shap'] / max_importance
    
    # Add nodes
    for feature in interaction_df.columns:
        # Size nodes by feature importance
        size = normalized_importance.get(feature, 0.1) * 1000
        G.add_node(feature, size=size)
    
    # Add edges
    for i in range(len(interaction_df)):
        for j in range(i+1, len(interaction_df)):
            strength = interaction_df.iloc[i, j]
            if strength > threshold:
                G.add_edge(
                    interaction_df.columns[i], 
                    interaction_df.columns[j], 
                    weight=strength
                )
    
    # Create plot
    plt.figure(figsize=(16, 16))
    pos = nx.spring_layout(G, k=0.3, iterations=50)
    
    # Draw nodes
    node_sizes = [G.nodes[n].get('size', 300) for n in G.nodes]
    node_colors = ['red' if n == 'totaltar' else 'skyblue' for n in G.nodes]
    
    nx.draw_networkx_nodes(G, pos, 
                          node_size=node_sizes,
                          node_color=node_colors, 
                          alpha=0.8)
    
    # Draw edges with weights as thickness
    edge_weights = [G.edges[e]['weight'] * 10 for e in G.edges]
    nx.draw_networkx_edges(G, pos, width=edge_weights, alpha=0.5)
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=12)
    
    # Add explanation text to the figure
    plt.figtext(0.5, 0.01, 
                "HOW TO READ THIS PLOT:\n"
                "• Node size represents feature importance (larger = more important)\n"
                "• Edge thickness represents interaction strength between features\n"
                "• Red node highlights 'totaltar' variable\n"
                "• Closely positioned nodes have stronger interactions",
                ha="center", fontsize=12, bbox={"facecolor":"lightgrey", "alpha":0.5, "pad":5})
    
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('interaction_network.png', bbox_inches='tight')
    return G

# Create the network graph
interaction_network = plot_interaction_network(
    interaction_df, 
    shap_importance, 
    threshold=0.01
)
print("Interaction network graph saved as 'interaction_network.png'")

# %%
# SAFE Transformer Implementation
print("\nImplementing SAFE (Surrogate-Assisted Feature Extraction)...")

def safe_transform_numeric(X, surrogate_model, feature_name, penalty=0.8):
    """Transform a numeric feature using the SAFE method"""
    # Get the column index
    feature_idx = list(X.columns).index(feature_name)
    
    # Get feature values
    feature_values = X[feature_name].values
    min_val, max_val = np.min(feature_values), np.max(feature_values)
    
    # Create a grid of values
    grid_points = np.linspace(min_val, max_val, 1000)
    
    # Create copies of X with the feature set to each grid point
    pdp_data = []
    for point in grid_points:
        X_copy = X.copy()
        X_copy[feature_name] = point
        pdp_data.append(surrogate_model.predict(X_copy).mean())
    
    # Find changepoints
    algo = rpt.Pelt(model="l2").fit(np.array(pdp_data))
    changepoints = algo.predict(pen=penalty)
    
    # Convert changepoint indices to feature values
    changepoint_values = [grid_points[i] for i in changepoints[:-1]]
    
    # Create transformed features
    transformed_feature_data = np.zeros((len(X), len(changepoint_values)))
    for i, x in enumerate(feature_values):
        for j, threshold in enumerate(changepoint_values):
            if x >= threshold:
                transformed_feature_data[i, j] = 1
    
    # Create column names for the transformed features
    if not changepoint_values:
        # If no changepoints found, return binary feature based on median
        median_val = np.median(feature_values)
        transformed_feature_data = np.zeros((len(X), 1))
        transformed_feature_data[:, 0] = (feature_values >= median_val).astype(int)
        column_names = [f"{feature_name}_>={median_val:.2f}"]
    else:
        column_names = [f"{feature_name}_>={val:.2f}" for val in changepoint_values]
    
    return pd.DataFrame(transformed_feature_data, columns=column_names, index=X.index)

def apply_safe_transformation(X_train, X_test, surrogate_model, important_features, penalty=0.8):
    """Apply SAFE transformation to selected features"""
    # Always include totaltar
    if 'totaltar' in X_train.columns and 'totaltar' not in important_features:
        important_features = list(important_features) + ['totaltar']
    
    # Transform each important feature
    transformed_train_dfs = []
    transformed_test_dfs = []
    
    for feature in important_features:
        print(f"Transforming feature: {feature}")
        
        # Apply transformation to training data
        transformed_train = safe_transform_numeric(X_train, surrogate_model, feature, penalty)
        transformed_train_dfs.append(transformed_train)
        
        # Apply transformation to test data using the SAME transformation
        # Get the column names from the transformed train data
        train_col_names = transformed_train.columns
        
        # Get threshold values from column names
        thresholds = []
        for col in train_col_names:
            # Extract threshold value from column name (e.g., "feature_>=10.5" -> 10.5)
            threshold_str = col.split('>=')[1]
            thresholds.append(float(threshold_str))
        
        # Apply thresholds to test data
        feature_values = X_test[feature].values
        transformed_data = np.zeros((len(X_test), len(thresholds)))
        
        for i, x in enumerate(feature_values):
            for j, threshold in enumerate(thresholds):
                if x >= threshold:
                    transformed_data[i, j] = 1
        
        # Create DataFrame with the same column names as training
        transformed_test = pd.DataFrame(transformed_data, columns=train_col_names, index=X_test.index)
        transformed_test_dfs.append(transformed_test)
    
    # Combine transformed features
    X_train_transformed = pd.concat(transformed_train_dfs, axis=1)
    X_test_transformed = pd.concat(transformed_test_dfs, axis=1)
    
    return X_train_transformed, X_test_transformed

# Select important features for transformation
# Choose top features but ensure totaltar is included
important_features = list(importance_df.index[:10])
if 'totaltar' not in important_features and 'totaltar' in X_train.columns:
    important_features.append('totaltar')

# Apply SAFE transformation
X_train_transformed, X_test_transformed = apply_safe_transformation(
    X_train, X_test, surrogate_model, important_features
)

print(f"\nTransformed training data shape: {X_train_transformed.shape}")
print(f"Transformed test data shape: {X_test_transformed.shape}")

# %%
# Create interaction features
print("\nCreating explicit interaction features...")

def create_interaction_features(X_train, X_test, interaction_df, threshold=0.01):
    """Create explicit interaction features for strongly interacting pairs"""
    train_interaction_features = pd.DataFrame(index=X_train.index)
    test_interaction_features = pd.DataFrame(index=X_test.index)
    
    # Get pairs above threshold
    pairs = []
    for i in range(len(interaction_df)):
        for j in range(i+1, len(interaction_df)):
            if interaction_df.iloc[i, j] > threshold:
                pairs.append((interaction_df.columns[i], interaction_df.columns[j]))
    
    print(f"Creating {len(pairs)} interaction features...")
    
    for i, j in pairs:
        # Create multiplicative interaction
        name = f"{i}_x_{j}"
        train_interaction_features[name] = X_train[i] * X_train[j]
        test_interaction_features[name] = X_test[i] * X_test[j]
        
        # Create binary threshold interactions (using median)
        median_i = X_train[i].median()
        median_j = X_train[j].median()
        
        name = f"{i}_>{median_i:.2f}_AND_{j}_>{median_j:.2f}"
        train_interaction_features[name] = ((X_train[i] > median_i) & (X_train[j] > median_j)).astype(int)
        test_interaction_features[name] = ((X_test[i] > median_i) & (X_test[j] > median_j)).astype(int)
    
    return train_interaction_features, test_interaction_features

# Create interaction features
interaction_train, interaction_test = create_interaction_features(
    X_train, X_test, interaction_df, threshold=0.02
)

# Combine with SAFE transformed features
X_train_final = pd.concat([X_train_transformed, interaction_train], axis=1)
X_test_final = pd.concat([X_test_transformed, interaction_test], axis=1)

print(f"\nFinal training data shape: {X_train_final.shape}")
print(f"Final test data shape: {X_test_final.shape}")

# %%
# Train decision tree on transformed features
print("\nTraining decision tree on transformed features...")

def train_decision_tree(X_train, y_train, X_test, y_test, max_depth=5):
    """Train a decision tree with controlled complexity"""
    # Ensure X_test has the same columns as X_train
    missing_cols = set(X_train.columns) - set(X_test.columns)
    for col in missing_cols:
        X_test[col] = 0  # Add missing columns with zeros
    
    # Ensure columns are in the same order
    X_test = X_test[X_train.columns]
    
    dt_model = DecisionTreeRegressor(max_depth=max_depth, random_state=42)
    dt_model.fit(X_train, y_train)
    
    # Evaluate performance
    train_score = r2_score(y_train, dt_model.predict(X_train))
    test_score = r2_score(y_test, dt_model.predict(X_test))
    test_mse = mean_squared_error(y_test, dt_model.predict(X_test))
    test_mae = mean_absolute_error(y_test, dt_model.predict(X_test))
    
    print(f"Decision Tree Training R²: {train_score:.4f}")
    print(f"Decision Tree Test R²: {test_score:.4f}")
    print(f"Decision Tree Test MSE: {test_mse:.4f}")
    print(f"Decision Tree Test MAE: {test_mae:.4f}")
    
    return dt_model

# Train the decision tree
dt_model = train_decision_tree(X_train_final, y_train, X_test_final, y_test, max_depth=5)
train_score = r2_score(y_train, dt_model.predict(X_train_final))
test_score = r2_score(y_test, dt_model.predict(X_test_final))

# Compare with surrogate model performance
surrogate_test_score = r2_score(y_test, surrogate_model.predict(X_test))
surrogate_test_mse = mean_squared_error(y_test, surrogate_model.predict(X_test))

print(f"\nSurrogate Model Test R²: {surrogate_test_score:.4f}")
print(f"Surrogate Model Test MSE: {surrogate_test_mse:.4f}")

# %%
# Visualize decision tree
print("\nVisualizing decision tree...")

plt.figure(figsize=(20, 12))
plot_tree(dt_model, filled=True, feature_names=X_train_final.columns, 
          rounded=True, fontsize=10, proportion=True, precision=2)
plt.tight_layout()
plt.savefig('decision_tree.png')
print("Decision tree visualization saved as 'decision_tree.png'")

# Get text representation of the tree
tree_rules = export_text(dt_model, feature_names=list(X_train_final.columns))
with open('decision_tree_rules.txt', 'w') as f:
    f.write(tree_rules)
print("Decision tree rules saved to 'decision_tree_rules.txt'")

# %%
# Feature importance in the final decision tree
print("\nAnalyzing feature importance in the final decision tree...")

dt_importance = pd.Series(dt_model.feature_importances_, index=X_train_final.columns)
dt_importance = dt_importance.sort_values(ascending=False)

print("\nTop 10 features in the decision tree:")
print(dt_importance.head(10))

plt.figure(figsize=(12, 8))
dt_importance.head(20).plot(kind='barh')
plt.title('Feature Importance in Decision Tree')
plt.tight_layout()
plt.savefig('dt_feature_importance.png')
print("Decision tree feature importance plot saved as 'dt_feature_importance.png'")

# %%
# Analyze path frequency and contribution
print("\nAnalyzing decision path frequency and contribution...")

def analyze_decision_paths(dt_model, X, y):
    """Analyze decision paths in the tree and their contributions"""
    # Get decision path
    decision_path = dt_model.decision_path(X)
    leaf_ids = dt_model.apply(X)
    
    # Count samples in each leaf
    leaf_counts = np.bincount(leaf_ids, minlength=dt_model.tree_.node_count)
    
    # Calculate average target value for each leaf
    leaf_values = {}
    for leaf_id in np.unique(leaf_ids):
        leaf_samples = np.where(leaf_ids == leaf_id)[0]
        leaf_values[leaf_id] = y.iloc[leaf_samples].mean()
    
    # Convert to DataFrame for easier analysis
    leaf_df = pd.DataFrame({
        'leaf_id': list(leaf_values.keys()),
        'samples': [leaf_counts[i] for i in leaf_values.keys()],
        'mean_target': list(leaf_values.values())
    }).sort_values('samples', ascending=False)
    
    return leaf_df

# Analyze paths in the training set
leaf_analysis = analyze_decision_paths(dt_model, X_train_final, y_train)
print("\nMost frequent decision paths:")
print(leaf_analysis.head(5))

# %%
# Time-based feature importance analysis
print("\nPerforming time-based feature importance analysis...")

def analyze_temporal_importance(X, y, feature_names, window_size=10):
    """Analyze how feature importance changes over time"""
    # Ensure X and y are pandas objects with indices
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X, columns=feature_names)
    if not isinstance(y, pd.Series):
        y = pd.Series(y)
    
    # Create windows
    n_windows = max(1, len(X) // window_size)
    temporal_importance = []
    
    for i in range(n_windows):
        start_idx = i * window_size
        end_idx = min(len(X), (i + 1) * window_size)
        
        # Skip windows that are too small
        if end_idx - start_idx < 5:
            continue
        
        # Get window data
        X_window = X.iloc[start_idx:end_idx]
        y_window = y.iloc[start_idx:end_idx]
        
        # Skip if window has no variation in y
        if len(y_window.unique()) < 2:
            continue
        
        # Fit model on window
        try:
            window_model = GradientBoostingRegressor(
                n_estimators=100, max_depth=3, learning_rate=0.1, 
                random_state=42
            ).fit(X_window, y_window)
            
            # Calculate feature importance
            window_importance = pd.Series(
                window_model.feature_importances_, 
                index=X_window.columns
            )
            
            temporal_importance.append({
                'window': i,
                'start_idx': start_idx,
                'end_idx': end_idx,
                'importance': window_importance
            })
        except Exception as e:
            print(f"Error in window {i}: {e}")
            continue
    
    return temporal_importance

# We'll use the original X data for this analysis
# First, combine train and test back together in chronological order
X_combined = pd.concat([X_train, X_test])
y_combined = pd.concat([y_train, y_test])

# Analyze temporal importance
temporal_importance = analyze_temporal_importance(
    X_combined, y_combined, X_combined.columns, window_size=10
)

# Create a plot of how feature importance changes over time
plt.figure(figsize=(15, 10))

time_periods = [item['window'] for item in temporal_importance]
top_features = importance_df.head(5).index.tolist()

if 'totaltar' not in top_features and 'totaltar' in X_combined.columns:
    top_features.append('totaltar')

for feature in top_features:
    importance_values = [item['importance'].get(feature, 0) for item in temporal_importance]
    plt.plot(time_periods, importance_values, marker='o', label=feature)

plt.title('Feature Importance Over Time')
plt.xlabel('Time Window')
plt.ylabel('Importance Score')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('temporal_importance.png')
print("Temporal importance plot saved as 'temporal_importance.png'")

# %%
# Evaluate the impact of totaltar specifically
print("\nEvaluating the specific impact of 'totaltar'...")

# Check if totaltar is in the dataset
if 'totaltar' not in X_combined.columns:
    print("Warning: 'totaltar' not found in the dataset.")
else:
    # Output totaltar importance metrics without visualization
    print("\nTotaltar importance across different analyses:")
    if 'totaltar' in importance_df.index:
        print(f"Gradient Boosting MDI: {importance_df.loc['totaltar', 'MDI']:.6f}")
        print(f"Permutation Importance: {importance_df.loc['totaltar', 'Permutation']:.6f}")
    
    if 'totaltar' in shap_importance.index:
        print(f"SHAP Importance: {shap_importance.loc['totaltar', 'mean_abs_shap']:.6f}")
    
    # Check totaltar's interactions
    if 'totaltar' in interaction_df.index:
        totaltar_interactions = interaction_df.loc['totaltar'].sort_values(ascending=False)
        print("\nTop 5 features interacting with totaltar:")
        print(totaltar_interactions.head(5))

# %%
# Create scatterplots to visualize feature interactions and non-linear relationships
print("\nCreating feature interaction scatterplots...")

# Get list of numerical features (excluding date and target)
price_features = [col for col in data_processed.columns if col not in ['date', 'target']]

# Create a correlation matrix for ordering features
corr_matrix = data_processed[price_features].corr()

# Create pairplots with annotations
plt.figure(figsize=(20, 15))
g = sns.pairplot(data_processed, vars=price_features, 
               diag_kind="kde", markers=".", plot_kws={"alpha": 0.6})

# Add title to each subplot
for i, var_i in enumerate(price_features):
    for j, var_j in enumerate(price_features):
        if i != j:  # Skip diagonal
            # Calculate correlation
            corr = data_processed[var_i].corr(data_processed[var_j])
            # Add correlation text to the plot
            ax = g.axes[i, j]
            ax.text(0.05, 0.95, f"r = {corr:.2f}", transform=ax.transAxes, 
                    fontsize=10, va='top', ha='left',
                    bbox=dict(facecolor='white', alpha=0.5))
            
            # Add LOESS smoothing to visualize non-linear relationships
            try:
                x = data_processed[var_j]
                y = data_processed[var_i]
                sns.regplot(x=x, y=y, scatter=False, lowess=True, 
                          line_kws={'color': 'red', 'lw': 1}, ax=ax)
            except:
                pass  # Skip if LOESS fails

plt.suptitle('Pairwise Relationships Between Price Variables', fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig('feature_relationships.png', bbox_inches='tight')
print("Feature relationship plots saved as 'feature_relationships.png'")

# Create heatmap of feature correlations
plt.figure(figsize=(12, 10))
mask = np.zeros_like(corr_matrix)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(corr_matrix, mask=mask, cmap='coolwarm', annot=True, fmt='.2f',
            linewidths=0.5, vmin=-1, vmax=1, center=0, square=True)
plt.title('Feature Correlation Heatmap', fontsize=15)
plt.tight_layout()
plt.savefig('feature_correlation_heatmap.png')
print("Feature correlation heatmap saved as 'feature_correlation_heatmap.png'")

# Create a scatter plot with totaltar coloring
if 'totaltar' in data_processed.columns:
    # Identify top correlated features with target
    target_corr = data_processed.corr()['target'].abs().sort_values(ascending=False)
    top_features = target_corr.index[1:3]  # Top 2 features (excluding target itself)
    
    # Create scatter plot with totaltar as color
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        data_processed[top_features[0]], 
        data_processed[top_features[1]],
        c=data_processed['totaltar'], 
        cmap='viridis', 
        alpha=0.7,
        s=80,
        edgecolors='w',
        linewidths=0.5
    )
    
    # Add colorbar and labels
    cbar = plt.colorbar(scatter)
    cbar.set_label('totaltar value', rotation=270, labelpad=20)
    plt.xlabel(top_features[0])
    plt.ylabel(top_features[1])
    plt.title(f'Relationship between {top_features[0]} and {top_features[1]} colored by totaltar')
    
    # Add annotations for high totaltar values
    high_totaltar = data_processed[data_processed['totaltar'] > data_processed['totaltar'].quantile(0.75)]
    for idx, row in high_totaltar.iterrows():
        plt.annotate(
            f"totaltar={row['totaltar']:.1f}", 
            xy=(row[top_features[0]], row[top_features[1]]),
            xytext=(10, 5),
            textcoords='offset points',
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7)
        )
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('totaltar_relationship_scatter.png')
    print("Totaltar relationship scatter plot saved as 'totaltar_relationship_scatter.png'")

# Add additional visualizations for feature interactions excluding totaltar
# Add at the end of the file

# %%
# Create additional visualizations for feature interactions and non-linear relationships
print("\nCreating additional visualizations for feature interactions and non-linearities...")

# Get variables excluding date, target, and totaltar
price_features_no_totaltar = [col for col in data_processed.columns 
                             if col not in ['date', 'target', 'totaltar']]

# Pairplots excluding totaltar
plt.figure(figsize=(20, 15))
g = sns.pairplot(data_processed, vars=price_features_no_totaltar, 
                diag_kind="kde", markers=".", plot_kws={"alpha": 0.6})

# Add title to each subplot
for i, var_i in enumerate(price_features_no_totaltar):
    for j, var_j in enumerate(price_features_no_totaltar):
        if i != j:  # Skip diagonal
            # Calculate correlation
            corr = data_processed[var_i].corr(data_processed[var_j])
            # Add correlation text to the plot
            ax = g.axes[i, j]
            ax.text(0.05, 0.95, f"r = {corr:.2f}", transform=ax.transAxes, 
                    fontsize=10, va='top', ha='left',
                    bbox=dict(facecolor='white', alpha=0.5))
            
            # Add LOESS smoothing to visualize non-linear relationships
            try:
                x = data_processed[var_j]
                y = data_processed[var_i]
                sns.regplot(x=x, y=y, scatter=False, lowess=True, 
                          line_kws={'color': 'red', 'lw': 1}, ax=ax)
            except:
                pass  # Skip if LOESS fails

plt.suptitle('Pairwise Relationships Between Price Variables (Excluding totaltar)', fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig('feature_relationships_no_totaltar.png', bbox_inches='tight')
print("Feature relationship plots without totaltar saved as 'feature_relationships_no_totaltar.png'")

# Create correlation heatmap excluding totaltar
corr_matrix_no_totaltar = data_processed[price_features_no_totaltar].corr()
plt.figure(figsize=(12, 10))
mask = np.zeros_like(corr_matrix_no_totaltar)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(corr_matrix_no_totaltar, mask=mask, cmap='coolwarm', annot=True, fmt='.2f',
            linewidths=0.5, vmin=-1, vmax=1, center=0, square=True)
plt.title('Correlation Between Price Variables (Excluding totaltar)')
plt.tight_layout()
plt.savefig('feature_correlation_heatmap_no_totaltar.png')
print("Feature correlation heatmap without totaltar saved as 'feature_correlation_heatmap_no_totaltar.png'")

# Visualize top feature interactions based on SHAP analysis
# Get top interaction pairs
non_self_interactions = []
for i in range(len(interaction_df)):
    for j in range(i+1, len(interaction_df)):
        if interaction_df.columns[i] != 'totaltar' and interaction_df.columns[j] != 'totaltar':
            strength = interaction_df.iloc[i, j]
            if strength > 0:
                non_self_interactions.append((
                    interaction_df.columns[i], 
                    interaction_df.columns[j], 
                    strength
                ))

# Sort by interaction strength
non_self_interactions.sort(key=lambda x: x[2], reverse=True)

# Function to create enhanced interaction PDP plots
def create_interaction_pdp(surrogate_model, X_train, features, feature_names=None, grid_resolution=20):
    """Create detailed partial dependence interaction plots"""
    if feature_names is None:
        feature_names = features
        
    # Calculate partial dependence
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
        for i, point in enumerate(grid_points):
            X_temp = X_train.copy()
            X_temp[features[0]] = point[0]
            X_temp[features[1]] = point[1]
            z_values[i] = np.mean(surrogate_model.predict(X_temp))
        
        z_values = z_values.reshape(x_values.shape)
        
    except Exception as e:
        print(f"Error in PDP calculation: {e}")
        return None, None
        
    # Create 2x2 grid of visualizations
    fig, axs = plt.subplots(2, 2, figsize=(18, 16))
    
    # 1. Contour plot
    contour = axs[0, 0].contourf(
        x_values, y_values, z_values, 
        cmap='viridis', levels=15
    )
    axs[0, 0].set_xlabel(feature_names[0])
    axs[0, 0].set_ylabel(feature_names[1])
    axs[0, 0].set_title('PDP Interaction Contour')
    plt.colorbar(contour, ax=axs[0, 0], label='Predicted Value')
    
    # 2. 3D Surface plot
    axs[0, 1] = fig.add_subplot(2, 2, 2, projection='3d')
    surface = axs[0, 1].plot_surface(
        x_values, y_values, z_values, 
        cmap='viridis', 
        edgecolor='none', 
        alpha=0.8
    )
    axs[0, 1].set_xlabel(feature_names[0])
    axs[0, 1].set_ylabel(feature_names[1])
    axs[0, 1].set_zlabel('Predicted Value')
    axs[0, 1].set_title('PDP Interaction 3D Surface')
    plt.colorbar(surface, ax=axs[0, 1], shrink=0.5, label='Predicted Value')
    
    # 3. Heatmap with annotations
    sns.heatmap(
        z_values, 
        ax=axs[1, 0],
        cmap='viridis', 
        annot=True, 
        fmt='.1f', 
        xticklabels=np.round(feature_values[0], 2),
        yticklabels=np.round(feature_values[1], 2)
    )
    axs[1, 0].set_xlabel(feature_names[0])
    axs[1, 0].set_ylabel(feature_names[1])
    axs[1, 0].set_title('PDP Interaction Heatmap')
    
    # 4. Overlay with actual data points
    contour = axs[1, 1].contourf(
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
    
    # Calculate and display interaction strength
    try:
        # Calculate individual PDPs
        pdp1_values = np.zeros(len(feature_values[0]))
        pdp2_values = np.zeros(len(feature_values[1]))
        
        # Calculate PDP for first feature
        for i, val in enumerate(feature_values[0]):
            X_temp = X_train.copy()
            X_temp[features[0]] = val
            pdp1_values[i] = np.mean(surrogate_model.predict(X_temp))
            
        # Calculate PDP for second feature
        for i, val in enumerate(feature_values[1]):
            X_temp = X_train.copy()
            X_temp[features[1]] = val
            pdp2_values[i] = np.mean(surrogate_model.predict(X_temp))
        
        # Compute interaction strength (H-statistic)
        pdp1_mesh, pdp2_mesh = np.meshgrid(pdp1_values, pdp2_values)
        additive_effect = pdp1_mesh + pdp2_mesh - np.mean(pdp1_values) - np.mean(pdp2_values) + np.mean(z_values)
        interaction_strength = np.mean(np.abs(z_values - additive_effect)) / np.std(z_values)
        
        plt.figtext(
            0.5, 0.01, 
            f"Interaction Strength (H-statistic): {interaction_strength:.4f}\n" +
            f"Higher values indicate stronger interaction effects between {feature_names[0]} and {feature_names[1]}.",
            ha='center', 
            fontsize=12, 
            bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5')
        )
    except Exception as e:
        print(f"Error calculating interaction strength: {e}")
        interaction_strength = None
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    
    return fig, interaction_strength

# Create interaction PDP plots for top 3 interactions
interaction_strengths = {}
for i, (feat1, feat2, _) in enumerate(non_self_interactions[:3]):
    print(f"Creating PDP interaction plot for {feat1} and {feat2}")
    plt.figure(figsize=(18, 16))
    fig, h_stat = create_interaction_pdp(
        surrogate_model, X_train, 
        [feat1, feat2], 
        [feat1, feat2]
    )
    interaction_strengths[(feat1, feat2)] = h_stat
    plt.savefig(f'pdp_interaction_{feat1}_vs_{feat2}.png')
    print(f"PDP interaction plot saved as 'pdp_interaction_{feat1}_vs_{feat2}.png'")

# Add enhanced threshold analysis from SAFE workflow
print("\nPerforming enhanced threshold analysis from SAFE transformation...")

def analyze_thresholds(X_train_transformed, surrogate_model, original_columns):
    """Analyze thresholds identified in the SAFE transformation"""
    thresholds = {}
    impact_scores = {}
    
    # Extract thresholds from column names
    for col in X_train_transformed.columns:
        if '>=' in col and 'AND' not in col and '_x_' not in col:
            feature, threshold_str = col.split('>=')
            feature = feature.rstrip('_')
            threshold = float(threshold_str)
            
            if feature not in thresholds:
                thresholds[feature] = []
            thresholds[feature].append(threshold)
            
            # Measure threshold impact using surrogate model
            temp_data = X_train.copy()
            temp_data[f"{feature}_threshold"] = (temp_data[feature] >= threshold).astype(int)
            
            # Calculate feature importance of this threshold in surrogate model context
            base_pred = surrogate_model.predict(X_train)
            # Compare predictions when forcing points below threshold
            temp_data_below = X_train.copy()
            mask = temp_data_below[feature] >= threshold
            if mask.sum() > 0:  # Only proceed if there are points above threshold
                orig_values = temp_data_below.loc[mask, feature].copy()
                # Set to just below threshold
                temp_data_below.loc[mask, feature] = threshold - 0.0001
                pred_below = surrogate_model.predict(temp_data_below)
                # Calculate average impact of crossing this threshold
                impact_above = np.abs(base_pred[mask] - pred_below[mask]).mean()
                # Store impact score
                impact_scores[(feature, threshold)] = impact_above
                # Reset the original values
                temp_data_below.loc[mask, feature] = orig_values
    
    # Sort thresholds for each feature
    for feature in thresholds:
        thresholds[feature].sort()
    
    # Create summary dataframe of thresholds and impacts
    impact_df = pd.DataFrame([
        {
            'Feature': feature,
            'Threshold': threshold,
            'Impact': impact_scores.get((feature, threshold), 0),
            'Data_Percentage_Above': (X_train[feature] >= threshold).mean() * 100
        }
        for (feature, threshold_list) in thresholds.items()
        for threshold in threshold_list
    ])
    
    impact_df = impact_df.sort_values('Impact', ascending=False)
    
    return thresholds, impact_df

# Run the threshold analysis
thresholds, impact_df = analyze_thresholds(X_train_transformed, surrogate_model, X_train.columns)

# Print top impactful thresholds
print("\nTop 10 most impactful thresholds:")
print(impact_df.head(10))

# Visualize threshold impacts
plt.figure(figsize=(14, 8))
top_impacts = impact_df.head(15)  # Top 15 thresholds

# Create labels that include both feature name and threshold value
labels = [f"{row['Feature']}\n>= {row['Threshold']:.2f}" for _, row in top_impacts.iterrows()]

# Create bar chart
bar = sns.barplot(x=top_impacts['Impact'], y=labels)
plt.title('Impact of Top Thresholds Identified by SAFE Transformation')
plt.xlabel('Average Impact on Prediction')
plt.ylabel('Feature Threshold')

# Add percentage annotations
for i, (_, row) in enumerate(top_impacts.iterrows()):
    bar.text(
        row['Impact'] + 0.1, 
        i, 
        f"{row['Data_Percentage_Above']:.1f}%", 
        va='center'
    )

plt.tight_layout()
plt.savefig('threshold_impact_analysis.png')
print("Threshold impact analysis saved as 'threshold_impact_analysis.png'")

# Visualize thresholds for top features
plt.figure(figsize=(16, 12))
top_features = impact_df['Feature'].value_counts().head(5).index.tolist()

# Create a multi-panel plot (1 row, len(top_features) columns)
fig, axes = plt.subplots(len(top_features), 1, figsize=(14, 4*len(top_features)))

for i, feature in enumerate(top_features):
    feature_thresholds = impact_df[impact_df['Feature'] == feature].sort_values('Threshold')
    
    # Get feature data
    data = X_train[feature]
    
    # Create histogram with thresholds as vertical lines
    sns.histplot(data, ax=axes[i], kde=True)
    
    # Add vertical lines for thresholds
    max_height = axes[i].get_ylim()[1]
    
    for _, row in feature_thresholds.iterrows():
        threshold = row['Threshold']
        impact = row['Impact']
        # Scale line height with impact
        height = max_height * (0.3 + 0.7 * impact / impact_df['Impact'].max())
        axes[i].plot(
            [threshold, threshold], 
            [0, height], 
            color='red', 
            linestyle='--', 
            linewidth=2, 
            alpha=0.7
        )
        # Add annotation
        axes[i].text(
            threshold, 
            height, 
            f" Impact: {impact:.2f}", 
            ha='left', 
            va='top', 
            color='red',
            fontweight='bold',
            bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.1')
        )
    
    axes[i].set_title(f'Distribution of {feature} with Identified Thresholds')
    axes[i].set_xlabel(feature)
    axes[i].set_ylabel('Frequency')

plt.tight_layout()
plt.savefig('threshold_distribution_analysis.png')
print("Threshold distribution analysis saved as 'threshold_distribution_analysis.png'")

# Add expanded analysis on thresholds and rules from the surrogate decision tree model
print("\nExtracting and analyzing rules from the surrogate decision tree...")

# Function to extract rules from a decision tree
def extract_rules_from_tree(tree, feature_names):
    """Extract decision rules from a fitted tree"""
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != -2 else "undefined!"  # -2 is the new undefined value
        for i in tree_.feature
    ]
    
    paths = []
    path = []
    
    def dfs(node, path, paths):
        if tree_.feature[node] != -2:  # -2 is the new undefined value
            name = feature_name[node]
            threshold = tree_.threshold[node]
            
            # Left path (<=)
            path.append((name, "<=", threshold))
            dfs(tree_.children_left[node], path, paths)
            path.pop()
            
            # Right path (>)
            path.append((name, ">", threshold))
            dfs(tree_.children_right[node], path, paths)
            path.pop()
        else:
            # Leaf node
            path_copy = list(path)
            paths.append(path_copy)
    
    dfs(0, path, paths)
    
    # Create rule strings from paths
    rules = []
    for path in paths:
        rule = "IF "
        for p in path:
            if rule != "IF ":
                rule += " AND "
            rule += f"{p[0]} {p[1]} {p[2]:.2f}"
        rule += f" THEN prediction = {tree_.value[0].flatten()[0]:.2f}"
        rules.append(rule)
    
    return rules, paths

# Create a surrogate decision tree model with limited depth for interpretability
from sklearn.tree import DecisionTreeRegressor, export_graphviz
import graphviz
from sklearn import tree as _tree

surrogate_dt = DecisionTreeRegressor(max_depth=3, random_state=42)
surrogate_dt.fit(X_train, surrogate_model.predict(X_train))

# Extract rules
rules, paths = extract_rules_from_tree(surrogate_dt, X_train.columns)

# Create rule impact analysis
rule_impacts = []
for i, (rule, path) in enumerate(zip(rules, paths)):
    # Create boolean mask for all samples matching this rule
    mask = np.ones(len(X_train), dtype=bool)
    for condition in path:
        feature, operator, threshold = condition
        if operator == "<=":
            mask = mask & (X_train[feature] <= threshold)
        else:  # ">"
            mask = mask & (X_train[feature] > threshold)
    
    # Calculate metrics for this rule
    samples_count = mask.sum()
    samples_percentage = samples_count / len(X_train) * 100
    
    # Get predictions
    leaf_prediction = surrogate_dt.predict(X_train[mask].iloc[0:1])[0]
    
    # Calculate average target value for samples matching this rule
    if samples_count > 0:
        avg_target = y_train[mask].mean()
        target_std = y_train[mask].std() if samples_count > 1 else 0
    else:
        avg_target = None
        target_std = None
    
    rule_impacts.append({
        'Rule': rule,
        'Samples': samples_count,
        'Percentage': samples_percentage,
        'Prediction': leaf_prediction,
        'Avg_Target': avg_target,
        'Target_Std': target_std,
        'Rule_Complexity': len(path)
    })

# Convert to DataFrame
rule_df = pd.DataFrame(rule_impacts)
rule_df['Error'] = rule_df['Prediction'] - rule_df['Avg_Target']
rule_df['Abs_Error'] = np.abs(rule_df['Error'])
rule_df = rule_df.sort_values('Samples', ascending=False)

print("\nTop surrogate decision tree rules:")
for i, (_, row) in enumerate(rule_df.head(5).iterrows()):
    print(f"{i+1}. {row['Rule']}")
    print(f"   Covers {row['Samples']} samples ({row['Percentage']:.1f}% of training data)")
    print(f"   Prediction: {row['Prediction']:.2f}, Actual Average: {row['Avg_Target']:.2f}")
    print(f"   Absolute Error: {row['Abs_Error']:.2f}")
    print()

# Visualize rule impact with enhanced explanation
plt.figure(figsize=(14, 10))  # Increased height to accommodate explanation
rule_labels = [f"Rule {i+1}\n({row['Samples']} samples)" for i, (_, row) in enumerate(rule_df.head(8).iterrows())]
rule_errors = rule_df.head(8)['Abs_Error'].values
rule_coverage = rule_df.head(8)['Percentage'].values

# Create bar chart
bars = plt.bar(rule_labels, rule_errors, color='skyblue')

# Add coverage as text
for i, (bar, coverage) in enumerate(zip(bars, rule_coverage)):
    plt.text(
        bar.get_x() + bar.get_width()/2, 
        bar.get_height() + 0.1,
        f"{coverage:.1f}%",
        ha='center', 
        va='bottom'
    )

# Visualize rule impact with enhanced explanation and actual rule conditions
plt.figure(figsize=(18, 12))  # Larger figure to accommodate rule text
top_rules = rule_df.head(5)  # Limit to top 5 rules for readability

# Create shortened rule descriptions by extracting key components of each rule
rule_texts = []
for i, (_, row) in enumerate(top_rules.iterrows()):
    # Simplify the rule text to fit in chart
    rule_text = row['Rule']
    # Extract the main conditions (the part between IF and THEN)
    if 'IF ' in rule_text and ' THEN ' in rule_text:
        conditions = rule_text.split('IF ')[1].split(' THEN ')[0]
        # Shorten further if needed
        if len(conditions) > 70:
            conditions = conditions[:67] + "..."
    else:
        conditions = rule_text[:70] + "..." if len(rule_text) > 70 else rule_text
    rule_texts.append(conditions)

# Calculate x positions for bars (with extra space)
x_pos = np.arange(len(top_rules)) * 1.5

# Create bar chart with actual rule conditions and gradient colors
colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(top_rules)))
bars = plt.bar(
    x_pos, 
    top_rules['Abs_Error'].values, 
    color=colors,
    width=0.8,
    edgecolor='black',
    linewidth=0.5
)

# Add coverage as text on top of bars
for i, (bar, coverage) in enumerate(zip(bars, top_rules['Percentage'].values)):
    plt.text(
        bar.get_x() + bar.get_width()/2, 
        bar.get_height() + 0.05,
        f"Coverage: {coverage:.1f}%",
        ha='center', 
        va='bottom',
        fontweight='bold',
        color='black',
        bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2')
    )

# Add annotations below bars with actual rule text
for i, (rule_text, x) in enumerate(zip(rule_texts, x_pos)):
    plt.text(
        x,
        -0.15,  # Position below x-axis
        rule_text,
        ha='center',
        va='top',
        rotation=0,  # Keep text horizontal
        fontsize=9,
        wrap=True,
        bbox=dict(facecolor='lightyellow', alpha=0.9, boxstyle='round,pad=0.3')
    )

# Add explanation text
explanation_text = """
HOW TO INTERPRET THIS PLOT:

1. Each bar shows a decision rule from the surrogate tree with its actual conditions below
2. Bar height represents the absolute error between predicted and actual values
3. Lower bars indicate more accurate rules (better predictions)
4. Coverage % shows how much training data is covered by each rule
5. The most common rules (higher coverage) tend to be more reliable
"""

plt.figtext(0.5, 0.02, explanation_text, ha='center', fontsize=10,
            bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))

plt.title('Error Analysis of Top Surrogate Decision Tree Rules with Conditions', fontsize=14)
plt.ylabel('Absolute Error', fontsize=12)
plt.ylim(-0.2, max(top_rules['Abs_Error'].values) * 1.2)  # Make room for text above and below
plt.xticks(x_pos, [f"Rule {i+1}" for i in range(len(top_rules))], fontsize=10)
plt.axhline(y=0, color='black', linestyle='-', linewidth=0.8)  # Add x-axis line
plt.grid(axis='y', linestyle='--', alpha=0.3)

plt.tight_layout()
plt.subplots_adjust(bottom=0.22)  # Make room for the explanation
plt.savefig('surrogate_tree_rule_analysis.png')
print("Surrogate tree rule analysis saved as 'surrogate_tree_rule_analysis.png'")

# Visualize surrogate tree
plt.figure(figsize=(20, 12))
_tree.plot_tree(
    surrogate_dt, 
    feature_names=X_train.columns, 
    filled=True, 
    rounded=True,
    fontsize=10, 
    proportion=True, 
    precision=2
)
plt.title('Surrogate Decision Tree (Max Depth 3)')
plt.tight_layout()
plt.savefig('surrogate_decision_tree.png')
print("Surrogate decision tree visualization saved as 'surrogate_decision_tree.png'")

# Create a more detailed visualization using graphviz
dot_data = export_graphviz(
    surrogate_dt,
    out_file=None,
    feature_names=X_train.columns,
    filled=True,
    rounded=True,
    special_characters=True,
    proportion=True
)
graph = graphviz.Source(dot_data)
graph.render("surrogate_tree_detailed")
print("Detailed surrogate tree visualization saved as 'surrogate_tree_detailed.pdf'")

# Create a feature interaction heatmap from surrogate tree
def extract_feature_interactions_from_tree(tree, feature_names):
    """Extract feature interactions from decision paths in a tree"""
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != -2 else "undefined!"  # -2 is the new undefined value
        for i in tree_.feature
    ]
    
    interactions = np.zeros((len(feature_names), len(feature_names)))
    
    def count_interactions(node, path, interactions):
        if tree_.feature[node] != -2:  # -2 is the new undefined value
            current_feature = feature_names.index(feature_name[node])
            
            # Count interactions with features in the path
            for prev_feature in path:
                interactions[prev_feature, current_feature] += 1
                interactions[current_feature, prev_feature] += 1
            
            # Continue traversal
            path.append(current_feature)
            count_interactions(tree_.children_left[node], path, interactions)
            count_interactions(tree_.children_right[node], path, interactions)
            path.pop()
    
    count_interactions(0, [], interactions)
    return interactions

interaction_matrix = extract_feature_interactions_from_tree(surrogate_dt, list(X_train.columns))
interaction_df_tree = pd.DataFrame(
    interaction_matrix, 
    index=X_train.columns, 
    columns=X_train.columns
)

plt.figure(figsize=(14, 14))  # Increased height to accommodate explanation
mask = np.zeros_like(interaction_df_tree)
mask[np.triu_indices_from(mask, k=1)] = True
sns.heatmap(
    interaction_df_tree, 
    mask=mask, 
    cmap='YlOrRd', 
    annot=True, 
    fmt='g',
    linewidths=0.5, 
    square=True
)
plt.title('Feature Interactions from Surrogate Decision Tree')

# Add explanation text
explanation_text = """
HOW TO INTERPRET THIS HEATMAP:

1. Each cell shows the number of times two features appear together in decision paths
2. Darker colors and higher numbers indicate stronger interactions
3. The diagonal is masked as self-interactions are not meaningful
4. Key insights:
   • Higher values suggest features that frequently work together to make predictions
   • Zero values indicate features that never appear together in decision paths
   • This reveals the hierarchical structure of the decision-making process
"""

plt.figtext(0.5, 0.02, explanation_text, ha='center', fontsize=10, 
            bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))

plt.tight_layout()
plt.subplots_adjust(bottom=0.3)  # Make room for the explanation
plt.savefig('surrogate_tree_interactions.png')
print("Surrogate tree interactions heatmap saved as 'surrogate_tree_interactions.png'")

# Add PDP interaction effect plots
print("\nGenerating PDP interaction effect plots...")

# Create interaction PDP plots for top 3 interactions
interaction_strengths = {}
for i, (feat1, feat2, _) in enumerate(non_self_interactions[:3]):
    print(f"Creating PDP interaction plot for {feat1} and {feat2}")
    plt.figure(figsize=(18, 16))
    fig, h_stat = create_interaction_pdp(
        surrogate_model, X_train, 
        [feat1, feat2], 
        [feat1, feat2]
    )
    interaction_strengths[(feat1, feat2)] = h_stat
    plt.savefig(f'pdp_interaction_{feat1}_vs_{feat2}.png')
    print(f"PDP interaction plot saved as 'pdp_interaction_{feat1}_vs_{feat2}.png'")

# Add a comprehensive function to calculate Friedman's H-statistic for multi-way interactions
def calculate_friedman_h_statistic(model, X, features, grid_resolution=20):
    """
    Calculate Friedman's H-statistic for feature interactions.
    
    Parameters:
    -----------
    model : fitted model with predict method
        The model for which to calculate interaction strength
    X : pandas DataFrame
        The data used for calculating partial dependence
    features : list
        List of feature names to evaluate interactions for
    grid_resolution : int
        Number of grid points for each feature dimension
        
    Returns:
    --------
    h_stat : float
        The H-statistic value (between 0 and 1), with higher values indicating stronger interactions
    partial_dependence_dict : dict
        Dictionary containing the calculated partial dependence values
    """
    import itertools
    
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
    sum_individual_effects = np.zeros(tuple(grid_resolution for _ in features))
    
    for i, f in enumerate(features):
        # Create broadcasting shape
        broadcast_shape = [1] * len(features)
        broadcast_shape[i] = grid_resolution
        
        # Reshape individual effect for proper broadcasting
        reshaped_pd = individual_pd[f].reshape(broadcast_shape)
        
        # Add to sum of effects
        sum_individual_effects += reshaped_pd
    
    # Calculate mean prediction (constant effect)
    mean_prediction = model.predict(X).mean()
    
    # Calculate joint partial dependence
    grid_combinations = list(itertools.product(*[grid_points[f] for f in features]))
    joint_pd = np.zeros(tuple(grid_resolution for _ in features))
    
    # Fill multi-dimensional grid
    grid_indices = list(itertools.product(*[range(grid_resolution) for _ in features]))
    
    for idx, grid_combo in zip(grid_indices, grid_combinations):
        X_temp = X.copy()
        
        for j, feature in enumerate(features):
            X_temp[feature] = grid_combo[j]
            
        joint_pd[idx] = model.predict(X_temp).mean()
    
    # Calculate H-statistic
    interaction_effect = joint_pd - (sum_individual_effects - mean_prediction * (len(features) - 1))
    
    # Variance of the interaction effect
    var_interaction = np.var(interaction_effect)
    
    # Variance of the joint partial dependence
    var_joint = np.var(joint_pd)
    
    # Calculate H-statistic (normalized measure of interaction strength)
    h_stat = var_interaction / (var_joint + 1e-8)  # Add small constant to avoid division by zero
    
    return h_stat, {
        'individual_pd': individual_pd,
        'joint_pd': joint_pd,
        'interaction_effect': interaction_effect,
        'grid_points': grid_points
    }

def visualize_three_way_interaction(model, X, features, grid_resolution=10):
    """
    Create visualizations for three-way feature interactions.
    
    Parameters:
    -----------
    model : fitted model with predict method
        The model to analyze
    X : pandas DataFrame
        The data used for calculating interactions
    features : list of str
        Three feature names to visualize interaction for
    grid_resolution : int
        Resolution of grid for each feature dimension
    
    Returns:
    --------
    fig : matplotlib figure
        Figure containing the visualizations
    h_stat : float
        Friedman's H-statistic for the three-way interaction
    """
    if len(features) != 3:
        raise ValueError("This function requires exactly 3 features")
    
    h_stat, pd_dict = calculate_friedman_h_statistic(model, X, features, grid_resolution)
    
    # Create a 3x3 grid of figures showing different slices and projections
    fig = plt.figure(figsize=(20, 16))
    fig.suptitle(f"Three-way Interaction: {', '.join(features)}\nFriedman H-statistic: {h_stat:.4f}", fontsize=16)
    
    grid_points = pd_dict['grid_points']
    interaction_effect = pd_dict['interaction_effect']
    
    # Define colormaps
    cmap = plt.cm.viridis
    norm = plt.Normalize(interaction_effect.min(), interaction_effect.max())
    
    # Feature combinations for 2D slices
    combinations = list(itertools.combinations(range(3), 2))
    fixed_feature_positions = [grid_resolution // 4, grid_resolution // 2, 3 * grid_resolution // 4]
    
    # Create subplots for each combination and fixed position
    for i, (x_idx, y_idx) in enumerate(combinations):
        z_idx = [j for j in range(3) if j not in [x_idx, y_idx]][0]
        
        for j, fixed_pos in enumerate(fixed_feature_positions):
            ax = fig.add_subplot(3, 3, i*3 + j + 1)
            
            # Create slice indices
            slice_indices = [0, 0, 0]
            slice_indices[z_idx] = fixed_pos
            slice_indices[x_idx] = slice(None)
            slice_indices[y_idx] = slice(None)
            
            # Extract the 2D slice
            slice_data = interaction_effect[tuple(slice_indices)]
            
            # Create heatmap
            im = ax.imshow(slice_data.T, origin='lower', aspect='auto', cmap=cmap, norm=norm,
                         extent=[
                             grid_points[features[x_idx]].min(), 
                             grid_points[features[x_idx]].max(),
                             grid_points[features[y_idx]].min(), 
                             grid_points[features[y_idx]].max()
                         ])
            
            # Add contour lines
            ax.contour(
                np.linspace(grid_points[features[x_idx]].min(), grid_points[features[x_idx]].max(), slice_data.shape[0]),
                np.linspace(grid_points[features[y_idx]].min(), grid_points[features[y_idx]].max(), slice_data.shape[1]),
                slice_data.T,
                colors='white', alpha=0.5, linewidths=0.5
            )
            
            # Label axes
            ax.set_xlabel(features[x_idx])
            ax.set_ylabel(features[y_idx])
            
            fixed_value = grid_points[features[z_idx]][fixed_pos]
            ax.set_title(f"{features[z_idx]} = {fixed_value:.2f}")
    
    # Add a colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax)
    cbar.set_label('Interaction Effect Strength')
    
    # Add explanation
    explanation_text = """
    HOW TO INTERPRET THIS PLOT:
    
    1. Each heatmap shows the interaction effect strength at different feature values
    2. Brighter colors indicate stronger interactions (red/yellow = strong, blue = weak)
    3. Each row shows different 2D slices of the 3D interaction space
    4. Contour lines highlight regions with similar interaction strength
    5. The Friedman H-statistic (0-1) quantifies overall interaction strength
    6. Higher H-statistic values indicate stronger interactions beyond individual effects
    """
    
    fig.text(0.5, 0.02, explanation_text, ha='center', fontsize=12, 
             bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9, right=0.9, bottom=0.2)
    
    return fig, h_stat

# Modify the existing code to include enhanced Friedman H-statistic analysis
# ... existing code ...

# Add enhanced Friedman H-statistic analysis after the interaction network graph
print("\nPerforming enhanced Friedman H-statistic interaction analysis...")

# Calculate pairwise Friedman H-statistics for all feature pairs
feature_pairs = list(itertools.combinations(X_train.columns, 2))
pairwise_h_stats = {}

print("Calculating pairwise Friedman H-statistics...")
# Take a reasonable number of top pairs to analyze
top_n_pairs = min(15, len(feature_pairs))
selected_pairs = feature_pairs[:top_n_pairs]

for pair in tqdm(selected_pairs):
    h_stat, _ = calculate_friedman_h_statistic(surrogate_model, X_train, list(pair))
    pairwise_h_stats[pair] = h_stat

# Sort and display top interaction pairs
pairwise_h_df = pd.DataFrame(
    [(f1, f2, h_stat) for (f1, f2), h_stat in pairwise_h_stats.items()],
    columns=['Feature 1', 'Feature 2', 'H-statistic']
).sort_values('H-statistic', ascending=False)

print("\nTop 10 feature pairs by Friedman H-statistic:")
print(pairwise_h_df.head(10))

# Visualize pairwise H-statistics
plt.figure(figsize=(14, 8))
plt.bar(
    [f"{row['Feature 1']} vs {row['Feature 2']}" for _, row in pairwise_h_df.head(10).iterrows()],
    pairwise_h_df.head(10)['H-statistic'].values
)
plt.xticks(rotation=90)
plt.ylabel('Friedman H-statistic')
plt.title('Top 10 Feature Pairs by Interaction Strength (Friedman H-statistic)')

# Add explanation text
explanation_text = """
HOW TO INTERPRET FRIEDMAN H-STATISTIC:

1. H-statistic measures interaction strength beyond individual feature effects
2. Values range from 0 (no interaction) to 1 (strong interaction)
3. Higher values indicate features that work together synergistically
4. Values above 0.2 typically indicate significant interactions
5. This helps identify which feature combinations to further investigate
"""

plt.figtext(0.5, 0.01, explanation_text, ha='center', fontsize=10,
            bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))

plt.tight_layout()
plt.subplots_adjust(bottom=0.3)
plt.savefig('pairwise_h_statistics.png')
print("Pairwise H-statistics plot saved as 'pairwise_h_statistics.png'")

# Create heatmap of pairwise H-statistics
h_matrix = np.zeros((len(X_train.columns), len(X_train.columns)))
for (f1, f2), h_stat in pairwise_h_stats.items():
    i = list(X_train.columns).index(f1)
    j = list(X_train.columns).index(f2)
    h_matrix[i, j] = h_stat
    h_matrix[j, i] = h_stat  # symmetric

plt.figure(figsize=(12, 10))
sns.heatmap(
    pd.DataFrame(h_matrix, index=X_train.columns, columns=X_train.columns),
    cmap='YlOrRd',
    annot=True,
    fmt='.3f'
)
plt.title('Pairwise Friedman H-statistics Heatmap')
plt.tight_layout()
plt.savefig('h_statistic_heatmap.png')
print("H-statistic heatmap saved as 'h_statistic_heatmap.png'")

# Calculate and visualize three-way interactions
print("\nCalculating three-way Friedman H-statistics...")

# Take top features from earlier analyses
top_features = importance_df.head(5).index.tolist()
three_way_combinations = list(itertools.combinations(top_features, 3))

# Limit to reasonable number of combinations
max_threeway = min(5, len(three_way_combinations))
three_way_h_stats = {}

for combo in tqdm(three_way_combinations[:max_threeway]):
    try:
        fig, h_stat = visualize_three_way_interaction(
            surrogate_model, X_train, list(combo), grid_resolution=8
        )
        three_way_h_stats[combo] = h_stat
        filename = f'three_way_interaction_{"_".join(combo)}.png'
        plt.savefig(filename)
        plt.close(fig)
        print(f"Three-way interaction plot saved as '{filename}'")
    except Exception as e:
        print(f"Error calculating three-way interaction for {combo}: {e}")
        plt.close()

# Create summary of three-way interactions
three_way_df = pd.DataFrame(
    [(f1, f2, f3, h_stat) for (f1, f2, f3), h_stat in three_way_h_stats.items()],
    columns=['Feature 1', 'Feature 2', 'Feature 3', 'H-statistic']
).sort_values('H-statistic', ascending=False)

print("\nThree-way feature interactions by Friedman H-statistic:")
print(three_way_df)

# Create a summary visualization of three-way interactions
plt.figure(figsize=(12, 6))
plt.bar(
    [f"{row['Feature 1'][:5]}_{row['Feature 2'][:5]}_{row['Feature 3'][:5]}" 
     for _, row in three_way_df.iterrows()],
    three_way_df['H-statistic'].values
)

# Create a summary visualization of three-way interactions with improved labeling
plt.figure(figsize=(14, 8))

# Create better labels that are more readable
feature_labels = []
for _, row in three_way_df.iterrows():
    # Create abbreviated but distinct feature names
    f1 = row['Feature 1'].split('_')[0][:7]  # First word, up to 7 chars
    f2 = row['Feature 2'].split('_')[0][:7]  # First word, up to 7 chars
    f3 = row['Feature 3'].split('_')[0][:7]  # First word, up to 7 chars
    feature_labels.append(f"{f1} + {f2} + {f3}")

# Plot with better formatting
bars = plt.bar(
    feature_labels,
    three_way_df['H-statistic'].values,
    color=plt.cm.viridis(np.linspace(0.2, 0.8, len(three_way_df))),  # Use colormap for better distinction
    width=0.6  # Make bars narrower for better separation
)

# Add H-statistic values on top of each bar
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{height:.3f}', ha='center', va='bottom', fontsize=10)

plt.xticks(rotation=45, ha='right')
plt.ylabel('Friedman H-statistic', fontsize=12)
plt.ylim(0, max(three_way_df['H-statistic'].values) * 1.15)  # Add some headroom for labels
plt.title('Three-way Feature Interactions by Strength', fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add explanation
explanation_text = """
HOW TO INTERPRET THREE-WAY INTERACTIONS:

1. H-statistic measures complex interaction strength among three features
2. Higher values indicate a combined effect beyond pairwise interactions
3. Three-way interactions reveal complex decision boundaries and non-linear relationships
4. These interactions may explain model behavior that simpler analyses miss
5. Focus on three-way combinations with H-statistic > 0.1 for deeper analysis
"""

plt.figtext(0.5, 0.01, explanation_text, ha='center', fontsize=10,
            bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))

plt.tight_layout()
plt.subplots_adjust(bottom=0.35)
plt.savefig('three_way_h_statistics.png')
print("Three-way H-statistics plot saved as 'three_way_h_statistics.png'")

# Create an integrated analysis comparing pairwise and three-way interactions
plt.figure(figsize=(14, 10))

# Setup for parallel coordinates plot
from pandas.plotting import parallel_coordinates

# Combine interaction data
combined_interactions = []

# Add pairwise interactions
for i, row in pairwise_h_df.head(10).iterrows():
    combined_interactions.append({
        'Feature Combination': f"{row['Feature 1'][:7]}_{row['Feature 2'][:7]}",
        'Interaction Type': 'Pairwise',
        'H-statistic': row['H-statistic'],
        'Rank': i + 1
    })

# Add three-way interactions
for i, row in three_way_df.iterrows():
    combined_interactions.append({
        'Feature Combination': f"{row['Feature 1'][:5]}_{row['Feature 2'][:5]}_{row['Feature 3'][:5]}",
        'Interaction Type': 'Three-way',
        'H-statistic': row['H-statistic'],
        'Rank': i + 1
    })

combined_df = pd.DataFrame(combined_interactions)

# Create the comparison visualization
plt.subplot(2, 1, 1)
sns.barplot(x='Feature Combination', y='H-statistic', hue='Interaction Type', data=combined_df)
plt.xticks(rotation=90)
plt.title('Comparison of Pairwise and Three-way Interaction Strengths')
plt.legend(loc='upper right')

# Add a more appropriate visualization instead of parallel coordinates
plt.subplot(2, 1, 2)
# Create a scatter plot showing H-statistic by rank colored by interaction type
sns.scatterplot(
    x='Rank', 
    y='H-statistic', 
    hue='Interaction Type',
    size='H-statistic',
    sizes=(50, 400),
    alpha=0.7,
    data=combined_df
)
plt.title('Interaction Strength by Rank')
plt.xlabel('Rank (lower = stronger)')
plt.ylabel('H-statistic (higher = stronger interaction)')

# Add explanation
explanation_text = """
HOW TO INTERPRET THIS PLOT:

1. Each bar represents a feature combination
2. Color indicates interaction type (Pairwise or Three-way)
3. Size of points represents interaction strength
4. H-statistic values are shown on the y-axis
5. This visualization helps identify which feature combinations are most influential
"""

plt.figtext(0.5, 0.01, explanation_text, ha='center', fontsize=10,
            bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))

plt.tight_layout()
plt.subplots_adjust(hspace=0.3, bottom=0.25)
plt.savefig('interaction_comparison.png')
print("Interaction comparison visualization saved as 'interaction_comparison.png'")

# ... continue with existing code ...

# Modify the PDP interaction plot creation to include the H-statistic in the title
print("\nGenerating enhanced PDP interaction effect plots with Friedman H-statistics...")

# Get all original features excluding target, date, and derived features
original_features = [col for col in data_processed.columns 
                    if col not in ['date', 'target'] 
                    and not col.endswith('_x_') 
                    and not col.endswith('_AND_')
                    and not '>=' in col]

# Create all possible pairs of features
feature_pairs = list(itertools.combinations(original_features, 2))

# First calculate H-statistics for all pairs to filter
print("\nCalculating H-statistics for all feature pairs...")
interaction_strengths = {}
for feat1, feat2 in tqdm(feature_pairs, desc="Calculating H-statistics"):
    try:
        h_stat, _ = calculate_friedman_h_statistic(surrogate_model, X_train, [feat1, feat2])
        interaction_strengths[(feat1, feat2)] = h_stat
    except Exception as e:
        print(f"Error calculating H-statistic for {feat1} and {feat2}: {e}")

# Filter pairs by H-statistic threshold
significant_pairs = [(f1, f2) for (f1, f2), h_stat in interaction_strengths.items() 
                    if h_stat is not None and h_stat > 0.1]

print(f"\nFound {len(significant_pairs)} feature pairs with H-statistic > 0.1")
print(f"Generating PDP interaction plots for significant pairs...")

# Create PDP plots only for significant pairs
for feat1, feat2 in tqdm(significant_pairs, desc="Generating PDP plots"):
    print(f"\nCreating PDP interaction plot for {feat1} and {feat2}")
    plt.figure(figsize=(18, 16))
    try:
        fig, h_stat = create_interaction_pdp(
            surrogate_model, X_train, 
            [feat1, feat2], 
            [feat1, feat2]
        )
        
        if h_stat is not None:
            plt.suptitle(f"PDP Interaction: {feat1} vs {feat2}\nFriedman H-statistic: {h_stat:.4f}", fontsize=16)
        
        # Create a clean filename by removing special characters
        filename = f'pdp_interaction_{feat1.replace(" ", "_")}_{feat2.replace(" ", "_")}.png'
        plt.savefig(filename)
        plt.close()
        print(f"Enhanced PDP interaction plot saved as '{filename}'")
    except Exception as e:
        print(f"Error creating PDP plot for {feat1} and {feat2}: {e}")
        plt.close()

# Create a summary of significant interactions
print("\nCreating interaction strength summary...")
summary_data = []

for (feat1, feat2), h_stat in interaction_strengths.items():
    if h_stat is not None and h_stat > 0.1:
        summary_data.append({
            'Feature 1': feat1,
            'Feature 2': feat2,
            'H-statistic': h_stat
        })

# Convert to DataFrame and sort by interaction strength
summary_df = pd.DataFrame(summary_data).sort_values('H-statistic', ascending=False)

# Create a summary visualization
plt.figure(figsize=(15, 10))
plt.bar(
    range(len(summary_df)),
    summary_df['H-statistic'],
    alpha=0.8
)

# Create labels
labels = [f"{row['Feature 1'][:10]}...\nvs\n{row['Feature 2'][:10]}..." 
          for _, row in summary_df.iterrows()]
plt.xticks(range(len(summary_df)), labels, rotation=90)
plt.ylabel('Friedman H-statistic')
plt.title('Interaction Strength Summary for Significant Feature Pairs (H-statistic > 0.1)')

# Add explanation text
plt.figtext(0.5, 0.02,
            "Interaction Strength Summary:\n" +
            "• Only showing feature pairs with H-statistic > 0.1\n" +
            "• Higher H-statistic values indicate stronger interactions\n" +
            "• These pairs represent the most meaningful feature interactions in the model",
            ha='center', fontsize=10,
            bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))

plt.tight_layout()
plt.subplots_adjust(bottom=0.2)
plt.savefig('significant_pdp_interactions_summary.png')
plt.close()

print("\nInteraction strength summary for significant pairs:")
print(summary_df)
print("\nSummary visualization saved as 'significant_pdp_interactions_summary.png'")

# ... continue with existing code ...

# Add comparison of three-way interactions with their constituent pairwise interactions
print("\nComparing three-way interactions with constituent pairwise interactions...")

# Create a visualization to compare three-way interactions with their constituent pairwise interactions
plt.figure(figsize=(15, 12))

# For each three-way interaction, get its components
three_way_vs_pairwise = []

for idx, row in three_way_df.iterrows():
    f1, f2, f3 = row['Feature 1'], row['Feature 2'], row['Feature 3']
    three_way_h = row['H-statistic']
    
    # Get corresponding pairwise interactions
    pairs = [(f1, f2), (f1, f3), (f2, f3)]
    pair_names = [f"{p[0]} & {p[1]}" for p in pairs]
    
    # Find H-statistics for each pair
    pair_h_stats = []
    for pair in pairs:
        # Try both orderings as the dataframe may have them in either order
        try:
            pair_h = pairwise_h_df[(pairwise_h_df['Feature 1'] == pair[0]) & 
                                 (pairwise_h_df['Feature 2'] == pair[1])]['H-statistic'].values[0]
        except (IndexError, KeyError):
            try:
                pair_h = pairwise_h_df[(pairwise_h_df['Feature 1'] == pair[1]) & 
                                     (pairwise_h_df['Feature 2'] == pair[0])]['H-statistic'].values[0]
            except (IndexError, KeyError):
                # If not found, assume a low value
                pair_h = 0.001
        
        pair_h_stats.append(pair_h)
    
    # Calculate the synergy (how much more the three-way interaction explains beyond pairwise)
    # Using max pairwise as baseline
    max_pairwise = max(pair_h_stats)
    synergy = three_way_h - max_pairwise
    
    # Add to dataframe
    three_way_vs_pairwise.append({
        'Three-way Interaction': f"{f1[:5]}_{f2[:5]}_{f3[:5]}",
        'Three-way H': three_way_h,
        'Max Pairwise H': max_pairwise,
        'Synergy': synergy,
        'Pair 1': pair_names[0],
        'Pair 1 H': pair_h_stats[0],
        'Pair 2': pair_names[1],
        'Pair 2 H': pair_h_stats[1],
        'Pair 3': pair_names[2],
        'Pair 3 H': pair_h_stats[2],
    })

comparison_df = pd.DataFrame(three_way_vs_pairwise)

# Create a subplot with multiple visualizations
plt.subplot(2, 2, 1)
# Create a bar chart comparing three-way vs max pairwise H-statistic
comparison_data = pd.DataFrame({
    'Three-way H': comparison_df['Three-way H'],
    'Max Pairwise H': comparison_df['Max Pairwise H']
})
comparison_data.plot(kind='bar', ax=plt.gca())
plt.title('Three-way vs Max Pairwise H-statistic')
plt.ylabel('H-statistic')
plt.xlabel('Three-way Interaction Index')
plt.xticks(rotation=45)

# Create a heatmap showing all values in a matrix
plt.subplot(2, 2, 2)
# Reshape data for heatmap
heatmap_data = []
for _, row in comparison_df.iterrows():
    heatmap_data.append([row['Three-way H'], row['Pair 1 H'], row['Pair 2 H'], row['Pair 3 H']])

heatmap_df = pd.DataFrame(
    heatmap_data, 
    columns=['Three-way', 'Pair 1', 'Pair 2', 'Pair 3'],
    index=[f"3way_{i+1}" for i in range(len(comparison_df))]
)
sns.heatmap(heatmap_df, annot=True, cmap='YlOrRd', fmt='.3f')
plt.title('H-statistic Matrix: Three-way vs Pairwise')

# Create a scatter plot showing synergy
plt.subplot(2, 2, 3)
plt.bar(comparison_df['Three-way Interaction'], comparison_df['Synergy'])
plt.axhline(y=0, color='black', linestyle='--')
plt.title('Interaction Synergy (Three-way H - Max Pairwise H)')
plt.ylabel('Synergy Value')
plt.xticks(rotation=90)

# Create a radar/spider chart to visualize the components
plt.subplot(2, 2, 4, polar=True)
for i, row in comparison_df.iterrows():
    values = [row['Three-way H'], row['Pair 1 H'], row['Pair 2 H'], row['Pair 3 H'], row['Three-way H']]
    angles = np.linspace(0, 2*np.pi, len(values), endpoint=True)
    plt.plot(angles, values, linewidth=2, label=f"3way_{i+1}")
    plt.fill(angles, values, alpha=0.1)

plt.xticks(angles[:-1], ['Three-way', 'Pair 1', 'Pair 2', 'Pair 3'])
plt.title('Radar Chart of Interaction Strengths')
plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

# Add explanation
explanation_text = """
HOW TO INTERPRET THESE PLOTS:

1. Bar Chart: Compares three-way interaction strength to the strongest constituent pairwise interaction
2. Heatmap: Shows the full matrix of interactions; warmer colors = stronger interactions
3. Synergy Chart: Shows how much additional variance is explained by three-way interactions
   - Positive values: three-way interaction adds value beyond pairwise interactions
   - Negative values: three-way interaction explains less than the best pairwise interaction
4. Radar Chart: Visualizes the pattern of interaction strengths across different combinations

Key Insight: Three-way interactions with high synergy values capture complex relationships that would be 
missed by only analyzing pairwise interactions.
"""

plt.figtext(0.5, 0.01, explanation_text, ha='center', fontsize=10,
            bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))

plt.tight_layout()
plt.subplots_adjust(bottom=0.3)
plt.savefig('three_way_pairwise_comparison.png')
print("Three-way vs pairwise interaction comparison saved as 'three_way_pairwise_comparison.png'")

# Generate a summary table with detailed statistics
summary_table = comparison_df[['Three-way Interaction', 'Three-way H', 'Max Pairwise H', 'Synergy']]
summary_table = summary_table.sort_values('Synergy', ascending=False)

print("\nThree-way interaction synergy analysis:")
print(summary_table)

# End of script

def create_three_way_pdp(surrogate_model, X_train, features, feature_names=None, grid_resolution=10):
    """Create detailed partial dependence interaction plots for three features"""
    if feature_names is None:
        feature_names = features
    
    # Calculate grid points for each feature
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
    
    # Create 3D grid
    grid_points = list(itertools.product(*feature_values))
    pdp_values = np.zeros((grid_resolution, grid_resolution, grid_resolution))
    
    # Calculate PDP values
    for i, val1 in enumerate(feature_values[0]):
        for j, val2 in enumerate(feature_values[1]):
            for k, val3 in enumerate(feature_values[2]):
                X_temp = X_train.copy()
                X_temp[features[0]] = val1
                X_temp[features[1]] = val2
                X_temp[features[2]] = val3
                pdp_values[i, j, k] = np.mean(surrogate_model.predict(X_temp))
    
    # Create figure with multiple views
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Slice views at different levels
    for idx, level in enumerate([0.25, 0.5, 0.75]):
        ax = fig.add_subplot(2, 3, idx+1)
        level_idx = int(level * (grid_resolution-1))
        
        # Create slice
        slice_data = pdp_values[:, :, level_idx]
        
        # Plot heatmap
        sns.heatmap(
            slice_data,
            ax=ax,
            cmap='viridis',
            xticklabels=np.round(feature_values[0], 2),
            yticklabels=np.round(feature_values[1], 2)
        )
        
        ax.set_title(f'Slice at {features[2]} = {feature_values[2][level_idx]:.2f}')
        ax.set_xlabel(features[0])
        ax.set_ylabel(features[1])
    
    # 2. 3D scatter plot with important points
    ax = fig.add_subplot(2, 3, 4, projection='3d')
    
    # Get important points (high PDP values)
    threshold = np.percentile(pdp_values, 90)
    important_points = np.where(pdp_values > threshold)
    
    scatter = ax.scatter(
        [feature_values[0][i] for i in important_points[0]],
        [feature_values[1][i] for i in important_points[1]],
        [feature_values[2][i] for i in important_points[2]],
        c=pdp_values[important_points],
        cmap='viridis'
    )
    
    ax.set_xlabel(features[0])
    ax.set_ylabel(features[1])
    ax.set_zlabel(features[2])
    ax.set_title('High Impact Regions (Top 10%)')
    plt.colorbar(scatter, ax=ax, label='Predicted Value')
    
    # 3. Interaction strength analysis
    ax = fig.add_subplot(2, 3, 5)
    
    # Calculate pairwise interactions at different levels
    interaction_strengths = []
    for level_idx in range(grid_resolution):
        h_stat, _ = calculate_friedman_h_statistic(
            surrogate_model,
            X_train,
            [features[0], features[1]],
            grid_resolution=grid_resolution
        )
        interaction_strengths.append(h_stat)
    
    plt.plot(feature_values[2], interaction_strengths, marker='o')
    plt.xlabel(features[2])
    plt.ylabel('Pairwise H-statistic')
    plt.title('Interaction Strength Variation')
    
    # 4. Distribution of PDP values
    ax = fig.add_subplot(2, 3, 6)
    sns.histplot(pdp_values.flatten(), kde=True, ax=ax)
    plt.xlabel('Predicted Value')
    plt.ylabel('Frequency')
    plt.title('Distribution of Predictions')
    
    # Calculate three-way H-statistic
    h_stat, _ = calculate_friedman_h_statistic(
        surrogate_model,
        X_train,
        features,
        grid_resolution=grid_resolution
    )
    
    # Add overall title with H-statistic
    plt.suptitle(
        f"Three-way PDP Interaction Analysis\n"
        f"Features: {', '.join(features)}\n"
        f"H-statistic: {h_stat:.4f}",
        fontsize=16
    )
    
    # Add explanation text
    explanation_text = """
    HOW TO INTERPRET THIS VISUALIZATION:
    
    1. Slice Views (top): Show interaction between two features at different levels of the third feature
    2. 3D Scatter (bottom left): Highlights regions of strong three-way interaction
    3. Interaction Strength (bottom middle): Shows how pairwise interactions vary with third feature
    4. Distribution (bottom right): Shows the range and frequency of predicted values
    
    The H-statistic measures the strength of the three-way interaction:
    • Values close to 0 indicate weak or no interaction
    • Values above 0.1 suggest meaningful interactions
    • Values above 0.3 indicate strong three-way dependencies
    """
    
    plt.figtext(
        0.5, 0.02,
        explanation_text,
        ha='center',
        fontsize=10,
        bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5')
    )
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.85, bottom=0.2)
    
    return fig, h_stat

# Generate three-way PDP interaction plots
print("\nGenerating three-way PDP interaction plots...")

# Get top features based on importance
top_features = importance_df.head(5).index.tolist()

# Create combinations of three features
feature_triplets = list(itertools.combinations(top_features, 3))

# Generate plots for each triplet
three_way_results = []
for triplet in tqdm(feature_triplets[:5], desc="Generating three-way PDP plots"):
    print(f"\nCreating three-way PDP plot for {', '.join(triplet)}")
    try:
        fig, h_stat = create_three_way_pdp(
            surrogate_model,
            X_train,
            list(triplet),
            grid_resolution=8
        )
        
        # Save the plot
        filename = f"three_way_pdp_{'_'.join([t.split('_')[0] for t in triplet])}.png"
        plt.savefig(filename, bbox_inches='tight', dpi=300)
        plt.close()
        
        # Store results
        three_way_results.append({
            'Features': triplet,
            'H-statistic': h_stat
        })
        
        print(f"Three-way PDP plot saved as '{filename}'")
        
    except Exception as e:
        print(f"Error creating three-way PDP plot for {triplet}: {e}")
        plt.close()

# Create summary of three-way interactions
if three_way_results:
    summary_df = pd.DataFrame(three_way_results)
    summary_df['Feature_Names'] = summary_df['Features'].apply(lambda x: ' + '.join(t.split('_')[0] for t in x))
    
    plt.figure(figsize=(12, 6))
    plt.bar(summary_df['Feature_Names'], summary_df['H-statistic'])
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Three-way H-statistic')
    plt.title('Summary of Three-way Interaction Strengths')
    
    plt.tight_layout()
    plt.savefig('three_way_pdp_summary.png')
    plt.close()
    
    print("\nThree-way interaction summary:")
    print(summary_df.sort_values('H-statistic', ascending=False))

def create_three_way_interaction_heatmap(three_way_df, pairwise_h_df):
    """
    Create a comprehensive heatmap visualization for three-way interactions.
    
    Parameters:
    -----------
    three_way_df : pandas DataFrame
        DataFrame containing three-way interaction results
    pairwise_h_df : pandas DataFrame
        DataFrame containing pairwise interaction results
    """
    # Create a matrix to store three-way interaction strengths
    features = list(set(
        list(three_way_df['Feature 1'].unique()) +
        list(three_way_df['Feature 2'].unique()) +
        list(three_way_df['Feature 3'].unique())
    ))
    n_features = len(features)
    
    # Initialize 3D matrix for three-way interactions
    interaction_matrix = np.zeros((n_features, n_features, n_features))
    
    # Fill the matrix with three-way H-statistics
    for _, row in three_way_df.iterrows():
        i = features.index(row['Feature 1'])
        j = features.index(row['Feature 2'])
        k = features.index(row['Feature 3'])
        h_stat = row['H-statistic']
        
        # Fill all permutations to make it symmetric
        for indices in itertools.permutations([i, j, k]):
            interaction_matrix[indices] = h_stat
    
    # Create subplots for different slices of the 3D matrix
    fig = plt.figure(figsize=(20, 15))
    
    # Calculate number of slices to show (use quartiles of features)
    slice_positions = [int(n_features * p) for p in [0.25, 0.5, 0.75]]
    
    # Create heatmaps for different slices
    for idx, slice_pos in enumerate(slice_positions):
        ax = fig.add_subplot(2, 2, idx + 1)
        slice_data = interaction_matrix[:, :, slice_pos]
        
        # Create heatmap
        sns.heatmap(
            pd.DataFrame(slice_data, index=features, columns=features),
            cmap='YlOrRd',
            annot=True,
            fmt='.3f',
            ax=ax
        )
        
        ax.set_title(f'Three-way Interaction Slice at {features[slice_pos]}')
        
        # Rotate x-axis labels for better readability
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        plt.setp(ax.get_yticklabels(), rotation=0)
    
    # Add a summary heatmap showing maximum interaction strength
    ax = fig.add_subplot(2, 2, 4)
    max_interactions = np.max(interaction_matrix, axis=2)
    
    sns.heatmap(
        pd.DataFrame(max_interactions, index=features, columns=features),
        cmap='YlOrRd',
        annot=True,
        fmt='.3f',
        ax=ax
    )
    ax.set_title('Maximum Three-way Interaction Strength')
    
    # Rotate x-axis labels for better readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    plt.setp(ax.get_yticklabels(), rotation=0)
    
    # Add explanation text
    explanation_text = """
    HOW TO INTERPRET THIS VISUALIZATION:
    
    1. Each heatmap shows a slice of three-way interactions at different feature levels
    2. Colors indicate interaction strength (darker = stronger interaction)
    3. Numbers show the exact H-statistic values
    4. The bottom-right heatmap shows the maximum interaction strength across all combinations
    5. Look for:
       • Dark regions indicating strong three-way interactions
       • Patterns across different slices
       • Features that consistently show strong interactions
    """
    
    plt.figtext(0.5, 0.02, explanation_text, ha='center', fontsize=10,
                bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    plt.suptitle('Three-way Interaction Heatmap Analysis', fontsize=16, y=0.95)
    plt.tight_layout()
    plt.subplots_adjust(top=0.90, bottom=0.20)
    
    return fig

# Create and save the three-way interaction heatmap
print("\nCreating three-way interaction heatmap visualization...")
interaction_heatmap_fig = create_three_way_interaction_heatmap(three_way_df, pairwise_h_df)
plt.savefig('three_way_interaction_heatmap.png')
plt.close()
print("Three-way interaction heatmap saved as 'three_way_interaction_heatmap.png'")

# ... existing code ...

def generate_summary_report(data_processed, importance_df, shap_importance, interaction_df, three_way_df, 
                        impact_df, dt_model, train_score, test_score, rule_df):
    """
    Generate a dynamic summary report based on the actual analysis results.
    
    Parameters:
    -----------
    data_processed : pandas DataFrame
        The preprocessed dataset
    importance_df : pandas DataFrame
        Feature importance results from multiple methods
    shap_importance : pandas DataFrame
        SHAP-based feature importance results
    interaction_df : pandas DataFrame
        Pairwise interaction results
    three_way_df : pandas DataFrame
        Three-way interaction results
    impact_df : pandas DataFrame
        Threshold impact analysis results
    dt_model : DecisionTreeRegressor
        The trained decision tree model
    train_score : float
        R² score on training data
    test_score : float
        R² score on test data
    rule_df : pandas DataFrame
        Decision tree rule analysis results
    """
    
    # Dataset statistics
    n_samples, n_features = data_processed.shape
    numeric_features = data_processed.select_dtypes(include=[np.number]).columns.tolist()
    
    # Get top features from different methods
    top_shap_features = shap_importance.head(5).index.tolist()
    top_importance_features = importance_df.head(5).index.tolist()
    
    # Get significant interactions
    significant_pairs = []
    for i in range(len(interaction_df)):
        for j in range(i+1, len(interaction_df)):
            if interaction_df.iloc[i, j] > 0.1:  # H-statistic threshold
                significant_pairs.append({
                    'features': (interaction_df.columns[i], interaction_df.columns[j]),
                    'h_stat': interaction_df.iloc[i, j]
                })
    
    # Get top thresholds
    top_thresholds = impact_df.head(5)
    
    report = f"""# Dynamic Analysis Summary Report

## 1. Dataset Overview
- **Dataset Size**: {n_samples} samples, {n_features} features
- **Feature Types**: {len(numeric_features)} numeric features
- **Date Range**: {data_processed['date'].min()} to {data_processed['date'].max()}

## 2. Key Features and Their Importance

### Top Features by SHAP Analysis:
"""
    
    # Add SHAP importance details
    for feature in top_shap_features:
        shap_value = shap_importance.loc[feature, 'mean_abs_shap']
        report += f"- `{feature}` (SHAP value: {shap_value:.3f})\n"
    
    report += "\n### Top Features by Combined Importance Metrics:\n"
    for feature in top_importance_features:
        mdi = importance_df.loc[feature, 'MDI']
        perm = importance_df.loc[feature, 'Permutation']
        lofo = importance_df.loc[feature, 'LOFO']
        report += f"- `{feature}`:\n  - MDI: {mdi:.3f}\n  - Permutation: {perm:.3f}\n  - LOFO: {lofo:.3f}\n"
    
    report += "\n## 3. Feature Interactions\n\n### Significant Pairwise Interactions (H-statistic > 0.1):\n"
    for pair in significant_pairs:
        report += f"- `{pair['features'][0]}` × `{pair['features'][1]}` (H = {pair['h_stat']:.3f})\n"
    
    report += "\n### Top Three-way Interactions:\n"
    for _, row in three_way_df.head(3).iterrows():
        report += f"- {row['Feature 1']} × {row['Feature 2']} × {row['Feature 3']} (H = {row['H-statistic']:.3f})\n"
    
    report += "\n## 4. Critical Thresholds\n\n### Top Impactful Thresholds:\n"
    for _, row in top_thresholds.iterrows():
        report += f"- `{row['Feature']}` at {row['Threshold']:.2f}:\n"
        report += f"  - Impact: {row['Impact']:.2f}\n"
        report += f"  - Affects {row['Data_Percentage_Above']:.1f}% of the data\n"
    
    report += f"""
## 5. Model Performance

### Decision Tree Results:
- Training R²: {train_score:.4f}
- Test R²: {test_score:.4f}

### Top Decision Rules:
"""
    
    for i, (_, row) in enumerate(rule_df.head(3).iterrows()):
        report += f"- Rule {i+1}:\n"
        report += f"  - Covers {row['Samples']} samples ({row['Percentage']:.1f}% of data)\n"
        report += f"  - Prediction accuracy: {100 - row['Abs_Error']:.1f}%\n"
    
    report += """
## 6. Key Insights and Recommendations

### Feature Relationships:"""
    
    # Add insights about feature relationships
    if significant_pairs:
        report += "\n- Strong interactions detected between crack spread features"
        report += "\n- Multiple threshold effects suggest non-linear relationships"
    
    report += "\n\n### Model Implications:"
    if test_score < train_score - 0.3:
        report += "\n- Evidence of overfitting suggests need for regularization"
        report += "\n- Consider using simpler models or gathering more data"
    
    report += "\n\n### Recommendations:"
    report += "\n1. Focus monitoring on identified threshold levels"
    report += "\n2. Consider interaction effects in feature engineering"
    report += "\n3. Monitor strongest feature pairs for regime changes"
    
    report += f"""
## 7. Limitations
- Dataset size ({n_samples} samples) may limit generalizability
- Training-test performance gap indicates potential overfitting
- Market conditions and relationships may change over time
"""
    
    # Save the report
    with open('dynamic_analysis_summary.md', 'w') as f:
        f.write(report)
    print("Dynamic analysis summary report saved as 'dynamic_analysis_summary.md'")

# Update the script to call the dynamic report generation
# Add after all analyses are complete:
print("\nGenerating dynamic analysis summary report...")
generate_summary_report(
    data_processed=data_processed,
    importance_df=importance_df,
    shap_importance=shap_importance,
    interaction_df=interaction_df,
    three_way_df=three_way_df,
    impact_df=impact_df,
    dt_model=dt_model,
    train_score=train_score,
    test_score=test_score,
    rule_df=rule_df
)