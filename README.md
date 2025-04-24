# Surrogate-Assisted Feature Extraction (SAFE) Analysis

This repository implements a comprehensive workflow inspired by Surrogate-Assisted Feature Extraction (SAFE) techniques to analyze and interpret complex machine learning models, particularly tree-based ensembles like Random Forests and XGBoost.

The primary script, `safe_analysis.py`, trains a surrogate model on a given dataset and then applies various interpretability methods to understand feature importance, interactions, critical conditions, and potential causal relationships.

## Features & Analyses Implemented

The script performs the following analyses:

1.  **Data Preparation:** Loads data, handles time indexing, optional filtering, NaN imputation, and time series splitting.
2.  **Surrogate Model Training:** Trains a specified surrogate model (`RandomForestRegressor` or `XGBoostRegressor`) on the training data and evaluates its performance (RMSE, R²) on both training and test sets.
3.  **Rule/Condition Extraction:**
    *   Extracts decision rules and conditions (feature <= threshold, feature > threshold) from the surrogate Random Forest model.
    *   Visualizes sample decision trees.
    *   Analyzes the frequency of unique thresholds used per feature.
4.  **Feature Importance:** Calculates feature importance using multiple methods:
    *   Permutation Importance (on train set)
    *   Leave-One-Feature-Out (LOFO) Importance
    *   SHAP (Mean Absolute Values, Summary Plot)
    *   Mean Decrease in Impurity (MDI - if applicable)
    *   Combines ranks from different methods into a `mean_rank`.
5.  **Condition Importance Analysis:**
    *   Creates binary features based on the most frequent conditions extracted from the surrogate model.
    *   Trains a secondary Random Forest model using original + condition features.
    *   Calculates Permutation, MDI, and SHAP importance for these *condition features* within the secondary model.
    *   Trains Linear Regression, RidgeCV, and LassoCV models using *only* the condition features to assess their linear predictive power and analyze coefficients.
6.  **Rule Extraction (RuleFit):**
    *   Trains an `imodels.RuleFitRegressor` to extract interpretable rules (combinations of conditions).
    *   Visualizes the importance (coefficient magnitude) of the top rules.
7.  **Feature Interaction Analysis:**
    *   Calculates approximate pairwise Friedman's H-statistic and visualizes it as a heatmap.
    *   Calculates interaction stability for top pairs using bootstrapping.
    *   Calculates approximate 3-way H-statistic for triplets formed from top features and visualizes results.
    *   Generates detailed pairwise Partial Dependence Plots (PDP) and Individual Conditional Expectation (ICE) plots using scikit-learn.
    *   Generates custom 4-panel Matplotlib interaction plots (Contour, 3D Surface, Heatmap, Overlay) for all feature pairs.
    *   Generates interactive 3D PDP plots using Plotly for pairs formed from top N features.
    *   Calculates SHAP interaction values.
8.  **Causality Analysis:**
    *   Performs ADF stationarity tests on features.
    *   Performs Granger causality tests to check if features help predict the target variable.
9.  **Feature Engineering Comparison:**
    *   Optionally creates engineered features based on the most important conditions and interactions identified.
    *   Trains a simple Linear Regression model with original features and another with original + engineered features.
    *   Compares the performance (RMSE, R²) of the two linear models on the test set.
10. **Reporting:**
    *   Generates a high-level summary report in Markdown (`analysis_summary_report.md`).
    *   Saves detailed findings (rankings, coefficients, etc.) to a text file (`summary_findings_detailed.txt`).
    *   Saves key numerical results, DataFrames, and configurations to a JSON file (`analysis_results.json`).

## Prerequisites

*   Python 3.x
*   Required libraries (see `requirements.txt`). Install using:
    ```bash
    pip install -r requirements.txt
    ```
*   **(Optional for XGBoost Tree Viz)** Graphviz: If using XGBoost and wanting tree visualizations, you need to install the Graphviz library (`pip install graphviz`) and ensure the Graphviz binaries are installed on your system and added to your PATH.
*   **(Optional for Static Plotly)** Kaleido: Needed to save Plotly figures as static images (`.png`). Install via `pip install -U kaleido`.

## Configuration

Key parameters can be configured directly at the top of the `safe_analysis.py` script:

*   `DATA_PATH`: Path to the input CSV data file.
*   `TARGET_VARIABLE`: Name of the target column in the data.
*   `DATE_COLUMN`: Name of the date/time column for time series indexing.
*   `GROUPING_COLUMN`, `FILTER_VALUE`: Optional parameters to filter the dataset before analysis.
*   `FEATURES_TO_INCLUDE`: Controls which features are used. Default is `"all"` (uses all numeric features except the target). Can be set to a list of column names (e.g., `["feature1", "feature3"]`) to use only those specific features.
*   `TEST_MONTHS`: Number of months to use for the time series test split.
*   `SURROGATE_MODEL_TYPE`: Choose between `'random_forest'` or `'xgboost'`.
*   `N_TOP_FEATURES_FOR_PDP_3D`: Limits the generation of computationally intensive Plotly 3D PDP plots to pairs formed from these top features.
*   `OUTPUT_DIR`: Directory where all reports, plots, and results will be saved.

## Usage

1.  Place your input data CSV file in the `_data/` directory (or update `DATA_PATH` in the script).
2.  Ensure all prerequisites are installed (`pip install -r requirements.txt`).
3.  Configure the parameters at the top of `safe_analysis.py` as needed.
4.  Run the script from your terminal:
    ```bash
    python safe_analysis.py
    ```
    *(Note: The script can take a significant amount of time to run due to computationally intensive calculations like interaction analysis and SHAP values.)*

## Outputs

All outputs are saved into the directory specified by `OUTPUT_DIR` (default: `safe_analysis/`).

*   **Reports:**
    *   `analysis_summary_report.md`: High-level summary report in Markdown format.
    *   `summary_findings_detailed.txt`: Detailed text file containing full importance rankings, interaction scores, condition importance, model coefficients, etc.
*   **Data:**
    *   `analysis_results.json`: JSON file containing key results (performance metrics, importance scores, interaction values, conditions, rules) in a structured format.
*   **Plots:** Numerous plots saved as `.html` (interactive Plotly) and/or `.png` (static Matplotlib/Seaborn/Plotly), covering:
    *   Feature Importance (Permutation, LOFO, SHAP, MDI, Condition Importance)
    *   RuleFit Rule Importance
    *   Interactions (H-statistic heatmap, 3-way H-statistic bar, custom PDPs, Plotly 3D PDPs)
    *   PDP/ICE plots
    *   Surrogate Tree visualization
    *   Engineered Feature Distributions
    *   Granger Causality
    *   SHAP Summary & Force plots

## Limitations

*   **Approximations:** Some calculations, like Friedman's H-statistic, are approximations based on PDP variances and may differ from other implementations.
*   **Computational Cost:** Some analyses (LOFO, SHAP interactions, interaction stability, custom PDPs for all pairs) can be computationally expensive and time-consuming, especially on large datasets.
*   **Assumptions:** Granger causality assumes stationarity, which might be violated by some features (ADF tests are performed).
*   **Data Dependence:** Results are specific to the dataset and the chosen surrogate model.
*   **SHAP Additivity:** The SHAP calculation for condition features uses `check_additivity=False` to bypass potential errors, which might indicate minor inconsistencies.
