# Surrogate-Assisted Feature Extraction (SAFE)

This repository provides a Python implementation for **Surrogate-Assisted Feature Extraction (SAFE)**. It helps analyze complex datasets by building simpler, interpretable surrogate models (like Random Forests or XGBoost) to understand feature importance, interactions, and their impact on a target variable, especially in time-series contexts.

## Overview

Understanding complex machine learning models ("black boxes") can be challenging. SAFE tackles this by:

1.  **Training a Surrogate Model:** A simpler, more interpretable model (e.g., Random Forest) is trained to mimic the behavior of the original data or a complex model (if one were present).
2.  **Analyzing the Surrogate:** Techniques like SHAP, Permutation Importance, LOFO, and rule extraction are applied to the surrogate model.
3.  **Extracting Insights:** The analysis reveals key features, critical thresholds, significant interactions, and potential causal relationships.
4.  **Generating Reports:** Comprehensive reports with visualizations are generated to summarize the findings.

## Key Features

-   **Multiple Importance Methods:**
    -   SHAP (SHapley Additive exPlanations)
    -   Permutation Feature Importance
    -   MDI (Mean Decrease in Impurity) for tree-based models
    -   LOFO (Leave One Feature Out)
    -   RuleFit for extracting interpretable rules
-   **Interaction Analysis:**
    -   Approximate Friedman's H-statistic (pairwise & three-way)
    -   Interaction stability analysis via bootstrapping
    -   SHAP interaction values
    -   Custom Partial Dependence Plots (PDP) for visualizing interactions
-   **Condition/Threshold Analysis:**
    -   Identifies frequent decision rules/thresholds from tree models.
    -   Calculates the importance of these conditions using secondary models (Permutation, LOFO).
    -   Compares linear models built using only these condition-based features.
-   **Time Series Capabilities:**
    -   Time-based train/test splitting.
    -   Stationarity testing (ADF).
    -   Granger causality testing (feature -> target and condition feature -> target).
-   **Comprehensive Reporting:**
    -   Generates detailed Markdown reports.
    -   Exports key results to JSON and text summary files.
    -   Saves numerous plots (HTML, PNG) for visualization.
-   **Configurability:**
    -   Supports Random Forest and XGBoost as surrogate models.
    -   Allows filtering data based on categories.
    -   Allows specifying which features (exogenous variables) to include.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/surrogate-assisted-feature-extraction-gemini.git
    cd surrogate-assisted-feature-extraction-gemini
    ```
    *(Replace `yourusername` with the actual username/organization)*

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python3 -m venv venv  # Use python3 or python depending on your system
    # On macOS/Linux:
    source venv/bin/activate
    # On Windows:
    .\venv\Scripts\activate
    ```

3.  **Install dependencies:** The `requirements.txt` file contains the exact versions of all necessary packages.
    ```bash
    pip install -r requirements.txt
    ```
    *Note: Some plotting functions (like XGBoost tree visualization or SHAP plots) might require additional system libraries like Graphviz. Consult the respective library documentation if you encounter issues.* 

## Usage

1.  **Prepare Data:** Place your data file (e.g., `your_data.csv`) in the `_data/` directory.
2.  **Configure Analysis:** Modify the configuration constants at the top of `safe_analysis.py`:
    -   `DATA_PATH`: Path to your input CSV file (relative to the project root).
    -   `TARGET_VARIABLE`: The name of the column containing the target variable.
    -   `DATE_COLUMN`: The name of the column containing dates/timestamps (required for time series analysis).
    -   `GROUPING_COLUMN`: (Optional) Column name to filter data by (e.g., 'country', 'category'). Set to `None` to disable filtering.
    -   `FILTER_VALUE`: (Optional) The specific value to keep when filtering using `GROUPING_COLUMN`.
    -   `TEST_MONTHS`: Number of *most recent* months to use for the test set.
    -   `SURROGATE_MODEL_TYPE`: Choose either `'random_forest'` or `'xgboost'`.
    -   `EXOGENOUS_VARIABLES`: Controls which features are used.
        -   `'all'` (default): Uses all numeric columns except the target and date/grouping columns.
        -   `['feature1', 'feature2', ...]` : A list of specific column names to use as features.
    -   `OUTPUT_DIR`: Directory where all results (plots, reports, JSON) will be saved.
    -   `N_TOP_FEATURES`: Number of features to show in summary tables/plots.
    -   `N_BOOTSTRAP_SAMPLES`: Number of bootstrap iterations for interaction stability analysis (set to 0 to disable).
    -   `N_TREES_TO_VISUALIZE`: For tree models, how many individual trees to visualize.
3.  **Run Analysis:**
    -   Option A (Using the provided run script): Modify `run_analysis.py` if needed (it's currently set up for the example) and run:
        ```bash
        python run_analysis.py
        ```
    -   Option B (Integrating into your workflow): Import `ReportGenerator` and relevant functions from `safe_analysis.py` into your own script. You'll need to load data, split it, set the `.X_train`, `.y_train`, `.X_test`, `.y_test`, `.feature_names`, `.target_name` attributes of the `ReportGenerator` instance, and then call the desired analysis and reporting methods.

## Outputs

The analysis generates several outputs in the specified `OUTPUT_DIR` (default: `safe_analysis/`):

-   **`analysis_summary_report.md`:** A detailed Markdown report summarizing all findings, including:
    -   Dataset Overview
    -   Key Feature Importance (SHAP, Permutation, LOFO, MDI, Combined Rank)
    -   Feature Interactions (H-statistic, SHAP Interactions, Stability)
    -   Critical Thresholds/Conditions (Importance, Linear Model Comparison)
    -   Surrogate Model Performance
    -   RuleFit Extracted Rules (if applicable)
    -   Stationarity Tests
    -   Granger Causality Analysis (Features & Conditions)
    -   Limitations
-   **`analysis_results.json`:** A JSON file containing detailed numerical results, importance scores, rankings, stability metrics, etc.
-   **`summary_findings_detailed.txt`:** A text file with more verbose summaries of importance rankings, conditions, interactions, and model comparisons.
-   **Numerous Plots (`.png`, `.html`):**
    -   Feature importance plots (Permutation, LOFO, SHAP, MDI, RuleFit)
    -   Condition importance plots
    -   Interaction heatmaps (H-statistic)
    -   Interaction PDPs (Contour, 3D, Heatmap)
    -   Individual PDP/ICE plots for top features
    -   Granger causality bar charts
    -   Decision tree visualizations (if applicable)
    -   Engineered feature distributions
    -   SHAP summary and force plots

## Project Structure

```
.
├── safe_analysis.py       # Core analysis logic and ReportGenerator class
├── run_analysis.py        # Example script to run the analysis
├── requirements.txt       # Python package dependencies (pinned versions)
├── README.md              # This file
├── .gitignore             # Specifies intentionally untracked files
├── _data/                 # Directory for input data files (e.g., *.csv)
└── safe_analysis/         # Default output directory (created on run)
    ├── analysis_results.json
    ├── analysis_summary_report.md
    ├── summary_findings_detailed.txt
    ├── *.png                # Various plot image files
    └── *.html               # Interactive plotly plots
```

## Contributing

Contributions are welcome! Please follow these steps:

1.  Fork the repository.
2.  Create a new branch for your feature or bug fix (`git checkout -b feature/your-feature-name`).
3.  Make your changes and commit them (`git commit -am 'Add some feature'`).
4.  Push your changes to the branch (`git push origin feature/your-feature-name`).
5.  Open a Pull Request against the main branch of the original repository.

## License

This project is licensed under the MIT License - see the LICENSE file (if one exists) or contact the repository owner for details.

## Acknowledgments

-   This work builds upon concepts from surrogate modeling for machine learning interpretability.
-   Utilizes libraries like Scikit-learn, XGBoost, SHAP, LOFO, Statsmodels, Plotly, and others.
-   Inspired by the need for better understanding of complex model predictions.
