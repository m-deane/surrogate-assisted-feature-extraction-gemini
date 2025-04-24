"""
Main Dash application for the Model Studio.
"""
import dash
import dash_bootstrap_components as dbc # Add this import back

# --- Set Matplotlib Backend --- #
# Must be done BEFORE any other matplotlib imports (e.g., indirect ones via shap/dalex)
import matplotlib
matplotlib.use('Agg') # Use non-interactive backend suitable for servers/threading
# -------------------------- #

from dash import html, dcc, Input, Output, State, callback, dash_table, no_update
import plotly.graph_objects as go
import plotly.express as px # For simpler plots like distributions/scatter
import plotly.io as pio # Import plotly IO
import traceback # For detailed error logging
import os
import pandas as pd # Needed for observation selection
import numpy as np # Needed for residuals calculation
import scipy.stats as stats # Import scipy for QQ plot
import mlflow # Added
import mlflow.sklearn # Added

# Import data loader and explainer creator
from data_loader import load_data
# from explainer import create_explainer # We create explainer directly now
import dalex as dx # Import dalex
# import joblib # No longer needed

# Imports for reference tab
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression # Added for regression example

# --- Plotly Template --- #
pio.templates.default = "plotly_white" # Set default template closer to demo

# --- Constants --- (Define paths at the top)
# MODEL_PATH = 'trained_model/model.joblib' # Old path
MODEL_PATH = os.path.join('trained_model', 'model_mlflow') # MLflow model DIRECTORY
DATA_PATH = '_data/preem.csv'
TARGET_COLUMN = 'target'
# Assuming 'date' is the name in the CSV and gets cleaned to 'date'
# Adjust if clean_col_names changes it differently
DATE_COLUMN = 'date'

# --- Available Plot Choices --- #
# Match case/naming with DALEX functions where possible
PLOT_CHOICES = {
    # Instance Level
    'break_down': 'Break Down',
    'shap': 'Shapley Values (Instance)',
    'lime_explanation': 'LIME Explanation (Local)',
    'predict_profile': 'Ceteris Paribus Profile (CP)',
    # Feature Level
    'model_profile_pdp': 'Partial Dependence Plot (PDP)',
    'model_profile_ale': 'Accumulated Local Effects (ALE)',
    'model_profile_ice': 'Individual Conditional Expectation (ICE)',
    'feature_distribution': 'Feature Distribution',
    'target_vs_feature': 'Target vs Feature',
    # Model Level
    'variable_importance': 'Permutation Feature Importance',
    'shap_importance': 'SHAP Feature Importance (Avg Abs)',
    'model_performance': 'Model Performance Metrics',
    'model_diagnostics': 'Residuals vs Fitted',
    'residuals_qq': 'Residuals QQ Plot',
    'scale_location': 'Scale-Location Plot',
    'residuals_vs_feature': 'Residuals vs Feature',
    'residuals_acf': 'Residuals Autocorrelation (ACF)',
    'residuals_pacf': 'Residuals Partial Autocorrelation (PACF)',
    'feature_correlation': 'Feature Correlation Heatmap',
    # Interactions (Require 2 Variables)
    'model_profile_pdp_2d': '2D Partial Dependence (PDP)',
    'model_profile_ale_2d': '2D Accumulated Local Effects (ALE)',
}
# Plot types requiring observation selection
NEEDS_OBSERVATION = ['break_down', 'shap', 'predict_profile', 'lime_explanation']
# Plot types requiring a single variable selection
NEEDS_VARIABLE = [
    'model_profile_pdp',
    'model_profile_ale',
    'model_profile_ice',
    'residuals_vs_feature',
    'feature_distribution',
    'target_vs_feature'
]
# Plots requiring a *second* variable selector
NEEDS_VARIABLE_2 = ['model_profile_pdp_2d', 'model_profile_ale_2d']
# Plot types that *don't* need observation or variable
STANDALONE_PLOTS = [
    'variable_importance',
    'shap_importance',
    'model_performance',
    'model_diagnostics',
    'residuals_qq',
    'scale_location',
    'residuals_acf',
    'residuals_pacf',
    'feature_correlation'
]

# --- Reference Text Content --- #
REFERENCE_TEXT = {
    'break_down':
"""
### Break Down Plot

**What it is:** Shows the contribution of each feature to a *single* prediction, starting from the average prediction and sequentially adding each feature's impact.

**How to interpret:**
*   Each bar represents the change in prediction caused by that specific feature, given the values of the features above it.
*   Positive values increase the prediction; negative values decrease it.
*   The final value (`prediction`) should match the model's output for the selected observation.

**Use for understanding:** Helps understand *why* a specific prediction was made by attributing it to individual feature contributions. Useful for debugging surprising predictions or explaining outcomes to stakeholders.
""",
    'shap':
"""
### Shapley Values (Instance) Plot

**What it is:** Explains a *single* prediction using Shapley values...
*(rest of explanation)*
""",
    'lime_explanation':
"""
### LIME Explanation (Local)

**What it is:** Explains a *single* prediction by fitting a simple, interpretable linear model to the behavior of the complex model in the *local neighborhood* around that specific instance.

**How to interpret:**
*   The bar chart shows the features most influential for that specific prediction *according to the local linear approximation*.
*   Positive weights (e.g., green) indicate the feature value pushed the prediction higher (locally); negative weights (e.g., red) pushed it lower.
*   The weights indicate the *local* linear importance, which might differ from global importance.

**Use for understanding:** Provides an alternative local explanation, focusing on linear effects in the instance's vicinity. Useful for gaining trust by showing a simple approximation matches the complex model locally.
""",
    'predict_profile':
"""
### Ceteris Paribus Profile (CP)

**What it is:** Shows how the model's prediction for a *single selected observation* changes as the value of *one selected feature* is varied, while all other features are held constant at the observation's original values.

**How to interpret:**
*   The x-axis represents the values of the selected feature.
*   The y-axis shows the corresponding model prediction.
*   The curve reveals the model's sensitivity to that specific feature *around the point* of the selected observation.
*   A vertical line often indicates the actual value of the feature for the selected observation.

**Use for understanding:** Allows detailed examination of how a single feature affects the prediction for a specific instance. Helps understand local feature effects and non-linearities.
""",
    'model_profile_pdp':
"""
### Partial Dependence Plot (PDP)

**What it is:** Shows the *average* marginal effect of one (or two) features on the model's prediction, across all observations in the dataset.

**How to interpret (1D):**
*   The x-axis is the value of the selected feature.
*   The y-axis is the average prediction value for that feature value, marginalizing over the distribution of other features.
*   The curve shows the average trend of how the prediction changes as the feature changes.

**Use for understanding:** Reveals the average relationship between a feature and the prediction, isolating its effect from other features. Helps understand global feature importance and identify non-linearities on average.
""",
    'model_profile_ale':
"""
### Accumulated Local Effects (ALE) Plot

**What it is:** Similar to PDP, it shows the average effect of a feature on predictions. However, ALE is generally considered more reliable than PDP when features are correlated, as it calculates effects based on *conditional* distributions rather than marginalizing.

**How to interpret (1D):**
*   The x-axis is the feature value.
*   The y-axis represents the *accumulated* change in the prediction centered around the mean effect. A flat line indicates no effect.
*   Steeper slopes indicate stronger local effects.

**Use for understanding:** Provides a less biased estimate of the average feature effect compared to PDP, especially with correlated features. Helps understand the global impact of a feature while mitigating issues caused by unrealistic feature combinations.
""",
    'model_profile_ice':
"""
### Individual Conditional Expectation (ICE) Plot

**What it is:** Shows how the prediction for *each individual observation* changes as one feature varies (similar to Ceteris Paribus, but plotted for many/all observations simultaneously).

**How to interpret:**
*   Each thin line represents one observation.
*   The x-axis is the feature value, the y-axis is the prediction.
*   It visualizes the heterogeneity of the feature effect across the dataset. Lines parallel to the (often overlaid) PDP line suggest homogeneous effects; diverging lines indicate interactions.

**Use for understanding:** Uncovers heterogeneity and potential interactions missed by average plots like PDP/ALE. Allows seeing if the average effect shown by PDP is representative of individual instances.
""",
    'feature_distribution':
"""
### Feature Distribution Plot

**What it is:** A simple histogram showing the distribution of values for a *single selected feature* in the dataset.

**How to interpret:**
*   Shows the frequency (count) of different value ranges for the feature.
*   Reveals skewness, modality (number of peaks), and potential outliers.

**Use for understanding:** Essential for basic data exploration. Helps understand the range and common values of a feature, which provides context for interpreting other plots like PDP or CP.
""",
    'target_vs_feature':
"""
### Target vs Feature Plot

**What it is:** A scatter plot showing the relationship between a *single selected feature* (x-axis) and the *actual target variable* (y-axis).

**How to interpret:**
*   Reveals the raw relationship between the feature and the target before model fitting.
*   A trendline (like LOWESS) can help visualize the central tendency.

**Use for understanding:** Basic data exploration. Helps assess the predictive potential of a feature and identify potential non-linear relationships that the model might need to capture.
""",
    'variable_importance':
"""
### Permutation Feature Importance Plot

**What it is:** Ranks features based on how much the model's overall performance (e.g., R-squared, RMSE) drops when that feature's values are randomly shuffled...
*(rest of explanation)*
""",
    'shap_importance':
"""
### SHAP Feature Importance (Avg Abs) Plot

**What it is:** Ranks features based on the average *absolute* Shapley value across all instances in the dataset. It measures the average magnitude of impact a feature has on the predictions.

**How to interpret:**
*   Features with higher average absolute Shapley values have a larger impact on the model's output magnitude, regardless of direction.
*   It provides a different view of global importance compared to permutation importance, focusing on contribution magnitude.

**Use for understanding:** Identifies features with the largest average impact on prediction magnitude. Complements permutation importance.
""",
    'model_performance':
"""
### Model Performance Metrics

**What it is:** Displays quantitative metrics evaluating the model's overall accuracy on the dataset (or a defined test set).

**How to interpret:**
*   Metrics depend on model type (regression/classification).
*   **Regression:** Lower RMSE/MAE is better; R-squared closer to 1 is better.
*   **Classification:** Accuracy, Precision, Recall, F1-score (closer to 1 is better), AUC (closer to 1 is better).

**Use for understanding:** Provides a crucial baseline measure of how well the model performs *overall*. Essential context before diving into explaining *why* it makes certain predictions.
""",
    'model_diagnostics':
"""
### Residuals vs Fitted Plot

**What it is:** Plots the model's errors (residuals: actual - predicted) against the predicted values.

**How to interpret:**
*   Ideally, points should scatter randomly around the horizontal line at zero with roughly constant variance.
*   Patterns (like a curve or a funnel shape) indicate potential problems: non-linearity not captured by the model, heteroscedasticity (non-constant variance), or outliers.

**Use for understanding:** A fundamental diagnostic plot to check if the model's assumptions are met and if the errors are well-behaved across the range of predictions.
""",
    'residuals_qq':
"""
### Residuals QQ Plot

**What it is:** Compares the quantiles of the model residuals against the theoretical quantiles of a normal distribution.

**How to interpret:**
*   If residuals are normally distributed, points should fall closely along the diagonal reference line.
*   Deviations (e.g., S-shapes, points far from the line) indicate non-normality, which might affect confidence intervals or subsequent statistical tests.

**Use for understanding:** Checks the normality assumption of the model's errors.
""",
    'scale_location':
"""
### Scale-Location Plot

**What it is:** Plots the square root of the absolute residuals against the fitted (predicted) values.

**How to interpret:**
*   Helps check the assumption of equal variance (homoscedasticity).
*   If the variance is constant, points should spread roughly evenly along the y-axis across the range of fitted values. A funnel shape (spread increasing with fitted values) indicates heteroscedasticity.
*   The trendline should ideally be flat.

**Use for understanding:** Diagnoses whether the error variance depends on the predicted value, which can invalidate certain statistical inferences.
""",
    'residuals_vs_feature':
"""
### Residuals vs Feature Plot

**What it is:** Plots the model's errors (residuals) against the values of a *single selected feature*.

**How to interpret:**
*   Similar to Residuals vs Fitted, but checks for patterns related to a specific predictor.
*   Ideally, points should scatter randomly around zero without obvious trends.
*   Patterns might suggest the model hasn't fully captured the relationship with that feature (e.g., remaining non-linearity).

**Use for understanding:** Helps identify if the model's errors are systematically related to a particular input feature.
""",
    'feature_correlation':
"""
### Feature Correlation Heatmap

**What it is:** A heatmap showing the pairwise Pearson correlation coefficient between all numerical features (and often the target variable).

**How to interpret:**
*   Colors indicate the strength and direction of linear correlation (e.g., red for positive, blue for negative, white/light for near zero).
*   Values closer to 1 or -1 indicate strong linear relationships.
*   Helps identify multicollinearity (high correlation between predictor features).

**Use for understanding:** Data exploration tool to understand linear relationships between variables. High correlation between predictors can affect model coefficient stability and interpretation.
""",
    'model_profile_pdp_2d':
"""
### 2D Partial Dependence Plot (PDP)

**What it is:** Shows the average marginal effect of *two selected features* on the model prediction simultaneously.

**How to interpret:**
*   Typically shown as a heatmap or contour plot.
*   Axes represent the values of the two selected features.
*   Color/height indicates the average prediction value for combinations of those two features.
*   Patterns reveal how the prediction changes in response to both features and can highlight interaction effects (e.g., non-parallel contour lines).

**Use for understanding:** Visualizes potential interaction effects between two features and their combined impact on the average prediction.
""",
    'model_profile_ale_2d':
"""
### 2D Accumulated Local Effects (ALE) Plot

**What it is:** Similar to 2D PDP, but uses ALE calculations, making it more robust to correlated features.

**How to interpret:**
*   Axes represent the two features.
*   Color/height represents the centered ALE value (main effects of both variables removed), highlighting the interaction effect itself.
*   Flat regions indicate no interaction; varying colors/heights indicate interactions.

**Use for understanding:** A more reliable way to visualize two-way interaction effects, especially when the two features are correlated.
""",
    'residuals_acf':
"""
### Residuals Autocorrelation (ACF) Plot

**What it is:** Shows the correlation of the model residuals with lagged versions of themselves.

**How to interpret:**
*   The x-axis represents the lag (how many time steps back).
*   The y-axis shows the correlation coefficient.
*   Bars extending outside the shaded confidence interval indicate statistically significant autocorrelation at that lag.
*   Ideally, for a well-specified model (especially in time series), residuals should be uncorrelated, meaning most bars (except lag 0) should be within the confidence interval.

**Use for understanding:** Primarily used for time series models to check if there is remaining temporal structure in the errors that the model hasn't captured. Significant autocorrelation suggests the model could be improved (e.g., by adding lagged variables).
""",
    'residuals_pacf':
"""
### Residuals Partial Autocorrelation (PACF) Plot

**What it is:** Shows the correlation between residuals at different lags after removing the effects of correlations at shorter lags.

**How to interpret:**
*   Similar to ACF, bars outside the confidence interval indicate significant partial autocorrelation.
*   PACF helps identify the order of an Autoregressive (AR) process. Significant spikes at specific lags (e.g., lag 1 and 2, but not beyond) might suggest an AR(2) component is missing from the model.

**Use for understanding:** Used alongside ACF for time series residual analysis to identify the nature of remaining temporal dependencies (AR or MA components).
"""
}

# --- Data Loading, Explainers, Initial Calcs ---
print("Loading main data and creating main explainer...")
explainer = None # Main explainer for user data
ref_explainer = None # Reference explainer for Iris
residuals_df = None
model_perf_fig = None
correlation_matrix = None
error_message = None
data_length = 0
feature_names = []
date_series = None
date_strings = []
initial_figure = go.Figure().update_layout(title="Select Plot Type")

try:
    # --- Main Explainer Setup --- #
    print("Loading main data...")
    X, y, df = load_data(DATA_PATH, target_col=TARGET_COLUMN)
    if X is None: raise ValueError("Main data loading failed.")

    print(f"Loading model from MLflow path: {MODEL_PATH}...")
    # Load model using MLflow
    try:
        model = mlflow.sklearn.load_model(MODEL_PATH)
    except Exception as load_err:
        print(f"Error loading MLflow model: {load_err}")
        print(f"Ensure the model was saved correctly to {MODEL_PATH} by running model_trainer.py")
        raise load_err # Re-raise after logging

    print("Creating main DALEX explainer...")
    explainer = dx.Explainer(model, X, y, label="User Model (RF Preem)")
    if not explainer: raise ValueError("Main explainer creation failed.")

    print("Main explainer created. Pre-calculating...")
    data_length = len(explainer.data)
    feature_names = explainer.data.columns.tolist()
    print(f"Using feature names: {feature_names}")

    # Store and format dates if date column exists in the loaded df
    if DATE_COLUMN in df.columns:
        date_series = df[DATE_COLUMN]
        date_strings = date_series.dt.strftime('%Y-%m-%d').tolist() # Format as YYYY-MM-DD
    else:
        print(f"Warning: Date column '{DATE_COLUMN}' not found in loaded DataFrame.")
        date_series = pd.Series(range(data_length)) # Fallback to index if no date
        date_strings = [str(i) for i in range(data_length)]

    # Calculate residuals once
    model_diag = explainer.model_diagnostics()
    residuals_df = pd.DataFrame({
        'residuals': model_diag.result['residuals'].values,
        'fitted_values': model_diag.result['y_hat'].values
    })
    residuals_df.index = explainer.data.index
    print("Residuals calculated.")

    # Performance
    perf = explainer.model_performance()
    model_perf_fig = perf.plot(show=False, title="Model Performance")
    # Apply layout updates now if needed
    model_perf_fig.update_layout(
         margin=dict(l=40, r=20, t=40, b=40),
         title_font_size=14, title_x=0.5
    )

    # Correlation Matrix (use final feature names from explainer.data)
    df_for_corr = explainer.data.copy()
    df_for_corr[TARGET_COLUMN] = explainer.y # Assume target name is simple
    correlation_matrix = df_for_corr.corr()

    print("Main pre-calculations complete.")

    # --- Reference Explainer Setup --- #
    print("Setting up reference explainer (Iris Regression)...")
    ref_residuals_df = None # Initialize ref data
    ref_correlation_matrix = None
    ref_model_perf_fig = None
    try:
        iris = load_iris()
        # Use petal_width (index 3) as the regression target
        # Use first 3 features as predictors
        feature_cols_ref = [c.replace(' (cm)', '').replace(' ', '_') for c in iris.feature_names[:3]]
        target_col_ref = iris.feature_names[3].replace(' (cm)', '').replace(' ', '_')

        X_ref = pd.DataFrame(iris.data[:, :3], columns=feature_cols_ref)
        y_ref = iris.data[:, 3] # Petal width as target

        X_train_ref, X_test_ref, y_train_ref, y_test_ref = train_test_split(
            X_ref, y_ref, test_size=0.3, random_state=42
        )

        # Use Linear Regression model
        ref_model = LinearRegression()
        ref_model.fit(X_train_ref, y_train_ref)

        # Create explainer using test data for realistic examples
        ref_explainer = dx.Explainer(
            ref_model,
            X_test_ref,
            y_test_ref,
            label="Reference Model (Iris Regression)",
            verbose=False
        )
        print("Reference regression explainer created successfully.")

        if ref_explainer:
            # Calculate ref residuals
            ref_model_diag = ref_explainer.model_diagnostics()
            ref_residuals_df = pd.DataFrame({
                'residuals': ref_model_diag.result['residuals'].values,
                'fitted_values': ref_model_diag.result['y_hat'].values
            })
            ref_residuals_df.index = ref_explainer.data.index

            # Calculate ref performance plot
            ref_perf = ref_explainer.model_performance()
            ref_model_perf_fig = ref_perf.plot(show=False, title="Reference Model Performance")
            ref_model_perf_fig.update_layout(margin=dict(l=40, r=20, t=40, b=40), title_font_size=14, title_x=0.5)

            # Calculate ref correlation matrix
            ref_df_for_corr = ref_explainer.data.copy()
            ref_df_for_corr['target'] = ref_explainer.y
            ref_correlation_matrix = ref_df_for_corr.corr()

            print("Reference explainer created successfully and data pre-calculated.")

    except Exception as ref_e:
        print(f"Warning: Failed to create reference explainer or pre-calculate data - {ref_e}")
        ref_explainer = None # Ensure it's None if setup fails

except Exception as e:
    print(f"An error occurred during app initialization: {e}")
    error_message = f"App Init Error: {e}. Check console logs."
    print(traceback.format_exc())
    explainer = None
    ref_explainer = None
    initial_figure = go.Figure().update_layout(title=error_message)
    model_perf_fig = initial_figure

# --- Initialize the Dash app ---
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server # Expose server for deployment

# --- Define the layout ---
# We will use dbc components for a cleaner layout, similar to the R version

# Helper function for plot selectors
def create_plot_selector(id_suffix):
    return dcc.Dropdown(
        id=f'plot-selector-{id_suffix}',
        options=[{'label': name, 'value': key} for key, name in PLOT_CHOICES.items()],
        value=None, # Default to nothing selected
        placeholder="Select Plot Type...",
        clearable=False,
        disabled=explainer is None
    )

# Sidebar Controls
controls = dbc.Card([
    dbc.CardBody([
        # Add the logo at the top of the controls panel
        html.Img(src=app.get_asset_url('617f744b-91d0-4fb9-b7cf-9236cfe6ffa7.png'),
                 style={'width': '25%', 'margin-bottom': '10px', 'backgroundColor': 'white'}), # Add white background
        html.H4("Controls", className="card-title"),
        html.Hr(),

        # Add Fitted vs Actual Time Series Plot
        html.P("Fitted vs Actual Values:", style={'font-weight':'bold'}),
        dcc.Loading(dcc.Graph(id='fitted-vs-actual-ts', config={'displayModeBar': False})), # Add graph, hide modebar
        html.Hr(),

        # Observation Selection (Slider + Input + Date Display)
        html.P("Select Observation:"),
        dcc.Slider(
            id='observation-slider',
            min=0,
            max=data_length - 1 if data_length > 0 else 0,
            step=1,
            value=0,
            marks=None,
            tooltip={"placement": "bottom", "always_visible": True},
            disabled=explainer is None
        ),
        # Display selected date/index
        html.Div([
            "Selected:",
            html.Strong(id='selected-observation-display', children=date_strings[0] if date_strings else "N/A", className="ms-1")
        ], className="mt-2 text-center text-muted small"),
        # Optional: Keep numeric input for direct index entry if desired
        # dbc.Input(id='observation-input', ... ),
        html.Hr(),

        # Variable Selection
        html.P("Select Variable:"),
        dcc.Dropdown(
            id='variable-selector',
            options=[{'label': i, 'value': i} for i in feature_names],
            value=feature_names[0] if feature_names else None,
            clearable=False,
            disabled=explainer is None
        ),
        # Second variable selector (for 2D plots), initially hidden
        html.Div([
            html.P("Select Second Variable (for 2D plots):"),
            dcc.Dropdown(
                id='variable-selector-2',
                options=[{'label': i, 'value': i} for i in feature_names],
                value=feature_names[1] if len(feature_names) > 1 else None,
                clearable=False,
                disabled=explainer is None
            ),
        ], id='variable-selector-2-container', style={'display': 'none'}, className="mt-3"),
        html.Hr(),

        # Plot Selectors
        html.P("Top-Left Plot:"),
        create_plot_selector('tl'),
        html.P("Top-Right Plot:", className="mt-3"),
        create_plot_selector('tr'),
        html.P("Bottom-Left Plot:", className="mt-3"),
        create_plot_selector('bl'),
        html.P("Bottom-Right Plot:", className="mt-3"),
        create_plot_selector('br'),

    ])
], className="mb-3 controls-card") # Added class

# Main Content Area with 2x2 Grid AND Reference Tab
# We need to redefine content_plots to be dbc.Tabs
content_area = dbc.Tabs([
    dbc.Tab(label="Model Explorer", children=[
        html.Div([
            dbc.Row([
                dbc.Col(dcc.Loading(dcc.Graph(id='plot-top-left', figure=initial_figure)), width=6),
                dbc.Col(dcc.Loading(dcc.Graph(id='plot-top-right', figure=initial_figure)), width=6),
            ]),
            dbc.Row([
                dbc.Col(dcc.Loading(dcc.Graph(id='plot-bottom-left', figure=initial_figure)), width=6),
                dbc.Col(dcc.Loading(dcc.Graph(id='plot-bottom-right', figure=initial_figure)), width=6),
            ], className="mt-3"),
        ], className="plots-container mt-3") # Add margin top to tab content
    ]),
    dbc.Tab(label="Plot Reference", children=[
        dbc.Row([
            dbc.Col([
                html.P("Select plot type for reference:", className="mt-3"),
                dcc.Dropdown(
                    id='reference-plot-selector',
                    options=[{'label': name, 'value': key} for key, name in PLOT_CHOICES.items()],
                    value=list(PLOT_CHOICES.keys())[0], # Default to first plot type
                    clearable=False,
                ),
                dcc.Loading(
                    dcc.Graph(id='reference-plot-display', className="mt-3")
                ),
                dcc.Markdown(id='reference-explanation-display', className="mt-3 reference-text")
            ], width=12)
        ])
    ])
])

# Final App Layout
app.layout = html.Div([ # Main container
    dcc.Location(id='url', refresh=False), # Add dcc.Location component
    html.Link(
        rel='stylesheet',
        href='/assets/custom.css' # Path to custom CSS
    ),
    dbc.Container([
        # Remove the image row from here
        # dbc.Row([dbc.Col(html.Img(src=app.get_asset_url('617f744b-91d0-4fb9-b7cf-9236cfe6ffa7.png'), style={'height':'50px'}), # Adjust height as needed
        #                  width={'size': 6, 'offset': 3}, # Center the logo (adjust size/offset if needed)
        #                  className='text-center') # Center align the image
        #         ], className="mb-3 app-title"),
        dbc.Row([
            dbc.Col(controls, id='controls-column', width=4), # Increased controls width
            dbc.Col(content_area, id='plots-column', width=8) # Decreased plots width
        ])
    ], fluid=True)
])

# Populate the layout placeholders (No longer needed if defined directly above)
# app.layout["controls-column"].children = controls
# app.layout["plots-column"].children = content_plots

# --- Add Callbacks ---

def create_error_figure(title, error_message):
    "Helper to create a Plotly figure showing an error."
    fig = go.Figure().update_layout(title=title)
    fig.add_annotation(text=f"Error: {error_message}", showarrow=False)
    return fig

# NEW Callback to control visibility of the second variable selector
@callback(
    Output('variable-selector-2-container', 'style'),
    # Depend on *all* plot selectors
    Input('plot-selector-tl', 'value'),
    Input('plot-selector-tr', 'value'),
    Input('plot-selector-bl', 'value'),
    Input('plot-selector-br', 'value'),
)
def toggle_variable_selector_2(plot_tl, plot_tr, plot_bl, plot_br):
    # Check if any selected plot needs a second variable
    selected_plots = [plot_tl, plot_tr, plot_bl, plot_br]
    needs_second_var = any(p in NEEDS_VARIABLE_2 for p in selected_plots if p is not None)

    if needs_second_var:
        return {'display': 'block'} # Show the container
    else:
        return {'display': 'none'}  # Hide the container

# Main callback to update all plots (add variable_2 input)
@callback(
    Output('plot-top-left', 'figure'),
    Output('plot-top-right', 'figure'),
    Output('plot-bottom-left', 'figure'),
    Output('plot-bottom-right', 'figure'),
    Input('plot-selector-tl', 'value'),
    Input('plot-selector-tr', 'value'),
    Input('plot-selector-bl', 'value'),
    Input('plot-selector-br', 'value'),
    Input('observation-slider', 'value'),
    Input('variable-selector', 'value'),
    Input('variable-selector-2', 'value') # Add input for second variable
)
def update_plots(plot_tl, plot_tr, plot_bl, plot_br, obs_index, variable, variable_2):
    if explainer is None:
        return initial_figure, initial_figure, initial_figure, initial_figure

    # Generate figures for each quadrant
    fig_tl = generate_figure(plot_tl, obs_index, variable, variable_2)
    fig_tr = generate_figure(plot_tr, obs_index, variable, variable_2)
    fig_bl = generate_figure(plot_bl, obs_index, variable, variable_2)
    fig_br = generate_figure(plot_br, obs_index, variable, variable_2)

    return fig_tl, fig_tr, fig_bl, fig_br

# Callback to sync slider and input field
@callback(
    Output('selected-observation-display', 'children'),
    Input('observation-slider', 'value')
)
def update_observation_display(selected_index):
    if date_strings and 0 <= selected_index < len(date_strings):
        return date_strings[selected_index]
    elif 0 <= selected_index < data_length: # Fallback if dates weren't loaded
        return f"Index: {selected_index}"
    return "N/A"

# NEW Callback for Reference Tab
@callback(
    Output('reference-plot-display', 'figure'),
    Output('reference-explanation-display', 'children'),
    Input('reference-plot-selector', 'value')
)
def update_reference_section(selected_plot_key):
    if ref_explainer is None:
        no_ref_explainer_text = "Reference explainer could not be created. Cannot show examples."
        return create_error_figure("Reference Error", no_ref_explainer_text), no_ref_explainer_text

    if not selected_plot_key:
        return go.Figure().update_layout(title="Select Plot Type"), "Please select a plot type from the dropdown."

    # Generate reference figure using the new function
    ref_fig = generate_reference_figure(selected_plot_key)

    # Get explanation text
    explanation_text = REFERENCE_TEXT.get(selected_plot_key, "Explanation not available for this plot type.")

    return ref_fig, explanation_text

# NEW Callback for Fitted vs Actual Time Series
@callback(
    Output('fitted-vs-actual-ts', 'figure'),
    Input('url', 'pathname') # Add Input to trigger on load
)
def update_fitted_actual_ts(pathname): # Add pathname argument
    if explainer is None or residuals_df is None or date_series is None:
        return create_error_figure("Error", "Data for plot not available.")

    fig = go.Figure()

    # Add Actual values trace
    fig.add_trace(go.Scatter(x=date_series, y=explainer.y,
                         mode='lines', name='Actual',
                         line=dict(color='blue')))

    # Add Fitted values trace
    fig.add_trace(go.Scatter(x=date_series, y=residuals_df['fitted_values'],
                         mode='lines', name='Fitted',
                         line=dict(color='red')))

    # Update layout for compactness
    fig.update_layout(
        title=None, # No title needed, context is clear
        xaxis_title=None, # Hide x-axis title
        yaxis_title="Value",
        margin=dict(l=40, r=10, t=30, b=10), # Increased top margin slightly for legend
        height=150, # Make plot shorter
        showlegend=True,
        # Move legend to top, reduce font size
        legend=dict(orientation="h", yanchor="top", y=1.15, xanchor="center", x=0.5),
        font=dict(size=10)
    )
    # Optionally hide x-axis ticks/labels if too cluttered
    # fig.update_xaxes(showticklabels=False)

    return fig

# --- Helper Plotting Functions (To be implemented) --- #

def plot_residuals_vs_feature(explainer_obj, residuals_data, feature_name):
    """Generates a scatter plot of residuals vs. a specific feature.
    Args:
        explainer_obj: The DALEX explainer object.
        residuals_data (pd.DataFrame): DataFrame with a 'residuals' column.
        feature_name (str): Name of the feature.
    """
    if feature_name not in explainer_obj.data.columns:
        raise ValueError(f"Feature '{feature_name}' not found in explainer data.")
    if residuals_data is None or 'residuals' not in residuals_data.columns:
         raise ValueError("Valid residuals data is not available.")

    plot_data = pd.DataFrame({
        'residuals': residuals_data['residuals'],
        feature_name: explainer_obj.data[feature_name]
    })

    fig = px.scatter(
        plot_data, x=feature_name, y='residuals',
        title=f"Residuals vs {feature_name}",
        labels={'residuals': 'Residuals', feature_name: feature_name},
        trendline="lowess", trendline_color_override="red"
    )
    fig.add_hline(y=0, line_dash="dash", line_color="grey")
    return fig

def plot_feature_distribution(explainer_obj, feature_name):
    """Generates a histogram for the distribution of a specific feature.
    Args:
        explainer_obj: The DALEX explainer object.
        feature_name (str): Name of the feature.
    """
    if feature_name not in explainer_obj.data.columns:
        raise ValueError(f"Feature '{feature_name}' not found in explainer data.")

    fig = px.histogram(
        explainer_obj.data, x=feature_name,
        title=f"Distribution of {feature_name}",
        labels={feature_name: feature_name}
    )
    return fig

def plot_target_vs_feature(explainer_obj, feature_name):
    """Generates a scatter plot of the target variable vs. a specific feature.
    Args:
        explainer_obj: The DALEX explainer object.
        feature_name (str): Name of the feature.
    """
    if feature_name not in explainer_obj.data.columns:
        raise ValueError(f"Feature '{feature_name}' not found in explainer data.")

    plot_data = pd.DataFrame({
        'target': explainer_obj.y,
        feature_name: explainer_obj.data[feature_name]
    })

    fig = px.scatter(
        plot_data, x=feature_name, y='target',
        title=f"Target vs {feature_name}",
        labels={'target': 'Target', feature_name: feature_name},
        trendline="lowess", trendline_color_override="red"
    )
    return fig

def plot_residuals_qq(residuals_data):
    """Generates a QQ plot for residuals.
    Args:
        residuals_data (pd.DataFrame): DataFrame with a 'residuals' column.
    """
    if residuals_data is None or 'residuals' not in residuals_data.columns:
         raise ValueError("Valid residuals data is not available.")
    residuals = residuals_data['residuals'].dropna()
    osm, osr = stats.probplot(residuals, dist="norm")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=osm[0], y=osr, mode='markers', name='Residuals'))
    fig.add_trace(go.Scatter(x=osm[0], y=osm[1], mode='lines', name='Normal', line=dict(color='red')))
    fig.update_layout(
        title="Normal Q-Q Plot of Residuals",
        xaxis_title="Theoretical Quantiles",
        yaxis_title="Standardized Residuals",
        showlegend=False
    )
    return fig

def plot_scale_location(residuals_data):
    """Generates a Scale-Location plot (sqrt(|residuals|) vs fitted).
    Args:
        residuals_data (pd.DataFrame): DataFrame with 'residuals' and 'fitted_values' columns.
    """
    if residuals_data is None or not all(c in residuals_data.columns for c in ['residuals', 'fitted_values']):
        raise ValueError("Valid residuals and fitted values data is not available.")
    sqrt_abs_residuals = np.sqrt(np.abs(residuals_data['residuals']))
    plot_data = pd.DataFrame({
        'sqrt_abs_residuals': sqrt_abs_residuals,
        'fitted_values': residuals_data['fitted_values']
    })
    fig = px.scatter(
        plot_data, x='fitted_values', y='sqrt_abs_residuals',
        title="Scale-Location Plot",
        labels={'fitted_values': 'Fitted Values', 'sqrt_abs_residuals': 'Sqrt(|Residuals|)'},
        trendline="lowess", trendline_color_override="red"
    )
    return fig

def plot_correlation_heatmap(corr_matrix, title="Feature Correlation Heatmap"):
    """Generates a heatmap of a given correlation matrix.
    Args:
        corr_matrix (pd.DataFrame): The correlation matrix.
        title (str): Plot title.
    """
    if corr_matrix is None:
        raise ValueError("Correlation matrix is not available.")
    fig = px.imshow(corr_matrix, text_auto='.2f', aspect="auto", title=title)
    fig.update_layout(coloraxis_colorbar=dict(title="Corr"), xaxis_tickangle=-45)
    fig.update_traces(hovertemplate='X: %{x}<br>Y: %{y}<br>Corr: %{z:.2f}<extra></extra>')
    return fig

# --- Main Plot Generation Logic --- #
def generate_figure(plot_key, obs_index, variable, variable_2):
    """Main function to generate a figure based on plot_key and inputs"""
    print(f"Generating figure for key: {plot_key}, obs: {obs_index}, var: {variable}, var_2: {variable_2}")
    if explainer is None:
        return create_error_figure("Error", "Explainer not available")
    if plot_key is None:
        return go.Figure().update_layout(title="Select Plot Type")

    fig = None
    try:
        # --- Standalone Plots --- #
        if plot_key == 'variable_importance':
            var_imp = explainer.model_parts()
            fig = var_imp.plot(show=False, title=PLOT_CHOICES[plot_key])
        elif plot_key == 'shap_importance':
            var_imp_shap = explainer.model_parts(type='shap_wrapper')
            fig = var_imp_shap.plot(show=False, title=PLOT_CHOICES[plot_key])
        elif plot_key == 'model_performance':
            # Use pre-calculated figure for main explainer
            if model_perf_fig:
                return model_perf_fig # Return directly, layout already applied
            else:
                return create_error_figure(PLOT_CHOICES[plot_key], "Performance data not calculated.")
        elif plot_key == 'model_diagnostics':
            model_diag = explainer.model_diagnostics()
            fig = model_diag.plot(show=False, title=PLOT_CHOICES[plot_key])
        elif plot_key == 'residuals_qq':
            # Pass main residuals_df to helper
            fig = plot_residuals_qq(residuals_df)
        elif plot_key == 'scale_location':
            # Pass main residuals_df to helper
            fig = plot_scale_location(residuals_df)
        elif plot_key == 'residuals_acf':
            fig = plot_acf_pacf(residuals_df, plot_type='acf')
        elif plot_key == 'residuals_pacf':
            fig = plot_acf_pacf(residuals_df, plot_type='pacf')
        elif plot_key == 'feature_correlation':
            # Pass main correlation_matrix to helper
            fig = plot_correlation_heatmap(correlation_matrix, title=PLOT_CHOICES[plot_key])

        # --- Plots requiring Observation --- #
        elif plot_key in NEEDS_OBSERVATION:
            if obs_index is None or not (0 <= obs_index < data_length):
                return create_error_figure(PLOT_CHOICES[plot_key], f"Invalid Observation Index: {obs_index}")
            observation = explainer.data.iloc[[obs_index]]
            actual_value = explainer.y[obs_index]
            title_base = PLOT_CHOICES[plot_key]

            if plot_key == 'break_down' or plot_key == 'shap':
                title = f"{PLOT_CHOICES[plot_key]} (Obs: {obs_index}, Actual: {actual_value:.2f})"
                result = explainer.predict_parts(observation, type=plot_key)
                fig = result.plot(show=False, title=title)
            elif plot_key == 'predict_profile':
                if variable is None or variable not in feature_names:
                     return create_error_figure(PLOT_CHOICES[plot_key], f"Invalid Variable Selected: {variable}")
                title = f"{title_base} for '{variable}' (Obs: {obs_index}, Actual: {actual_value:.2f})"
                result = explainer.predict_profile(observation, variables=[variable])
                fig = result.plot(show=False, title=title)
            elif plot_key == 'lime_explanation':
                fig = plot_lime_explanation(explainer, obs_index)

        # --- Plots requiring Variable --- #
        elif plot_key in NEEDS_VARIABLE:
            if variable is None or variable not in feature_names:
                return create_error_figure(PLOT_CHOICES[plot_key], f"Invalid Variable Selected: {variable}")
            title = f"{PLOT_CHOICES[plot_key]} for {variable}"

            if plot_key == 'model_profile_pdp':
                profile = explainer.model_profile(variables=[variable], N=100, type='partial')
                fig = profile.plot(show=False, title=title)
            elif plot_key == 'model_profile_ale':
                profile = explainer.model_profile(variables=[variable], N=100, type='accumulated')
                fig = profile.plot(show=False, title=title)
            elif plot_key == 'model_profile_ice':
                profile = explainer.model_profile(variables=[variable], N=None, type='conditional')
                fig = profile.plot(show=False, title=title)
            elif plot_key == 'residuals_vs_feature':
                fig = plot_residuals_vs_feature(explainer, residuals_df, variable)
            elif plot_key == 'feature_distribution':
                fig = plot_feature_distribution(explainer, variable)
            elif plot_key == 'target_vs_feature':
                fig = plot_target_vs_feature(explainer, variable)

        # --- Plots requiring Two Variables --- #
        elif plot_key in NEEDS_VARIABLE_2:
            if variable is None or variable not in feature_names:
                return create_error_figure(PLOT_CHOICES[plot_key], f"Invalid Variable 1 Selected: {variable}")
            if variable_2 is None or variable_2 not in feature_names:
                return create_error_figure(PLOT_CHOICES[plot_key], f"Invalid Variable 2 Selected: {variable_2}")
            if variable == variable_2:
                return create_error_figure(PLOT_CHOICES[plot_key], "Please select two different variables.")

            title = f"{PLOT_CHOICES[plot_key]} for {variable} and {variable_2}"
            variables_list = [variable, variable_2]

            if plot_key == 'model_profile_pdp_2d':
                profile = explainer.model_profile(variables=variables_list, N=50, type='partial')
                fig = profile.plot(show=False, title=title)
            elif plot_key == 'model_profile_ale_2d':
                profile = explainer.model_profile(variables=variables_list, N=50, type='accumulated')
                fig = profile.plot(show=False, title=title)

        else:
            return create_error_figure("Error", f"Unknown plot key: {plot_key}")

        # --- Apply Common Layout Updates --- #
        if fig:
            fig.update_layout(
                margin=dict(l=40, r=20, t=50, b=60), # Increased bottom margin slightly for rotated labels
                legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5), # Adjusted legend y
                title_font_size=14, title_x=0.5,
                plot_bgcolor='white', # Explicit white background
                yaxis_showgrid=True, # Show only horizontal gridlines
                xaxis_showgrid=False
            )
            # Apply x-axis tick rotation
            fig.update_xaxes(tickangle=-45)

            # Hide legend for plots where it adds little value
            plots_no_legend = [
                'feature_distribution', 'model_diagnostics', 'residuals_vs_feature',
                'target_vs_feature', 'residuals_qq', 'scale_location', 'feature_correlation',
                'model_profile_pdp_2d', 'model_profile_ale_2d',
                'residuals_acf', 'residuals_pacf', 'lime_explanation'
            ]
            if plot_key in plots_no_legend:
                 fig.update_layout(showlegend=False)
        else:
            return create_error_figure("Plot Error", f"Could not generate plot for {plot_key}")

        return fig

    except Exception as e:
        print(f"Error generating plot for {plot_key}: {e}")
        print(traceback.format_exc())
        return create_error_figure(f"Error: {PLOT_CHOICES.get(plot_key, plot_key)}", str(e))

# --- Reference Plot Generation Logic --- #
def generate_reference_figure(plot_key):
    """Generates a plot for the Reference tab using the ref_explainer."""
    print(f"Generating reference figure for key: {plot_key}")
    if ref_explainer is None:
        return create_error_figure("Reference Error", "Reference explainer not available")
    if plot_key is None:
        return go.Figure().update_layout(title="Select Plot Type for Reference")

    # Define fixed inputs for reference plots
    ref_obs_index = 0 # Use the first observation in the test set
    ref_feature_names = ref_explainer.data.columns.tolist()
    ref_variable = ref_feature_names[0]
    ref_variable_2 = ref_feature_names[1] if len(ref_feature_names) > 1 else ref_variable

    fig = None
    try:
        # --- Standalone Plots --- #
        if plot_key == 'variable_importance':
            var_imp = ref_explainer.model_parts()
            fig = var_imp.plot(show=False, title=f"Example: {PLOT_CHOICES[plot_key]}")
        elif plot_key == 'shap_importance':
            var_imp_shap = ref_explainer.model_parts(type='shap_wrapper')
            fig = var_imp_shap.plot(show=False, title=f"Example: {PLOT_CHOICES[plot_key]}")
        elif plot_key == 'model_performance':
            if ref_model_perf_fig:
                return ref_model_perf_fig # Already calculated & formatted
            else:
                 return create_error_figure(f"Example: {PLOT_CHOICES[plot_key]}", "Ref Perf data unavailable.")
        elif plot_key == 'model_diagnostics':
            model_diag = ref_explainer.model_diagnostics()
            fig = model_diag.plot(show=False, title=f"Example: {PLOT_CHOICES[plot_key]}")
        elif plot_key == 'residuals_qq':
            fig = plot_residuals_qq(ref_residuals_df)
            fig.update_layout(title=f"Example: {PLOT_CHOICES[plot_key]}")
        elif plot_key == 'scale_location':
            fig = plot_scale_location(ref_residuals_df)
            fig.update_layout(title=f"Example: {PLOT_CHOICES[plot_key]}")
        elif plot_key == 'feature_correlation':
            fig = plot_correlation_heatmap(ref_correlation_matrix, title=f"Example: {PLOT_CHOICES[plot_key]}")

        # --- Plots requiring Observation --- #
        elif plot_key in NEEDS_OBSERVATION:
            observation = ref_explainer.data.iloc[[ref_obs_index]]
            actual_value = ref_explainer.y[ref_obs_index]
            title_base = f"Example: {PLOT_CHOICES[plot_key]}"

            if plot_key == 'break_down' or plot_key == 'shap':
                title = f"{title_base} (Obs: {ref_obs_index}, Actual: {actual_value:.2f})"
                result = ref_explainer.predict_parts(observation, type=plot_key)
                fig = result.plot(show=False, title=title)
            elif plot_key == 'predict_profile':
                title = f"{title_base} for '{ref_variable}' (Obs: {ref_obs_index}, Actual: {actual_value:.2f})"
                result = ref_explainer.predict_profile(observation, variables=[ref_variable])
                fig = result.plot(show=False, title=title)
            elif plot_key == 'lime_explanation':
                return create_error_figure(f"Example: {PLOT_CHOICES[plot_key]}", "LIME example generation skipped for simplicity.")

        # --- Plots requiring Variable --- #
        elif plot_key in NEEDS_VARIABLE:
            title = f"Example: {PLOT_CHOICES[plot_key]} for {ref_variable}"
            if plot_key == 'model_profile_pdp':
                profile = ref_explainer.model_profile(variables=[ref_variable], N=100, type='partial')
                fig = profile.plot(show=False, title=title)
            elif plot_key == 'model_profile_ale':
                profile = ref_explainer.model_profile(variables=[ref_variable], N=100, type='accumulated')
                fig = profile.plot(show=False, title=title)
            elif plot_key == 'model_profile_ice':
                profile = ref_explainer.model_profile(variables=[ref_variable], N=None, type='conditional')
                fig = profile.plot(show=False, title=title)
            elif plot_key == 'residuals_vs_feature':
                fig = plot_residuals_vs_feature(ref_explainer, ref_residuals_df, ref_variable)
                fig.update_layout(title=f"Example: {PLOT_CHOICES[plot_key]} for {ref_variable}")
            elif plot_key == 'feature_distribution':
                fig = plot_feature_distribution(ref_explainer, ref_variable)
                fig.update_layout(title=f"Example: {PLOT_CHOICES[plot_key]} for {ref_variable}")
            elif plot_key == 'target_vs_feature':
                fig = plot_target_vs_feature(ref_explainer, ref_variable)
                fig.update_layout(title=f"Example: {PLOT_CHOICES[plot_key]} for {ref_variable}")

        # --- Plots requiring Two Variables --- #
        elif plot_key in NEEDS_VARIABLE_2:
            title = f"Example: {PLOT_CHOICES[plot_key]} for {ref_variable} and {ref_variable_2}"
            variables_list = [ref_variable, ref_variable_2]
            if plot_key == 'model_profile_pdp_2d':
                profile = ref_explainer.model_profile(variables=variables_list, N=50, type='partial')
                fig = profile.plot(show=False, title=title)
            elif plot_key == 'model_profile_ale_2d':
                profile = ref_explainer.model_profile(variables=variables_list, N=50, type='accumulated')
                fig = profile.plot(show=False, title=title)
        else:
            return create_error_figure("Reference Error", f"Unknown plot key: {plot_key}")

        # --- Apply Common Layout Updates --- #
        if fig:
            fig.update_layout(
                margin=dict(l=40, r=20, t=50, b=60),
                legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
                title_font_size=14, title_x=0.5,
                plot_bgcolor='white',
                yaxis_showgrid=True,
                xaxis_showgrid=False
            )
            fig.update_xaxes(tickangle=-45)
            # Hide legend for plots where it adds little value
            plots_no_legend = [
                'feature_distribution', 'model_diagnostics', 'residuals_vs_feature',
                'target_vs_feature', 'residuals_qq', 'scale_location', 'feature_correlation',
                'model_profile_pdp_2d', 'model_profile_ale_2d',
                'residuals_acf', 'residuals_pacf', 'lime_explanation'
            ]
            if plot_key in plots_no_legend:
                 fig.update_layout(showlegend=False)
        else:
             return create_error_figure("Plot Error", f"Could not generate reference plot for {plot_key}")

        return fig

    except Exception as e:
        print(f"Error generating reference plot for {plot_key}: {e}")
        print(traceback.format_exc())
        return create_error_figure(f"Error: Example {PLOT_CHOICES.get(plot_key, plot_key)}", str(e))

# --- New Helper Plot Functions --- #

from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf # Keep for potential future use if mpl works
import lime
import lime.lime_tabular

def plot_acf_pacf(residuals_data, plot_type='acf'):
    """Generates ACF or PACF plot for residuals using statsmodels calculation
       and manual Plotly plotting.
    Args:
        residuals_data (pd.DataFrame): DataFrame with a 'residuals' column.
        plot_type (str): 'acf' or 'pacf'.
    """
    if residuals_data is None or 'residuals' not in residuals_data.columns:
        raise ValueError("Valid residuals data is not available.")

    residuals = residuals_data['residuals'].dropna()
    nlags = min(40, len(residuals) // 2 - 1) # Default lags, ensure sufficient data

    if plot_type == 'acf':
        values, confint = acf(residuals, nlags=nlags, alpha=0.05)
        title = "Residuals Autocorrelation (ACF)"
    elif plot_type == 'pacf':
        # Method 'ols' is default and generally good
        values, confint = pacf(residuals, nlags=nlags, alpha=0.05, method='ols')
        title = "Residuals Partial Autocorrelation (PACF)"
    else:
        raise ValueError("Invalid plot_type for plot_acf_pacf. Use 'acf' or 'pacf'.")

    lags = np.arange(len(values))
    conf_upper = confint[:, 1] - values
    conf_lower = values - confint[:, 0]

    fig = go.Figure()
    # Add confidence intervals bands (using fill)
    fig.add_trace(go.Scatter(
        x=np.concatenate([lags, lags[::-1]]),
        y=np.concatenate([conf_upper, conf_lower[::-1]]),
        fill='toself',
        fillcolor='rgba(0,100,80,0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        showlegend=False,
        name='Confidence Interval'
    ))
    # Add ACF/PACF bars/markers
    fig.add_trace(go.Bar(
        x=lags, y=values,
        name=plot_type.upper(),
        marker_color='blue'
    ))

    # Add line at lag 0 for ACF
    if plot_type == 'acf':
        fig.data[1].update(marker_line_color='blue', marker_line_width=0.5) # Style bars

    fig.update_layout(
        title=title,
        xaxis_title="Lag",
        yaxis_title="Correlation",
        yaxis=dict(range=[-1, 1], zeroline=True, zerolinewidth=1, zerolinecolor='grey'),
        xaxis=dict(range=[-0.5, nlags + 0.5]),
        showlegend=False
    )
    return fig

def plot_lime_explanation(explainer_obj, obs_index):
    """Generates a LIME explanation plot for a specific observation.
    Args:
        explainer_obj: The DALEX explainer object (used for data and predict func).
        obs_index (int): The index of the observation to explain.
    """
    if not (0 <= obs_index < len(explainer_obj.data)):
        raise ValueError(f"Observation index {obs_index} is out of bounds.")

    # LIME needs predict_proba for regression, needs wrapper
    def predict_wrapper(data_np):
        data_pd = pd.DataFrame(data_np, columns=explainer_obj.data.columns)
        # Return as shape (n_samples,) for regression
        return explainer_obj.predict(data_pd)

    # Create LIME explainer inside the function
    lime_explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=explainer_obj.data.values, # LIME prefers numpy
        feature_names=explainer_obj.data.columns.tolist(),
        class_names=['prediction'], # Placeholder for regression
        mode='regression'
    )

    print(f"Explaining instance {obs_index} with LIME...")
    instance = explainer_obj.data.iloc[obs_index].values
    explanation = lime_explainer.explain_instance(
        data_row=instance,
        predict_fn=predict_wrapper,
        num_features=len(explainer_obj.data.columns) # Explain using all features
    )

    # Extract explanations and create Plotly bar chart
    exp_list = explanation.as_list()
    exp_df = pd.DataFrame(exp_list, columns=['feature', 'weight']).sort_values(by='weight', ascending=False)

    # Assign colors based on positive/negative weight
    colors = ['green' if w > 0 else 'red' for w in exp_df['weight']]

    fig = px.bar(
        exp_df,
        x='weight',
        y='feature',
        orientation='h',
        title=f"LIME Explanation (Obs: {obs_index})",
        labels={'weight': 'Feature Weight', 'feature': 'Feature'}
    )
    fig.update_traces(marker_color=colors)
    fig.update_layout(yaxis={'categoryorder':'total ascending'}) # Show features sorted by weight
    return fig

# --- Run the app ---
if __name__ == '__main__':
    if error_message:
        print("############################################")
        print(f"ERROR during initialization: {error_message}")
        print("App will start, but plots may be unavailable.")
        print("############################################")

    print("Starting Dash server...")
    app.run_server(debug=True) 