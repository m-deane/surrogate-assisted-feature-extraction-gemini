# SAFE Analysis Summary Report
Generated on: 2025-04-24 12:50:56 

## Executive Summary
This report summarizes findings from the Surrogate-Assisted Feature Extraction (SAFE) workflow.
It highlights key features, interactions, and decision boundaries identified by analyzing a
surrogate model trained on the data.

## 1. Dataset Overview
- **Dataset Size**: 57 samples, 9 columns (including target/date)
- **Features Analyzed**: 8 numeric features
- **Date Range**: 2020-04-01 00:00:00 to 2024-12-01 00:00:00


## 5. Model Performance
### Surrogate Model (random_forest) Performance
- **Training RMSE**: 24.5702
- **Training R²**: 0.4856
- **Test RMSE**: 36.0617
- **Test R²**: -1.9990

*Interpretation Guidance:*
* *R²*: Proportion of variance explained (closer to 1 is better). Negative R² indicates the model performs worse than a horizontal line.
* *RMSE*: Root Mean Squared Error (lower is better), in the units of the target variable.
* *Train vs. Test Gap*: A large difference often indicates overfitting.


## 3b. SHAP Interaction Analysis
Top pairwise feature interactions based on Mean Absolute SHAP Interaction values.

- `mean_med_diesel_crack_input1_trade_month_lag2` <> `mean_nwe_hsfo_crack_trade_month_lag1`: 0.0000
- `mean_med_diesel_crack_input1_trade_month_lag2` <> `mean_nwe_lsfo_crack_trade_month`: 0.0000
- `mean_sing_gasoline_vs_vlsfo_trade_month` <> `mean_nwe_lsfo_crack_trade_month`: 0.0000
- `mean_sing_gasoline_vs_vlsfo_trade_month` <> `mean_nwe_ulsfo_crack_trade_month lag3`: 0.0000
- `mean_sing_gasoline_vs_vlsfo_trade_month` <> `mean_sing_vlsfo_crack_trade_month_lag3`: 0.0000
- `mean_sing_gasoline_vs_vlsfo_trade_month` <> `new_sweet_sr_margin`: 0.0000
- `mean_sing_gasoline_vs_vlsfo_trade_month` <> `totaltar`: 0.0000
- `mean_sing_vlsfo_crack_trade_month_lag3` <> `mean_med_diesel_crack_input1_trade_month_lag2`: 0.0000
- `mean_sing_vlsfo_crack_trade_month_lag3` <> `mean_nwe_hsfo_crack_trade_month_lag1`: 0.0000
- `mean_sing_vlsfo_crack_trade_month_lag3` <> `mean_nwe_lsfo_crack_trade_month`: 0.0000


## 3. Feature Interactions
### Significant Pairwise Interactions (Approx. H-statistic):
- `mean_nwe_hsfo_crack_trade_month_lag1` × `mean_med_diesel_crack_input1_trade_month_lag2` (H ≈ 0.062)
- `mean_sing_vlsfo_crack_trade_month_lag3` × `mean_med_diesel_crack_input1_trade_month_lag2` (H ≈ 0.030)
- `mean_med_diesel_crack_input1_trade_month_lag2` × `new_sweet_sr_margin` (H ≈ 0.022)
- `new_sweet_sr_margin` × `mean_sing_gasoline_vs_vlsfo_trade_month` (H ≈ 0.017)
- `mean_sing_gasoline_vs_vlsfo_trade_month` × `mean_sing_vlsfo_crack_trade_month_lag3` (H ≈ 0.012)

### Three-way Interactions (Approx. H-statistic for Top Feature Triplets):
- `mean_med_diesel_crack_input1_trade_month_lag2 × mean_nwe_hsfo_crack_trade_month_lag1 × mean_sing_gasoline_vs_vlsfo_trade_month` (H ≈ 0.0502)
- `new_sweet_sr_margin × mean_med_diesel_crack_input1_trade_month_lag2 × mean_nwe_hsfo_crack_trade_month_lag1` (H ≈ 0.0452)
- `new_sweet_sr_margin × mean_med_diesel_crack_input1_trade_month_lag2 × mean_sing_gasoline_vs_vlsfo_trade_month` (H ≈ 0.0305)
- `new_sweet_sr_margin × mean_nwe_hsfo_crack_trade_month_lag1 × mean_sing_gasoline_vs_vlsfo_trade_month` (H ≈ 0.0175)


## 2. Key Features and Their Importance
### Top Features by SHAP Analysis (Train Set):
- `new_sweet_sr_margin` (Mean Abs SHAP: 6.149)
- `mean_med_diesel_crack_input1_trade_month_lag2` (Mean Abs SHAP: 4.717)
- `mean_nwe_hsfo_crack_trade_month_lag1` (Mean Abs SHAP: 2.801)
- `mean_sing_gasoline_vs_vlsfo_trade_month` (Mean Abs SHAP: 2.469)
- `mean_sing_vlsfo_crack_trade_month_lag3` (Mean Abs SHAP: 1.833)

### Top Features by Combined Importance Rank (Train Set):
- `new_sweet_sr_margin` (Mean Rank: 1.00)
- `mean_med_diesel_crack_input1_trade_month_lag2` (Mean Rank: 2.33)
- `mean_nwe_hsfo_crack_trade_month_lag1` (Mean Rank: 2.67)
- `mean_sing_gasoline_vs_vlsfo_trade_month` (Mean Rank: 4.00)
- `mean_sing_vlsfo_crack_trade_month_lag3` (Mean Rank: 5.00)


## 8a. Stationarity Tests (ADF)
ADF Test performed on training features to check for stationarity (required for Granger causality). A p-value <= 0.05 suggests stationarity.

- **`mean_med_diesel_crack_input1_trade_month_lag2`**: ADF Stat=-3.099, p-value=0.027 -> Stationary
- **`mean_nwe_hsfo_crack_trade_month_lag1`**: ADF Stat=-5.086, p-value=0.000 -> Stationary
- **`mean_nwe_lsfo_crack_trade_month`**: ADF Stat=-4.679, p-value=0.000 -> Stationary
- **`mean_nwe_ulsfo_crack_trade_month lag3`**: ADF Stat=-2.385, p-value=0.146 -> Non-Stationary
- **`mean_sing_gasoline_vs_vlsfo_trade_month`**: ADF Stat=-2.271, p-value=0.182 -> Non-Stationary
- **`mean_sing_vlsfo_crack_trade_month_lag3`**: ADF Stat=-2.895, p-value=0.046 -> Stationary
- **`new_sweet_sr_margin`**: ADF Stat=-6.161, p-value=0.000 -> Stationary
- **`totaltar`**: ADF Stat=-4.497, p-value=0.000 -> Stationary

Summary: 6 out of 8 features appear stationary (p<=0.05).
Non-stationary features may violate Granger assumptions.


## 7. Limitations
- Dataset size might limit generalizability.
- Training-test performance gap may indicate potential overfitting or concept drift.
- H-statistic values are approximations.
- Condition importance is a proxy derived from a secondary model.
- Temporal dynamics beyond simple train/test split are not explicitly modeled in this script (e.g., complex seasonality, autocorrelation in residuals).

