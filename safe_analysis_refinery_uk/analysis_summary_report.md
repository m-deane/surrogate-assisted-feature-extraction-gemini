# SAFE Analysis Summary Report
Generated on: 2025-04-24 20:31:44 

## Executive Summary
This report summarizes findings from the Surrogate-Assisted Feature Extraction (SAFE) workflow.
It highlights key features, interactions, and decision boundaries identified by analyzing a
surrogate model trained on the data.

## 1. Dataset Overview
- **Dataset Size**: 192 samples, 18 columns (including target/date)
- **Features Analyzed**: 17 numeric features
- **Date Range**: 2006-01-01 00:00:00 to 2021-12-01 00:00:00


## 5. Model Performance
### Surrogate Model (random_forest) Performance
- **Training RMSE**: 81.6184
- **Training R²**: 0.8684
- **Test RMSE**: 239.8323
- **Test R²**: -3.4646

*Interpretation Guidance:*
* *R²*: Proportion of variance explained (closer to 1 is better). Negative R² indicates the model performs worse than a horizontal line.
* *RMSE*: Root Mean Squared Error (lower is better), in the units of the target variable.
* *Train vs. Test Gap*: A large difference often indicates overfitting.


## 3b. SHAP Interaction Analysis
Top pairwise feature interactions based on Mean Absolute SHAP Interaction values.

- `brent` <> `dubai`: 0.0000
- `dubai_cracking_singapore` <> `brent_cracking_nw_europe`: 0.0000
- `dubai_cracking_singapore` <> `urals_cracking_med`: 0.0000
- `dubai_cracking_singapore` <> `es_sider_hydroskimming_med`: 0.0000
- `dubai_cracking_singapore` <> `es_sider_cracking_med`: 0.0000
- `dubai_cracking_singapore` <> `urals_hydroskimming_nw_europe`: 0.0000
- `dubai_cracking_singapore` <> `urals_cracking_nw_europe`: 0.0000
- `dubai_cracking_singapore` <> `brent_hydroskimming_nw_europe`: 0.0000
- `dubai_cracking_singapore` <> `wti`: 0.0000
- `brent` <> `wti`: 0.0000


## 8c. Causality Analysis (Granger Condition Feature -> Target)
Pairwise Granger causality tests performed for each *condition-derived* feature predicting the target variable, up to lag 3. Lower p-values suggest a condition feature Granger-causes the target (helps predict its future values).\n\nSee the bar chart plot (`causality_granger_condition_bar.png/.html`) for details.\nNote: Granger causality checks for predictive power, not necessarily true causation, and assumes stationarity.\n

## 6b.3 LOFO Condition Importance
LOFO importance calculated for condition-derived features using the secondary model.\nHigher positive scores indicate greater importance.\n\nSee the bar chart plot (`featimp_lofo_condition.png`) for details.\n**Top 5 Condition Features by LOFO Importance:**\n- `cond_urals_cracking_med_le_6p62`: 0.0407 (`('urals_cracking_med', '<=6.620')`)\n- `cond_x50_50_hls_lls_cracking_usgc_le_0p24`: 0.0100 (`('x50_50_hls_lls_cracking_usgc', '<=0.245')`)\n- `cond_x50_50_hls_lls_cracking_usgc_le_0p20`: -0.1235 (`('x50_50_hls_lls_cracking_usgc', '<=0.195')`)\n- `cond_x50_50_hls_lls_cracking_usgc_gt_0p24`: -0.1486 (`('x50_50_hls_lls_cracking_usgc', '> 0.245')`)\n- `cond_brent_cracking_nw_europe_gt_4p24`: -0.2733 (`('brent_cracking_nw_europe', '> 4.235')`)\n

## 3. Feature Interactions
### Significant Pairwise Interactions (Approx. H-statistic):
- `urals_cracking_med` × `x50_50_hls_lls_cracking_usgc` (H ≈ 0.075)
- `wti` × `dubai_cracking_singapore` (H ≈ 0.056)
- `wti` × `urals_hydroskimming_med` (H ≈ 0.040)
- `tapis_hydroskimming_singapore` × `wti` (H ≈ 0.030)
- `x50_50_hls_lls_cracking_usgc` × `urals_hydroskimming_med` (H ≈ 0.025)

### Three-way Interactions (Approx. H-statistic for Top Feature Triplets):
- `x50_50_hls_lls_cracking_usgc × urals_cracking_med × urals_hydroskimming_med` (H ≈ 0.0907)
- `x50_50_hls_lls_cracking_usgc × urals_cracking_med × wti` (H ≈ 0.0725)
- `x50_50_hls_lls_cracking_usgc × wti × urals_hydroskimming_med` (H ≈ 0.0583)
- `urals_cracking_med × wti × urals_hydroskimming_med` (H ≈ 0.0221)


## 2. Key Features and Their Importance
### Top Features by SHAP Analysis (Train Set):
- `x50_50_hls_lls_cracking_usgc` (Mean Abs SHAP: 84.376)
- `urals_cracking_med` (Mean Abs SHAP: 53.779)
- `wti` (Mean Abs SHAP: 50.429)
- `urals_hydroskimming_med` (Mean Abs SHAP: 20.566)
- `urals_cracking_nw_europe` (Mean Abs SHAP: 9.615)

### Top Features by Combined Importance Rank (Train Set):
- `x50_50_hls_lls_cracking_usgc` (Mean Rank: 1.00)
- `urals_cracking_med` (Mean Rank: 2.00)
- `wti` (Mean Rank: 3.00)
- `urals_hydroskimming_med` (Mean Rank: 4.00)
- `dubai_cracking_singapore` (Mean Rank: 5.33)


## 8a. Stationarity Tests (ADF)
ADF Test performed on training features to check for stationarity (required for Granger causality). A p-value <= 0.05 suggests stationarity.

- **`brent`**: ADF Stat=-2.704, p-value=0.073 -> Non-Stationary
- **`dubai`**: ADF Stat=-2.111, p-value=0.240 -> Non-Stationary
- **`wti`**: ADF Stat=-2.900, p-value=0.045 -> Stationary
- **`brent_cracking_nw_europe`**: ADF Stat=-3.487, p-value=0.008 -> Stationary
- **`brent_hydroskimming_nw_europe`**: ADF Stat=-3.880, p-value=0.002 -> Stationary
- **`urals_cracking_nw_europe`**: ADF Stat=-2.314, p-value=0.167 -> Non-Stationary
- **`urals_hydroskimming_nw_europe`**: ADF Stat=-3.524, p-value=0.007 -> Stationary
- **`es_sider_cracking_med`**: ADF Stat=-3.651, p-value=0.005 -> Stationary
- **`es_sider_hydroskimming_med`**: ADF Stat=-3.979, p-value=0.002 -> Stationary
- **`urals_cracking_med`**: ADF Stat=-2.184, p-value=0.212 -> Non-Stationary
- **`urals_hydroskimming_med`**: ADF Stat=-3.127, p-value=0.025 -> Stationary
- **`dubai_cracking_singapore`**: ADF Stat=-4.611, p-value=0.000 -> Stationary
- **`dubai_hydroskimming_singapore`**: ADF Stat=-5.445, p-value=0.000 -> Stationary
- **`tapis_hydroskimming_singapore`**: ADF Stat=-2.534, p-value=0.107 -> Non-Stationary
- **`x50_50_hls_lls_cracking_usgc`**: ADF Stat=-2.564, p-value=0.101 -> Non-Stationary
- **`x30_70_wcs_bakken_cracking_usmc`**: ADF Stat=-4.908, p-value=0.000 -> Stationary
- **`bakken_coking_usmc`**: ADF Stat=-2.335, p-value=0.161 -> Non-Stationary

Summary: 10 out of 17 features appear stationary (p<=0.05).
Non-stationary features may violate Granger assumptions.


## 7. Limitations
- Dataset size might limit generalizability.
- Training-test performance gap may indicate potential overfitting or concept drift.
- H-statistic values are approximations.
- Condition importance is a proxy derived from a secondary model.
- Temporal dynamics beyond simple train/test split are not explicitly modeled in this script (e.g., complex seasonality, autocorrelation in residuals).

