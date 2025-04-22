# Dynamic Analysis Summary Report

## 1. Dataset Overview
- **Dataset Size**: 57 samples, 10 features
- **Feature Types**: 9 numeric features
- **Date Range**: 2020-04-01 00:00:00 to 2024-12-01 00:00:00

## 2. Key Features and Their Importance

### Top Features by SHAP Analysis:
- `mean_nwe_hsfo_crack_trade_month_lag1` (SHAP value: 4.944)
- `mean_med_diesel_crack_input1_trade_month_lag2` (SHAP value: 4.873)
- `mean_sing_gasoline_vs_vlsfo_trade_month` (SHAP value: 4.467)
- `new_sweet_sr_margin` (SHAP value: 3.428)
- `mean_sing_vlsfo_crack_trade_month_lag3` (SHAP value: 2.801)

### Top Features by Combined Importance Metrics:
- `mean_med_diesel_crack_input1_trade_month_lag2`:
  - MDI: 0.098
  - Permutation: -0.170
  - LOFO: 0.694
- `mean_sing_gasoline_vs_vlsfo_trade_month`:
  - MDI: 0.277
  - Permutation: -0.067
  - LOFO: 0.657
- `mean_nwe_hsfo_crack_trade_month_lag1`:
  - MDI: 0.144
  - Permutation: -0.096
  - LOFO: 0.454
- `mean_nwe_ulsfo_crack_trade_month lag3`:
  - MDI: 0.084
  - Permutation: 0.081
  - LOFO: 0.373
- `mean_nwe_lsfo_crack_trade_month`:
  - MDI: 0.032
  - Permutation: 0.054
  - LOFO: 0.133

## 3. Feature Interactions

### Significant Pairwise Interactions (H-statistic > 0.1):
- `mean_med_diesel_crack_input1_trade_month_lag2` × `mean_nwe_hsfo_crack_trade_month_lag1` (H = 1.402)
- `mean_med_diesel_crack_input1_trade_month_lag2` × `mean_nwe_lsfo_crack_trade_month` (H = 0.169)
- `mean_med_diesel_crack_input1_trade_month_lag2` × `mean_sing_gasoline_vs_vlsfo_trade_month` (H = 0.297)
- `mean_med_diesel_crack_input1_trade_month_lag2` × `mean_sing_vlsfo_crack_trade_month_lag3` (H = 0.195)
- `mean_med_diesel_crack_input1_trade_month_lag2` × `new_sweet_sr_margin` (H = 1.192)
- `mean_nwe_hsfo_crack_trade_month_lag1` × `mean_nwe_lsfo_crack_trade_month` (H = 0.611)
- `mean_nwe_hsfo_crack_trade_month_lag1` × `mean_nwe_ulsfo_crack_trade_month lag3` (H = 0.183)
- `mean_nwe_hsfo_crack_trade_month_lag1` × `mean_sing_gasoline_vs_vlsfo_trade_month` (H = 0.513)
- `mean_nwe_hsfo_crack_trade_month_lag1` × `new_sweet_sr_margin` (H = 0.347)
- `mean_nwe_lsfo_crack_trade_month` × `mean_nwe_ulsfo_crack_trade_month lag3` (H = 0.119)
- `mean_nwe_lsfo_crack_trade_month` × `new_sweet_sr_margin` (H = 0.325)
- `mean_nwe_ulsfo_crack_trade_month lag3` × `mean_sing_gasoline_vs_vlsfo_trade_month` (H = 0.237)
- `mean_nwe_ulsfo_crack_trade_month lag3` × `mean_sing_vlsfo_crack_trade_month_lag3` (H = 0.581)
- `mean_nwe_ulsfo_crack_trade_month lag3` × `new_sweet_sr_margin` (H = 0.550)
- `mean_sing_gasoline_vs_vlsfo_trade_month` × `mean_sing_vlsfo_crack_trade_month_lag3` (H = 0.533)
- `mean_sing_gasoline_vs_vlsfo_trade_month` × `new_sweet_sr_margin` (H = 0.365)
- `mean_sing_vlsfo_crack_trade_month_lag3` × `new_sweet_sr_margin` (H = 0.291)
- `new_sweet_sr_margin` × `totaltar` (H = 0.224)

### Top Three-way Interactions:
- mean_med_diesel_crack_input1_trade_month_lag2 × mean_nwe_hsfo_crack_trade_month_lag1 × mean_nwe_lsfo_crack_trade_month (H = 0.252)
- mean_med_diesel_crack_input1_trade_month_lag2 × mean_nwe_hsfo_crack_trade_month_lag1 × mean_nwe_ulsfo_crack_trade_month lag3 (H = 0.224)
- mean_med_diesel_crack_input1_trade_month_lag2 × mean_sing_gasoline_vs_vlsfo_trade_month × mean_nwe_hsfo_crack_trade_month_lag1 (H = 0.109)

## 4. Critical Thresholds

### Top Impactful Thresholds:
- `mean_sing_gasoline_vs_vlsfo_trade_month` at 4.57:
  - Impact: 45.54
  - Affects 6.7% of the data
- `new_sweet_sr_margin` at -6.44:
  - Impact: 39.48
  - Affects 97.8% of the data
- `new_sweet_sr_margin` at -6.36:
  - Impact: 34.70
  - Affects 97.8% of the data
- `new_sweet_sr_margin` at -5.35:
  - Impact: 33.85
  - Affects 95.6% of the data
- `new_sweet_sr_margin` at -6.20:
  - Impact: 33.09
  - Affects 97.8% of the data

## 5. Model Performance

### Decision Tree Results:
- Training R²: 0.8031
- Test R²: -5.8168

### Top Decision Rules:
- Rule 1:
  - Covers 36 samples (80.0% of data)
  - Prediction accuracy: 100.0%
- Rule 2:
  - Covers 5 samples (11.1% of data)
  - Prediction accuracy: 100.0%
- Rule 3:
  - Covers 1 samples (2.2% of data)
  - Prediction accuracy: 100.0%

## 6. Key Insights and Recommendations

### Feature Relationships:
- Strong interactions detected between crack spread features
- Multiple threshold effects suggest non-linear relationships

### Model Implications:
- Evidence of overfitting suggests need for regularization
- Consider using simpler models or gathering more data

### Recommendations:
1. Focus monitoring on identified threshold levels
2. Consider interaction effects in feature engineering
3. Monitor strongest feature pairs for regime changes
## 7. Limitations
- Dataset size (57 samples) may limit generalizability
- Training-test performance gap indicates potential overfitting
- Market conditions and relationships may change over time
