# Detailed Analysis Interpretation

## Feature Importance Analysis

### Key Features

#### 1. mean_nwe_ulsfo_crack_trade_month lag3
- Relative Importance: 33.38%
- Cumulative Importance: 33.38%
- Distribution Statistics:
  * Median: 13.14
  * IQR: [10.72, 16.15]
- Decision Thresholds:
  * 8.22 (splits data 8.8% / 91.2%)

#### 2. new_sweet_sr_margin
- Relative Importance: 28.75%
- Cumulative Importance: 62.13%
- Distribution Statistics:
  * Median: -0.87
  * IQR: [-2.53, 0.65]
- Decision Thresholds:
  * -7.54 (splits data 1.8% / 98.2%)
  * -5.03 (splits data 5.3% / 94.7%)
  * -2.31 (splits data 28.1% / 71.9%)

#### 3. mean_sing_gasoline_vs_vlsfo_trade_month
- Relative Importance: 21.47%
- Cumulative Importance: 83.60%
- Distribution Statistics:
  * Median: -3.86
  * IQR: [-7.22, -0.48]
- Decision Thresholds:
  * -6.38 (splits data 31.6% / 68.4%)
  * -4.83 (splits data 40.4% / 59.6%)
  * 4.78 (splits data 96.5% / 3.5%)
  * 5.76 (splits data 98.2% / 1.8%)

#### 4. mean_nwe_hsfo_crack_trade_month_lag1
- Relative Importance: 7.42%
- Cumulative Importance: 91.02%
- Distribution Statistics:
  * Median: -9.81
  * IQR: [-12.84, -5.86]
- Decision Thresholds:
  * 13.42 (splits data 94.7% / 5.3%)

#### 5. totaltar
- Relative Importance: 6.18%
- Cumulative Importance: 97.20%
- Distribution Statistics:
  * Median: 0.00
  * IQR: [0.00, 4.60]
- Decision Thresholds:
  * 0.47 (splits data 71.9% / 28.1%)

#### 6. mean_nwe_lsfo_crack_trade_month
- Relative Importance: 2.10%
- Cumulative Importance: 99.30%
- Distribution Statistics:
  * Median: -3.44
  * IQR: [-7.35, 0.07]
- Decision Thresholds:
  * -6.81 (splits data 33.3% / 66.7%)

#### 7. mean_sing_vlsfo_crack_trade_month_lag3
- Relative Importance: 0.49%
- Cumulative Importance: 99.79%
- Distribution Statistics:
  * Median: 11.28
  * IQR: [9.29, 14.24]
- Decision Thresholds:
  * 13.99 (splits data 71.9% / 28.1%)

#### 8. mean_med_diesel_crack_input1_trade_month_lag2
- Relative Importance: 0.21%
- Cumulative Importance: 100.00%
- Distribution Statistics:
  * Median: -75.81
  * IQR: [-82.67, -53.08]
- Decision Thresholds:
  * -41.65 (splits data 82.5% / 17.5%)


## Feature Interactions

### Strong Interaction Patterns

#### mean_nwe_ulsfo_crack_trade_month lag3 × new_sweet_sr_margin
- Interaction Strength: 42.1023
- Correlation: 0.1481
- Interpretation: Low correlation suggests complementary information

#### new_sweet_sr_margin × totaltar
- Interaction Strength: 23.0494
- Correlation: -0.1923
- Interpretation: Low correlation suggests complementary information

#### mean_nwe_ulsfo_crack_trade_month lag3 × mean_sing_gasoline_vs_vlsfo_trade_month
- Interaction Strength: 22.7676
- Correlation: -0.1304
- Interpretation: Low correlation suggests complementary information

#### mean_sing_gasoline_vs_vlsfo_trade_month × new_sweet_sr_margin
- Interaction Strength: 20.6033
- Correlation: 0.0477
- Interpretation: Low correlation suggests complementary information

#### mean_nwe_hsfo_crack_trade_month_lag1 × new_sweet_sr_margin
- Interaction Strength: 19.8506
- Correlation: -0.1620
- Interpretation: Low correlation suggests complementary information

