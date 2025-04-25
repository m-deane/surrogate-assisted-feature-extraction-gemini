import pandas as pd
from safe_analysis import ReportGenerator, DATA_PATH, TARGET_VARIABLE, DATE_COLUMN, GROUPING_COLUMN, FILTER_VALUE, TEST_MONTHS

def main():
    # Load and prepare data
    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    
    # Convert date column to datetime
    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN])
    
    # Filter for United Kingdom
    if GROUPING_COLUMN and FILTER_VALUE:
        df = df[df[GROUPING_COLUMN] == FILTER_VALUE]
        print(f"Filtered data for {GROUPING_COLUMN}={FILTER_VALUE}")
    
    # Sort by date
    df = df.sort_values(DATE_COLUMN)
    
    # Split into train and test sets based on time
    split_date = df[DATE_COLUMN].max() - pd.DateOffset(months=TEST_MONTHS)
    train_df = df[df[DATE_COLUMN] <= split_date]
    test_df = df[df[DATE_COLUMN] > split_date]
    
    print(f"Train set size: {len(train_df)}")
    print(f"Test set size: {len(test_df)}")
    
    # Initialize report generator
    report_gen = ReportGenerator(
        output_dir='safe_analysis',
        surrogate_model_type='random_forest'
    )
    
    # Set data
    feature_cols = [col for col in df.columns if col not in [TARGET_VARIABLE, DATE_COLUMN, GROUPING_COLUMN]]
    report_gen.feature_names = feature_cols
    report_gen.target_name = TARGET_VARIABLE
    
    # Prepare train and test data
    report_gen.X_train = train_df[feature_cols]
    report_gen.y_train = train_df[TARGET_VARIABLE]
    report_gen.X_test = test_df[feature_cols]
    report_gen.y_test = test_df[TARGET_VARIABLE]
    
    # Add dataset overview
    report_gen.add_dataset_overview(
        df_shape=df.shape,
        features_list=feature_cols,
        date_range=(df[DATE_COLUMN].min(), df[DATE_COLUMN].max())
    )
    
    # Analyze condition features
    report_gen.analyze_condition_features()
    
    # Generate and save report
    report_gen.generate_report("safe_analysis_report.md")
    print("Analysis complete. Check safe_analysis_report.md for results.")

if __name__ == "__main__":
    main() 