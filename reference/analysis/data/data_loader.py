"""
Data loader module for the Preem analysis project.
This module handles loading and preprocessing of the dataset.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict
from scipy import stats

class PreemDataLoader:
    def __init__(self, file_path: str):
        """
        Initialize the data loader.
        
        Args:
            file_path (str): Path to the preem.csv file
        """
        self.file_path = file_path
        self.data = None
        self.X = None
        self.y = None
        self.feature_names = None
        self.feature_stats = None
        
    def load_data(self) -> pd.DataFrame:
        """
        Load the data from CSV file and perform initial preprocessing.
        
        Returns:
            pd.DataFrame: Processed DataFrame
        """
        # Load the data
        self.data = pd.read_csv(self.file_path)
        
        # Convert date column to datetime
        self.data['date'] = pd.to_datetime(self.data['date'])
        
        # Sort by date to ensure temporal order
        self.data = self.data.sort_values('date')
        
        # Set date as index
        self.data.set_index('date', inplace=True)
        
        # Calculate and store basic statistics
        self._calculate_feature_statistics()
        
        return self.data
    
    def _calculate_feature_statistics(self):
        """
        Calculate comprehensive statistics for each feature.
        """
        stats_dict = {}
        
        for col in self.data.columns:
            col_stats = {
                'mean': self.data[col].mean(),
                'std': self.data[col].std(),
                'min': self.data[col].min(),
                'max': self.data[col].max(),
                'skew': stats.skew(self.data[col].dropna()),
                'kurtosis': stats.kurtosis(self.data[col].dropna()),
                'missing_pct': (self.data[col].isnull().sum() / len(self.data)) * 100,
                'unique_values': len(self.data[col].unique()),
                'autocorr_lag1': self.data[col].autocorr(lag=1),
                'autocorr_lag5': self.data[col].autocorr(lag=5),
                'stationarity': self._check_stationarity(self.data[col])
            }
            stats_dict[col] = col_stats
            
        self.feature_stats = stats_dict
    
    def _check_stationarity(self, series: pd.Series) -> Dict:
        """
        Check stationarity using Augmented Dickey-Fuller test.
        
        Args:
            series (pd.Series): Time series to test
            
        Returns:
            Dict: Stationarity test results
        """
        from statsmodels.tsa.stattools import adfuller
        result = adfuller(series.dropna())
        return {
            'adf_statistic': result[0],
            'p_value': result[1],
            'is_stationary': result[1] < 0.05
        }
    
    def prepare_features(self, target_col: str = 'target', 
                        handle_outliers: bool = True,
                        handle_missing: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features and target variables for modeling.
        No scaling is applied to preserve interpretability.
        
        Args:
            target_col (str): Name of the target column
            handle_outliers (bool): Whether to handle outliers
            handle_missing (bool): Whether to handle missing values
            
        Returns:
            tuple: (X, y) where X is the feature matrix and y is the target vector
        """
        # Separate features and target
        self.y = self.data[target_col]
        self.X = self.data.drop(columns=[target_col])
        
        if handle_missing:
            self._handle_missing_values()
            
        if handle_outliers:
            self._handle_outliers()
        
        # Store feature names
        self.feature_names = self.X.columns.tolist()
        
        return self.X, self.y
    
    def _handle_missing_values(self):
        """
        Handle missing values using appropriate methods for time series data.
        """
        # Forward fill first (use previous value)
        self.X = self.X.fillna(method='ffill')
        
        # Then backward fill any remaining NAs at the start
        self.X = self.X.fillna(method='bfill')
        
        # If any NAs remain, fill with median
        self.X = self.X.fillna(self.X.median())
    
    def _handle_outliers(self, threshold: float = 3.0):
        """
        Handle outliers using rolling statistics for time series data.
        
        Args:
            threshold (float): Number of standard deviations to use as threshold
        """
        for column in self.X.columns:
            # Calculate rolling mean and std
            rolling_mean = self.X[column].rolling(window=5, center=True).mean()
            rolling_std = self.X[column].rolling(window=5, center=True).std()
            
            # Identify outliers
            lower_bound = rolling_mean - (threshold * rolling_std)
            upper_bound = rolling_mean + (threshold * rolling_std)
            
            # Replace outliers with rolling mean
            outlier_mask = (self.X[column] < lower_bound) | (self.X[column] > upper_bound)
            self.X.loc[outlier_mask, column] = rolling_mean[outlier_mask]
    
    def get_train_test_split(self, test_months: int = 6, 
                            validation_months: Optional[int] = 3) -> Tuple[pd.DataFrame, ...]:
        """
        Split the data into training, validation, and testing sets based on time.
        
        Args:
            test_months (int): Number of months to use for testing
            validation_months (Optional[int]): Number of months for validation
            
        Returns:
            tuple: (X_train, X_val, X_test, y_train, y_val, y_test) if validation_months is specified
                  (X_train, X_test, y_train, y_test) otherwise
        """
        if self.X is None or self.y is None:
            raise ValueError("Must call prepare_features before splitting the data")
        
        # Calculate cutoff dates
        test_cutoff = self.X.index[-1] - pd.DateOffset(months=test_months)
        
        if validation_months:
            val_cutoff = test_cutoff - pd.DateOffset(months=validation_months)
            
            # Split the data
            X_train = self.X[self.X.index <= val_cutoff]
            X_val = self.X[(self.X.index > val_cutoff) & (self.X.index <= test_cutoff)]
            X_test = self.X[self.X.index > test_cutoff]
            
            y_train = self.y[self.y.index <= val_cutoff]
            y_val = self.y[(self.y.index > val_cutoff) & (self.y.index <= test_cutoff)]
            y_test = self.y[self.y.index > test_cutoff]
            
            print(f"\nTrain period: {X_train.index[0]} to {X_train.index[-1]}")
            print(f"Validation period: {X_val.index[0]} to {X_val.index[-1]}")
            print(f"Test period: {X_test.index[0]} to {X_test.index[-1]}")
            print(f"Train samples: {len(X_train)}, Val samples: {len(X_val)}, Test samples: {len(X_test)}")
            
            return X_train, X_val, X_test, y_train, y_val, y_test
        else:
            # Split without validation set
            X_train = self.X[self.X.index <= test_cutoff]
            X_test = self.X[self.X.index > test_cutoff]
            y_train = self.y[self.y.index <= test_cutoff]
            y_test = self.y[self.y.index > test_cutoff]
            
            print(f"\nTrain period: {X_train.index[0]} to {X_train.index[-1]}")
            print(f"Test period: {X_test.index[0]} to {X_test.index[-1]}")
            print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
            
            return X_train, X_test, y_train, y_test
    
    def get_feature_names(self) -> list:
        """
        Get the list of feature names.
        
        Returns:
            list: List of feature names
        """
        return self.feature_names
    
    def get_feature_statistics(self) -> Dict:
        """
        Get comprehensive feature statistics.
        
        Returns:
            Dict: Dictionary containing feature statistics
        """
        if self.feature_stats is None:
            self._calculate_feature_statistics()
        return self.feature_stats 