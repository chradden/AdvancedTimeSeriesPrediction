"""
Data preprocessing utilities for time series
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple, List
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class TimeSeriesPreprocessor:
    """Preprocessing utilities for time series data"""
    
    def __init__(self):
        self.scaler = None
        self.feature_cols = None
        
    def handle_missing_values(
        self,
        df: pd.DataFrame,
        value_col: str = 'value',
        method: str = 'interpolate',
        **kwargs
    ) -> pd.DataFrame:
        """
        Handle missing values in time series
        
        Args:
            df: DataFrame with time series
            value_col: Name of the value column
            method: Method to use ('interpolate', 'ffill', 'bfill', 'mean', 'drop')
            **kwargs: Additional arguments for the method
            
        Returns:
            DataFrame with handled missing values
        """
        df = df.copy()
        
        if method == 'interpolate':
            df[value_col] = df[value_col].interpolate(method='linear', **kwargs)
        elif method == 'ffill':
            df[value_col] = df[value_col].fillna(method='ffill', **kwargs)
        elif method == 'bfill':
            df[value_col] = df[value_col].fillna(method='bfill', **kwargs)
        elif method == 'mean':
            df[value_col] = df[value_col].fillna(df[value_col].mean())
        elif method == 'drop':
            df = df.dropna(subset=[value_col])
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return df
    
    def handle_outliers(
        self,
        df: pd.DataFrame,
        value_col: str = 'value',
        method: str = 'iqr',
        threshold: float = 1.5
    ) -> pd.DataFrame:
        """
        Handle outliers in time series
        
        Args:
            df: DataFrame with time series
            value_col: Name of the value column
            method: Method to use ('iqr', 'zscore', 'clip')
            threshold: Threshold for outlier detection
            
        Returns:
            DataFrame with handled outliers
        """
        df = df.copy()
        
        if method == 'iqr':
            Q1 = df[value_col].quantile(0.25)
            Q3 = df[value_col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - threshold * IQR
            upper = Q3 + threshold * IQR
            df[value_col] = df[value_col].clip(lower, upper)
            
        elif method == 'zscore':
            mean = df[value_col].mean()
            std = df[value_col].std()
            df[value_col] = df[value_col].clip(
                mean - threshold * std,
                mean + threshold * std
            )
            
        elif method == 'clip':
            df[value_col] = df[value_col].clip(
                df[value_col].quantile(0.01),
                df[value_col].quantile(0.99)
            )
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return df
    
    def create_time_features(
        self,
        df: pd.DataFrame,
        timestamp_col: str = 'timestamp'
    ) -> pd.DataFrame:
        """
        Create time-based features
        
        Args:
            df: DataFrame with timestamp column
            timestamp_col: Name of timestamp column
            
        Returns:
            DataFrame with added features
        """
        df = df.copy()
        
        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
            df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        
        # Extract time components
        df['year'] = df[timestamp_col].dt.year
        df['month'] = df[timestamp_col].dt.month
        df['day'] = df[timestamp_col].dt.day
        df['dayofweek'] = df[timestamp_col].dt.dayofweek
        df['dayofyear'] = df[timestamp_col].dt.dayofyear
        df['weekofyear'] = df[timestamp_col].dt.isocalendar().week
        df['quarter'] = df[timestamp_col].dt.quarter
        df['hour'] = df[timestamp_col].dt.hour
        
        # Cyclical encoding for time features
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
        df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        
        # Boolean features
        df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
        df['is_month_start'] = df[timestamp_col].dt.is_month_start.astype(int)
        df['is_month_end'] = df[timestamp_col].dt.is_month_end.astype(int)
        
        return df
    
    def create_lag_features(
        self,
        df: pd.DataFrame,
        value_col: str = 'value',
        lags: List[int] = [1, 2, 3, 24, 168]
    ) -> pd.DataFrame:
        """
        Create lag features
        
        Args:
            df: DataFrame
            value_col: Column to create lags from
            lags: List of lag periods
            
        Returns:
            DataFrame with lag features
        """
        df = df.copy()
        
        for lag in lags:
            df[f'lag_{lag}'] = df[value_col].shift(lag)
        
        return df
    
    def create_rolling_features(
        self,
        df: pd.DataFrame,
        value_col: str = 'value',
        windows: List[int] = [24, 168],
        functions: List[str] = ['mean', 'std', 'min', 'max']
    ) -> pd.DataFrame:
        """
        Create rolling window features
        
        Args:
            df: DataFrame
            value_col: Column to create features from
            windows: List of window sizes
            functions: List of aggregation functions
            
        Returns:
            DataFrame with rolling features
        """
        df = df.copy()
        
        for window in windows:
            for func in functions:
                df[f'rolling_{window}_{func}'] = df[value_col].rolling(
                    window=window, min_periods=1
                ).agg(func)
        
        return df
    
    def scale_data(
        self,
        train_df: pd.DataFrame,
        test_df: Optional[pd.DataFrame] = None,
        feature_cols: Optional[List[str]] = None,
        method: str = 'standard'
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Scale features
        
        Args:
            train_df: Training DataFrame
            test_df: Optional test DataFrame
            feature_cols: Columns to scale (None = all numeric)
            method: 'standard' or 'minmax'
            
        Returns:
            Scaled train and test DataFrames
        """
        if feature_cols is None:
            feature_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
        
        self.feature_cols = feature_cols
        
        # Initialize scaler
        if method == 'standard':
            self.scaler = StandardScaler()
        elif method == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaling method: {method}")
        
        # Fit and transform train
        train_scaled = train_df.copy()
        train_scaled[feature_cols] = self.scaler.fit_transform(train_df[feature_cols])
        
        # Transform test if provided
        if test_df is not None:
            test_scaled = test_df.copy()
            test_scaled[feature_cols] = self.scaler.transform(test_df[feature_cols])
            return train_scaled, test_scaled
        
        return train_scaled, None
    
    def inverse_scale(self, df: pd.DataFrame) -> pd.DataFrame:
        """Inverse scale the data"""
        if self.scaler is None:
            raise ValueError("Scaler not fitted yet")
        
        df_inv = df.copy()
        df_inv[self.feature_cols] = self.scaler.inverse_transform(df[self.feature_cols])
        return df_inv


def train_test_split_temporal(
    df: pd.DataFrame,
    test_size: float = 0.2,
    val_size: Optional[float] = None
) -> Tuple:
    """
    Split time series data chronologically
    
    Args:
        df: DataFrame to split
        test_size: Fraction for test set
        val_size: Optional fraction for validation set
        
    Returns:
        train, val, test DataFrames (val is None if val_size not specified)
    """
    n = len(df)
    
    if val_size is not None:
        train_end = int(n * (1 - test_size - val_size))
        val_end = int(n * (1 - test_size))
        
        train = df.iloc[:train_end].copy()
        val = df.iloc[train_end:val_end].copy()
        test = df.iloc[val_end:].copy()
        
        return train, val, test
    else:
        split_idx = int(n * (1 - test_size))
        train = df.iloc[:split_idx].copy()
        test = df.iloc[split_idx:].copy()
        
        return train, test


if __name__ == "__main__":
    # Example usage
    print("Preprocessing utilities loaded successfully")
    
    # Create example data
    dates = pd.date_range('2023-01-01', periods=1000, freq='H')
    values = np.sin(np.linspace(0, 10*np.pi, 1000)) * 100 + 500 + np.random.randn(1000) * 20
    
    df = pd.DataFrame({'timestamp': dates, 'value': values})
    
    # Add some missing values
    df.loc[10:15, 'value'] = np.nan
    
    # Test preprocessing
    prep = TimeSeriesPreprocessor()
    df_clean = prep.handle_missing_values(df)
    df_features = prep.create_time_features(df_clean)
    
    print(f"\nOriginal shape: {df.shape}")
    print(f"With features shape: {df_features.shape}")
    print(f"\nNew columns: {df_features.columns.tolist()}")
