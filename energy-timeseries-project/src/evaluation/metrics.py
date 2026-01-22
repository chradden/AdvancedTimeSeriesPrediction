"""
Evaluation metrics for time series forecasting

This module provides comprehensive evaluation metrics for comparing
different forecasting models.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, List
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class MetricsCalculator:
    """Calculate various forecasting metrics"""
    
    @staticmethod
    def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Absolute Error"""
        return mean_absolute_error(y_true, y_pred)
    
    @staticmethod
    def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Root Mean Squared Error"""
        return np.sqrt(mean_squared_error(y_true, y_pred))
    
    @staticmethod
    def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Mean Absolute Percentage Error
        
        Note: Returns infinity if any y_true value is zero.
        Use smape() for datasets with zeros.
        """
        mask = y_true != 0
        if not mask.any():
            return np.inf
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    
    @staticmethod
    def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Symmetric Mean Absolute Percentage Error
        
        Better for datasets with zeros or very small values.
        """
        numerator = np.abs(y_pred - y_true)
        denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
        
        # Avoid division by zero
        mask = denominator != 0
        if not mask.any():
            return 0.0
            
        return np.mean(numerator[mask] / denominator[mask]) * 100
    
    @staticmethod
    def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """R² Score (Coefficient of Determination)"""
        return r2_score(y_true, y_pred)
    
    @staticmethod
    def mase(y_true: np.ndarray, y_pred: np.ndarray, 
             y_train: Optional[np.ndarray] = None, seasonality: int = 1) -> float:
        """
        Mean Absolute Scaled Error
        
        Args:
            y_true: True values
            y_pred: Predicted values
            y_train: Training data (for baseline calculation)
            seasonality: Seasonal period (1 for non-seasonal, 24 for hourly with daily seasonality)
        """
        mae_model = np.mean(np.abs(y_true - y_pred))
        
        if y_train is not None:
            # Use training data for baseline
            baseline_errors = np.abs(np.diff(y_train, n=seasonality))
        else:
            # Use test data for baseline (not ideal but works)
            baseline_errors = np.abs(np.diff(y_true, n=seasonality))
        
        mae_baseline = np.mean(baseline_errors)
        
        if mae_baseline == 0:
            return np.inf
            
        return mae_model / mae_baseline


def calculate_metrics(
    y_true: np.ndarray, 
    y_pred: np.ndarray,
    y_train: Optional[np.ndarray] = None,
    seasonality: int = 1,
    prefix: str = ""
) -> Dict[str, float]:
    """
    Calculate all relevant metrics
    
    Args:
        y_true: True values
        y_pred: Predicted values
        y_train: Optional training data for MASE calculation
        seasonality: Seasonal period for MASE
        prefix: Prefix for metric names (e.g., 'train_', 'test_')
        
    Returns:
        Dictionary with all metrics
    """
    calc = MetricsCalculator()
    
    metrics = {
        f'{prefix}mae': calc.mae(y_true, y_pred),
        f'{prefix}rmse': calc.rmse(y_true, y_pred),
        f'{prefix}mape': calc.mape(y_true, y_pred),
        f'{prefix}smape': calc.smape(y_true, y_pred),
        f'{prefix}r2': calc.r2(y_true, y_pred),
        f'{prefix}mase': calc.mase(y_true, y_pred, y_train, seasonality)
    }
    
    return metrics


def print_metrics(metrics: Dict[str, float], title: str = "Metrics") -> None:
    """
    Print metrics in a nice format
    
    Args:
        metrics: Dictionary of metrics
        title: Title to display
    """
    print("=" * 60)
    print(f"{title:^60}")
    print("=" * 60)
    
    for name, value in metrics.items():
        if 'mape' in name.lower() or 'smape' in name.lower():
            print(f"  {name:15s}: {value:10.2f}%")
        elif 'r2' in name.lower():
            print(f"  {name:15s}: {value:10.4f}")
        else:
            print(f"  {name:15s}: {value:10.2f}")
    
    print("=" * 60)


def compare_models(
    results: Dict[str, Dict[str, float]],
    sort_by: str = 'rmse',
    ascending: bool = True
) -> pd.DataFrame:
    """
    Compare multiple models based on their metrics
    
    Args:
        results: Dictionary of {model_name: metrics_dict}
        sort_by: Metric to sort by
        ascending: Sort ascending (True) or descending (False)
        
    Returns:
        DataFrame with comparison
    """
    df = pd.DataFrame(results).T
    
    # Sort by specified metric
    if sort_by in df.columns:
        df = df.sort_values(sort_by, ascending=ascending)
    
    # Highlight best values
    def highlight_best(col):
        """Highlight the best value in each column"""
        if col.name in ['r2', 'r2_score']:
            # Higher is better for R²
            best_idx = col.idxmax()
        else:
            # Lower is better for error metrics
            best_idx = col.idxmin()
        
        return ['font-weight: bold' if idx == best_idx else '' 
                for idx in col.index]
    
    return df


def residual_analysis(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Analyze residuals for model diagnostics
    
    Returns:
        Dictionary with residual statistics
    """
    residuals = y_true - y_pred
    
    return {
        'residual_mean': np.mean(residuals),
        'residual_std': np.std(residuals),
        'residual_min': np.min(residuals),
        'residual_max': np.max(residuals),
        'residual_q25': np.percentile(residuals, 25),
        'residual_q50': np.percentile(residuals, 50),
        'residual_q75': np.percentile(residuals, 75)
    }


# Convenience functions for common use cases
def quick_eval(y_true: np.ndarray, y_pred: np.ndarray, 
               model_name: str = "Model") -> None:
    """Quick evaluation with print"""
    metrics = calculate_metrics(y_true, y_pred)
    print_metrics(metrics, title=f"{model_name} Performance")


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    y_true = np.random.randn(100) * 10 + 50
    y_pred = y_true + np.random.randn(100) * 2
    
    print("Example Metrics Calculation\n")
    metrics = calculate_metrics(y_true, y_pred)
    print_metrics(metrics, "Example Model")
    
    print("\nResidual Analysis:")
    residuals = residual_analysis(y_true, y_pred)
    for key, value in residuals.items():
        print(f"  {key}: {value:.3f}")
