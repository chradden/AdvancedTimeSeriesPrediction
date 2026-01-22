"""
Plotting utilities for time series forecasting

Provides reusable plotting functions for:
- Time series visualization
- Forecast comparisons
- Residual analysis
- Seasonal decomposition
- Prediction intervals
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, List, Tuple, Dict
from datetime import datetime


# Set default style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('husl')


def plot_time_series(
    timestamps: pd.Series,
    values: pd.Series,
    title: str = "Time Series",
    xlabel: str = "Time",
    ylabel: str = "Value",
    figsize: Tuple[int, int] = (15, 6),
    color: str = 'blue',
    alpha: float = 0.7,
    save_path: Optional[Path] = None
):
    """
    Plot a simple time series
    
    Args:
        timestamps: Time index
        values: Values to plot
        title: Plot title
        xlabel, ylabel: Axis labels
        figsize: Figure size
        color: Line color
        alpha: Line transparency
        save_path: Optional path to save figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(timestamps, values, color=color, alpha=alpha, linewidth=1)
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        save_figure(fig, save_path)
    
    plt.show()


def plot_forecast(
    timestamps_train: pd.Series,
    values_train: pd.Series,
    timestamps_test: pd.Series,
    values_test: pd.Series,
    predictions: np.ndarray,
    model_name: str = "Model",
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 6),
    save_path: Optional[Path] = None
):
    """
    Plot training data, actual test data, and predictions
    
    Args:
        timestamps_train: Training timestamps
        values_train: Training values
        timestamps_test: Test timestamps
        values_test: True test values
        predictions: Predicted values
        model_name: Name of the model
        title: Custom title (optional)
        figsize: Figure size
        save_path: Optional path to save figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot training data
    ax.plot(timestamps_train, values_train, 
            label='Training Data', color='blue', alpha=0.6, linewidth=1)
    
    # Plot actual test data
    ax.plot(timestamps_test, values_test,
            label='Actual', color='green', alpha=0.8, linewidth=1.5)
    
    # Plot predictions
    ax.plot(timestamps_test, predictions,
            label=f'{model_name} Forecast', color='red', 
            alpha=0.8, linewidth=1.5, linestyle='--')
    
    # Title
    if title is None:
        title = f'{model_name} - Forecast vs Actual'
    ax.set_title(title, fontsize=16, fontweight='bold')
    
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        save_figure(fig, save_path)
    
    plt.show()


def plot_multiple_forecasts(
    timestamps_test: pd.Series,
    values_test: pd.Series,
    forecasts: Dict[str, np.ndarray],
    title: str = "Model Comparison",
    figsize: Tuple[int, int] = (16, 7),
    show_actual: bool = True,
    save_path: Optional[Path] = None
):
    """
    Plot multiple model forecasts for comparison
    
    Args:
        timestamps_test: Test timestamps
        values_test: True test values
        forecasts: Dictionary of {model_name: predictions}
        title: Plot title
        figsize: Figure size
        show_actual: Whether to show actual values
        save_path: Optional path to save figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot actual values
    if show_actual:
        ax.plot(timestamps_test, values_test,
                label='Actual', color='black', alpha=0.8, 
                linewidth=2, zorder=10)
    
    # Plot each model's forecast
    colors = plt.cm.tab10.colors
    for i, (model_name, predictions) in enumerate(forecasts.items()):
        ax.plot(timestamps_test, predictions,
                label=model_name, color=colors[i % len(colors)],
                alpha=0.7, linewidth=1.5, linestyle='--')
    
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        save_figure(fig, save_path)
    
    plt.show()


def plot_residuals(
    residuals: np.ndarray,
    timestamps: Optional[pd.Series] = None,
    model_name: str = "Model",
    figsize: Tuple[int, int] = (16, 10),
    save_path: Optional[Path] = None
):
    """
    Comprehensive residual analysis plots
    
    Args:
        residuals: Residual values (y_true - y_pred)
        timestamps: Optional timestamps for time plot
        model_name: Name of the model
        figsize: Figure size
        save_path: Optional path to save figure
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(f'{model_name} - Residual Analysis', 
                 fontsize=16, fontweight='bold', y=1.00)
    
    # 1. Residuals over time (if timestamps provided)
    if timestamps is not None:
        axes[0, 0].plot(timestamps, residuals, alpha=0.6, linewidth=0.5)
        axes[0, 0].axhline(y=0, color='red', linestyle='--', linewidth=2)
        axes[0, 0].set_title('Residuals Over Time')
        axes[0, 0].set_xlabel('Time')
        axes[0, 0].set_ylabel('Residual')
        axes[0, 0].grid(True, alpha=0.3)
    else:
        axes[0, 0].plot(residuals, alpha=0.6, linewidth=0.5)
        axes[0, 0].axhline(y=0, color='red', linestyle='--', linewidth=2)
        axes[0, 0].set_title('Residuals')
        axes[0, 0].set_xlabel('Index')
        axes[0, 0].set_ylabel('Residual')
        axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Histogram of residuals
    axes[0, 1].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
    axes[0, 1].axvline(x=0, color='red', linestyle='--', linewidth=2)
    axes[0, 1].set_title('Residual Distribution')
    axes[0, 1].set_xlabel('Residual')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # 3. Q-Q plot
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title('Q-Q Plot')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. ACF of residuals
    from statsmodels.graphics.tsaplots import plot_acf
    plot_acf(residuals, lags=40, ax=axes[1, 1])
    axes[1, 1].set_title('Residual Autocorrelation')
    
    plt.tight_layout()
    
    if save_path:
        save_figure(fig, save_path)
    
    plt.show()


def plot_prediction_intervals(
    timestamps_test: pd.Series,
    values_test: pd.Series,
    predictions: np.ndarray,
    lower_bound: np.ndarray,
    upper_bound: np.ndarray,
    model_name: str = "Model",
    confidence: float = 0.95,
    figsize: Tuple[int, int] = (16, 6),
    save_path: Optional[Path] = None
):
    """
    Plot predictions with confidence/prediction intervals
    
    Args:
        timestamps_test: Test timestamps
        values_test: True test values
        predictions: Point predictions
        lower_bound: Lower confidence bound
        upper_bound: Upper confidence bound
        model_name: Name of the model
        confidence: Confidence level (e.g., 0.95 for 95%)
        figsize: Figure size
        save_path: Optional path to save figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot actual values
    ax.plot(timestamps_test, values_test,
            label='Actual', color='black', alpha=0.8, linewidth=2)
    
    # Plot predictions
    ax.plot(timestamps_test, predictions,
            label=f'{model_name} Forecast', color='red', 
            alpha=0.8, linewidth=1.5, linestyle='--')
    
    # Plot confidence interval
    ax.fill_between(timestamps_test, lower_bound, upper_bound,
                     alpha=0.2, color='red',
                     label=f'{int(confidence*100)}% Confidence Interval')
    
    ax.set_title(f'{model_name} - Predictions with {int(confidence*100)}% CI',
                 fontsize=16, fontweight='bold')
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        save_figure(fig, save_path)
    
    plt.show()


def plot_seasonal_decomposition(
    decomposition,
    figsize: Tuple[int, int] = (16, 12),
    save_path: Optional[Path] = None
):
    """
    Plot seasonal decomposition results
    
    Args:
        decomposition: Result from seasonal_decompose
        figsize: Figure size
        save_path: Optional path to save figure
    """
    fig, axes = plt.subplots(4, 1, figsize=figsize)
    fig.suptitle('Seasonal Decomposition', fontsize=16, fontweight='bold')
    
    # Original
    decomposition.observed.plot(ax=axes[0], color='blue', alpha=0.7)
    axes[0].set_ylabel('Observed')
    axes[0].grid(True, alpha=0.3)
    
    # Trend
    decomposition.trend.plot(ax=axes[1], color='green', alpha=0.7)
    axes[1].set_ylabel('Trend')
    axes[1].grid(True, alpha=0.3)
    
    # Seasonal
    decomposition.seasonal.plot(ax=axes[2], color='orange', alpha=0.7)
    axes[2].set_ylabel('Seasonal')
    axes[2].grid(True, alpha=0.3)
    
    # Residual
    decomposition.resid.plot(ax=axes[3], color='red', alpha=0.7)
    axes[3].set_ylabel('Residual')
    axes[3].grid(True, alpha=0.3)
    axes[3].axhline(y=0, color='black', linestyle='--', linewidth=1)
    
    plt.tight_layout()
    
    if save_path:
        save_figure(fig, save_path)
    
    plt.show()


def save_figure(fig, path: Path, dpi: int = 300):
    """
    Save figure to file
    
    Args:
        fig: Matplotlib figure
        path: Path to save to
        dpi: Resolution
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    print(f"Figure saved to: {path}")


if __name__ == "__main__":
    # Example usage
    print("Visualization utilities loaded successfully")
    
    # Create example data
    np.random.seed(42)
    timestamps = pd.date_range('2023-01-01', periods=100, freq='D')
    values = np.sin(np.linspace(0, 4*np.pi, 100)) * 10 + 50 + np.random.randn(100) * 2
    
    # Example plot
    plot_time_series(timestamps, values, title="Example Time Series")
