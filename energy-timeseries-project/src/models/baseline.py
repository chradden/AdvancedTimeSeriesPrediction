"""
Baseline models for time series forecasting
"""

import numpy as np
import pandas as pd
from typing import Optional


class NaiveForecaster:
    """Naive forecast: y_hat(t) = y(t-1)"""
    
    def __init__(self):
        self.last_value = None
        
    def fit(self, y_train: np.ndarray):
        """Store last training value"""
        self.last_value = y_train[-1]
        return self
    
    def predict(self, steps: int = 1) -> np.ndarray:
        """Predict by repeating last value"""
        return np.full(steps, self.last_value)


class SeasonalNaiveForecaster:
    """Seasonal naive: y_hat(t) = y(t-seasonality)"""
    
    def __init__(self, seasonality: int = 24):
        """
        Args:
            seasonality: Seasonal period (e.g., 24 for daily pattern in hourly data)
        """
        self.seasonality = seasonality
        self.last_season = None
        
    def fit(self, y_train: np.ndarray):
        """Store last seasonal period"""
        self.last_season = y_train[-self.seasonality:]
        return self
    
    def predict(self, steps: int = 1) -> np.ndarray:
        """Predict by repeating seasonal pattern"""
        n_full_seasons = steps // self.seasonality
        remainder = steps % self.seasonality
        
        predictions = np.tile(self.last_season, n_full_seasons + 1)
        return predictions[:steps]


class MovingAverageForecaster:
    """Simple moving average forecast"""
    
    def __init__(self, window: int = 7):
        """
        Args:
            window: Window size for moving average
        """
        self.window = window
        self.last_values = None
        
    def fit(self, y_train: np.ndarray):
        """Store last window values"""
        self.last_values = y_train[-self.window:]
        return self
    
    def predict(self, steps: int = 1) -> np.ndarray:
        """Predict using moving average"""
        avg = np.mean(self.last_values)
        return np.full(steps, avg)


class DriftForecaster:
    """Drift method: Linear extrapolation from first to last value"""
    
    def __init__(self):
        self.slope = None
        self.last_value = None
        
    def fit(self, y_train: np.ndarray):
        """Calculate drift (slope)"""
        self.last_value = y_train[-1]
        # Slope from first to last value
        self.slope = (y_train[-1] - y_train[0]) / (len(y_train) - 1)
        return self
    
    def predict(self, steps: int = 1) -> np.ndarray:
        """Predict by extrapolating drift"""
        return self.last_value + self.slope * np.arange(1, steps + 1)


class MeanForecaster:
    """Average method: Forecast is the mean of historical data"""
    
    def __init__(self):
        self.mean = None
        
    def fit(self, y_train: np.ndarray):
        """Calculate mean"""
        self.mean = np.mean(y_train)
        return self
    
    def predict(self, steps: int = 1) -> np.ndarray:
        """Predict using mean"""
        return np.full(steps, self.mean)


if __name__ == "__main__":
    # Example usage
    print("Baseline models loaded successfully\n")
    
    # Generate example data with seasonality
    np.random.seed(42)
    t = np.arange(200)
    seasonal = 10 * np.sin(2 * np.pi * t / 24)  # Daily pattern
    trend = 0.1 * t
    noise = np.random.randn(200) * 2
    y = 50 + trend + seasonal + noise
    
    # Split train/test
    y_train = y[:150]
    y_test = y[150:]
    
    # Test all models
    models = {
        'Naive': NaiveForecaster(),
        'Seasonal Naive': SeasonalNaiveForecaster(seasonality=24),
        'Moving Average': MovingAverageForecaster(window=7),
        'Drift': DriftForecaster(),
        'Mean': MeanForecaster()
    }
    
    print("Testing baseline models:\n")
    for name, model in models.items():
        model.fit(y_train)
        predictions = model.predict(steps=len(y_test))
        mae = np.mean(np.abs(y_test - predictions))
        print(f"{name:20s}: MAE = {mae:.2f}")
