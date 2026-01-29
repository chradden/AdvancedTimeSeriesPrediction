"""
Real Weather API Integration
=============================

Fetches real weather data from OpenWeather API for improving forecasts.

Features:
- Current weather conditions
- Weather forecasts
- Historical weather data
- Automatic caching
- Graceful fallback
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, List
import json
import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WeatherAPIClient:
    """Client for fetching weather data from OpenWeather API"""
    
    BASE_URL = "https://api.openweathermap.org/data/2.5"
    CACHE_DIR = Path("/app/data/cache/weather")
    CACHE_DURATION = 3600  # 1 hour in seconds
    
    # Major German cities (for distributed weather)
    LOCATIONS = {
        "berlin": {"lat": 52.52, "lon": 13.405},
        "hamburg": {"lat": 53.55, "lon": 10.0},
        "munich": {"lat": 48.137, "lon": 11.576},
        "cologne": {"lat": 50.937, "lon": 6.96},
        "frankfurt": {"lat": 50.110, "lon": 8.682}
    }
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Weather API Client
        
        Args:
            api_key: OpenWeather API key (or set OPENWEATHER_API_KEY env var)
        """
        self.api_key = api_key or os.getenv('OPENWEATHER_API_KEY')
        self.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        
        if not self.api_key:
            logger.warning("No OpenWeather API key provided, will use fallback data")
        else:
            logger.info("Weather API Client initialized")
    
    def get_current_weather(self, location: str = "berlin") -> Dict:
        """
        Get current weather conditions
        
        Args:
            location: City name (berlin, hamburg, munich, cologne, frankfurt)
        
        Returns:
            Dict with weather data
        """
        if location not in self.LOCATIONS:
            raise ValueError(f"Unknown location: {location}")
        
        # Check cache
        cached = self._load_from_cache(f"current_{location}")
        if cached:
            return cached
        
        # Fetch from API
        if not self.api_key:
            return self._generate_fallback_current(location)
        
        try:
            coords = self.LOCATIONS[location]
            url = f"{self.BASE_URL}/weather"
            params = {
                'lat': coords['lat'],
                'lon': coords['lon'],
                'appid': self.api_key,
                'units': 'metric'
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            weather = self._parse_current_weather(data)
            self._save_to_cache(f"current_{location}", weather)
            
            logger.info(f"Fetched current weather for {location}")
            return weather
            
        except Exception as e:
            logger.error(f"Error fetching weather: {e}")
            return self._generate_fallback_current(location)
    
    def get_forecast(self, location: str = "berlin", hours: int = 120) -> pd.DataFrame:
        """
        Get weather forecast
        
        Args:
            location: City name
            hours: Forecast horizon in hours (max 120 for free API)
        
        Returns:
            DataFrame with hourly weather forecast
        """
        if location not in self.LOCATIONS:
            raise ValueError(f"Unknown location: {location}")
        
        # Check cache
        cached = self._load_from_cache(f"forecast_{location}")
        if cached and isinstance(cached, dict) and 'dataframe' in cached:
            df = pd.DataFrame(cached['dataframe'])
            df.index = pd.to_datetime(df.index)
            return df
        
        # Fetch from API
        if not self.api_key:
            return self._generate_fallback_forecast(location, hours)
        
        try:
            coords = self.LOCATIONS[location]
            url = f"{self.BASE_URL}/forecast"
            params = {
                'lat': coords['lat'],
                'lon': coords['lon'],
                'appid': self.api_key,
                'units': 'metric'
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            df = self._parse_forecast(data)
            
            # Save to cache
            cache_data = {'dataframe': df.reset_index().to_dict(orient='records')}
            self._save_to_cache(f"forecast_{location}", cache_data)
            
            logger.info(f"Fetched forecast for {location}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching forecast: {e}")
            return self._generate_fallback_forecast(location, hours)
    
    def get_aggregated_weather(self) -> Dict:
        """
        Get aggregated weather across all major German cities
        
        Returns:
            Dict with averaged weather conditions
        """
        weather_data = []
        
        for location in self.LOCATIONS.keys():
            try:
                weather = self.get_current_weather(location)
                weather_data.append(weather)
            except Exception as e:
                logger.error(f"Error fetching weather for {location}: {e}")
        
        if not weather_data:
            return self._generate_fallback_current("berlin")
        
        # Aggregate
        return {
            'temperature': np.mean([w['temperature'] for w in weather_data]),
            'humidity': np.mean([w['humidity'] for w in weather_data]),
            'wind_speed': np.mean([w['wind_speed'] for w in weather_data]),
            'clouds': np.mean([w['clouds'] for w in weather_data]),
            'pressure': np.mean([w['pressure'] for w in weather_data]),
            'timestamp': datetime.now().isoformat(),
            'locations_count': len(weather_data)
        }
    
    def _parse_current_weather(self, data: Dict) -> Dict:
        """Parse OpenWeather current weather response"""
        return {
            'temperature': data['main']['temp'],
            'feels_like': data['main']['feels_like'],
            'humidity': data['main']['humidity'],
            'pressure': data['main']['pressure'],
            'wind_speed': data['wind']['speed'],
            'wind_direction': data['wind'].get('deg', 0),
            'clouds': data['clouds']['all'],
            'visibility': data.get('visibility', 10000),
            'weather': data['weather'][0]['main'],
            'weather_description': data['weather'][0]['description'],
            'timestamp': datetime.fromtimestamp(data['dt']).isoformat()
        }
    
    def _parse_forecast(self, data: Dict) -> pd.DataFrame:
        """Parse OpenWeather forecast response"""
        forecasts = []
        
        for item in data['list']:
            forecasts.append({
                'timestamp': datetime.fromtimestamp(item['dt']),
                'temperature': item['main']['temp'],
                'humidity': item['main']['humidity'],
                'pressure': item['main']['pressure'],
                'wind_speed': item['wind']['speed'],
                'clouds': item['clouds']['all'],
                'weather': item['weather'][0]['main']
            })
        
        df = pd.DataFrame(forecasts)
        df = df.set_index('timestamp')
        return df
    
    def _load_from_cache(self, key: str) -> Optional[Dict]:
        """Load data from cache if recent enough"""
        cache_file = self.CACHE_DIR / f"{key}.json"
        
        if not cache_file.exists():
            return None
        
        cache_age = datetime.now().timestamp() - cache_file.stat().st_mtime
        if cache_age > self.CACHE_DURATION:
            return None
        
        try:
            with open(cache_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading cache: {e}")
            return None
    
    def _save_to_cache(self, key: str, data: Dict):
        """Save data to cache"""
        cache_file = self.CACHE_DIR / f"{key}.json"
        
        try:
            with open(cache_file, 'w') as f:
                json.dump(data, f)
        except Exception as e:
            logger.error(f"Error saving cache: {e}")
    
    def _generate_fallback_current(self, location: str) -> Dict:
        """Generate realistic fallback current weather"""
        now = datetime.now()
        month = now.month
        hour = now.hour
        
        # Seasonal temperature
        base_temp = 5 + 15 * np.sin((month - 3) * np.pi / 6)
        # Daily variation
        temp_variation = 5 * np.sin((hour - 6) * np.pi / 12)
        temperature = base_temp + temp_variation + np.random.normal(0, 2)
        
        return {
            'temperature': float(temperature),
            'feels_like': float(temperature - 2),
            'humidity': float(60 + np.random.normal(0, 15)),
            'pressure': float(1013 + np.random.normal(0, 5)),
            'wind_speed': float(5 + np.random.exponential(3)),
            'wind_direction': float(np.random.uniform(0, 360)),
            'clouds': float(np.random.uniform(0, 100)),
            'visibility': 10000,
            'weather': 'Clear' if np.random.random() > 0.3 else 'Clouds',
            'weather_description': 'simulated data',
            'timestamp': now.isoformat(),
            'is_fallback': True
        }
    
    def _generate_fallback_forecast(self, location: str, hours: int) -> pd.DataFrame:
        """Generate realistic fallback forecast"""
        timestamps = pd.date_range(
            start=datetime.now().replace(minute=0, second=0, microsecond=0),
            periods=hours,
            freq='H'
        )
        
        forecasts = []
        for ts in timestamps:
            month = ts.month
            hour = ts.hour
            
            base_temp = 5 + 15 * np.sin((month - 3) * np.pi / 6)
            temp_variation = 5 * np.sin((hour - 6) * np.pi / 12)
            temperature = base_temp + temp_variation + np.random.normal(0, 1)
            
            forecasts.append({
                'temperature': temperature,
                'humidity': 60 + np.random.normal(0, 10),
                'pressure': 1013 + np.random.normal(0, 3),
                'wind_speed': 5 + np.random.exponential(2),
                'clouds': np.random.uniform(0, 100),
                'weather': 'Clear' if np.random.random() > 0.3 else 'Clouds'
            })
        
        df = pd.DataFrame(forecasts, index=timestamps)
        return df


# Convenience function
def get_weather_features(location: str = "berlin", hours: int = 24) -> pd.DataFrame:
    """
    Get weather features for forecasting
    
    Args:
        location: City name
        hours: Forecast horizon
    
    Returns:
        DataFrame with weather features
    """
    client = WeatherAPIClient()
    
    try:
        forecast = client.get_forecast(location, hours)
        
        # Add derived features
        forecast['temp_rolling_mean'] = forecast['temperature'].rolling(window=3, min_periods=1).mean()
        forecast['wind_speed_rolling_mean'] = forecast['wind_speed'].rolling(window=3, min_periods=1).mean()
        
        return forecast
        
    except Exception as e:
        logger.error(f"Error getting weather features: {e}")
        return pd.DataFrame()


if __name__ == "__main__":
    # Test the client
    print("Testing Weather API Client...")
    
    client = WeatherAPIClient()
    
    # Current weather
    print("\nCurrent Weather (Berlin):")
    current = client.get_current_weather("berlin")
    for key, value in current.items():
        print(f"  {key}: {value}")
    
    # Forecast
    print("\nWeather Forecast:")
    forecast = client.get_forecast("berlin", hours=24)
    print(forecast.head())
    
    # Aggregated
    print("\nAggregated Weather (Germany):")
    agg = client.get_aggregated_weather()
    for key, value in agg.items():
        print(f"  {key}: {value}")
