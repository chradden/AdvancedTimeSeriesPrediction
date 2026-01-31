"""
Data loading utilities for SMARD API (Bundesnetzagentur)

SMARD API Documentation: https://www.smard.de/home/downloadcenter/download-marktdaten/
API Endpoint: https://www.smard.de/app/chart_data/{filter}/{region}/{resolution}_{timestamp}.json

Filter Codes (important ones):
- 1223: Photovoltaik (Solar Generation)
- 1224: Wind Offshore
- 1225: Wind Onshore  
- 410: Stromverbrauch (Consumption)
- 4169: Day-Ahead Auction (Prices)
- 4387: Realized Generation Total
- 4359: Forecast Generation Total

Region Codes:
- DE: Germany
- DE-LU: Germany/Luxembourg

Resolution:
- hour: Hourly data
- quarterhour: 15-minute data
- day: Daily data
- week: Weekly data
- month: Monthly data
- year: Yearly data
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
from pathlib import Path
from typing import Optional, List, Tuple
import json


class SMARDDataLoader:
    """Load data from SMARD API (Bundesnetzagentur)"""
    
    BASE_URL = "https://www.smard.de/app"
    
    # Filter codes
    # NOTE: Filter 1223 liefert FALSCHE Daten (invertiert/nicht Solar)
    # Filter 4068 = Korrekte Solar-Daten (Generation Actual)
    FILTERS = {
        'solar': 4068,  # ✅ KORRIGIERT: War 1223 (falsch!) → Jetzt 4068 (korrekt)
        'wind_offshore': 1224,
        'wind_onshore': 1225,
        'consumption': 410,
        'price_day_ahead': 4169,
        'generation_total': 4387,
        'generation_forecast': 4359,
        'biomass': 1227,
        'hydro_run': 1228,
        'hydro_pump': 4070,
        'others': 1229,
        'nuclear': 1221,
        'brown_coal': 1216,
        'hard_coal': 1215,
        'gas': 1222,
    }
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize SMARD data loader
        
        Args:
            cache_dir: Directory to cache downloaded data
        """
        self.cache_dir = cache_dir or Path("../data/raw")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def get_available_timestamps(
        self, 
        filter_code: int, 
        region: str = "DE",
        resolution: str = "hour"
    ) -> List[int]:
        """
        Get available timestamps for a given filter
        
        Args:
            filter_code: SMARD filter code (e.g., 1223 for solar)
            region: Region code (default: DE)
            resolution: Time resolution (hour, day, etc.)
            
        Returns:
            List of available UNIX timestamps (in milliseconds)
        """
        url = f"{self.BASE_URL}/chart_data/{filter_code}/{region}/index_{resolution}.json"
        
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            timestamps = data.get('timestamps', [])
            return timestamps
        except Exception as e:
            print(f"Error getting timestamps: {e}")
            return []
    
    def download_data(
        self,
        filter_code: int,
        timestamp: int,
        region: str = "DE",
        resolution: str = "hour"
    ) -> Optional[pd.DataFrame]:
        """
        Download data for a specific timestamp
        
        Args:
            filter_code: SMARD filter code
            timestamp: UNIX timestamp in milliseconds
            region: Region code
            resolution: Time resolution
            
        Returns:
            DataFrame with columns ['timestamp', 'value']
        """
        url = f"{self.BASE_URL}/chart_data/{filter_code}/{region}/{filter_code}_{region}_{resolution}_{timestamp}.json"
        
        try:
            print(f"  Requesting: {url}")
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            # Data comes as list of [timestamp_ms, value] pairs
            series = data.get('series', [])
            if not series:
                print(f"    Warning: No series data in response")
                return None
                
            df = pd.DataFrame(series, columns=['timestamp', 'value'])
            
            # Convert timestamp from milliseconds to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Handle missing values (represented as None)
            df['value'] = pd.to_numeric(df['value'], errors='coerce')
            
            print(f"    Downloaded {len(df)} records")
            return df
            
        except requests.exceptions.HTTPError as e:
            print(f"    HTTP Error {e.response.status_code}: {url}")
            return None
        except Exception as e:
            print(f"    Error downloading data: {type(e).__name__}: {e}")
            return None
    
    def load_data(
        self,
        filter_name: str,
        start_date: str,
        end_date: str,
        region: str = "DE",
        resolution: str = "hour",
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Load data for a date range
        
        Args:
            filter_name: Name from FILTERS dict (e.g., 'solar', 'consumption')
            start_date: Start date as string 'YYYY-MM-DD'
            end_date: End date as string 'YYYY-MM-DD'
            region: Region code
            resolution: Time resolution
            use_cache: Whether to use cached data if available
            
        Returns:
            DataFrame with complete time series
        """
        filter_code = self.FILTERS.get(filter_name)
        if filter_code is None:
            raise ValueError(f"Unknown filter: {filter_name}. Available: {list(self.FILTERS.keys())}")
        
        # Check cache
        cache_file = self.cache_dir / f"{filter_name}_{start_date}_{end_date}_{resolution}.csv"
        if use_cache and cache_file.exists():
            print(f"Loading from cache: {cache_file}")
            return pd.read_csv(cache_file, parse_dates=['timestamp'])
        
        print(f"Downloading {filter_name} data from SMARD API...")
        print(f"Period: {start_date} to {end_date}")
        
        # Get available timestamps
        timestamps = self.get_available_timestamps(filter_code, region, resolution)
        if not timestamps:
            raise ValueError("No timestamps available")
        
        # Filter timestamps to date range
        start_ts = int(pd.Timestamp(start_date).timestamp() * 1000)
        end_ts = int(pd.Timestamp(end_date).timestamp() * 1000)
        
        relevant_timestamps = [
            ts for ts in timestamps 
            if start_ts <= ts <= end_ts
        ]
        
        if not relevant_timestamps:
            print(f"Warning: No data found in range. Available range:")
            print(f"  From: {pd.Timestamp(min(timestamps), unit='ms')}")
            print(f"  To: {pd.Timestamp(max(timestamps), unit='ms')}")
            # Download closest available data
            relevant_timestamps = timestamps[-5:]  # Get last 5 available chunks
        
        print(f"Downloading {len(relevant_timestamps)} chunks...")
        
        # Download all chunks
        dfs = []
        for i, ts in enumerate(relevant_timestamps, 1):
            print(f"  Chunk {i}/{len(relevant_timestamps)}: {pd.Timestamp(ts, unit='ms').date()}", end='\r')
            df = self.download_data(filter_code, ts, region, resolution)
            if df is not None and not df.empty:
                dfs.append(df)
            time.sleep(0.5)  # Be polite to the API
        
        print("\nCombining data...")
        
        if not dfs:
            raise ValueError("No data downloaded")
        
        # Combine all chunks
        result = pd.concat(dfs, ignore_index=True)
        
        # Remove duplicates and sort
        result = result.drop_duplicates(subset=['timestamp']).sort_values('timestamp')
        
        # Filter to exact date range
        result = result[
            (result['timestamp'] >= start_date) & 
            (result['timestamp'] <= end_date + ' 23:59:59')
        ]
        
        # Reset index
        result = result.reset_index(drop=True)
        
        print(f"Downloaded {len(result)} records")
        print(f"Date range: {result['timestamp'].min()} to {result['timestamp'].max()}")
        print(f"Missing values: {result['value'].isna().sum()} ({result['value'].isna().sum()/len(result)*100:.1f}%)")
        
        # Save to cache
        result.to_csv(cache_file, index=False)
        print(f"Saved to cache: {cache_file}")
        
        return result


def load_smard_data(
    filter_name: str = 'solar',
    start_date: str = '2022-01-01',
    end_date: str = '2024-12-31',
    resolution: str = 'hour',
    cache_dir: Optional[Path] = None
) -> pd.DataFrame:
    """
    Convenience function to load SMARD data
    
    Args:
        filter_name: Data type to load (solar, consumption, price_day_ahead, etc.)
        start_date: Start date YYYY-MM-DD
        end_date: End date YYYY-MM-DD
        resolution: hour, day, etc.
        cache_dir: Cache directory
        
    Returns:
        DataFrame with timestamp and value columns
    """
    loader = SMARDDataLoader(cache_dir=cache_dir)
    return loader.load_data(filter_name, start_date, end_date, resolution=resolution)


if __name__ == "__main__":
    # Example usage
    print("SMARD Data Loader - Example Usage\n")
    
    # Load solar generation data for 2023
    df = load_smard_data(
        filter_name='solar',
        start_date='2023-01-01',
        end_date='2023-12-31',
        resolution='hour'
    )
    
    print("\nData shape:", df.shape)
    print("\nFirst rows:")
    print(df.head())
    print("\nBasic statistics:")
    print(df['value'].describe())
