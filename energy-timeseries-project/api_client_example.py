"""
API Client Example
==================

Demonstrates how to use the Energy Time Series Forecasting API
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

# API Base URL
BASE_URL = "http://localhost:8000"

def check_health():
    """Check API health"""
    response = requests.get(f"{BASE_URL}/health")
    print("Health Check:")
    print(json.dumps(response.json(), indent=2))
    print()

def get_models():
    """Get available models"""
    response = requests.get(f"{BASE_URL}/models")
    print("Available Models:")
    print(json.dumps(response.json(), indent=2))
    print()

def get_metrics():
    """Get model metrics"""
    response = requests.get(f"{BASE_URL}/metrics")
    print("Model Metrics:")
    print(json.dumps(response.json(), indent=2))
    print()

def predict_solar_simple():
    """Simple solar prediction example"""
    print("\n" + "="*80)
    print("SOLAR PREDICTION EXAMPLE - Next 24 Hours")
    print("="*80)
    
    # Generate sample historical data (last 7 days for context)
    start_date = datetime(2024, 1, 1)
    timestamps = [(start_date + timedelta(hours=i)).isoformat() 
                  for i in range(168)]  # 7 days
    
    # Simulate solar generation pattern
    values = []
    for i in range(168):
        hour = i % 24
        # Day/night pattern
        if 6 <= hour <= 18:
            base = 500 + 300 * np.sin(np.pi * (hour - 6) / 12)
            noise = np.random.normal(0, 50)
            values.append(max(0, base + noise))
        else:
            values.append(0)
    
    # Request 24 hours forecast
    payload = {
        "historical_data": {
            "timestamps": timestamps,
            "values": values
        },
        "forecast_horizon": 24,  # 24 hours ahead
        "model": "xgboost"
    }
    
    # API Call
    response = requests.post(f"{BASE_URL}/predict/solar", json=payload)
    
    if response.status_code == 200:
        result = response.json()
        print("✅ Forecast successful!")
        print(f"\nModel Used: {result['model_used']}")
        print(f"Forecast Horizon: {result['metadata']['forecast_horizon']} hours")
        print(f"\nFirst 10 Predictions:")
        for i in range(min(10, len(result['predictions']))):
            timestamp = result['timestamps'][i]
            value = result['predictions'][i]
            hour = pd.to_datetime(timestamp).hour
            print(f"  {timestamp} (Hour {hour:02d}): {value:.2f} MW")
        
        if len(result['predictions']) > 10:
            print(f"\n  ... ({len(result['predictions'])} total predictions)")
        
        # Statistics
        predictions = result['predictions']
        print(f"\n📊 Forecast Statistics:")
        print(f"  Mean: {np.mean(predictions):.2f} MW")
        print(f"  Max: {np.max(predictions):.2f} MW")
        print(f"  Min: {np.min(predictions):.2f} MW")
    else:
        print(f"❌ Error: {response.status_code}")
        print(response.text)

def predict_multi_series():
    """Multi-series prediction example"""
    print("\n" + "="*80)
    print("MULTI-SERIES PREDICTION EXAMPLE")
    print("="*80)
    
    payload = {
        "forecast_horizon": 24,
        "series": ["solar", "wind_offshore", "consumption"]
    }
    
    response = requests.post(f"{BASE_URL}/predict/multi", json=payload)
    
    if response.status_code == 200:
        result = response.json()
        print("✅ Multi-series forecast successful!")
        print(f"\nGenerated at: {result['generated_at']}")
        
        for series, data in result['forecasts'].items():
            if 'error' in data:
                print(f"\n{series}: {data['error']}")
            else:
                predictions = data['predictions']
                print(f"\n{series}:")
                print(f"  Forecast Horizon: {data['forecast_horizon']} hours")
                print(f"  Sample Predictions: {predictions[:3]} ... (showing first 3)")
    else:
        print(f"❌ Error: {response.status_code}")
        print(response.text)

def load_real_data_and_predict():
    """Load real data from CSV and predict"""
    print("\n" + "="*80)
    print("PREDICTION WITH REAL DATA - Next 48 Hours")
    print("="*80)
    
    try:
        # Load real solar data
        df = pd.read_csv('data/raw/solar_2022-01-01_2024-12-31_hour.csv', 
                        parse_dates=['DateTime'])
        
        # Take last 7 days as historical context
        historical = df.tail(168)
        
        timestamps = historical['DateTime'].dt.strftime('%Y-%m-%dT%H:%M:%S').tolist()
        values = historical['Value_MWh'].tolist()
        
        payload = {
            "historical_data": {
                "timestamps": timestamps,
                "values": values
            },
            "forecast_horizon": 48,  # 48 hours (2 days ahead)
            "model": "xgboost"
        }
        
        response = requests.post(f"{BASE_URL}/predict/solar", json=payload)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Forecast with real data successful!")
            print(f"\nHistorical Context: {result['metadata']['historical_samples']} samples")
            print(f"Forecast Generated: {result['metadata']['generated_at']}")
            
            # Display summary statistics
            predictions = result['predictions']
            print(f"\n📊 Forecast Statistics (48 hours):")
            print(f"  Mean: {np.mean(predictions):.2f} MW")
            print(f"  Max: {np.max(predictions):.2f} MW")
            print(f"  Min: {np.min(predictions):.2f} MW")
            
            # Show predictions by day
            print(f"\n📅 Daily Breakdown:")
            for day in range(2):  # 2 days
                day_start = day * 24
                day_end = (day + 1) * 24
                day_preds = predictions[day_start:day_end]
                day_date = pd.to_datetime(result['timestamps'][day_start]).date()
                print(f"  Day {day+1} ({day_date}):")
                print(f"    Mean: {np.mean(day_preds):.2f} MW")
                print(f"    Max: {np.max(day_preds):.2f} MW (Hour {np.argmax(day_preds)})")
            
            # Save to CSV
            output_df = pd.DataFrame({
                'timestamp': result['timestamps'],
                'predicted_solar_mw': predictions
            })
            output_df.to_csv('forecast_output.csv', index=False)
            print(f"\n✅ Forecast saved to forecast_output.csv")
            print(f"   Total predictions: {len(predictions)} hours")
            
        else:
            print(f"❌ Error: {response.status_code}")
            print(response.text)
            
    except FileNotFoundError:
        print("⚠️ Real data file not found. Using simulated data instead.")
        predict_solar_simple()

def main():
    """Main demo"""
    print("="*80)
    print("ENERGY TIME SERIES FORECASTING API CLIENT")
    print("="*80)
    print()
    
    try:
        # Check if API is running
        check_health()
        
        # Get available models
        get_models()
        
        # Get model metrics
        get_metrics()
        
        # Simple prediction
        predict_solar_simple()
        
        # Multi-series prediction
        predict_multi_series()
        
        # Prediction with real data
        load_real_data_and_predict()
        
        print("\n" + "="*80)
        print("✨ Demo completed!")
        print("="*80)
        
    except requests.exceptions.ConnectionError:
        print("❌ Error: Cannot connect to API")
        print("Make sure the API is running: python api.py")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
