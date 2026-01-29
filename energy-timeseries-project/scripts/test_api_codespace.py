#!/usr/bin/env python3
"""
API Test für GitHub Codespaces
================================

Testet die API über die Codespace-URL
"""

import requests
import json
import os
from datetime import datetime

# Codespace URL konstruieren
CODESPACE_NAME = os.getenv('CODESPACE_NAME', 'unknown')
PORT = 8000

# Für Codespaces verwenden wir localhost, da wir im gleichen Container sind
API_BASE = "http://localhost:8000"

print("=" * 70)
print("🧪 Energy Forecasting API - Codespace Test")
print("=" * 70)
print(f"Codespace: {CODESPACE_NAME}")
print(f"API URL: {API_BASE}")
print(f"Public URL: https://{CODESPACE_NAME}-{PORT}.app.github.dev")
print("=" * 70)
print()

# Test 1: Health check
print("1️⃣  Health Check...")
try:
    response = requests.get(f"{API_BASE}/health", timeout=5)
    print(f"   ✅ Status: {response.status_code}")
    data = response.json()
    print(f"   Status: {data['status']}")
    print(f"   Model loaded: {data['model_loaded']}")
except Exception as e:
    print(f"   ❌ Error: {e}")
print()

# Test 2: Models endpoint
print("2️⃣  Available Models...")
try:
    response = requests.get(f"{API_BASE}/models", timeout=5)
    print(f"   ✅ Status: {response.status_code}")
    data = response.json()
    print(f"   Models: {', '.join(data['available_models'])}")
    print(f"   Type: {data['model_type']}")
except Exception as e:
    print(f"   ❌ Error: {e}")
print()

# Test 3: 24-hour forecast
print("3️⃣  24-Hour Solar Forecast...")
try:
    start_time = datetime.now()
    response = requests.post(
        f"{API_BASE}/predict/solar",
        json={"hours": 24},
        timeout=10
    )
    elapsed = (datetime.now() - start_time).total_seconds() * 1000
    
    print(f"   ✅ Status: {response.status_code}")
    print(f"   ⚡ Response time: {elapsed:.1f} ms")
    
    data = response.json()
    preds = data['predictions']
    
    print(f"\n   📊 Forecast Results:")
    print(f"      Predictions: {len(preds)} hours")
    print(f"      Model: {data['model']}")
    print(f"      Expected MAE: {data['mae_expected']} MW")
    print(f"      Expected R²: {data['r2_expected']}")
    
    print(f"\n   📈 Sample predictions:")
    for i in [0, 6, 12, 18, 23]:
        if i < len(preds):
            print(f"      {data['timestamps'][i]}: {preds[i]:>7.2f} MW")
    
    print(f"\n   📊 Statistics:")
    print(f"      Mean: {sum(preds)/len(preds):>7.2f} MW")
    print(f"      Min:  {min(preds):>7.2f} MW")
    print(f"      Max:  {max(preds):>7.2f} MW")
    
    # Save to CSV
    import pandas as pd
    df = pd.DataFrame({
        'timestamp': data['timestamps'],
        'prediction_MW': preds
    })
    csv_file = 'api_forecast_result.csv'
    df.to_csv(csv_file, index=False)
    print(f"\n   💾 Saved to: {csv_file}")
    
except Exception as e:
    print(f"   ❌ Error: {e}")
print()

print("=" * 70)
print("✅ Test completed!")
print("=" * 70)
print()
print("🌐 Access API Documentation:")
print(f"   https://{CODESPACE_NAME}-{PORT}.app.github.dev/docs")
print()
print("💡 Make sure the port visibility is set to 'Public' in:")
print("   VS Code → Ports Panel → Right-click Port 8000 → Port Visibility")
print()
