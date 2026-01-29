#!/usr/bin/env python3
"""
Quick API Test - Energy Forecasting API
========================================

Testet alle Endpoints der laufenden API.
"""

import requests
import json
from datetime import datetime

API_BASE = "http://localhost:8000"

print("=" * 70)
print("🧪 Energy Forecasting API - Quick Test")
print("=" * 70)
print()

# Test 1: Root endpoint
print("1️⃣  Testing root endpoint...")
try:
    response = requests.get(f"{API_BASE}/")
    print(f"   ✅ Status: {response.status_code}")
    print(f"   Response: {json.dumps(response.json(), indent=2)}")
except Exception as e:
    print(f"   ❌ Error: {e}")
print()

# Test 2: Health check
print("2️⃣  Testing health endpoint...")
try:
    response = requests.get(f"{API_BASE}/health")
    print(f"   ✅ Status: {response.status_code}")
    data = response.json()
    print(f"   Status: {data['status']}")
    print(f"   Model loaded: {data['model_loaded']}")
    print(f"   Timestamp: {data['timestamp']}")
except Exception as e:
    print(f"   ❌ Error: {e}")
print()

# Test 3: List models
print("3️⃣  Testing models endpoint...")
try:
    response = requests.get(f"{API_BASE}/models")
    print(f"   ✅ Status: {response.status_code}")
    print(f"   Response: {json.dumps(response.json(), indent=2)}")
except Exception as e:
    print(f"   ❌ Error: {e}")
print()

# Test 4: Solar forecast (6 hours)
print("4️⃣  Testing 6-hour solar forecast...")
try:
    response = requests.post(
        f"{API_BASE}/predict/solar",
        json={"hours": 6}
    )
    print(f"   ✅ Status: {response.status_code}")
    data = response.json()
    print(f"   Model: {data['model']}")
    print(f"   Expected MAE: {data['mae_expected']} MW")
    print(f"   Expected R²: {data['r2_expected']}")
    print(f"   Predictions ({len(data['predictions'])} values):")
    for i, (pred, ts) in enumerate(zip(data['predictions'][:6], data['timestamps'][:6])):
        print(f"      {ts}: {pred:.2f} MW")
except Exception as e:
    print(f"   ❌ Error: {e}")
print()

# Test 5: Solar forecast (24 hours)
print("5️⃣  Testing 24-hour solar forecast...")
try:
    start_time = datetime.now()
    response = requests.post(
        f"{API_BASE}/predict/solar",
        json={"hours": 24}
    )
    elapsed = (datetime.now() - start_time).total_seconds() * 1000
    
    print(f"   ✅ Status: {response.status_code}")
    data = response.json()
    print(f"   Response time: {elapsed:.1f} ms")
    print(f"   Predictions: {len(data['predictions'])} values")
    print(f"   First 3 predictions:")
    for i in range(3):
        print(f"      {data['timestamps'][i]}: {data['predictions'][i]:.2f} MW")
    print(f"   ...")
    print(f"   Last prediction:")
    print(f"      {data['timestamps'][-1]}: {data['predictions'][-1]:.2f} MW")
    
    # Statistics
    preds = data['predictions']
    print(f"\n   📊 Statistics:")
    print(f"      Mean: {sum(preds)/len(preds):.2f} MW")
    print(f"      Min: {min(preds):.2f} MW")
    print(f"      Max: {max(preds):.2f} MW")
    
except Exception as e:
    print(f"   ❌ Error: {e}")
print()

print("=" * 70)
print("✅ All tests completed!")
print("=" * 70)
print()
print("🌐 Interactive API Documentation:")
print(f"   {API_BASE}/docs")
print()
