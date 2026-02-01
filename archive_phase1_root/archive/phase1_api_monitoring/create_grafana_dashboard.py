#!/usr/bin/env python3
"""
Script to create Grafana dashboard with Prometheus datasource
"""
import requests
import json
import time

GRAFANA_URL = "http://localhost:3000"
ADMIN_USER = "admin"
ADMIN_PASSWORD = "admin"

def create_datasource():
    """Create Prometheus datasource"""
    url = f"{GRAFANA_URL}/api/datasources"
    
    # Try with default Grafana API token first
    headers = {"Content-Type": "application/json"}
    
    payload = {
        "name": "Prometheus",
        "type": "prometheus",
        "url": "http://prometheus:9090",
        "access": "proxy",
        "isDefault": True,
        "jsonData": {
            "timeInterval": "15s"
        }
    }
    
    try:
        # First attempt: basic auth
        resp = requests.post(
            url,
            json=payload,
            auth=(ADMIN_USER, ADMIN_PASSWORD),
            headers=headers,
            timeout=10
        )
        print(f"Create datasource (basic auth): {resp.status_code}")
        print(f"Response: {resp.text[:200]}")
        
        if resp.status_code in [200, 409]:  # 409 = already exists
            try:
                data = resp.json()
                if "id" in data:
                    return data["id"]
            except:
                pass
            return 1
    except Exception as e:
        print(f"Error creating datasource: {e}")
    
    return None

def create_dashboard(ds_id):
    """Create dashboard"""
    url = f"{GRAFANA_URL}/api/dashboards/db"
    
    dashboard = {
        "dashboard": {
            "title": "Energy Forecasting Monitoring",
            "tags": ["energy", "monitoring"],
            "timezone": "browser",
            "schemaVersion": 36,
            "version": 0,
            "refresh": "30s",
            "time": {"from": "now-1h", "to": "now"},
            "panels": [
                {
                    "id": 1,
                    "title": "Prediction Count",
                    "type": "timeseries",
                    "gridPos": {"x": 0, "y": 0, "w": 12, "h": 8},
                    "targets": [
                        {
                            "expr": "rate(energy_predictions_total[5m])",
                            "refId": "A",
                            "datasourceUid": f"ds-{ds_id}"
                        }
                    ]
                },
                {
                    "id": 2,
                    "title": "API Requests",
                    "type": "timeseries",
                    "gridPos": {"x": 12, "y": 0, "w": 12, "h": 8},
                    "targets": [
                        {
                            "expr": "rate(energy_api_requests_total[5m])",
                            "refId": "A",
                            "datasourceUid": f"ds-{ds_id}"
                        }
                    ]
                },
                {
                    "id": 3,
                    "title": "Prediction Latency (p95)",
                    "type": "timeseries",
                    "gridPos": {"x": 0, "y": 8, "w": 12, "h": 8},
                    "targets": [
                        {
                            "expr": "histogram_quantile(0.95, rate(energy_prediction_latency_seconds_bucket[5m]))",
                            "refId": "A",
                            "datasourceUid": f"ds-{ds_id}"
                        }
                    ]
                }
            ]
        },
        "overwrite": True
    }
    
    try:
        resp = requests.post(
            url,
            json=dashboard,
            auth=(ADMIN_USER, ADMIN_PASSWORD),
            timeout=10
        )
        print(f"Create dashboard: {resp.status_code}")
        if resp.status_code in [200, 201]:
            print(f"Dashboard created successfully")
            return True
    except Exception as e:
        print(f"Error creating dashboard: {e}")
    
    return False

if __name__ == "__main__":
    print("Waiting for Grafana...")
    time.sleep(3)
    
    print("Creating datasource...")
    ds_id = create_datasource()
    print(f"Datasource ID: {ds_id}")
    
    if ds_id:
        print("Creating dashboard...")
        create_dashboard(ds_id)
    
    print("Done!")
