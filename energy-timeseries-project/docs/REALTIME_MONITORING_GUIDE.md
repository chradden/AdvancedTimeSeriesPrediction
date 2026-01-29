# 🔄 Real-Time Pipeline & Monitoring Guide

## Overview

This guide describes the real-time pipeline, monitoring, and weather API integration features added to the Energy Forecasting System.

## 🚀 New Features

### 1. Real-Time SMARD API Integration

**Module**: `src/data/smard_realtime.py`

Automatically fetches live energy data from the SMARD API with intelligent caching.

#### Features:
- ✅ Live data updates every 15 minutes
- ✅ Automatic caching to reduce API calls
- ✅ Graceful fallback when API is unavailable
- ✅ Support for all energy types

#### Usage:

```python
from src.data.smard_realtime import get_realtime_data

# Fetch latest solar data
data = get_realtime_data("solar", hours=168)
print(f"Latest value: {data['value'].iloc[-1]}")
```

#### API Endpoint:
```bash
# The real-time data is automatically used in predictions
curl -X POST "http://localhost:8000/api/predict/solar" \
  -H "Content-Type: application/json" \
  -d '{"hours": 24}'
```

---

### 2. Monitoring & Alerting System

**Module**: `src/monitoring/metrics.py`

Comprehensive monitoring with Prometheus metrics and model drift detection.

#### Tracked Metrics:
- ✅ Prediction count by model and energy type
- ✅ Prediction latency (p50, p95, p99)
- ✅ Mean Absolute Error (MAE) over rolling windows
- ✅ Mean Absolute Percentage Error (MAPE)
- ✅ Model drift scores
- ✅ Data quality scores
- ✅ API request rates and error rates

#### Prometheus Metrics Endpoint:
```bash
curl http://localhost:8000/metrics
```

#### Monitoring Status Endpoint:
```bash
curl http://localhost:8000/monitoring/status
```

**Response:**
```json
{
  "solar": {
    "summary": {
      "total_predictions": 150,
      "predictions_with_actuals": 45,
      "recent_mae": 245.3,
      "recent_mape": 3.1,
      "latest_prediction": "2026-01-29T15:00:00",
      "model_name": "XGBoost"
    },
    "drift": {
      "drift_detected": false,
      "drift_score": 0.08,
      "current_mae": 245.3,
      "baseline_mae": 249.03,
      "degradation_pct": 8.0
    }
  }
}
```

#### Model Drift Detection:

The system automatically compares current performance against baseline metrics:

```python
from src.monitoring.metrics import get_monitor

monitor = get_monitor()

# Set baseline (from training)
monitor.set_baseline_metrics('solar', {
    'mae': 249.03,
    'mape': 3.2,
    'r2': 0.9825
})

# Check for drift
drift = monitor.detect_drift('solar')
if drift['drift_detected']:
    print(f"⚠️  Model drift detected! Score: {drift['drift_score']}")
```

**Drift Thresholds:**
- `drift_score > 0.2`: Model performance degraded by 20%+
- `drift_score > 0.3`: Warning level - consider retraining
- `drift_score > 0.5`: Critical level - retraining required

---

### 3. Real Weather API Integration

**Module**: `src/data/weather_api.py`

Fetches real weather data from OpenWeather API to improve forecast accuracy.

#### Setup:

1. Get API key from [OpenWeather](https://openweathermap.org/api)
2. Set environment variable:
```bash
export OPENWEATHER_API_KEY="your_api_key_here"
```

#### Features:
- ✅ Current weather conditions
- ✅ Weather forecasts (up to 5 days)
- ✅ Aggregated weather across major German cities
- ✅ Automatic caching
- ✅ Realistic fallback data

#### Usage:

```python
from src.data.weather_api import WeatherAPIClient

client = WeatherAPIClient()

# Current weather
current = client.get_current_weather("berlin")
print(f"Temperature: {current['temperature']}°C")
print(f"Wind Speed: {current['wind_speed']} m/s")

# Forecast
forecast = client.get_forecast("berlin", hours=48)
print(forecast[['temperature', 'wind_speed', 'clouds']])

# Germany-wide aggregate
agg = client.get_aggregated_weather()
print(f"National avg temp: {agg['temperature']}°C")
```

#### API Endpoints:
```bash
# Current weather
curl "http://localhost:8000/weather/current?location=berlin"

# Weather forecast
curl "http://localhost:8000/weather/forecast?location=munich&hours=48"
```

#### Supported Cities:
- Berlin
- Hamburg
- Munich
- Cologne
- Frankfurt

---

## 📊 Grafana Dashboards

### Setup

1. Start the monitoring stack:
```bash
docker-compose --profile monitoring up
```

2. Access Grafana: `http://localhost:3000`
   - Username: `admin`
   - Password: `admin`

3. Import dashboard: `monitoring/grafana-dashboard.json`

### Dashboard Panels:

1. **Prediction Count** - Predictions per minute by energy type
2. **Model Drift Score** - Real-time drift detection
3. **Prediction MAE** - Error metrics over time
4. **Prediction MAPE** - Percentage errors
5. **Data Quality Score** - Data health monitoring
6. **Prediction Latency** - Response time p95
7. **API Request Rate** - Request throughput

### Alerts

Configured alerts in `monitoring/alerts.yml`:

| Alert | Condition | Severity |
|-------|-----------|----------|
| High Model Drift | drift_score > 0.5 for 10min | Critical |
| Moderate Model Drift | drift_score > 0.3 for 30min | Warning |
| Low Data Quality | quality_score < 0.5 for 15min | Critical |
| High Prediction Error | MAE > 1000 for 20min | Warning |
| High MAPE | MAPE > 15% for 20min | Warning |
| High API Error Rate | error_rate > 10% for 10min | Critical |

---

## 🔧 Configuration

### Docker Compose Profile

The monitoring components are in a separate profile:

```yaml
# Start API only
docker-compose up

# Start API + Monitoring
docker-compose --profile monitoring up
```

### Environment Variables

```bash
# Weather API (optional)
OPENWEATHER_API_KEY=your_key_here

# Monitoring (optional)
PROMETHEUS_PORT=9090
GRAFANA_PORT=3000
GRAFANA_ADMIN_PASSWORD=secure_password
```

---

## 📈 Monitoring Workflow

### 1. Baseline Setup

After training your models:

```python
from src.monitoring.metrics import get_monitor

monitor = get_monitor()

# Set baselines for all energy types
for energy_type, metrics in trained_models.items():
    monitor.set_baseline_metrics(energy_type, {
        'mae': metrics['test_mae'],
        'mape': metrics['test_mape'],
        'r2': metrics['test_r2']
    })
```

### 2. Continuous Monitoring

The system automatically:
1. Records all predictions with timestamps
2. Compares performance against baseline
3. Calculates drift scores
4. Exports metrics to Prometheus
5. Triggers alerts if thresholds exceeded

### 3. Handling Alerts

**Model Drift Detected:**
1. Review recent predictions in Grafana
2. Check data quality metrics
3. Verify external factors (weather, grid events)
4. Retrain model if sustained degradation

**Data Quality Issues:**
1. Check SMARD API status
2. Verify cache freshness
3. Review missing data percentage
4. Contact data provider if needed

---

## 🧪 Testing

### Test Real-Time Data
```bash
python src/data/smard_realtime.py
```

### Test Monitoring
```bash
python src/monitoring/metrics.py
```

### Test Weather API
```bash
python src/data/weather_api.py
```

### Test Full Integration
```bash
# Start API
python api_simple.py

# Make predictions
curl -X POST "http://localhost:8000/api/predict/solar" \
  -H "Content-Type: application/json" \
  -d '{"hours": 24}'

# Check metrics
curl http://localhost:8000/metrics

# Check monitoring status
curl http://localhost:8000/monitoring/status
```

---

## 📊 Performance Impact

| Feature | Latency Impact | Memory Impact |
|---------|----------------|---------------|
| Real-time Data | +50-200ms (cached: +5ms) | +10MB |
| Monitoring | +5-10ms per request | +50MB |
| Weather API | +100-300ms (cached: +5ms) | +5MB |

**Total**: ~100ms additional latency with caching, ~500ms without

---

## 🔮 Future Enhancements

### Planned (Q2 2026):
- [ ] Kafka streaming for high-throughput
- [ ] Apache Spark for distributed processing
- [ ] Time-series database (InfluxDB)
- [ ] Advanced anomaly detection
- [ ] Automated model retraining pipeline
- [ ] A/B testing framework

### Under Consideration:
- [ ] Multi-region deployment
- [ ] Edge computing for local predictions
- [ ] Federated learning
- [ ] Explainable AI dashboard

---

## 📚 References

- [Prometheus Documentation](https://prometheus.io/docs/)
- [Grafana Dashboards](https://grafana.com/docs/)
- [OpenWeather API](https://openweathermap.org/api)
- [SMARD API](https://www.smard.de/home/downloadcenter/download-marktdaten/)

---

**Last Updated**: 2026-01-29  
**Version**: 2.0.0  
**Status**: Production Ready ✅
