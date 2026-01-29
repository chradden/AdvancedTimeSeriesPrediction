# 🚀 Quick Start Guide

Get the Energy Forecasting Dashboard running in 2 minutes!

## Prerequisites

- Docker & Docker Compose installed
- OR Python 3.10+ with pip

## 🐳 Option 1: Docker with Monitoring Stack (Recommended!)

### For GitHub Codespaces or Local Development

This is the **easiest way** to get everything running with Grafana Monitoring!

```bash
cd AdvancedTimeSeriesPrediction/energy-timeseries-project
./start_monitoring.sh
```

That's it! ✨

**What gets started:**
- ✅ FastAPI (http://localhost:8000)
- ✅ Grafana Dashboard (http://localhost:3000)
- ✅ Prometheus Metrics (http://localhost:9090)
- ✅ Automatic monitoring & predictions

**Next steps in Codespaces:**
1. Open "PORTS" panel in VS Code
2. Make ports 8000, 3000, 9090 public (eye icon)
3. Open Grafana: http://localhost:3000 (admin/admin)

---

## 🐳 Option 2: Standard Docker Compose

### Just the API (no monitoring):

```bash
cd AdvancedTimeSeriesPrediction/energy-timeseries-project
docker-compose up
```

**Access**: http://localhost:8000/ui

### With monitoring profile:

```bash
docker-compose --profile monitoring up
```

---

## 🐍 Option 3: Python Local (Advanced)

### Step 1: Clone & Navigate

```bash
git clone https://github.com/chradden/AdvancedTimeSeriesPrediction.git
cd AdvancedTimeSeriesPrediction/energy-timeseries-project
```

### Step 2: Install Dependencies

```bash
pip install -r requirements-api.txt
```

### Step 3: Run the API

```bash
python api_simple.py
```

### Step 4: Open the Dashboard

**Web UI**: http://localhost:8000/ui

---

## 📊 Understanding the Grafana Dashboard

### The 7 Key Charts Explained

**1. Prediction Count** - How many forecasts have been generated?
**2. Model Drift Score** - Is our model still accurate? (0=good, 1=bad)
**3. Prediction MAE** - Average error in MW (lower is better)
**4. Prediction MAPE** - Average error in % (lower is better)
**5. Data Quality Score** - Are our input data clean? (higher is better)
**6. Prediction Latency** - How fast are predictions? (lower is faster)
**7. API Request Rate** - How many users are using the system?

👉 **Read the full guide**: [docs/GRAFANA_DASHBOARD_GUIDE_DE.md](energy-timeseries-project/docs/GRAFANA_DASHBOARD_GUIDE_DE.md)

---

## 🔄 GitHub Codespaces Setup

If you're running in a Codespaces environment:

```bash
# Simple 2-step setup:
cd energy-timeseries-project
./start_monitoring.sh

# Then:
# 1. Look for port notifications
# 2. Or manually: open PORTS tab → make 8000, 3000 public
# 3. Click "Open in Browser" on port 3000
```

### API Example

```python
import requests

response = requests.post(
    "http://localhost:8000/api/predict/solar",
    json={"hours": 24}
)

data = response.json()
print(f"First prediction: {data['predictions'][0]} MW")
print(f"Model R²: {data['r2_expected']}")
```

### cURL Example

```bash
curl -X POST "http://localhost:8000/api/predict/solar" \
  -H "Content-Type: application/json" \
  -d '{"hours": 24}'
```

---

## 🛑 Stopping the Application

Docker:
```bash
docker-compose down
```

Python:
Press `Ctrl+C` in the terminal

---

## 🔧 Troubleshooting

### Port 8000 Already in Use

```bash
# Stop any existing containers
docker-compose down

# Or change the port in docker-compose.yml
ports:
  - "8001:8000"
```

### Static Files Not Found

The static files should be automatically included in the Docker build. If not:

```bash
# Rebuild the Docker image
docker-compose up --build
```

### Container Won't Start

Check logs:
```bash
docker-compose logs api
```

---

## 📚 Next Steps

- Explore the [Jupyter Notebooks](notebooks/) for detailed analysis
- Read the [Full Documentation](docs/)
- Check the [API Documentation](http://localhost:8000/docs)
- Try different forecast horizons and energy types

---

## 🆘 Need Help?

- Check the main [README](../README.md)
- Review [Documentation](docs/)
- Open an issue on GitHub

---

**Enjoy forecasting! ⚡**
