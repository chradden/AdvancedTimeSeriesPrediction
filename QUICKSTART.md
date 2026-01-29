# 🚀 Quick Start Guide

Get the Energy Forecasting Dashboard running in 2 minutes!

## Prerequisites

- Docker & Docker Compose installed
- OR Python 3.10+ with pip

## 🐳 Option 1: Docker (Recommended)

### Step 1: Clone the Repository

```bash
git clone https://github.com/chradden/AdvancedTimeSeriesPrediction.git
cd AdvancedTimeSeriesPrediction/energy-timeseries-project
```

### Step 2: Start the Application

```bash
docker-compose up
```

### Step 3: Open the Dashboard

**Web UI**: http://localhost:8000/ui
**API Docs**: http://localhost:8000/docs

That's it! 🎉

---

## 🐍 Option 2: Python Local

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

## 🌐 GitHub Codespaces

If you're running in a Codespace:

1. Start the app: `docker-compose up`
2. The port will be automatically forwarded
3. Click on the "Ports" tab and open port 8000
4. Access the UI at the provided URL + `/ui`

---

## 📊 Using the Dashboard

### Generate a Forecast

1. **Select Energy Type**: Solar, Wind Offshore, Wind Onshore, Consumption, or Price
2. **Set Forecast Horizon**: 1-168 hours (default: 24h)
3. **Click "Vorhersage generieren"**
4. View the results in the interactive chart and table

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
