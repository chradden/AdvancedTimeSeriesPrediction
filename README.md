# ⚡ Advanced Time Series Prediction for Energy Data

> **Comprehensive Time Series Forecasting Project**: Comparative analysis of statistical, machine learning, and deep learning methods for German energy market prediction.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Production-success.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://www.docker.com/)
[![Avg R²](https://img.shields.io/badge/Avg%20R²-0.978-brightgreen.svg)]()

---

## 📋 Schnellzugang

**📌 Wichtigste Dokumente:**
- **[`PROJECT_OVERVIEW.md`](PROJECT_OVERVIEW.md)** - Komplette Projektübersicht & aktuelle Strategie
- [`QUICKSTART.md`](QUICKSTART.md) - Schnelleinstieg in 5 Minuten
- [`energy-timeseries-project/README.md`](energy-timeseries-project/README.md) - Technische Details

---

## 🎯 Project Overview

Production-ready forecasting system for the German energy market, comparing 15+ different modeling approaches to identify optimal forecasting methods for different energy types.

### 📊 Performance Results

| Energy Type | Best Model | R² Score | MAE | MAPE | Status |
|-------------|-----------|----------|-----|------|--------|
| 🌊 Wind Offshore | XGBoost | **0.996** | 16 MW | 2.0% | 🏆 Production |
| 🏭 Consumption | XGBoost | **0.996** | 484 MW | 0.9% | ✅ Production |
| ☀️ Solar | XGBoost | **0.980** | 255 MW | 3.2% | ✅ Production |
| 💨 Wind Onshore | XGBoost | **0.969** | 252 MW | 6.1% | ✅ Production |
| 💰 Price | XGBoost | **0.952** | 7.25 €/MWh | 11.1% | 🔬 Research |

**Average R² across all datasets: 0.978** 🎉

## 🚀 Quick Start

### Option 1: Docker (Recommended)

```bash
cd energy-timeseries-project
docker-compose up
```

Then open: **http://localhost:8000/ui**

### Option 2: Local Development

```bash
cd energy-timeseries-project
pip install -r requirements.txt
python api_simple.py
```

## 🏗️ Project Structure

```
AdvancedTimeSeriesPrediction/
├── energy-timeseries-project/      # Main project directory
│   ├── api_simple.py               # Production API (FastAPI)
│   ├── docker-compose.yml          # Docker orchestration
│   ├── Dockerfile                  # Container definition
│   ├── data/                       # Energy datasets
│   │   ├── raw/                    # Original SMARD data
│   │   └── processed/              # Preprocessed datasets
│   ├── notebooks/                  # Analysis notebooks (16 notebooks)
│   │   ├── 01_data_exploration.ipynb
│   │   ├── 05_ml_tree_models.ipynb
│   │   ├── 06_deep_learning_models.ipynb
│   │   └── ...
│   ├── src/                        # Source code modules
│   │   ├── data/                   # Data loading & processing
│   │   ├── models/                 # Model implementations
│   │   ├── evaluation/             # Metrics & evaluation
│   │   └── visualization/          # Plotting utilities
│   ├── results/                    # Model outputs & figures
│   ├── static/                     # Web UI
│   ├── scripts/                    # Utility scripts
│   └── docs/                       # Documentation
└── README.md                       # This file
```

## 🔬 Methods Implemented

### Statistical Models
- Naive & Seasonal Naive baselines
- SARIMA (Seasonal ARIMA)
- ETS (Exponential Smoothing)
- Prophet (Facebook)

### Machine Learning
- **XGBoost** ⭐ (Best performer)
- LightGBM
- CatBoost
- Random Forest

### Deep Learning
- LSTM & GRU networks
- Bi-directional LSTM
- Temporal Fusion Transformer (TFT)
- N-BEATS
- DeepAR (Probabilistic)

### Advanced Methods
- Ensemble methods (Stacking, Voting)
- Chronos (Pretrained LLM for time series)
- Multivariate forecasting
- External weather features integration

## 📊 Data Source

**SMARD API** (Bundesnetzagentur - German Federal Network Agency)
- **Period**: 2022-2024 (3 years)
- **Resolution**: Hourly data
- **Variables**: 
  - Solar generation (MW)
  - Wind offshore/onshore generation (MW)
  - Energy consumption (MW)
  - Day-ahead prices (€/MWh)

## 🎨 Web Dashboard

Interactive web interface for real-time forecasting:

**Features:**
- 📈 Interactive charts (Chart.js)
- 🎛️ Multiple energy type selection
- ⏱️ Configurable forecast horizons (1-168 hours)
- 📊 Model performance metrics
- 📋 Detailed prediction tables

**Access:** http://localhost:8000/ui

## 🔧 API Endpoints

### Forecasting
- `POST /api/predict/solar` - Solar energy forecast
- `POST /api/predict/wind_offshore` - Wind offshore forecast
- `POST /api/predict/wind_onshore` - Wind onshore forecast
- `POST /api/predict/consumption` - Consumption forecast
- `POST /api/predict/price` - Price forecast

### System
- `GET /health` - Health check
- `GET /docs` - API documentation (Swagger UI)

### Example Request

```python
import requests

response = requests.post(
    "http://localhost:8000/api/predict/solar",
    json={"hours": 24}
)
data = response.json()
print(f"Predictions: {data['predictions']}")
```

## 📈 Key Findings

1. **XGBoost dominates** across all energy types
2. **Feature engineering is crucial**: Time features, lags, rolling statistics
3. **Deep learning** shows promise but requires more data/tuning
4. **Ensemble methods** provide marginal improvements over single models
5. **Wind prediction** benefits from weather data integration

## 🛠️ Technologies

- **Python 3.10+**
- **FastAPI** - Modern API framework
- **XGBoost, LightGBM, CatBoost** - Gradient boosting
- **PyTorch** - Deep learning
- **Chart.js** - Interactive visualizations
- **Docker & Docker Compose** - Containerization
- **Pandas, NumPy** - Data manipulation
- **Scikit-learn** - ML utilities

## 📚 Notebooks Overview

| Notebook | Description | Status |
|----------|-------------|--------|
| 01 | Data Exploration | ✅ Complete |
| 02 | Data Preprocessing | ✅ Complete |
| 03 | Baseline Models | ✅ Complete |
| 04 | Statistical Models | ✅ Complete |
| 05 | ML Tree Models | ✅ Complete |
| 06 | Deep Learning | ✅ Complete |
| 07 | Generative Models | ✅ Complete |
| 08 | Advanced Models | ✅ Complete |
| 09 | Model Comparison | ✅ Complete |
| 10 | Multi-Series Analysis | ✅ Complete |
| 11 | XGBoost Tuning | ✅ Complete |
| 12 | LLM Time Series | ✅ Complete |
| 13 | Ensemble Methods | ✅ Complete |
| 14 | Multivariate Forecasting | ✅ Complete |
| 15 | External Weather Features | ✅ Complete |
| 16 | Chronos Finetuning | ✅ Complete |

## 🎯 Future Enhancements

- [ ] Real-time data integration via SMARD API
- [ ] Probabilistic forecasting (prediction intervals)
- [ ] Automated model retraining pipeline
- [ ] Multi-horizon forecasting optimization
- [ ] Model explainability (SHAP values)
- [ ] Production monitoring & alerting

## 📝 Documentation

Detailed documentation available in `energy-timeseries-project/docs/`:
- `FINAL_PROJECT_SUMMARY.md` - Complete project summary
- `FORECAST_24H_GUIDE.md` - 24-hour forecasting guide
- Session logs and debugging notes

## 🤝 Contributing

This is an academic project, but suggestions and improvements are welcome!

## 📄 License

Academic project for educational purposes.

## 👤 Author

**Christian Radden**
- GitHub: [@chradden](https://github.com/chradden)

---

⭐ **Star this repo if you find it useful!**

🔗 **Live Demo**: [Access the dashboard](http://localhost:8000/ui) after starting the application
