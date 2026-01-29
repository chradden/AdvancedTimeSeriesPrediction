# 📁 Project Structure

## Overview

```
AdvancedTimeSeriesPrediction/
├── 📄 README.md                    # Main project documentation
├── 📄 QUICKSTART.md                # Quick start guide
├── 📄 PROJEKTPLAN_ENERGIE_ZEITREIHEN.md  # German project plan
└── energy-timeseries-project/      # Main project directory
    ├── 🐳 docker-compose.yml       # Docker orchestration
    ├── 🐳 Dockerfile               # Container definition
    ├── 📄 README.md                # Project-specific README
    │
    ├── 📊 data/                    # Data directory
    │   ├── raw/                    # Original SMARD API data
    │   └── processed/              # Preprocessed datasets
    │
    ├── 📓 notebooks/               # Jupyter notebooks (16 total)
    │   ├── 01_data_exploration.ipynb
    │   ├── 02_data_preprocessing.ipynb
    │   ├── 03_baseline_models.ipynb
    │   ├── 04_statistical_models.ipynb
    │   ├── 05_ml_tree_models.ipynb          # XGBoost, LightGBM, CatBoost
    │   ├── 06_deep_learning_models.ipynb    # LSTM, GRU
    │   ├── 07_generative_models.ipynb       # VAE, GAN
    │   ├── 08_advanced_models.ipynb         # TFT, N-BEATS
    │   ├── 09_model_comparison.ipynb        # Comparative analysis
    │   ├── 10_multi_series_analysis.ipynb   # Multi-dataset analysis
    │   ├── 11_xgboost_tuning.ipynb          # Hyperparameter tuning
    │   ├── 12_llm_time_series_models.ipynb  # LLM-based models
    │   ├── 13_ensemble_methods.ipynb        # Stacking, voting
    │   ├── 14_multivariate_forecasting.ipynb
    │   ├── 15_external_weather_features.ipynb
    │   └── 16_chronos_finetuning.ipynb
    │
    ├── 💻 src/                     # Source code modules
    │   ├── __init__.py
    │   ├── data/                   # Data loading & processing
    │   ├── models/                 # Model implementations
    │   ├── evaluation/             # Metrics & evaluation
    │   └── visualization/          # Plotting utilities
    │
    ├── 📊 results/                 # Model outputs
    │   ├── figures/                # Generated plots
    │   └── metrics/                # Performance metrics
    │
    ├── 🌐 static/                  # Web UI
    │   └── index.html              # Dashboard interface
    │
    ├── 📜 scripts/                 # Utility scripts
    │   ├── analyze_*.py            # Analysis scripts
    │   ├── debug_*.py              # Debugging tools
    │   ├── test_*.py               # Test scripts
    │   └── validate_*.py           # Validation tools
    │
    ├── 📚 docs/                    # Documentation
    │   ├── FINAL_PROJECT_SUMMARY.md
    │   ├── FORECAST_24H_GUIDE.md
    │   ├── LSTM_MAPE_ANALYSE.md
    │   ├── PROJECT_COMPLETION_REPORT.md
    │   ├── SESSION_*.md            # Session logs
    │   └── PRÄSENTATION_*.md       # Presentation materials
    │
    ├── 🚀 API Files
    │   ├── api_simple.py           # Production API (FastAPI)
    │   ├── api.py                  # Full-featured API
    │   └── api_client_example.py   # API usage examples
    │
    ├── 🏃 Run Scripts
    │   ├── quickstart.py           # Quick demo script
    │   ├── run_chronos_forecasting.py
    │   ├── run_complete_multi_series.py
    │   ├── run_deep_learning_retrain.py
    │   ├── run_ensemble_methods.py
    │   └── run_xgboost_tuning.py
    │
    └── 📦 Dependencies
        ├── requirements.txt        # Full dependencies
        └── requirements-api.txt    # Minimal API dependencies
```

## 📝 Key Files

### Entry Points

| File | Purpose | Usage |
|------|---------|-------|
| `api_simple.py` | Production API | `python api_simple.py` |
| `docker-compose.yml` | Container orchestration | `docker-compose up` |
| `quickstart.py` | Quick demo | `python quickstart.py` |

### Configuration

| File | Purpose |
|------|---------|
| `requirements.txt` | Full Python dependencies |
| `requirements-api.txt` | Minimal API dependencies |
| `Dockerfile` | Container definition |
| `docker-compose.yml` | Multi-container setup |

### Documentation

| File | Description |
|------|-------------|
| `README.md` (root) | Main project overview |
| `QUICKSTART.md` | Quick start guide |
| `docs/FINAL_PROJECT_SUMMARY.md` | Complete project summary |
| `docs/FORECAST_24H_GUIDE.md` | 24-hour forecasting guide |

## 🔍 Directory Details

### `/data`
- **raw/**: Original CSV files from SMARD API (cached)
- **processed/**: Train/val/test splits, scaled data

### `/notebooks`
16 Jupyter notebooks covering the complete ML pipeline from EDA to deployment

### `/src`
Reusable Python modules:
- `data/`: Data loaders, preprocessors
- `models/`: Model wrappers and custom implementations
- `evaluation/`: Metrics calculation
- `visualization/`: Plotting functions

### `/results`
Generated outputs:
- `figures/`: PNG/PDF plots
- `metrics/`: JSON performance metrics

### `/static`
Web interface files (HTML, CSS, JavaScript)

### `/scripts`
Standalone utility scripts for analysis, debugging, and validation

### `/docs`
Project documentation, session logs, and presentation materials

## 🚀 Typical Workflow

1. **Explore Data**: `notebooks/01_data_exploration.ipynb`
2. **Preprocess**: `notebooks/02_data_preprocessing.ipynb`
3. **Build Models**: `notebooks/05-08_*.ipynb`
4. **Compare**: `notebooks/09_model_comparison.ipynb`
5. **Deploy**: `docker-compose up` → API + Web UI

## 📦 Dependencies

### Production (API)
- FastAPI
- Uvicorn
- XGBoost
- Pandas, NumPy

### Development (Full)
- All production dependencies
- Jupyter
- PyTorch
- Scikit-learn
- Matplotlib, Seaborn
- Prophet, statsmodels
- LightGBM, CatBoost

## 🔗 Quick Links

- [Main README](../README.md)
- [Quick Start](../QUICKSTART.md)
- [API Documentation](http://localhost:8000/docs) (when running)
- [Web Dashboard](http://localhost:8000/ui) (when running)
