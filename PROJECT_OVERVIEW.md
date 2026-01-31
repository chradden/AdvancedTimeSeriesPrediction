# Advanced Time Series Prediction - Projektübersicht

**Stand:** Januar 2026  
**Status:** Produktionsreif mit erweiterten Pipelines

## 📊 Projektbeschreibung

Umfassendes Machine-Learning-Projekt zur Vorhersage von Energiezeitreihen (Solar, Wind Offshore, Wind Onshore, Consumption, Price) mit mehreren Modellkategorien und produktionsreifer API.

## 🎯 Aktuelle Strategie

### 1. **Automatisierte Pipelines** (Empfohlen für schnelle Durchläufe)
Alle Datenquellen haben jetzt standardisierte Extended Pipelines mit 9 Phasen:

**Verfügbare Skripte:**
- `scripts/run_solar_extended_pipeline.py`
- `scripts/run_wind_offshore_extended_pipeline.py`
- `scripts/run_wind_onshore_extended_pipeline.py`
- `scripts/run_consumption_extended_pipeline.py`
- `scripts/run_price_extended_pipeline.py`

**9 Phasen pro Pipeline:**
1. Data Exploration
2. Preprocessing & Feature Engineering
3. Baseline Models (Naive, Seasonal Naive, Moving Average, Drift, Mean)
4. Statistical Models (ARIMA, SARIMA, ETS)
5. ML Tree Models (Random Forest, XGBoost, LightGBM, CatBoost)
6. Deep Learning (RNN, LSTM, GRU)
7. Generative Models (Overview, in Notebooks detailliert)
8. Advanced Models (N-BEATS, N-HiTS)
9. Final Comparison & Visualizations

**Vorteile:**
- ✅ Reproduzierbar
- ✅ Schneller als Notebooks
- ✅ Automatische Metrics & Plots
- ✅ CSV-Export für alle Modelle

### 2. **Jupyter Notebooks** (Für detaillierte Analysen)
Für jede Datenquelle gibt es 9 thematische Notebooks:

**Struktur (am Beispiel Solar):**
```
notebooks/solar/
├── 01_data_exploration.ipynb
├── 02_data_preprocessing.ipynb
├── 03_baseline_models.ipynb
├── 04_statistical_models.ipynb
├── 05_ml_tree_models.ipynb
├── 06_deep_learning_models.ipynb
├── 07_generative_models.ipynb
├── 08_advanced_models.ipynb
└── 09_model_comparison.ipynb
```

**Weitere Serien:**
- `notebooks/wind_offshore/` (5 Notebooks)
- `notebooks/wind_onshore/` (9 Notebooks)
- `notebooks/price/` (9 Notebooks)

### 3. **Production API**
FastAPI-basierte REST-API für 24h-Forecasts:

**Dateien:**
- `api.py` - Haupt-API (empfohlen)
- `api_simple.py` - Vereinfachte Version
- `api_client_example.py` - Client-Beispiele

**Endpoints:**
- `POST /predict/solar` - 24h Solar-Forecast
- `POST /predict/wind_offshore` - 24h Wind-Forecast
- `POST /predict/consumption` - 24h Verbrauchs-Forecast
- `GET /health` - Health Check

**Start:**
```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

### 4. **Monitoring & Dashboards**
- Prometheus + Grafana Setup
- Echtzeit-Metriken
- Visualisierung von Predictions vs. Actuals

## 📁 Projektstruktur

```
energy-timeseries-project/
├── data/
│   ├── raw/                    # Original SMARD-Daten
│   └── processed/              # Aufbereitete Train/Val/Test-Splits
├── scripts/                    # Automatisierte Pipelines
│   ├── run_*_extended_pipeline.py
│   ├── run_*_advanced_testing.py
│   └── test_*.py
├── notebooks/                  # Jupyter-Notebooks pro Serie
│   ├── solar/
│   ├── wind_offshore/
│   ├── wind_onshore/
│   └── price/
├── src/                        # Source-Code-Module
│   ├── data/                   # Datenlade- & Preprocessing-Tools
│   ├── models/                 # Modellimplementierungen
│   ├── evaluation/             # Metriken & Evaluation
│   └── visualization/          # Plot-Funktionen
├── results/
│   ├── figures/                # Generierte Plots
│   └── metrics/                # CSV/JSON-Ergebnisse
├── archive/                    # Archivierte alte Entwicklungen
│   ├── old_scripts/            # Debug-/Analyse-Skripte
│   ├── old_docs/               # Session-Logs
│   └── old_root_files/         # Veraltete Root-Skripte
├── monitoring/                 # Prometheus/Grafana-Configs
├── docs/                       # Aktuelle Dokumentation
├── api.py                      # Production API
└── requirements.txt
```

## 🚀 Quick Start

### Option 1: Automatisierte Pipeline ausführen
```bash
cd energy-timeseries-project
python scripts/run_solar_extended_pipeline.py
```

Ergebnisse:
- `results/metrics/solar_all_models_extended.csv`
- `results/figures/solar_extended_09_comparison.png`

### Option 2: Notebooks interaktiv
```bash
jupyter lab
# Öffne notebooks/solar/01_data_exploration.ipynb
```

### Option 3: API starten
```bash
cd energy-timeseries-project
uvicorn api:app --reload
# Öffne http://localhost:8000/docs
```

## 📈 Ergebnisse & Performance

**Best Models (Stand: Januar 2026):**
- **Solar:** LightGBM (R² ≈ 0.98, RMSE ≈ 1000 MW)
- **Wind Offshore:** LightGBM (R² ≈ 0.85)
- **Wind Onshore:** XGBoost/LightGBM (R² ≈ 0.92)
- **Price:** LightGBM (R² ≈ 0.98, RMSE ≈ 9 EUR/MWh)

Details: `results/metrics/*_all_models_extended.csv`

## 🛠️ Technologie-Stack

**Core Libraries:**
- pandas, numpy, scipy
- scikit-learn (1.7+)
- xgboost, lightgbm, catboost
- statsmodels, pmdarima
- torch, tensorflow/keras
- darts (N-BEATS, N-HiTS)
- FastAPI, uvicorn

**Visualisierung:**
- matplotlib, seaborn, plotly
- Grafana + Prometheus

## 📚 Wichtige Dokumentation

**Hauptdokumente:**
- `README.md` - Projekt-README
- `QUICKSTART.md` - Schnelleinstieg
- `MASTERPLAN.md` - Gesamtstrategie
- `STRUCTURE.md` - Detaillierte Struktur

**In `docs/`:**
- `FINAL_PROJECT_SUMMARY.md` - Finaler Projektbericht
- `PROJECT_COMPLETION_REPORT.md` - Abschlussbericht
- `GRAFANA_DASHBOARD_GUIDE_DE.md` - Dashboard-Anleitung
- `MONITORING_SETUP.md` - Monitoring-Setup
- `REALTIME_MONITORING_GUIDE.md` - Echtzeit-Monitoring

**In `notebooks/`:**
- `README.md` - Notebook-Übersicht
- `WO_SIND_DIE_ERGEBNISSE.md` - Wo finde ich Ergebnisse?
- `RESULTS_VIEWER.ipynb` - Interaktiver Ergebnis-Viewer

## 🗂️ Archivierte Entwicklungen

Alte Debug-/Analyse-Skripte und Session-Logs befinden sich in:
- `energy-timeseries-project/archive/`
- `archive_root/` (Root-Level)

Diese sind für die produktive Nutzung nicht mehr relevant, wurden aber für die Historie bewahrt.

## 🎓 Akademischer Hintergrund

Dieses Projekt kombiniert modernste Zeitreihen-Methoden:
- Classical Statistics (ARIMA, ETS)
- Tree Boosting (XGBoost, LightGBM, CatBoost)
- Deep Learning (LSTM, GRU)
- Neural Architectures (N-BEATS, N-HiTS)
- Foundation Models (Chronos - experimentell)

## 📞 Nächste Schritte

1. **Für neue Analysen:** Nutze `scripts/run_*_extended_pipeline.py`
2. **Für Experimente:** Kopiere & modifiziere Notebooks
3. **Für Production:** Nutze `api.py` mit Docker/Docker-Compose
4. **Für Monitoring:** Starte Grafana-Stack mit `start_monitoring.sh`

---

**Letzte Aktualisierung:** 31. Januar 2026
