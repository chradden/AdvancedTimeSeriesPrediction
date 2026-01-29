# Energie-Zeitreihen-Analyse & -Vorhersage ⚡🔋

**Advanced Time Series Prediction Project**

> Comprehensive comparison of time series forecasting methods applied to German energy data

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Production-success.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://www.docker.com/)
[![Avg R²](https://img.shields.io/badge/Avg%20R²-0.978-brightgreen.svg)]()

## 🚀 Quick Start

### Launch the Web Dashboard

```bash
docker-compose up
```

**Open**:
- 🎯 **Prognose-UI**: 
  - Localhost: http://localhost:8000/ui
  - Codespace: https://<codespace-name>-8000.app.github.dev/ui
- 📈 **Grafana Monitoring**: 
  - Localhost: http://localhost:3000 (admin/admin)
  - Codespace: https://<codespace-name>-3000.app.github.dev (admin/admin)
- 🔧 **API Docs**: 
  - Localhost: http://localhost:8000/docs
  - Codespace: https://<codespace-name>-8000.app.github.dev/docs

## 🎯 Project Results

| Dataset | Best Model | R² Score | MAE | MAPE | Status |
|---------|------------|----------|-----|------|--------|
| 🌊 Wind Offshore | XGBoost | **0.996** | 16 MW | 2.0% | 🏆 Spectacular |
| 🏭 Consumption | XGBoost | **0.996** | 484 MW | 0.9% | 🟢 Production |
| ☀️ Solar | XGBoost | **0.980** | 255 MW | 3.2% | 🟢 Production |
| 💨 Wind Onshore | XGBoost | **0.969** | 252 MW | 6.1% | 🟢 Production |
| 💰 Price | XGBoost | **0.952** | 7.25 €/MWh | 11.1% | 🟡 Research |

**🎉 Average R² across all datasets: 0.978** → Produktionsreife erreicht!

## 📊 Übersicht

Dieses Projekt analysiert Energiezeitreihen der deutschen Stromversorgung mit verschiedenen State-of-the-Art Forecasting-Methoden:

- **Datenquelle:** [SMARD](https://www.smard.de/home) (Bundesnetzagentur)
- **Zeitreihen:** Stromerzeugung (Solar, Wind), Verbrauch, Preise
- **Zeitraum:** 2022-2024 (3 Jahre stündliche Daten)
- **Ziel:** Vergleich von statistischen, ML- und Deep-Learning-Modellen

## 🗂️ Projektstruktur

```
energy-timeseries-project/
├── data/
│   ├── raw/              # Rohdaten von SMARD API (gecached)
│   ├── processed/        # Aufbereitete Daten
│   └── external/         # Zusätzliche Daten (Wetter, Feiertage)
├── notebooks/
│   ├── 01_data_exploration.ipynb       # ✅ Explorative Datenanalyse
│   ├── 02_data_preprocessing.ipynb     # ✅ Datenaufbereitung & Feature Engineering
│   ├── 03_baseline_models.ipynb        # ✅ Baseline-Modelle (Naive, Seasonal)
│   ├── 04_statistical_models.ipynb     # ✅ SARIMA, ETS
│   ├── 05_ml_tree_models.ipynb         # ✅ XGBoost, LightGBM, CatBoost
│   ├── 06_deep_learning_models.ipynb   # ✅ LSTM, GRU, Bi-LSTM
│   ├── 07_generative_models.ipynb      # ✅ VAE, GAN, DeepAR
│   ├── 08_advanced_models.ipynb        # ✅ TFT, N-BEATS
│   ├── 09_model_comparison.ipynb       # ✅ Vergleich aller Modelle
│   ├── 10_multi_series_analysis.ipynb  # ✅ 5 Zeitreihen parallel
│   ├── 11_xgboost_tuning.ipynb         # ✅ XGBoost Hyperparameter-Optimierung
│   ├── 12_llm_time_series_models.ipynb # ✅ Foundation Models (Chronos)
│   ├── 09_model_comparison.ipynb       # ✅ Finaler Modellvergleich
│   ├── 10_multi_series_analysis.ipynb  # ✅ Multi-Series Pipeline (alle 5 Datensätze)
│   └── 11_xgboost_tuning.ipynb         # ✅ Hyperparameter-Optimierung
├── src/
│   ├── data/             # Daten-Loading (SMARD API) & Preprocessing
│   ├── models/           # Model-Implementierungen
│   ├── visualization/    # Plotting-Funktionen
│   └── evaluation/       # Metriken & Evaluation (MAE, RMSE, R², MAPE)
├── results/
│   ├── figures/          # Plots & Visualisierungen
│   └── metrics/          # Performance-Metriken (CSV)
│       ├── RESULTS_SUMMARY.md                 # Zusammenfassung aller Ergebnisse
│       ├── INTERPRETATION_UND_NEXT_STEPS.md   # Interpretation & nächste Schritte
│       ├── PROJECT_COMPLETION_REPORT.md       # Finale Projekt-Dokumentation
│       └── multi_series_comparison_UPDATED.csv # Multi-Series Finale Ergebnisse
├── README.md
├── requirements.txt
└── PROJEKTPLAN_ENERGIE_ZEITREIHEN.md   # Detaillierter Plan
```

## 🚀 Quick Start

### 1. Installation

```bash
# Repository klonen
cd c:\Users\Christian\Coding\AdvancedTimeSeriesPrediction
cd energy-timeseries-project

# Virtual Environment erstellen (empfohlen)
python -m venv venv
.\venv\Scripts\activate  # Windows

# Dependencies installieren
pip install -r requirements.txt

# Jupyter Notebook starten
jupyter notebook
```

### 2. Erste Schritte

Öffne `notebooks/01_data_exploration.ipynb` und führe die Zellen aus!

Das Notebook wird:
- ✅ Daten von SMARD API laden (automatisches Caching)
- ✅ Explorative Datenanalyse durchführen
- ✅ Saisonalität & Trends visualisieren
- ✅ Stationaritätstests durchführen

## 📈 Geplante Modelle

### Baseline
- ✅ Naive Forecast
- ✅ Seasonal Naive
- ✅ Moving Average

### Statistische Modelle
- ✅ ARIMA
- ✅ SARIMA
- ✅ SARIMAX (mit exogenen Variablen)
- ✅ ETS (Exponential Smoothing)

### Machine Learning
- ✅ XGBoost
- ✅ LightGBM
- ✅ CatBoost
- ✅ Random Forest

## 📈 Implementierte Modelle

### ✅ Baseline Models
- Naive Forecast
- Seasonal Naive  
- Moving Average

### ✅ Statistische Modelle
- ARIMA / SARIMA
- SARIMAX (mit exogenen Variablen)
- ETS (Exponential Smoothing)

### ✅ Machine Learning (Winner 🏆)
- **XGBoost** → Best overall (R² = 0.978)
- LightGBM → Close second
- CatBoost
- Random Forest

### ✅ Deep Learning
- LSTM (Long Short-Term Memory)
- GRU (Gated Recurrent Unit)
- Bidirectional LSTM

### ✅ Generative Models
- Autoencoders für Anomalie-Erkennung
- VAEs (Variational Autoencoders)
- GANs (Generative Adversarial Networks)
- DeepAR (Probabilistische Vorhersagen)

### ✅ Advanced Deep Learning
- Temporal Fusion Transformer (TFT)
- N-BEATS
- N-HiTS

### ✅ Foundation Models (LLMs)
- **Chronos** (Amazon): T5-based zero-shot forecasting
- **TimeGPT** (Nixtla): GPT-ähnliche Architektur
- **Lag-Llama** (ServiceNow): Llama-basiert
- **Moirai** (Salesforce): Multi-Scale Transformer

**Ergebnis**: Foundation Models zeigen beeindruckende Zero-Shot-Fähigkeiten, aber bei domänenspezifischen Problemen mit reichlich Trainingsdaten sind XGBoost/LSTM noch überlegen (XGBoost: MAE=249MW vs. Chronos: MAE=4418MW). Hauptvorteil: Rapid Prototyping ohne Training.

## 📊 Evaluation-Metriken

Alle Modelle werden verglichen anhand von:

- **MAE** (Mean Absolute Error) → Primäre Metrik
- **RMSE** (Root Mean Squared Error) → Outlier-Sensitivität
- **R² Score** → Erklärte Varianz (0-1, höher = besser)
- **MAPE** (Mean Absolute Percentage Error) → Relative Fehler
- **Trainingszeit** → Effizienz
- **Inferenzzeit** → Produktionseinsatz

## 🔍 Wichtige Erkenntnisse

### Feature Engineering ist entscheidend
- **31 Features** entwickelt: Zeit-Features, zyklische Encodings, Lags (1h-7d), Rolling Stats
- 18 fehlende Features führten zu 15% Performance-Drop (R² 0.83 → 0.98)

### Test-Split-Strategie kritisch
- Naive "letzte 30 Tage" führte bei Wind Offshore zu R²=0.00 (100% Nullwerte im Test)
- **Smart Test Splits**: Datensatz-spezifische Perioden mit repräsentativer Verteilung
- Wind Offshore: Oct 2022 statt Jan 2024 → R² von 0.00 auf 0.996 🚀

### Model Performance
- **XGBoost dominiert**: Gewinnt bei allen 5 Datensätzen
- Deep Learning: Vergleichbare Accuracy, aber 10x längeres Training
- Statistical Models: Gut für Interpretation, schwächer bei Multivariaten Daten

### Data Quality Matters
- Wind Offshore hatte 9 Monate Downtime (Mai 2023 - Jan 2024)
- Automatische Datenqualitätsprüfung verhindert falsche Splits

## 🔧 Verwendete Tools & Libraries

### Daten & Preprocessing
- `pandas`, `numpy`, `scipy`
- `sklearn.preprocessing`
- `holidays` (deutsche Feiertage)

### Visualisierung
- `matplotlib`, `seaborn`, `plotly`

### Statistische Modelle
- `statsmodels` (ARIMA, SARIMA)
- `pmdarima` (auto_arima)

### Machine Learning
- `scikit-learn`
- `xgboost`, `lightgbm`, `catboost`

### Deep Learning
- `pytorch`, `tensorflow`
- `darts` (Forecasting-Framework)
- `pytorch-forecasting` (TFT)
- `neuralforecast` (N-BEATS, N-HiTS)

### Optimierung
- `optuna` (Hyperparameter-Tuning)

## 📚 Datenquellen

### Primäre Quelle: SMARD
API der Bundesnetzagentur: https://www.smard.de/home

**Verfügbare Zeitreihen:**
- ✅ Photovoltaik-Erzeugung
- ✅ Wind Onshore
- ✅ Wind Offshore
- ✅ Stromverbrauch Deutschland
- ✅ Day-Ahead Strompreise
- ✅ Gesamterzeugung
- Und weitere...

**Auflösung:** Stündlich, täglich, wöchentlich, monatlich

### Alternative Quellen
- [Energy-Charts](https://www.energy-charts.info/?l=de&c=DE) (Fraunhofer ISE)
- [Bundesnetzagentur Datenportal](https://www.bundesnetzagentur.de/DE/Fachthemen/Datenportal/start.html)

### Externe Daten (optional)
- Wetterdaten: [Open-Meteo](https://open-meteo.com/)
- Feiertage: Python `holidays` Library

## 🎯 Projektziele & Status

1. ✅ **Datenverständnis:** Tiefe explorative Analyse der Energiedaten  
2. ✅ **Methodenvergleich:** Systematischer Vergleich von 20+ Modellen  
3. ✅ **Best Practice:** Reproduzierbare, gut dokumentierte Analyse mit 11 Notebooks  
4. ✅ **Praktische Relevanz:** Produktionsreife Modelle für Energiesektor (R² > 0.95)  
5. ✅ **Technische Tiefe:** State-of-the-Art Feature Engineering & Smart Test Splits

**Status: PROJEKT ABGESCHLOSSEN** ✅

## 📝 Projekt-Verlauf & Lessons Learned

### Phase 1: Datenexploration & Baseline (Notebooks 01-03)
- [x] Projektstruktur erstellt
- [x] SMARD API-Integration (automatisches Caching)
- [x] Explorative Datenanalyse (Seasonalität, Trends, Stationarität)
- [x] Preprocessing & Feature Engineering (31 Features)
- [x] Train/Test/Validation Split
- [x] Baseline-Modelle (Naive, Seasonal, MA)

### Phase 2: Klassische ML & Stats (Notebooks 04-05)
- [x] Statistische Modelle (SARIMA, ETS)
- [x] ML Tree Models (XGBoost, LightGBM, CatBoost)
- [x] **Key Finding:** XGBoost bestes Modell für Solar (R² = 0.98)

### Phase 3: Deep Learning (Notebooks 06-08)
- [x] Grundlagen: LSTM, GRU, Bi-LSTM
- [x] Generative Models: VAE, GAN, DeepAR
- [x] Advanced: TFT, N-BEATS, N-HiTS
- [x] **Key Finding:** Vergleichbare Accuracy, aber 10x längeres Training

### Phase 4: Multi-Series & Optimization (Notebooks 09-11)
- [x] Model Comparison Solar (9 Model-Kategorien)
- [x] Multi-Series Analysis (alle 5 Datensätze)
- [x] Hyperparameter Tuning (XGBoost)

### Phase 5: Critical Debugging 🐛
**Problem 1:** Solar R² Drop (0.98 → 0.83)  
- **Root Cause:** 18 fehlende Features in Notebook 10  
- **Solution:** create_features() auf 31 Features erweitert  
- **Result:** R² = 0.98 ✅

**Problem 2:** Wind Offshore Catastrophic Failure (R² = 0.00)  
- **Root Cause:** Test-Split in 9-Monats-Downtime (100% Nullwerte)  
- **Solution:** Smart Test Split Strategy implementiert  
- **Result:** R² = 0.996 🚀  
- **Lesson:** Datenqualität > Algorithmus

### Phase 6: Production Deployment
- [x] Multi-Series Pipeline (run_complete_multi_series.py)
- [x] Alle 5 Datensätze mit finalen Features & Smart Splits
- [x] Comprehensive Documentation (3 Markdown Reports, 10 Debug Scripts)
- [x] **Result:** Avg R² = 0.978 across all datasets ✅

## 📂 Key Files & Documentation

### Notebooks (Execution Order)
1. [01_data_exploration.ipynb](notebooks/01_data_exploration.ipynb) - EDA
2. [02_data_preprocessing.ipynb](notebooks/02_data_preprocessing.ipynb) - Feature Engineering  
3. [03_baseline_models.ipynb](notebooks/03_baseline_models.ipynb) - Simple Baselines
4. [04_statistical_models.ipynb](notebooks/04_statistical_models.ipynb) - SARIMA, ETS
5. [05_ml_tree_models.ipynb](notebooks/05_ml_tree_models.ipynb) - XGBoost, LightGBM  
6. [06_deep_learning_models.ipynb](notebooks/06_deep_learning_models.ipynb) - LSTM, GRU
7. [07_generative_models.ipynb](notebooks/07_generative_models.ipynb) - VAE, GAN, DeepAR
8. [08_advanced_models.ipynb](notebooks/08_advanced_models.ipynb) - TFT, N-BEATS  
9. [09_model_comparison.ipynb](notebooks/09_model_comparison.ipynb) - Solar Comparison
10. [10_multi_series_analysis.ipynb](notebooks/10_multi_series_analysis.ipynb) - All 5 Datasets
11. [11_xgboost_tuning.ipynb](notebooks/11_xgboost_tuning.ipynb) - Hyperparameter Optimization

### Reports & Documentation
- [RESULTS_SUMMARY.md](results/metrics/RESULTS_SUMMARY.md) - Zusammenfassung aller Modell-Ergebnisse
- [INTERPRETATION_UND_NEXT_STEPS.md](results/metrics/INTERPRETATION_UND_NEXT_STEPS.md) - Interpretation & Roadmap
- [PROJECT_COMPLETION_REPORT.md](results/metrics/PROJECT_COMPLETION_REPORT.md) - Finale Projekt-Dokumentation mit Debugging-Details
- [SESSION_2_DEBUGGING.md](SESSION_2_DEBUGGING.md) - Detaillierte Debugging-Session
- [multi_series_comparison_UPDATED.csv](results/metrics/multi_series_comparison_UPDATED.csv) - Finale Ergebnisse

### Scripts
- [quickstart.py](quickstart.py) - Schneller Einstieg & Daten-Download
- [run_complete_multi_series.py](run_complete_multi_series.py) - Production Pipeline  
- 10 Debug/Validation Scripts (siehe PROJECT_COMPLETION_REPORT.md)

## 🚀 Reproduktion der Ergebnisse

```bash
# 1. Environment Setup
cd energy-timeseries-project
pip install -r requirements.txt

# 2. Data Download
python quickstart.py  # Lädt alle 5 Datensätze von SMARD

# 3a. Run Full Pipeline (empfohlen)
python run_complete_multi_series.py

# 3b. OR: Run Notebooks sequentiell
jupyter notebook
# Notebooks 01-11 der Reihe nach ausführen
```

**Expected Runtime:**  
- Full Pipeline: ~30-45 Minuten  
- Individual Notebooks: 5-10 Minuten each  
- Deep Learning Notebooks: 15-20 Minuten each

## 💡 Key Takeaways

1. **Feature Engineering > Model Complexity**  
   31 sorgfältig konstruierte Features schlagen komplexe Deep Learning Modelle

2. **Data Quality is King**  
   Smart Test Splits & Datenvalidierung sind kritisch für valide Ergebnisse

3. **XGBoost ist der praktische Gewinner**  
   Best Performance + Fast Training + Easy Deployment = Production Ready

4. **Deep Learning hat seinen Platz**  
   Wenn Daten >100k und Komplexität hoch → LSTM/TFT können lohnen

5. **Documentation Matters**  
   10 Debug-Scripts + 3 Reports ermöglichen vollständige Reproduzierbarkeit

## 📞 Kontakt & Weiterführendes

**Datenquelle:** [SMARD - Bundesnetzagentur](https://www.smard.de/home)  
**Energy Charts:** [Fraunhofer ISE](https://www.energy-charts.info/?l=de&c=DE)

---

**Projekt-Status:** ✅ PRODUCTION READY  
**Letzte Aktualisierung:** 2026-01-22  
**Durchschnittliche R²-Score:** 0.978
- [ ] Hyperparameter-Tuning
- [ ] Finale Dokumentation & Visualisierung

## 🔗 Referenzen

### Kursmaterial
- [TimeSeriesPrediction Repository](../TimeSeriesPrediction)
- Week02: SARIMA, Week04: Trees, Week05: LSTM
- Week08: VAEs, GANs, DeepAR ✅
- Week09: Transformers, Week10: N-BEATS

### Erfolgreiche Projekte
- [Energy Timeseries Project](https://github.com/Timson1235/energy-timeseries-project) (VDE Prize Winner)
- [Solar Prediction](https://github.com/AnnaValentinaHirsch/solar-prediction)
- [German Energy Analysis](https://github.com/worldmansist/German-energy-Time-Series-analysis-)



## 📄 Lizenz

Dieses Projekt ist für Bildungszwecke erstellt.

---

**Status:** ✅ Alle 9 Notebooks erstellt - Bereit zum Ausführen!  
**Letzte Aktualisierung:** 2026-01-21
