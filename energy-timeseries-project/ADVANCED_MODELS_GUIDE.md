# 🚀 Advanced Models Guide - Erweiterte Modelle

**Datum**: 1. Februar 2026  
**Status**: ✅ Bereit für Experimente

---

## 📋 Übersicht

Dieses Dokument beschreibt die **erweiterten Modellierungs-Ansätze** für die Energiezeitreihen-Prognose, die über die Basis-Modelle (RF, LSTM, XGBoost) hinausgehen.

---

## 🎯 Neue Notebooks

### 1. 🔥 **Extended Deep Learning (Google Colab GPU)**
**File**: `scripts/LSTM_Optimization_Extended_Colab.ipynb`

**Modelle**:
- ✅ **GRU** - Gated Recurrent Unit (schneller als LSTM!)
- ✅ **Bi-LSTM** - Bidirectional LSTM
- ✅ **Autoencoder** - Dimensionsreduktion + Forecasting
- ✅ **VAE** - Variational Autoencoder (Unsicherheitsschätzung)
- ✅ **N-BEATS** - Neural Basis Expansion (State-of-the-Art)
- ✅ **N-HiTS** - Hierarchical Interpolation
- ✅ **DeepAR** - Amazon's probabilistisches Modell
- ✅ **TFT** - Temporal Fusion Transformer (Google Research)
- ⚠️ **TimeGAN** - Generative Adversarial Network (optional, experimentell)

**Rechenzeit (GPU T4)**:
- Schnell (<5 min): LSTM, GRU, Bi-LSTM, Autoencoder, VAE
- Mittel (5-15 min): N-BEATS, N-HiTS, DeepAR
- Langsam (15-45 min): TFT, TimeGAN

**Setup**:
```python
# In Colab: Runtime → Change runtime type → GPU (T4 empfohlen)
SERIES_NAME = 'solar'  # Ändern für andere Zeitreihen

# Model Selection
RUN_BASIC = True          # LSTM, GRU, Bi-LSTM
RUN_GENERATIVE = True     # Autoencoder, VAE
RUN_GAN = False           # TimeGAN (experimentell)
RUN_ADVANCED = True       # N-BEATS, N-HiTS
RUN_PROBABILISTIC = True  # DeepAR
RUN_TFT = False           # TFT (30-45 min!)
```

**Output**: `results/metrics/deep_learning_extended_{series_name}.csv`

---

### 2. 📊 **Multivariate Zeitreihenanalyse (Codespace)**
**File**: `notebooks/multivariate_VAR_VECM_analysis.ipynb`

**Modelle**:
- ✅ **VAR** - Vector Autoregression (Standard)
- ✅ **VECM** - Vector Error Correction Model (bei Kointegration)
- ✅ **VARMA** - Vector ARMA (mit MA-Komponente)
- ✅ **Granger Causality Tests** - Kausalitätsanalyse

**Tests**:
- 🧪 Stationaritätstests (ADF, KPSS)
- 🧪 Kointegrations-Test (Johansen)
- 🧪 Granger Causality Matrix

**Warum multivariate Verfahren?**
Unsere Energiezeitreihen sind **stark gekoppelt**:
- ☀️ Solar → 💰 Price (viel Sonne = niedriger Preis)
- 💨 Wind → 💰 Price (viel Wind = niedriger Preis)
- ☀️ Solar + 💨 Wind → 🏭 Consumption

**VAR/VECM** modellieren diese **Cross-Effects**!

**Vorteile**:
- 📊 Modelliert interdependenzen
- 🔍 Kausalität testbar
- 💡 Ökonomisch interpretierbar
- 🎯 Gut für Policy-Analysen

**Setup**:
```bash
# Im Codespace ausführen (CPU reicht)
# Keine GPU nötig!
```

---

### 3. 🌊 **Zeitreihen-spezifische Notebooks (Colab)**

Vorbereitet für alle Zeitreihen:
- `scripts/LSTM_Optimization_Colab_wind_offshore.ipynb` ✅
- `scripts/LSTM_Optimization_Colab_wind_onshore.ipynb` (in Arbeit)
- `scripts/LSTM_Optimization_Colab_price.ipynb` (in Arbeit)
- `scripts/LSTM_Optimization_Colab_consumption.ipynb` (in Arbeit)

**Gleiche Modelle wie Extended Edition**, aber optimiert für spezifische Zeitreihe.

---

## 📊 Ergebnisse: Solar (Google Colab)

| Modell | R² | RMSE (MW) | MAE (MW) | Training Zeit |
|--------|-----|-----------|----------|---------------|
| **Bi-LSTM** ✅ | **0.9955** | - | - | ~30s |
| **Baseline LSTM** | **0.9934** | - | - | ~25s |
| **Autoencoder** | **0.9515** | - | - | ~40s |
| **VAE** | **0.9255** | - | - | ~60s |
| **N-BEATS** ⚠️ | -18.93 | 23,316 | 16,348 | ~977s |
| **N-HiTS** ⚠️ | -4.22 | 11,930 | 8,211 | ~138s |

**Erkenntnisse**:
- ✅ **Bi-LSTM** erreicht beste Performance (R²=0.9955)
- ✅ **GPU-Beschleunigung**: 30-50x schneller als CPU
- ⚠️ **N-BEATS/N-HiTS** zeigen negative R² - möglicherweise Skalierungsprobleme
- 💡 **Random Forest (R²=0.9994)** bleibt dennoch bestes Gesamtmodell

---

## 🔄 Workflow-Empfehlung

### Phase 1: Basis-Experimente (Colab)
1. Starte mit **Extended Colab Notebook**
2. Aktiviere nur schnelle Modelle:
   ```python
   RUN_BASIC = True          # LSTM, GRU, Bi-LSTM
   RUN_GENERATIVE = True     # Autoencoder, VAE
   RUN_ADVANCED = True       # N-BEATS, N-HiTS
   RUN_TFT = False           # Zunächst überspringen
   ```
3. Laufzeit: ~10-15 Minuten
4. Evaluiere Ergebnisse

### Phase 2: State-of-the-Art (Colab)
1. Falls Zeit/Ressourcen verfügbar:
   ```python
   RUN_TFT = True            # Temporal Fusion Transformer
   RUN_GAN = True            # TimeGAN (experimentell)
   ```
2. Laufzeit: +30-45 Minuten
3. Vergleiche mit Basis-Modellen

### Phase 3: Multivariate Analyse (Codespace)
1. Führe `multivariate_VAR_VECM_analysis.ipynb` aus
2. Analysiere Granger-Kausalitäten
3. Teste VAR vs. VECM
4. Laufzeit: ~5-10 Minuten

### Phase 4: Vergleich & Dokumentation
1. Vergleiche alle Ansätze:
   - **Univariate** (RF, LSTM, GRU)
   - **Advanced DL** (N-BEATS, TFT, DeepAR)
   - **Multivariate** (VAR, VECM)
2. Dokumentiere in `PHASE2_EVALUATION_SUMMARY.md`
3. Erstelle finale Empfehlungen

---

## 🎯 Modell-Auswahl-Matrix

| Kriterium | Empfohlenes Modell | Begründung |
|-----------|-------------------|------------|
| **Höchste Genauigkeit** | Random Forest, Bi-LSTM | R² > 0.99 |
| **Schnellste Inferenz** | GRU, Linear Regression | <1ms pro Vorhersage |
| **Unsicherheitsschätzung** | DeepAR, VAE | Probabilistische Outputs |
| **Interpretierbarkeit** | VAR, VECM | Ökonomisch klar |
| **Kausalitätsanalyse** | VAR + Granger Tests | Cross-Series Effects |
| **State-of-the-Art** | TFT, N-BEATS | Neueste Forschung |
| **Anomalieerkennung** | Autoencoder, VAE | Reconstruction Error |
| **Produktionsreife** | Random Forest, LightGBM | Robust, schnell, stabil |

---

## 🔍 Fehlende Modelle & Limitationen

### Noch nicht implementiert:
- ❌ **Chronos** - Zu groß (mehrere GB), benötigt viel RAM
- ❌ **TimeGAN** - Sehr experimentell, komplex
- ❌ **Informer** - Transformer für lange Sequenzen
- ❌ **PatchTST** - State-of-the-Art (2023)

### Machbar, aber nicht priorisiert:
- ⚠️ **Prophet** - Facebook's Tool (bereits getestet, schlecht performt)
- ⚠️ **ARCH/GARCH** - Für Volatilität, nicht Forecasting
- ⚠️ **Wavelet Transform** - Feature Engineering, kein Modell

---

## 💡 Wichtige Erkenntnisse

### 1. GPU-Beschleunigung ist kritisch
- LSTM/GRU: **30-50x** schneller auf GPU
- N-BEATS/N-HiTS: Nur auf GPU praktikabel
- TFT: GPU **essentiell** (sonst Stunden!)

### 2. Multivariate ≠ Bessere Accuracy
- VAR/VECM haben oft **niedrigere R²** als univariate RF
- **ABER**: Modellieren Kausalitäten, ökonomisch wertvoller!

### 3. Komplexität ≠ Performance
- Einfache Modelle (RF, GRU) oft **besser** als komplexe (N-BEATS)
- Problem: Daten-Skalierung, Hyperparameter-Tuning

### 4. Negative R² bei N-BEATS/N-HiTS
- Wahrscheinliche Ursachen:
  - Falsche Daten-Skalierung
  - Zu kleine Trainingsdaten
  - Hyperparameter nicht optimal
- **Fix**: Mehr Tuning, andere Scaler (MinMaxScaler?)

---

## 📚 Referenzen & Literatur

### Papers:
1. **N-BEATS**: Oreshkin et al. (2019) - "N-BEATS: Neural basis expansion analysis for interpretable time series forecasting"
2. **TFT**: Lim et al. (2021) - "Temporal Fusion Transformers for interpretable multi-horizon time series forecasting"
3. **DeepAR**: Salinas et al. (2020) - "DeepAR: Probabilistic forecasting with autoregressive recurrent networks"
4. **VAR**: Sims (1980) - "Macroeconomics and Reality"
5. **VECM**: Johansen (1988) - "Statistical analysis of cointegration vectors"

### Code/Tools:
- **Darts**: https://github.com/unit8co/darts
- **GluonTS**: https://ts.gluon.ai/
- **statsmodels**: https://www.statsmodels.org/
- **PyTorch Forecasting**: https://pytorch-forecasting.readthedocs.io/

---

## ✅ Next Steps

1. ✅ **Solar Extended Colab** - Ergebnisse in PHASE2_EVALUATION_SUMMARY.md ✅
2. 🔄 **Alle Zeitreihen** - Extended Colab für Wind/Price/Consumption ausführen
3. 📊 **Multivariate Analyse** - VAR/VECM im Codespace testen
4. 📈 **Vergleichstabelle** - Alle Modelle über alle Zeitreihen
5. 🎯 **Produktionsempfehlung** - Finale Model Selection

---

**Viel Erfolg mit den Advanced Models! 🚀**
