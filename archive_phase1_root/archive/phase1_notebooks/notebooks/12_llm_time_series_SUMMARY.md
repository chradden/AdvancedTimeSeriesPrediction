# 🤖 Kapitel 12: Foundation Models für Time Series

## 📊 Evaluationsergebnisse

### Chronos-T5-Small (Zero-Shot)
- **MAE**: 4,417.93 MW
- **RMSE**: 5,084.72 MW  
- **R²**: -2.97
- **MAPE**: 49.94%
- **Inference**: 134s (56ms/sample)

### Vergleich mit traditionellen Modellen

| Modell | MAE (MW) | R² | MAPE (%) | Training | Typ |
|--------|----------|-----|----------|----------|-----|
| XGBoost (Tuned) | 249.03 | 0.9825 | 3.15 | 7.6 min | ML |
| LSTM | 251.53 | 0.9822 | 3.48 | 3.4 min | DL |
| GRU | 252.32 | 0.9820 | 3.49 | 4.7 min | DL |
| XGBoost (Baseline) | 269.47 | 0.9817 | 3.41 | 0.6 s | ML |
| **Chronos-T5-Small** | **4417.93** | **-2.97** | **49.94** | **Zero-Shot** | **FM** |

## 🎯 Wichtigste Erkenntnisse

### ✅ Foundation Models Vorteile
1. **Keine Training-Daten benötigt**: Zero-Shot Forecasting
2. **Generalisierung**: Funktioniert über viele Domänen
3. **Probabilistische Vorhersagen**: Unsicherheitsquantifizierung
4. **Rapid Prototyping**: Sofort einsetzbar

### ⚠️ Foundation Models Limitationen  
1. **Domänenspezifische Performance**: 18x schlechter als XGBoost
2. **Inference-Zeit**: 56ms vs. <1ms bei ML-Modellen
3. **Ressourcen**: ~200MB Modellgröße
4. **Keine Feature Engineering**: Nutzt nur historische Werte

## 📈 Wann welches Modell?

### 🏆 XGBoost/LSTM/GRU verwenden wenn:
- ✅ Reichlich domänenspezifische Trainingsdaten vorhanden
- ✅ Optimale Accuracy erforderlich  
- ✅ Feature Engineering möglich (Wetter, Kalender, etc.)
- ✅ Niedrige Latenz wichtig
- ✅ Interpretierbarkeit gefordert

### 🤖 Chronos/Foundation Models verwenden wenn:
- ✅ Wenig/keine Trainingsdaten
- ✅ Mehrere unterschiedliche Zeitreihen  
- ✅ Schnelles Prototyping
- ✅ Probabilistische Vorhersagen benötigt
- ✅ Domänenwechsel häufig

## 🔬 Technische Details

### Chronos Architecture
- **Basis**: T5 Transformer (Text-to-Text)
- **Pre-Training**: 100B+ Zeitreihenpunkte
- **Context Window**: 512 Tokens (168h in unserem Fall)
- **Prediction**: Autoregressive Generierung
- **Samples**: 20 probabilistische Trajektorien

### Weitere Foundation Models
- **TimeGPT** (Nixtla): GPT-ähnliche Architektur
- **Lag-Llama** (ServiceNow): Llama-basiert
- **Moirai** (Salesforce): Multi-Scale

## 💡 Best Practices

1. **Hybrid-Ansatz**: Chronos für Cold-Start, dann Fine-Tuning mit XGBoost
2. **Ensemble**: Kombiniere Zero-Shot + Domain-Specific Models
3. **Scaling**: Normalisiere Daten vor Chronos Inference
4. **Context**: Nutze mindestens 7 Tage Historie für Saisonalität

## 🚀 Zukunft

Foundation Models werden besser sobald:
- Größere Modelle verfügbar (T5-Large, -XL)
- Domain-Adaptation Methoden entwickelt  
- Multimodale Integration (Text + Time Series)
- Fine-Tuning für spezifische Domänen

---

**Fazit**: Foundation Models sind vielversprechend, aber für domänenspezifische Probleme mit reichlich Daten sind traditionelle ML/DL-Methoden noch überlegen. Der Hauptvorteil liegt in der Zero-Shot-Fähigkeit für neue Domänen ohne Training.
