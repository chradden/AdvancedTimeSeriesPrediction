# 🌞 24-Stunden Solar Forecast - Quick Start

## Übersicht

Die API kann jetzt **24 Stunden (oder mehr) in die Zukunft** prognostizieren mit rolling forecasts und automatischen Feature-Updates.

## 🚀 Schnellstart

### 1. API starten
```bash
cd energy-timeseries-project
python api.py
```

### 2. Test ausführen
```bash
# In einem neuen Terminal
python test_24h_forecast.py
```

## 📊 Was wurde verbessert?

### Vorher:
- ❌ Vereinfachte iterative Prediction ohne Feature-Updates
- ❌ Features wurden nicht für jeden Zeitschritt aktualisiert
- ❌ Ungenauigkeit über längere Horizonte

### Jetzt:
- ✅ **Rolling Forecast** mit korrekten Feature-Updates
- ✅ Automatische Berechnung von Lags und Rolling Statistics
- ✅ Genaue Vorhersagen über 24+ Stunden
- ✅ Non-negative Constraint (Solar kann nicht negativ sein)

## 🎯 Verwendung

### Python API Call
```python
import requests

payload = {
    "historical_data": {
        "timestamps": [...],  # Letzte 7 Tage (168 Stunden)
        "values": [...]       # Solar Generation in MW
    },
    "forecast_horizon": 24,   # 24 Stunden vorhersagen
    "model": "xgboost"
}

response = requests.post("http://localhost:8000/predict/solar", json=payload)
result = response.json()

# result enthält:
# - timestamps: 24 Zeitstempel
# - predictions: 24 Vorhersagen in MW
# - model_used: "xgboost"
# - metadata: Zusätzliche Infos
```

### CURL Example
```bash
curl -X POST "http://localhost:8000/predict/solar" \
  -H "Content-Type: application/json" \
  -d '{
    "historical_data": {
      "timestamps": ["2024-01-01T00:00:00", ...],
      "values": [0, 0, 150, 500, ...]
    },
    "forecast_horizon": 24,
    "model": "xgboost"
  }'
```

## 📈 Forecast Horizons

Die API unterstützt verschiedene Vorhersage-Zeiträume:

| Horizon | Stunden | Beschreibung |
|---------|---------|--------------|
| 24 | 1 Tag | **Standard** - Bester Use Case |
| 48 | 2 Tage | Gut für Planung |
| 72 | 3 Tage | Mittel- bis langfristig |
| 168 | 1 Woche | Maximum empfohlen |

**⚠️ Hinweis**: Je länger der Horizon, desto weniger genau die Vorhersage (normale Eigenschaft aller Forecasting-Modelle).

## 🧪 Test-Szenarien

### Szenario 1: Standard 24h Forecast
```python
python test_24h_forecast.py
```

### Szenario 2: Mit echten Daten
```python
python api_client_example.py
```

### Szenario 3: Verschiedene Horizons
```python
import requests

for horizon in [24, 48, 72]:
    payload = {
        "historical_data": {...},
        "forecast_horizon": horizon,
        "model": "xgboost"
    }
    response = requests.post("http://localhost:8000/predict/solar", json=payload)
    print(f"{horizon}h forecast: {len(response.json()['predictions'])} predictions")
```

## 📊 Output Format

### JSON Response
```json
{
  "timestamps": [
    "2024-01-08T01:00:00",
    "2024-01-08T02:00:00",
    ...
  ],
  "predictions": [
    0.0,
    0.0,
    125.43,
    456.78,
    ...
  ],
  "model_used": "xgboost",
  "metadata": {
    "historical_samples": 168,
    "forecast_horizon": 24,
    "generated_at": "2026-01-29T12:34:56.789"
  }
}
```

### CSV Export
Das Test-Script erstellt automatisch `forecast_24h.csv`:
```csv
timestamp,solar_mw
2024-01-08T01:00:00,0.00
2024-01-08T02:00:00,0.00
2024-01-08T03:00:00,0.00
2024-01-08T08:00:00,125.43
2024-01-08T13:00:00,789.12
...
```

## 🔍 Feature Updates

Bei jedem Vorhersage-Schritt werden diese Features neu berechnet:

1. **Zeit-Features**
   - Stunde, Tag, Monat
   - Wochentag, Wochenende
   - Zyklische Features (sin/cos)

2. **Lag-Features**
   - lag_1, lag_2, lag_6, lag_12, lag_24, lag_48, lag_168

3. **Rolling Statistics**
   - Rolling Mean (6h, 12h, 24h, 168h)
   - Rolling Std, Min, Max

4. **Predicted Values**
   - Vorherige Predictions werden als neue historische Daten verwendet

## 💡 Best Practices

1. **Historische Daten**: Mindestens 7 Tage (168 Stunden) für gute Lag-Features
2. **Aktualisierung**: Vorhersagen regelmäßig mit neuen Daten aktualisieren
3. **Validierung**: Predictions mit echten Werten vergleichen
4. **Monitoring**: Performance über Zeit tracken

## 🐛 Troubleshooting

### "Insufficient historical data"
- **Problem**: Zu wenig historische Daten
- **Lösung**: Mindestens 168 Stunden (7 Tage) bereitstellen

### Predictions sind konstant
- **Problem**: Features können nicht berechnet werden
- **Lösung**: Mehr Varianz in historischen Daten

### API antwortet nicht
- **Problem**: Server nicht gestartet
- **Lösung**: `python api.py` ausführen

## 📚 Weitere Informationen

- API Dokumentation: http://localhost:8000/docs
- Health Check: http://localhost:8000/health
- Verfügbare Modelle: http://localhost:8000/models

---

**✨ Viel Erfolg mit deinen 24-Stunden Prognosen!**
