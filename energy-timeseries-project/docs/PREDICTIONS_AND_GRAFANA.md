# 📋 Integration: API Prognosen + Grafana

## Situation

Es gibt 2 Dashboards:

### 1. **API Web Dashboard** (Prognosen)
```
Localhost: http://localhost:8000/ui
Codespace: https://<codespace-name>-8000.app.github.dev/ui
```
- 🎯 Zeigt aktuelle Vorhersagen an
- 📊 Interaktive Charts pro Energietyp
- 🎮 Manual vorhersagen generieren
- 💾 Ergebnisse in Tabelle

**Beste für:** Live-Vorhersagen testen & Parameter ändern

---

### 2. **Grafana Dashboard** (Monitoring)  
```
Localhost: http://localhost:3000
Codespace: https://<codespace-name>-3000.app.github.dev
Login: admin / admin
```
- 📈 Performance-Metriken
- 🔍 Model Drift, MAE, MAPE
- 📊 API-Request-Rate
- 🕐 Historische Trends

**Beste für:** Langzeit-Monitoring & System-Health

---

## 🔄 Wie sie zusammenhängen

```
API-UI generiert Prediction
    ↓
Prediction wird gemacht
    ↓
Metriken gehen an Prometheus
    ↓
Grafana zeigt Metriken
```

**Ablauf:**
1. Du klickst in API-UI "Vorhersage generieren"
2. API macht Prediction → sendet an Prometheus
3. Im Grafana refreshen → Neue Prediction Count + MAE/MAPE sichtbar

---

## 🎯 Quick Navigation

### Für Vorhersagen testen:
```
Localhost: http://localhost:8000/ui
Codespace: https://<codespace-name>-8000.app.github.dev/ui
```
- Solar/Wind/etc. auswählen
- "Vorhersage generieren" klicken
- Chart + Tabelle sehen

### Für Monitoring:
```
Localhost: http://localhost:3000 (admin/admin)
Codespace: https://<codespace-name>-3000.app.github.dev (admin/admin)
```
- Prediction Count steigt
- Model Drift anschauen
- Data Quality prüfen

---

## 💡 Best Practice Workflow

### Option A: Schnelle Tests
1. Öffne API-UI: 
   - Localhost: http://localhost:8000/ui
   - Codespace: https://<codespace-name>-8000.app.github.dev/ui
2. Generiere mehrere Prognosen (verschiedene Typen)
3. Schau die Charts an
4. Dann: Öffne Grafana um Metriken zu sehen

### Option B: Production Monitoring
1. Nur Grafana offen:
   - Localhost: http://localhost:3000
   - Codespace: https://<codespace-name>-3000.app.github.dev
2. API läuft im Hintergrund und generiert Prognosen
3. Schau nur Monitoring-Metriken an
4. Bei Problemen: Logs prüfen oder API-UI öffnen

### Option C: Vergleich
- **Linkes Fenster:** API-UI 
  - Localhost: http://localhost:8000/ui
  - Codespace: https://<codespace-name>-8000.app.github.dev/ui
- **Rechtes Fenster:** Grafana
  - Localhost: http://localhost:3000
  - Codespace: https://<codespace-name>-3000.app.github.dev
- Side-by-side vergleichen

---

## 🔗 API Endpoints für Prognosen

### Web UI
```
GET /ui
```
→ Öffnet das schöne Vorhersage-Dashboard

### API (JSON Responses)
```bash
POST /api/predict/solar
POST /api/predict/wind_offshore
POST /api/predict/wind_onshore
POST /api/predict/consumption
POST /api/predict/price

# Mit Payload:
{
  "hours": 24
}
```

### Beispiel: Curl
```bash
curl -X POST http://localhost:8000/api/predict/solar \
  -H "Content-Type: application/json" \
  -d '{"hours":24}'
```

**Response:**
```json
{
  "predictions": [100, 200, 300, ...],
  "timestamps": ["2026-01-29T19:00:00", ...],
  "model": "XGBoost (Production Model)",
  "mae_expected": 249.03,
  "r2_expected": 0.9825
}
```

---

## 📊 Grafana Chart für Live-Predictions

Falls du ein **Live-Prediction-Panel in Grafana** möchtest, können wir das ergänzen:

### Option 1: JSON Data Source (Einfach)
- Grafana verbindet sich direkt zur API
- Panel zeigt aktuelle Prognosen
- Nachteil: Nur die letzte Prognose

### Option 2: InfluxDB/TimeSeries (Complex)
- Prognosen in TimeSeries-DB speichern
- Grafana kann dann historische Prognosen zeigen
- Vergleich: Prognose vs. Realität

### Option 3: Aktuell (Empfohlen)
- Nutze API-UI für Prognosen: http://localhost:8000/ui
- Nutze Grafana für Performance-Metriken: http://localhost:3000
- Beide Dashboards sind optimiert für ihre Aufgabe!

---

## 🚀 Die 3 wichtigsten URLs merken:

| URL | Zweck | Nutzer |
|-----|-------|--------|
| `http://localhost:8000/ui` | 📊 Prognosen generieren & visualisieren | Analyst, Power-User |
| `http://localhost:3000` | 📈 System-Monitoring & Metriken | Ops, Infrastruktur |
| `http://localhost:8000/docs` | 🔧 API Technical Docs | Developer |

---

## 📱 Mobile/Remote Access (Codespaces)

### Ports öffnen:
1. VS Code → "PORTS" Panel
2. Port 8000 (API) → Public
3. Port 3000 (Grafana) → Public
4. Dann URLs direkt öffnen

### Remote teilen:
```bash
# Beide URLs sind von außen erreichbar:
https://sturdy-space-...-8000.app.github.dev/ui
https://sturdy-space-...-3000.app.github.dev
```

---

## ✅ Checkliste zum Start

- [ ] API läuft? (`docker compose ps`)
- [ ] API-UI öffnen: http://localhost:8000/ui
- [ ] Prognose generieren → Chart sichtbar?
- [ ] Grafana öffnen: http://localhost:3000
- [ ] Prediction Count Chart → steigt?
- [ ] Beide Seiten im Split-View öffnen

---

## 💬 Häufige Fragen

**F: Kann ich Prognosen direkt in Grafana sehen?**
A: Ja, aber die API-UI ist besser dafür optimiert. Grafana zeigt dir eher die Performance-Metriken.

**F: Warum 2 Dashboards?**
A: API-UI = "Wie gut ist die Vorhersage?" / Grafana = "Wie geht es dem System?"

**F: Können die sich synchronisieren?**
A: Sie tun das bereits! API-UI generiert Prognosen → Grafana zeigt die Metriken sofort.

**F: Welches Dashboard für Production?**
A: Grafana! Die API-UI ist nur für Testing/Exploration.

---

**Version:** 1.0 | **Datum:** 2026-01-29
