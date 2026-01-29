# 🎯 Überblick: 2 Dashboards im Energy Forecasting System

## Die zwei Welten

```
┌─────────────────────────────────────────────────────────────────┐
│                    ENERGY FORECASTING SYSTEM                    │
├──────────────────────────────┬──────────────────────────────────┤
│                              │                                  │
│   🎯 API-UI Dashboard        │     📈 Grafana Monitoring       │
│   (Prognosen generieren)     │     (Performance überwachen)    │
│                              │                                  │
│   http://localhost:8000/ui   │     http://localhost:3000       │
│                              │                                  │
│   ✅ Aktuelle Vorhersagen    │     ✅ System-Performance       │
│   ✅ Live Charts             │     ✅ Model Drift Detection    │
│   ✅ Energie-Typen wählen    │     ✅ Fehlermetriken (MAE)    │
│   ✅ Manuell generieren      │     ✅ Datenqualität           │
│   ✅ Sofort visualisieren    │     ✅ API Request Rate        │
│                              │     ✅ Latenz-Messung         │
│                              │                                  │
└──────────────────────────────┴──────────────────────────────────┘
```

---

## 🎯 API-UI Dashboard
**Zweck:** Prognosen generieren und visualisieren

### Was kannst du dort tun?
- 🌞 Solar-Erzeugung vorhersagen
- 💨 Wind Offshore/Onshore vorhersagen
- 🔋 Stromverbrauch vorhersagen
- 💰 Preise vorhersagen
- 📊 Charts in Echtzeit sehen
- 📋 Tabelle mit genauen Werten

### Wer nutzt das?
- Analysten (testen Modelle)
- Energieplaner (schauen Prognosen an)
- Developer (debuggen Vorhersagen)

### URL
```
http://localhost:8000/ui
```

---

## 📈 Grafana Dashboard
**Zweck:** System-Performance und Modell-Qualität überwachen

### Was siehst du dort?
1. **Prediction Count** - Wie viele Prognosen wurden gemacht?
2. **Model Drift Score** - Ist das Modell noch gut? (0-1)
3. **Prediction MAE** - Durchschnittlicher Fehler in MW
4. **Prediction MAPE** - Durchschnittlicher Fehler in %
5. **Data Quality Score** - Sind die Daten sauber?
6. **Prediction Latency** - Wie schnell ist die API?
7. **API Request Rate** - Wie viele Nutzer?

### Wer nutzt das?
- DevOps / Operations (überwachen System)
- Technische Manager (Performance-Reports)
- Data Scientists (Model-Monitoring)

### URL
```
http://localhost:3000
Login: admin / admin
```

---

## 🔄 Wie sie zusammenhängen

```
Du klickst in API-UI
"Vorhersage generieren"
       ↓
API macht Berechnung
       ↓
Sendet Metriken an
Prometheus
       ↓
Grafana zeigt
Metriken live
```

**Das Ergebnis:** Beide Dashboards sind synchronized! 🔗

---

## 💡 Die beste Nutzung

### Für schnelle Prognose-Checks
👉 Nur **API-UI** öffnen

```
http://localhost:8000/ui
```

### Für System-Überwachung
👉 Nur **Grafana** öffnen

```
http://localhost:3000
```

### Für vollständige Analyse (Empfohlen!)
👉 **Beide Side-by-Side**

```
┌─────────────────────────────┬─────────────────────────────┐
│   API-UI                    │   Grafana                   │
│ localhost:8000/ui           │   localhost:3000            │
│                             │                             │
│ 1. Generiere Prognose      │ 1. Beobachte Metriken      │
│ 2. Schaue Chart            │ 2. Prediction Count steigt  │
│ 3. Vergleiche mit Grafana  │ 3. Model Drift anschauen   │
│                             │                             │
└─────────────────────────────┴─────────────────────────────┘
```

---

## 📚 Dokumentationen

| Dokument | Inhalt | Für wen? |
|----------|--------|---------|
| [QUICKSTART.md](energy-timeseries-project/QUICKSTART.md) | Erste Schritte | Alle |
| [GRAFANA_DASHBOARD_GUIDE_DE.md](energy-timeseries-project/docs/GRAFANA_DASHBOARD_GUIDE_DE.md) | Was bedeuten die Grafana-Charts? | Anfänger |
| [PREDICTIONS_AND_GRAFANA.md](energy-timeseries-project/docs/PREDICTIONS_AND_GRAFANA.md) | Übersicht beider Dashboards | Alle |
| [HOW_TO_USE_BOTH_DASHBOARDS.md](energy-timeseries-project/docs/HOW_TO_USE_BOTH_DASHBOARDS.md) | Praktische Workflows | Power-User |
| [MONITORING_SETUP.md](energy-timeseries-project/docs/MONITORING_SETUP.md) | Technische Details | Developer/Ops |

---

## 🚀 Los geht's

### Schritt 1: Starten
```bash
cd energy-timeseries-project
./start_monitoring.sh
```

### Schritt 2: API-UI öffnen
```
http://localhost:8000/ui
```

### Schritt 3: Prognose generieren
1. Energy-Type wählen (z.B. Solar)
2. "Vorhersage generieren" klicken
3. Chart anschauen 📊

### Schritt 4: Grafana öffnen
```
http://localhost:3000 (admin/admin)
```

### Schritt 5: Vergleichen
- Prediction Count sollte steigen
- Model Drift anschauen
- Datenqualität prüfen

---

## ✅ Checkliste

- [ ] Beide Dashboards erreichbar?
- [ ] API-UI: Kann Prognose generieren?
- [ ] Grafana: Sieht Metriken?
- [ ] Prognose in API-UI → Metrik wächst in Grafana?

---

## 🎓 Was du jetzt weißt

✅ API-UI = Prognosen testen & visualisieren
✅ Grafana = Performance & Qualität überwachen
✅ Beide synchronisiert = Vollständige Lösung
✅ Unterschiedliche Zielgruppen = Unterschiedliche Tools

**Status:** ✨ Fertig zum Ausprobieren!

---

**Version:** 1.0 | **Datum:** 2026-01-29 | **Nächste Schritte:** Siehe QUICKSTART.md
