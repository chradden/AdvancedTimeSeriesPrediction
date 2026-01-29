# 🎯 So setzt du die API-UI und Grafana zusammen ein

## Szenarien

### 1️⃣ Entwickler: "Ich teste neue Modelle"

```
1. API-UI öffnen: 
   Localhost: http://localhost:8000/ui
   Codespace: https://<codespace-name>-8000.app.github.dev/ui
2. Energy-Type auswählen (z.B. Solar)
3. "Vorhersage generieren" klicken
4. Chart anschauen → Sieht die Vorhersage gut aus?
5. Paralleles Grafana-Tab: Sind die Metriken besser geworden?
```

**Workflow:** 
- Links: API-UI (Prognosen)
- Rechts: Grafana (Metriken)
- Side-by-side vergleichen

---

### 2️⃣ Analyst: "Ich brauche den aktuellen Prognose-Status"

**Option A - Schnell:**
```
Localhost: http://localhost:8000/ui
Codespace: https://<codespace-name>-8000.app.github.dev/ui

→ Sieht alle 5 Energietypen
→ Aktuelle Vorhersagen im Chart
→ Tabelle mit genauen Werten
```

**Option B - Ausführlich:**
```
1. Grafana: Schaue "Prediction Count" → Wie viele Prognosen?
2. Grafana: Schaue "Model Drift" → Sind Modelle noch gut?
3. API-UI: Generiere neue Prognose
4. Vergleiche die Charts
```

---

### 3️⃣ Operations: "Ich überwache das System"

```
Localhost: http://localhost:3000
Codespace: https://<codespace-name>-3000.app.github.dev

Schau diese Panels:
- Model Drift Score → Zu hoch? ⚠️
- Data Quality → Zu niedrig? ⚠️
- API Request Rate → Lädt das System?
- Prediction Latency → Zu langsam? 🐌
```

**Wenn Problem:** Logs prüfen
```bash
docker compose logs api | tail -100
```

---

## 📊 Das perfekte Setup

### Browser Split-View (Empfohlen!)

```
┌─────────────────────────────┬─────────────────────────────┐
│   API-UI (Prognosen)        │   Grafana (Monitoring)      │
│ Localhost: 8000/ui          │   Localhost: 3000           │
│ Codespace: 8000 port        │   Codespace: 3000 port      │
│                             │                             │
│ - Energy Type wählen        │ - Live Metrics              │
│ - "Generieren" klicken      │ - Model Drift anschauen     │
│ - Chart sehen               │ - MAE/MAPE sehen            │
│ - Tabelle sehen             │ - Refreshen (alle 30s)      │
│                             │                             │
└─────────────────────────────┴─────────────────────────────┘
```

**So öffnest du Split-View:**
1. Rechts-Klick auf API-UI URL → "In neuem Tab öffnen"
2. In VS Code: `Strg+K Strg+O` (Split Editor öffnen)
3. API-UI links, Grafana rechts
4. Bei Bedarf: F11 Fullscreen für mehr Platz

---

## 🔄 Datenfluss

```
API-UI generiert Prognose
    ↓
API macht Calculation
    ↓
Schreibt an Prometheus
    ↓
Grafana liest von Prometheus
    ↓
Grafana zeigt im Chart
```

**Timing:** Meist < 1 Sekunde!

---

## 🎮 Interaktive Tests

### Test 1: "Funktioniert die Integration?"

```bash
# Terminal 1: Predictions generieren
for i in {1..5}; do
  curl -X POST http://localhost:8000/api/predict/solar \
    -H "Content-Type: application/json" \
    -d '{"hours":24}' &
done
wait

# Dann: Grafana refreshen und "Prediction Count" schauen
# Chart sollte 5 neue Punkte oben bekommen!
```

### Test 2: "Wie schnell ist die API?"

```
1. API-UI öffnen: http://localhost:8000/ui
2. Solar auswählen
3. "Generieren" klicken (mehrmals schnell)
4. Stopuhr starten bis Chart aktualisiert
5. In Grafana: "Prediction Latency" anschauen
```

**Normal:** < 0.5 Sekunden

### Test 3: "Funktioniert das Monitoring?"

```
1. Grafana öffnen: http://localhost:3000
2. "Prediction Count" Panel anschauen
3. Note der aktuellen Wert
4. API-UI öffnen und viele Prognosen generieren
5. Grafana refreshen (oder 30s warten)
6. Chart sollte Anstieg zeigen
```

---

## 📱 Mobile / Codespaces Remote

### Setup für Remote-Zugriff:

1. Codespaces → "PORTS" Panel
2. Port 8000 (API) → Rechts-Klick → "Make public"
3. Port 3000 (Grafana) → Rechts-Klick → "Make public"
4. URLs kopieren und teilen

**Beispiel:**
```
Prognose-UI: https://sturdy-space-...-8000.app.github.dev/ui
Grafana:     https://sturdy-space-...-3000.app.github.dev
```

---

## ✅ Checkliste: Alles funkioniert!

- [ ] API läuft? (`docker compose ps` zeigt alle 3 grün)
- [ ] API-UI erreichbar? (http://localhost:8000/ui)
- [ ] API-UI Prognose generierbar? (Button clickbar)
- [ ] Chart in API-UI sichtbar? (Nach Generieren)
- [ ] Grafana erreichbar? (http://localhost:3000, Login: admin/admin)
- [ ] Grafana "Prediction Count" Panel? (Sollte Linien haben)
- [ ] Nach API-UI Prognose in Grafana aktualisiert? (Chart wächst)

---

## 🚨 Troubleshooting

### Problem: API-UI ist leer / funktioniert nicht

```bash
# Container neu starten
docker compose restart api

# Logs prüfen
docker compose logs api | tail -50

# Alle neu starten
docker compose down -v
docker compose up
```

### Problem: Grafana zeigt keine Metriken

```bash
# Prometheus offen: http://localhost:9090
# Targets checken: http://localhost:9090/targets
# API sollte "UP" sein (grüner Status)

# Wenn rot:
docker compose logs prometheus
```

### Problem: Prognosen sind falsch

```
1. Schaue API-UI Chart → Realistische Werte?
2. Schaue Grafana "Data Quality" → > 0.9?
3. Schaue Grafana "Model Drift" → < 0.5?

Wenn Drift zu hoch:
→ Modell sollte retrainiert werden
```

---

## 💡 Pro-Tipps

1. **Favorites in Browser setzen:** Beide URLs bookmarken
2. **Grafana-Einstellungen:** Time-Range auf "Last 1 hour" setzen für Live-Monitoring
3. **API-UI für Tests:** Verschiedene Energietypen nacheinander testen
4. **Grafana für Reporting:** Screenshots der Panels machen für Reports
5. **Both aktiv:** Im Team: Eine Person API-UI, eine Grafana - beide synchronized

---

**Version:** 1.0 | **Datum:** 2026-01-29
