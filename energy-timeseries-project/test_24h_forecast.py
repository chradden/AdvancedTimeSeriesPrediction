"""
Quick Test: 24-Hour Solar Forecast
===================================

Testet die API mit einer 24-Stunden Vorhersage
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

BASE_URL = "http://localhost:8000"

def test_24h_forecast():
    """Test 24-hour solar forecast"""
    print("="*80)
    print("24-HOUR SOLAR FORECAST TEST")
    print("="*80)
    
    # Generiere historische Daten (letzte 7 Tage als Kontext)
    start_date = datetime(2024, 6, 15)  # Sommer für gute Solar-Werte
    timestamps = []
    values = []
    
    for i in range(168):  # 7 Tage
        timestamp = start_date + timedelta(hours=i)
        timestamps.append(timestamp.isoformat())
        
        hour = timestamp.hour
        # Realistische Solar-Generierung
        if 6 <= hour <= 20:
            # Peak um 13 Uhr
            peak_factor = np.sin(np.pi * (hour - 6) / 14)
            base = 800 * peak_factor  # Bis zu 800 MW
            noise = np.random.normal(0, 50)
            values.append(max(0, base + noise))
        else:
            values.append(0)  # Keine Generation nachts
    
    # API Request
    payload = {
        "historical_data": {
            "timestamps": timestamps,
            "values": values
        },
        "forecast_horizon": 24,  # 24 Stunden
        "model": "xgboost"
    }
    
    print(f"\n📊 Historische Daten:")
    print(f"  Zeitraum: {len(values)} Stunden (7 Tage)")
    print(f"  Von: {timestamps[0]}")
    print(f"  Bis: {timestamps[-1]}")
    print(f"  Durchschnitt: {np.mean(values):.2f} MW")
    
    print(f"\n🔮 Anfrage: 24-Stunden Vorhersage...")
    
    try:
        response = requests.post(f"{BASE_URL}/predict/solar", json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            predictions = result['predictions']
            forecast_times = result['timestamps']
            
            print(f"\n✅ Vorhersage erfolgreich!")
            print(f"\n📅 Vorhersage-Details:")
            print(f"  Modell: {result['model_used']}")
            print(f"  Horizon: {result['metadata']['forecast_horizon']} Stunden")
            print(f"  Generiert: {result['metadata']['generated_at']}")
            
            print(f"\n📈 Vorhersage-Statistiken:")
            print(f"  Anzahl Stunden: {len(predictions)}")
            print(f"  Durchschnitt: {np.mean(predictions):.2f} MW")
            print(f"  Maximum: {np.max(predictions):.2f} MW")
            print(f"  Minimum: {np.min(predictions):.2f} MW")
            
            # Zeige stündliche Vorhersagen
            print(f"\n⏰ Stündliche Vorhersagen (nächste 24h):")
            print("-" * 80)
            
            for i, (ts, pred) in enumerate(zip(forecast_times, predictions)):
                dt = pd.to_datetime(ts)
                hour_str = dt.strftime("%Y-%m-%d %H:%M")
                bar_length = int(pred / 20)  # Scale für Visualisierung
                bar = "█" * bar_length
                print(f"  {hour_str} | {pred:7.2f} MW | {bar}")
            
            print("-" * 80)
            
            # Tages-/Nacht-Analyse
            day_hours = [p for i, p in enumerate(predictions) if 6 <= (i % 24) <= 20]
            night_hours = [p for i, p in enumerate(predictions) if (i % 24) < 6 or (i % 24) > 20]
            
            print(f"\n☀️ Tag-Analyse (6-20 Uhr):")
            if day_hours:
                print(f"  Durchschnitt: {np.mean(day_hours):.2f} MW")
                print(f"  Maximum: {np.max(day_hours):.2f} MW")
            
            print(f"\n🌙 Nacht-Analyse (21-5 Uhr):")
            if night_hours:
                print(f"  Durchschnitt: {np.mean(night_hours):.2f} MW")
            
            # Speichern
            df = pd.DataFrame({
                'timestamp': forecast_times,
                'solar_mw': predictions
            })
            df.to_csv('forecast_24h.csv', index=False)
            print(f"\n💾 Vorhersage gespeichert: forecast_24h.csv")
            
        else:
            print(f"\n❌ Fehler {response.status_code}:")
            print(response.text)
            
    except requests.exceptions.ConnectionError:
        print("\n❌ Verbindung fehlgeschlagen!")
        print("   Stelle sicher, dass die API läuft: python api.py")
    except Exception as e:
        print(f"\n❌ Fehler: {e}")

def check_api():
    """Check if API is running"""
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            print("✅ API ist erreichbar")
            return True
        else:
            print("⚠️ API antwortet, aber Status != 200")
            return False
    except:
        print("❌ API ist nicht erreichbar!")
        print("   Starte die API mit: python api.py")
        return False

if __name__ == "__main__":
    print("\n🚀 Starte 24-Stunden Forecast Test...\n")
    
    # Check API
    if not check_api():
        print("\n⚠️ Test abgebrochen - API nicht verfügbar")
        exit(1)
    
    # Run test
    test_24h_forecast()
    
    print("\n" + "="*80)
    print("✨ Test abgeschlossen!")
    print("="*80)
