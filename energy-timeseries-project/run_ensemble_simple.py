"""
Simplified Ensemble Methods - nur sklearn-basiert
Keine Deep Learning oder LLM Modelle benötigt
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import VotingRegressor, StackingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("ENSEMBLE METHODS - SOLAR FORECASTING")
print("=" * 60)

# Daten laden
print("\n📊 Lade Daten...")
df_train = pd.read_csv('data/processed/solar_train.csv')
df_val = pd.read_csv('data/processed/solar_val.csv')
df_test = pd.read_csv('data/processed/solar_test.csv')

X_train = df_train.drop(['value', 'timestamp'], axis=1)
y_train = df_train['value']
X_val = df_val.drop(['value', 'timestamp'], axis=1)
y_val = df_val['value']
X_test = df_test.drop(['value', 'timestamp'], axis=1)
y_test = df_test['value']

print(f"Train: {len(X_train)} samples")
print(f"Val: {len(X_val)} samples")
print(f"Test: {len(X_test)} samples")

# Base Models aus Notebook 05 replizieren
print("\n🤖 Trainiere Base Models...")

# RandomForest - schnell
print("  - RandomForest...")
rf = RandomForestRegressor(n_estimators=100, max_depth=15, min_samples_split=5, 
                           random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
pred_rf = rf.predict(X_test)
rmse_rf = np.sqrt(mean_squared_error(y_test, pred_rf))
mae_rf = mean_absolute_error(y_test, pred_rf)
r2_rf = r2_score(y_test, pred_rf)

# Ridge Regression - schnell
print("  - Ridge Regression...")
ridge = Ridge(alpha=10.0, random_state=42)
ridge.fit(X_train, y_train)
pred_ridge = ridge.predict(X_test)
rmse_ridge = np.sqrt(mean_squared_error(y_test, pred_ridge))
mae_ridge = mean_absolute_error(y_test, pred_ridge)
r2_ridge = r2_score(y_test, pred_ridge)

# Gradient Boosting (sklearn) - langsamer
print("  - Gradient Boosting...")
from sklearn.ensemble import GradientBoostingRegressor
gb = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5,
                               random_state=42)
gb.fit(X_train, y_train)
pred_gb = gb.predict(X_test)
rmse_gb = np.sqrt(mean_squared_error(y_test, pred_gb))
mae_gb = mean_absolute_error(y_test, pred_gb)
r2_gb = r2_score(y_test, pred_gb)

print("\n✅ Base Models trainiert")
print(f"RandomForest: RMSE={rmse_rf:.1f} MW, MAE={mae_rf:.1f}, R²={r2_rf:.4f}")
print(f"Ridge: RMSE={rmse_ridge:.1f} MW, MAE={mae_ridge:.1f}, R²={r2_ridge:.4f}")
print(f"GradientBoosting: RMSE={rmse_gb:.1f} MW, MAE={mae_gb:.1f}, R²={r2_gb:.4f}")

# Ensemble Methods
print("\n" + "=" * 60)
print("ENSEMBLE METHODS")
print("=" * 60)

# 1. Simple Average
print("\n1️⃣ Simple Average Ensemble...")
pred_avg = (pred_rf + pred_ridge + pred_gb) / 3
rmse_avg = np.sqrt(mean_squared_error(y_test, pred_avg))
mae_avg = mean_absolute_error(y_test, pred_avg)
r2_avg = r2_score(y_test, pred_avg)
print(f"   RMSE={rmse_avg:.1f} MW, MAE={mae_avg:.1f}, R²={r2_avg:.4f}")

# 2. Weighted Average (Performance-based)
print("\n2️⃣ Weighted Average (Performance-based)...")
# Gewichte basierend auf 1/RMSE
total_inv_rmse = 1/rmse_rf + 1/rmse_ridge + 1/rmse_gb
w_rf = (1/rmse_rf) / total_inv_rmse
w_ridge = (1/rmse_ridge) / total_inv_rmse
w_gb = (1/rmse_gb) / total_inv_rmse

pred_weighted = w_rf * pred_rf + w_ridge * pred_ridge + w_gb * pred_gb
rmse_weighted = np.sqrt(mean_squared_error(y_test, pred_weighted))
mae_weighted = mean_absolute_error(y_test, pred_weighted)
r2_weighted = r2_score(y_test, pred_weighted)
print(f"   Weights: RF={w_rf:.2f}, Ridge={w_ridge:.2f}, GB={w_gb:.2f}")
print(f"   RMSE={rmse_weighted:.1f} MW, MAE={mae_weighted:.1f}, R²={r2_weighted:.4f}")

# 3. Optimized Weights (Grid Search auf Validation Set)
print("\n3️⃣ Optimized Weights (Grid Search)...")
pred_rf_val = rf.predict(X_val)
pred_ridge_val = ridge.predict(X_val)
pred_gb_val = gb.predict(X_val)

best_rmse = float('inf')
best_weights = None

for w1 in np.linspace(0.1, 0.6, 20):
    for w2 in np.linspace(0.1, 0.6, 20):
        w3 = 1 - w1 - w2
        if w3 >= 0.1 and w3 <= 0.6:
            pred_val = w1*pred_rf_val + w2*pred_ridge_val + w3*pred_gb_val
            rmse_val = np.sqrt(mean_squared_error(y_val, pred_val))
            if rmse_val < best_rmse:
                best_rmse = rmse_val
                best_weights = (w1, w2, w3)

pred_optimized = best_weights[0]*pred_rf + best_weights[1]*pred_ridge + best_weights[2]*pred_gb
rmse_optimized = np.sqrt(mean_squared_error(y_test, pred_optimized))
mae_optimized = mean_absolute_error(y_test, pred_optimized)
r2_optimized = r2_score(y_test, pred_optimized)
print(f"   Optimal Weights: RF={best_weights[0]:.2f}, Ridge={best_weights[1]:.2f}, GB={best_weights[2]:.2f}")
print(f"   RMSE={rmse_optimized:.1f} MW, MAE={mae_optimized:.1f}, R²={r2_optimized:.4f}")

# 4. Stacking mit Ridge Meta-Learner
print("\n4️⃣ Stacking Regressor (Ridge Meta-Learner)...")
estimators = [('rf', rf), ('ridge', ridge), ('gb', gb)]
stacking = StackingRegressor(
    estimators=estimators, 
    final_estimator=Ridge(alpha=1.0),
    cv=3,
    n_jobs=-1
)
stacking.fit(X_train, y_train)
pred_stacking = stacking.predict(X_test)
rmse_stacking = np.sqrt(mean_squared_error(y_test, pred_stacking))
mae_stacking = mean_absolute_error(y_test, pred_stacking)
r2_stacking = r2_score(y_test, pred_stacking)
print(f"   RMSE={rmse_stacking:.1f} MW, MAE={mae_stacking:.1f}, R²={r2_stacking:.4f}")

# 5. Voting Regressor
print("\n5️⃣ Voting Regressor...")
voting = VotingRegressor(estimators=estimators)
voting.fit(X_train, y_train)
pred_voting = voting.predict(X_test)
rmse_voting = np.sqrt(mean_squared_error(y_test, pred_voting))
mae_voting = mean_absolute_error(y_test, pred_voting)
r2_voting = r2_score(y_test, pred_voting)
print(f"   RMSE={rmse_voting:.1f} MW, MAE={mae_voting:.1f}, R²={r2_voting:.4f}")

# Ergebnisse zusammenfassen
print("\n" + "=" * 60)
print("FINALE ERGEBNISSE")
print("=" * 60)

results = pd.DataFrame({
    'Methode': [
        'RandomForest (Base)',
        'Ridge (Base)',
        'GradientBoosting (Base)',
        'Simple Average',
        'Weighted Average',
        'Optimized Weights',
        'Stacking',
        'Voting'
    ],
    'RMSE': [rmse_rf, rmse_ridge, rmse_gb, rmse_avg, rmse_weighted, 
             rmse_optimized, rmse_stacking, rmse_voting],
    'MAE': [mae_rf, mae_ridge, mae_gb, mae_avg, mae_weighted, 
            mae_optimized, mae_stacking, mae_voting],
    'R²': [r2_rf, r2_ridge, r2_gb, r2_avg, r2_weighted,
           r2_optimized, r2_stacking, r2_voting]
})

results = results.sort_values('RMSE')
print("\n" + results.to_string(index=False))

# Beste Methode
best = results.iloc[0]
print(f"\n🏆 BESTE ENSEMBLE-METHODE: {best['Methode']}")
print(f"   RMSE: {best['RMSE']:.1f} MW")
print(f"   MAE: {best['MAE']:.1f} MW")
print(f"   R²: {best['R²']:.4f}")

# Improvement über beste Base Model
best_base_rmse = min(rmse_rf, rmse_ridge, rmse_gb)
improvement = ((best_base_rmse - best['RMSE']) / best_base_rmse) * 100
print(f"\n📈 Verbesserung über beste Base Model: {improvement:.2f}%")

# Speichern
results.to_csv('results/metrics/solar_ensemble_simple_results.csv', index=False)
print("\n💾 Gespeichert: results/metrics/solar_ensemble_simple_results.csv")

print("\n" + "=" * 60)
print("✅ ENSEMBLE ANALYSE ABGESCHLOSSEN")
print("=" * 60)
