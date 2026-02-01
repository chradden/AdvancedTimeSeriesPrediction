# 📊 Price Forecasting Notebooks

This directory contains the **complete 9-notebook analysis pipeline** for **electricity price (day-ahead)** forecasting, matching the structure of the Solar analysis.

## 📁 Notebook Structure (9 Notebooks)

### 1. **01_price_data_exploration.ipynb**
- Timeline visualization (2022-2024)
- Data quality assessment
- Statistical properties analysis
- Negative price detection and analysis
- Temporal patterns (hourly, daily, weekly, seasonal)
- Autocorrelation analysis
- Volatility and spike detection
- Train/Val/Test split definition

**Key Findings:**
- Date range: 2022-2024 (3 years, hourly data)
- Negative prices present (oversupply situations)
- High volatility with spikes
- Strong hourly patterns
- Most challenging energy type to forecast

---

### 2. **02_price_preprocessing.ipynb**
- Feature engineering (46 features total):
  - Time features (19): hour, day, month, cyclic encoding, etc.
  - Lag features (8): 1h, 2h, 3h, 6h, 12h, 24h, 48h, 168h
  - Rolling statistics (20): mean, std, min, max for 5 windows
  - Difference features (3): 1h, 24h, 168h differences
  - Price-specific features (5): negative indicator, volatility, momentum
- Handling negative prices (kept as valid data)
- Train/Val/Test split (70%/15%/15%)
- Feature scaling (StandardScaler)
- Save processed datasets

**Output:**
- `price_train.csv`
- `price_val.csv`
- `price_test.csv`
- `price_feature_info.json`

---

### 3. **03_price_baseline_models.ipynb**
Simple forecasting methods to establish performance threshold:

1. **Naive** - Last observation
2. **Seasonal Naive (24h)** - Same hour yesterday
3. **Moving Average (24h)** - Rolling mean
4. **Drift** - Linear trend
5. **Mean** - Historical average

**Expected:** Negative or very low R² due to volatility

---

### 4. **04_price_statistical_models.ipynb**
Classical time series models:

1. **SARIMA** - Seasonal ARIMA with 24h seasonality
2. **ETS** - Exponential Smoothing (Additive)

**Features:**
- Stationarity testing (ADF test)
- Residual analysis
- Model diagnostics

**Expected:** Better than baselines but limited by volatility

---

### 5. **05_price_ml_tree_models.ipynb**
Machine learning tree-based models:

1. **Random Forest** - Ensemble of decision trees
2. **XGBoost** - Gradient boosting (optimized)
3. **LightGBM** - Fast gradient boosting
4. **CatBoost** - Categorical boosting

**Features:**
- Feature importance analysis
- Error analysis
- Comparison with baselines and statistical models

**Expected:** R² 0.85-0.92 (best performance category)

---

### 6. **06_price_deep_learning.ipynb**
Deep learning sequence models:

1. **LSTM** - Long Short-Term Memory
2. **GRU** - Gated Recurrent Unit
3. **BiLSTM** - Bidirectional LSTM

**Features:**
- Sequence preparation (24-hour lookback)
- Training with early stopping
- Training history visualization
- Final comparison of ALL models

**Expected:** Comparable to ML models, may handle spikes better

---

### 7. **07_price_generative_models.ipynb**
Conceptual overview of generative approaches:

1. **Autoencoder** - Anomaly detection
2. **VAE** - Probabilistic forecasting
3. **GAN** - Scenario generation
4. **DeepAR** - Quantile predictions

**Features:**
- Uncertainty quantification concepts
- Anomaly detection approaches
- Practical recommendations

**Reality Check:** LightGBM (R²=0.9798) already excellent  
**Recommendation:** Use LightGBM Quantile for uncertainty instead

---

### 8. **08_price_advanced_models.ipynb**
State-of-the-art deep learning models (conceptual):

1. **N-BEATS** - Neural Basis Expansion (pure univariate)
2. **TFT (Temporal Fusion Transformer)** - Attention-based with interpretability

**Features:**
- Architecture overviews
- Implementation guides
- Performance vs complexity trade-off

**Reality Check:** Very complex, long training (30min-2h GPU)  
**Expected:** R²~0.92-0.97 (unlikely to beat LightGBM)  
**Recommendation:** Not worth the complexity for price

---

### 9. **09_price_model_comparison.ipynb**
Final comprehensive comparison of all models:

**Analyses:**
- Model ranking (all phases)
- Performance vs Complexity visualization
- Model selection matrix for different use cases
- Production deployment guide
- Final recommendations

**Features:**
- Best per category identification
- Trade-off analysis
- Practical deployment instructions
- Monitoring and retraining strategies

**Output:** Complete production-ready documentation

---

## 🎯 Expected Results

Based on the Masterplan, price forecasting is the **most challenging** energy type:

| Metric | Expected Range | Reason |
|--------|----------------|--------|
| **R²** | 0.85 - 0.92 | High volatility, spikes, negative prices |
| **Best Model** | LightGBM or XGBoost | Tree models handle feature interactions well |
| **Challenges** | Negative prices, extreme spikes | Require special handling |

---

## 🔄 Workflow

Run notebooks in order:

```bash
1. 01_price_data_exploration.ipynb     # Understand the data
2. 02_price_preprocessing.ipynb        # Prepare features
3. 03_price_baseline_models.ipynb      # Set baseline
4. 04_price_statistical_models.ipynb   # Classical approaches
5. 05_price_ml_tree_models.ipynb       # Best performance expected
6. 06_price_deep_learning.ipynb        # RNN models
7. 07_price_generative_models.ipynb    # Conceptual (Autoencoder, VAE, GAN)
8. 08_price_advanced_models.ipynb      # Conceptual (N-BEATS, TFT)
9. 09_price_model_comparison.ipynb     # Final comparison & deployment
```

**Or use automated pipeline:**

```bash
python scripts/run_price_extended_pipeline.py
```
→ Executes all phases in ~7 minutes!

---

## 📈 Key Differences from Other Energy Types

**Price vs Solar/Wind/Consumption:**
- ❌ More volatile (higher CV)
- ❌ Negative values possible
- ❌ Extreme spikes more common
- ✅ Strong hourly patterns
- ✅ Clear peak/off-peak structure
- ⚠️ Lower expected R² (0.85-0.92 vs 0.99+)

---

## 💾 Output Files

### Processed Data
- `data/processed/price_train.csv`
- `data/processed/price_val.csv`
- `data/processed/price_test.csv`
- `data/processed/price_feature_info.json`

### Results
- `results/metrics/price_baseline_metrics.csv`
- `results/metrics/price_statistical_metrics.csv`
- `results/metrics/price_ml_tree_metrics.csv`
- `results/metrics/price_deep_learning_metrics.csv`
- `results/metrics/price_all_models_final.csv`
- `results/metrics/price_feature_importance_xgb.csv`
- `results/metrics/price_feature_importance_lgb.csv`

### Visualizations
- `results/figures/price_*_timeline.png`
- `results/figures/price_*_patterns.png`
- `results/figures/price_*_forecast.png`
- `results/figures/price_*_comparison.png`
- `results/figures/price_feature_importance.png`
- And many more...

---

## 🔍 Key Insights

### What Makes Price Forecasting Challenging?
1. **Volatility**: Prices can change dramatically hour-to-hour
2. **Negative Prices**: Oversupply situations → negative EUR/MWh
3. **Spikes**: Sudden extreme values (both high and low)
4. **Multiple Drivers**: Weather, demand, supply, market dynamics
5. **Non-stationarity**: Mean and variance change over time

### What Helps?
✅ Lag features (recent prices are good predictors)  
✅ Rolling statistics (capture local trends)  
✅ Hour/day features (strong hourly patterns)  
✅ Tree-based models (handle non-linearity well)  

---

## 🚀 Next Steps After Completion

1. **Hyperparameter Tuning**: Optimize top 3 models
2. **Ensemble Methods**: Combine predictions
3. **External Features**: Add weather, demand forecasts
4. **Cross-Series Analysis**: Update `10_multi_series_analysis.ipynb`
5. **Production Deployment**: Choose best model for API

---

## 📚 References

- **Masterplan**: See `MASTERPLAN.md` in project root
- **Template Structure**: Based on Solar pipeline (9 notebooks)
- **Expected Performance**: R² 0.850-0.920 (Table in Masterplan line 172)

---

**Status**: ✅ All 6 notebooks created and ready for execution

**Author**: Energy Time Series Forecasting Framework  
**Date**: 2026-01-31  
**Energy Type**: Price (Day-Ahead)
