#!/usr/bin/env python3
"""
Chronos Foundation Model - Time Series Forecasting
Amazon's pre-trained T5-based model for zero-shot forecasting
"""

import sys
sys.path.append('src')

import pandas as pd
import numpy as np
import torch
from pathlib import Path
import warnings
import time
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tqdm import tqdm
warnings.filterwarnings('ignore')

print("=" * 80)
print("🤖 Chronos Foundation Model - Zero-Shot Time Series Forecasting")
print("=" * 80)

# Check device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n🖥️  Device: {device}")
print(f"   PyTorch: {torch.__version__}")

# ==============================================================================
# 1. Load Data
# ==============================================================================
print("\n📂 Loading data...")
DATA_TYPE = 'solar'
data_dir = Path('data/processed')

train_df = pd.read_csv(data_dir / f'{DATA_TYPE}_train.csv', parse_dates=['timestamp'])
test_df = pd.read_csv(data_dir / f'{DATA_TYPE}_test.csv', parse_dates=['timestamp'])

print(f"✅ Train: {len(train_df):,} samples")
print(f"✅ Test:  {len(test_df):,} samples")
print(f"✅ Value range: [{train_df['value'].min():.0f}, {train_df['value'].max():.0f}] MW")

train_values = train_df['value'].values
test_values = test_df['value'].values

# ==============================================================================
# 2. Load Chronos Model
# ==============================================================================
print("\n" + "=" * 80)
print("📥 Loading Chronos-T5-Small model...")
print("=" * 80)
print("   This will download ~200MB on first run")
print("   Model: amazon/chronos-t5-small")

from chronos import ChronosPipeline

try:
    pipeline = ChronosPipeline.from_pretrained(
        "amazon/chronos-t5-small",
        device_map="cpu",  # Force CPU for stability
        torch_dtype=torch.float32,
    )
    print("✅ Chronos model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    print("   Trying alternative loading method...")
    pipeline = ChronosPipeline.from_pretrained(
        "amazon/chronos-t5-small",
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    print("✅ Model loaded with alternative method!")

# ==============================================================================
# 3. Configuration
# ==============================================================================
CONTEXT_LENGTH = 168  # Use last 7 days (168 hours)
PREDICTION_LENGTH = 24  # Predict next 24 hours
NUM_SAMPLES = 20  # Number of probabilistic samples

print(f"\n📊 Forecast Configuration:")
print(f"   Context window: {CONTEXT_LENGTH} hours (7 days)")
print(f"   Prediction horizon: {PREDICTION_LENGTH} hours (1 day)")
print(f"   Probabilistic samples: {NUM_SAMPLES}")

# ==============================================================================
# 4. Rolling Forecast
# ==============================================================================
print("\n" + "=" * 80)
print("🔮 Running Zero-Shot Forecasts...")
print("=" * 80)

n_predictions = min(len(test_values) // PREDICTION_LENGTH, 100)  # Limit to 100 chunks for speed
predictions_chronos = []

print(f"   Forecasting {n_predictions} chunks ({n_predictions * PREDICTION_LENGTH} hours)")
print(f"   Estimated time: {n_predictions * 2:.0f}-{n_predictions * 4:.0f} seconds\n")

start_time = time.time()

for i in tqdm(range(n_predictions), desc="Chronos forecasting"):
    # Get context
    if i == 0:
        context = train_values[-CONTEXT_LENGTH:]
    else:
        all_data = np.concatenate([train_values, predictions_chronos])
        context = all_data[-CONTEXT_LENGTH:]
    
    # Convert to tensor
    context_tensor = torch.tensor(context, dtype=torch.float32).unsqueeze(0)
    
    # Generate forecast
    try:
        with torch.no_grad():
            forecast = pipeline.predict(
                inputs=context_tensor,
                prediction_length=PREDICTION_LENGTH,
                num_samples=NUM_SAMPLES,
            )
        
        # Take median of samples
        forecast_median = forecast.median(dim=1).values.squeeze().cpu().numpy()
        predictions_chronos.extend(forecast_median)
        
    except Exception as e:
        print(f"\n⚠️  Error at chunk {i}: {e}")
        # Use simple repeat of last value as fallback
        predictions_chronos.extend([context[-1]] * PREDICTION_LENGTH)

predictions_chronos = np.array(predictions_chronos)
inference_time = time.time() - start_time

print(f"\n✅ Forecasting complete!")
print(f"   Generated: {len(predictions_chronos):,} predictions")
print(f"   Total time: {inference_time:.1f}s")
print(f"   Per sample: {inference_time/len(predictions_chronos)*1000:.1f}ms")

# ==============================================================================
# 5. Evaluate
# ==============================================================================
print("\n" + "=" * 80)
print("📊 Evaluation Results")
print("=" * 80)

# Match lengths
y_test_chronos = test_values[:len(predictions_chronos)]

# Calculate metrics
mae_chronos = mean_absolute_error(y_test_chronos, predictions_chronos)
rmse_chronos = np.sqrt(mean_squared_error(y_test_chronos, predictions_chronos))
r2_chronos = r2_score(y_test_chronos, predictions_chronos)
mape_chronos = np.mean(np.abs((y_test_chronos - predictions_chronos) / y_test_chronos)) * 100

print(f"\n🏆 Chronos-T5-Small (Zero-Shot):")
print(f"   MAE:  {mae_chronos:.2f} MW")
print(f"   RMSE: {rmse_chronos:.2f} MW")
print(f"   R²:   {r2_chronos:.4f}")
print(f"   MAPE: {mape_chronos:.2f}%")
print(f"\n   Inference: {inference_time:.1f}s ({inference_time/len(predictions_chronos)*1000:.1f}ms/sample)")

# ==============================================================================
# 6. Comparison with Previous Models
# ==============================================================================
print("\n" + "=" * 80)
print("📈 Model Comparison")
print("=" * 80)

comparison_data = {
    'Model': [
        'XGBoost (Tuned)',
        'Chronos-T5-Small',
        'LSTM',
        'GRU',
        'XGBoost (Baseline)',
    ],
    'MAE_MW': [
        249.03,
        mae_chronos,
        251.53,
        252.32,
        269.47,
    ],
    'R2': [
        0.9825,
        r2_chronos,
        0.9822,
        0.9820,
        0.9817,
    ],
    'MAPE_%': [
        3.15,
        mape_chronos,
        3.48,
        3.49,
        3.41,
    ],
    'Training': [
        '7.6 min',
        'Zero-Shot',
        '3.4 min',
        '4.7 min',
        '0.6 s',
    ],
    'Type': [
        'Gradient Boosting',
        'Foundation Model',
        'Deep Learning',
        'Deep Learning',
        'Gradient Boosting',
    ]
}

comparison_df = pd.DataFrame(comparison_data)
comparison_df = comparison_df.sort_values('MAE_MW')

print("\n")
print(comparison_df.to_string(index=False))

# ==============================================================================
# 7. Save Results
# ==============================================================================
print("\n💾 Saving results...")

results_dir = Path('results/metrics')
comparison_df.to_csv(results_dir / 'solar_llm_comparison.csv', index=False)

# Save detailed results
chronos_results = pd.DataFrame({
    'Model': ['Chronos-T5-Small'],
    'MAE_MW': [mae_chronos],
    'RMSE_MW': [rmse_chronos],
    'R2': [r2_chronos],
    'MAPE_%': [mape_chronos],
    'Inference_Time_s': [inference_time],
    'Samples_Predicted': [len(predictions_chronos)]
})
chronos_results.to_csv(results_dir / 'chronos_results.csv', index=False)

print(f"✅ Results saved:")
print(f"   - {results_dir}/solar_llm_comparison.csv")
print(f"   - {results_dir}/chronos_results.csv")

# ==============================================================================
# 8. Final Summary
# ==============================================================================
print("\n" + "=" * 80)
print("🎉 Chronos Foundation Model Evaluation Complete!")
print("=" * 80)

print("\n🔍 Key Insights:")
if mae_chronos < 270:
    print("   ✅ Chronos performs competitively with tuned traditional models")
    print("   ✅ Zero-shot capability is impressive - no domain-specific training!")
    rank = (comparison_df['Model'] == 'Chronos-T5-Small').idxmax() + 1
    print(f"   ✅ Ranks #{rank} out of 5 models tested")
else:
    print("   ℹ️  Traditional ML (XGBoost) still leads for this specific domain")
    print("   ℹ️  Foundation models excel when training data is limited")

print("\n💡 When to use Chronos:")
print("   • Limited training data available")
print("   • Multiple time series domains")
print("   • Rapid prototyping needed")
print("   • Probabilistic forecasts required")

print("\n💡 When to use XGBoost:")
print("   • Domain-specific optimization possible")
print("   • Abundant training data")
print("   • Low latency requirements")
print("   • Interpretability needed")

print("\n" + "=" * 80)
print("Foundation Models represent the future of time series forecasting!")
print("=" * 80)
