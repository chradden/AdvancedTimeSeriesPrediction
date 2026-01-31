#!/usr/bin/env python
# Fix wind onshore advanced testing script file references

with open('scripts/run_wind_onshore_advanced_testing.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Fix file references
replacements = {
    "'wind_power_train.csv'": "'wind_onshore_train.csv'",
    "'wind_power_val.csv'": "'wind_onshore_val.csv'",
    "'wind_power_test.csv'": "'wind_onshore_test.csv'",
    "'wind_power_all_models_extended.csv'": "'wind_onshore_all_models_extended.csv'",
    "wind_power_autoencoder_anomalies": "wind_onshore_autoencoder_anomalies",
    "wind_power_quantile_results": "wind_onshore_quantile_results",
    "wind_power_nbeats_results": "wind_onshore_nbeats_results",
    "wind_power_advanced_models_results": "wind_onshore_advanced_models_results",
    "wind_power_ADVANCED_TESTING_SUMMARY": "wind_onshore_ADVANCED_TESTING_SUMMARY",
    "wind_power_advanced": "wind_onshore_advanced"
}

for old, new in replacements.items():
    content = content.replace(old, new)

with open('scripts/run_wind_onshore_advanced_testing.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("✅ Fixed all file references!")
