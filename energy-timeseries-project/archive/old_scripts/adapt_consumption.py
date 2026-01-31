#!/usr/bin/env python
# Quick script to adapt wind onshore to consumption

with open('scripts/run_consumption_extended_pipeline.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Global replacements
replacements = {
    'wind_onshore': 'consumption',
    'Wind Onshore': 'Consumption',
    'WIND ONSHORE': 'CONSUMPTION',
    'wind_power': 'consumption',
    'Wind Power': 'Consumption',
    'MW': 'MW',  # Keep MW (consumption is also in MW)
}

for old, new in replacements.items():
    content = content.replace(old, new)

with open('scripts/run_consumption_extended_pipeline.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("✅ Adapted consumption script!")
