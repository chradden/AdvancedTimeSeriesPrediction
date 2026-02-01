"""
Quick Start Script - Test SMARD API Connection

This script downloads a small sample of energy data to verify everything works.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from data.smard_loader import load_smard_data
import pandas as pd


def main():
    print("=" * 70)
    print("ENERGIE-ZEITREIHEN-PROJEKT - QUICK START")
    print("=" * 70)
    print("\n🔄 Teste SMARD API-Verbindung...\n")
    
    try:
        # Download small sample - one week from 2023 (known to have data)
        start_date = '2023-01-01'
        end_date = '2023-01-07'
        
        print(f"Loading Solar data from {start_date} to {end_date}...")
        print("(This is just a test - your actual project will use 2-3 years of data)\n")
        
        df = load_smard_data(
            filter_name='solar',
            start_date=start_date,
            end_date=end_date,
            resolution='hour',
            cache_dir=Path('data/raw')
        )
        
        print("\n" + "=" * 70)
        print("✅ SUCCESS!")
        print("=" * 70)
        print(f"\nLoaded {len(df)} data points")
        print(f"\nFirst 5 rows:")
        print(df.head().to_string())
        print(f"\nBasic statistics:")
        print(df['value'].describe())
        
        print("\n" + "=" * 70)
        print("🎉 READY TO START!")
        print("=" * 70)
        print("\nNext steps:")
        print("1. Install all dependencies: pip install -r requirements.txt")
        print("2. Open Jupyter: jupyter notebook")
        print("3. Start with: notebooks/01_data_exploration.ipynb")
        print("\nHappy forecasting! 🚀")
        
    except Exception as e:
        print("\n" + "=" * 70)
        print("❌ ERROR")
        print("=" * 70)
        print(f"\n{type(e).__name__}: {e}")
        print("\nPossible solutions:")
        print("1. Check internet connection")
        print("2. Install required packages: pip install pandas requests")
        print("3. Try again in a few minutes (API might be temporarily unavailable)")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
