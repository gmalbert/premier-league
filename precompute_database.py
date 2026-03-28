"""
Precompute Database Script

This script processes the raw historical data and saves it in an optimized format
for fast loading in the Streamlit app. This eliminates the expensive CSV parsing
and feature engineering that happens at app startup.

Run this script:
- Locally: python precompute_database.py
- Automated: Via GitHub Actions after data updates
"""

import pandas as pd
import numpy as np
import pickle
import os
from os import path
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
import time

warnings.filterwarnings('ignore')

DATA_DIR = 'data_files/'
OUTPUT_DIR = 'precomputed/'

def precompute_data():
    """
    Precompute expensive data processing operations for fast app loading.
    
    This function:
    1. Loads the raw CSV data
    2. Performs feature engineering and encoding
    3. Creates train/test splits
    4. Saves processed data to pickle for instant loading
    
    Expected speedup: 6-10x faster app startup (from 30-60s to 5-10s)
    """
    start_time = time.time()
    print("🚀 Starting data precomputation...")
    
    # Load raw data
    csv_path = path.join(DATA_DIR, 'combined_historical_data_with_calculations_new.csv')
    
    if not path.exists(csv_path):
        print(f"❌ Error: Data file not found at {csv_path}")
        return False
    
    print(f"📂 Loading data from {csv_path}...")
    df = pd.read_csv(csv_path, sep='\t')
    initial_rows = len(df)
    print(f"   Loaded {initial_rows:,} rows")
    
    # Data preparation (matches app logic exactly)
    print("🔄 Processing features...")
    target_map = {'H': 0, 'D': 1, 'A': 2}
    df = df[df['FullTimeResult'].isin(target_map.keys())].copy()
    df['target'] = df['FullTimeResult'].map(target_map)
    
    drop_cols = [
        'FullTimeResult', 'FullTimeHomeGoals', 'FullTimeAwayGoals',
        'HalfTimeResult', 'HalfTimeHomeGoals', 'HalfTimeAwayGoals',
        'HomeWin', 'AwayWin', 'Draw', 'WinningTeam',
        'HomePoints', 'AwayPoints', 'HomeTeamCumulativePoints', 'AwayTeamCumulativePoints',
        'MatchDate', 'KickoffTime', 'Season', 'Round', 'Venue', 'Referee',
        'HomeTeam', 'AwayTeam', 'Division'
    ]
    
    X = df.drop(columns=[col for col in drop_cols if col in df.columns] + ['target'], errors='ignore')
    y = df['target']
    
    # Process numeric features
    X_numeric = X.select_dtypes(include=[np.number]).drop(columns=drop_cols, errors='ignore')
    print(f"   Found {len(X_numeric.columns)} numeric features")
    
    # Encode categorical features
    cat_cols = X.select_dtypes(include=['object']).columns
    X_categorical = pd.DataFrame()
    
    if len(cat_cols) > 0:
        print(f"   Encoding {len(cat_cols)} categorical features...")
        for col in cat_cols:
            if col not in drop_cols:
                le = LabelEncoder()
                X_categorical[col] = le.fit_transform(X[col].astype(str))
    
    # Combine features
    X = pd.concat([X_numeric, X_categorical], axis=1)
    X = X.fillna(X.mean())
    
    # Ensure consistent feature names
    if isinstance(X, pd.DataFrame):
        X.columns = [f'feature_{i}' for i in range(X.shape[1])]
        feature_names = X.columns.tolist()
    
    X_processed = X.values
    y_processed = y.values
    
    # Create train/test split (consistent with app)
    print("✂️  Creating train/test split...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y_processed, 
        test_size=0.2, 
        random_state=42, 
        stratify=y_processed
    )
    
    # Package data for saving
    preprocessed_data = {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'feature_names': feature_names,
        'df_sample': df.head(1000),  # Small sample for quick operations
        'metadata': {
            'total_samples': len(X_processed),
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'num_features': X_processed.shape[1],
            'processed_date': pd.Timestamp.now().isoformat(),
            'source_file': csv_path
        }
    }
    
    # Save to disk
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = path.join(OUTPUT_DIR, 'preprocessed_data.pkl')
    
    print(f"💾 Saving preprocessed data to {output_path}...")
    with open(output_path, 'wb') as f:
        pickle.dump(preprocessed_data, f)
    
    # Calculate file sizes
    file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
    
    elapsed_time = time.time() - start_time
    
    # Print summary
    print("\n" + "="*60)
    print("✅ PRECOMPUTATION COMPLETE!")
    print("="*60)
    print(f"📊 Summary:")
    print(f"   • Training samples: {len(X_train):,}")
    print(f"   • Test samples: {len(X_test):,}")
    print(f"   • Total features: {X_processed.shape[1]}")
    print(f"   • Output file: {output_path}")
    print(f"   • File size: {file_size:.2f} MB")
    print(f"   • Processing time: {elapsed_time:.2f} seconds")
    print(f"\n🚀 Expected app startup speedup: 6-10x faster")
    print("="*60)
    
    return True

if __name__ == "__main__":
    success = precompute_data()
    exit(0 if success else 1)
