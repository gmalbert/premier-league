# Precomputed Data

This folder contains preprocessed data for fast Streamlit app loading.

## What's Here
- `preprocessed_data.pkl` - Processed training/test data ready for ML models

## Why Precompute?
The app processes a large CSV file with feature engineering at startup. By precomputing this data:
- **Startup time**: Reduced from 30-60s to 5-10s (6-10x faster)
- **User experience**: Nearly instant loading
- **Server resources**: Less CPU usage on Streamlit Cloud

## How It Works
1. **Daily automation**: GitHub Actions runs `precompute_database.py` after data updates
2. **Processing**: CSV → Feature engineering → Train/test splits → Pickle file
3. **App loading**: Streamlit loads the pickle directly (fast) instead of processing CSV (slow)

## Updating
The data is automatically updated by GitHub Actions:
- **Trigger**: After CSV data changes or on schedule (2 AM UTC daily)
- **Workflow**: `.github/workflows/precompute-app-data.yml`
- **Manual**: Run `python precompute_database.py` locally

## File Details
- **Format**: Python pickle (binary)
- **Size**: ~5-10 MB
- **Contents**: NumPy arrays, feature names, metadata
- **Updated**: Automatically by CI/CD pipeline

## Fallback
If this file is missing or corrupted, the app automatically falls back to real-time CSV processing (slower but functional).
