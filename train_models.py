"""
Pre-train ML models for Premier League predictions
This script is run by the automated nightly pipeline to pre-train models
so they're ready for the next day, improving app startup performance.
"""

import pandas as pd
import numpy as np
import pickle
import os
import time
from os import path
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, accuracy_score
from sklearn.preprocessing import LabelEncoder
from models.ensemble_predictor import create_simple_ensemble
from models.neural_predictor import train_neural_model, predict_neural
from models.poisson_evaluation import evaluate_poisson_file
from optimize_model import optimize_xgboost

DATA_DIR = 'data_files/'
MODELS_DIR = 'models/'

def load_and_preprocess_data():
    """Load processed data and prepare for training"""
    csv_path = path.join(DATA_DIR, 'combined_historical_data_with_calculations_new.csv')

    if not path.exists(csv_path):
        raise FileNotFoundError(f"Processed data file not found: {csv_path}")

    df = pd.read_csv(csv_path, sep='\t')

    # Clean column names to be XGBoost compatible (remove <, >, [, ])
    df.columns = df.columns.str.replace('<', '_').str.replace('>', '_').str.replace('[', '_').str.replace(']', '_')

    # Target variable (3-class: Home Win=0, Draw=1, Away Win=2)
    target_map = {'H': 0, 'D': 1, 'A': 2}
    y = df['FullTimeResult'].map(target_map)

    # Feature selection (exclude leaky columns)
    exclude_cols = [
        'FullTimeResult', 'FullTimeHomeGoals', 'FullTimeAwayGoals',
        'HalfTimeResult', 'HalfTimeHomeGoals', 'HalfTimeAwayGoals',
        'HomeWin', 'AwayWin', 'Draw', 'WinningTeam',
        'HomePoints', 'AwayPoints', 'HomeTeamCumulativePoints', 'AwayTeamCumulativePoints',
        'MatchDate', 'KickoffTime', 'Season', 'Round', 'Venue', 'Referee',
        'HomeTeam', 'AwayTeam', 'Division'
    ]

    # Get numeric features only
    X_numeric = df.select_dtypes(include=[np.number]).drop(columns=exclude_cols, errors='ignore')

    # Handle categorical columns by encoding them
    cat_cols = df.select_dtypes(include=['object']).columns
    X_categorical = pd.DataFrame()
    for col in cat_cols:
        if col not in exclude_cols:
            le = LabelEncoder()
            X_categorical[col] = le.fit_transform(df[col].astype(str))

    # Combine numeric and categorical features
    X = pd.concat([X_numeric, X_categorical], axis=1)

    # Fill any remaining NaN values
    X = X.fillna(X.mean())

    # Ensure X is a DataFrame with clean column names
    if isinstance(X, pd.DataFrame):
        # Reset column names to generic names to avoid XGBoost issues
        X.columns = [f'feature_{i}' for i in range(X.shape[1])]

    # Convert to numpy array to ensure compatibility with XGBoost
    X = X.values

    return X, y

def train_and_save_models():
    """Train all models and save them to disk"""
    start_time = time.time()
    print(f"🚀 Starting model training pipeline at {time.strftime('%H:%M:%S')}")
    print(f"📋 Code version: XGBoost column fix v1.1 - {time.strftime('%Y-%m-%d %H:%M:%S')}")

    print("Loading and preprocessing data...")
    data_start = time.time()
    X, y = load_and_preprocess_data()
    print(".2f")

    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Create models directory if it doesn't exist
    os.makedirs(MODELS_DIR, exist_ok=True)

    # 1. Train XGBoost baseline
    print("Training XGBoost baseline...")
    xgb_start = time.time()
    xgb_model = XGBClassifier(eval_metric='mlogloss', random_state=42)
    xgb_model.fit(X_train, y_train)

    xgb_pred = xgb_model.predict(X_test)
    xgb_acc = accuracy_score(y_test, xgb_pred)
    xgb_mae = mean_absolute_error(y_test, xgb_pred)

    print(".3f")
    print(".2f")

    # Save XGBoost model
    with open(path.join(MODELS_DIR, 'xgb_baseline.pkl'), 'wb') as f:
        pickle.dump(xgb_model, f)

    # 2. Train Ensemble model
    print("Training Ensemble model...")
    ensemble_start = time.time()
    ensemble_model = create_simple_ensemble()
    ensemble_model.fit(X_train, y_train)

    ensemble_pred = ensemble_model.predict(X_test)
    ensemble_acc = accuracy_score(y_test, ensemble_pred)
    ensemble_mae = mean_absolute_error(y_test, ensemble_pred)

    print(".3f")
    print(".2f")

    # Save Ensemble model
    with open(path.join(MODELS_DIR, 'ensemble_model.pkl'), 'wb') as f:
        pickle.dump(ensemble_model, f)

    # 3. Train Neural Network
    print("Training Neural Network (this may take several minutes)...")
    neural_start = time.time()
    try:
        neural_model, neural_scaler = train_neural_model(X_train, y_train, epochs=50, batch_size=32)

        neural_pred_proba = predict_neural(neural_model, neural_scaler, X_test)
        neural_pred = np.argmax(neural_pred_proba, axis=1)
        neural_acc = accuracy_score(y_test, neural_pred)
        neural_mae = mean_absolute_error(y_test, neural_pred)

        print(".3f")
        print(".2f")

        # Save Neural Network model and scaler
        with open(path.join(MODELS_DIR, 'neural_model.pkl'), 'wb') as f:
            pickle.dump(neural_model, f)
        with open(path.join(MODELS_DIR, 'neural_scaler.pkl'), 'wb') as f:
            pickle.dump(neural_scaler, f)
            
    except Exception as e:
        print(f"⚠️  Neural Network training failed: {e}")
        print("Continuing without neural network model...")
        neural_model = None
        neural_scaler = None
        neural_acc = neural_mae = 0

    # 4. Train Optimized XGBoost
    print("Training Optimized XGBoost (hyperparameter search)...")
    opt_start = time.time()
    optimized_xgb_model = optimize_xgboost(X_train, y_train)

    opt_xgb_pred = optimized_xgb_model.predict(X_test)
    opt_xgb_acc = accuracy_score(y_test, opt_xgb_pred)
    opt_xgb_mae = mean_absolute_error(y_test, opt_xgb_pred)

    print(".3f")
    print(".2f")

    # Save Optimized XGBoost model
    with open(path.join(MODELS_DIR, 'optimized_xgb.pkl'), 'wb') as f:
        pickle.dump(optimized_xgb_model, f)

    # Save model performance metrics
    performance = {
        'xgb_baseline': {'accuracy': xgb_acc, 'mae': xgb_mae},
        'ensemble': {'accuracy': ensemble_acc, 'mae': ensemble_mae},
        'optimized_xgb': {'accuracy': opt_xgb_acc, 'mae': opt_xgb_mae}
    }
    
    # Only include neural network if it was successfully trained
    if neural_model is not None:
        performance['neural'] = {'accuracy': neural_acc, 'mae': neural_mae}

    # evaluate poisson metrics on the same historical data
    try:
        poisson_metrics = evaluate_poisson_file(path.join(DATA_DIR, 'combined_historical_data_with_calculations_new.csv'))
        performance['poisson'] = poisson_metrics
        # append to history CSV
        hist_path = path.join(DATA_DIR, 'poisson_metrics_history.csv')
        hist_df = pd.DataFrame([{
            'date': pd.Timestamp.now(),
            'home_mae': poisson_metrics['home_mae'],
            'away_mae': poisson_metrics['away_mae'],
            'home_rmse': poisson_metrics['home_rmse'],
            'away_rmse': poisson_metrics['away_rmse'],
            'outcome_acc': poisson_metrics['outcome_acc']
        }])
        if path.exists(hist_path):
            hist_df.to_csv(hist_path, mode='a', header=False, index=False)
        else:
            hist_df.to_csv(hist_path, index=False)
    except Exception as e:
        print(f"⚠️  Poisson evaluation failed: {e}")

    with open(path.join(MODELS_DIR, 'model_performance.pkl'), 'wb') as f:
        pickle.dump(performance, f)

    total_time = time.time() - start_time
    print("All models trained and saved successfully!")
    print(".2f")
    print("\nModel Performance Summary:")
    for model_name, metrics in performance.items():
        if model_name == 'poisson':
            # poisson entry contains detailed metrics
            print(f"poisson: home_mae={metrics['home_mae']:.3f}, away_mae={metrics['away_mae']:.3f}, outcome_acc={metrics['outcome_acc']:.3f}")
        else:
            print(f"{model_name}: Accuracy={metrics['accuracy']:.3f}, MAE={metrics['mae']:.3f}")

if __name__ == "__main__":
    train_and_save_models()