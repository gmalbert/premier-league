"""
Model Versioning with MLflow
Wraps train_models.py to record every training run in MLflow so you can
compare experiments, roll back to prior versions, and promote the best model.

Usage:
    python models/train_with_mlflow.py

View UI:
    mlflow ui --backend-store-uri sqlite:///mlflow.db
    # then open http://localhost:5000
"""

import os
import sys
import time
import pickle
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from pathlib import Path
from os import path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error, f1_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

# Ensure imports resolve from project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from models.ensemble_predictor import create_simple_ensemble

DATA_DIR = 'data_files/'
MODELS_DIR = 'models/'
MLFLOW_DB = 'sqlite:///mlflow.db'
EXPERIMENT_NAME = 'premier-league-predictor'

# ── MLflow setup ──────────────────────────────────────────────────────────────
mlflow.set_tracking_uri(MLFLOW_DB)
mlflow.set_experiment(EXPERIMENT_NAME)


# ── Data loading (mirrors train_models.py) ────────────────────────────────────

def load_and_preprocess_data():
    csv_path = path.join(DATA_DIR, 'combined_historical_data_with_calculations_new.csv')
    if not path.exists(csv_path):
        raise FileNotFoundError(f"Processed data not found: {csv_path}")

    df = pd.read_csv(csv_path, sep='\t')
    df.columns = (df.columns
                  .str.replace('<', '_').str.replace('>', '_')
                  .str.replace('[', '_').str.replace(']', '_'))

    target_map = {'H': 0, 'D': 1, 'A': 2}
    y = df['FullTimeResult'].map(target_map)

    exclude_cols = [
        'FullTimeResult', 'FullTimeHomeGoals', 'FullTimeAwayGoals',
        'HalfTimeResult', 'HalfTimeHomeGoals', 'HalfTimeAwayGoals',
        'HomeWin', 'AwayWin', 'Draw', 'WinningTeam',
        'HomePoints', 'AwayPoints', 'HomeTeamCumulativePoints', 'AwayTeamCumulativePoints',
        'MatchDate', 'KickoffTime', 'Season', 'Round', 'Venue', 'Referee',
        'HomeTeam', 'AwayTeam', 'Division',
    ]

    X_numeric = df.select_dtypes(include=[np.number]).drop(columns=exclude_cols, errors='ignore')
    cat_cols = df.select_dtypes(include=['object']).columns
    X_categorical = pd.DataFrame()
    for col in cat_cols:
        if col not in exclude_cols:
            le = LabelEncoder()
            X_categorical[col] = le.fit_transform(df[col].astype(str))

    X = pd.concat([X_numeric, X_categorical], axis=1).fillna(0)
    X.columns = [f'feature_{i}' for i in range(X.shape[1])]
    return X.values, y.values


# ── Training helpers ──────────────────────────────────────────────────────────

def _log_metrics(y_true, y_pred, prefix=''):
    acc = accuracy_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, average='weighted')
    mlflow.log_metrics({
        f'{prefix}accuracy': acc,
        f'{prefix}mae': mae,
        f'{prefix}f1': f1,
    })
    return acc, mae, f1


def train_xgboost(X_train, X_test, y_train, y_test, params: dict | None = None):
    """Train XGBoost baseline and log to MLflow."""
    default_params = dict(eval_metric='mlogloss', random_state=42)
    if params:
        default_params.update(params)

    with mlflow.start_run(run_name='xgb_baseline', nested=True):
        mlflow.log_params(default_params)
        model = XGBClassifier(**default_params)
        model.fit(X_train, y_train)
        acc, mae, f1 = _log_metrics(y_test, model.predict(X_test), prefix='test_')
        mlflow.xgboost.log_model(model, artifact_path='xgb_model')
        print(f"  XGBoost  acc={acc:.3f}  mae={mae:.3f}  f1={f1:.3f}")
    return model


def train_ensemble(X_train, X_test, y_train, y_test):
    """Train simple VotingClassifier ensemble and log to MLflow."""
    with mlflow.start_run(run_name='ensemble', nested=True):
        model = create_simple_ensemble()
        model.fit(X_train, y_train)
        acc, mae, f1 = _log_metrics(y_test, model.predict(X_test), prefix='test_')
        mlflow.sklearn.log_model(model, artifact_path='ensemble_model')
        print(f"  Ensemble acc={acc:.3f}  mae={mae:.3f}  f1={f1:.3f}")
    return model


# ── Main parent run ───────────────────────────────────────────────────────────

def run_tracked_training():
    print("Loading data...")
    X, y = load_and_preprocess_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Train: {X_train.shape}  Test: {X_test.shape}")

    os.makedirs(MODELS_DIR, exist_ok=True)

    with mlflow.start_run(run_name=f'full_pipeline_{time.strftime("%Y%m%d_%H%M%S")}'):
        mlflow.log_param('n_train', len(X_train))
        mlflow.log_param('n_test', len(X_test))
        mlflow.log_param('n_features', X_train.shape[1])

        print("Training XGBoost baseline...")
        xgb_model = train_xgboost(X_train, X_test, y_train, y_test)

        print("Training Ensemble...")
        ensemble_model = train_ensemble(X_train, X_test, y_train, y_test)

        # Persist alongside existing pkl models so the app can still load them
        with open(path.join(MODELS_DIR, 'xgb_baseline.pkl'), 'wb') as f:
            pickle.dump(xgb_model, f)
        with open(path.join(MODELS_DIR, 'ensemble_model.pkl'), 'wb') as f:
            pickle.dump(ensemble_model, f)

        print("Models saved to models/ and logged to MLflow.")
        print(f"View runs:  mlflow ui --backend-store-uri {MLFLOW_DB}")


# ── Promotion helper ──────────────────────────────────────────────────────────

def load_best_model(metric='test_accuracy'):
    """
    Load the best-performing XGBoost run from MLflow by a given metric.
    Falls back to the local pkl if no runs exist.
    """
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        print("No MLflow experiment found; falling back to local pkl.")
        with open(path.join(MODELS_DIR, 'xgb_baseline.pkl'), 'rb') as f:
            return pickle.load(f)

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="tags.mlflow.runName = 'xgb_baseline'",
        order_by=[f"metrics.{metric} DESC"],
        max_results=1,
    )
    if not runs:
        print("No qualifying runs found; falling back to local pkl.")
        with open(path.join(MODELS_DIR, 'xgb_baseline.pkl'), 'rb') as f:
            return pickle.load(f)

    best_run = runs[0]
    model_uri = f"runs:/{best_run.info.run_id}/xgb_model"
    print(f"Loading best run {best_run.info.run_id}  {metric}={best_run.data.metrics.get(metric, '?'):.3f}")
    return mlflow.xgboost.load_model(model_uri)


if __name__ == '__main__':
    run_tracked_training()
