# Technical Infrastructure Roadmap

## Status Update (2026-03-28)
- ✅ CSV-based architecture working
- ⚪ DB migration (SQLite/PostgreSQL) pending
- ⚪ API layer pending
- ✅ Automated Pipeline — `automation/scheduler.py` created (schedule-based nightly runs)
- ✅ Model Versioning — `models/train_with_mlflow.py` created (MLflow experiment tracking)
- ✅ Caching Layer — `cache/redis_cache.py` created (Redis + in-memory fallback)

## Current Architecture
- **Framework:** Streamlit
- **ML:** XGBoost, scikit-learn
- **Data:** Pandas, NumPy
- **Storage:** CSV files (tab-separated)
- **Deployment:** Local

---

## Recommended Infrastructure Improvements

### 1. Database Migration
**Priority:** High  
**Effort:** Medium  
**Impact:** High

Move from CSV files to SQLite/PostgreSQL for better performance.

```python
# Create: database/setup.py
import sqlite3
import pandas as pd
from os import path

DB_PATH = 'data_files/premier_league.db'

def create_database():
    """Initialize database schema"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Matches table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS matches (
            match_id INTEGER PRIMARY KEY AUTOINCREMENT,
            match_date DATE NOT NULL,
            kickoff_time TIME,
            home_team VARCHAR(50) NOT NULL,
            away_team VARCHAR(50) NOT NULL,
            full_time_result CHAR(1),
            home_goals INTEGER,
            away_goals INTEGER,
            home_shots INTEGER,
            away_shots INTEGER,
            home_shots_on_target INTEGER,
            away_shots_on_target INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            INDEX idx_date (match_date),
            INDEX idx_teams (home_team, away_team)
        )
    ''')
    
    # Teams table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS teams (
            team_id INTEGER PRIMARY KEY AUTOINCREMENT,
            team_name VARCHAR(50) UNIQUE NOT NULL,
            current_form VARCHAR(10),
            home_goals_avg FLOAT,
            away_goals_avg FLOAT,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Predictions table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
            match_id INTEGER,
            prediction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            home_win_prob FLOAT,
            draw_prob FLOAT,
            away_win_prob FLOAT,
            model_version VARCHAR(20),
            FOREIGN KEY (match_id) REFERENCES matches(match_id)
        )
    ''')
    
    conn.commit()
    conn.close()

def migrate_csv_to_db():
    """Migrate existing CSV data to database"""
    conn = sqlite3.connect(DB_PATH)
    
    # Read CSV
    df = pd.read_csv('data_files/combined_historical_data_with_calculations.csv', sep='\t')
    
    # Prepare columns
    df_clean = df[[
        'MatchDate', 'KickoffTime', 'HomeTeam', 'AwayTeam',
        'FullTimeResult', 'FullTimeHomeGoals', 'FullTimeAwayGoals',
        'HomeShots', 'AwayShots', 'HomeShotsOnTarget', 'AwayShotsOnTarget'
    ]].rename(columns={
        'MatchDate': 'match_date',
        'KickoffTime': 'kickoff_time',
        'HomeTeam': 'home_team',
        'AwayTeam': 'away_team',
        'FullTimeResult': 'full_time_result',
        'FullTimeHomeGoals': 'home_goals',
        'FullTimeAwayGoals': 'away_goals',
        'HomeShots': 'home_shots',
        'AwayShots': 'away_shots',
        'HomeShotsOnTarget': 'home_shots_on_target',
        'AwayShotsOnTarget': 'away_shots_on_target'
    })
    
    # Write to database
    df_clean.to_sql('matches', conn, if_exists='replace', index=False)
    
    conn.close()
    print(f"✓ Migrated {len(df_clean)} matches to database")

# Query helper
def query_matches(start_date=None, end_date=None, team=None):
    """Query matches with filters"""
    conn = sqlite3.connect(DB_PATH)
    
    query = "SELECT * FROM matches WHERE 1=1"
    params = []
    
    if start_date:
        query += " AND match_date >= ?"
        params.append(start_date)
    
    if end_date:
        query += " AND match_date <= ?"
        params.append(end_date)
    
    if team:
        query += " AND (home_team = ? OR away_team = ?)"
        params.extend([team, team])
    
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    
    return df
```

**Usage in app:**
```python
# Replace CSV reads with database queries
from database.setup import query_matches

df = query_matches(start_date='2025-01-01')
```

---

### 2. API Development
**Priority:** Medium  
**Effort:** High  
**Impact:** High

Create REST API for predictions.

```python
# Create: api/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import date
import pandas as pd
import pickle

app = FastAPI(title="Premier League Predictor API")

# Load model at startup
with open('models/xgboost_model.pkl', 'rb') as f:
    model = pickle.load(f)

class MatchPredictionRequest(BaseModel):
    home_team: str
    away_team: str
    match_date: date

class PredictionResponse(BaseModel):
    home_win_prob: float
    draw_prob: float
    away_win_prob: float
    predicted_winner: str
    confidence: float

@app.get("/")
def read_root():
    return {"message": "Premier League Predictor API", "version": "1.0"}

@app.get("/teams")
def get_teams():
    """Get list of all Premier League teams"""
    teams = query_matches()['home_team'].unique().tolist()
    return {"teams": sorted(teams)}

@app.post("/predict", response_model=PredictionResponse)
def predict_match(request: MatchPredictionRequest):
    """Predict match outcome"""
    
    # Get team stats
    team_stats = pd.read_csv('data_files/all_teams.csv', sep='\t')
    
    # Prepare features
    features = prepare_prediction_features(
        request.home_team,
        request.away_team,
        team_stats
    )
    
    # Make prediction
    proba = model.predict_proba([features])[0]
    
    # Determine winner
    winner_idx = proba.argmax()
    winners = ['Home', 'Draw', 'Away']
    
    return PredictionResponse(
        home_win_prob=float(proba[0]),
        draw_prob=float(proba[1]),
        away_win_prob=float(proba[2]),
        predicted_winner=winners[winner_idx],
        confidence=float(proba[winner_idx])
    )

@app.get("/upcoming")
def get_upcoming_fixtures():
    """Get upcoming fixtures with predictions"""
    df = pd.read_csv('data_files/upcoming_fixtures.csv')
    return df.to_dict(orient='records')

@app.get("/history/{team}")
def get_team_history(team: str, limit: int = 10):
    """Get recent match history for a team"""
    matches = query_matches(team=team)
    recent = matches.sort_values('match_date', ascending=False).head(limit)
    return recent.to_dict(orient='records')
```

**Run API:**
```bash
pip install fastapi uvicorn
uvicorn api.main:app --reload
```

**API Documentation:** Auto-generated at `http://localhost:8000/docs`

---

### 3. Automated Data Pipeline
**Priority:** High  
**Effort:** Medium  
**Impact:** Very High

Schedule automatic data updates.

```python
# Create: automation/scheduler.py
import schedule
import time
import subprocess
from datetime import datetime

def update_historical_data():
    """Run data update pipeline"""
    print(f"[{datetime.now()}] Starting data update...")
    
    # Step 1: Download raw data
    result1 = subprocess.run(['python', 'combine_raw_data.py'], capture_output=True)
    if result1.returncode != 0:
        print(f"Error downloading data: {result1.stderr}")
        return
    
    # Step 2: Process data
    result2 = subprocess.run(['python', 'prepare_model_data.py'], capture_output=True)
    if result2.returncode != 0:
        print(f"Error processing data: {result2.stderr}")
        return
    
    print(f"[{datetime.now()}] Data update complete!")

def update_fixtures():
    """Fetch upcoming fixtures"""
    print(f"[{datetime.now()}] Updating fixtures...")
    subprocess.run(['python', 'fetch_upcoming_fixtures.py'])
    print(f"[{datetime.now()}] Fixtures updated!")

# Schedule tasks
schedule.every().day.at("03:00").do(update_historical_data)  # 3 AM daily
schedule.every().hour.do(update_fixtures)  # Every hour

if __name__ == '__main__':
    print("Scheduler started...")
    
    # Run immediately on startup
    update_historical_data()
    update_fixtures()
    
    # Keep running
    while True:
        schedule.run_pending()
        time.sleep(60)
```

**Run as background service:**
```bash
# Windows (using Task Scheduler)
# Create task: python automation/scheduler.py

# Linux (using cron)
# Add to crontab:
# 0 3 * * * cd /path/to/project && python automation/scheduler.py
```

---

### 4. Model Versioning
**Priority:** Medium  
**Effort:** Low  
**Impact:** Medium

Track and version models with MLflow.

```python
# Create: models/train_with_mlflow.py
import mlflow
import mlflow.sklearn
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("premier-league-predictor")

def train_and_log_model(X_train, X_test, y_train, y_test, params):
    """Train model and log to MLflow"""
    
    with mlflow.start_run():
        # Log parameters
        mlflow.log_params(params)
        
        # Train model
        model = XGBClassifier(**params)
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1)
        
        # Log model
        mlflow.sklearn.log_model(model, "model")
        
        # Log feature importance
        importance_df = pd.DataFrame({
            'feature': X_train.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        importance_df.to_csv('feature_importance.csv', index=False)
        mlflow.log_artifact('feature_importance.csv')
        
        print(f"Model logged! Accuracy: {accuracy:.3f}")
        
        return model

# Load best model
def load_production_model():
    """Load best performing model from MLflow"""
    client = mlflow.tracking.MlflowClient()
    
    # Get best run
    experiment = client.get_experiment_by_name("premier-league-predictor")
    runs = client.search_runs(
        experiment.experiment_id,
        order_by=["metrics.accuracy DESC"],
        max_results=1
    )
    
    if runs:
        best_run = runs[0]
        model_uri = f"runs:/{best_run.info.run_id}/model"
        model = mlflow.sklearn.load_model(model_uri)
        return model
    
    return None
```

**Install MLflow:**
```bash
pip install mlflow
```

**View UI:**
```bash
mlflow ui
# Visit http://localhost:5000
```

---

### 5. Caching Layer
**Priority:** Medium  
**Effort:** Low  
**Impact:** High

Add Redis caching for frequently accessed data.

```python
# Create: cache/redis_cache.py
import redis
import json
import pandas as pd
from functools import wraps

# Redis connection
redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)

def cache_result(expiration=3600):
    """Decorator to cache function results"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key
            cache_key = f"{func.__name__}:{str(args)}:{str(kwargs)}"
            
            # Try to get from cache
            cached = redis_client.get(cache_key)
            if cached:
                return json.loads(cached)
            
            # Execute function
            result = func(*args, **kwargs)
            
            # Store in cache
            redis_client.setex(
                cache_key,
                expiration,
                json.dumps(result, default=str)
            )
            
            return result
        
        return wrapper
    return decorator

# Usage
@cache_result(expiration=1800)  # 30 minutes
def get_team_stats(team_name):
    """Cached team statistics lookup"""
    df = pd.read_csv('data_files/all_teams.csv', sep='\t')
    team_data = df[df['Team'] == team_name]
    return team_data.to_dict(orient='records')
```

---

### 6. Logging System
**Priority:** High  
**Effort:** Low  
**Impact:** Medium

Comprehensive logging for debugging and monitoring.

```python
# Create: utils/logger.py
import logging
from logging.handlers import RotatingFileHandler
import os

def setup_logger(name, log_file='app.log', level=logging.INFO):
    """Setup application logger"""
    
    # Create logs directory
    os.makedirs('logs', exist_ok=True)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # File handler (with rotation)
    file_handler = RotatingFileHandler(
        f'logs/{log_file}',
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(level)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# Usage in scripts
from utils.logger import setup_logger

logger = setup_logger('data_fetcher')
logger.info("Starting data fetch...")
logger.error("Failed to fetch data", exc_info=True)
```

---

### 7. Testing Framework
**Priority:** Medium  
**Effort:** Medium  
**Impact:** High

Add unit and integration tests.

```python
# Create: tests/test_predictions.py
import pytest
import pandas as pd
from models.ensemble_predictor import create_ensemble_model

def test_model_prediction_shape():
    """Test that model outputs correct shape"""
    model = create_ensemble_model()
    
    # Mock data
    X = pd.DataFrame([[1, 2, 3, 4, 5]] * 10)
    y = pd.Series([0, 1, 2] * 3 + [0])
    
    model.fit(X, y)
    predictions = model.predict_proba(X)
    
    assert predictions.shape == (10, 3), "Should output 3 probabilities per sample"
    assert predictions.sum(axis=1).allclose(1.0), "Probabilities should sum to 1"

def test_data_loading():
    """Test data loading functionality"""
    df = pd.read_csv('data_files/combined_historical_data_with_calculations.csv', sep='\t')
    
    assert len(df) > 0, "Data file should not be empty"
    assert 'HomeTeam' in df.columns, "Should have HomeTeam column"
    assert 'AwayTeam' in df.columns, "Should have AwayTeam column"

# Run tests
# pytest tests/
```

---

## Infrastructure Timeline

**Phase 1 (Week 1):**
- Logging system
- Automated data pipeline

**Phase 2 (Week 2-3):**
- Database migration
- Model versioning with MLflow

**Phase 3 (Month 2):**
- API development
- Caching layer

**Phase 4 (Month 3):**
- Testing framework
- CI/CD pipeline
- Production deployment
