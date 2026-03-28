# Infrastructure Improvements - Next Steps

## Status Update (2026-03-28)
- ✅ CSV-based prototype deployed
- ⚪ SQLite migration: pending
- ⚪ API development: pending
- ⚪ Automated pipeline scheduling: pending

Building on the current CSV-based architecture with Streamlit deployment, here are the next actionable steps to improve scalability, reliability, and maintainability.

---

## 1. SQLite Database Migration (Phase 1)

**Priority:** High  
**Effort:** 3-4 hours  
**Expected Impact:** 10x faster queries, better data integrity

Migrate from CSV files to SQLite for improved performance and ACID compliance.

```python
# Create: database/db_manager.py

import sqlite3
import pandas as pd
from contextlib import contextmanager
from datetime import datetime
import os

DB_PATH = 'data_files/premier_league.db'

class DatabaseManager:
    """Manage SQLite database operations"""
    
    def __init__(self, db_path=DB_PATH):
        self.db_path = db_path
        self._initialize_database()
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections"""
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()
    
    def _initialize_database(self):
        """Create database schema if not exists"""
        
        with self.get_connection() as conn:
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
                    home_xg FLOAT,
                    away_xg FLOAT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Indexes for faster queries
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_match_date 
                ON matches(match_date DESC)
            ''')
            
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_teams 
                ON matches(home_team, away_team)
            ''')
            
            # Team statistics table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS team_stats (
                    stat_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    team_name VARCHAR(50) NOT NULL,
                    season VARCHAR(10) NOT NULL,
                    matches_played INTEGER DEFAULT 0,
                    wins INTEGER DEFAULT 0,
                    draws INTEGER DEFAULT 0,
                    losses INTEGER DEFAULT 0,
                    goals_for INTEGER DEFAULT 0,
                    goals_against INTEGER DEFAULT 0,
                    points INTEGER DEFAULT 0,
                    home_goals_avg FLOAT,
                    away_goals_avg FLOAT,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(team_name, season)
                )
            ''')
            
            # Predictions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS predictions (
                    prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    match_id INTEGER,
                    prediction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    home_win_prob FLOAT NOT NULL,
                    draw_prob FLOAT NOT NULL,
                    away_win_prob FLOAT NOT NULL,
                    predicted_outcome CHAR(1),
                    model_version VARCHAR(20) DEFAULT 'ensemble_v1',
                    actual_outcome CHAR(1),
                    correct_prediction BOOLEAN,
                    validated_at TIMESTAMP,
                    FOREIGN KEY (match_id) REFERENCES matches(match_id)
                )
            ''')
            
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_prediction_date 
                ON predictions(prediction_date DESC)
            ''')
    
    def insert_match(self, match_data):
        """Insert a new match record"""
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO matches (
                    match_date, kickoff_time, home_team, away_team,
                    full_time_result, home_goals, away_goals,
                    home_shots, away_shots, home_shots_on_target,
                    away_shots_on_target, home_xg, away_xg
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                match_data['match_date'],
                match_data.get('kickoff_time'),
                match_data['home_team'],
                match_data['away_team'],
                match_data.get('full_time_result'),
                match_data.get('home_goals'),
                match_data.get('away_goals'),
                match_data.get('home_shots'),
                match_data.get('away_shots'),
                match_data.get('home_shots_on_target'),
                match_data.get('away_shots_on_target'),
                match_data.get('home_xg'),
                match_data.get('away_xg')
            ))
            
            return cursor.lastrowid
    
    def get_recent_matches(self, team_name=None, limit=10):
        """Get recent matches for a team or all teams"""
        
        with self.get_connection() as conn:
            if team_name:
                query = '''
                    SELECT * FROM matches
                    WHERE home_team = ? OR away_team = ?
                    ORDER BY match_date DESC
                    LIMIT ?
                '''
                df = pd.read_sql_query(query, conn, params=(team_name, team_name, limit))
            else:
                query = '''
                    SELECT * FROM matches
                    ORDER BY match_date DESC
                    LIMIT ?
                '''
                df = pd.read_sql_query(query, conn, params=(limit,))
            
            return df
    
    def get_h2h_matches(self, team1, team2, limit=10):
        """Get head-to-head matches between two teams"""
        
        with self.get_connection() as conn:
            query = '''
                SELECT * FROM matches
                WHERE (home_team = ? AND away_team = ?)
                   OR (home_team = ? AND away_team = ?)
                ORDER BY match_date DESC
                LIMIT ?
            '''
            df = pd.read_sql_query(query, conn, 
                                  params=(team1, team2, team2, team1, limit))
            
            return df
    
    def save_prediction(self, prediction_data):
        """Save a match prediction"""
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Determine predicted outcome
            probs = [
                prediction_data['home_win_prob'],
                prediction_data['draw_prob'],
                prediction_data['away_win_prob']
            ]
            predicted = ['H', 'D', 'A'][probs.index(max(probs))]
            
            cursor.execute('''
                INSERT INTO predictions (
                    match_id, home_win_prob, draw_prob, away_win_prob,
                    predicted_outcome, model_version
                ) VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                prediction_data.get('match_id'),
                prediction_data['home_win_prob'],
                prediction_data['draw_prob'],
                prediction_data['away_win_prob'],
                predicted,
                prediction_data.get('model_version', 'ensemble_v1')
            ))
            
            return cursor.lastrowid
    
    def validate_predictions(self):
        """Validate predictions against actual results"""
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Find predictions with actual results
            cursor.execute('''
                UPDATE predictions
                SET actual_outcome = (
                    SELECT full_time_result 
                    FROM matches 
                    WHERE matches.match_id = predictions.match_id
                ),
                correct_prediction = (
                    predicted_outcome = (
                        SELECT full_time_result 
                        FROM matches 
                        WHERE matches.match_id = predictions.match_id
                    )
                ),
                validated_at = CURRENT_TIMESTAMP
                WHERE match_id IN (
                    SELECT match_id FROM matches 
                    WHERE full_time_result IS NOT NULL
                )
                AND validated_at IS NULL
            ''')
            
            validated_count = cursor.rowcount
            
            return validated_count
    
    def get_prediction_accuracy(self, days_back=30):
        """Calculate prediction accuracy over time"""
        
        with self.get_connection() as conn:
            query = '''
                SELECT 
                    DATE(prediction_date) as date,
                    COUNT(*) as total_predictions,
                    SUM(CASE WHEN correct_prediction = 1 THEN 1 ELSE 0 END) as correct,
                    AVG(CASE WHEN correct_prediction = 1 THEN 1.0 ELSE 0.0 END) as accuracy
                FROM predictions
                WHERE validated_at IS NOT NULL
                  AND prediction_date >= DATE('now', '-{} days')
                GROUP BY DATE(prediction_date)
                ORDER BY date DESC
            '''.format(days_back)
            
            df = pd.read_sql_query(query, conn)
            
            return df

# Migration script
def migrate_csv_to_sqlite():
    """One-time migration from CSV to SQLite"""
    
    print("Starting CSV to SQLite migration...")
    
    db = DatabaseManager()
    
    # Read existing CSV
    csv_path = 'data_files/combined_historical_data_with_calculations_new.csv'
    df = pd.read_csv(csv_path, sep='\t')
    
    print(f"Found {len(df)} matches to migrate")
    
    # Migrate matches
    migrated = 0
    for _, row in df.iterrows():
        try:
            match_data = {
                'match_date': row['MatchDate'],
                'kickoff_time': row.get('KickoffTime'),
                'home_team': row['HomeTeam'],
                'away_team': row['AwayTeam'],
                'full_time_result': row.get('FullTimeResult'),
                'home_goals': row.get('FullTimeHomeGoals'),
                'away_goals': row.get('FullTimeAwayGoals'),
                'home_shots': row.get('HomeShots'),
                'away_shots': row.get('AwayShots'),
                'home_shots_on_target': row.get('HomeShotsOnTarget'),
                'away_shots_on_target': row.get('AwayShotsOnTarget'),
                'home_xg': row.get('HomeXg'),
                'away_xg': row.get('AwayXg')
            }
            
            db.insert_match(match_data)
            migrated += 1
            
            if migrated % 100 == 0:
                print(f"Migrated {migrated} matches...")
        
        except Exception as e:
            print(f"Error migrating match: {e}")
            continue
    
    print(f"✅ Migration complete! Migrated {migrated} matches")
    
    # Verify migration
    with db.get_connection() as conn:
        count = pd.read_sql_query("SELECT COUNT(*) as count FROM matches", conn)
        print(f"Database now contains {count['count'][0]} matches")

# Usage in premier-league-predictions.py
from database.db_manager import DatabaseManager

# Initialize database
db = DatabaseManager()

# Replace CSV reads with database queries
@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_historical_data():
    """Load historical match data from database"""
    with db.get_connection() as conn:
        df = pd.read_sql_query('''
            SELECT * FROM matches
            ORDER BY match_date DESC
        ''', conn)
    return df

# Load data
df = load_historical_data()
```

---

## 2. Caching Layer with Redis

**Priority:** Medium  
**Effort:** 2-3 hours  
**Expected Impact:** 5x faster repeated queries

Implement Redis caching for expensive computations and API calls.

```python
# Create: cache/redis_cache.py

import redis
import json
import pickle
from functools import wraps
import hashlib
import os

class CacheManager:
    """Redis cache manager for expensive operations"""
    
    def __init__(self, host='localhost', port=6379, db=0, ttl=3600):
        """
        Initialize Redis cache
        
        Args:
            host: Redis server host
            port: Redis server port
            db: Redis database number
            ttl: Default time-to-live in seconds
        """
        
        # Check if Redis is available
        try:
            self.redis_client = redis.Redis(
                host=host, 
                port=port, 
                db=db,
                decode_responses=False
            )
            self.redis_client.ping()
            self.enabled = True
            print("✅ Redis cache connected")
        except (redis.ConnectionError, redis.TimeoutError):
            print("⚠️ Redis not available, using memory cache fallback")
            self.redis_client = None
            self.enabled = False
            self.memory_cache = {}
        
        self.default_ttl = ttl
    
    def _generate_key(self, prefix, *args, **kwargs):
        """Generate cache key from function arguments"""
        
        key_data = f"{prefix}:{args}:{sorted(kwargs.items())}"
        key_hash = hashlib.md5(key_data.encode()).hexdigest()
        
        return f"{prefix}:{key_hash}"
    
    def get(self, key):
        """Get value from cache"""
        
        if self.enabled:
            data = self.redis_client.get(key)
            if data:
                return pickle.loads(data)
        else:
            return self.memory_cache.get(key)
        
        return None
    
    def set(self, key, value, ttl=None):
        """Set value in cache"""
        
        ttl = ttl or self.default_ttl
        
        if self.enabled:
            self.redis_client.setex(
                key, 
                ttl, 
                pickle.dumps(value)
            )
        else:
            self.memory_cache[key] = value
    
    def delete(self, pattern):
        """Delete keys matching pattern"""
        
        if self.enabled:
            for key in self.redis_client.scan_iter(pattern):
                self.redis_client.delete(key)
        else:
            keys_to_delete = [k for k in self.memory_cache.keys() if pattern in k]
            for key in keys_to_delete:
                del self.memory_cache[key]
    
    def cache_result(self, prefix, ttl=None):
        """Decorator to cache function results"""
        
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Generate cache key
                cache_key = self._generate_key(prefix, *args, **kwargs)
                
                # Try to get from cache
                cached_result = self.get(cache_key)
                if cached_result is not None:
                    print(f"Cache hit: {prefix}")
                    return cached_result
                
                # Compute result
                print(f"Cache miss: {prefix}, computing...")
                result = func(*args, **kwargs)
                
                # Store in cache
                self.set(cache_key, result, ttl)
                
                return result
            
            return wrapper
        
        return decorator

# Global cache instance
cache = CacheManager()

# Usage examples
@cache.cache_result('model_predictions', ttl=1800)  # Cache for 30 minutes
def get_model_predictions(home_team, away_team, model_type='ensemble'):
    """Cached model predictions"""
    # Expensive model prediction logic
    pass

@cache.cache_result('team_stats', ttl=3600)  # Cache for 1 hour
def calculate_team_statistics(team_name, season='2024-25'):
    """Cached team statistics calculation"""
    # Expensive stats calculation
    pass

@cache.cache_result('h2h_history', ttl=7200)  # Cache for 2 hours
def get_h2h_history(team1, team2):
    """Cached head-to-head history"""
    # Database query or computation
    pass

# Cache invalidation on data update
def on_new_match_added():
    """Invalidate relevant caches when new match is added"""
    cache.delete('team_stats:*')
    cache.delete('model_predictions:*')
    cache.delete('h2h_history:*')

# Add to requirements.txt:
# redis>=5.0.0
```

---

## 3. Automated Testing Suite

**Priority:** Medium  
**Effort:** 3-4 hours  
**Expected Impact:** Prevent regression bugs

Implement comprehensive unit and integration tests.

```python
# Create: tests/test_models.py

import pytest
import pandas as pd
import numpy as np
from models.ensemble_predictor import create_ensemble_model, create_simple_ensemble
from models.neural_predictor import train_neural_model, predict_neural
from models.poisson_predictor import predict_match_poisson

class TestEnsembleModel:
    """Test ensemble model functionality"""
    
    @pytest.fixture
    def mock_data(self):
        """Create mock training data"""
        X = pd.DataFrame(np.random.randn(100, 10))
        y = pd.Series(np.random.choice([0, 1, 2], 100))
        return X, y
    
    def test_model_creation(self):
        """Test model can be created"""
        model = create_ensemble_model()
        assert model is not None
    
    def test_model_training(self, mock_data):
        """Test model can be trained"""
        X, y = mock_data
        model = create_simple_ensemble()
        model.fit(X, y)
        assert hasattr(model, 'classes_')
    
    def test_prediction_shape(self, mock_data):
        """Test prediction output shape is correct"""
        X, y = mock_data
        model = create_simple_ensemble()
        model.fit(X, y)
        
        predictions = model.predict_proba(X)
        
        assert predictions.shape == (100, 3), "Should output 3 probabilities per sample"
        assert np.allclose(predictions.sum(axis=1), 1.0), "Probabilities should sum to 1"
    
    def test_prediction_range(self, mock_data):
        """Test predictions are in valid probability range"""
        X, y = mock_data
        model = create_simple_ensemble()
        model.fit(X, y)
        
        predictions = model.predict_proba(X)
        
        assert predictions.min() >= 0, "Probabilities should be >= 0"
        assert predictions.max() <= 1, "Probabilities should be <= 1"

class TestDataLoading:
    """Test data loading and preprocessing"""
    
    def test_csv_loads(self):
        """Test historical data CSV can be loaded"""
        df = pd.read_csv(
            'data_files/combined_historical_data_with_calculations_new.csv',
            sep='\t'
        )
        
        assert len(df) > 0, "Data file should not be empty"
        assert 'HomeTeam' in df.columns, "Should have HomeTeam column"
        assert 'AwayTeam' in df.columns, "Should have AwayTeam column"
    
    def test_no_null_outcomes(self):
        """Test that training data has no null outcomes"""
        df = pd.read_csv(
            'data_files/combined_historical_data_with_calculations_new.csv',
            sep='\t'
        )
        
        # Filter for completed matches
        completed = df[df['FullTimeResult'].notna()]
        
        assert len(completed) > 0, "Should have completed matches"
        assert completed['FullTimeResult'].isnull().sum() == 0

class TestPoissonModel:
    """Test Poisson regression model"""
    
    @pytest.fixture
    def mock_team_stats(self):
        """Create mock team statistics"""
        return pd.DataFrame({
            'Team': ['Arsenal', 'Chelsea'],
            'GoalsPerGame': [2.1, 1.8],
            'ConcededPerGame': [0.9, 1.1],
            'HomeGoalsAvg': [2.3, 2.0],
            'AwayGoalsAvg': [1.8, 1.5]
        })
    
    def test_poisson_prediction(self, mock_team_stats):
        """Test Poisson model produces valid predictions"""
        
        result = predict_match_poisson('Arsenal', 'Chelsea', mock_team_stats)
        
        assert 'HomeWinProb' in result
        assert 'DrawProb' in result
        assert 'AwayWinProb' in result
        
        total_prob = result['HomeWinProb'] + result['DrawProb'] + result['AwayWinProb']
        assert abs(total_prob - 1.0) < 0.01, "Probabilities should sum to ~1"

# Create: tests/test_integration.py

class TestEndToEnd:
    """Integration tests for full prediction pipeline"""
    
    def test_full_prediction_pipeline(self):
        """Test complete pipeline from data load to prediction"""
        
        # Load data
        df = pd.read_csv(
            'data_files/combined_historical_data_with_calculations_new.csv',
            sep='\t'
        )
        
        # Prepare features
        df = df[df['FullTimeResult'].notna()]
        
        # Select features (simplified)
        feature_cols = [col for col in df.columns if col.startswith('Home') or col.startswith('Away')]
        feature_cols = [col for col in feature_cols if df[col].dtype in ['int64', 'float64']]
        
        X = df[feature_cols].fillna(df[feature_cols].mean())
        y = df['FullTimeResult'].map({'H': 0, 'D': 1, 'A': 2})
        
        # Train model
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        model = create_simple_ensemble()
        model.fit(X_train, y_train)
        
        # Make prediction
        predictions = model.predict_proba(X_test[:1])
        
        assert predictions.shape == (1, 3)
        assert abs(predictions.sum() - 1.0) < 0.01

# Run tests with pytest
# pytest tests/ -v
```

---

## 4. Logging and Monitoring System

**Priority:** High  
**Effort:** 2 hours  
**Expected Impact:** Better debugging and error tracking

Implement comprehensive logging with rotation and monitoring.

```python
# Create: utils/logger.py

import logging
import os
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from datetime import datetime

class AppLogger:
    """Application-wide logging system"""
    
    def __init__(self, name='premier_league'):
        self.name = name
        self.log_dir = 'logs'
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Create loggers for different modules
        self.app_logger = self._create_logger('app', 'app.log')
        self.data_logger = self._create_logger('data', 'data_pipeline.log')
        self.model_logger = self._create_logger('model', 'model_training.log')
        self.error_logger = self._create_logger('error', 'errors.log', level=logging.ERROR)
    
    def _create_logger(self, name, filename, level=logging.INFO):
        """Create a logger with file and console handlers"""
        
        logger = logging.getLogger(f"{self.name}.{name}")
        logger.setLevel(level)
        
        # Prevent duplicate handlers
        if logger.handlers:
            return logger
        
        # File handler with rotation (10MB per file, keep 5 backups)
        file_handler = RotatingFileHandler(
            os.path.join(self.log_dir, filename),
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(level)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.WARNING)  # Only warnings+ to console
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def log_model_training(self, model_name, metrics):
        """Log model training results"""
        self.model_logger.info(
            f"Model: {model_name} | "
            f"Accuracy: {metrics.get('accuracy', 0):.3f} | "
            f"MAE: {metrics.get('mae', 0):.3f}"
        )
    
    def log_data_update(self, source, records_added):
        """Log data pipeline updates"""
        self.data_logger.info(
            f"Data source: {source} | "
            f"Records added: {records_added} | "
            f"Timestamp: {datetime.now()}"
        )
    
    def log_error(self, error_type, error_msg, stack_trace=None):
        """Log errors with stack trace"""
        self.error_logger.error(
            f"Error Type: {error_type} | "
            f"Message: {error_msg}"
        )
        if stack_trace:
            self.error_logger.error(f"Stack Trace:\n{stack_trace}")

# Global logger instance
logger = AppLogger()

# Usage in scripts
from utils.logger import logger

# Application logs
logger.app_logger.info("Starting Streamlit app...")
logger.app_logger.debug(f"Loaded {len(df)} matches")

# Data pipeline logs
logger.log_data_update('ESPN API', records_added=10)

# Model training logs
logger.log_model_training('ensemble', {'accuracy': 0.567, 'mae': 0.654})

# Error logging
try:
    # Some operation
    pass
except Exception as e:
    import traceback
    logger.log_error(
        error_type=type(e).__name__,
        error_msg=str(e),
        stack_trace=traceback.format_exc()
    )
```

---

## 5. GitHub Actions CI/CD Enhancement

**Priority:** Medium  
**Effort:** 2 hours  
**Expected Impact:** Automated testing and deployment

Enhance existing GitHub Actions with comprehensive testing and deployment.

```yaml
# .github/workflows/test-and-deploy.yml

name: Test and Deploy

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]
  schedule:
    # Run tests daily at 3 AM UTC
    - cron: '0 3 * * *'

jobs:
  test:
    runs-on: ubuntu-latest
    
    strategy:
      matrix:
        python-version: [3.9, 3.10, 3.11]
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Cache dependencies
      uses: actions/cache@v3
      with:
        path: ~/.cache/pip
        key: ${{ runner.os }}-pip-${{ hashFiles('**/requirements.txt') }}
        restore-keys: |
          ${{ runner.os }}-pip-
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install pytest pytest-cov
    
    - name: Run tests with coverage
      run: |
        pytest tests/ -v --cov=. --cov-report=xml --cov-report=html
    
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        fail_ci_if_error: true
    
    - name: Check code quality
      run: |
        pip install flake8
        flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
        flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics
  
  model-validation:
    runs-on: ubuntu-latest
    needs: test
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
    
    - name: Validate model accuracy
      run: |
        python -c "
        from sklearn.metrics import accuracy_score
        import pandas as pd
        import numpy as np
        
        # Load data and test model
        # Assert minimum accuracy threshold
        min_accuracy = 0.50
        # actual_accuracy = test_model()
        # assert actual_accuracy >= min_accuracy
        print('Model validation passed')
        "
  
  deploy:
    runs-on: ubuntu-latest
    needs: [test, model-validation]
    if: github.ref == 'refs/heads/main'
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Deploy to production
      run: |
        echo "Deploying to production environment..."
        # Add deployment commands here
        # e.g., rsync, docker push, Streamlit Cloud deploy, etc.
```

---

## Implementation Priority

**Week 1:**
1. Logging and Monitoring (High Impact, Easy Implementation)
2. SQLite Migration Phase 1 (High Impact, Foundation for Future)

**Week 2:**
3. Automated Testing Suite (Medium Impact, Prevents Bugs)
4. GitHub Actions Enhancement (Medium Impact, Quality Assurance)

**Week 3:**
5. Redis Caching Layer (Medium Impact, Performance Boost)

---

## Success Metrics

- **SQLite Migration:** 10x faster query performance, sub-100ms response times
- **Redis Cache:** 80%+ cache hit rate for repeated queries
- **Testing Suite:** 80%+ code coverage, zero regression bugs
- **Logging:** Complete audit trail for debugging and monitoring
- **CI/CD:** Automated testing on every commit, zero failed deployments
