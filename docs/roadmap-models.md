# Model Improvements Roadmap

## Status Update (2026-03-28)
- ✅ XGBoost baseline implemented
- ✅ Ensemble model implemented
- ✅ Neural network implemented
- ✅ Poisson model implemented
- ✅ LSTM model implemented
- ✅ Hyperparameter tuning implemented
- ⚪ Ongoing evaluation/productionization

## Current Model: XGBoost Classifier
- **Type:** Gradient Boosting Decision Tree
- **Target:** 3-class (Home Win, Draw, Away Win)
- **Current Accuracy:** ~50-60% (typical for football prediction)

---

## Recommended Model Enhancements

### 1. Ensemble Model Approach ✅ **COMPLETED**
**Priority:** High  
**Complexity:** Medium  
**Expected Improvement:** +5-10% accuracy  
**Actual Improvement:** +3.5% accuracy, -0.038 MAE

Combine multiple models for more robust predictions. Implemented as VotingClassifier with XGBoost, Random Forest, Gradient Boosting, and Logistic Regression using soft voting with weighted probabilities.

**Features:**
- Pre-trained nightly via automated pipeline
- Session state persistence for immediate availability
- Simple ensemble for fast loading vs full ensemble for accuracy

```python
# Create: models/ensemble_predictor.py
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
import numpy as np

def create_ensemble_model():
    """Create ensemble of multiple classifiers"""
    
    # Individual models
    xgb = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        random_state=42
    )
    
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        random_state=42
    )
    
    gb = GradientBoostingClassifier(
        n_estimators=150,
        max_depth=5,
        random_state=42
    )
    
    lr = LogisticRegression(
        max_iter=1000,
        random_state=42
    )
    
    # Voting ensemble (soft voting for probabilities)
    ensemble = VotingClassifier(
        estimators=[
            ('xgb', xgb),
            ('rf', rf),
            ('gb', gb),
            ('lr', lr)
        ],
        voting='soft',
        weights=[2, 1.5, 1, 0.5]  # Higher weight for XGB
    )
    
    return ensemble

# Usage in premier-league-predictions.py
model = create_ensemble_model()
model.fit(X_train, y_train)
```

---

### 2. Neural Network with PyTorch ✅ **COMPLETED**
**Priority:** Medium  
**Complexity:** High  
**Expected Improvement:** +3-7% accuracy  
**Actual Improvement:** +4.9% accuracy vs XGBoost baseline

Deep learning approach using PyTorch with 3-layer neural network (128→64→32 neurons), batch normalization, and dropout regularization. Successfully implemented and integrated into the model comparison framework with UI button activation.

**Features:**
- Pre-trained nightly via automated pipeline
- Session state persistence of trained models
- 50 epochs with batch normalization and dropout
- On-demand retraining available via UI button
- Automatic integration with model comparison dashboard

**Architecture:**
- Input layer → 128 neurons → 64 neurons → 32 neurons → 3 outputs
- ReLU activation, batch normalization, 30% dropout
- Cross-entropy loss with Adam optimizer

---

### 3. Poisson Regression for Goal Prediction ✅ **COMPLETED**
**Priority:** Medium  
**Complexity:** Low  
**Expected Improvement:** Better for goal-based betting  
**Status:** ✅ IMPLEMENTED - Poisson regression model integrated into Streamlit app with model selection UI

Complete Poisson distribution-based goal prediction system with expected goals modeling, scoreline probability matrices, and match outcome conversion. Successfully integrated as an alternative prediction model in the Streamlit app with radio button selection between Simple Ensemble and Poisson Regression.

**Features:**
- Expected goals calculation using team attacking/defensive strengths
- Poisson probability mass function for goal distributions
- Scoreline probability matrix generation (up to 5 goals each)
- Match outcome probability conversion from scorelines
- Most likely score prediction
- Integrated into Streamlit app with model selection UI
- Real-time predictions for upcoming matches

**Architecture:**
- `models/poisson_predictor.py` - Core Poisson predictor class
- Team strength-based expected goals estimation
- Statistical modeling using scipy.stats.poisson
- UI integration with radio button model selection
- Automatic team statistics loading from `all_teams.csv`

**Usage:**
```python
from models.poisson_predictor import predict_match_poisson

# Predict Arsenal vs Chelsea
result = predict_match_poisson('Arsenal', 'Chelsea', team_stats_df)
print(f"Home Win: {result['HomeWinProb']:.3f}")
print(f"Draw: {result['DrawProb']:.3f}")
print(f"Away Win: {result['AwayWinProb']:.3f}")
print(f"Expected Goals: {result['ExpectedHomeGoals']:.2f}-{result['ExpectedAwayGoals']:.2f}")
print(f"Most Likely Score: {result['MostLikelyScore']}")
```

---

### 4. Time Series LSTM for Momentum ✅ **COMPLETED**
**Priority:** Low  
**Complexity:** High  
**Expected Improvement:** Captures temporal patterns  
**Status:** ✅ IMPLEMENTED - LSTM neural network for temporal momentum analysis integrated into Streamlit app

Complete Long Short-Term Memory (LSTM) implementation for capturing team momentum and temporal patterns in football performance. The model analyzes sequences of recent matches to predict future outcomes using deep learning.

**Features:**
- PyTorch-based LSTM neural network with configurable hidden layers
- Sequence-based feature extraction from recent team performance
- Temporal pattern recognition for momentum analysis
- Automatic sequence preparation from historical match data
- Integrated into Streamlit app with model selection UI
- Early stopping and validation during training
- Model persistence with pickle serialization

**Architecture:**
- LSTM layers with dropout regularization
- Fully connected layers with ReLU activation
- Sequence length: 5 matches per prediction
- Input features: shots, corners, fouls, cards, match results
- Output: 3-class probabilities (Home Win, Draw, Away Win)

**Training Process:**
- Prepares time series sequences from historical data
- Standardizes features using scikit-learn
- Trains with Adam optimizer and cross-entropy loss
- Implements early stopping to prevent overfitting
- Validates on held-out dataset during training

**Usage:**
```python
from models.lstm_predictor import train_lstm_model, predict_match_lstm

# Train model
predictor = train_lstm_model(historical_df, sequence_length=5, epochs=50)

# Predict match
result = predict_match_lstm('Arsenal', 'Chelsea', historical_df)
print(f"Home Win: {result['HomeWinProb']:.3f}")
print(f"Draw: {result['DrawProb']:.3f}")
print(f"Away Win: {result['AwayWinProb']:.3f}")
```

---

### 5. Hyperparameter Optimization ✅ **COMPLETED**
**Priority:** High  
**Complexity:** Low  
**Expected Improvement:** +2-5% accuracy  
**Actual Improvement:** +0.87% accuracy, -0.023 MAE

Implemented RandomizedSearchCV for XGBoost hyperparameter optimization with UI button integration. Users can now trigger expensive hyperparameter optimization on-demand rather than on every app startup, improving performance while maintaining access to advanced features.

**Features:**
- Pre-trained nightly via automated pipeline
- On-demand re-optimization available via UI button
- Session state persistence of optimization results
- Reduced search space (10 iterations × 3-fold CV) for reasonable runtime
- Real-time progress indicators and status updates
- Integration with model comparison dashboard

**Best Parameters Found:**
- `subsample`: 0.8
- `n_estimators`: 100  
- `min_child_weight`: 5
- `max_depth`: 3
- `learning_rate`: 0.1
- `gamma`: 0
- `colsample_bytree`: 0.8

---

## Model Comparison Framework

```python
# Create: compare_models.py
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd

def compare_all_models(X_train, X_test, y_train, y_test):
    """Compare performance of different models"""
    
    models = {
        'XGBoost': XGBClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'Ensemble': create_ensemble_model()
    }
    
    results = []
    
    for name, model in models.items():
        # Train
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        results.append({
            'Model': name,
            'Accuracy': accuracy,
            'Predictions': y_pred
        })
    
    # Display comparison
    comparison_df = pd.DataFrame(results)[['Model', 'Accuracy']]
    comparison_df = comparison_df.sort_values('Accuracy', ascending=False)
    
    return comparison_df
```

---

## Recommended Next Steps

1. ✅ **COMPLETED:** Implement ensemble model (+3.5% accuracy improvement)
2. ✅ **COMPLETED:** Experiment with neural networks (+4.9% accuracy vs XGBoost baseline) - Now with UI button activation
3. ✅ **COMPLETED:** Optimize current XGBoost hyperparameters (+0.87% accuracy improvement) - Now with UI button activation
4. ✅ **COMPLETED:** Add Poisson regression for goal predictions - Integrated into Streamlit app with model selection
5. ✅ **COMPLETED:** Implement LSTM Time Series for momentum analysis - Integrated into Streamlit app with model selection
6. **Month 1:** Build comprehensive model comparison dashboard
