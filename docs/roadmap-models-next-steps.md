# Model Improvements - Next Steps

## Status Update (2026-03-28)
- ✅ Ensemble/NN/Poisson/LSTM/hyperparameter optimization: completed
- ✅ Comprehensive Model Comparison Dashboard — plotly charts in tab2, all models compared
- ✅ Gradient Boosting Variants — `models/gradient_boosting_variants.py` (LightGBM + CatBoost + ensemble)
- ✅ Confidence Calibration — Platt Scaling via `CalibratedClassifierCV` in tab2
- ✅ SHAP Feature Analysis — `models/feature_analysis.py`, TreeExplainer charts in tab2
- ✅ Time-Based Cross-Validation — `TimeSeriesSplit` 5-fold button in tab2

Building on the successfully implemented ensemble, neural network, Poisson, LSTM, and hyperparameter optimization models, here are the next actionable steps to further enhance prediction accuracy and capabilities.

---

## 1. Comprehensive Model Comparison Dashboard

**Priority:** High  
**Effort:** 1-2 hours  
**Expected Impact:** Better visibility into model performance and selection

Create a unified dashboard comparing all implemented models with historical validation and performance metrics.

```python
# Add to premier-league-predictions.py in tab2

import plotly.graph_objects as go
from plotly.subplots import make_subplots

def create_model_comparison_dashboard(X_test, y_test, models_dict):
    """
    Create comprehensive model comparison with visualizations
    
    Args:
        X_test: Test features
        y_test: Test labels
        models_dict: Dictionary of {model_name: (model, scaler)} 
    """
    
    results = []
    predictions_by_model = {}
    
    for model_name, (model, scaler) in models_dict.items():
        # Get predictions
        if scaler is not None:
            X_scaled = scaler.transform(X_test)
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X_scaled)
            else:
                proba = model(torch.FloatTensor(X_scaled)).detach().numpy()
        else:
            proba = model.predict_proba(X_test)
        
        pred = np.argmax(proba, axis=1)
        
        # Calculate metrics
        acc = accuracy_score(y_test, pred)
        mae = mean_absolute_error(y_test, pred)
        
        # Per-class accuracy
        class_acc = {}
        for cls in [0, 1, 2]:  # Home, Draw, Away
            mask = y_test == cls
            if mask.sum() > 0:
                class_acc[cls] = accuracy_score(y_test[mask], pred[mask])
        
        results.append({
            'Model': model_name,
            'Overall Accuracy': acc,
            'MAE': mae,
            'Home Win Acc': class_acc.get(0, 0),
            'Draw Acc': class_acc.get(1, 0),
            'Away Win Acc': class_acc.get(2, 0)
        })
        
        predictions_by_model[model_name] = pred
    
    results_df = pd.DataFrame(results)
    
    # Create visualizations
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Overall Accuracy Comparison', 
                       'MAE Comparison',
                       'Per-Class Accuracy',
                       'Prediction Distribution'),
        specs=[[{'type': 'bar'}, {'type': 'bar'}],
               [{'type': 'bar'}, {'type': 'bar'}]]
    )
    
    # Overall Accuracy
    fig.add_trace(
        go.Bar(x=results_df['Model'], y=results_df['Overall Accuracy'], 
               name='Accuracy', marker_color='lightblue'),
        row=1, col=1
    )
    
    # MAE
    fig.add_trace(
        go.Bar(x=results_df['Model'], y=results_df['MAE'], 
               name='MAE', marker_color='lightcoral'),
        row=1, col=2
    )
    
    # Per-class accuracy
    for cls, label, color in [(0, 'Home Win', 'green'), 
                               (1, 'Draw', 'yellow'), 
                               (2, 'Away Win', 'red')]:
        fig.add_trace(
            go.Bar(x=results_df['Model'], 
                   y=results_df[f'{label} Acc'],
                   name=label, marker_color=color),
            row=2, col=1
        )
    
    # Prediction distribution for each model
    for model_name, preds in predictions_by_model.items():
        pred_counts = pd.Series(preds).value_counts().sort_index()
        fig.add_trace(
            go.Bar(x=['Home', 'Draw', 'Away'], 
                   y=[pred_counts.get(i, 0) for i in [0, 1, 2]],
                   name=model_name),
            row=2, col=2
        )
    
    fig.update_layout(height=800, showlegend=True, title_text="Model Performance Dashboard")
    
    return fig, results_df

# Usage in Predictive Data tab
st.subheader("📊 Model Comparison Dashboard")

if st.button("Generate Comprehensive Model Comparison"):
    with st.spinner("Comparing all models..."):
        models_to_compare = {
            'XGBoost': (xgb_model, None),
            'Ensemble': (ensemble_model, None)
        }
        
        if neural_available and neural_model:
            models_to_compare['Neural Network'] = (neural_model, neural_scaler)
        
        if optimized_available and optimized_xgb_model:
            models_to_compare['Optimized XGBoost'] = (optimized_xgb_model, None)
        
        fig, comparison_df = create_model_comparison_dashboard(
            X_test, y_test, models_to_compare
        )
        
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(comparison_df, hide_index=True)
```

---

## 2. Gradient Boosting Variants (LightGBM, CatBoost)

**Priority:** Medium  
**Effort:** 2-3 hours  
**Expected Impact:** +1-3% accuracy improvement

Implement alternative gradient boosting algorithms known for efficiency and performance.

```python
# Create: models/gradient_boosting_variants.py

import lightgbm as lgb
import catboost as cb
from sklearn.metrics import accuracy_score

class LightGBMPredictor:
    """LightGBM model for football match prediction"""
    
    def __init__(self):
        self.model = lgb.LGBMClassifier(
            objective='multiclass',
            num_class=3,
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1
        )
    
    def fit(self, X_train, y_train):
        """Train the model"""
        self.model.fit(X_train, y_train)
        return self
    
    def predict_proba(self, X):
        """Get prediction probabilities"""
        return self.model.predict_proba(X)
    
    def predict(self, X):
        """Get class predictions"""
        return self.model.predict(X)

class CatBoostPredictor:
    """CatBoost model for football match prediction"""
    
    def __init__(self):
        self.model = cb.CatBoostClassifier(
            iterations=200,
            depth=6,
            learning_rate=0.05,
            loss_function='MultiClass',
            classes_count=3,
            random_seed=42,
            verbose=False
        )
    
    def fit(self, X_train, y_train):
        """Train the model"""
        self.model.fit(X_train, y_train)
        return self
    
    def predict_proba(self, X):
        """Get prediction probabilities"""
        return self.model.predict_proba(X)
    
    def predict(self, X):
        """Get class predictions"""
        return self.model.predict(X)

def create_gradient_boosting_ensemble():
    """Create ensemble of gradient boosting variants"""
    from sklearn.ensemble import VotingClassifier
    from xgboost import XGBClassifier
    
    lgbm = LightGBMPredictor().model
    catb = CatBoostPredictor().model
    xgb = XGBClassifier(n_estimators=200, max_depth=6, random_state=42)
    
    ensemble = VotingClassifier(
        estimators=[
            ('lightgbm', lgbm),
            ('catboost', catb),
            ('xgboost', xgb)
        ],
        voting='soft',
        weights=[1.5, 1.5, 1.0]
    )
    
    return ensemble

# Add to requirements.txt:
# lightgbm>=4.0.0
# catboost>=1.2.0
```

**Integration into UI:**
```python
# In premier-league-predictions.py, add to model selection
model_options['Gradient Boosting Ensemble'] = create_gradient_boosting_ensemble()
```

---

## 3. Confidence Calibration with Platt Scaling

**Priority:** Medium  
**Effort:** 1 hour  
**Expected Impact:** More reliable probability estimates

Calibrate probability outputs to better reflect true prediction confidence.

```python
# Add to models/ directory

from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import cross_val_predict

def calibrate_model_probabilities(model, X_train, y_train, method='sigmoid'):
    """
    Calibrate model probability outputs using Platt scaling or isotonic regression
    
    Args:
        model: Trained classifier
        X_train: Training features
        y_train: Training labels
        method: 'sigmoid' for Platt scaling or 'isotonic' for isotonic regression
    
    Returns:
        Calibrated model
    """
    
    calibrated_model = CalibratedClassifierCV(
        model, 
        method=method,
        cv=5
    )
    
    calibrated_model.fit(X_train, y_train)
    
    return calibrated_model

# Usage in premier-league-predictions.py
if st.checkbox("Use Calibrated Probabilities"):
    st.info("Calibrating model probabilities for more accurate confidence estimates...")
    
    calibrated_ensemble = calibrate_model_probabilities(
        ensemble_model, X_train, y_train, method='sigmoid'
    )
    
    # Compare calibrated vs uncalibrated
    uncalibrated_proba = ensemble_model.predict_proba(X_test)
    calibrated_proba = calibrated_ensemble.predict_proba(X_test)
    
    # Display calibration comparison
    from sklearn.calibration import calibration_curve
    
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    
    for cls in range(3):
        fraction_of_positives_uncal, mean_predicted_value_uncal = calibration_curve(
            y_test == cls, uncalibrated_proba[:, cls], n_bins=10
        )
        fraction_of_positives_cal, mean_predicted_value_cal = calibration_curve(
            y_test == cls, calibrated_proba[:, cls], n_bins=10
        )
        
        ax[cls].plot(mean_predicted_value_uncal, fraction_of_positives_uncal, 
                     's-', label='Uncalibrated')
        ax[cls].plot(mean_predicted_value_cal, fraction_of_positives_cal, 
                     'o-', label='Calibrated')
        ax[cls].plot([0, 1], [0, 1], 'k--', label='Perfect')
        ax[cls].set_title(['Home Win', 'Draw', 'Away Win'][cls])
        ax[cls].legend()
    
    st.pyplot(fig)
```

---

## 4. Feature Importance Analysis with SHAP

**Priority:** Medium  
**Effort:** 2 hours  
**Expected Impact:** Better model interpretability

Use SHAP (SHapley Additive exPlanations) for advanced feature importance analysis.

```python
# Create: models/feature_analysis.py

import shap
import matplotlib.pyplot as plt
import streamlit as st

def analyze_feature_importance_shap(model, X_train, X_test, feature_names):
    """
    Generate SHAP values and visualizations for model interpretability
    
    Args:
        model: Trained model
        X_train: Training features
        X_test: Test features
        feature_names: List of feature names
    """
    
    # Create SHAP explainer
    explainer = shap.TreeExplainer(model)
    
    # Calculate SHAP values
    shap_values = explainer.shap_values(X_test)
    
    # Summary plot (shows feature importance)
    fig1, ax1 = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test, feature_names=feature_names, 
                     show=False, plot_type='bar', class_names=['Home', 'Draw', 'Away'])
    plt.tight_layout()
    
    # Detailed summary plot for each class
    figs = []
    for i, class_name in enumerate(['Home Win', 'Draw', 'Away Win']):
        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(shap_values[i], X_test, feature_names=feature_names,
                         show=False, title=f'Feature Impact on {class_name}')
        plt.tight_layout()
        figs.append(fig)
    
    return fig1, figs, shap_values

# Integration in Streamlit
if st.checkbox("Show SHAP Feature Analysis"):
    with st.spinner("Generating SHAP explanations..."):
        summary_fig, class_figs, shap_vals = analyze_feature_importance_shap(
            ensemble_model, X_train, X_test, feature_names=X_train.columns
        )
        
        st.subheader("Overall Feature Importance (SHAP)")
        st.pyplot(summary_fig)
        
        tabs = st.tabs(['Home Win', 'Draw', 'Away Win'])
        for i, tab in enumerate(tabs):
            with tab:
                st.pyplot(class_figs[i])

# Add to requirements.txt:
# shap>=0.42.0
```

---

## 5. Time-Based Cross-Validation

**Priority:** High  
**Effort:** 1 hour  
**Expected Impact:** More realistic performance estimates

Implement proper time-series cross-validation to prevent data leakage.

```python
# Add to train_models.py or premier-league-predictions.py

from sklearn.model_selection import TimeSeriesSplit

def time_based_cross_validation(X, y, model, n_splits=5):
    """
    Perform time-series cross-validation for temporal data
    
    Returns accuracy scores for each fold
    """
    
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = []
    
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train_fold = X.iloc[train_idx]
        X_test_fold = X.iloc[test_idx]
        y_train_fold = y.iloc[train_idx]
        y_test_fold = y.iloc[test_idx]
        
        # Train and evaluate
        model.fit(X_train_fold, y_train_fold)
        y_pred = model.predict(X_test_fold)
        acc = accuracy_score(y_test_fold, y_pred)
        
        scores.append({
            'Fold': fold + 1,
            'Accuracy': acc,
            'Train_Size': len(train_idx),
            'Test_Size': len(test_idx)
        })
    
    return pd.DataFrame(scores)

# Usage in UI
if st.button("Run Time-Based Cross-Validation"):
    st.subheader("⏱️ Time-Series Cross-Validation Results")
    
    with st.spinner("Running 5-fold time-series CV..."):
        cv_results = time_based_cross_validation(
            X, y, create_ensemble_model(), n_splits=5
        )
        
        st.dataframe(cv_results, hide_index=True)
        
        st.metric(
            "Average CV Accuracy",
            f"{cv_results['Accuracy'].mean():.3f}",
            f"±{cv_results['Accuracy'].std():.3f}"
        )
```

---

## Implementation Priority

**Week 1:**
1. Comprehensive Model Comparison Dashboard (High Impact, Quick Win)
2. Time-Based Cross-Validation (High Impact, Important for Accuracy)

**Week 2:**
3. Confidence Calibration with Platt Scaling (Medium Impact, Improves UX)

**Week 3:**
4. Gradient Boosting Variants (Medium Impact, Performance Boost)

**Week 4:**
5. SHAP Feature Analysis (Medium Impact, Better Insights)

---

## Success Metrics

- **Model Comparison Dashboard:** Used by users to select best model for their needs
- **Gradient Boosting Variants:** +1-3% accuracy improvement over current ensemble
- **Calibration:** Probability estimates within 5% of actual outcomes
- **SHAP Analysis:** Clear identification of top 10 most influential features
- **Time-Series CV:** Realistic accuracy estimates with <2% variance between folds
