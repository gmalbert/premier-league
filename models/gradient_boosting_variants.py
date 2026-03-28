"""
Gradient Boosting Variants: LightGBM + CatBoost predictors and ensemble.
"""

import pickle
import numpy as np
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.ensemble import VotingClassifier
from xgboost import XGBClassifier

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False


class LightGBMPredictor:
    """LightGBM model for football match prediction."""

    def __init__(self):
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("lightgbm is not installed. Run: pip install lightgbm")
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
            verbose=-1,
        )

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def predict(self, X):
        return self.model.predict(X)


class CatBoostPredictor:
    """CatBoost model for football match prediction."""

    def __init__(self):
        if not CATBOOST_AVAILABLE:
            raise ImportError("catboost is not installed. Run: pip install catboost")
        self.model = cb.CatBoostClassifier(
            iterations=200,
            depth=6,
            learning_rate=0.05,
            loss_function='MultiClass',
            classes_count=3,
            random_seed=42,
            verbose=False,
        )

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def predict(self, X):
        return self.model.predict(X)


def create_gradient_boosting_ensemble():
    """
    Create a soft-voting ensemble of LightGBM, CatBoost, and XGBoost.
    Raises ImportError if lightgbm or catboost are not installed.
    """
    lgbm = LightGBMPredictor().model
    catb = CatBoostPredictor().model
    xgb = XGBClassifier(n_estimators=200, max_depth=6, random_state=42, eval_metric='mlogloss')

    ensemble = VotingClassifier(
        estimators=[
            ('lightgbm', lgbm),
            ('catboost', catb),
            ('xgboost', xgb),
        ],
        voting='soft',
        weights=[1.5, 1.5, 1.0],
    )
    return ensemble


def train_gradient_boosting_variants(X_train, y_train, X_test, y_test):
    """
    Train LightGBM, CatBoost and a combined ensemble; return metrics dict and fitted models.

    Returns:
        dict: {
            'lgbm': {'model': ..., 'accuracy': ..., 'mae': ...},
            'catboost': {'model': ..., 'accuracy': ..., 'mae': ...},
            'gb_ensemble': {'model': ..., 'accuracy': ..., 'mae': ...},
        }
    """
    results = {}

    if LIGHTGBM_AVAILABLE:
        lgbm_predictor = LightGBMPredictor()
        lgbm_predictor.fit(X_train, y_train)
        lgbm_pred = lgbm_predictor.predict(X_test)
        results['lgbm'] = {
            'model': lgbm_predictor,
            'accuracy': accuracy_score(y_test, lgbm_pred),
            'mae': mean_absolute_error(y_test, lgbm_pred),
        }

    if CATBOOST_AVAILABLE:
        catb_predictor = CatBoostPredictor()
        catb_predictor.fit(X_train, y_train)
        catb_pred = catb_predictor.predict(X_test)
        results['catboost'] = {
            'model': catb_predictor,
            'accuracy': accuracy_score(y_test, catb_pred),
            'mae': mean_absolute_error(y_test, catb_pred),
        }

    if LIGHTGBM_AVAILABLE and CATBOOST_AVAILABLE:
        gb_ensemble = create_gradient_boosting_ensemble()
        gb_ensemble.fit(X_train, y_train)
        gb_pred = gb_ensemble.predict(X_test)
        results['gb_ensemble'] = {
            'model': gb_ensemble,
            'accuracy': accuracy_score(y_test, gb_pred),
            'mae': mean_absolute_error(y_test, gb_pred),
        }

    return results


def save_gradient_boosting_models(results, models_dir='models'):
    """Persist trained gradient boosting models to disk."""
    import os
    for name, info in results.items():
        path = os.path.join(models_dir, f'{name}_model.pkl')
        with open(path, 'wb') as f:
            pickle.dump(info['model'], f)
