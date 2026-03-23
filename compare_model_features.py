"""
compare_model_features.py
=========================
Compare model accuracy BEFORE and AFTER the new feature engineering additions
from api-football-integration-guide.md (Enhancements 3, 4, 6, 7).

NEW FEATURES ADDED:
  Enhancement 3 - Team Season Statistics (from existing data + API stats)
    HomeCleanSheetPct_L10, AwayCleanSheetPct_L10
    HomeFailedToScorePct_L10, AwayFailedToScorePct_L10
    HomeDefVsAwayAttack, AwayDefVsHomeAttack
    API_Home_<stat>, API_Away_<stat>  (clean sheets, formations, goals/min, etc.)

  Enhancement 4 - League Position Features
    HomePointsPerGame, AwayPointsPerGame, PointsPerGameDiff
    HomePointsZScore, AwayPointsZScore, PointsZScoreDiff
    HomeGoalDiffSeason, AwayGoalDiffSeason
    API_Home_Standing_Rank, API_Away_Standing_Rank, API_StandingsRankDiff

  Home/Away Split Form
    HomeTeamPointsLast5_HomeOnly, AwayTeamPointsLast5_AwayOnly
    VenueAdjustedFormDiff
    HomeWinStreakHome_L5, AwayWinStreakAway_L5

Usage:
    python compare_model_features.py
"""

import pandas as pd
import numpy as np
from os import path
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.inspection import permutation_importance
import xgboost as xgb
import warnings

warnings.filterwarnings('ignore')

DATA_DIR = 'data_files/'
OLD_FILE = path.join(DATA_DIR, 'combined_historical_data_with_calculations.csv')
NEW_FILE = path.join(DATA_DIR, 'combined_historical_data_with_calculations_new.csv')

# Columns that leak the match result and must be excluded from features
LEAK_COLS = [
    'FullTimeResult', 'FullTimeHomeGoals', 'FullTimeAwayGoals',
    'HalfTimeResult', 'HalfTimeHomeGoals', 'HalfTimeAwayGoals',
    'HomeWin', 'AwayWin', 'Draw', 'WinningTeam',
    'HomePoints', 'AwayPoints',
    'MatchDate', 'KickoffTime', 'Season', 'Round', 'Venue', 'Referee',
    'HomeTeam', 'AwayTeam', 'Division',
]

# New feature columns added by the latest enhancements
NEW_FEATURE_COLS = [
    'HomePointsPerGame', 'AwayPointsPerGame', 'PointsPerGameDiff',
    'HomePointsZScore', 'AwayPointsZScore', 'PointsZScoreDiff',
    'HomeGoalDiffSeason', 'AwayGoalDiffSeason',
    'HomeCleanSheetPct_L10', 'AwayCleanSheetPct_L10',
    'HomeFailedToScorePct_L10', 'AwayFailedToScorePct_L10',
    'HomeDefVsAwayAttack', 'AwayDefVsHomeAttack',
    'HomeTeamPointsLast5_HomeOnly', 'AwayTeamPointsLast5_AwayOnly',
    'VenueAdjustedFormDiff', 'HomeWinStreakHome_L5', 'AwayWinStreakAway_L5',
]


def prepare_dataset(filepath):
    """Load CSV, drop leaky cols, encode categoricals, return X, y, col_names."""
    print(f"  Loading {path.basename(filepath)} ...", end=' ')
    df = pd.read_csv(filepath, sep='\t')
    print(f"({len(df)} rows, {len(df.columns)} cols)")

    target_map = {'H': 0, 'D': 1, 'A': 2}
    df = df[df['FullTimeResult'].isin(target_map.keys())].copy()
    df['target'] = df['FullTimeResult'].map(target_map)

    X = df.drop(columns=[c for c in LEAK_COLS if c in df.columns] + ['target'],
                errors='ignore')

    # Encode categoricals
    for col in X.select_dtypes(include='object').columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

    X = X.select_dtypes(include=[np.number])
    X = X.fillna(X.mean())

    # Sanitize column names for XGBoost (no [, ], <, >)
    X.columns = [c.replace('[', '_').replace(']', '_').replace('<', '_lt_').replace('>', '_gt_')
                 .replace(' ', '_').replace(',', '_') for c in X.columns]

    y = df['target']

    return X, y


def train_and_evaluate(X, y, label, cv_folds=5):
    """Train XGBoost with cross-validation; return mean accuracy + std."""
    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        
        eval_metric='mlogloss',
        random_state=42,
        verbosity=0,
        n_jobs=-1,
    )
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
    print(f"  [{label}] CV accuracy: {scores.mean():.4f} ± {scores.std():.4f}  "
          f"(min {scores.min():.4f}, max {scores.max():.4f})")
    return scores


def get_feature_importances(X, y, top_n=20):
    """Train on 80% of data, compute permutation importance on held-out 20%."""
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                               random_state=42, stratify=y)
    model = xgb.XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        eval_metric='mlogloss',
        random_state=42, verbosity=0, n_jobs=-1,
    )
    model.fit(X_tr, y_tr)
    test_acc = accuracy_score(y_te, model.predict(X_te))

    perm = permutation_importance(model, X_te, y_te, n_repeats=10,
                                  random_state=42, n_jobs=-1)
    imp_df = pd.DataFrame({
        'Feature': X.columns,
        'ImportanceMean': perm.importances_mean,
        'ImportanceStd': perm.importances_std,
    }).sort_values('ImportanceMean', ascending=False)

    return imp_df.head(top_n), test_acc, model


def highlight_new_features(imp_df, new_cols):
    """Mark features that are newly added."""
    imp_df = imp_df.copy()
    imp_df['IsNew'] = imp_df['Feature'].isin(new_cols)
    return imp_df


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 70)
    print("FEATURE ENGINEERING IMPACT ANALYSIS")
    print("=" * 70)

    # -------- Check files exist --------
    if not path.exists(OLD_FILE):
        print(f"⚠  Old file not found: {OLD_FILE}")
        print("  Run the app once to generate it, or copy from a backup.")
        raise SystemExit(1)
    if not path.exists(NEW_FILE):
        print(f"⚠  New file not found: {NEW_FILE}")
        print("  Run `python prepare_model_data.py` first.")
        raise SystemExit(1)

    # -------- Prepare datasets --------
    print("\n[1] Loading datasets")
    X_old, y_old = prepare_dataset(OLD_FILE)
    X_new, y_new = prepare_dataset(NEW_FILE)

    print(f"\n  Old feature count : {X_old.shape[1]}")
    print(f"  New feature count : {X_new.shape[1]}")
    added = set(X_new.columns) - set(X_old.columns)
    removed = set(X_old.columns) - set(X_new.columns)
    print(f"  Features added    : {len(added)}")
    print(f"  Features removed  : {len(removed)}")

    # -------- Cross-Validated Accuracy --------
    print("\n[2] Cross-validated accuracy (5-fold XGBoost)")
    scores_old = train_and_evaluate(X_old, y_old, "BEFORE new features")
    scores_new = train_and_evaluate(X_new, y_new, "AFTER  new features")

    delta_mean = scores_new.mean() - scores_old.mean()
    direction = "📈 IMPROVED" if delta_mean > 0 else ("📉 DECLINED" if delta_mean < 0 else "➡️  UNCHANGED")
    print(f"\n  Accuracy delta    : {delta_mean:+.4f}  ({delta_mean * 100:+.2f} pp)  {direction}")

    # -------- t-test for significance --------
    from scipy import stats as sp_stats
    t_stat, p_val = sp_stats.ttest_rel(scores_new, scores_old)
    sig = "✅ statistically significant" if p_val < 0.05 else "❌ not statistically significant (p={:.3f})".format(p_val)
    print(f"  Paired t-test     : t={t_stat:.3f}, p={p_val:.4f}  {sig}")

    # -------- Per-class breakdown --------
    print("\n[3] Classification report — NEW model (held-out 20%)")
    X_tr, X_te, y_tr, y_te = train_test_split(X_new, y_new, test_size=0.2,
                                               random_state=42, stratify=y_new)
    final_model = xgb.XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric='mlogloss',
        random_state=42, verbosity=0, n_jobs=-1,
    )
    final_model.fit(X_tr, y_tr)
    y_pred = final_model.predict(X_te)
    print(classification_report(y_te, y_pred, target_names=['Home Win', 'Draw', 'Away Win']))

    # -------- Feature importances --------
    print("\n[4] Top 25 features by permutation importance (NEW model)")
    imp_df, test_acc, _ = get_feature_importances(X_new, y_new, top_n=25)
    imp_df = highlight_new_features(imp_df, set(added) | set(NEW_FEATURE_COLS))
    print(f"  Hold-out test accuracy: {test_acc:.4f}")
    print()
    col_width = 45
    print(f"  {'Feature':<{col_width}} {'Mean Imp':>10}  {'Std':>8}  New?")
    print(f"  {'-' * col_width} {'--------':>10}  {'---':>8}  ----")
    for _, row in imp_df.iterrows():
        marker = " ★ NEW" if row['IsNew'] else ""
        print(f"  {row['Feature']:<{col_width}} {row['ImportanceMean']:>10.5f}  "
              f"{row['ImportanceStd']:>8.5f}{marker}")

    # -------- New features specifically --------
    new_in_model = [c for c in NEW_FEATURE_COLS if c in X_new.columns]
    if new_in_model:
        print(f"\n[5] Importance of newly-added features (subset)")
        new_imp = imp_df[imp_df['Feature'].isin(new_in_model)]
        if new_imp.empty:
            # Recompute importances for all new features (they may not be in top25)
            from sklearn.inspection import permutation_importance as perm_imp
            X_tr2, X_te2, y_tr2, y_te2 = train_test_split(X_new, y_new, test_size=0.2,
                                                           random_state=42, stratify=y_new)
            m2 = xgb.XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                                    eval_metric='mlogloss',
                                    random_state=42, verbosity=0, n_jobs=-1)
            m2.fit(X_tr2, y_tr2)
            perm = perm_imp(m2, X_te2[new_in_model], y_te2, n_repeats=10,
                            random_state=42, n_jobs=-1)
            new_imp = pd.DataFrame({
                'Feature': new_in_model,
                'ImportanceMean': perm.importances_mean,
                'ImportanceStd': perm.importances_std,
            }).sort_values('ImportanceMean', ascending=False)

        print(f"  {'Feature':<{col_width}} {'Mean Imp':>10}  {'Std':>8}")
        print(f"  {'-' * col_width} {'--------':>10}  {'---':>8}")
        for _, row in new_imp.iterrows():
            print(f"  {row['Feature']:<{col_width}} {row['ImportanceMean']:>10.5f}  "
                  f"{row['ImportanceStd']:>8.5f}")

    # -------- Summary --------
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Old features : {X_old.shape[1]}")
    print(f"  New features : {X_new.shape[1]}  (+{X_new.shape[1] - X_old.shape[1]})")
    print(f"  Old CV acc   : {scores_old.mean():.4f}")
    print(f"  New CV acc   : {scores_new.mean():.4f}")
    print(f"  Delta        : {delta_mean:+.4f}  ({delta_mean * 100:+.2f} percentage points)")
    print(f"  Significance : {sig}")
    print("=" * 70)
