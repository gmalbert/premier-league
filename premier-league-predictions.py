import streamlit as st
import pandas as pd
import numpy as np
from os import path
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, accuracy_score
from sklearn.inspection import permutation_importance

DATA_DIR = 'data_files/'

st.set_page_config(page_title="Premier League Historical Data", layout="wide", page_icon=path.join(DATA_DIR, 'favicon.ico'))

st.image(path.join(DATA_DIR, 'Premier_league_logo.png'))
st.title("Premier League Historical Data Viewer & Predictor")

csv_path = path.join(DATA_DIR, 'combined_historical_data_with_calculations.csv')

if not path.exists(csv_path):
    st.warning(f"No historical data file found at `{csv_path}`. Please add your CSV file to get started.")
    st.stop()

df = pd.read_csv(csv_path, sep='\t')

if st.checkbox("Show Raw Data"):
    st.subheader("Historical Data")
    st.dataframe(df)

if st.checkbox("Show Predictive Data"):

    # --- Data Preparation ---
    # Assume columns: HomeTeam, AwayTeam, FullTimeResult, plus features
    # Encode target: 0 = HomeWin, 1 = Draw, 2 = AwayWin
    target_map = {'H': 0, 'D': 1, 'A': 2}
    df = df[df['FullTimeResult'].isin(target_map.keys())].copy()
    df['target'] = df['FullTimeResult'].map(target_map)

    # Drop columns not useful for modeling or that leak the result
    drop_cols = [
        'MatchDate', 'KickoffTime', 'FullTimeResult', 'HomeTeam', 'AwayTeam', 'WinningTeam',
        'HomeWin', 'AwayWin', 'Draw',  'HalfTimeHomeWin', 'HalfTimeAwayWin', 'HalfTimeDraw', 'FullTimeHomeGoals', 'FullTimeAwayGoals',
        'HalfTimeResult', 'HalfTimeHomeGoals', 'HalfTimeAwayGoals', 'HomePoints', 'AwayPoints'
    ]
    X = df.drop(columns=[col for col in drop_cols if col in df.columns] + ['target'], errors='ignore')
    y = df['target']

    # Fill NA and encode categoricals
    for col in X.select_dtypes(include='object').columns:
        X[col] = X[col].astype('category').cat.codes
    X = X.fillna(X.mean(numeric_only=True))

    # Select only numeric columns
    X = X.select_dtypes(include=[np.number])

    # Clean column names for XGBoost compatibility
    X.columns = [str(col).replace('[','').replace(']','').replace('<','').replace('>','').replace(' ', '_') for col in X.columns]

    # Remove columns that are not 1D or have object dtype
    bad_cols = []
    for col in X.columns:
        if isinstance(X[col].iloc[0], (pd.Series, pd.DataFrame)) or X[col].dtype == 'O':
            bad_cols.append(col)
    if bad_cols:
        # st.warning(f"Removing columns with unsupported types for XGBoost: {bad_cols}")
        X = X.drop(columns=bad_cols)

    # --- Train/Test Split ---
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # --- XGBoost Model ---
    model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
    model.fit(X_train, y_train)

    # --- Predictions & MAE ---
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)

    st.subheader("Model Performance")
    st.write(f"Mean Absolute Error (MAE): **{mae:.3f}**")
    st.write(f"Accuracy: **{acc:.3f}**")

    # --- Monte Carlo Permutation Importance ---
    st.subheader("Monte Carlo Feature Importance (Permutation)")

    n_runs = 20
    importances = np.zeros((n_runs, X.shape[1]))

    for i in range(n_runs):
        result = permutation_importance(model, X_test, y_test, n_repeats=1, random_state=42+i, scoring='accuracy')
        importances[i, :] = result.importances_mean

    mean_importance = (importances.mean(axis=0) * 100)
    std_importance = (importances.std(axis=0) * 100)
    importance_df = pd.DataFrame({
        'Feature': X.columns,
        'MeanImportance': mean_importance,
        'StdImportance': std_importance
    }).rename(columns={'MeanImportance': 'Mean Importance (%)', 'StdImportance': 'Std Importance (%)'}).sort_values('Mean Importance (%)', ascending=False)

    st.dataframe(importance_df, width=600, hide_index=True)

