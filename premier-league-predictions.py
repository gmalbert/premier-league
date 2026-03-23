import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
import warnings
from os import path
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.inspection import permutation_importance
from scipy import stats
from datetime import datetime

# Suppress XGBoost serialization warnings (models are retrained nightly)
warnings.filterwarnings('ignore', message='.*If you are loading a serialized model.*')
warnings.filterwarnings('ignore', category=UserWarning, module='xgboost')
from team_name_mapping import normalize_team_name
from generate_pdf_report import generate_statistical_report, generate_quick_report
from models.ensemble_predictor import create_ensemble_model, create_simple_ensemble
from models.neural_predictor import train_neural_model, predict_neural
from models.poisson_predictor import predict_match_poisson
from models.poisson_evaluation import evaluate_poisson_file

# cache the expensive historical evaluation so Streamlit doesn't recompute on every interaction
@st.cache_data
def get_poisson_metrics(csv_path: str = 'data_files/combined_historical_data_with_calculations.csv'):
    """Return historical Poisson evaluation metrics as plain Python types."""
    metrics = evaluate_poisson_file(csv_path)
    # convert any numpy types to native Python scalars for pickling
    cleaned = {}
    for k, v in metrics.items():
        try:
            cleaned[k] = float(v)
        except Exception:
            # if value is not directly convertible (e.g. nested dict), skip or keep as-is
            cleaned[k] = v
    return cleaned

from models.lstm_predictor import predict_match_lstm
from optimize_model import optimize_xgboost
from footer import add_betting_oracle_footer

DATA_DIR = 'data_files/'

st.set_page_config(page_title="Pitch Oracle - Premier League Historical Data", layout="wide", page_icon="⚽")

# Add cache status indicator
if 'app_load_count' not in st.session_state:
    st.session_state.app_load_count = 0
st.session_state.app_load_count += 1

st.image(path.join(DATA_DIR, 'logo.png'), width=250)
st.title("Premier League Predictor")

# Show cache status
# if st.session_state.app_load_count > 1:
#     st.success("⚡ Using cached data and models for instant loading!", icon="🚀")

# Show last update timestamp for fixtures data
fixtures_file = path.join(DATA_DIR, 'upcoming_fixtures.csv')
if path.exists(fixtures_file):
    last_updated = datetime.fromtimestamp(os.path.getmtime(fixtures_file))
    st.caption(f"Last updated: {last_updated.strftime('%Y-%m-%d %I:%M %p ET')}")

# Performance note
# st.info("🚀 **Performance Mode:** Models are pre-trained nightly. Advanced features like hyperparameter optimization and neural networks are available on-demand via buttons below.")

csv_path = path.join(DATA_DIR, 'combined_historical_data_with_calculations_new.csv')

def get_dataframe_height(df, row_height=35, header_height=38, padding=2, max_height=600):
    """
    Calculate the optimal height for a Streamlit dataframe based on number of rows.
    
    Args:
        df (pd.DataFrame): The dataframe to display
        row_height (int): Height per row in pixels. Default: 35
        header_height (int): Height of header row in pixels. Default: 38
        padding (int): Extra padding in pixels. Default: 2
        max_height (int): Maximum height cap in pixels. Default: 600 (None for no limit)
    
    Returns:
        int: Calculated height in pixels
    
    Example:
        height = get_dataframe_height(my_df)
        st.dataframe(my_df, height=height)
    """
    num_rows = len(df)
    calculated_height = (num_rows * row_height) + header_height + padding
    
    if max_height is not None:
        return min(calculated_height, max_height)
    return calculated_height


@st.cache_data(ttl=3600)  # Cache for 1 hour
def compute_live_standings(csv_path):
    """Compute current 2025-26 season standings from actual match results in the historical CSV."""
    _df = pd.read_csv(csv_path, sep='\t')
    _df['MatchDate'] = pd.to_datetime(_df['MatchDate'])
    current = _df[_df['MatchDate'] >= '2025-07-01'].copy()
    if current.empty:
        return pd.DataFrame()

    records = []
    for _, row in current.iterrows():
        home, away = row['HomeTeam'], row['AwayTeam']
        hg = int(row['FullTimeHomeGoals']) if pd.notna(row['FullTimeHomeGoals']) else 0
        ag = int(row['FullTimeAwayGoals']) if pd.notna(row['FullTimeAwayGoals']) else 0
        result = row['FullTimeResult']
        hw = hd = hl = aw = ad = al = 0
        if result == 'H':
            hw = 1; al = 1
        elif result == 'D':
            hd = 1; ad = 1
        elif result == 'A':
            hl = 1; aw = 1
        records.append({'Team': home, 'GF': hg, 'GA': ag, 'W': hw, 'D': hd, 'L': hl,
                        'IsHome': 1, 'MatchDate': row['MatchDate']})
        records.append({'Team': away, 'GF': ag, 'GA': hg, 'W': aw, 'D': ad, 'L': al,
                        'IsHome': 0, 'MatchDate': row['MatchDate']})

    mdf = pd.DataFrame(records)
    standings = mdf.groupby('Team').agg(
        Played=('GF', 'count'),
        Win=('W', 'sum'),
        Draw=('D', 'sum'),
        Lose=('L', 'sum'),
        GoalsFor=('GF', 'sum'),
        GoalsAgainst=('GA', 'sum'),
    ).reset_index()
    standings['GoalDifference'] = standings['GoalsFor'] - standings['GoalsAgainst']
    standings['Points'] = standings['Win'] * 3 + standings['Draw']
    standings = standings.sort_values(
        ['Points', 'GoalDifference', 'GoalsFor'], ascending=False
    ).reset_index(drop=True)
    standings['Rank'] = standings.index + 1

    def _form(team):
        rows = mdf[mdf['Team'] == team].sort_values('MatchDate').tail(5)
        return ''.join('W' if r['W'] else ('D' if r['D'] else 'L') for _, r in rows.iterrows())

    standings['Form'] = standings['Team'].apply(_form)
    return standings[['Rank', 'Team', 'Played', 'Win', 'Draw', 'Lose',
                       'GoalsFor', 'GoalsAgainst', 'GoalDifference', 'Points', 'Form']]


@st.cache_data(ttl=3600)  # Cache for 1 hour
def compute_current_team_stats(csv_path):
    """Compute per-team stats for the current 2025-26 season from match data."""
    _df = pd.read_csv(csv_path, sep='\t')
    _df['MatchDate'] = pd.to_datetime(_df['MatchDate'])
    current = _df[_df['MatchDate'] >= '2025-07-01'].copy()
    if current.empty:
        return pd.DataFrame()

    records = []
    for _, row in current.iterrows():
        home, away = row['HomeTeam'], row['AwayTeam']
        hg = int(row['FullTimeHomeGoals']) if pd.notna(row['FullTimeHomeGoals']) else 0
        ag = int(row['FullTimeAwayGoals']) if pd.notna(row['FullTimeAwayGoals']) else 0
        result = row['FullTimeResult']
        hw = hd = hl = aw = ad = al = 0
        if result == 'H':
            hw = 1; al = 1
        elif result == 'D':
            hd = 1; ad = 1
        elif result == 'A':
            hl = 1; aw = 1
        records.append({'Team': home, 'GF': hg, 'GA': ag, 'W': hw, 'D': hd, 'L': hl,
                        'IsHome': 1, 'CleanSheet': int(ag == 0), 'FTS': int(hg == 0)})
        records.append({'Team': away, 'GF': ag, 'GA': hg, 'W': aw, 'D': ad, 'L': al,
                        'IsHome': 0, 'CleanSheet': int(hg == 0), 'FTS': int(ag == 0)})

    mdf = pd.DataFrame(records)
    overall = mdf.groupby('Team').agg(
        TotalPlayed=('GF', 'count'),
        TotalWins=('W', 'sum'),
        TotalDraws=('D', 'sum'),
        TotalLosses=('L', 'sum'),
        GoalsForTotal=('GF', 'sum'),
        GoalsAgainstTotal=('GA', 'sum'),
        CleanSheetTotal=('CleanSheet', 'sum'),
        FailedToScoreTotal=('FTS', 'sum'),
    ).reset_index()
    home_stats = mdf[mdf['IsHome'] == 1].groupby('Team').agg(
        HomeWins=('W', 'sum'),
        HomeDraws=('D', 'sum'),
        HomeLosses=('L', 'sum'),
        HomeGF=('GF', 'sum'),
        HomeGA=('GA', 'sum'),
        CleanSheetHome=('CleanSheet', 'sum'),
        FailedToScoreHome=('FTS', 'sum'),
        HomePlayed=('GF', 'count'),
    ).reset_index()
    away_stats = mdf[mdf['IsHome'] == 0].groupby('Team').agg(
        AwayWins=('W', 'sum'),
        AwayDraws=('D', 'sum'),
        AwayLosses=('L', 'sum'),
        AwayGF=('GF', 'sum'),
        AwayGA=('GA', 'sum'),
        CleanSheetAway=('CleanSheet', 'sum'),
        FailedToScoreAway=('FTS', 'sum'),
        AwayPlayed=('GF', 'count'),
    ).reset_index()
    stats = overall.merge(home_stats, on='Team', how='left').merge(away_stats, on='Team', how='left')
    stats['GoalDifference'] = stats['GoalsForTotal'] - stats['GoalsAgainstTotal']
    stats['Points'] = stats['TotalWins'] * 3 + stats['TotalDraws']
    stats['GoalsForAvgHome'] = (stats['HomeGF'] / stats['HomePlayed'].clip(lower=1)).round(2)
    stats['GoalsAgainstAvgHome'] = (stats['HomeGA'] / stats['HomePlayed'].clip(lower=1)).round(2)
    stats['GoalsForAvgAway'] = (stats['AwayGF'] / stats['AwayPlayed'].clip(lower=1)).round(2)
    stats['GoalsAgainstAvgAway'] = (stats['AwayGA'] / stats['AwayPlayed'].clip(lower=1)).round(2)
    return stats


@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_precomputed_data():
    """
    Load precomputed data for fast app startup.
    Falls back to processing data if precomputed file doesn't exist.
    
    Cached to avoid repeated disk I/O.
    """
    precomputed_path = 'precomputed/preprocessed_data.pkl'

    if path.exists(precomputed_path):
        try:
            with open(precomputed_path, 'rb') as f:
                data = pickle.load(f)
            print("✅ Loaded precomputed data for fast startup")
            return data
        except Exception as e:
            print(f"⚠️  Could not load precomputed data: {e}")
            print("Falling back to real-time processing...")

    return None

@st.cache_resource  # Cache models in memory (they're not serializable)
def load_pretrained_models():
    """
    Load pre-trained models from the nightly pipeline.
    Falls back to training if models don't exist.
    
    Cached as a resource since ML models can't be serialized by Streamlit.
    This dramatically speeds up app loading on subsequent visits.
    """
    models_dir = 'models/'
    performance = {}

    try:
        # Load XGBoost baseline
        xgb_path = path.join(models_dir, 'xgb_baseline.pkl')
        if path.exists(xgb_path):
            with open(xgb_path, 'rb') as f:
                xgb_model = pickle.load(f)
            print("✅ Loaded pre-trained XGBoost baseline")
        else:
            raise FileNotFoundError("XGBoost model not found")

        # Load Ensemble model
        ensemble_path = path.join(models_dir, 'ensemble_model.pkl')
        if path.exists(ensemble_path):
            with open(ensemble_path, 'rb') as f:
                ensemble_model = pickle.load(f)
            print("✅ Loaded pre-trained Ensemble model")
        else:
            raise FileNotFoundError("Ensemble model not found")

        # Load Neural Network (if available)
        neural_model = None
        neural_scaler = None
        neural_path = path.join(models_dir, 'neural_model.pkl')
        scaler_path = path.join(models_dir, 'neural_scaler.pkl')
        if path.exists(neural_path) and path.exists(scaler_path):
            with open(neural_path, 'rb') as f:
                neural_model = pickle.load(f)
            with open(scaler_path, 'rb') as f:
                neural_scaler = pickle.load(f)
            print("✅ Loaded pre-trained Neural Network")
        else:
            print("⚠️  Neural Network not available, will train on-demand")

        # Load Optimized XGBoost (if available)
        optimized_xgb_model = None
        opt_xgb_path = path.join(models_dir, 'optimized_xgb.pkl')
        if path.exists(opt_xgb_path):
            with open(opt_xgb_path, 'rb') as f:
                optimized_xgb_model = pickle.load(f)
            print("✅ Loaded pre-trained Optimized XGBoost")
        else:
            print("⚠️  Optimized XGBoost not available, will optimize on-demand")

        # Load performance metrics
        perf_path = path.join(models_dir, 'model_performance.pkl')
        if path.exists(perf_path):
            with open(perf_path, 'rb') as f:
                performance = pickle.load(f)
            print("✅ Loaded model performance metrics")

        return {
            'xgb_model': xgb_model,
            'ensemble_model': ensemble_model,
            'neural_model': neural_model,
            'neural_scaler': neural_scaler,
            'optimized_xgb_model': optimized_xgb_model,
            'performance': performance
        }

    except Exception as e:
        print(f"⚠️  Could not load pre-trained models: {e}")
        print("Falling back to on-demand training...")
        return None

@st.cache_data(ttl=3600)  # Cache for 1 hour (expensive calculation)
def calculate_feature_importance(_model, X_test, y_test, feature_names, n_repeats=5):
    """
    Calculate permutation feature importance for the model with statistical significance testing.
    
    Args:
        _model: Trained model (underscore prefix prevents hashing)
        X_test: Test features
        y_test: Test targets
        feature_names: List of feature names
        n_repeats: Number of permutation repeats
    
    Returns:
        pd.DataFrame: Feature importance results with statistical significance
    
    Note: Cached because permutation importance is computationally expensive.
    """
    result = permutation_importance(_model, X_test, y_test, n_repeats=n_repeats, random_state=42, scoring='accuracy')
    
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': result.importances_mean,
        'Importance %': result.importances_mean * 100,
        'Std': result.importances_std
    })
    
    # Add statistical significance testing
    # Calculate z-scores and p-values for each feature
    # Null hypothesis: feature importance = 0 (no importance)
    importance_df['Z_Score'] = importance_df['Importance'] / (importance_df['Std'] + 1e-10)  # Add small epsilon to avoid division by zero
    importance_df['P_Value'] = 2 * (1 - stats.norm.cdf(abs(importance_df['Z_Score'])))  # Two-tailed test
    
    # Add significance level categories
    conditions = [
        (importance_df['P_Value'] < 0.001),
        (importance_df['P_Value'] < 0.01),
        (importance_df['P_Value'] < 0.05),
        (importance_df['P_Value'] < 0.10)
    ]
    choices = ['*** (p < 0.001)', '** (p < 0.01)', '* (p < 0.05)', '. (p < 0.10)']
    importance_df['Significance'] = np.select(conditions, choices, default='Not Significant')
    
    # Add confidence intervals (95%)
    confidence_level = 1.96  # 95% confidence
    importance_df['CI_Lower'] = importance_df['Importance'] - confidence_level * importance_df['Std']
    importance_df['CI_Upper'] = importance_df['Importance'] + confidence_level * importance_df['Std']
    
    importance_df = importance_df.sort_values('Importance', ascending=False)
    return importance_df

@st.cache_data(ttl=3600)  # Cache processed data for 1 hour
def load_and_process_data(csv_path):
    """
    Load CSV data and process it for ML model training.
    
    This function is cached to avoid re-processing the CSV file on every app reload.
    Processing includes:
    - Loading CSV
    - Target encoding
    - Feature engineering
    - Train/test split
    
    Args:
        csv_path: Path to the CSV file
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test, feature_names, df)
    """
    # Try to load precomputed data first (fastest path)
    print("Loading precomputed data...")
    precomputed_data = load_precomputed_data()

    if precomputed_data:
        # Use precomputed data
        X_train = precomputed_data['X_train']
        X_test = precomputed_data['X_test']
        y_train = precomputed_data['y_train']
        y_test = precomputed_data['y_test']
        feature_names = precomputed_data['feature_names']
        # For df, we'll load the  CSV (still faster than full processing)
        df = pd.read_csv(csv_path, sep='\t')
        
        print("✅ Using precomputed data for instant loading")
        return X_train, X_test, y_train, y_test, feature_names, df
    
    # Fallback: Process data in real-time (but cached after first run)
    print("Processing data in real-time (will be cached)...")
    df = pd.read_csv(csv_path, sep='\t')
    
    # Data preparation
    target_map = {'H': 0, 'D': 1, 'A': 2}
    df = df[df['FullTimeResult'].isin(target_map.keys())].copy()
    df['target'] = df['FullTimeResult'].map(target_map)

    # Drop columns not useful for modeling or that leak the result
    drop_cols = [
        'FullTimeResult', 'FullTimeHomeGoals', 'FullTimeAwayGoals',
        'HalfTimeResult', 'HalfTimeHomeGoals', 'HalfTimeAwayGoals',
        'HomeWin', 'AwayWin', 'Draw', 'WinningTeam',
        'HomePoints', 'AwayPoints', 'HomeTeamCumulativePoints', 'AwayTeamCumulativePoints',
        'MatchDate', 'KickoffTime', 'Season', 'Round', 'Venue', 'Referee',
        'HomeTeam', 'AwayTeam', 'Division'
    ]
    X = df.drop(columns=[col for col in drop_cols if col in df.columns] + ['target'], errors='ignore')
    y = df['target']

    # Get numeric features only
    X_numeric = X.select_dtypes(include=[np.number]).drop(columns=drop_cols, errors='ignore')

    # Handle categorical columns by encoding them
    cat_cols = X.select_dtypes(include=['object']).columns
    X_categorical = pd.DataFrame()
    for col in cat_cols:
        if col not in drop_cols:
            le = LabelEncoder()
            X_categorical[col] = le.fit_transform(X[col].astype(str))

    # Combine numeric and categorical features
    X = pd.concat([X_numeric, X_categorical], axis=1)

    # Fill any remaining NaN values
    X = X.fillna(X.mean())

    # Ensure X is a DataFrame with clean column names
    if isinstance(X, pd.DataFrame):
        # Reset column names to generic names to avoid XGBoost issues
        X.columns = [f'feature_{i}' for i in range(X.shape[1])]
        # Add dummy features to match expected 255
        current_features = X.shape[1]
        if current_features < 255:
            dummy_cols = {f'feature_{i}': 0 for i in range(current_features, 255)}
            dummy_df = pd.DataFrame(dummy_cols, index=X.index)
            X = pd.concat([X, dummy_df], axis=1)
        feature_names = X.columns.tolist()  # Store feature names for later use

    # Convert to numpy array to ensure compatibility with XGBoost
    X = X.values

    # --- Train/Test Split ---
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("✅ Data processed and cached")
    return X_train, X_test, y_train, y_test, feature_names, df


if not path.exists(csv_path):
    st.warning(f"No historical data file found at `{csv_path}`. Please add your CSV file to get started.")
    st.stop()

# Load and process data (cached after first run)
X_train, X_test, y_train, y_test, feature_names, df = load_and_process_data(csv_path)

# Reconstruct full datasets for parts of the app that need them
X = np.concatenate([X_train, X_test])
y = np.concatenate([y_train, y_test])

# --- Load Pre-trained Models or Train Fresh ---
print("Loading pre-trained models...")
pretrained_models = load_pretrained_models()

if pretrained_models:
    # Use pre-trained models
    xgb_model = pretrained_models['xgb_model']
    ensemble_model = pretrained_models['ensemble_model']
    neural_model = pretrained_models['neural_model']
    neural_scaler = pretrained_models['neural_scaler']
    optimized_xgb_model = pretrained_models['optimized_xgb_model']
    performance = pretrained_models['performance']

    # Get performance metrics
    xgb_acc = performance.get('xgb_baseline', {}).get('accuracy', 0)
    xgb_mae = performance.get('xgb_baseline', {}).get('mae', 0)
    ensemble_acc = performance.get('ensemble', {}).get('accuracy', 0)
    ensemble_mae = performance.get('ensemble', {}).get('mae', 0)
    neural_acc = performance.get('neural', {}).get('accuracy', 0) if neural_model else 0
    neural_mae = performance.get('neural', {}).get('mae', 0) if neural_model else 0
    opt_xgb_acc = performance.get('optimized_xgb', {}).get('accuracy', 0) if optimized_xgb_model else 0
    opt_xgb_mae = performance.get('optimized_xgb', {}).get('mae', 0) if optimized_xgb_model else 0

    # Generate predictions for feature importance (using test set)
    xgb_pred = xgb_model.predict(X_test)
    ensemble_pred = ensemble_model.predict(X_test)

    print("✅ Successfully loaded pre-trained models!")

else:
    # Fallback: Train models fresh (original behavior)
    print("Training models from scratch...")

    # Train XGBoost model (baseline)
    xgb_model = XGBClassifier(eval_metric='mlogloss', random_state=42)
    xgb_model.fit(X_train, y_train)
    xgb_pred = xgb_model.predict(X_test)
    xgb_mae = mean_absolute_error(y_test, xgb_pred)
    xgb_acc = accuracy_score(y_test, xgb_pred)

    # Train Ensemble model
    ensemble_model = create_simple_ensemble()
    ensemble_model.fit(X_train, y_train)
    ensemble_pred = ensemble_model.predict(X_test)
    ensemble_mae = mean_absolute_error(y_test, ensemble_pred)
    ensemble_acc = accuracy_score(y_test, ensemble_pred)

    # Initialize variables for optional models
    neural_model = None
    neural_scaler = None
    optimized_xgb_model = None
    neural_acc = neural_mae = opt_xgb_acc = opt_xgb_mae = 0

# Initialize session state for expensive model results
if 'optimized_available' not in st.session_state:
    st.session_state.optimized_available = optimized_xgb_model is not None
    st.session_state.opt_xgb_mae = opt_xgb_mae
    st.session_state.opt_xgb_acc = opt_xgb_acc

if 'neural_available' not in st.session_state:
    st.session_state.neural_available = neural_model is not None
    st.session_state.neural_mae = neural_mae
    st.session_state.neural_acc = neural_acc

# Use session state values
optimized_available = st.session_state.optimized_available
opt_xgb_mae = st.session_state.opt_xgb_mae
opt_xgb_acc = st.session_state.opt_xgb_acc
neural_available = st.session_state.neural_available
neural_mae = st.session_state.neural_mae
neural_acc = st.session_state.neural_acc

# Use ensemble as the main model for predictions
model = ensemble_model
y_pred = ensemble_pred
mae = ensemble_mae
acc = ensemble_acc

model_trained = True

# Create tabs for different sections
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Upcoming Matches", "Predictive Data", "Upcoming Predictions", "Statistics", "Raw Data", "Team Deep Dive"])

with tab1:
    # ── Live Standings (computed from current season match data) ───────────
    live_standings = compute_live_standings(csv_path)
    if not live_standings.empty:
        st.subheader("📊 Current Premier League Standings (2025-26)")
        st.dataframe(live_standings, height=get_dataframe_height(live_standings),
                     hide_index=True, use_container_width=True)
        st.divider()

    # ── Upcoming Fixtures ──────────────────────────────────────────────────
    upcoming_csv = path.join(DATA_DIR, 'upcoming_fixtures.csv')
    if not path.exists(upcoming_csv):
        st.warning(f"No upcoming fixtures file found at `{upcoming_csv}`. Please run `python fetch_upcoming_fixtures.py` to get upcoming matches.")
    else:
        upcoming_df_raw = pd.read_csv(upcoming_csv)

        # ── 7e. Match Preview Cards ────────────────────────────────────────
        # Enrich with current season standings
        if not live_standings.empty:
            standings_lookup = live_standings.set_index('Team')

            def _get_standing(team, col, default='—'):
                try:
                    return standings_lookup.loc[team, col]
                except (KeyError, TypeError):
                    return default

            st.subheader("🗓️ Upcoming Premier League Fixtures")
            st.write("*Times shown in Eastern Time (ET)*")
            for _, fixture in upcoming_df_raw.iterrows():
                home = fixture.get('HomeTeam', '?')
                away = fixture.get('AwayTeam', '?')
                date_str = fixture.get('Date', '')
                time_str = fixture.get('Time', '')

                home_rank = _get_standing(home, 'Rank')
                away_rank = _get_standing(away, 'Rank')
                home_pts  = _get_standing(home, 'Points')
                away_pts  = _get_standing(away, 'Points')
                home_form = _get_standing(home, 'Form')
                away_form = _get_standing(away, 'Form')
                home_gd   = _get_standing(home, 'GoalDifference')
                away_gd   = _get_standing(away, 'GoalDifference')

                with st.expander(f"**{home}** vs **{away}**  —  {date_str} {time_str}", expanded=False):
                    c1, c2, c3 = st.columns([2, 1, 2])
                    with c1:
                        st.markdown(f"**🏠 {home}**")
                        st.markdown(f"Rank: **#{home_rank}** | Pts: **{home_pts}** | GD: {home_gd}")
                        st.markdown(f"Form (last 5): `{home_form}`")
                    with c2:
                        st.markdown("### VS")
                    with c3:
                        st.markdown(f"**✈️ {away}**")
                        st.markdown(f"Rank: **#{away_rank}** | Pts: **{away_pts}** | GD: {away_gd}")
                        st.markdown(f"Form (last 5): `{away_form}`")
        else:
            st.subheader("Upcoming Premier League Matches")
            st.write(f"Found {len(upcoming_df_raw)} upcoming matches")
            st.write("*Times shown in Eastern Time (ET)*")
            st.dataframe(upcoming_df_raw, height=get_dataframe_height(upcoming_df_raw),
                         width=600, hide_index=True)

    add_betting_oracle_footer()

with tab2:
    st.subheader("Model Performance Comparison")
    st.write("**Current Model: Ensemble (XGBoost + Random Forest)**")

    # Model Accuracy Widget - Prominent display of current model performance
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            label="Model Accuracy",
            value=f"{acc:.1%}",
            delta=f"{acc - 0.5:.1%} vs random"
        )

    with col2:
        st.metric(
            label="Mean Absolute Error",
            value=f"{mae:.3f}",
            delta=f"{0.67 - mae:.3f} vs baseline"
        )

    with col3:
        predictions_count = len(y_test)
        st.metric(
            label="Test Predictions",
            value=predictions_count
        )

    st.divider()

    # Advanced Model Training Buttons
    st.subheader("🔬 Advanced Model Training")
    col1, col2 = st.columns(2)

    with col1:
        if st.button("🔧 Run Hyperparameter Optimization", 
                    help="Optimize XGBoost hyperparameters using RandomizedSearchCV (takes ~1-2 minutes)",
                    disabled=st.session_state.optimized_available):
            with st.spinner("Running hyperparameter optimization... This may take 1-2 minutes."):
                try:
                    optimized_xgb_model = optimize_xgboost(X_train, y_train)
                    opt_xgb_pred = optimized_xgb_model.predict(X_test)
                    st.session_state.opt_xgb_mae = mean_absolute_error(y_test, opt_xgb_pred)
                    st.session_state.opt_xgb_acc = accuracy_score(y_test, opt_xgb_pred)
                    st.session_state.optimized_available = True
                    st.success(f"✅ Optimization complete! Accuracy: {st.session_state.opt_xgb_acc:.3f}, MAE: {st.session_state.opt_xgb_mae:.3f}")
                    st.rerun()  # Refresh the page to update the comparison table
                except Exception as e:
                    st.error(f"❌ Optimization failed: {e}")

    with col2:
        if st.button("🧠 Train Neural Network", 
                    help="Train PyTorch neural network model (takes ~30-60 seconds)",
                    disabled=st.session_state.neural_available):
            with st.spinner("Training neural network... This may take 30-60 seconds."):
                try:
                    neural_model, neural_scaler = train_neural_model(X_train, y_train, epochs=50, batch_size=32)
                    neural_pred_proba = predict_neural(neural_model, neural_scaler, X_test)
                    neural_pred = np.argmax(neural_pred_proba, axis=1)
                    st.session_state.neural_mae = mean_absolute_error(y_test, neural_pred)
                    st.session_state.neural_acc = accuracy_score(y_test, neural_pred)
                    st.session_state.neural_available = True
                    st.success(f"✅ Neural network trained! Accuracy: {st.session_state.neural_acc:.3f}, MAE: {st.session_state.neural_mae:.3f}")
                    st.rerun()  # Refresh the page to update the comparison table
                except Exception as e:
                    st.error(f"❌ Neural network training failed: {e}")

    # Status indicators
    status_col1, status_col2 = st.columns(2)
    with status_col1:
        if st.session_state.optimized_available:
            if pretrained_models and pretrained_models['optimized_xgb_model']:
                st.success("✅ Hyperparameter optimization: Pre-loaded")
            else:
                st.success("✅ Hyperparameter optimization: Complete")
        else:
            st.info("⏳ Hyperparameter optimization: Not run")
    
    with status_col2:
        if st.session_state.neural_available:
            if pretrained_models and pretrained_models['neural_model']:
                st.success("✅ Neural network: Pre-loaded")
            else:
                st.success("✅ Neural network: Trained")
        else:
            st.info("⏳ Neural network: Not trained")

    st.divider()

    # Create comparison dataframe
    model_names = ['XGBoost (Baseline)', 'Ensemble (Current)']
    mae_values = [xgb_mae, ensemble_mae]
    acc_values = [xgb_acc, ensemble_acc]

    if optimized_available:
        model_names.insert(1, 'XGBoost (Optimized)')
        mae_values.insert(1, opt_xgb_mae)
        acc_values.insert(1, opt_xgb_acc)

    if neural_available:
        model_names.append('Neural Network (PyTorch)')
        mae_values.append(neural_mae)
        acc_values.append(neural_acc)

    performance_data = {
        'Model': model_names,
        'MAE': mae_values,
        'Accuracy': acc_values,
        'MAE Change': [0] + [mae - xgb_mae for mae in mae_values[1:]],
        'Accuracy Change': [0] + [acc - xgb_acc for acc in acc_values[1:]]
    }

    perf_df = pd.DataFrame(performance_data)

    # Display metrics with color coding
    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Mean Absolute Error (MAE)",
            f"{mae:.3f}",
            f"{(ensemble_mae - xgb_mae):+.3f} vs XGBoost",
            delta_color="inverse"  # Lower MAE is better (inverse)
        )

    with col2:
        st.metric(
            "Accuracy",
            f"{acc:.3f}",
            f"{(ensemble_acc - xgb_acc):+.3f} vs XGBoost",
            delta_color="normal"  # Higher accuracy is better
        )

    # Detailed comparison table
    st.subheader("Model Comparison Details")
    styled_df = perf_df.style.format({
        'MAE': '{:.3f}',
        'Accuracy': '{:.3f}',
        'MAE Change': '{:+.3f}',
        'Accuracy Change': '{:+.3f}'
    })

    # Highlight the best performing model
    if neural_available and neural_acc > ensemble_acc:
        styled_df = styled_df.apply(lambda x: ['background-color: #d4edda' if x.name == 'Neural Network (PyTorch)' else '' for i in x], axis=1)
    else:
        styled_df = styled_df.apply(lambda x: ['background-color: #d4edda' if x.name == 'Ensemble (Current)' else '' for i in x], axis=1)

    st.dataframe(styled_df, hide_index=True, height=get_dataframe_height(styled_df.data), width=600)

    # Performance interpretation
    mae_improvement = xgb_mae - ensemble_mae
    acc_improvement = ensemble_acc - xgb_acc

    st.success(f"✅ **Ensemble Model Improvement:** MAE reduced by {mae_improvement:.3f}, Accuracy improved by {acc_improvement:.3f}")

    if optimized_available:
        opt_mae_improvement = xgb_mae - opt_xgb_mae
        opt_acc_improvement = opt_xgb_acc - xgb_acc
        st.info(f"🔧 **XGBoost Optimization:** MAE reduced by {opt_mae_improvement:.3f}, Accuracy improved by {opt_acc_improvement:.3f}")

    if neural_available:
        neural_mae_improvement = xgb_mae - neural_mae
        neural_acc_improvement = neural_acc - xgb_acc

        if neural_acc > ensemble_acc:
            st.success(f"🎉 **Neural Network Best Performer:** MAE reduced by {neural_mae_improvement:.3f}, Accuracy improved by {neural_acc_improvement:.3f}")
        elif neural_acc >= ensemble_acc * 0.98:  # Within 2% of ensemble
            st.info(f"🧠 **Neural Network Competitive:** MAE reduced by {neural_mae_improvement:.3f}, Accuracy improved by {neural_acc_improvement:.3f}")
        else:
            st.info(f"🧠 **Neural Network Trained:** MAE reduced by {neural_mae_improvement:.3f}, Accuracy improved by {neural_acc_improvement:.3f} (Ensemble still better)")
    else:
        st.warning("⚠️ Neural Network training failed - check PyTorch installation")

    # --- Monte Carlo Permutation Importance ---
    st.subheader("Monte Carlo Feature Importance (Permutation)")
    
    if st.button("Calculate Feature Importance", key="calc_importance"):
        with st.spinner("Calculating feature importance... This may take a moment."):
            importance_df = calculate_feature_importance(model, X_test, y_test, feature_names)
            st.session_state['importance_df'] = importance_df
    
    # Display results if importance_df is available (either just calculated or from session state)
    if 'importance_df' in st.session_state:
        importance_df = st.session_state['importance_df']
        
        # Display statistical significance legend
        st.info("""
        **📊 Statistical Significance Legend:**
        - *** (p < 0.001): Extremely significant
        - ** (p < 0.01): Very significant  
        - * (p < 0.05): Significant
        - . (p < 0.10): Marginally significant
        - Not Significant: No statistical significance
        """)
        
        # Display top 20 features with enhanced formatting
        st.subheader("Top 20 Features by Importance")
        top_20 = importance_df.head(20).copy()
        
        # Format for display
        display_df = top_20[['Feature', 'Importance %', 'Significance', 'Z_Score', 'P_Value']].copy()
        display_df['Importance %'] = display_df['Importance %'].round(3)
        display_df['Z_Score'] = display_df['Z_Score'].round(2)
        display_df['P_Value'] = display_df['P_Value'].apply(lambda x: f"{x:.2e}" if x < 0.001 else f"{x:.4f}")
        
        # Color code based on significance
        def color_significance(val):
            if val == '*** (p < 0.001)':
                return 'background-color: #d4edda; color: #155724'  # Green
            elif val == '** (p < 0.01)':
                return 'background-color: #d1ecf1; color: #0c5460'  # Blue
            elif val == '* (p < 0.05)':
                return 'background-color: #fff3cd; color: #856404'  # Yellow
            elif val == '. (p < 0.10)':
                return 'background-color: #f8d7da; color: #721c24'  # Red
            else:
                return ''
        
        styled_df = display_df.style.apply(lambda x: [color_significance(val) if col == 'Significance' else '' for col, val in x.items()], axis=1)
        st.dataframe(styled_df, hide_index=True, height=get_dataframe_height(display_df))
        
        # Summary statistics
        st.subheader("Statistical Summary")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            significant_count = len(importance_df[importance_df['P_Value'] < 0.05])
            st.metric("Statistically Significant Features", f"{significant_count}/{len(importance_df)}", 
                     f"{(significant_count/len(importance_df)*100):.1f}%")
        
        with col2:
            highly_significant = len(importance_df[importance_df['P_Value'] < 0.01])
            st.metric("Highly Significant (p < 0.01)", highly_significant)
        
        with col3:
            top_importance = importance_df['Importance'].max()
            st.metric("Max Importance Score", f"{top_importance:.4f}")
        
        # Feature categories analysis
        st.subheader("Feature Category Analysis")
        
        # Define feature categories
        categories = {
            'Team Performance': ['Points', 'Goals', 'Shots', 'Differential', 'Momentum'],
            'Betting Odds': ['Bet365', 'Pinnacle', 'William'],
            'Manager': ['Manager', 'MGR'],
            'Referee': ['Referee', 'Ref'],
            'Weather': ['Weather', 'Temperature'],
            'Poisson': ['Poisson'],
            'Shooting': ['ShootingEff', 'xG']
        }
        
        category_stats = []
        for cat_name, keywords in categories.items():
            mask = importance_df['Feature'].str.contains('|'.join(keywords), case=False)
            if mask.any():
                cat_importance = importance_df[mask]['Importance'].sum()
                cat_count = mask.sum()
                significant_in_cat = (importance_df[mask]['P_Value'] < 0.05).sum()
                category_stats.append({
                    'Category': cat_name,
                    'Total Importance': cat_importance,
                    'Feature Count': cat_count,
                    'Significant Features': significant_in_cat,
                    'Significance Rate': significant_in_cat / cat_count if cat_count > 0 else 0
                })
        
        if category_stats:
            cat_df = pd.DataFrame(category_stats)
            cat_df = cat_df.sort_values('Total Importance', ascending=False)
            cat_df['Significance Rate'] = (cat_df['Significance Rate'] * 100).round(1).astype(str) + '%'
            st.dataframe(cat_df, hide_index=True)
        
        # PDF Export Section
        st.markdown("---")
        st.subheader("📄 Export Statistical Report")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📊 Generate Full PDF Report", key="full_pdf", help="Generate comprehensive PDF report with charts and detailed analysis"):
                with st.spinner("Generating comprehensive PDF report..."):
                    # Prepare data for PDF generation
                    model_metrics = {
                        'mae': mae,
                        'accuracy': acc,
                        'total_features': len(importance_df),
                        'significant_features': len(importance_df[importance_df['P_Value'] < 0.05]),
                        'significance_rate': len(importance_df[importance_df['P_Value'] < 0.05]) / len(importance_df) * 100,
                        'top_category': cat_df.iloc[0]['Category'] if len(cat_df) > 0 else 'N/A',
                        'top_reliability': float(cat_df.iloc[0]['Significance Rate'].rstrip('%')) if len(cat_df) > 0 else 0
                    }
                    
                    # Convert category stats for PDF
                    pdf_category_stats = []
                    for _, row in cat_df.iterrows():
                        pdf_category_stats.append({
                            'Category': row['Category'],
                            'Features': row['Feature Count'],
                            'Significant': row['Significant Features'],
                            'Reliability': float(row['Significance Rate'].rstrip('%'))
                        })
                    
                    # Generate PDF
                    pdf_path = generate_statistical_report(importance_df, model_metrics, pdf_category_stats)
                    
                    # Read PDF and create download button
                    with open(pdf_path, "rb") as f:
                        pdf_bytes = f.read()
                    
                    st.success("✅ Comprehensive PDF report generated!")
                    st.download_button(
                        label="📥 Download Full Report",
                        data=pdf_bytes,
                        file_name=pdf_path,
                        mime="application/pdf",
                        key="download_full_pdf"
                    )
                    
                    # Clean up file
                    os.remove(pdf_path)
        
        with col2:
            if st.button("📋 Generate Quick Summary PDF", key="quick_pdf", help="Generate concise PDF summary for quick review"):
                with st.spinner("Generating quick PDF summary..."):
                    # Prepare data for quick PDF
                    model_metrics = {
                        'mae': mae,
                        'accuracy': acc,
                        'total_features': len(importance_df),
                        'significant_features': len(importance_df[importance_df['P_Value'] < 0.05]),
                        'significance_rate': len(importance_df[importance_df['P_Value'] < 0.05]) / len(importance_df) * 100
                    }
                    
                    pdf_category_stats = []  # Not needed for quick report
                    
                    # Generate quick PDF
                    pdf_bytes = generate_quick_report(importance_df, model_metrics, pdf_category_stats)
                    
                    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
                    filename = f"premier_league_quick_report_{timestamp}.pdf"
                    
                    st.success("✅ Quick PDF summary generated!")
                    st.download_button(
                        label="📥 Download Quick Summary",
                        data=pdf_bytes,
                        file_name=filename,
                        mime="application/pdf",
                        key="download_quick_pdf"
                    )
    
    else:
        st.info("Click 'Calculate Feature Importance' to analyze feature significance and enable PDF export.")

    # Prediction Performance Tracker
    st.markdown("---")
    if st.checkbox("Show Prediction Performance Tracker"):
        st.subheader("📈 Model Prediction Accuracy Over Time")
        
        # Import the tracking functions
        from track_predictions import validate_predictions
        
        perf = validate_predictions()
        
        if perf is not None and len(perf) > 0:
            completed = perf[perf['Correct'].notna()]
            if len(completed) > 0:
                accuracy = completed['Correct'].mean()
                st.metric("Prediction Accuracy", f"{accuracy:.1%}", 
                         f"{len(completed)} predictions validated")
                st.dataframe(completed[['PredictionDate', 'MatchDate', 'HomeTeam', 'AwayTeam', 
                                       'PredHomeWin', 'PredDraw', 'PredAwayWin', 'ActualResult', 'Correct']], 
                           width='stretch', hide_index=True, height=get_dataframe_height(completed))
            else:
                st.info("No predictions have been validated yet. Predictions will be validated after match results are available.")
        else:
            st.info("No prediction history found. Start making predictions to track performance over time.")
    
    add_betting_oracle_footer()

with tab3:
    # Load upcoming fixtures
    upcoming_csv = path.join(DATA_DIR, 'upcoming_fixtures.csv')
    if not path.exists(upcoming_csv):
        st.warning(f"No upcoming fixtures file found at `{upcoming_csv}`. Please run fetch_upcoming_fixtures.py to get upcoming matches.")
        st.stop()

    upcoming_df = pd.read_csv(upcoming_csv)
    
    # Normalize team names to match historical data
    upcoming_df['HomeTeam'] = upcoming_df['HomeTeam'].apply(normalize_team_name)
    upcoming_df['AwayTeam'] = upcoming_df['AwayTeam'].apply(normalize_team_name)

    # Load team stats
    all_teams = pd.read_csv(path.join(DATA_DIR, 'all_teams.csv'), sep='\t')

    # Merge home team stats (using their home performance stats)
    home_cols = ['Team', 'TeamId', 'HomeGoalsAve', 'HomeGoalsTotal', 'HomeGoalsHalfAve', 'HomeGoalsHalfTotal', 
                 'HomeShotsAve', 'HomeShotsTotal', 'HomeShotsOnTargetAve', 'HomeFirstHalfDifferentialAve', 
                 'HomeGameDifferentialAve', 'HomeFirstToSecondHalfGoalRatioAve']
    upcoming_df = pd.merge(
        upcoming_df,
        all_teams[home_cols],
        left_on='HomeTeam', right_on='Team', how='left'
    )
    upcoming_df.drop(columns=['Team'], inplace=True)
    
    # Merge away team stats (using their away performance stats)
    away_cols = ['Team', 'AwayGoalsAve', 'AwayGoalsTotal', 'AwayGoalsHalfAve', 'AwayGoalsHalfTotal', 
                 'AwayShotsAve', 'AwayShotsTotal', 'AwayShotsOnTargetAve', 'AwayFirstHalfDifferentialAve', 
                 'AwayGameDifferentialAve', 'AwayFirstToSecondHalfGoalRatioAve']
    upcoming_df = pd.merge(
        upcoming_df,
        all_teams[away_cols],
        left_on='AwayTeam', right_on='Team', how='left'
    )
    upcoming_df.drop(columns=['Team'], inplace=True)
    
    # Load and merge referee data if available
    referee_csv = path.join(DATA_DIR, 'scraped_referees_test.csv')
    if path.exists(referee_csv):
        referee_df = pd.read_csv(referee_csv)
        referee_df['HomeTeam'] = referee_df['HomeTeam'].apply(normalize_team_name)
        referee_df['AwayTeam'] = referee_df['AwayTeam'].apply(normalize_team_name)
        
        # Merge referee assignments with upcoming fixtures
        upcoming_df = pd.merge(
            upcoming_df,
            referee_df[['Date', 'HomeTeam', 'AwayTeam', 'Referee']],
            on=['Date', 'HomeTeam', 'AwayTeam'],
            how='left'
        )
        
        # Load historical referee statistics
        historical_referee_stats = df[['Referee', 'RefYellowCardsPerGame', 'RefRedCardsPerGame', 'RefFoulsPerGame', 
                                       'RefHomeAdvantageYellow', 'RefHomeWinRate', 'RefAwayWinRate', 'RefDrawRate']].drop_duplicates()
        
        # Merge referee statistics
        upcoming_df = pd.merge(
            upcoming_df,
            historical_referee_stats,
            on='Referee',
            how='left'
        )
        
        # Fill missing referee stats with league averages
        ref_cols = ['RefYellowCardsPerGame', 'RefRedCardsPerGame', 'RefFoulsPerGame', 
                   'RefHomeAdvantageYellow', 'RefHomeWinRate', 'RefAwayWinRate', 'RefDrawRate']
        for col in ref_cols:
            if col in upcoming_df.columns:
                upcoming_df[col] = upcoming_df[col].fillna(df[col].mean())
    
    # Ensure upcoming_df has all the columns that the training data has
    # Get the expected columns from training data processing
    expected_columns = [
        'HomeShots', 'AwayShots', 'HomeShotsOnTarget', 'AwayShotsOnTarget', 'HomeFouls', 'AwayFouls', 
        'HomeCorners', 'AwayCorners', 'HomeYellowCards', 'AwayYellowCards', 'HomeRedCards', 'AwayRedCards', 
        'Bet365_HomeWinOdds', 'Bet365_DrawOdds', 'Bet365_AwayWinOdds', 'BetWin_HomeWinOdds', 'BetWin_DrawOdds', 
        'BetWin_AwayWinOdds', 'Interwetten_HomeWinOdds', 'Interwetten_DrawOdds', 'Interwetten_AwayWinOdds', 
        'Pinnacle_HomeWinOdds', 'Pinnacle_DrawOdds', 'Pinnacle_AwayWinOdds', 'WilliamHill_HomeWinOdds', 
        'WilliamHill_DrawOdds', 'WilliamHill_AwayWinOdds', 'VCBet_HomeWinOdds', 'VCBet_DrawOdds', 'VCBet_AwayWinOdds', 
        'Max_HomeWinOdds', 'Max_DrawOdds', 'Max_AwayWinOdds', 'Avg_HomeWinOdds', 'Avg_DrawOdds', 'Avg_AwayWinOdds', 
        'Bet365_Over2_5GoalsOdds', 'Bet365_Under2_5GoalsOdds', 'P>2.5', 'P<2.5', 'Max>2.5', 'Max<2.5', 'Avg>2.5', 
        'Avg<2.5', 'AHh', 'Bet365_AH_HomeOdds', 'Bet365_AH_AwayOdds', 'PAHH', 'PAHA', 'MaxAHH', 'MaxAHA', 'AvgAHH', 
        'AvgAHA', 'Bet365_ClosingHomeOdds', 'Bet365_ClosingDrawOdds', 'Bet365_ClosingAwayOdds', 'BWCH', 'BWCD', 
        'BWCA', 'IWCH', 'IWCD', 'IWCA', 'Pinnacle_ClosingHomeOdds', 'Pinnacle_ClosingDrawOdds', 
        'Pinnacle_ClosingAwayOdds', 'WHCH', 'WHCD', 'WHCA', 'VCCH', 'VCCD', 'VCCA', 'MaxCH', 'MaxCD', 'MaxCA', 
        'AvgCH', 'AvgCD', 'AvgCA', 'B365C>2.5', 'B365C<2.5', 'PC>2.5', 'PC<2.5', 'MaxC>2.5', 'MaxC<2.5', 
        'AvgC>2.5', 'AvgC<2.5', 'AHCh', 'B365CAHH', 'B365CAHA', 'PCAHH', 'PCAHA', 'MaxCAHH', 'MaxCAHA', 'AvgCAHH', 
        'AvgCAHA', 'BFH', 'BFD', 'BFA', '1XBH', '1XBD', '1XBA', 'BFEH', 'BFED', 'BFEA', 'BFE>2.5', 'BFE<2.5', 
        'BFEAHH', 'BFEAHA', 'BFCH', 'BFCD', 'BFCA', '1XBCH', '1XBCD', '1XBCA', 'BFECH', 'BFECD', 'BFECA', 
        'BFEC>2.5', 'BFEC<2.5', 'BFECAHH', 'BFECAHA', 'BFDH', 'BFDD', 'BFDA', 'BMGMH', 'BMGMD', 'BMGMA', 'BVH', 
        'BVD', 'BVA', 'CLH', 'CLD', 'CLA', 'Ladbrokes_HomeWinOdds', 'Ladbrokes_DrawOdds', 'Ladbrokes_AwayWinOdds', 
        'BFDCH', 'BFDCD', 'BFDCA', 'BMGMCH', 'BMGMCD', 'BMGMCA', 'BVCH', 'BVCD', 'BVCA', 'CLCH', 'CLCD', 'CLCA', 
        'LBCH', 'LBCD', 'LBCA', 'HalfTimeHomeWin', 'HalfTimeAwayWin', 'HalfTimeDraw', 'HomeTeamPointsLast5', 
        'AwayTeamPointsLast5', 'HomeH2HWinLast5', 'AwayH2HWinLast5', 'H2HDrawLast5', 'HomeRestDays', 'AwayRestDays', 
        'HomeGoalsAve', 'HomeGoalsTotal', 'HomeGoalsHalfAve', 'HomeGoalsHalfTotal', 'HomeShotsAve', 'HomeShotsTotal', 
        'HomeShotsOnTargetAve', 'HomeFirstHalfDifferentialAve', 'HomeGameDifferentialAve', 
        'HomeFirstToSecondHalfGoalRatioAve', 'AwayGoalsAve', 'AwayGoalsTotal', 'AwayGoalsHalfAve', 'AwayGoalsHalfTotal', 
        'AwayShotsAve', 'AwayShotsTotal', 'AwayShotsOnTargetAve', 'AwayFirstHalfDifferentialAve', 
        'AwayGameDifferentialAve', 'AwayFirstToSecondHalfGoalRatioAve', 'ImpliedProb_HomeWin', 'ImpliedProb_Draw', 
        'ImpliedProb_AwayWin', 'ImpliedProb_HomeWin_Norm', 'ImpliedProb_Draw_Norm', 'ImpliedProb_AwayWin_Norm', 
        'Bet365_MarketMargin', 'OddsMovement_Home', 'OddsMovement_Away', 'OddsMovement_Draw', 'Bet365_Value_Home', 
        'Bet365_Value_Away', 'Bet365_Value_Draw', 'Bet365_HomeVsDraw_Ratio', 'Bet365_AwayVsDraw_Ratio', 
        'Bet365_HomeVsAway_Ratio', 'Bet365_OverUnder_Margin', 'Bet365_ExpectedTotalGoals', 'Bet365_AH_Margin', 
        'HomeInjuryCount', 'AwayInjuryCount', 'InjuryAdvantage', 'Temperature', 'Humidity', 'WindSpeed', 
        'Precipitation', 'HomexG_Avg_L5', 'HomeShootingEff_Avg_L5', 'HomeMomentum_L3', 'HomeGoalDiff_Avg_L5', 
        'AwayxG_Avg_L5', 'AwayShootingEff_Avg_L5', 'AwayMomentum_L3', 'AwayGoalDiff_Avg_L5', 'TeamId', 
        'WeatherCondition', 'WeatherDescription', 'WeatherImpact'
    ]
    
    # Add missing columns with default values
    missing_cols = [col for col in expected_columns if col not in upcoming_df.columns]
    if missing_cols:
        default_values = {}
        for col in missing_cols:
            if col in df.columns and df[col].dtype in ['int64', 'float64']:
                # Use mean from historical data for numeric columns
                default_values[col] = df[col].mean()
            elif 'Odds' in col or 'Prob' in col or 'Margin' in col or 'Value' in col or 'Ratio' in col:
                # For betting-related columns, use neutral values
                if 'HomeWin' in col or 'AwayWin' in col:
                    default_values[col] = 2.0
                elif 'Draw' in col:
                    default_values[col] = 3.5
                else:
                    default_values[col] = 2.0
            elif 'Weather' in col or 'Description' in col:
                # For weather, use a default string
                default_values[col] = 'Clear'
            else:
                # For other missing columns, use 0
                default_values[col] = 0
        
        # Add all missing columns at once to avoid fragmentation
        new_df = pd.DataFrame(default_values, index=upcoming_df.index)
        upcoming_df = pd.concat([upcoming_df, new_df], axis=1)
    
    # Prepare features for prediction model
    X_upcoming = upcoming_df.drop(columns=['Date', 'Time', 'HomeTeam', 'AwayTeam'], errors='ignore')
    
    # Apply the same data processing as training
    # Get numeric features only
    X_upcoming_numeric = X_upcoming.select_dtypes(include=[np.number])
    
    # Handle categorical columns by encoding them
    cat_cols = X_upcoming.select_dtypes(include=['object']).columns
    X_upcoming_categorical = pd.DataFrame()
    for col in cat_cols:
        le = LabelEncoder()
        X_upcoming_categorical[col] = le.fit_transform(X_upcoming[col].astype(str))
    
    # Combine numeric and categorical features
    X_upcoming = pd.concat([X_upcoming_numeric, X_upcoming_categorical], axis=1)
    
    # Fill any remaining NaN values
    X_upcoming = X_upcoming.fillna(X_upcoming.mean())
    
    # Ensure X is a DataFrame with clean column names
    if isinstance(X_upcoming, pd.DataFrame):
        # Reset column names to generic names to match training
        X_upcoming.columns = [f'feature_{i}' for i in range(X_upcoming.shape[1])]
        # Add dummy features to match expected 255
        current_features = X_upcoming.shape[1]
        if current_features < 255:
            dummy_cols = {f'feature_{i}': 0 for i in range(current_features, 255)}
            dummy_df = pd.DataFrame(dummy_cols, index=X_upcoming.index)
            X_upcoming = pd.concat([X_upcoming, dummy_df], axis=1)
    
    # Convert to numpy array to ensure compatibility with XGBoost
    X_upcoming = X_upcoming.values
    
    # Model Selection
    st.subheader("🤖 Choose Prediction Model")
    model_options = {
        "Simple Ensemble": "Fast, reliable predictions using multiple ML algorithms",
        "Poisson Regression": "Goal-based predictions using statistical modeling of expected goals",
        "LSTM Time Series": "Deep learning model capturing team momentum and temporal patterns"
    }
    
    selected_model = st.radio(
        "Select prediction model:",
        options=list(model_options.keys()),
        index=0,  # Default to Simple Ensemble
        help="Choose which model to use for predicting match outcomes"
    )
    
    st.info(f"📊 **{selected_model}**: {model_options[selected_model]}")
    
    # Filter historical data to same features (both datasets now have same generic column names)
    # Since both X and X_upcoming are processed identically, they have the same features
    X_simple = X  # Use all features since they're aligned

    if selected_model == "Simple Ensemble":
        # Train simple ensemble model
        simple_model = create_simple_ensemble()
        simple_model.fit(X_simple, y)
        
        # Predict probabilities using simple ensemble model
        proba = simple_model.predict_proba(X_upcoming)
        
        # Add predictions to df
        prediction_cols = {
            'HomeWin_Prob': proba[:, 0],
            'Draw_Prob': proba[:, 1],
            'AwayWin_Prob': proba[:, 2]
        }
        pred_df = pd.DataFrame(prediction_cols, index=upcoming_df.index)
        upcoming_df = pd.concat([upcoming_df, pred_df], axis=1)
        
    elif selected_model == "Poisson Regression":
        # Use Poisson regression for predictions
        st.info("⚽ Using Poisson regression based on expected goals...")
        
        # Initialize prediction arrays
        home_win_probs = []
        draw_probs = []
        away_win_probs = []
        
        # Get team stats for Poisson predictions
        team_stats_df = all_teams.copy()
        
        for idx, match in upcoming_df.iterrows():
            home_team = match['HomeTeam']
            away_team = match['AwayTeam']
            
            # Get Poisson predictions
            poisson_result = predict_match_poisson(home_team, away_team, team_stats_df)
            
            home_win_probs.append(poisson_result['HomeWinProb'])
            draw_probs.append(poisson_result['DrawProb'])
            away_win_probs.append(poisson_result['AwayWinProb'])
        
        # Add predictions to df
        prediction_cols = {
            'HomeWin_Prob': home_win_probs,
            'Draw_Prob': draw_probs,
            'AwayWin_Prob': away_win_probs
        }
        pred_df = pd.DataFrame(prediction_cols, index=upcoming_df.index)
        upcoming_df = pd.concat([upcoming_df, pred_df], axis=1)

        # show evaluation metrics for the Poisson model using historical data
        with st.expander('📈 Poisson Model Historical Metrics'):
            metrics = get_poisson_metrics()
            st.write('League average goals used for fitting:', round(metrics['league_avg'], 3))
            cols = st.columns(2)
            cols[0].metric('Home goals MAE', f"{metrics['home_mae']:.3f}")
            cols[1].metric('Away goals MAE', f"{metrics['away_mae']:.3f}")
            cols = st.columns(2)
            cols[0].metric('Home goals RMSE', f"{metrics['home_rmse']:.3f}")
            cols[1].metric('Away goals RMSE', f"{metrics['away_rmse']:.3f}")
            st.write('Outcome accuracy:', f"{metrics['outcome_acc']:.3f}")
            st.write('Brier scores (home/draw/away):',
                     f"{metrics['brier_home']:.3f}, {metrics['brier_draw']:.3f}, {metrics['brier_away']:.3f}")

            # historical chart
            hist_path = path.join(DATA_DIR, 'poisson_metrics_history.csv')
            if path.exists(hist_path):
                hist_df = pd.read_csv(hist_path, parse_dates=['date'])
                hist_df = hist_df.set_index('date')
                # show only the date portion on the x-axis
                hist_df.index = hist_df.index.strftime('%Y-%m-%d')
                st.write('### Historical Poisson Metrics')
                st.line_chart(hist_df[['home_mae', 'away_mae']])
        
    elif selected_model == "LSTM Time Series":
        # Use LSTM time series model for predictions
        st.info("🧠 Using LSTM deep learning model for temporal momentum analysis...")
        
        # Initialize prediction arrays
        home_win_probs = []
        draw_probs = []
        away_win_probs = []
        
        # Load historical data for LSTM predictions
        historical_df = df.copy()  # Use the main historical dataframe
        
        for idx, match in upcoming_df.iterrows():
            home_team = match['HomeTeam']
            away_team = match['AwayTeam']
            
            # Get LSTM predictions (currently uses home team momentum)
            lstm_result = predict_match_lstm(home_team, away_team, historical_df)
            
            home_win_probs.append(lstm_result['HomeWinProb'])
            draw_probs.append(lstm_result['DrawProb'])
            away_win_probs.append(lstm_result['AwayWinProb'])
        
        # Add predictions to df
        prediction_cols = {
            'HomeWin_Prob': home_win_probs,
            'Draw_Prob': draw_probs,
            'AwayWin_Prob': away_win_probs
        }
        pred_df = pd.DataFrame(prediction_cols, index=upcoming_df.index)
        upcoming_df = pd.concat([upcoming_df, pred_df], axis=1)
    def calculate_prediction_risk(home_prob, draw_prob, away_prob):
        """
        Calculate prediction risk score (0-100) based on probability distribution.
        Lower scores = higher confidence, higher scores = higher risk.
        Adapted from HenryOnilude's variance-based risk scoring.
        """
        # Get the maximum probability (most likely outcome)
        max_prob = max(home_prob, draw_prob, away_prob)

        # Calculate entropy as a measure of uncertainty
        # Higher entropy = more evenly distributed probabilities = higher risk
        probs = np.array([home_prob, draw_prob, away_prob])
        # Add small epsilon to avoid log(0)
        probs = np.clip(probs, 1e-10, 1.0)
        entropy = -np.sum(probs * np.log(probs))

        # Normalize entropy to 0-1 scale (max entropy for 3 outcomes is log(3) ≈ 1.099)
        normalized_entropy = entropy / np.log(3)

        # Calculate confidence score (inverse of entropy)
        confidence_score = 1 - normalized_entropy

        # Calculate variance from uniform distribution as additional risk factor
        uniform_prob = 1/3
        variance = np.sum((probs - uniform_prob) ** 2) / 3

        # Combine factors: lower confidence + higher variance = higher risk
        risk_score = (1 - confidence_score) * 50 + variance * 50

        # Scale to 0-100 range (no additional multiplication needed)
        risk_score = min(100, max(0, risk_score))

        return risk_score, confidence_score

    # Apply risk scoring to all predictions
    risk_scores = []
    confidence_scores = []
    for idx, row in upcoming_df.iterrows():
        risk, confidence = calculate_prediction_risk(
            row['HomeWin_Prob'],
            row['Draw_Prob'],
            row['AwayWin_Prob']
        )
        risk_scores.append(risk)
        confidence_scores.append(confidence)

    # Add risk categories adjusted for match prediction with limited data
    # Based on actual distribution: most scores are 40-50, need broader low risk band
    def get_risk_category(risk_score):
        if risk_score > 47:
            return "Critical Risk", "🚨"
        elif risk_score > 40:
            return "High Risk", "🔴"
        elif risk_score > 30:
            return "Moderate Risk", "🟡"
        else:
            return "Low Risk", "🟢"

    risk_categories = []
    risk_emojis = []
    for risk in risk_scores:
        category, emoji = get_risk_category(risk)
        risk_categories.append(category)
        risk_emojis.append(emoji)

    # Add betting recommendations based on risk
    def get_betting_recommendation(home_prob, draw_prob, away_prob, risk_score):
        max_prob = max(home_prob, draw_prob, away_prob)
        confidence_threshold = 0.6  # 60% confidence minimum

        if max_prob >= confidence_threshold and risk_score <= 30:
            # High confidence, low risk - recommend betting
            if home_prob == max_prob:
                return "Bet Home Win", "💰"
            elif draw_prob == max_prob:
                return "Bet Draw", "💰"
            else:
                return "Bet Away Win", "💰"
        elif max_prob >= 0.5 and risk_score <= 50:
            # Moderate confidence - consider betting
            if home_prob == max_prob:
                return "Consider Home", "🤔"
            elif draw_prob == max_prob:
                return "Consider Draw", "🤔"
            else:
                return "Consider Away", "🤔"
        else:
            # Low confidence or high risk - avoid betting
            return "Avoid Betting", "❌"

    betting_recs = []
    betting_emojis = []
    for i, risk_score in enumerate(risk_scores):
        home_prob = upcoming_df.iloc[i]['HomeWin_Prob']
        draw_prob = upcoming_df.iloc[i]['Draw_Prob']
        away_prob = upcoming_df.iloc[i]['AwayWin_Prob']
        rec, emoji = get_betting_recommendation(
            home_prob,
            draw_prob,
            away_prob,
            risk_score
        )
        betting_recs.append(rec)
        betting_emojis.append(emoji)

    # Add all new columns at once to avoid fragmentation
    new_columns = {
        'Risk_Score': risk_scores,
        'Confidence_Score': confidence_scores,
        'Risk_Category': risk_categories,
        'Risk_Emoji': risk_emojis,
        'Betting_Recommendation': betting_recs,
        'Bet_Emoji': betting_emojis,
        'Expected_Home_Goals': upcoming_df['HomeGoalsAve'].round(2),
        'Expected_Away_Goals': upcoming_df['AwayGoalsAve'].round(2)
    }
    new_df = pd.DataFrame(new_columns, index=upcoming_df.index)
    upcoming_df = pd.concat([upcoming_df, new_df], axis=1)

    # Prepare display dataframe with human-readable columns and percentages
    display_cols = ['Date', 'Time', 'HomeTeam', 'AwayTeam', 'HomeWin_Prob', 'Draw_Prob', 'AwayWin_Prob',
                   'Expected_Home_Goals', 'Expected_Away_Goals', 'Risk_Score', 'Risk_Category', 'Confidence_Score', 'Betting_Recommendation']
    if 'Referee' in upcoming_df.columns:
        display_cols.insert(4, 'Referee')

    display_df = upcoming_df[display_cols].copy()
    display_df.columns = ['Match Date', 'Kickoff Time', 'Home Team', 'Away Team'] + \
                        (['Referee'] if 'Referee' in upcoming_df.columns else []) + \
                        ['Home Win %', 'Draw %', 'Away Win %', 'Exp. Home Goals', 'Exp. Away Goals', 'Risk Score', 'Risk Level', 'Confidence %', 'Betting Tip']
    display_df['Home Win %'] = (display_df['Home Win %'] * 100).round(2)
    display_df['Draw %'] = (display_df['Draw %'] * 100).round(2)
    display_df['Away Win %'] = (display_df['Away Win %'] * 100).round(2)
    display_df['Exp. Home Goals'] = display_df['Exp. Home Goals'].round(2)
    display_df['Exp. Away Goals'] = display_df['Exp. Away Goals'].round(2)
    display_df['Confidence %'] = (display_df['Confidence %'] * 100).round(2)
    display_df['Risk Score'] = display_df['Risk Score'].round(2)

    st.subheader("🎯 Upcoming Match Predictions with Risk Assessment")
    st.write("*Times shown in Eastern Time (ET)*")
    if 'Referee' in upcoming_df.columns:
        st.write("✅ **Referee data integrated** - Predictions now include referee statistics from historical matches")

    # Risk level filter
    st.subheader("🔍 Filter by Risk Level")
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        show_all = st.button("📊 All Matches", use_container_width=True, type="secondary")

    with col2:
        show_low = st.button("🟢 Low Risk", use_container_width=True,
                           help="Risk score ≤30: Relatively more confident predictions")

    with col3:
        show_moderate = st.button("🟡 Moderate Risk", use_container_width=True,
                                help="Risk score 31-40: Moderate confidence predictions")

    with col4:
        show_high = st.button("🔴 High Risk", use_container_width=True,
                            help="Risk score 41-47: Lower confidence predictions")

    with col5:
        show_critical = st.button("🚨 Critical Risk", use_container_width=True,
                                help="Risk score >47: Very low confidence predictions")

    # Determine which filter is active - only one can be true at a time
    active_filters = [show_all, show_low, show_moderate, show_high, show_critical]
    active_filter_count = sum(active_filters)

    if active_filter_count == 0:
        # Default to showing all
        filtered_df = display_df.copy()
        active_filter = "All Matches"
    elif active_filter_count == 1:
        # Only one filter is active
        if show_all:
            filtered_df = display_df.copy()
            active_filter = "All Matches"
        elif show_low:
            filtered_df = display_df[display_df['Risk Score'] <= 30].copy()
            active_filter = "Low Risk (≤30)"
        elif show_moderate:
            filtered_df = display_df[(display_df['Risk Score'] > 30) & (display_df['Risk Score'] <= 40)].copy()
            active_filter = "Moderate Risk (31-40)"
        elif show_high:
            filtered_df = display_df[(display_df['Risk Score'] > 40) & (display_df['Risk Score'] <= 47)].copy()
            active_filter = "High Risk (41-47)"
        elif show_critical:
            filtered_df = display_df[display_df['Risk Score'] > 47].copy()
            active_filter = "Critical Risk (>47)"
    else:
        # Multiple filters clicked - show all as fallback
        filtered_df = display_df.copy()
        active_filter = "All Matches (multiple selections detected)"

    # Debug: Show risk score distribution
    with st.expander("🔍 Risk Score Debug (Click to expand)", expanded=False):
        risk_counts = {
            'Low (≤30)': len(display_df[display_df['Risk Score'] <= 30]),
            'Moderate (31-40)': len(display_df[(display_df['Risk Score'] > 30) & (display_df['Risk Score'] <= 40)]),
            'High (41-47)': len(display_df[(display_df['Risk Score'] > 40) & (display_df['Risk Score'] <= 47)]),
            'Critical (>47)': len(display_df[display_df['Risk Score'] > 47])
        }
        st.write("**Risk Score Distribution:**")
        for category, count in risk_counts.items():
            st.write(f"- {category}: {count} matches")

        st.write("**Sample Risk Scores:**")
        sample_df = display_df.head(5)[['Home Team', 'Away Team', 'Risk Score', 'Home Win %', 'Draw %', 'Away Win %']]
        st.dataframe(sample_df, hide_index=True)

    # Risk scoring explanation
    with st.expander("📊 Risk Scoring Methodology", expanded=False):
        st.markdown("""
        **Risk Assessment Framework** (Adapted from HenryOnilude's statistical variance analysis):

        **Risk Score (0-100):**
        - 🟢 **Low Risk (0-30)**: Relatively more confident predictions
        - 🟡 **Moderate Risk (31-40)**: Moderate confidence predictions
        - 🔴 **High Risk (41-47)**: Lower confidence predictions
        - 🚨 **Critical Risk (>47)**: Very low confidence predictions (limited data available)

        **Confidence Score:** Measures prediction certainty (inverse of entropy)
        **Betting Recommendations:** Risk-adjusted suggestions based on confidence and risk levels
        """)

    # Summary statistics (based on filtered data)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if len(filtered_df) > 0:
            low_risk_pct = (filtered_df['Risk Score'] <= 13).sum() / len(filtered_df) * 100
            st.metric("Low Risk in Filter", f"{(filtered_df['Risk Score'] <= 13).sum()}/{len(filtered_df)}", f"{low_risk_pct:.1f}%")
        else:
            st.metric("Low Risk in Filter", "0/0", "0.0%")
    with col2:
        if len(filtered_df) > 0:
            high_conf_pct = (filtered_df['Confidence %'] >= 60).sum() / len(filtered_df) * 100
            st.metric("High Confidence in Filter", f"{(filtered_df['Confidence %'] >= 60).sum()}/{len(filtered_df)}", f"{high_conf_pct:.1f}%")
        else:
            st.metric("High Confidence in Filter", "0/0", "0.0%")
    with col3:
        if len(filtered_df) > 0:
            bettable_pct = filtered_df['Betting Tip'].str.contains('Bet|Consider').sum() / len(filtered_df) * 100
            st.metric("Recommended Bets in Filter", f"{filtered_df['Betting Tip'].str.contains('Bet|Consider').sum()}/{len(filtered_df)}", f"{bettable_pct:.1f}%")
        else:
            st.metric("Recommended Bets in Filter", "0/0", "0.0%")
    with col4:
        if len(filtered_df) > 0:
            avg_risk = filtered_df['Risk Score'].mean()
            st.metric("Average Risk in Filter", f"{avg_risk:.1f}/100")
        else:
            st.metric("Average Risk in Filter", "N/A")

    # Add color styling to the dataframe based on risk levels
    def color_risk_rows(row):
        risk_score = row['Risk Score']
        if risk_score <= 30:
            return ['background-color: #d4edda; color: #155724'] * len(row)  # Green for low risk
        elif risk_score <= 40:
            return ['background-color: #fff3cd; color: #856404'] * len(row)  # Yellow for moderate risk
        elif risk_score <= 47:
            return ['background-color: #f8d7da; color: #721c24'] * len(row)  # Red for high risk
        else:
            return ['background-color: #f5c6cb; color: #721c24'] * len(row)  # Dark red for critical risk

    # Apply styling and display filtered dataframe
    if len(filtered_df) > 0:
        # Apply styling to numeric data and format display
        styled_df = filtered_df.style.apply(color_risk_rows, axis=1).format({
            'Home Win %': '{:.2f}',
            'Draw %': '{:.2f}',
            'Away Win %': '{:.2f}',
            'Exp. Home Goals': '{:.2f}',
            'Exp. Away Goals': '{:.2f}',
            'Confidence %': '{:.2f}',
            'Risk Score': '{:.2f}'
        })
        st.dataframe(styled_df, width='stretch', hide_index=True, height=get_dataframe_height(filtered_df))

        # ── 7c. Injury Intelligence Panel ─────────────────────────────────
        injuries_file = path.join(DATA_DIR, 'api_injuries.csv')
        if path.exists(injuries_file):
            st.markdown("---")
            st.subheader("🏥 Injury Intelligence (2024-25 Season)")
            st.caption("Source: API-Football v3 — 2024 season injury records")
            inj_df = pd.read_csv(injuries_file, sep='\t')
            if not inj_df.empty:
                inj_counts = inj_df.groupby('Team').size().reset_index(name='TotalInjuries')
                inj_types = inj_df.groupby(['Team', 'Type']).size().reset_index(name='Count')
                inj_display = inj_counts.sort_values('TotalInjuries', ascending=False)
                col_inj1, col_inj2 = st.columns([1, 2])
                with col_inj1:
                    st.markdown("**Team Injury Counts (2024-25)**")
                    st.dataframe(inj_display, hide_index=True,
                                 height=get_dataframe_height(inj_display, max_height=400))
                with col_inj2:
                    selected_team = st.selectbox(
                        "View injury details for team:",
                        options=['— Select —'] + sorted(inj_display['Team'].tolist()),
                        key='injury_team_select'
                    )
                    if selected_team != '— Select —':
                        team_inj = inj_df[inj_df['Team'] == selected_team][
                            ['PlayerName', 'Type', 'Reason', 'FixtureDate']
                        ].drop_duplicates().sort_values('FixtureDate', ascending=False)
                        st.markdown(f"**{selected_team} injury records ({len(team_inj)})**")
                        st.dataframe(team_inj, hide_index=True,
                                     height=get_dataframe_height(team_inj, max_height=350))

        # Add prediction logging functionality
        st.markdown("---")
        col1, col2 = st.columns([1, 1])
        
        with col1:
            if st.button("📊 Log Predictions for Tracking", key="log_predictions", 
                        help="Save these predictions to track accuracy over time"):
                from track_predictions import log_prediction
                
                logged_count = 0
                for idx, row in upcoming_df.iterrows():
                    try:
                        log_prediction(
                            row['Date'],
                            row['HomeTeam'], 
                            row['AwayTeam'],
                            row['HomeWin_Prob'],
                            row['Draw_Prob'], 
                            row['AwayWin_Prob']
                        )
                        logged_count += 1
                    except Exception as e:
                        st.error(f"Error logging prediction for {row['HomeTeam']} vs {row['AwayTeam']}: {e}")
                
                if logged_count > 0:
                    st.success(f"✅ Successfully logged {logged_count} predictions for future accuracy tracking!")
                    st.info("Predictions will be automatically validated against actual results as matches are played.")
        
        with col2:
            if st.button("🔄 Refresh & Validate", key="validate_predictions",
                        help="Check for completed matches and update prediction accuracy"):
                from track_predictions import validate_predictions
                perf = validate_predictions()
                if perf is not None:
                    completed = perf[perf['Correct'].notna()]
                    if len(completed) > 0:
                        accuracy = completed['Correct'].mean()
                        st.success(f"✅ Validation complete! Current accuracy: {accuracy:.1%} ({len(completed)} predictions)")
                    else:
                        st.info("No predictions ready for validation yet.")
                else:
                    st.info("No prediction history to validate.")
                    
    else:
        st.info("No matches found for the selected risk level. Try selecting 'All Matches' or a different risk category.")
    
    add_betting_oracle_footer()

with tab4:
    st.subheader("📊 Team Form Guide")
    st.write("Recent performance analysis for all Premier League teams (last 5 matches)")

    # ── Live Standings (2025-26) & Top Players ────────────────────────────
    top_players_file = path.join(DATA_DIR, 'api_top_players.csv')
    _tab4_standings = compute_live_standings(csv_path)
    if not _tab4_standings.empty:
        with st.expander("📋 Current League Table (2025-26 Season)", expanded=False):
            # Home vs Away splits from current season stats
            _cs = compute_current_team_stats(csv_path)
            st.dataframe(_tab4_standings, hide_index=True, use_container_width=True,
                         height=get_dataframe_height(_tab4_standings, max_height=500))
            if not _cs.empty and 'HomeWins' in _cs.columns:
                st.markdown("**Home vs Away breakdown (2025-26):**")
                split_cols_data = _cs[['Team', 'HomeWins', 'HomeDraws', 'HomeLosses',
                                       'AwayWins', 'AwayDraws', 'AwayLosses']].copy()
                split_cols_data.columns = ['Team', 'Home W', 'Home D', 'Home L',
                                           'Away W', 'Away D', 'Away L']
                merged_splits = _tab4_standings[['Rank', 'Team']].merge(split_cols_data, on='Team', how='left')
                st.dataframe(merged_splits.sort_values('Rank'), hide_index=True, use_container_width=True)

    if path.exists(top_players_file):
        with st.expander("⭐ Top Players (2024-25 Season — API Data)", expanded=False):
            tp_df = pd.read_csv(top_players_file, sep='\t')
            scorers = tp_df[tp_df['Category'] == 'TopScorer'][
                ['PlayerName', 'Team', 'Position', 'Appearances', 'Goals', 'Assists', 'Rating']
            ].drop_duplicates().sort_values('Goals', ascending=False)
            assists = tp_df[tp_df['Category'] == 'TopAssist'][
                ['PlayerName', 'Team', 'Position', 'Appearances', 'Goals', 'Assists', 'Rating']
            ].drop_duplicates().sort_values('Assists', ascending=False)
            c_sc, c_as = st.columns(2)
            with c_sc:
                st.markdown("**Top Scorers**")
                st.dataframe(scorers, hide_index=True,
                             height=get_dataframe_height(scorers, max_height=400))
            with c_as:
                st.markdown("**Top Assists**")
                st.dataframe(assists, hide_index=True,
                             height=get_dataframe_height(assists, max_height=400))

    from analyze_team_form import get_team_form_stats
    
    teams = sorted(df['HomeTeam'].unique())
    
    form_data = []
    for team in teams:
        stats = get_team_form_stats(team, num_matches=5)
        form_data.append({
            'Team': team,
            'Last 5': stats['form_string'],
            'Wins': stats['wins'],
            'Draws': stats['draws'],
            'Losses': stats['losses'],
            'Points': stats['points'],
            'Form Score': stats['wins'] * 3 + stats['draws']  # Points-based scoring
        })
    
    form_df = pd.DataFrame(form_data).sort_values('Form Score', ascending=False)
    
    # Format the display
    form_display = form_df.copy()
    form_display['Form Score'] = form_display['Form Score'].astype(int)
    
    st.write(f"**Total Teams Analyzed:** {len(form_df)}")
    st.write("**Key Metrics:**")
    st.write("- **Last 5**: Recent match results (W=Win, D=Draw, L=Loss)")
    st.write("- **Points**: Total points from last 5 matches (3 per win, 1 per draw)")
    st.write("- **Form Score**: Points-based ranking (higher = better form)")
    
    # Add color coding for form strings
    def color_form_results(form_string):
        if not form_string:
            return ''
        colored = []
        for result in form_string:
            if result == 'W':
                colored.append('🟢')  # Green for wins
            elif result == 'D':
                colored.append('🟡')  # Yellow for draws
            else:
                colored.append('🔴')  # Red for losses
        return ' '.join(colored)
    
    # Create a display version with colored form
    display_df = form_display.copy()
    display_df['Form Visual'] = display_df['Last 5'].apply(color_form_results)
    display_df = display_df[['Team', 'Form Visual', 'Last 5', 'Wins', 'Draws', 'Losses', 'Points', 'Form Score']]
    display_df.columns = ['Team', 'Form Visual', 'Results', 'Wins', 'Draws', 'Losses', 'Points', 'Form Score']
    
    st.dataframe(display_df, width='stretch', hide_index=True, height=get_dataframe_height(display_df))
    
    # Add form summary
    st.subheader("Form Summary")
    st.write("**League-wide form analysis:**")
    
    # Calculate summary statistics
    total_matches = form_df['Wins'].sum() + form_df['Draws'].sum() + form_df['Losses'].sum()
    avg_points = form_df['Points'].mean()
    best_form_team = form_df.loc[form_df['Form Score'].idxmax(), 'Team']
    best_form_score = form_df['Form Score'].max()
    
    summary_stats = {
        'Total Matches Analyzed': total_matches,
        'Average Points per Team': f"{avg_points:.1f}",
        'Best Form Team': f"{best_form_team} ({best_form_score} points)",
        'Teams with Perfect Form': len(form_df[form_df['Losses'] == 0]),
        'Teams Winless': len(form_df[form_df['Wins'] == 0])
    }
    
    summary_df = pd.DataFrame(list(summary_stats.items()), columns=['Metric', 'Value'], dtype=object)
    summary_df['Value'] = summary_df['Value'].astype(str)
    summary_df = summary_df.astype(str)
    st.dataframe(summary_df, width=600, hide_index=True, height=get_dataframe_height(summary_df))

    st.markdown("---")
    st.subheader("Manager Statistics")
    st.write("Historical manager performance metrics calculated from Premier League matches (2021-2026)")
    
    # Extract manager statistics from historical data
    manager_cols = ['HomeManager', 'HomeManagerWinRate', 'HomeManagerGoalsPerGame', 'HomeManagerDefensiveSolidity', 
                   'HomeManagerAttackingThreat', 'HomeManagerTacticalFlexibility']
    
    # Get unique managers and their stats
    manager_stats_df = df[manager_cols].drop_duplicates(subset=['HomeManager']).sort_values('HomeManagerWinRate', ascending=False)
    
    # Rename columns for better display
    manager_stats_df.columns = ['Manager', 'Win Rate', 'Goals per Game', 'Defensive Solidity', 
                               'Attacking Threat', 'Tactical Flexibility']
    
    # Format percentages
    percentage_cols = ['Win Rate']
    for col in percentage_cols:
        if col in manager_stats_df.columns:
            manager_stats_df[col] = (manager_stats_df[col] * 100).round(1)
    
    # Format decimal columns
    decimal_cols = ['Goals per Game', 'Defensive Solidity', 'Attacking Threat', 'Tactical Flexibility']
    for col in decimal_cols:
        if col in manager_stats_df.columns:
            manager_stats_df[col] = manager_stats_df[col].round(2)
    
    st.write(f"**Total Managers:** {len(manager_stats_df)}")
    st.write("**Key Metrics:**")
    st.write("- **Win Rate**: Historical winning percentage as manager")
    st.write("- **Goals per Game**: Average goals scored per match under their management")
    st.write("- **Defensive Solidity**: Rating of defensive organization (higher = better defense)")
    st.write("- **Attacking Threat**: Rating of offensive capability (higher = more dangerous attack)")
    st.write("- **Tactical Flexibility**: Rating of adaptability to different tactical situations")
    
    st.dataframe(manager_stats_df, width=800, hide_index=True, height=get_dataframe_height(manager_stats_df))
    
    # Add summary statistics
    st.subheader("Manager Summary Statistics")
    st.write("**League-wide averages across all managers:**")
    
    # Calculate averages for numeric columns
    numeric_cols = ['Goals per Game', 'Defensive Solidity', 'Attacking Threat', 'Tactical Flexibility']
    percentage_cols = ['Win Rate']
    
    # Create summary dataframe
    summary_data = {}
    
    # Numeric columns - calculate mean
    for col in numeric_cols:
        if col in manager_stats_df.columns:
            summary_data[col] = manager_stats_df[col].mean()
    
    # Percentage columns - calculate mean and format as percentage
    for col in percentage_cols:
        if col in manager_stats_df.columns:
            raw_values = manager_stats_df[col] / 100  # Convert back from percentage
            summary_data[col] = raw_values.mean()
    
    # Create summary dataframe
    summary_df = pd.DataFrame([summary_data])
    
    # Format the display
    summary_display = summary_df.copy()
    for col in percentage_cols:
        if col in summary_display.columns:
            summary_display[col] = (summary_display[col] * 100).round(1)
    
    for col in ['Goals per Game', 'Defensive Solidity', 'Attacking Threat', 'Tactical Flexibility']:
        if col in summary_display.columns:
            summary_display[col] = summary_display[col].round(2)
    
    st.dataframe(summary_display, width=600, hide_index=True)
    
    st.subheader("Referee Statistics")
    st.write("Historical referee performance metrics calculated from Premier League matches (2021-2026)")
    
    # Extract referee statistics from historical data
    referee_cols = ['Referee', 'RefTotalMatches', 'RefYellowCardsPerGame', 'RefRedCardsPerGame', 
                   'RefFoulsPerGame', 'RefHomeAdvantageYellow', 'RefHomeWinRate', 'RefAwayWinRate', 'RefDrawRate']
    
    referee_stats_df = df[referee_cols].drop_duplicates().sort_values('RefTotalMatches', ascending=False)
    
    # Rename columns for better display
    referee_stats_df.columns = ['Referee', 'Total Matches', 'Yellow Cards/Game', 'Red Cards/Game', 
                               'Fouls/Game', 'Home Yellow Advantage', 'Home Win Rate', 'Away Win Rate', 'Draw Rate']
    
    # Format percentages
    percentage_cols = ['Home Win Rate', 'Away Win Rate', 'Draw Rate']
    for col in percentage_cols:
        if col in referee_stats_df.columns:
            referee_stats_df[col] = (referee_stats_df[col] * 100).round(1)
    
    # Format decimal columns
    decimal_cols = ['Yellow Cards/Game', 'Red Cards/Game', 'Fouls/Game', 'Home Yellow Advantage']
    for col in decimal_cols:
        if col in referee_stats_df.columns:
            referee_stats_df[col] = referee_stats_df[col].round(2)
    
    st.write(f"**Total Referees:** {len(referee_stats_df)}")
    st.write("**Key Metrics:**")
    st.write("- **Yellow/Red Cards per Game**: Disciplinary strictness indicators")
    st.write("- **Home Yellow Advantage**: Positive values indicate referees give more yellow cards to home teams")
    st.write("- **Win Rates**: Historical outcome percentages when referee officiates")
    
    st.dataframe(referee_stats_df, width='stretch', hide_index=True, height=get_dataframe_height(referee_stats_df))
    
    # Add summary statistics
    st.subheader("Summary Statistics")
    st.write("**League-wide averages across all referees:**")
    
    # Calculate averages for numeric columns
    numeric_cols = ['Total Matches', 'Yellow Cards/Game', 'Red Cards/Game', 'Fouls/Game', 'Home Yellow Advantage']
    percentage_cols = ['Home Win Rate', 'Away Win Rate', 'Draw Rate']
    
    # Create summary dataframe
    summary_data = {}
    
    # Numeric columns - calculate mean
    for col in numeric_cols:
        if col in referee_stats_df.columns:
            # Convert back to raw values for percentage columns
            if col in ['Home Win Rate', 'Away Win Rate', 'Draw Rate']:
                raw_values = referee_stats_df[col] / 100  # Convert back from percentage
                summary_data[col] = raw_values.mean()
            else:
                summary_data[col] = referee_stats_df[col].mean()
    
    # Percentage columns - calculate mean and format as percentage
    for col in percentage_cols:
        if col in referee_stats_df.columns:
            raw_values = referee_stats_df[col] / 100  # Convert back from percentage
            summary_data[col] = raw_values.mean()
    
    # Create summary dataframe
    summary_df = pd.DataFrame([summary_data])
    
    # Format the display
    summary_display = summary_df.copy()
    for col in percentage_cols:
        if col in summary_display.columns:
            summary_display[col] = (summary_display[col] * 100).round(1)
    
    for col in ['Yellow Cards/Game', 'Red Cards/Game', 'Fouls/Game', 'Home Yellow Advantage']:
        if col in summary_display.columns:
            summary_display[col] = summary_display[col].round(2)
    
    # Rename columns for display
    summary_display.columns = ['Avg Total Matches', 'Avg Yellow Cards/Game', 'Avg Red Cards/Game', 
                              'Avg Fouls/Game', 'Avg Home Yellow Advantage', 'Avg Home Win %', 
                              'Avg Away Win %', 'Avg Draw %']
    
    st.dataframe(summary_display, width='stretch', hide_index=True)

    st.subheader("Formation Statistics")
    st.write("Historical performance analysis by tactical formation (2021-2026)")
    
    # Extract formation statistics from historical data
    formation_cols = ['HomeManagerFormation', 'AwayManagerFormation', 'FullTimeResult']
    
    # Analyze home formations
    home_formation_stats = df.groupby('HomeManagerFormation').agg({
        'FullTimeResult': ['count', lambda x: (x == 'H').mean(), lambda x: (x == 'D').mean(), lambda x: (x == 'A').mean()]
    }).round(4)
    
    home_formation_stats.columns = ['Matches', 'Home_Win_Rate', 'Draw_Rate', 'Away_Win_Rate']
    home_formation_stats = home_formation_stats.sort_values('Home_Win_Rate', ascending=False)
    
    # Analyze away formations
    away_formation_stats = df.groupby('AwayManagerFormation').agg({
        'FullTimeResult': ['count', lambda x: (x == 'H').mean(), lambda x: (x == 'D').mean(), lambda x: (x == 'A').mean()]
    }).round(4)
    
    away_formation_stats.columns = ['Matches', 'Home_Win_Rate', 'Draw_Rate', 'Away_Win_Rate']
    away_formation_stats = away_formation_stats.sort_values('Away_Win_Rate', ascending=False)
    
    # Create combined display
    combined_stats = []
    for formation in home_formation_stats.index:
        if formation in away_formation_stats.index:
            home_stats = home_formation_stats.loc[formation]
            away_stats = away_formation_stats.loc[formation]
            combined_stats.append({
                'Formation': formation,
                'Home Matches': int(home_stats['Matches']),
                'Home Win Rate': home_stats['Home_Win_Rate'],
                'Away Matches': int(away_stats['Matches']),
                'Away Win Rate': away_stats['Away_Win_Rate'],
                'Total Matches': int(home_stats['Matches'] + away_stats['Matches'])
            })
    
    formation_df = pd.DataFrame(combined_stats).sort_values('Total Matches', ascending=False)
    
    # Format percentages
    percentage_cols = ['Home Win Rate', 'Away Win Rate']
    for col in percentage_cols:
        if col in formation_df.columns:
            formation_df[col] = (formation_df[col] * 100).round(1)
    
    st.write(f"**Total Formations Analyzed:** {len(formation_df)}")
    st.write("**Key Insights:**")
    st.write("- **Home Win Rate**: Percentage of matches won when using this formation at home")
    st.write("- **Away Win Rate**: Percentage of matches won when using this formation away")
    st.write("- **Note**: Formations are associated with managers and may correlate with team quality")
    
    st.dataframe(formation_df, width='stretch', hide_index=True, height=get_dataframe_height(formation_df))
    
    # Add formation summary
    st.subheader("Formation Summary")
    st.write("**Formation popularity and performance overview:**")
    
    # Calculate summary statistics
    summary_stats = {
        'Most Popular Formation': formation_df.loc[formation_df['Total Matches'].idxmax(), 'Formation'],
        'Highest Home Win Rate': f"{formation_df.loc[formation_df['Home Win Rate'].idxmax(), 'Formation']} ({formation_df['Home Win Rate'].max()}%)",
        'Highest Away Win Rate': f"{formation_df.loc[formation_df['Away Win Rate'].idxmax(), 'Formation']} ({formation_df['Away Win Rate'].max()}%)",
        'Average Home Win Rate': f"{formation_df['Home Win Rate'].mean():.1f}%",
        'Average Away Win Rate': f"{formation_df['Away Win Rate'].mean():.1f}%"
    }
    
    summary_df = pd.DataFrame(list(summary_stats.items()), columns=['Metric', 'Value'], dtype=object)
    summary_df['Value'] = summary_df['Value'].astype(str)
    summary_df = summary_df.astype(str)
    st.dataframe(summary_df, width=600, hide_index=True, height=get_dataframe_height(summary_df))

    st.markdown("---")
    st.subheader("🏆 Head-to-Head Analyzer")
    st.write("Compare historical match results between any two Premier League teams")

    # Head-to-Head History function
    def get_h2h_stats(home_team, away_team, num_matches=10):
        """Get head-to-head statistics between two teams"""
        h2h = df[
            ((df['HomeTeam'] == home_team) & (df['AwayTeam'] == away_team)) |
            ((df['HomeTeam'] == away_team) & (df['AwayTeam'] == home_team))
        ].sort_values('MatchDate', ascending=False).head(num_matches)

        return h2h[['MatchDate', 'HomeTeam', 'AwayTeam', 'FullTimeHomeGoals', 'FullTimeAwayGoals', 'FullTimeResult']]

    # UI Component for team selection
    col1, col2 = st.columns(2)
    with col1:
        team1 = st.selectbox("Select Team 1", sorted(df['HomeTeam'].unique()), key="h2h_team1")
    with col2:
        team2 = st.selectbox("Select Team 2", sorted(df['AwayTeam'].unique()), key="h2h_team2")

    if st.button("🔍 Analyze Head-to-Head History", key="h2h_button"):
        if team1 != team2:
            h2h_df = get_h2h_stats(team1, team2, num_matches=10)

            if len(h2h_df) > 0:
                st.success(f"Found {len(h2h_df)} historical matches between {team1} and {team2}")

                # Calculate H2H statistics
                team1_wins = 0
                team2_wins = 0
                draws = 0
                team1_goals = 0
                team2_goals = 0

                for _, match in h2h_df.iterrows():
                    if match['HomeTeam'] == team1:
                        team1_goals += match['FullTimeHomeGoals']
                        team2_goals += match['FullTimeAwayGoals']
                        if match['FullTimeResult'] == 'H':
                            team1_wins += 1
                        elif match['FullTimeResult'] == 'A':
                            team2_wins += 1
                        else:
                            draws += 1
                    else:  # team1 is away
                        team1_goals += match['FullTimeAwayGoals']
                        team2_goals += match['FullTimeHomeGoals']
                        if match['FullTimeResult'] == 'A':
                            team1_wins += 1
                        elif match['FullTimeResult'] == 'H':
                            team2_wins += 1
                        else:
                            draws += 1

                # Display summary statistics
                st.subheader("📊 Head-to-Head Summary")
                summary_cols = st.columns(4)
                with summary_cols[0]:
                    st.metric(f"{team1} Wins", team1_wins)
                with summary_cols[1]:
                    st.metric(f"{team2} Wins", team2_wins)
                with summary_cols[2]:
                    st.metric("Draws", draws)
                with summary_cols[3]:
                    st.metric("Total Goals", f"{team1_goals}-{team2_goals}")

                # Display recent matches
                st.subheader("📅 Recent Matches")
                st.write("Most recent matches first:")

                # Format the dataframe for better display
                display_h2h = h2h_df.copy()
                display_h2h['Match Date'] = pd.to_datetime(display_h2h['MatchDate']).dt.strftime('%m/%d/%Y')

                # Add result interpretation
                def format_result(row):
                    if row['HomeTeam'] == team1:
                        if row['FullTimeResult'] == 'H':
                            return f"{team1} {row['FullTimeHomeGoals']}-{row['FullTimeAwayGoals']} {team2} (W)"
                        elif row['FullTimeResult'] == 'A':
                            return f"{team1} {row['FullTimeHomeGoals']}-{row['FullTimeAwayGoals']} {team2} (L)"
                        else:
                            return f"{team1} {row['FullTimeHomeGoals']}-{row['FullTimeAwayGoals']} {team2} (D)"
                    else:
                        if row['FullTimeResult'] == 'A':
                            return f"{team2} {row['FullTimeHomeGoals']}-{row['FullTimeAwayGoals']} {team1} (W)"
                        elif row['FullTimeResult'] == 'H':
                            return f"{team2} {row['FullTimeHomeGoals']}-{row['FullTimeAwayGoals']} {team1} (L)"
                        else:
                            return f"{team2} {row['FullTimeHomeGoals']}-{row['FullTimeAwayGoals']} {team1} (D)"

                display_h2h['Result'] = display_h2h.apply(format_result, axis=1)
                display_h2h = display_h2h[['Match Date', 'Result']]

                st.dataframe(display_h2h, width=400, hide_index=True, height=get_dataframe_height(display_h2h))
            else:
                st.warning(f"No historical matches found between {team1} and {team2}")
        else:
            st.warning("Please select two different teams to compare")
    
    add_betting_oracle_footer()

with tab5:
    st.subheader("Historical Data")
    df_sorted = df.sort_values(by=['MatchDate', 'KickoffTime'], ascending=[False, False])
    st.dataframe(df_sorted, height=get_dataframe_height(df_sorted), width='stretch', hide_index=True)
    
    add_betting_oracle_footer()

# ── Tab 6: Team Deep Dive (Enhancement 7a) ────────────────────────────────────
with tab6:
    st.subheader("🔬 Team Deep Dive")
    st.write("Detailed season statistics and performance breakdowns for the current 2025-26 season.")

    top_players_file_t6 = path.join(DATA_DIR, 'api_top_players.csv')
    injuries_file_t6 = path.join(DATA_DIR, 'api_injuries.csv')
    team_stats_file = path.join(DATA_DIR, 'api_team_statistics.csv')  # 2024-25 supplemental data

    _t6_stats = compute_current_team_stats(csv_path)
    available_teams = sorted(_t6_stats['Team'].dropna().unique().tolist()) if not _t6_stats.empty else []

    selected_team_dive = st.selectbox("Select a team:", available_teams, key='team_deep_dive')

    if selected_team_dive and not _t6_stats.empty:
        team_row = _t6_stats[_t6_stats['Team'] == selected_team_dive]
        if team_row.empty:
            st.warning(f"No 2025-26 data found for {selected_team_dive}")
        else:
            team_row = team_row.iloc[0]
            played = max(int(team_row.get('TotalPlayed', 1)), 1)
            wins = int(team_row.get('TotalWins', 0))
            cs = int(team_row.get('CleanSheetTotal', 0))
            gf = int(team_row.get('GoalsForTotal', 0))
            ga = int(team_row.get('GoalsAgainstTotal', 0))
            pts = int(team_row.get('Points', 0))

            # ── Season overview KPIs ─────────────────────────────────────
            st.markdown(f"### {selected_team_dive} — 2025-26 Season Statistics")
            k1, k2, k3, k4, k5 = st.columns(5)
            with k1:
                st.metric("Matches Played", played)
            with k2:
                st.metric("Points", pts)
            with k3:
                st.metric("Win Rate", f"{wins / played:.0%}")
            with k4:
                st.metric("Clean Sheets", cs, f"{cs / played:.0%}")
            with k5:
                st.metric("Goals For / Against", f"{gf} / {ga}", f"GD: {gf - ga:+d}")

            st.divider()
            col_a, col_b = st.columns(2)

            # ── Home vs Away stats ────────────────────────────────────────
            with col_a:
                st.markdown("**Home vs Away Performance (2025-26)**")
                ha_data = {
                    'Metric': ['Played', 'Wins', 'Goals Scored (avg)', 'Goals Conceded (avg)',
                               'Clean Sheets', 'Failed to Score'],
                    'Home': [
                        int(team_row.get('HomePlayed', 0)),
                        int(team_row.get('HomeWins', 0)),
                        round(float(team_row.get('GoalsForAvgHome', 0)), 2),
                        round(float(team_row.get('GoalsAgainstAvgHome', 0)), 2),
                        int(team_row.get('CleanSheetHome', 0)),
                        int(team_row.get('FailedToScoreHome', 0)),
                    ],
                    'Away': [
                        int(team_row.get('AwayPlayed', 0)),
                        int(team_row.get('AwayWins', 0)),
                        round(float(team_row.get('GoalsForAvgAway', 0)), 2),
                        round(float(team_row.get('GoalsAgainstAvgAway', 0)), 2),
                        int(team_row.get('CleanSheetAway', 0)),
                        int(team_row.get('FailedToScoreAway', 0)),
                    ],
                }
                st.dataframe(pd.DataFrame(ha_data), hide_index=True, use_container_width=True)

            # ── Supplemental 2024-25 detail (goals by minute, penalties, streaks) ──
            with col_b:
                if path.exists(team_stats_file):
                    ts_df = pd.read_csv(team_stats_file, sep='\t')
                    ts_row_data = ts_df[ts_df['Team'] == selected_team_dive]
                    if not ts_row_data.empty:
                        ts_row = ts_row_data.iloc[0]
                        st.markdown("**Goals Scored by Time Period** *(2024-25 API data)*")
                        period_cols = {
                            '0–15': 'GoalsFor_0_15', '16–30': 'GoalsFor_16_30',
                            '31–45': 'GoalsFor_31_45', '46–60': 'GoalsFor_46_60',
                            '61–75': 'GoalsFor_61_75', '76–90': 'GoalsFor_76_90',
                            '91+': 'GoalsFor_91_105',
                        }
                        period_data = [
                            {'Period': lbl, 'Goals': int(ts_row.get(col, 0))}
                            for lbl, col in period_cols.items()
                            if ts_row.get(col) is not None and not pd.isna(ts_row.get(col, None))
                        ]
                        if period_data:
                            st.bar_chart(pd.DataFrame(period_data).set_index('Period')['Goals'])
                        formation = ts_row.get('MostUsedFormation', '—')
                        pen_scored = ts_row.get('PenaltyScored', 0)
                        pen_missed = ts_row.get('PenaltyMissed', 0)
                        pen_total = (pen_scored or 0) + (pen_missed or 0)
                        pen_rate = (pen_scored or 0) / max(pen_total, 1) if pen_total else 0
                        st.markdown(f"**Preferred Formation (2024-25):** {formation}")
                        if pen_total:
                            st.markdown(
                                f"**Penalties (2024-25):** {int(pen_scored or 0)} scored / "
                                f"{int(pen_missed or 0)} missed — {pen_rate:.0%} conversion"
                            )
                        st.markdown(
                            f"**Best Streaks (2024-25):** "
                            f"Wins: {int(ts_row.get('BiggestStreakWins', 0))}, "
                            f"Draws: {int(ts_row.get('BiggestStreakDraws', 0))}, "
                            f"Losses: {int(ts_row.get('BiggestStreakLoses', 0))}"
                        )
                    else:
                        st.info("No supplemental 2024-25 details available for this team.")
                else:
                    st.info("Run `python fetch_api_football.py --weekly-mon` to load supplemental stats.")

            st.divider()

            # ── Top players for this team ─────────────────────────────────
            if path.exists(top_players_file_t6):
                tp_t6 = pd.read_csv(top_players_file_t6, sep='\t')
                team_players = tp_t6[tp_t6['Team'] == selected_team_dive][
                    ['PlayerName', 'Category', 'Position', 'Appearances', 'Goals', 'Assists', 'Rating']
                ].drop_duplicates()
                if not team_players.empty:
                    st.markdown("**Key Players — 2024-25 API Data** *(top-scorer & top-assist lists)*")
                    st.dataframe(team_players.sort_values('Goals', ascending=False),
                                 hide_index=True, use_container_width=True)

            # ── Current injuries ──────────────────────────────────────
            if path.exists(injuries_file_t6):
                inj_t6 = pd.read_csv(injuries_file_t6, sep='\t')
                team_inj = inj_t6[inj_t6['Team'] == selected_team_dive][
                    ['PlayerName', 'Type', 'Reason', 'FixtureDate']
                ].drop_duplicates().sort_values('FixtureDate', ascending=False)
                if not team_inj.empty:
                    st.markdown(f"**Injury Records — 2024-25 API Data ({len(team_inj)} records)**")
                    with st.expander(f"View all {selected_team_dive} injury records", expanded=False):
                        st.dataframe(team_inj, hide_index=True, use_container_width=True,
                                     height=get_dataframe_height(team_inj, max_height=350))

            # ── Historical form from existing data ───────────────────
            st.markdown("**Historical Performance (all seasons in dataset)**")
            team_home_matches = df[df['HomeTeam'] == selected_team_dive].copy()
            team_away_matches = df[df['AwayTeam'] == selected_team_dive].copy()
            total_played = len(team_home_matches) + len(team_away_matches)
            if total_played > 0:
                home_wins = (team_home_matches['FullTimeResult'] == 'H').sum()
                away_wins = (team_away_matches['FullTimeResult'] == 'A').sum()
                draws_h = (team_home_matches['FullTimeResult'] == 'D').sum()
                draws_a = (team_away_matches['FullTimeResult'] == 'D').sum()
                total_wins = home_wins + away_wins
                total_draws = draws_h + draws_a
                hk1, hk2, hk3, hk4 = st.columns(4)
                with hk1:
                    st.metric("Total Matches", total_played)
                with hk2:
                    st.metric("All-Time Win Rate", f"{total_wins / total_played:.1%}")
                with hk3:
                    st.metric("All-Time Draw Rate", f"{total_draws / total_played:.1%}")
                with hk4:
                    avg_gf = (team_home_matches['FullTimeHomeGoals'].sum()
                              + team_away_matches['FullTimeAwayGoals'].sum()) / total_played
                    st.metric("Avg Goals / Match", f"{avg_gf:.2f}")

    add_betting_oracle_footer()

