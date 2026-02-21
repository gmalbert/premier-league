"""Shared evaluation utilities for the Poisson goals predictor."""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, brier_score_loss

from .poisson_predictor import PoissonPredictor


def evaluate_poisson_dataframe(df: pd.DataFrame) -> dict:
    """Evaluate a PoissonPredictor on a historical dataframe.

    Args:
        df: DataFrame containing at least the following columns:
            ['HomeTeam','AwayTeam','HomeGoalsAve','AwayGoalsAve',
             'FullTimeHomeGoals','FullTimeAwayGoals','FullTimeResult']

    Returns:
        dict with keys:
            league_avg, home_mae, away_mae, home_rmse, away_rmse,
            outcome_acc, brier_home, brier_draw, brier_away
    """
    # Copy to avoid modifying original
    df = df.copy()

    # compute goals conceded averages if not present
    if 'HomeGoalsConcededAve' not in df.columns:
        df['HomeGoalsConcededAve'] = df.groupby('HomeTeam')['FullTimeAwayGoals'].transform('mean')
    if 'AwayGoalsConcededAve' not in df.columns:
        df['AwayGoalsConcededAve'] = df.groupby('AwayTeam')['FullTimeHomeGoals'].transform('mean')

    # drop rows missing stats
    required = ['HomeGoalsAve', 'AwayGoalsAve', 'HomeGoalsConcededAve', 'AwayGoalsConcededAve']
    df = df.dropna(subset=required)

    pred = PoissonPredictor()
    league_avg = (df['FullTimeHomeGoals'] + df['FullTimeAwayGoals']).mean() / 2
    pred.league_avg_goals = league_avg

    home_exp = []
    away_exp = []
    outcome_probs = []

    for _, row in df.iterrows():
        h_exp, a_exp = pred.estimate_goals(
            home_attack=row['HomeGoalsAve'],
            home_defense=row['HomeGoalsConcededAve'],
            away_attack=row['AwayGoalsAve'],
            away_defense=row['AwayGoalsConcededAve'],
        )
        home_exp.append(h_exp)
        away_exp.append(a_exp)
        score_mat = pred.poisson_scoreline_probabilities(h_exp, a_exp, max_goals=10)
        outcome_probs.append(pred.predict_match_outcome(score_mat))

    home_exp = np.array(home_exp)
    away_exp = np.array(away_exp)
    pred_outcome = np.argmax(np.vstack(outcome_probs), axis=1)

    y_home = df['FullTimeHomeGoals'].values
    y_away = df['FullTimeAwayGoals'].values
    y_outcome = df['FullTimeResult'].map({'H': 0, 'D': 1, 'A': 2}).values

    y_home_win = (y_outcome == 0).astype(int)
    y_draw = (y_outcome == 1).astype(int)
    y_away_win = (y_outcome == 2).astype(int)
    prob_array = np.vstack(outcome_probs)

    metrics = {
        'league_avg': league_avg,
        'home_mae': mean_absolute_error(y_home, home_exp),
        'away_mae': mean_absolute_error(y_away, away_exp),
        'home_rmse': np.sqrt(mean_squared_error(y_home, home_exp)),
        'away_rmse': np.sqrt(mean_squared_error(y_away, away_exp)),
        'outcome_acc': accuracy_score(y_outcome, pred_outcome),
        'brier_home': brier_score_loss(y_home_win, prob_array[:, 0]),
        'brier_draw': brier_score_loss(y_draw, prob_array[:, 1]),
        'brier_away': brier_score_loss(y_away_win, prob_array[:, 2]),
    }
    return metrics


def evaluate_poisson_file(path: str) -> dict:
    """Convenience wrapper to load a csv and evaluate it."""
    df = pd.read_csv(path, sep='\t')
    return evaluate_poisson_dataframe(df)
