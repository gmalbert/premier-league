"""Basic unit test for the Poisson evaluation functionality."""
import math
from models.poisson_evaluation import evaluate_poisson_file


def test_poisson_metrics_exist():
    metrics = evaluate_poisson_file('data_files/combined_historical_data_with_calculations.csv')
    # Expected keys
    expected = [
        'league_avg', 'home_mae', 'away_mae', 'home_rmse', 'away_rmse',
        'outcome_acc', 'brier_home', 'brier_draw', 'brier_away'
    ]
    for key in expected:
        assert key in metrics, f"Missing metric {key}"
        assert not math.isnan(metrics[key]), f"Metric {key} is NaN"
    # Basic sanity: MAE should be >=0 and less than league_avg*3
    assert metrics['home_mae'] >= 0
    assert metrics['away_mae'] >= 0
    assert metrics['home_rmse'] >= metrics['home_mae']
    assert metrics['away_rmse'] >= metrics['away_mae']
    assert 0 <= metrics['outcome_acc'] <= 1
    # Brier scores between 0 and 1
    assert 0 <= metrics['brier_home'] <= 1
    assert 0 <= metrics['brier_draw'] <= 1
    assert 0 <= metrics['brier_away'] <= 1

    # history file should exist and have correct columns
    hist_path = 'data_files/poisson_metrics_history.csv'
    import os, pandas as pd
    assert os.path.exists(hist_path), 'History CSV missing'
    hist = pd.read_csv(hist_path)
    for col in ['home_mae', 'away_mae', 'home_rmse', 'away_rmse', 'outcome_acc']:
        assert col in hist.columns
    # last entry should be non-negative
    last = hist.iloc[-1]
    assert last['home_mae'] >= 0
    assert last['away_mae'] >= 0
