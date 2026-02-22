import pandas as pd
from premier_league_predictions import get_poisson_metrics

# create dummy history
hist = pd.DataFrame({
    'date': ['2026-02-21 15:00', '2026-02-22 16:00'],
    'home_mae': [1.0, 1.1],
    'away_mae': [1.05, 1.08],
    'home_rmse': [1.3, 1.4],
    'away_rmse': [1.2, 1.25],
    'outcome_acc': [0.4, 0.41]
})
hist.to_csv('data_files/poisson_metrics_history.csv', index=False)
print('created dummy history')

# call cached function
print(get_poisson_metrics())
