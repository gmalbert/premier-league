from models.poisson_evaluation import evaluate_poisson_file


def main():
    csv_path = 'data_files/combined_historical_data_with_calculations.csv'
    metrics = evaluate_poisson_file(csv_path)

    print('\n=== Poisson Goal Prediction Metrics ===')
    print(f"League average goals (used): {metrics['league_avg']:.3f}")
    print(f"Home goals MAE: {metrics['home_mae']:.3f}")
    print(f"Away goals MAE: {metrics['away_mae']:.3f}")
    print(f"Home goals RMSE: {metrics['home_rmse']:.3f}")
    print(f"Away goals RMSE: {metrics['away_rmse']:.3f}")

    print('\n=== Derived Outcome Metrics (from Poisson probabilities) ===')
    print(f"Classification accuracy: {metrics['outcome_acc']:.3f}")
    print(f"Brier score (home win): {metrics['brier_home']:.3f}")
    print(f"Brier score (draw): {metrics['brier_draw']:.3f}")
    print(f"Brier score (away win): {metrics['brier_away']:.3f}")


if __name__ == '__main__':
    main()
