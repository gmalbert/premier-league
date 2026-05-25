<p align="center">
	<img src="data_files/logo.png" alt="Pitch Oracle Logo" width="320" />
</p>

# Premier League Predictor

A friendly app and data pipeline for predicting English Premier League match outcomes.

--

## Table of Contents

- [What this project does (for fans)](#what-this-project-does-for-fans)
- [What's already implemented](#whats-already-implemented)
- [Roadmaps (plans & code samples)](#roadmaps-plans--code-samples)
- [How to run (technical)](#how-to-run-technical)
- [Data & Credits](#data--credits)
- [Back to top](#premier-league-predictor)

--

## What this project does (for fans)

This project predicts the likely outcome of upcoming Premier League matches (home win, draw, or away win) using historical match data and machine learning. It also shows upcoming fixtures, kickoff times, live standings, and explanatory analytics so fans can quickly understand which team is favored.

The app now includes:
- A dedicated **Standings** tab with current Premier League table computation from the historical dataset.
- A **Upcoming Matches** tab with fixtures filtered to include today and sorted chronologically.
- A **Upcoming Predictions** tab with matchup probabilities, model selection, and improved feature alignment for more realistic predictions.
- A **Statistics** tab with referee analytics, team form, manager statistics, and league-level performance metrics.
- A **Raw Data** tab for instant access to the underlying processed dataset.

The prediction engine now supports:
- **Ensemble modeling** combining XGBoost, Random Forest, Gradient Boosting, and Logistic Regression for stronger outcome forecasts.
- **Neural network prediction** for non-linear patterns.
- **LSTM time series prediction** for momentum and seasonal dynamics.
- **Poisson regression diagnostics** for goal-based forecasts and MAE/RMSE performance tracking.

[Back to top](#premier-league-predictor)

--

## What's already implemented

- A data pipeline that combines historical match CSVs into a processed dataset.
- A Streamlit app (`premier-league-predictions.py`) that:
	- Displays upcoming fixtures, home/away teams, kickoff times, and match countdowns.
	- Uses a dedicated **Standings** tab for current table display.
	- Uses a dedicated **Statistics** tab for referees, team form, manager data, and league analytics.
	- Supports **model comparison** across ensemble, Poisson, neural network, and LSTM models.
	- Includes an **Upcoming Predictions** tab with probabilities calculated from aligned training features.
	- Shows a placeholder message when no upcoming fixtures are available.
	- Places the fixture refresh button directly below the upcoming fixtures heading.
- A fit-for-purpose model workflow that includes:
	- **Ensemble modeling** for better accuracy and robustness.
	- **Neural network support** for deep learning predictions.
	- **LSTM time series modeling** for momentum-aware forecasts.
- An ESPN-based fixture fetcher (`fetch_upcoming_fixtures.py`) that pulls upcoming matches and saves them to `data_files/upcoming_fixtures.csv`.
- Referee data integration: Scrapes referee assignments and merges referee stats from Playmaker Stats.
- Team form and performance tracking using rolling averages and historical match statistics.
- GitHub Actions pipeline updates to support longer pipeline execution time.

[Back to top](#premier-league-predictor)

--

## Roadmaps (plans & code samples)

Detailed roadmaps and code samples have been added in the `docs/` folder. These break the work into features, model ideas, data improvements and infrastructure steps. Pick a roadmap to explore:

- [Features Roadmap](docs/roadmap-features.md)
- [Models Roadmap](docs/roadmap-models.md)
- [Data Roadmap](docs/roadmap-data.md)
- [Quick Wins](docs/roadmap-quick-wins.md)
- [Infrastructure Roadmap](docs/roadmap-infrastructure.md)
- [Roadmap Index](docs/README.md)

[Back to top](#premier-league-predictor)

--

## How to run (technical)

These instructions are for developers or power users who want to run the app locally.

Prerequisites

- Python 3.9+ (Windows, macOS, or Linux)
- A virtual environment (recommended)

Install dependencies (example):

```bash
python -m venv venv
venv\Scripts\Activate.ps1  # Windows PowerShell
# or: source venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
```

Fetch upcoming fixtures (optional) and generate processed data:

```bash
python fetch_upcoming_fixtures.py  # pulls upcoming matches from ESPN API
python combineHistorical.py        # combine raw CSVs (if you maintain raw files)
python prepare_model_data.py       # process and generate features
```

Run the Poisson evaluation script to compute goal‑prediction metrics (used in the app and CI):

```bash
python evaluate_poisson.py
# or to verify via unit test (use module form so the runner is found)
python -m pytest test_poisson_evaluation.py
```

Run the Streamlit app:

```bash
streamlit run premier-league-predictions.py
```

Notes for developers

- The Streamlit UI has tabs for: Upcoming Matches, Standings, Predictive Data, Upcoming Predictions, Statistics, and Raw Data.
- Upcoming matches are filtered to include today and are sorted chronologically.
- The fixture refresh button is shown under the Upcoming Fixtures heading rather than in the sidebar.
- The Statistics tab displays referee performance metrics, manager statistics, team form analysis, and league-wide averages.
- **NEW: Ensemble model** combines XGBoost, Random Forest, Gradient Boosting, and Logistic Regression using soft voting for improved accuracy.
- **NEW: Neural network support** using PyTorch with 3-layer architecture, batch normalization, and dropout regularization.
- **NEW: LSTM time series support** for momentum-aware forecasts.
- Models are trained in-memory when you open the 'Predictive Data' section; for production you may want to train offline and load a saved model.
- If you add third-party APIs (e.g., weather, injuries), add keys to a local `.env` and do not commit them.

[Back to top](#premier-league-predictor)

--

## Data & Credits

- Historical match data is pulled from CSVs sourced from football-data.co.uk and processed into `data_files/combined_historical_data_with_calculations.csv`.
- Upcoming fixtures are fetched via the ESPN API (site.api.espn.com).
- Libraries used: `pandas`, `numpy`, `xgboost`, `scikit-learn`, `streamlit`, `requests`, `beautifulsoup4`, `torch`, `torchvision`.
- **Model Enhancement:** Ensemble approach using scikit-learn's VotingClassifier combines multiple algorithms for improved accuracy. Neural network support via PyTorch provides deep learning capabilities.

If you reuse data or publish results, please credit the original data sources.

[Back to top](#premier-league-predictor)
