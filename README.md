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

This project predicts the likely outcome of upcoming Premier League matches (home win, draw, or away win) using historical match data and machine learning. It also shows upcoming fixtures, kickoff times (in Eastern Time), and simple explanations of the model's predictions so fans can quickly understand which team is favored. Additionally, it provides detailed referee statistics, manager performance metrics, and team form analysis to help fans understand how different referees, managers, and recent performance might influence match outcomes.

**Latest Enhancement:** The prediction model now uses an ensemble approach combining multiple machine learning algorithms, resulting in **3.5% higher accuracy** compared to the previous XGBoost-only model. Neural network and LSTM time series support have also been added for advanced deep learning predictions.

[Back to top](#premier-league-predictor)

--

## What's already implemented

- A data pipeline that combines historical match CSVs into a processed dataset.
- A Streamlit app (`premier-league-predictions.py`) that:
	- Displays historical match data and model metrics
	- **NEW: Trains an ensemble model** combining XGBoost, Random Forest, Gradient Boosting, and Logistic Regression for improved accuracy (+3.5% vs XGBoost alone)
	- **NEW: Neural network support** using PyTorch for deep learning predictions (+4.9% vs XGBoost baseline)
	- **NEW: LSTM time series model** for capturing team momentum and temporal patterns in performance
	- Shows upcoming fixtures and predicted probabilities with risk assessment
	- Displays kickoff times converted to Eastern Time (ET)
	- **NEW: Statistics tab** with referee performance metrics, manager statistics, team form analysis, and league-wide averages
- **NEW: Poisson regression diagnostics** include a live display of goal‑prediction MAE/RMSE and a historical chart of those metrics over time (visible when selecting Poisson model)
	- **NEW: Model comparison** showing performance improvements between baseline XGBoost, ensemble, neural network, and LSTM time series models
- An ESPN-based fixture fetcher (`fetch_upcoming_fixtures.py`) that pulls upcoming matches from ESPN's API and saves them to `data_files/upcoming_fixtures.csv`.
- Referee data integration: Scrapes referee assignments from Playmaker Stats and calculates historical referee statistics (disciplinary tendencies, win rates, home advantage bias).
- Team form tracking: Analyzes recent performance for all Premier League teams with visual indicators.
- Several helper scripts and data files in `data_files/` such as `combined_historical_data_with_calculations.csv` and `all_teams.csv`.

If you'd like a quick view, open the Streamlit app and check the "Show Upcoming Matches", "Show Upcoming Predictions", and "Statistics" sections.

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

- The Streamlit UI has tabs for: Upcoming Matches, Predictive Data, Upcoming Predictions, Statistics, and Raw Data.
- The Statistics tab displays referee performance metrics, manager statistics, team form analysis, and league-wide averages.
- **NEW: Ensemble model** combines XGBoost, Random Forest, Gradient Boosting, and Logistic Regression using soft voting for improved accuracy.
- **NEW: Neural network support** using PyTorch with 3-layer architecture, batch normalization, and dropout regularization.
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
