# Premier League Predictor — Architecture

## Overview
Streamlit app predicting English Premier League match outcomes using XGBoost classification on historical match data, team form metrics, referee statistics, and betting odds features.

## Data Flow
```
football-data.co.uk (E0.csv per season)
        ↓
combine_raw_data.py → data_files/combined_historical_data.csv
        ↓
prepare_model_data.py → Feature Engineering
    [team aggregates, rolling form (L5), H2H, rest days, ref stats, odds]
        ↓
data_files/combined_historical_data_with_calculations.csv
        ↓
scrape_referees.py → data_files/scraped_referees_test.csv (Playmaker Stats)
fetch_upcoming_fixtures.py → data_files/upcoming_fixtures.csv (ESPN API)
        ↓
predictions.py (Streamlit entry)
    → XGBoost H/D/A classifier
    → Model metrics, permutation feature importance, referee stats
```

## ML Model
- **Algorithm**: XGBoost multi-class classifier (H/D/A)
- **Target encoding**: H=0, D=1, A=2
- **Features**: Team aggregates, rolling last-5 points, H2H history, rest days, referee history, B365 odds implied probs
- **No leakage**: Drop result columns, goals; use only pre-game data
- Feature names: sanitised (spaces → `_`, special chars removed) for XGBoost

## API Integrations
| Source | Purpose | Key |
|--------|---------|-----|
| football-data.co.uk | E0.csv per season | None (public download) |
| ESPN `site.api.espn.com` | Next 30 days fixtures | None (public) |
| Playmaker Stats | Referee assignments | None (scraped, BeautifulSoup) |

## Key Components
- `combine_raw_data.py` — downloads + concatenates yearly E0.csv files
- `prepare_model_data.py` — column renaming (`FTHG` → `FullTimeHomeGoals`), team stats, rolling windows, referee stats
- `fetch_upcoming_fixtures.py` — ESPN API for next 30 days of PL matches
- `scrape_referees.py` — scrapes referee assignments from Playmaker Stats
- `predictions.py` — Streamlit entry + shared-core app factory
- `team_name_mapping.py` — normalises team names across data sources

## Feature Engineering Patterns
- **Team aggregates**: `groupby('Team').agg({'col': 'mean'})`
- **Rolling form**: `groupby('Team')['Points'].rolling(window=5).sum()`
- **H2H history**: filter same-team matches, sort by date, last N
- **Rest days**: `shift(1)` date differences
- **Betting features**: odds → implied probabilities, market margins

## Column Naming Convention
- Descriptive names: e.g. `Bet365_HomeWinOdds`, `FullTimeHomeGoals`
- Date: `pd.to_datetime(..., dayfirst=True)` (UK format)
- Team IDs: lowercase, underscores, no apostrophes

## Storage
- `data_files/` — tab-separated CSVs (all CSVs use `sep='\t'`)
- `models/` — XGBoost model artifacts (if persisted)

## Development Workflow
```powershell
python combine_raw_data.py        # refresh historical data
python prepare_model_data.py      # regenerate processed CSVs
python fetch_upcoming_fixtures.py # get next 30 days of fixtures
python scrape_referees.py         # get referee assignments
streamlit run predictions.py
```
