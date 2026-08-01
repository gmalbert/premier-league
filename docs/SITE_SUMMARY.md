> **AI Onboarding Guide** — See also `.github/copilot-instructions.md` for full coding conventions.

# Premier League Predictor — Site Summary

## What This App Does

Streamlit app that predicts English Premier League match outcomes using an XGBoost classifier trained on historical match data from football-data.co.uk. Includes team form metrics, head-to-head history, rest days, referee statistics, and betting odds as features. A referee dashboard surfaces card rates and home-win bias per official.

## Quick Start

```bash
# 1. Activate virtual environment
.\.venv\Scripts\Activate.ps1        # Windows
source .venv/bin/activate           # macOS/Linux

# 2. Update and process data
python combine_raw_data.py          # Download yearly E0.csv files and concatenate
python prepare_model_data.py        # Feature engineering → combined_historical_data_with_calculations.csv

# 3. (Optional) Fetch upcoming fixtures and referee assignments
python fetch_upcoming_fixtures.py   # ESPN API → upcoming_fixtures.csv
python scrape_referees.py           # Playmaker Stats → scraped_referees_test.csv

# 4. Run the app
streamlit run predictions.py
```

## Tech Stack

| Layer | Technology |
|---|---|
| UI | Streamlit (single-file app + sidebar) |
| ML | XGBoost classifier (trained in-app on load) |
| Data | pandas, NumPy |
| Visualization | Plotly (permutation feature importance, referee stats) |
| Scraping | requests, BeautifulSoup4 |
| Config | No API keys required for primary data |

## Key Files

| File | Purpose |
|---|---|
| `predictions.py` | Main Streamlit entry page using the shared-core app factory |
| `combine_raw_data.py` | Downloads and concatenates yearly `E0.csv` from football-data.co.uk |
| `prepare_model_data.py` | Feature engineering: column renaming, rolling form, H2H stats, rest days, referee stats |
| `fetch_upcoming_fixtures.py` | Fetches next 30 days of EPL fixtures from ESPN API |
| `scrape_referees.py` | Scrapes referee assignments from Playmaker Stats |
| `team_name_mapping.py` | Normalizes team names across ESPN / football-data.co.uk / Playmaker Stats |
| `data_files/combined_historical_data_with_calculations.csv` | Processed feature matrix — primary model input |

## Data Flow

1. **Raw data**: `combine_raw_data.py` downloads `E0.csv` per season (football-data.co.uk) → `combined_historical_data.csv`
2. **Feature engineering**: `prepare_model_data.py` adds team aggregates, rolling form (last 5), H2H history, rest days, referee stats → `combined_historical_data_with_calculations.csv`
3. **Upcoming fixtures**: `fetch_upcoming_fixtures.py` (ESPN) → `upcoming_fixtures.csv`; referee assignments: `scrape_referees.py` → `scraped_referees_test.csv`
4. **Model training**: Streamlit app trains XGBoost on load using processed CSV; displays metrics and feature importance
5. **Predictions**: Upcoming fixtures merged with referee data → XGBoost predictions displayed in UI

## Environment Variables

No API keys are required for primary data. The ESPN upcoming fixtures endpoint is public.

## External APIs & Rate Limits

| API | Notes |
|---|---|
| football-data.co.uk | Static CSV download, no key needed; file is `E0.csv` per season |
| ESPN (site.api.espn.com) | Public JSON endpoint for EPL schedule; no key required |
| Playmaker Stats | Referee assignment scraping — rate-limit cautiously |

## Critical Conventions

- Model trains **on every app load** — the training step is inside the Streamlit script, not pre-cached
- Column naming: descriptive (e.g., `Bet365_HomeWinOdds`), consistent across all scripts
- Date handling: `pd.to_datetime(..., dayfirst=True)` for UK date format
- Drop leaky columns before training: anything containing actual goals, results, or final odds
- Target encoding: H=0, D=1, A=2 (note: different from sibling soccer apps which use A=0, D=1, H=2)
- Clean column names for XGBoost: remove special characters, replace spaces with `_`
- All expensive computations should be wrapped in `@st.cache_data`

## Common Gotchas

- `prepare_model_data.py` must be re-run after `combine_raw_data.py` to regenerate the feature-engineered CSV
- Referee name normalization uses `team_name_mapping.py` — if a referee name doesn't match, the merge silently drops the assignment
- Rolling form calculations: always sort by date before `groupby().rolling()` or results will be wrong
- `fetch_upcoming_fixtures.py` only fetches next 30 days — extend the window for mid-season use
