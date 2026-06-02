# Premier League Predictor — Next 5 Features to Implement

> **Based on:** Codebase gap analysis as of July 2025

---

## Feature 1: Expanded Referee Impact Dashboard

**Why:** `scrape_referees.py` already collects referee assignments and disciplinary data, and `analyze_referee_impact.py` exists. However, referee statistics are only displayed as a sidebar table — there is no dedicated page for deep analysis. Referee selection is one of the few match factors that bookmakers consistently misprice.

**How:**
1. Add `pages/referee_impact.py` as a dedicated multi-chart page
2. Include: per-referee home-win%, yellow/red cards per game, penalty award rate, fouls called per game
3. Add a fixture overlay: for each upcoming match, show the assigned referee's historical stats
4. Use `scrape_referees.py` data + the combined historical data CSV to build the full stats table

**Complexity:** Low

---

## Feature 2: xG-Based Expected Goals Features

**Why:** The current model uses only betting odds, shots, goals, and form features. FBref provides EPL xGF and xGA (comp ID `9`) — adding xG rolling averages would add predictive signal independent of betting-market pricing.

**How:**
1. Create `fetch_fbref_xg.py` (adapt from sister repos) for EPL (FBref comp ID `9`)
2. Add `home_xg_l5`, `home_xga_l5`, `away_xg_l5`, `away_xga_l5` to the feature set in `prepare_model_data.py`
3. Use `groupby("Team").shift(1).rolling(5).mean()` to prevent leakage
4. Re-run `combine_raw_data.py` + `prepare_model_data.py` to regenerate the processed CSV

**Complexity:** Medium

---

## Feature 3: Live Odds Integration with Edge Calculation

**Why:** The model generates win probabilities but currently shows no comparison against market implied probabilities. Adding The Odds API (`soccer_england_premier_league`) would let users see model edge per fixture — the most actionable output for bettors.

**How:**
1. Add `fetch_odds.py` using The Odds API key (reuse the pattern from sister repos)
2. Merge odds into `upcoming_fixtures.csv` (convert American/decimal to implied probability)
3. Add an "Edge" column to the predictions table: `model_prob - implied_prob`
4. Color-code edge: green > 3%, yellow 1–3%, grey < 1%

**Complexity:** Medium

---

## Feature 4: Multi-Season Model Comparison

**Why:** The model currently trains on all data in one pass and reports a single accuracy number. Season-by-season AUC and Brier score would reveal drift, helping identify when the model needs recalibration (e.g., when a new dominant team emerges or tactics shift).

**How:**
1. In `prepare_model_data.py`, add a walk-forward cross-validation loop keyed by `Season` column
2. For each season, train on all prior seasons, predict on the held-out season, compute AUC and Brier score
3. Add a `pages/model_comparison.py` Streamlit page with a per-season bar chart

**Complexity:** Low

---

## Feature 5: VAR Decision Tracker

**Why:** VAR was introduced in EPL from the 2019/20 season. Teams and referees have measurably different VAR decision rates. This is a novel signal that is not in any public EPL dataset but can be scraped from BBC Sport or premierleague.com VAR decision logs.

**How:**
1. Create `scrape_var_decisions.py` using BeautifulSoup to extract VAR decisions from a public source
2. Encode per-team and per-referee VAR incident rates as features
3. Initially: surface as a contextual display only (not a model feature) to validate data quality before training
4. Add to the Referee Impact Dashboard as a new tab

**Complexity:** High
