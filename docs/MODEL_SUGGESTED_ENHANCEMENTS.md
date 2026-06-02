# Premier League Predictor — Model Suggested Enhancements

## Priority 1: Ensemble Improvements

### XGBoost Hyperparameter Tuning
- Use `Optuna` for Bayesian optimisation on XGBoost. Current params are defaults; properly tuned XGBoost alone often matches a naive ensemble.

### Add LightGBM to Ensemble
- LightGBM handles sparse categorical features better than XGBoost alone. Add as a voter with weight 1.5.

### Calibration
- Apply `CalibratedClassifierCV` with isotonic regression over the ensemble. EPL models tend to underestimate draw probability.

## Priority 2: Feature Expansion

### Expected Goals (xG)
- FBref EPL xG is free and scrapable. Add `xg_l5_home`, `xga_l5_away`, `xg_differential_l10` to `FEATURE_COLS`.

### Referee Aggression Depth
- Current `scrape_referees.py` captures card rates. Extend to: `ref_fouls_pg`, `ref_home_advantage_bias` (does the referee award more fouls at home?), `ref_pen_rate`.

### Defensive Pressure Metric
- Add `ppda_l5` (passes allowed per defensive action) as a proxy for pressing intensity. Available from FBref.

### Squad Rotation Flag
- EPL clubs rotate heavily for League Cup / Champions League weeks. Add `cup_fixture_within_3_days` flag.

## Priority 3: Betting Intelligence

### Early Season Caution Flag
- First 5 games of the season have high variance. Apply a confidence discount for early-season predictions.

### Promoted Club Downgrade
- Newly promoted clubs from the Championship have a documented adjustment period. Add `seasons_in_epl` (capped at 5) as a feature.

## Priority 4: Infrastructure

- Auto-delete `models/` on new-season start to prevent training on stale pre-promotion data.
- Add model version tag (training data cutoff) displayed in the Streamlit header.
