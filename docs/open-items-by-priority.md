# Open Roadmap Items (Prioritized)

This file was generated after reviewing all files in `docs/`.

## ✅ Completed items (as of current repo state)

From `roadmap-data.md`:
- Player Statistics & Injuries (completed)
- Weather Data (completed)
- Referee Statistics (completed)
- Advanced Team Metrics (completed)
- Betting Market Data (completed)
- Referee Assignments for Upcoming Matches (completed)
- Manager & Tactical Data (completed)
- Missing Data Handling / KNN Imputation (completed)

From `roadmap-models.md`:
- Ensemble Model Approach (completed)
- Neural Network with PyTorch (completed)
- Poisson Regression (completed)
- Time Series LSTM for Momentum (completed)
- Hyperparameter Optimization (completed)

From `roadmap-quick-wins.md`:
- Last Update Timestamp (completed)

From `roadmap-infrastructure.md` (2026-03-28 session):
- Automated Data Pipeline — `automation/scheduler.py` (nightly at 3 AM, hourly fixtures)
- Model Versioning with MLflow — `models/train_with_mlflow.py` (experiment tracking, nested runs)
- Caching Layer — `cache/redis_cache.py` (Redis + in-memory fallback, `cache_result()` decorator)

## ⚡ Open items (categorized by priority)

### High priority

- `roadmap-data-next-steps.md`: Enhanced Historical Odds Data
- `roadmap-features-next-steps.md`: Live Match Tracker with Auto-Refresh
- `roadmap-features-next-steps.md`: Betting Odds Comparison Tool
- `roadmap-models-next-steps.md`: Comprehensive Model Comparison Dashboard
- `roadmap-infrastructure-next-steps.md`: SQLite Database Migration (phase 1)
- `roadmap-quick-wins.md`: Show Top Features Chart

### Medium priority

- `roadmap-data-next-steps.md`: Social Media Sentiment Analysis
- `roadmap-data-next-steps.md`: Transfer Market Activity
- `roadmap-data-next-steps.md`: Venue-Specific Data
- `roadmap-features-next-steps.md`: Interactive Match Commentary Generator
- `roadmap-features-next-steps.md`: Export Predictions to PDF Report
- `roadmap-infrastructure-next-steps.md`: API Development for predictions
- `roadmap-models-next-steps.md`: Gradient Boosting Variants (LightGBM, CatBoost)
- `roadmap-models-next-steps.md`: Confidence Calibration (Platt Scaling)
- `roadmap-quick-wins.md`: Add Match Commentary display
- `roadmap-quick-wins.md`: Color-Code Confidence Levels
- `roadmap-quick-wins.md`: Sorting in data tables
- `roadmap-quick-wins.md`: Export to CSV
- `roadmap-quick-wins.md`: Filter by Date Range

### Low priority

- `roadmap-data-next-steps.md`: Match Scheduling Data (rest days, fixture density, Europe involvement)
- `roadmap-infrastructure-next-steps.md`: Email Alerts for high-confidence predictions
- `roadmap-quick-wins.md`: (already mostly small tasks; remaining optional)

## 📌 Notes

- Many major improvements are already implemented in code; open tasks are largely feature extensions, presentation enhancements, and infrastructure maturity.
- Use this file as a single source for remaining action items and for staging immediate development sprints.
