# Premier League Predictor — 6-Month Feature Roadmap

## Month 1: Match Day

- **Weekend fixture view** — All EPL fixtures this round with H/D/A probabilities and referee assignment.
- **Referee profile badge** — Click-through to referee stats on each fixture card.
- **Live score banner** — Auto-refreshing in-play scores via ESPN API.
- **Form badges** — "Won last 4" / "Unbeaten in 8" displayed on team cards.

## Month 2: Team Pages

- **Team profile** — Season stats, xG trend, referee record, top scorer, squad rotation flag.
- **Big Six rivalry tracker** — Head-to-head history among Man City, Arsenal, Liverpool, Chelsea, Tottenham, Man Utd.
- **European competition flag** — Visual indicator when a club has midweek Champions/Europa League.

## Month 3: Betting Tools

- **Value bet finder** — Edge > 3% vs. B365 odds; sortable table.
- **Draw specialist dashboard** — Fixtures with both teams' draw rate >30%.
- **CLV tracker** — Opening vs. closing B365 lines for this week's games.

## Month 4: Analytics

- **Model accuracy report** — Per-team, per-outcome accuracy this season vs. last.
- **Referee influence analysis** — Compare home win rate when high-carding vs. low-carding referees officiate.
- **xG vs. actual goals** — Over- and under-performing teams this season.

## Month 5: Reports

- **Matchday PDF export** — One-click PDF of all weekend predictions with odds and referee info.
- **Season halfway report** — Automated summary at GW19: model accuracy, top over/underperformers.

## Month 6: Automation

- **Friday email** — Weekend fixture predictions email with referee notes.
- **Nightly Actions** — `fetch_upcoming_fixtures.py`, `scrape_referees.py`, `prepare_model_data.py`.
- **Model retraining trigger** — Monthly trigger using current-season data.
