# Premier League Predictions — 12-Month Feature Roadmap

> Generated: 2026-07-31 | Horizon: August 2026 – July 2027

---

## Executive Summary

This roadmap evolves the Premier League Predictions platform from a VotingClassifier
ensemble with static CSV data into a live-odds intelligence platform with deep
tactical analytics, automated pipeline, and API-first architecture.

---

## Q1 (Aug–Oct 2026) — Data Foundation & Live Intelligence

### Feature 1 — FBref Advanced Stats Full Integration

Ingest FBref xG, xGA, progressive carries, pressures, and PPDA per team per
season. Replace basic goal counts with shot-quality metrics.

```python
# fetch_fbref_advanced.py
import requests, pandas as pd
from bs4 import BeautifulSoup
from io import StringIO
import time

FBREF_BASE = "https://fbref.com/en/comps/9"

def fetch_fbref_team_stats(season: str = "2025-2026") -> pd.DataFrame:
    """Scrape EPL team shooting stats from FBref."""
    url = f"{FBREF_BASE}/{season}/shooting/Premier-League-Stats"
    resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
    soup = BeautifulSoup(resp.text, "lxml")
    table = soup.find("table", {"id": "stats_squads_shooting_for"})
    df = pd.read_html(StringIO(str(table)), header=1)[0]
    df.columns = ["_".join(c).strip("_") for c in df.columns]
    df = df[df["Squad"].notna() & (df["Squad"] != "Squad")]
    df = df.rename(columns={
        "Squad": "team", "Gls": "goals", "Sh": "shots",
        "SoT": "shots_on_target", "xG": "xg", "npxG": "npxg",
    })
    time.sleep(3)  # Rate limiting
    return df[["team", "goals", "shots", "shots_on_target", "xg", "npxg"]].reset_index(drop=True)

def build_rolling_xg_features(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    df = df.sort_values(["team", "MatchDate"])
    for col in ["xg", "npxg", "xga"]:
        if col in df.columns:
            df[f"{col}_r{window}"] = (
                df.groupby("team")[col]
                .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
            )
    return df
```

### Feature 2 — Live Odds Feed via The Odds API

Fetch live EPL moneyline, Asian handicap, and BTTS odds every hour.
Display consensus lines and detect sharp moves.

```python
# fetch_live_odds.py
import requests, os, json, pandas as pd
from datetime import datetime
from pathlib import Path

SPORT_KEY = "soccer_epl"
MARKETS = "h2h,spreads,btts"
SNAP_DIR = Path("data_files/raw/odds_snapshots")
SNAP_DIR.mkdir(parents=True, exist_ok=True)

def fetch_epl_odds() -> pd.DataFrame:
    resp = requests.get(
        f"https://api.the-odds-api.com/v4/sports/{SPORT_KEY}/odds/",
        params={
            "apiKey": os.environ["ODDS_API_KEY"],
            "regions": "uk,eu", "markets": MARKETS,
            "oddsFormat": "decimal",
        },
        timeout=15,
    )
    data = resp.json()
    snap_file = SNAP_DIR / f"snapshot_{datetime.utcnow().strftime('%Y%m%d_%H%M')}.json"
    snap_file.write_text(json.dumps(data))
    rows = []
    for game in data:
        for book in game.get("bookmakers", []):
            for mkt in book.get("markets", []):
                for outcome in mkt["outcomes"]:
                    rows.append({
                        "game_id": game["id"], "commence_time": game["commence_time"],
                        "home_team": game["home_team"], "away_team": game["away_team"],
                        "bookmaker": book["key"], "market": mkt["key"],
                        "outcome": outcome["name"], "price": outcome["price"],
                    })
    return pd.DataFrame(rows)

def get_best_odds(df: pd.DataFrame, market: str, outcome: str) -> pd.Series:
    return df[(df["market"] == market) & (df["outcome"] == outcome)].sort_values(
        "price", ascending=False
    ).iloc[0]
```

### Feature 3 — Manager Tactical Style Classifier

Classify each manager's system (high press, possession, counter-attack, direct)
using PPDA, deep completions, and touch % in opponent half. Encode as model feature.

```python
# features/manager_style.py
import pandas as pd
from sklearn.cluster import KMeans

TACTICAL_FEATURES = [
    "ppda_ratio",         # passes per defensive action (lower = more pressing)
    "avg_possession_pct", # ball possession %
    "progressive_carries_per_game",
    "final_third_touch_pct",
    "avg_pass_length",
    "counter_attack_rate",
]

def classify_manager_style(team_stats: pd.DataFrame) -> pd.DataFrame:
    """K-Means cluster teams into 4 tactical styles."""
    X = team_stats[TACTICAL_FEATURES].fillna(team_stats[TACTICAL_FEATURES].mean())
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    team_stats["tactical_cluster"] = kmeans.fit_predict(X)
    cluster_labels = {
        0: "High Press", 1: "Possession", 2: "Counter", 3: "Direct",
    }
    team_stats["tactical_style"] = team_stats["tactical_cluster"].map(cluster_labels)
    return team_stats
```

### Feature 4 — Big-6 Head-to-Head Intelligence

Special handling for matches between top-6 clubs (Man City, Liverpool,
Arsenal, Chelsea, Man Utd, Spurs). These games have significantly different
prediction dynamics.

```python
# features/big6_features.py
import pandas as pd

BIG_6 = {"Manchester City", "Liverpool", "Arsenal",
          "Chelsea", "Manchester United", "Tottenham"}

def add_big6_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["is_big6_home"] = df["HomeTeam"].isin(BIG_6).astype(int)
    df["is_big6_away"] = df["AwayTeam"].isin(BIG_6).astype(int)
    df["big6_clash"] = (df["is_big6_home"] & df["is_big6_away"]).astype(int)
    df["big6_vs_bottom"] = (
        (df["is_big6_home"] | df["is_big6_away"]) &
        ~df["big6_clash"]
    ).astype(int)

    # Big-6 H2H results deviate from league expectations
    # Shift model probs toward draw in big-6 clashes
    df["big6_draw_boost"] = df["big6_clash"] * 0.08
    return df
```

### Feature 5 — Player Availability Impact Score

Ingest injury/suspension list from football-data.org. Quantify team strength
impact when key players are unavailable (based on FBref performance shares).

```python
# features/availability.py
import requests, pandas as pd

FOOTBALL_DATA_BASE = "https://api.football-data.org/v4"

def fetch_injured_players(team_id: int, season: str = "2025") -> list[str]:
    resp = requests.get(
        f"{FOOTBALL_DATA_BASE}/teams/{team_id}",
        headers={"X-Auth-Token": os.environ["FOOTBALL_DATA_KEY"]},
        timeout=10,
    )
    team = resp.json()
    return [p["name"] for p in team.get("squad", [])
            if p.get("status", "") == "injured"]

def team_availability_score(team: str, injured: list[str],
                             player_war: pd.DataFrame) -> float:
    """Return fraction of expected WAR available (0-1)."""
    team_war = player_war[player_war["team"] == team]
    total_war = team_war["war"].sum()
    if total_war == 0:
        return 1.0
    injured_war = team_war[team_war["player"].isin(injured)]["war"].sum()
    return max(0.0, 1.0 - injured_war / total_war)
```

---

## Q2 (Nov 2026 – Jan 2027) — Model Depth & BTTS/Both Teams to Score

### Feature 6 — Both Teams to Score (BTTS) Model

Train a dedicated binary classifier for BTTS. Top features: team scoring
frequency, opponent xGA, H2H BTTS history, referee, match importance.

```python
# model_btts.py
import pandas as pd, numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import brier_score_loss
import joblib

BTTS_FEATURES = [
    "home_xg_r5", "away_xg_r5",
    "home_xga_r5", "away_xga_r5",
    "home_btts_rate_l10", "away_btts_rate_l10",
    "h2h_btts_rate", "referee_avg_goals",
    "home_clean_sheet_rate_l10", "away_clean_sheet_rate_l10",
    "home_scoring_frequency", "away_scoring_frequency",
    "match_importance_score",
]

def train_btts_model(df: pd.DataFrame) -> None:
    X = df[BTTS_FEATURES].fillna(df[BTTS_FEATURES].median())
    y = (df["FTHG"] > 0) & (df["FTAG"] > 0)
    y = y.astype(int)

    model = XGBClassifier(n_estimators=600, max_depth=4, learning_rate=0.03,
                          subsample=0.8, colsample_bytree=0.7,
                          scale_pos_weight=(~y.astype(bool)).sum() / y.sum())
    tscv = TimeSeriesSplit(n_splits=5)
    briers = []
    for tr, val in tscv.split(X):
        model.fit(X.iloc[tr], y.iloc[tr])
        preds = model.predict_proba(X.iloc[val])[:, 1]
        briers.append(brier_score_loss(y.iloc[val], preds))
    print(f"BTTS Brier Score: {np.mean(briers):.4f}")
    model.fit(X, y)
    joblib.dump(model, "models/btts_model.pkl")
```

### Feature 7 — Correct Score Model (Bivariate Poisson)

Train a bivariate Poisson regression for exact correct score probabilities.
Display top-5 most likely scores per match.

```python
# model_correct_score.py
import numpy as np, pandas as pd
from scipy.stats import poisson
from scipy.optimize import minimize
import joblib

def fit_goal_model(df: pd.DataFrame) -> dict:
    """Fit Dixon-Coles-style bivariate Poisson for EPL correct score."""
    teams = sorted(pd.concat([df["HomeTeam"], df["AwayTeam"]]).unique())
    n = len(teams)
    team_idx = {t: i for i, t in enumerate(teams)}

    def neg_ll(params):
        attack = params[:n]
        defense = params[n:2*n]
        home_adv = params[2*n]
        rho = params[2*n + 1]
        ll = 0
        for _, row in df.iterrows():
            i, j = team_idx[row["HomeTeam"]], team_idx[row["AwayTeam"]]
            lam = np.exp(attack[i] + defense[j] + home_adv)
            mu = np.exp(attack[j] + defense[i])
            hg, ag = int(row["FTHG"]), int(row["FTAG"])
            t_val = (1 - lam * mu * rho if hg == 0 and ag == 0 else
                     1 + lam * rho if hg == 0 and ag == 1 else
                     1 + mu * rho if hg == 1 and ag == 0 else
                     1 - rho if hg == 1 and ag == 1 else 1)
            ll += np.log(max(t_val, 1e-10)) + poisson.logpmf(hg, lam) + poisson.logpmf(ag, mu)
        return -ll

    x0 = np.zeros(2*n + 2)
    x0[2*n] = 0.3  # home advantage
    x0[2*n+1] = -0.05
    res = minimize(neg_ll, x0, method="L-BFGS-B", options={"maxiter": 500})
    return {"attack": dict(zip(teams, res.x[:n])),
            "defense": dict(zip(teams, res.x[n:2*n])),
            "home_adv": res.x[2*n], "rho": res.x[2*n+1], "teams": teams}

def predict_correct_scores(home: str, away: str, model: dict, top_n: int = 5) -> list[dict]:
    i = model["teams"].index(home)
    j = model["teams"].index(away)
    lam = np.exp(model["attack"][home] + model["defense"][away] + model["home_adv"])
    mu = np.exp(model["attack"][away] + model["defense"][home])
    scores = []
    for h in range(8):
        for a in range(8):
            p = poisson.pmf(h, lam) * poisson.pmf(a, mu)
            scores.append({"score": f"{h}-{a}", "prob": round(p, 4)})
    return sorted(scores, key=lambda x: -x["prob"])[:top_n]
```

### Feature 8 — Asian Handicap Calculator

Convert model win/draw/lose probabilities to AH prices. Compare against
book's AH lines to find value in Asian market.

```python
# utils/asian_handicap.py
import numpy as np

def ah_probability(home_win_prob: float, draw_prob: float,
                   away_win_prob: float, handicap: float) -> dict:
    """
    Convert 1X2 probabilities to Asian Handicap probabilities.
    handicap: negative = home favorite (e.g., -0.5, -1.0, -1.5)
    """
    if handicap == -0.5:
        ah_home = home_win_prob
        ah_away = draw_prob + away_win_prob
    elif handicap == 0.5:
        ah_home = home_win_prob + draw_prob
        ah_away = away_win_prob
    elif handicap == -1.0:
        ah_home = home_win_prob - draw_prob / 2  # split stake on draw
        ah_away = away_win_prob + draw_prob / 2
    elif handicap == -1.5:
        ah_home = max(0, home_win_prob - draw_prob - away_win_prob * 0.5)
        ah_away = 1 - ah_home
    else:
        # Generic linear interpolation
        goals_diff_expected = _model_expected_goal_diff(home_win_prob, away_win_prob)
        ah_home = _poisson_cover_prob(goals_diff_expected, handicap)
        ah_away = 1 - ah_home
    return {"ah_home_cover": round(ah_home, 4), "ah_away_cover": round(ah_away, 4)}

def _model_expected_goal_diff(hwp: float, awp: float) -> float:
    return (hwp - awp) * 2.5  # rough calibration

def _poisson_cover_prob(exp_diff: float, handicap: float) -> float:
    from scipy.stats import norm
    return float(norm.cdf(-handicap, loc=exp_diff, scale=1.5))
```

### Feature 9 — Match Importance Index

Quantify how much each match matters for title race, top-4, or relegation.
High-importance games have different prediction dynamics.

```python
# features/match_importance.py
import pandas as pd, numpy as np

def compute_match_importance(schedule: pd.DataFrame, standings: pd.DataFrame) -> pd.DataFrame:
    """Score each fixture by its importance to season outcomes."""
    df = schedule.merge(
        standings[["team", "position", "pts", "pts_from_top4",
                   "pts_from_relegation", "pts_from_title"]],
        left_on="HomeTeam", right_on="team",
    )
    df["home_importance"] = df.apply(_importance_score, axis=1)
    return df

def _importance_score(row) -> float:
    score = 0.0
    # Title race
    if row.get("pts_from_title", 999) <= 6:
        score += 3.0
    # Top 4 race
    elif row.get("pts_from_top4", 999) <= 3:
        score += 2.0
    # Relegation battle
    elif row.get("pts_from_relegation", 999) <= 3:
        score += 2.0
    # Derby factor
    if row.get("is_derby", 0):
        score += 1.0
    return score / 6.0  # normalize 0-1
```

### Feature 10 — Half-Time / Full-Time Double Result Model

Train a classifier for HT/FT double result (6 outcomes: H/H, H/D, H/A,
D/H, D/D, D/A, A/H, A/D, A/A). Expose value in high-odds HTFT markets.

```python
# model_htft.py
import pandas as pd, numpy as np
from sklearn.ensemble import GradientBoostingClassifier
import joblib

HTFT_LABELS = ["H/H", "H/D", "H/A", "D/H", "D/D", "D/A", "A/H", "A/D", "A/A"]

def encode_htft(row) -> str:
    ht = "H" if row["HTR"] == "H" else ("D" if row["HTR"] == "D" else "A")
    ft = "H" if row["FTR"] == "H" else ("D" if row["FTR"] == "D" else "A")
    return f"{ht}/{ft}"

def train_htft_model(df: pd.DataFrame) -> None:
    df = df.copy()
    df["htft_label"] = df.apply(encode_htft, axis=1)
    df["htft_code"] = df["htft_label"].map({l: i for i, l in enumerate(HTFT_LABELS)})
    features = ["home_win_prob", "draw_prob", "away_win_prob",
                "home_xg_r5", "away_xg_r5", "home_scored_first_rate",
                "away_scored_first_rate"]
    X = df[features].fillna(0)
    y = df["htft_code"]
    model = GradientBoostingClassifier(n_estimators=400, max_depth=3)
    model.fit(X, y)
    joblib.dump(model, "models/htft_model.pkl")
```

---

## Q3 (Feb–Apr 2027) — Visual Analytics & Dashboard

### Feature 11 — Expected Points Table

Show expected points (xPts) based on xG-derived win probabilities alongside
actual points. Highlight over/under-performing teams.

```python
# pages/xpts_table.py
import streamlit as st, pandas as pd, plotly.express as px
from scipy.stats import poisson

def compute_xpts(df: pd.DataFrame) -> pd.DataFrame:
    """Compute expected points from xG data for each match."""
    results = []
    for _, row in df.iterrows():
        h_xg, a_xg = row.get("home_xg", 1.5), row.get("away_xg", 1.2)
        p_hw, p_draw, p_aw = _xg_to_probs(h_xg, a_xg)
        results.append({
            "home_team": row["HomeTeam"], "away_team": row["AwayTeam"],
            "home_xpts": p_hw * 3 + p_draw * 1,
            "away_xpts": p_aw * 3 + p_draw * 1,
        })
    return pd.DataFrame(results)

def _xg_to_probs(lam: float, mu: float, max_g: int = 8):
    pm = [[poisson.pmf(i, lam) * poisson.pmf(j, mu) for j in range(max_g)]
          for i in range(max_g)]
    hw = sum(pm[i][j] for i in range(max_g) for j in range(max_g) if i > j)
    draw = sum(pm[i][i] for i in range(max_g))
    aw = 1 - hw - draw
    return hw, draw, aw

def render_xpts_table(games: pd.DataFrame) -> None:
    xpts = compute_xpts(games)
    teams = pd.concat([
        xpts[["home_team", "home_xpts"]].rename(columns={"home_team": "team", "home_xpts": "xpts"}),
        xpts[["away_team", "away_xpts"]].rename(columns={"away_team": "team", "away_xpts": "xpts"}),
    ])
    table = teams.groupby("team")["xpts"].sum().reset_index().sort_values("xpts", ascending=False)
    table["rank"] = range(1, len(table) + 1)
    st.title("xPts Table — Underlying Performance")
    st.dataframe(table, width="stretch")
```

### Feature 12 — Form Guide with ELO Heatmap

Rolling Elo ratings visualized as a heatmap over time. Color each team-week
cell by Elo rating to show rise/fall patterns visually.

```python
# pages/elo_heatmap.py
import streamlit as st, pandas as pd, plotly.express as px

INITIAL_ELO = 1500
K_FACTOR = 30

def build_elo_history(games: pd.DataFrame) -> pd.DataFrame:
    elo = {}
    records = []
    games = games.sort_values("MatchDate")
    for _, row in games.iterrows():
        h, a = row["HomeTeam"], row["AwayTeam"]
        elo_h = elo.get(h, INITIAL_ELO)
        elo_a = elo.get(a, INITIAL_ELO)
        exp_h = 1 / (1 + 10 ** ((elo_a - elo_h) / 400))
        result = row.get("FTR", "D")
        actual = 1.0 if result == "H" else (0.5 if result == "D" else 0.0)
        elo[h] = elo_h + K_FACTOR * (actual - exp_h)
        elo[a] = elo_a + K_FACTOR * ((1 - actual) - (1 - exp_h))
        records.append({"week": row.get("GW", 0), "HomeTeam": h, "home_elo": elo[h],
                        "AwayTeam": a, "away_elo": elo[a]})
    return pd.DataFrame(records)

def render_elo_heatmap(elo_history: pd.DataFrame) -> None:
    home = elo_history[["week", "HomeTeam", "home_elo"]].rename(
        columns={"HomeTeam": "team", "home_elo": "elo"})
    away = elo_history[["week", "AwayTeam", "away_elo"]].rename(
        columns={"AwayTeam": "team", "away_elo": "elo"})
    df = pd.concat([home, away])
    pivot = df.pivot_table(index="team", columns="week", values="elo", aggfunc="mean")
    fig = px.imshow(pivot, color_continuous_scale="RdYlGn", zmin=1300, zmax=1700,
                    title="Team Elo Rating by Gameweek", template="plotly_dark")
    st.plotly_chart(fig, width="stretch")
```

### Feature 13 — Referee & VAR Impact Tracker

Track penalty rates, red cards, and score reversal frequency per referee.
Show how referee assignment shifts expected outcomes.

```python
# pages/referee_tracker.py
import streamlit as st, pandas as pd

def render_referee_tracker(games: pd.DataFrame) -> None:
    ref_stats = (
        games.groupby("Referee")
        .agg(
            games=("MatchDate", "count"),
            avg_total=("FullTimeHomeGoals", lambda x: (x + games.loc[x.index, "FullTimeAwayGoals"]).mean()),
            avg_pens=("HY", lambda x: (x + games.loc[x.index, "AY"]).mean()),
            home_win_pct=("FullTimeResult", lambda x: (x == "H").mean()),
        )
        .reset_index()
    )
    ref_stats = ref_stats[ref_stats["games"] >= 20].sort_values("avg_total", ascending=False)
    st.title("Referee Impact Analysis")
    st.dataframe(ref_stats, width="stretch")
```

### Feature 14 — Gameweek Preview Cards

Rich preview card for each upcoming fixture: form guide (last 5),
xG comparison, H2H record, key injuries, model pick with confidence.

```python
# pages/gameweek_preview.py
import streamlit as st
from utils import render_table

def render_fixture_card(fixture: dict) -> None:
    with st.container(border=True):
        col1, col2, col3 = st.columns([2, 1, 2])
        with col1:
            st.markdown(f"### {fixture['HomeTeam']}")
            st.caption(f"Form: {fixture.get('home_form', 'N/A')}")
            st.metric("xG (L5)", f"{fixture.get('home_xg_r5', 0):.2f}")
        with col2:
            st.markdown("### VS")
            pick = fixture.get("model_pick", "?")
            color = "🟢" if fixture.get("edge", 0) > 0.03 else "🟡"
            st.markdown(f"**{color} {pick}**")
            st.caption(f"Edge: {fixture.get('edge', 0):+.1%}")
        with col3:
            st.markdown(f"### {fixture['AwayTeam']}")
            st.caption(f"Form: {fixture.get('away_form', 'N/A')}")
            st.metric("xG (L5)", f"{fixture.get('away_xg_r5', 0):.2f}")
```

### Feature 15 — Season-Long Betting Record Tracker

Cumulative P&L by bet type, by result type, and by confidence tier.
Exportable as PDF / CSV.

```python
# pages/betting_record.py
import streamlit as st, pandas as pd, plotly.express as px

def render_betting_record(history: pd.DataFrame) -> None:
    if history.empty:
        st.info("No betting history recorded yet.")
        return
    history["date"] = pd.to_datetime(history["date"])
    history = history.sort_values("date")
    history["cumulative_pnl"] = history["pnl"].cumsum()

    st.title("Season Betting Record")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total P&L", f"{history['pnl'].sum():+.2f} units")
    col2.metric("Win Rate", f"{(history['outcome']=='W').mean():.1%}")
    col3.metric("ROI", f"{history['pnl'].sum() / len(history):+.3f} units/bet")

    fig = px.line(history, x="date", y="cumulative_pnl",
                  title="Cumulative P&L", template="plotly_dark")
    fig.add_hline(y=0, line_dash="dash", line_color="red")
    st.plotly_chart(fig, width="stretch")
```

---

## Q4 (May–Jul 2027) — Automation & Intelligence

### Feature 16 — Automated Fixture Scraper (GitHub Actions)

Fetch upcoming fixtures from football-data.org at 5 AM UTC each Monday.
Populate upcoming_fixtures.csv automatically.

```yaml
# .github/workflows/weekly_fixtures.yml
name: EPL Weekly Fixture Refresh
on:
  schedule:
    - cron: '0 5 * * 1'
  workflow_dispatch:

jobs:
  refresh:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: '3.11' }
      - run: pip install -r requirements.txt
      - name: Fetch upcoming fixtures
        env:
          FOOTBALL_DATA_KEY: ${{ secrets.FOOTBALL_DATA_KEY }}
          ODDS_API_KEY: ${{ secrets.ODDS_API_KEY }}
        run: python fetch_upcoming_fixtures.py
      - name: Update historical data
        run: python combine_raw_data.py && python prepare_model_data.py
      - name: Export best bets
        run: python scripts/export_best_bets.py
      - uses: EndBug/add-and-commit@v9
        with:
          message: "Auto: weekly EPL data refresh"
```

### Feature 17 — Transfer Window Impact Model

During transfer windows (Jan, July-Sep), ingest squad changes and
recompute team strength ratings. Flag teams with significant inflows/outflows.

```python
# features/transfer_impact.py
import requests, pandas as pd, os

FOOTBALL_DATA_BASE = "https://api.football-data.org/v4"

def fetch_team_squad(team_id: int) -> pd.DataFrame:
    resp = requests.get(
        f"{FOOTBALL_DATA_BASE}/teams/{team_id}",
        headers={"X-Auth-Token": os.environ["FOOTBALL_DATA_KEY"]},
        timeout=10,
    )
    team = resp.json()
    squad = team.get("squad", [])
    return pd.DataFrame([{
        "player": p["name"], "position": p["position"],
        "nationality": p["nationality"],
        "age": (pd.Timestamp.now().year - pd.to_datetime(p.get("dateOfBirth", "2000-01-01")).year)
    } for p in squad])

def compute_squad_depth_score(squad: pd.DataFrame) -> float:
    """Score squad depth 0-100 based on positions covered."""
    positions = {"Goalkeeper", "Defender", "Midfielder", "Forward"}
    covered = len(positions.intersection(set(squad["position"].unique())))
    return covered / len(positions) * 50 + min(len(squad) / 25, 1.0) * 50
```

### Feature 18 — Value Bet Email Newsletter

Generate weekly HTML email with top upcoming EPL value bets.
Include model edge, odds, form guide, and key stats.

```python
# scripts/send_epl_newsletter.py
import smtplib, os
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import pandas as pd
from prepare_model_data import load_upcoming_predictions

def generate_newsletter_html(picks: pd.DataFrame) -> str:
    rows = ""
    for _, p in picks.head(10).iterrows():
        tier_emoji = "🔥" if p.get("edge", 0) > 0.05 else "✅"
        rows += f"""
        <tr style="background:{'#1a237e' if _ % 2 == 0 else '#0d47a1'}; color:white">
          <td>{p.get('HomeTeam', '')} vs {p.get('AwayTeam', '')}</td>
          <td>{tier_emoji} <b>{p.get('pick', '')}</b></td>
          <td>{p.get('best_odds', 'N/A')}</td>
          <td style="color:#4CAF50">{p.get('edge', 0):+.1%}</td>
        </tr>"""
    return f"""<html><body style="font-family:Arial;background:#0a0a2e;color:white">
    <h1>⚽ EPL Weekly Value Picks</h1>
    <table width="100%" border="0" cellpadding="8">
      <tr style="background:#283593"><th>Match</th><th>Pick</th><th>Best Odds</th><th>Edge</th></tr>
      {rows}
    </table>
    <p style="color:grey;font-size:11px">Model picks for entertainment only. Please gamble responsibly.</p>
    </body></html>"""

def send_newsletter(recipients: list[str]) -> None:
    picks = load_upcoming_predictions()
    html = generate_newsletter_html(picks)
    msg = MIMEMultipart("alternative")
    msg["Subject"] = "⚽ EPL Value Picks This Gameweek"
    msg["From"] = os.environ["SMTP_FROM"]
    msg["To"] = ", ".join(recipients)
    msg.attach(MIMEText(html, "html"))
    with smtplib.SMTP_SSL(os.environ["SMTP_HOST"], 465) as server:
        server.login(os.environ["SMTP_USER"], os.environ["SMTP_PASS"])
        server.sendmail(msg["From"], recipients, msg.as_string())
    print(f"Newsletter sent to {len(recipients)} recipients.")
```

### Feature 19 — Model vs Public Consensus Tracker

Compare model probabilities against aggregated public betting percentages
(from the Odds API or similar). Flag games where model disagrees with public.

```python
# utils/consensus_comparison.py
import pandas as pd

def compare_model_vs_public(model_probs: pd.DataFrame,
                              public_pcts: pd.DataFrame) -> pd.DataFrame:
    """Identify games where model disagrees with public money."""
    merged = model_probs.merge(public_pcts, on="game_id", how="inner")
    merged["model_vs_public_gap"] = (
        merged["model_home_win_prob"] - merged["public_home_pct"] / 100
    )
    merged["contrarian_pick"] = merged["model_vs_public_gap"].apply(
        lambda g: "Home vs Public" if g > 0.10
        else ("Away vs Public" if g < -0.10 else "With Public")
    )
    return merged.sort_values("model_vs_public_gap", key=abs, ascending=False)
```

### Feature 20 — Golden Boot Tracker

Track top scorers' per-90 rates, xG vs actual goals, expected remaining
goals, and outright winner model probability.

```python
# pages/golden_boot.py
import streamlit as st, pandas as pd

def render_golden_boot(player_stats: pd.DataFrame) -> None:
    st.title("⚽ Golden Boot Tracker")
    top_scorers = (
        player_stats
        .sort_values("goals", ascending=False)
        .head(20)
        .reset_index(drop=True)
    )
    top_scorers["goals_per_90"] = (
        top_scorers["goals"] / top_scorers["minutes"].clip(lower=1) * 90
    )
    top_scorers["xg_per_90"] = (
        top_scorers["xg"] / top_scorers["minutes"].clip(lower=1) * 90
    )
    top_scorers["luck_factor"] = top_scorers["goals"] - top_scorers["xg"]

    st.dataframe(
        top_scorers[["player", "team", "goals", "xg", "goals_per_90",
                      "xg_per_90", "luck_factor", "matches_remaining"]],
        width="stretch",
    )
```

### Feature 21 — Relegation Battle Probability

Daily Monte Carlo simulation for bottom-6 clubs. Output probability of
finishing bottom-3 and being relegated.

```python
# utils/relegation_simulator.py
import numpy as np, pandas as pd
from collections import defaultdict

def simulate_relegation(
    remaining: pd.DataFrame, standings: pd.DataFrame, n_sim: int = 5_000
) -> pd.DataFrame:
    bottom_6 = standings.nsmallest(6, "points")["team"].tolist()
    relegation_counts = defaultdict(int)

    for _ in range(n_sim):
        pts = standings.set_index("team")["points"].to_dict()
        for _, game in remaining.iterrows():
            home_wp = _get_win_prob(game["HomeTeam"], game["AwayTeam"])
            if np.random.random() < home_wp:
                pts[game["HomeTeam"]] = pts.get(game["HomeTeam"], 0) + 3
            else:
                pts[game["AwayTeam"]] = pts.get(game["AwayTeam"], 0) + 3
                if np.random.random() < 0.25:  # draw simulation
                    pts[game["HomeTeam"]] = pts.get(game["HomeTeam"], 0) + 1
                    pts[game["AwayTeam"]] -= 2  # revert to draw

        sorted_teams = sorted(pts, key=pts.get, reverse=True)
        relegated = sorted_teams[-3:]
        for t in relegated:
            relegation_counts[t] += 1

    return pd.DataFrame([
        {"team": t, "relegation_pct": relegation_counts[t] / n_sim}
        for t in bottom_6
    ]).sort_values("relegation_pct", ascending=False)

def _get_win_prob(home: str, away: str) -> float:
    return 0.45  # stub
```

### Feature 22 — Player Performance Cards

Individual player cards showing season stats, per-90 metrics, xG/xA,
form trend, and comparison to positional peers.

```python
# pages/player_cards.py
import streamlit as st, pandas as pd, plotly.express as px

def render_player_card(player: str, stats: pd.DataFrame) -> None:
    p = stats[stats["player"] == player].iloc[0]
    st.subheader(f"{p['player']} — {p['team']}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Goals", p.get("goals", 0))
    c2.metric("xG", f"{p.get('xg', 0):.1f}")
    c3.metric("Assists", p.get("assists", 0))
    c4.metric("xA", f"{p.get('xa', 0):.1f}")

    # Percentile radar vs position peers
    peers = stats[stats["position"] == p["position"]]
    pct = {col: (p[col] > peers[col]).mean() * 100
           for col in ["goals_per90", "xg_per90", "key_passes_per90",
                       "progressive_carries_per90"] if col in stats.columns}
    if pct:
        fig = px.bar_polar(pd.DataFrame.from_dict(pct, orient="index", columns=["pct"]).reset_index(),
                            r="pct", theta="index", template="plotly_dark",
                            title="Percentile vs Positional Peers")
        st.plotly_chart(fig, width="stretch")
```

---

## Timeline Summary

| Quarter | Focus | Key Deliverables |
|---------|-------|-----------------|
| Q1 Aug–Oct 2026 | Data foundation | FBref advanced, live odds, tactical classifier, Big-6 features, availability |
| Q2 Nov 2026–Jan 2027 | Model depth | BTTS model, correct score model, AH calculator, match importance, HTFT model |
| Q3 Feb–Apr 2027 | Visual analytics | xPts table, Elo heatmap, referee tracker, fixture cards, betting record |
| Q4 May–Jul 2027 | Automation & intelligence | GH Actions pipeline, transfer window model, newsletter, model vs public, Golden Boot, relegation simulator, player cards |
