# Bzzoiro Sports API — Integration Analysis & Implementation

## ⚠️ Coverage Assessment (Read First)

**TL;DR: Bzzoiro is useful for the Streamlit app UI layer but is NOT a viable training data source for the ML model.**

Tested April 2026. Findings:

| Question | Answer |
|---|---|
| Historical depth? | **Current season only** (2025/26). Zero data before August 2025. |
| How many PL seasons? | **1 of 5** (vs 5 full seasons from football-data.co.uk) |
| Finished PL matches available? | **309** (vs 1,821 in our training data) |
| Training data coverage? | **17%** — Bzzoiro features would be NaN for 83% of training rows |
| Actual xG coverage (current season)? | **100%** ✓ |
| Half-time score coverage? | **~60%** (only from Dec 2025 onward; we have 100% already) |
| Shotmap / Momentum / Avg Positions? | **0%** across all 309 finished PL matches |
| Player stats (rating, passes, duels)? | **~73%** — current season only |
| Predictions for upcoming matches? | **Working** ✓ — most useful feature |
| Live scores? | **Working** ✓ |

**Bottom line:** Using Bzzoiro as model features would add columns that are NaN for 83% of training history. Even with imputation, the signal-to-noise ratio is poor. The API's real value is as a **real-time/upcoming match display layer** in the Streamlit app, not as a training feature source.

---

## Overview

[Bzzoiro Sports Data](https://sports.bzzoiro.com/) provides a **free, unlimited** REST API
for football data. The `BZZOIRO_KEY` is already in `.env`. This document covers:

1. What the API offers
2. How it compares with the existing data stack
3. **Coverage assessment** — where data gaps make features unusable for ML
4. What IS genuinely useful (app layer vs model layer)
5. Recommended implementation priorities
6. Complete implementation code (`bzzoiro_football_api.py`)

---

## 1. Endpoint Reference

| Endpoint | Cache TTL | Key Notes |
|---|---|---|
| `GET /api/leagues/` | 5 min | All active leagues (PL = id 1, api_id 17) |
| `GET /api/teams/` | 5 min | 3,000+ teams; filter by `league` or `country` |
| `GET /api/events/` | 2 min | Matches by date range, league, team, status |
| `GET /api/events/{id}/` | 2 min | **Single match detail** — includes shotmap, momentum, incidents, average positions |
| `GET /api/live/` | **30 sec** | Live scores + possession, shots, corners, incidents |
| `GET /api/predictions/` | 2 min | CatBoost ML predictions: 1X2 probs, xG, O/U, BTTS, most likely score |
| `GET /api/players/` | 5 min | 21,000+ players; filter by team, position, nationality |
| `GET /api/player-stats/` | 5 min | 139,000+ per-match stat records (xG, xA, passes, duels, tackles, etc.) |
| `GET /img/{type}/{api_id}/` | 365 days | Team logos, league badges, player photos (no auth needed) |

**Base URL:** `https://sports.bzzoiro.com/api/`
**Auth:** `Authorization: Token BZZOIRO_KEY` header
**Pagination:** 50 results/page; follow `next` URL or use `?page=N`
**Price:** 100% free, unlimited requests, no credit card

---

## 2. Current Data Stack vs. Bzzoiro

| Capability | Current Source | Bzzoiro Endpoint | Verdict |
|---|---|---|---|
| Match results & basic stats | football-data.co.uk CSVs | `/api/events/` | CSVs are fine for historical; Bzzoiro adds live + spatial data |
| Upcoming fixtures | ESPN API | `/api/events/?status=notstarted` | **Supplement** — Bzzoiro has round numbers |
| Betting odds | football-data.co.uk (8 bookmakers) | Not provided | Keep existing — Bzzoiro doesn't include odds |
| Injuries | API-Football (100 req/day) | Not provided | Keep existing |
| Standings | API-Football | Not provided directly | Keep existing |
| Referee data | Playmaker Stats (scraped) | Not provided | Keep existing |
| Weather | Open-Meteo | Not provided | Keep existing |
| **Actual match xG** | **Not available** | `/api/events/{id}/` actual_home_xg / actual_away_xg | **NEW — verified populated** |
| **Match odds (H/D/A + O/U + BTTS)** | football-data.co.uk (8 bookmakers) | `/api/events/{id}/` odds_home/draw/away, odds_over/under, odds_btts | **Supplement** — additional odds source |
| **Per-shot xG with coordinates** | **Not available** | `/api/events/{id}/` shotmap | **NEW** (field exists, PL coverage building) |
| **Post-shot xGoT** | **Not available** | `/api/events/{id}/` shotmap | **NEW** (field exists, PL coverage building) |
| **Goal build-up sequences** | **Not available** | `/api/events/{id}/` incidents.sequence | **NEW** |
| **Minute-by-minute momentum** | **Not available** | `/api/events/{id}/` momentum | **NEW** (field exists, PL coverage building) |
| **Average player positions** | **Not available** | `/api/events/{id}/` average_positions | **NEW** (field exists, PL coverage building) |
| **Per-match player stats (xG, xA, passes, duels)** | **Not available at this granularity** | `/api/player-stats/` | **NEW — 30+ metrics per player per match** |
| **Player match ratings (1-10)** | **Not available** | `/api/player-stats/` rating | **NEW** |
| **Player market values** | **Not available** | `/api/players/` market_value | **NEW** |
| **External ML predictions** | **None** | `/api/predictions/` | **NEW — CatBoost second opinion** |
| **Live match stats** | **Not available** | `/api/live/` | **NEW — possession, shots in real-time** |
| **Half-time scores (live)** | Only post-match (football-data.co.uk) | `/api/events/` home_score_ht / away_score_ht | **Supplement** |
| **Team/player images** | **Not available** | `/img/{type}/{api_id}/` | **NEW — UI enhancement** |

---

### Verified Coverage — Tested April 2026

```
Bzzoiro PL historical coverage (tested live):
  2021/2022: 0 finished matches
  2022/2023: 0 finished matches
  2023/2024: 0 finished matches
  2024/2025: 0 finished matches
  2025/2026: 309 finished matches ✓

football-data.co.uk (our existing source):
  2021/2022: 380 matches ✓
  2022/2023: 380 matches ✓
  2023/2024: 380 matches ✓
  2024/2025: 380 matches ✓
  2025/2026: 301 matches (in progress) ✓
  Total: 1,821 matches with 100% HT score, 20+ bookmaker odds
```

**Match-level field coverage within the 309 available matches (2025/26):**

| Field | Coverage | Notes |
|---|---|---|
| actual_home_xg / actual_away_xg | **100%** | Every finished match |
| home_score / away_score | **100%** | |
| home_score_ht / away_score_ht | **~60%** | Only from Dec 2025 onward; our existing source has 100% |
| shotmap (per-shot xG) | **0%** | Field exists on API but null for all PL matches |
| momentum (per-minute) | **0%** | Same — field available in schema, not populated for PL |
| average_positions | **0%** | Same |
| incidents (goals, cards) | **0%** | Same |
| odds_home/draw/away, O/U, BTTS | **100%** | Additional odds source |

**Player stats (per-match) — 2025/26 only:**

| Field | Coverage | Notes |
|---|---|---|
| rating | ~73% | Solid where present |
| passes (total / accurate) | ~74% | |
| duels (won / lost) | ~61% | |
| expected_goals (xG) | ~37% | Patchy |
| expected_assists (xA) | ~35% | Patchy |

**Predictions:**
- Available from December 2025 only (~half the current season)
- 170 historical predictions available for calibration
- Measured accuracy on finished matches: **42.9%** (our XGBoost ensemble typically achieves ~55-60%)

---

### What this means for model features

Our training dataset has 1,821 matches (2021-2026). Any Bzzoiro-derived feature would be:
- **NaN for 1,512 rows (83%)** — the entire 2021-2025 history
- **Present for 309 rows (17%)** — the current season

Even for those 309 rows, rolling window features (last 5 matches) would be partially NaN for the first ~5 matches of each team's season. Adding features with 83% missingness to an XGBoost model is not inherently fatal (it handles NaN via `missing` parameter), but:
1. The model effectively cannot learn from those features against 4 seasons of historical variance
2. They would add noise rather than signal during training
3. For inference on upcoming matches, the features WOULD be populated — but only if the training signal is strong enough to have learned from 17% of data

**Conclusion:** Do not use Bzzoiro data as model training features at this time. Re-evaluate when 2-3 full seasons of data exist (2026/27 season at earliest).

---

## 3. Gaps Filled — New Data Not Available Anywhere Else

### 3a. Actual Match xG (Confirmed Working — Current Season Only)

The event detail endpoint returns `actual_home_xg` and `actual_away_xg` — the **post-match expected goals** for each team. This is independent of our Poisson-derived xG and provides a true benchmark.

> ⚠️ **Coverage caveat:** 100% populated for the 309 finished 2025/26 PL matches, but **zero historical data** exists before August 2025. Adding this as a model training feature would produce NaN for 83% of training rows. Viable for display and monitoring; not viable as a primary ML feature until 2-3 seasons accumulate.

**What it is useful for:**
- **Streamlit display**: Show actual xG alongside predictions on current season matches
- **Current-season monitoring**: Track which teams over/underperform xG in 2025/26
- **Poisson calibration check**: One-season comparison of our Poisson output vs Bzzoiro's actual_xg
- **Future feature (from 2026/27 onward)**: Once multi-season data exists, `Home_ActualXg_L5`, `Away_ActualXg_L5`, xG overperformance features will be viable

Example from API:
```
Man United vs Crystal Palace: xG 1.74 - 1.97 (actual score 2-1)
Aston Villa vs Brentford:    xG 2.14 - 0.43
```

### 3b. Match Odds Suite (H/D/A + Over/Under + BTTS)

Event detail includes a full odds suite not available from football-data.co.uk's bookmaker columns:
- `odds_home`, `odds_draw`, `odds_away` — 1X2 odds
- `odds_over_15`, `odds_over_25`, `odds_over_35` — over goals
- `odds_under_15`, `odds_under_25`, `odds_under_35` — under goals
- `odds_btts_yes`, `odds_btts_no` — both teams to score

**Why it matters:**
- Additional odds source for cross-validation with Bet365/Pinnacle
- Over/Under and BTTS odds can be converted to implied probabilities as features
- Useful when football-data.co.uk odds are missing for recent matches

### 3c. Per-Shot xG with Pitch Coordinates (Shotmap)

> ⚠️ **Not available for PL:** Coverage tests showed **0/20 finished matches** had any shotmap data — the field exists in the API schema but is null for all Premier League matches. This may populate in future. Do not implement PL shotmap features until coverage improves.

If/when this populates, the `shotmap` field on `/api/events/{id}/` would return every shot in a match with:
- **Pre-shot xG** — expected goals based on shot position and situation
- **Post-shot xGoT** (xG on Target) — factors in shot placement on the goal frame
- **Pitch coordinates** — `pos.x`, `pos.y` (105×68 grid)
- **Goal-mouth target** — `gm.y`, `gm.z` (goal frame coordinates)
- **Body part** — head, left-foot, right-foot
- **Situation** — open-play, set-piece, corner, counter-attack

**Why it matters for the model:**
Currently, xG features in `prepare_model_data.py` are derived from aggregate shot counts. Per-shot xG enables:
- **Shot quality features**: Average xG per shot (a team taking 2 high-quality chances vs 10 speculative shots)
- **xGoT features**: Post-shot quality is far more predictive than pre-shot xG alone
- **Set piece xG**: Proportion of xG from corners/free kicks vs open play
- **Big chance creation**: Count of shots with xG > 0.3

```json
"shotmap": [
  {
    "min": 26, "type": "goal", "sit": "corner", "body": "head",
    "home": true, "xg": 0.383, "xgot": 0.949,
    "pos": {"x": 3.9, "y": 45.8}, "gm": {"y": 54.4, "z": 13.9},
    "gml": "low-left", "pid": 548194
  }
]
```

### 3b. Minute-by-Minute Momentum

> ⚠️ **Not available for PL:** The `momentum` field is null for all tested PL matches (0/20 in coverage test). Same status as shotmap — field exists in schema, no data for Premier League.

If/when this populates, the momentum index (positive = home, negative = away) would enable:

**Potential future features:**
- **Late-game momentum features**: Average momentum in minutes 75-90 (do they push for goals late?)
- **Momentum consistency**: Standard deviation of momentum (steady vs erratic)
- **First-half dominance**: Average momentum in 1st vs 2nd half (fade factor)
- **Comeback tendency**: Frequency of switching from negative to positive momentum

```json
"momentum": [
  {"minute": 1, "value": 12},
  {"minute": 2, "value": -5},
  {"minute": 45, "value": 28},
  {"minute": 90, "value": 8}
]
```

### 3c. Average Player Positions (Formations)

> ⚠️ **Not available for PL:** The `average_positions` field is null for all tested PL matches (0/20 in coverage test).

If/when this populates, average positions would enable:
- **Formation encoding**: Detect 4-3-3, 4-4-2, 3-5-2 from positions (currently only available from API-Football as a string)
- **Defensive line height**: Average y-position of defenders (high line = vulnerable to counters)
- **Width of play**: Spread of player positions (compact vs expansive)
- **Tactical mismatch detection**: Compare two teams' shape to identify overloads

```json
"average_positions": {
  "home": [
    {"player": "D. Fry", "pid": 548194, "pos": {"x": 88.3, "y": 34.1}, "number": 9}
  ]
}
```

### 3d. Per-Match Player Statistics (Current Season Only)

> ⚠️ **Coverage caveat:** Player stats confirmed working but **2025/26 season only**. The 139,000+ figure is across all covered leagues globally — PL-specific data totals ~620-840 records per team (one season's worth). Field completeness within those records: rating ~73%, passes ~74%, duels ~61%, xG ~37%.

`/api/player-stats/` provides 30+ metrics per player per match:

| Category | Metrics |
|---|---|
| **Attack** | goals, assists, xG, xA, total_shots, shots_on_target |
| **Passing** | total_pass, accurate_pass, key_pass, total_cross, accurate_cross, long_balls |
| **Duels** | duel_won, duel_lost, aerial_won, aerial_lost |
| **Defense** | total_tackle, won_tackle, total_clearance, interception, ball_recovery |
| **Discipline** | fouls, was_fouled, yellow_card, red_card, dispossessed, possession_lost |
| **General** | minutes_played, rating (1-10), touches |
| **Goalkeeper** | saves, goals_conceded |

**Why it matters:**
- **Key player dependency**: If Salah's xG accounts for 60% of Liverpool's total, that's a fragility signal
- **Player form features**: Rolling average rating over last 5 matches per key player
- **Squad depth quality**: Average rating of non-starters who play (substitutes)
- **Passing network proxy**: Team-level pass accuracy as a style indicator
- **Duel dominance**: Team-level aerial/ground duel win rate → predicts set-piece effectiveness

### 3e. Player Market Values

`/api/players/` includes `market_value` in EUR for each player.

**Why it matters:**
- **Squad value as feature**: Total team market value is a strong predictor of league position
- **Relative squad strength**: Home squad value / Away squad value ratio
- **Bench quality**: Value of likely substitutes (players not starting)

### 3f. External CatBoost Predictions

> ⚠️ **Coverage caveat:** Predictions confirmed working but only from December 2025 onward (170 historical records). Measured accuracy on finished matches: **42.9%** — our XGBoost ensemble typically achieves ~55-60%. Use as a display overlay, not as a training feature.

`/api/predictions/` returns an independent ML model's output:
- H/D/A probabilities (0-100)
- Expected home/away goals
- Over 1.5/2.5/3.5 probabilities
- BTTS probability
- Most likely scoreline (e.g. "2-1")
- Model confidence (0-1)

**What it is useful for (app layer only):**
- **Streamlit display**: Show Bzzoiro predictions alongside ours as a second opinion
- **Divergence flag**: Highlight matches where the two models strongly disagree
- **BTTS and O/U display**: We don't currently output these in the app — Bzzoiro provides instant coverage
- **Not recommended as a model feature** until calibration improves (42.9% < our existing accuracy)

### 3g. Goal Build-Up Sequences

Goals include a `sequence` field with the full passing chain:
- Every touch: player name, event type (pass, cross, dribble, goal)
- Origin and destination coordinates per touch
- Assist flags, body part, GK position at moment of strike

**Why it matters:**
- **Attack pattern classification**: Counter-attack (few passes, long distances) vs build-up (many short passes)
- **Set-piece dependency**: Proportion of goals from corners/free kicks vs open play
- **Assist quality**: Average xA of the assisting pass

### 3h. Live Match Dashboard Data

`/api/live/` returns real-time stats during matches:
- Ball possession %, total shots, shots on target, corners, fouls, cards, offsides
- Incidents (goals, cards, substitutions with minute)

**Why it matters for the app (not the model):**
- **Live scores tab** in Streamlit — currently missing entirely
- **In-play analysis**: Show how a match is unfolding vs pre-match predictions

---

## 4. New Features This Enables

> **Coverage reality check:** None of Bzzoiro's data is suitable for model training features right now. The ML model trains on 1,821 matches across 5 seasons. Bzzoiro covers only 309 matches in a single season — adding Bzzoiro features would produce NaN for 83% of training rows. The tables below are organized by what is immediately viable vs what requires waiting for more seasons.

### Viable Now — Streamlit App Layer Only

| Display / Feature | Source | Status |
|---|---|---|
| Bzzoiro H/D/A prediction display | `/api/predictions/` | ✅ Working — show alongside our model |
| Bzzoiro xG display | `/api/predictions/` | ✅ Working — show alongside our Poisson xG |
| Expected scoreline display | `/api/predictions/` | ✅ Working |
| BTTS / O2.5 display | `/api/predictions/` | ✅ Working — we have no current BTTS output |
| Live match scores tab | `/api/live/` | ✅ Working |
| Actual xG for current matches | `/api/events/{id}/` | ✅ Working for 2025/26 |
| Squad value ratio display | `/api/players/` | ✅ Working |
| Model divergence flag | predictions vs ours | ✅ Useful for UI callouts |

### Not Viable for Model Training — Insufficient Coverage

| Feature | Blocker | Re-evaluate |
|---|---|---|
| `Home_ActualXg_L5` rolling feature | 83% NaN in training data | After 2026/27 season |
| `Home_XgOverperformance_L5` | Same | After 2026/27 season |
| All shotmap features (`AvgXgPerShot`, `xGoT`, etc.) | 0% PL coverage | Unknown — when PL shotmap populates |
| All momentum features | 0% PL coverage | Unknown |
| All average_position features | 0% PL coverage | Unknown |
| Player stat features (`PassAccuracy_L5`, `DuelWinRate_L5`, etc.) | Current season only | After 2026/27 season |
| `Bzz_ProbHomeWin` as training meta-feature | Only Dec 2025 onward, 42.9% accuracy | After accuracy improves |

---

## 5. Recommended Implementation Priorities

### Do Now (App Layer)

| Priority | Task | Bzzoiro Endpoint | Impact |
|---|---|---|---|
| **P0** | `bzzoiro_football_api.py` client | All | ✅ Already created |
| **P1** | Add Bzzoiro predictions panel to Streamlit | `/api/predictions/?league=1` | Second opinion on upcoming fixtures |
| **P1** | Add live scores tab to Streamlit | `/api/live/` | Missing feature entirely |
| **P1** | Collect actual xG for 2025/26 current season | `/api/events/{id}/` | Display alongside predictions |
| **P2** | Calibration comparison: Bzzoiro accuracy vs ours | historical predictions | Benchmarking |
| **P2** | Team logos / player photos | `/img/{type}/{api_id}/` | Visual polish |

### Wait (Model Training Features)

| Task | Blocker | Watch For |
|---|---|---|
| Actual xG as rolling model feature | Only 1 season of data | Re-check Sept 2026 (after 2026/27 starts) |
| Player stat features | Only 1 season of data | Same |
| Shotmap features | 0% PL coverage | API changelog / Bzzoiro support |
| Momentum / formation features | 0% PL coverage | Same |

---

## 6. Implementation — `bzzoiro_football_api.py`

The complete implementation is in [bzzoiro_football_api.py](../bzzoiro_football_api.py). Key functions:

### Core API Functions

```python
from bzzoiro_football_api import (
    get_pl_events,          # Fetch PL matches by date range
    get_event_detail,       # Single match with shotmap, momentum, positions
    get_live_matches,       # Currently live PL matches
    get_pl_predictions,     # CatBoost predictions for PL matches
    get_pl_players,         # PL team players with market values
    get_player_stats,       # Per-match player stats (xG, xA, passes, etc.)
    get_pl_league_id,       # Look up PL league ID
)
```

### Data Collection Scripts

```python
# Backfill shotmap data for historical PL matches
from bzzoiro_football_api import backfill_shotmap_data
backfill_shotmap_data(
    date_from="2024-08-01",
    date_to="2026-04-05",
    output_path="data_files/bzzoiro_shotmap.csv"
)

# Fetch player stats for all PL teams
from bzzoiro_football_api import backfill_player_stats
backfill_player_stats(output_path="data_files/bzzoiro_player_stats.csv")

# Fetch squad market values
from bzzoiro_football_api import fetch_squad_values
fetch_squad_values(output_path="data_files/bzzoiro_squad_values.csv")

# Fetch upcoming predictions
from bzzoiro_football_api import fetch_upcoming_predictions
fetch_upcoming_predictions(output_path="data_files/bzzoiro_predictions.csv")
```

### Feature Engineering (add to `prepare_model_data.py`)

```python
from bzzoiro_football_api import compute_shotmap_features, compute_player_stat_features

# After loading historical data:
shotmap_features = compute_shotmap_features("data_files/bzzoiro_shotmap.csv")
player_features = compute_player_stat_features("data_files/bzzoiro_player_stats.csv")

# Merge into match data
df = df.merge(shotmap_features, on=["HomeTeam", "AwayTeam", "MatchDate"], how="left")
df = df.merge(player_features, on=["HomeTeam", "AwayTeam", "MatchDate"], how="left")
```

### Streamlit Integration

```python
# Add to premier-league-predictions.py in the upcoming matches tab:
from bzzoiro_football_api import get_pl_predictions

@st.cache_data(ttl=120)
def load_bzzoiro_predictions():
    return get_pl_predictions()

preds = load_bzzoiro_predictions()

# Display per match:
for pred in preds:
    event = pred["event"]
    st.markdown(f"**{event['home_team']} vs {event['away_team']}**")
    st.caption(
        f"Bzzoiro CatBoost: H {pred['prob_home_win']:.1f}% | "
        f"D {pred['prob_draw']:.1f}% | A {pred['prob_away_win']:.1f}%  ·  "
        f"Score: {pred['most_likely_score']}  ·  "
        f"Confidence: {pred['confidence']:.0%}"
    )
```

### Live Scores Tab

```python
# Add a new tab in premier-league-predictions.py:
from bzzoiro_football_api import get_live_matches

with tab_live:
    st.subheader("Live Matches")
    live = get_live_matches()
    if not live:
        st.info("No Premier League matches currently in play.")
    for m in live:
        stats = m.get("live_stats") or {}
        home_stats = stats.get("home", {})
        away_stats = stats.get("away", {})
        st.markdown(
            f"**{m['current_minute']}'** — "
            f"**{m['home_team']}** {m['home_score']} – {m['away_score']} **{m['away_team']}**  \n"
            f"Possession: {home_stats.get('ball_possession', '?')}% – {away_stats.get('ball_possession', '?')}%  |  "
            f"Shots: {home_stats.get('total_shots', '?')} – {away_stats.get('total_shots', '?')}"
        )
```

---

## 7. Data Volume Estimates

| Data Type | Records | Storage | Fetch Time |
|---|---|---|---|
| PL events 2024-2026 (~760 matches) | 760 event details | ~50 MB JSON | ~15 min (serial) |
| Shotmap data (embedded in events) | ~12,000 shots | ~8 MB CSV | Included above |
| Player stats (20 PL teams × ~season) | ~25,000 records | ~15 MB CSV | ~10 min |
| Squad market values (20 teams) | ~500 players | ~100 KB CSV | <1 min |
| Predictions (upcoming) | 10-20 per matchweek | <10 KB | <1 sec |

No rate limits, so all backfills can run as fast as network allows.

---

## 8. Team Name Mapping

Bzzoiro uses full team names (e.g. "Manchester United", "Wolverhampton Wanderers").
Update `team_name_mapping.py` to include a `BZZOIRO_TEAM_MAP` or reuse the existing
`API_FOOTBALL_TEAM_MAP` from `fetch_api_football.py` since Bzzoiro uses the same naming
convention.

---

## 9. Summary

Bzzoiro fills **6 major data gaps** that no other source in the current stack provides:

1. **Per-shot xG + xGoT with coordinates** — enables shot quality features that are more predictive than aggregate xG
2. **Minute-by-minute momentum** — enables momentum consistency and late-game pressure features
3. **Per-match player-level stats** — enables key player dependency, team style, and squad quality features
4. **External CatBoost predictions** — instant second opinion for model consensus/divergence
5. **Player market values** — squad value as a strength proxy
6. **Average player positions** — formation and tactical features

The unlimited, free nature of the API means we can aggressively backfill data and poll
frequently without worrying about quotas (unlike API-Football's 100 req/day limit).
