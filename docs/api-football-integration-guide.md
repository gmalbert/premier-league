# API-Football v3 Integration Guide

> **API Version**: 3.9.3 | **Plan**: Free (100 requests/day) | **League ID**: 39 (Premier League) | **Current Season**: 2025

## Table of Contents

1. [API Overview & Authentication](#1-api-overview--authentication)
2. [Complete Endpoint Catalog](#2-complete-endpoint-catalog)
3. [Premier League Coverage Matrix](#3-premier-league-coverage-matrix)
4. [Current Features vs. API Capabilities — Gap Analysis](#4-current-features-vs-api-capabilities--gap-analysis)
5. [Recommended Data Pulls (Priority Order)](#5-recommended-data-pulls-priority-order)
6. [Model Enhancement Proposals](#6-model-enhancement-proposals)
7. [App Feature Enhancements](#7-app-feature-enhancements)
8. [Request Budget Strategy (100/day)](#8-request-budget-strategy-100day)
9. [Implementation Code](#9-implementation-code)
10. [Phased Rollout Plan](#10-phased-rollout-plan)

---

## 1. API Overview & Authentication

### Base URL
```
https://v3.football.api-sports.io/
```

### Authentication Header
```python
headers = {"x-apisports-key": "YOUR_API_KEY"}
```

### Rate Limits
| Metric | Value |
|--------|-------|
| Daily requests | 100 |
| Per-minute rate | Varies (don't exceed burst) |
| Logo/image calls | Free (don't count against quota) |
| `/status` endpoint | Free (doesn't count against quota) |

### Response Headers (monitor usage)
- `x-ratelimit-requests-limit` — daily allocation
- `x-ratelimit-requests-remaining` — remaining today
- `X-RateLimit-Limit` — per-minute cap
- `X-RateLimit-Remaining` — remaining this minute

### Python Base Client
```python
import requests
import os
import json
import time
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("SPORTS_API_KEY")
BASE_URL = "https://v3.football.api-sports.io"
LEAGUE_ID = 39  # Premier League
CURRENT_SEASON = 2025

HEADERS = {"x-apisports-key": API_KEY}

def api_get(endpoint, params=None):
    """Make a rate-aware GET request to API-Football."""
    url = f"{BASE_URL}/{endpoint}"
    response = requests.get(url, headers=HEADERS, params=params, timeout=30)
    data = response.json()

    remaining = response.headers.get("x-ratelimit-requests-remaining", "?")
    print(f"[API] {endpoint} — {data.get('results', 0)} results, {remaining} requests remaining today")

    if data.get("errors"):
        print(f"[API ERROR] {data['errors']}")
        return None

    return data["response"]
```

---

## 2. Complete Endpoint Catalog

### Reference / Metadata (Low cost — call once and cache)
| Endpoint | Description | Recommended Frequency |
|----------|-------------|----------------------|
| `GET /status` | Account status/quota (FREE) | Before each batch |
| `GET /timezone` | Available timezones | Once ever |
| `GET /countries` | Countries list | Once ever |
| `GET /leagues` | League info + coverage | 1x/day |
| `GET /leagues/seasons` | Available seasons | 1x/day |
| `GET /venues` | Stadium info (capacity, city, surface) | 1x/week |

### Teams
| Endpoint | Description | Recommended Frequency |
|----------|-------------|----------------------|
| `GET /teams?league=39&season=2025` | Team info (name, logo, venue) | 1x/week |
| `GET /teams/statistics?league=39&team={id}&season=2025` | **Season stats**: form string, goals by minute, cards by minute, clean sheets, lineups used, penalty record | 1x/day |
| `GET /teams/seasons?team={id}` | Seasons a team participated in | 1x/month |
| `GET /teams/countries` | Countries with teams | 1x/month |

### Fixtures
| Endpoint | Description | Recommended Frequency |
|----------|-------------|----------------------|
| `GET /fixtures?league=39&season=2025` | All fixtures (scheduled + completed) | 1x/day |
| `GET /fixtures?league=39&next=10` | Next 10 upcoming matches | 1x/day |
| `GET /fixtures?league=39&last=10` | Last 10 completed matches | After matchday |
| `GET /fixtures?id={fixture_id}` | Single fixture with events, lineups, stats, players | As needed |
| `GET /fixtures?ids={id1}-{id2}-...` | Up to 20 fixtures in one call (with full detail) | **Best for bulk** |
| `GET /fixtures/rounds?league=39&season=2025` | Matchday/round info | 1x/week |
| `GET /fixtures/headtohead?h2h={team1_id}-{team2_id}` | H2H history between two teams | As needed |
| `GET /fixtures/statistics?fixture={id}` | **Match statistics**: shots, possession, passes, corners, fouls, offsides, saves | After matchday |
| `GET /fixtures/events?fixture={id}` | Goals, cards, subs, VAR events | After matchday |
| `GET /fixtures/lineups?fixture={id}` | Formation, starting XI, substitutes, coach | 20-40 min pre-match |
| `GET /fixtures/players?fixture={id}` | **Per-player stats per match**: rating, shots, passes, tackles, duels, dribbles | After matchday |

### Players
| Endpoint | Description | Recommended Frequency |
|----------|-------------|----------------------|
| `GET /players?league=39&season=2025&team={id}` | Full player statistics (paginated, 20/page) | 1x/week |
| `GET /players/squads?team={id}` | Current squad (name, age, number, position, photo) | 1x/week |
| `GET /players/topscorers?league=39&season=2025` | Top 20 scorers | 1x/day |
| `GET /players/topassists?league=39&season=2025` | Top 20 assists | 1x/day |
| `GET /players/topyellowcards?league=39&season=2025` | Top 20 yellow cards | 1x/day |
| `GET /players/topredcards?league=39&season=2025` | Top 20 red cards | 1x/day |
| `GET /players/profiles?player={id}` | Player bio (nationality, height, weight, age) | 1x/month |
| `GET /players/seasons` | Available seasons for player stats | 1x/month |
| `GET /players/teams?player={id}` | Career history by team/season | 1x/month |

### Standings
| Endpoint | Description | Recommended Frequency |
|----------|-------------|----------------------|
| `GET /standings?league=39&season=2025` | Full league table (rank, points, W/D/L, GD, form) | 1x/day |

### Coaches
| Endpoint | Description | Recommended Frequency |
|----------|-------------|----------------------|
| `GET /coachs?team={id}` | Coach info + career history | 1x/week |

### Injuries & Sidelined
| Endpoint | Description | Recommended Frequency |
|----------|-------------|----------------------|
| `GET /injuries?league=39&season=2025` | Missing/questionable players per fixture | 1x/day |
| `GET /injuries?fixture={id}` | Injuries for specific fixture | Pre-match |
| `GET /sidelined?player={id}` | Player injury history | 1x/week |

### Transfers
| Endpoint | Description | Recommended Frequency |
|----------|-------------|----------------------|
| `GET /transfers?team={id}` | All transfers for a team | 1x/week |

### Predictions (API's own)
| Endpoint | Description | Recommended Frequency |
|----------|-------------|----------------------|
| `GET /predictions?fixture={id}` | API's built-in prediction: winner, under/over, advice, team comparison stats | Pre-match |

### Odds
| Endpoint | Description | Recommended Frequency |
|----------|-------------|----------------------|
| `GET /odds?fixture={id}` | Pre-match odds from multiple bookmakers | Pre-match |
| `GET /odds?league=39&season=2025` | All odds for season (paginated, 10/page) | 1x/3hr |
| `GET /odds/live` | In-play odds (live matches only) | During matches |
| `GET /odds/bookmakers` | Available bookmakers list | 1x/week |
| `GET /odds/bets` | Available bet types | 1x/week |
| `GET /odds/live/bets` | In-play bet types | 1x/week |
| `GET /odds/mapping` | Fixtures with available odds | 1x/day |

---

## 3. Premier League Coverage Matrix

Verified from API response for league 39, season 2025:

| Feature | Available | Since Season |
|---------|-----------|-------------|
| Fixture events (goals, cards, subs) | ✅ | 2010+ |
| Lineup data | ✅ | 2010+ |
| Fixture statistics (shots, possession, passes) | ✅ | 2018+ |
| Player statistics per fixture | ✅ | 2018+ |
| Standings | ✅ | 2010+ |
| Player season stats | ✅ | 2010+ |
| Top scorers / assists / cards | ✅ | 2010+ |
| Injuries | ✅ | 2020+ |
| Predictions | ✅ | 2010+ |
| Pre-match odds | ✅ | 2025 |
| Half-time statistics | ✅ | 2024+ |

---

## 4. Current Features vs. API Capabilities — Gap Analysis

### What You Already Have
| Feature | Current Source | Status |
|---------|---------------|--------|
| Historical match results & stats | football-data.co.uk CSVs | ✅ Solid |
| Upcoming fixtures | ESPN API | ✅ Working |
| Betting odds (15+ bookmakers) | football-data.co.uk CSVs | ✅ Comprehensive |
| Team rolling form (last 5) | Calculated from CSVs | ✅ Working |
| H2H history | Calculated from CSVs | ✅ Working |
| Rest days | Calculated from CSVs | ✅ Working |
| Weather data | Open-Meteo API | ✅ Working |
| Injury counts | footballinjurynews.com scraping | ⚠️ Fragile (scraping) |
| Referee stats | Playmaker Stats scraping | ⚠️ Fragile (scraping) |
| Manager data | Hardcoded in code | ⚠️ Stale |
| Player data | football-data.org (squad sizes only) | ⚠️ Limited |

### What API-Football Adds (NEW capabilities)

| Gap | API-Football Endpoint | Impact on Predictions |
|-----|----------------------|----------------------|
| **Individual player ratings** | `/players?league=39&season=2025&team={id}` | 🔴 HIGH — Know if a team's key players are in peak form |
| **Player-level match stats** | `/fixtures/players?fixture={id}` | 🔴 HIGH — Per-match player ratings, shots, passes, duels, dribbles |
| **Formation & tactical lineup** | `/fixtures/lineups?fixture={id}` | 🔴 HIGH — Formation matchups (e.g., 4-3-3 vs 3-5-2) |
| **Live standings & form** | `/standings?league=39&season=2025` | 🟡 MEDIUM — Real-time league position and form string |
| **Team season statistics** | `/teams/statistics?league=39&team={id}&season=2025` | 🔴 HIGH — Goals by minute, clean sheets, penalty record, most-used formation |
| **Detailed injury data** | `/injuries?league=39&season=2025` | 🔴 HIGH — Replace fragile web scraping with structured API data (player name, type, reason) |
| **Coach career data** | `/coachs?team={id}` | 🟡 MEDIUM — Replace hardcoded manager data with live career histories |
| **Transfer activity** | `/transfers?team={id}` | 🟡 MEDIUM — Track mid-season squad changes |
| **API predictions** | `/predictions?fixture={id}` | 🟡 MEDIUM — Use as ensemble feature or comparison benchmark |
| **Pre-match odds** | `/odds?fixture={id}` | 🟡 MEDIUM — Fresh pre-match odds from API (not just historical) |
| **Fixture events (VAR, subs)** | `/fixtures/events?fixture={id}` | 🟢 LOW — Enrich match event data |
| **Player sidelined history** | `/sidelined?player={id}` | 🟢 LOW — Injury-prone player identification |
| **Trophies** | `/trophies?player={id}` | 🟢 LOW — Squad quality proxy |
| **Half-time statistics** | `/fixtures/statistics?fixture={id}&half=true` | 🟡 MEDIUM — First-half performance patterns |

---

## 5. Recommended Data Pulls (Priority Order)

### Priority 1: Replace Fragile Scrapers (saves maintenance, improves reliability)

#### 1a. Injuries — Replace web scraping with API
**Replaces**: `scrape_injuries_web.py` (footballinjurynews.com scraping)
**Cost**: 1 request per matchday
**Benefit**: Structured data, player IDs, injury type, missing/questionable status

#### 1b. Fixtures — Replace ESPN API
**Replaces**: `fetch_upcoming_fixtures.py` (ESPN API)
**Cost**: 1 request
**Benefit**: Fixture IDs needed for all other endpoints, referee info, venue, status

### Priority 2: High-Impact New Features

#### 2a. Team Season Statistics
**New feature**: Goals by minute, clean sheet %, penalty record, most-used formation, home/away form string
**Cost**: 20 requests (1 per team)
**Benefit**: Rich season-level features not available from CSVs

#### 2b. Standings (League Table)
**New feature**: Real-time league position, points, GD, form string per team
**Cost**: 1 request
**Benefit**: League position is a strong predictor; current approach calculates from historical data

#### 2c. Player Statistics (Team Squads)
**New feature**: Player ratings, goals, assists, passes, key passes, tackles, dribbles per player
**Cost**: 20 requests (1 per team, paginated) — or use `/players/topscorers` + `/players/topassists` for 2 requests
**Benefit**: Star player availability weighting for injury impact

### Priority 3: Pre-Match Enrichment

#### 3a. Lineups & Formation
**New feature**: Starting XI, formation, coach
**Cost**: 1 request per upcoming fixture (available 20-40 min before kickoff)
**Benefit**: Formation stats, actual vs expected lineup strength

#### 3b. API Predictions as Ensemble Feature
**New feature**: API's own Poisson-based predictions
**Cost**: 1 request per upcoming fixture
**Benefit**: Independent model signal to boost ensemble accuracy

#### 3c. Pre-Match Odds
**New feature**: Fresh bookmaker odds for upcoming fixtures
**Cost**: 1 request per fixture or batch by league/season
**Benefit**: Most recent odds (not just historical); available 1-14 days before fixture

### Priority 4: Background Enrichment (Weekly)

#### 4a. Coach Data
**Replaces**: Hardcoded manager data in `manager_data.py`
**Cost**: 20 requests (1 per team)
**Benefit**: Dynamic manager experience, career history

#### 4b. Transfer Activity
**New feature**: Track mid-season arrivals/departures
**Cost**: 20 requests (1 per team)
**Benefit**: Squad disruption feature

---

## 6. Model Enhancement Proposals

### Enhancement 1: Player-Weighted Injury Impact
**Current**: `HomeInjuryCount` / `AwayInjuryCount` (all injuries equal)
**Proposed**: Weight injuries by player importance (rating, goals, assists)

```python
# New features to engineer:
# - HomeKeyPlayersMissing: count of missing players with rating > 7.0
# - AwayKeyPlayersMissing: same for away
# - HomeSquadStrengthReduction: sum of missing player ratings / total squad rating
# - AwaySquadStrengthReduction: same for away
# - HomeTopScorerAvailable: binary (is top scorer injured?)
```

### Enhancement 2: Formation-Based Features
**Current**: Not modeled
**Proposed**: Use lineup data to create formation matchup features

```python
# New features:
# - HomeFormation: encoded formation (e.g., "4-3-3" → numeric ID)
# - AwayFormation: encoded formation
# - FormationMatchup: interaction feature (how does A's formation do vs B's)
# - HomeFormationWinRate: historical win rate with this formation
```

### Enhancement 3: Team Season Statistics
**Current**: Calculated from rolling match-level data
**Proposed**: Use API's pre-aggregated season stats for richer features

```python
# New features from /teams/statistics:
# - HomeCleanSheetPct: % of matches with clean sheet
# - AwayCleanSheetPct
# - HomeGoalsByMinute_0_15: goals scored in first 15 min
# - HomeGoalsByMinute_76_90: goals scored in last 15 min (late-game strength)
# - HomePenaltyConversionRate: penalties scored / penalties taken
# - MostUsedFormation: team's preferred formation
# - HomeFailedToScorePct: % of matches with 0 goals
```

### Enhancement 4: League Position Features
**Current**: Not directly used (implied through form/points)
**Proposed**: Direct standing position and form from API

```python
# New features from /standings:
# - HomeLeaguePosition: integer 1-20
# - AwayLeaguePosition: integer 1-20
# - PositionDifference: home_pos - away_pos
# - HomePointsPerGame: total points / matches played
# - HomeGoalDifferenceStandings: from official table
# - HomeFormString: last 5 results (WWDLW → numeric encoding)
```

### Enhancement 5: API Predictions as Ensemble Signal
**Current**: Own models only (XGBoost, Ensemble, Neural, LSTM, Poisson)
**Proposed**: Feed API predictions as features into ensemble

```python
# New features from /predictions:
# - APIPrediction_HomeWinPct: API's home win probability
# - APIPrediction_DrawPct
# - APIPrediction_AwayWinPct
# - APIPrediction_UnderOver25
# - APIPrediction_Advice: encoded recommendation
```

### Enhancement 6: Player Performance Aggregates
**Current**: Team-level aggregates only
**Proposed**: Aggregate per-player stats into team-strength metrics

```python
# New features from /players:
# - HomeAvgPlayerRating: average rating of starting players
# - AwayAvgPlayerRating
# - HomeTopPlayerRating: best player's rating
# - HomeTotalGoalContributions: goals + assists of squad
# - HomeSquadDepth: number of players with 5+ appearances
# - HomeGoalkeeperSaveRate: GK saves / (saves + goals conceded)
```

### Enhancement 7: Pre-Match Odds (Fresh)
**Current**: Historical odds from football-data.co.uk CSVs (only available after match data is published)
**Proposed**: Real-time pre-match odds from API for upcoming fixtures

```python
# New features from /odds:
# - FreshOdds_HomeWin: latest bookmaker odds for home win
# - FreshOdds_Draw: latest draw odds
# - FreshOdds_AwayWin: latest away win odds
# - FreshOdds_ImpliedProbHome: 1 / odds_home
# - FreshOdds_Over25: over 2.5 goals odds
# - FreshOdds_BTTS: both teams to score odds
```

---

## 7. App Feature Enhancements

### 7a. New Streamlit Tab: "Team Deep Dive"
- Select a team → show season stats from API
- Formation breakdown, goals by minute chart, clean sheet %, card frequency
- Player ratings table sorted by contribution
- Recent fixture results with events timeline

### 7b. Enhanced Predictions Tab
- Show API's own prediction alongside your models
- Side-by-side comparison: Your Ensemble vs API Prediction
- Confidence calibration: track which model is more accurate over time
- Pre-match odds display with implied probabilities

### 7c. Injury Intelligence Panel
- Replace current basic injury count display
- Show specific missing players with their season rating and importance
- "Squad Strength Impact" visualization
- Compare starting XI strength vs weakened XI

### 7d. Standings Integration
- Live league table widget
- Position trend chart (how position changed over weeks)
- Relegation/title race probability from model

### 7e. Match Preview Card
For each upcoming fixture, show:
- Both teams' form strings
- League positions
- Key missing players (with ratings)
- H2H recent results
- API prediction vs your model prediction
- Fresh pre-match odds
- Formation (when available pre-match)

---

## 8. Request Budget Strategy (100/day)

### Daily Budget Allocation

| Category | Endpoints | Requests | Frequency |
|----------|-----------|----------|-----------|
| **Status check** | `/status` | 0 (free) | Every run |
| **Standings** | `/standings?league=39&season=2025` | 1 | Daily |
| **Fixtures (upcoming)** | `/fixtures?league=39&next=15` | 1 | Daily |
| **Fixtures (recent results)** | `/fixtures?league=39&last=10` | 1 | Daily |
| **Injuries** | `/injuries?league=39&season=2025` | 1 | Daily |
| **Top scorers** | `/players/topscorers?league=39&season=2025` | 1 | Daily |
| **Top assists** | `/players/topassists?league=39&season=2025` | 1 | Daily |
| **API Predictions** | `/predictions?fixture={id}` × upcoming | ~8 | Match days |
| **Pre-match odds** | `/odds?fixture={id}` × upcoming | ~8 | Match days |
| **Daily subtotal** | | **~22** | |

### Weekly Budget Allocation (spread across week)

| Category | Endpoints | Requests | Day |
|----------|-----------|----------|-----|
| **Team stats** | `/teams/statistics` × 20 teams | 20 | Monday |
| **Coach data** | `/coachs?team={id}` × 20 | 20 | Tuesday |
| **Player squads** | `/players/squads?team={id}` × 20 | 20 | Wednesday |
| **Venues** | `/venues?country=england` | 1 | Wednesday |
| **Transfers** | `/transfers?team={id}` × 20 | 20 | Thursday |
| **Weekly subtotal** | | **81** (spread over 4 days) | |

### Budget Summary
- **Match days** (2-3/week): ~22 requests/day used on daily data
- **Non-match days** (4-5/week): Use remaining budget for weekly enrichment
- **Buffer**: Always keep 10-15 requests in reserve for ad-hoc queries
- **Caching**: All responses should be cached locally as JSON/CSV

### Caching Strategy
```python
import json
import os
from datetime import datetime, timedelta

CACHE_DIR = "data_files/api_cache"

def get_cached_or_fetch(cache_key, endpoint, params, max_age_hours=24):
    """Return cached data if fresh enough, otherwise fetch from API."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_file = os.path.join(CACHE_DIR, f"{cache_key}.json")

    if os.path.exists(cache_file):
        modified = datetime.fromtimestamp(os.path.getmtime(cache_file))
        if datetime.now() - modified < timedelta(hours=max_age_hours):
            with open(cache_file, "r") as f:
                print(f"[CACHE HIT] {cache_key}")
                return json.load(f)

    data = api_get(endpoint, params)
    if data is not None:
        with open(cache_file, "w") as f:
            json.dump(data, f, indent=2)
    return data
```

---

## 9. Implementation Code

### 9a. Core API Client — `fetch_api_football.py`

```python
"""
Fetch data from API-Football v3.
Replaces ESPN fixtures, web-scraped injuries, and hardcoded manager data.
Adds team stats, standings, player data, odds, and predictions.

Usage:
    python fetch_api_football.py --daily        # Daily data refresh
    python fetch_api_football.py --weekly-mon   # Monday: team stats
    python fetch_api_football.py --weekly-tue   # Tuesday: coach data
    python fetch_api_football.py --weekly-wed   # Wednesday: squads + venues
    python fetch_api_football.py --weekly-thu   # Thursday: transfers
    python fetch_api_football.py --prematch     # Pre-match: predictions + odds
    python fetch_api_football.py --status       # Check quota (free)
"""

import requests
import pandas as pd
import json
import os
import sys
import time
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("SPORTS_API_KEY")
BASE_URL = "https://v3.football.api-sports.io"
LEAGUE_ID = 39
CURRENT_SEASON = 2025
HEADERS = {"x-apisports-key": API_KEY}
CACHE_DIR = "data_files/api_cache"
DATA_DIR = "data_files"

os.makedirs(CACHE_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def api_get(endpoint, params=None):
    """Rate-aware GET request. Returns response list or None."""
    url = f"{BASE_URL}/{endpoint}"
    resp = requests.get(url, headers=HEADERS, params=params, timeout=30)
    data = resp.json()

    remaining = resp.headers.get("x-ratelimit-requests-remaining", "?")
    results = data.get("results", 0)
    print(f"  [API] /{endpoint} → {results} results  ({remaining} requests left today)")

    if data.get("errors") and len(data["errors"]) > 0:
        print(f"  [ERROR] {data['errors']}")
        return None
    return data.get("response", [])


def get_cached_or_fetch(cache_key, endpoint, params, max_age_hours=24):
    """Cache-aware fetch. Returns parsed response list."""
    cache_file = os.path.join(CACHE_DIR, f"{cache_key}.json")
    if os.path.exists(cache_file):
        age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(cache_file))
        if age < timedelta(hours=max_age_hours):
            with open(cache_file, "r") as f:
                print(f"  [CACHE] {cache_key} (age: {age})")
                return json.load(f)

    data = api_get(endpoint, params)
    if data is not None:
        with open(cache_file, "w") as f:
            json.dump(data, f, indent=2)
    return data


def check_status():
    """Check account status (free call)."""
    url = f"{BASE_URL}/status"
    resp = requests.get(url, headers=HEADERS, timeout=10)
    info = resp.json().get("response", {})
    req = info.get("requests", {})
    print(f"  Plan: {info.get('subscription', {}).get('plan')}")
    print(f"  Requests today: {req.get('current')} / {req.get('limit_day')}")
    return req.get("current", 0), req.get("limit_day", 100)


# ---------------------------------------------------------------------------
# Team name mapping (API-Football names → your historical data names)
# ---------------------------------------------------------------------------

API_FOOTBALL_TEAM_MAP = {
    "Manchester United": "Man United",
    "Manchester City": "Man City",
    "Wolverhampton Wanderers": "Wolves",
    "Brighton And Hove Albion": "Brighton",
    "Brighton & Hove Albion": "Brighton",
    "Nottingham Forest": "Nott'm Forest",
    "AFC Bournemouth": "Bournemouth",
    "Tottenham Hotspur": "Tottenham",
    "West Ham United": "West Ham",
    "Newcastle United": "Newcastle",
    "Leicester City": "Leicester",
    "Sheffield United": "Sheffield United",
    "Leeds United": "Leeds",
    "Ipswich Town": "Ipswich",
    # Add others as seen in API responses
}

def normalize_team(api_name):
    """Map API-Football team name to historical data team name."""
    return API_FOOTBALL_TEAM_MAP.get(api_name, api_name)


# ---------------------------------------------------------------------------
# Daily data fetchers
# ---------------------------------------------------------------------------

def fetch_standings():
    """Fetch current league table. Cost: 1 request."""
    print("\n📊 Fetching standings...")
    data = get_cached_or_fetch(
        f"standings_{LEAGUE_ID}_{CURRENT_SEASON}",
        "standings",
        {"league": LEAGUE_ID, "season": CURRENT_SEASON},
        max_age_hours=12,
    )
    if not data:
        return None

    rows = []
    for league_entry in data:
        standings = league_entry.get("league", {}).get("standings", [[]])
        for group in standings:
            for team in group:
                rows.append({
                    "TeamAPI": team["team"]["name"],
                    "Team": normalize_team(team["team"]["name"]),
                    "TeamID": team["team"]["id"],
                    "Rank": team["rank"],
                    "Points": team["points"],
                    "Played": team["all"]["played"],
                    "Win": team["all"]["win"],
                    "Draw": team["all"]["draw"],
                    "Lose": team["all"]["lose"],
                    "GoalsFor": team["all"]["goals"]["for"],
                    "GoalsAgainst": team["all"]["goals"]["against"],
                    "GoalDifference": team["goalsDiff"],
                    "Form": team.get("form", ""),
                    "HomeWin": team.get("home", {}).get("win", 0),
                    "HomeDraw": team.get("home", {}).get("draw", 0),
                    "HomeLose": team.get("home", {}).get("lose", 0),
                    "AwayWin": team.get("away", {}).get("win", 0),
                    "AwayDraw": team.get("away", {}).get("draw", 0),
                    "AwayLose": team.get("away", {}).get("lose", 0),
                })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(DATA_DIR, "api_standings.csv"), index=False, sep="\t")
    print(f"  Saved {len(df)} teams to api_standings.csv")
    return df


def fetch_upcoming_fixtures():
    """Fetch next upcoming PL fixtures. Cost: 1 request."""
    print("\n📅 Fetching upcoming fixtures...")
    data = get_cached_or_fetch(
        f"fixtures_next_{LEAGUE_ID}",
        "fixtures",
        {"league": LEAGUE_ID, "next": 15},
        max_age_hours=12,
    )
    if not data:
        return None

    rows = []
    for f in data:
        fixture = f["fixture"]
        rows.append({
            "FixtureID": fixture["id"],
            "Date": fixture["date"],
            "Venue": fixture.get("venue", {}).get("name", ""),
            "VenueCity": fixture.get("venue", {}).get("city", ""),
            "Referee": fixture.get("referee", ""),
            "Status": fixture["status"]["long"],
            "HomeTeamAPI": f["teams"]["home"]["name"],
            "AwayTeamAPI": f["teams"]["away"]["name"],
            "HomeTeam": normalize_team(f["teams"]["home"]["name"]),
            "AwayTeam": normalize_team(f["teams"]["away"]["name"]),
            "HomeTeamID": f["teams"]["home"]["id"],
            "AwayTeamID": f["teams"]["away"]["id"],
            "Round": f.get("league", {}).get("round", ""),
        })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(DATA_DIR, "api_upcoming_fixtures.csv"), index=False, sep="\t")
    print(f"  Saved {len(df)} fixtures to api_upcoming_fixtures.csv")
    return df


def fetch_recent_results():
    """Fetch last 10 completed PL fixtures. Cost: 1 request."""
    print("\n📋 Fetching recent results...")
    data = get_cached_or_fetch(
        f"fixtures_last_{LEAGUE_ID}",
        "fixtures",
        {"league": LEAGUE_ID, "last": 10},
        max_age_hours=6,
    )
    if not data:
        return None

    rows = []
    for f in data:
        fixture = f["fixture"]
        rows.append({
            "FixtureID": fixture["id"],
            "Date": fixture["date"],
            "Referee": fixture.get("referee", ""),
            "HomeTeam": normalize_team(f["teams"]["home"]["name"]),
            "AwayTeam": normalize_team(f["teams"]["away"]["name"]),
            "HomeGoals": f["goals"]["home"],
            "AwayGoals": f["goals"]["away"],
            "HTHomeGoals": f.get("score", {}).get("halftime", {}).get("home"),
            "HTAwayGoals": f.get("score", {}).get("halftime", {}).get("away"),
            "Status": fixture["status"]["long"],
        })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(DATA_DIR, "api_recent_results.csv"), index=False, sep="\t")
    print(f"  Saved {len(df)} results to api_recent_results.csv")
    return df


def fetch_injuries():
    """Fetch current injuries for PL. Cost: 1 request."""
    print("\n🏥 Fetching injuries...")
    data = get_cached_or_fetch(
        f"injuries_{LEAGUE_ID}_{CURRENT_SEASON}",
        "injuries",
        {"league": LEAGUE_ID, "season": CURRENT_SEASON},
        max_age_hours=12,
    )
    if not data:
        return None

    rows = []
    for inj in data:
        rows.append({
            "PlayerID": inj["player"]["id"],
            "PlayerName": inj["player"]["name"],
            "PlayerPhoto": inj["player"]["photo"],
            "Type": inj["player"].get("type", ""),
            "Reason": inj["player"].get("reason", ""),
            "TeamAPI": inj["team"]["name"],
            "Team": normalize_team(inj["team"]["name"]),
            "TeamID": inj["team"]["id"],
            "FixtureID": inj["fixture"]["id"],
            "FixtureDate": inj["fixture"]["date"],
        })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(DATA_DIR, "api_injuries.csv"), index=False, sep="\t")
    print(f"  Saved {len(df)} injury records to api_injuries.csv")
    return df


def fetch_top_players():
    """Fetch top scorers + top assists. Cost: 2 requests."""
    print("\n⭐ Fetching top players...")
    params = {"league": LEAGUE_ID, "season": CURRENT_SEASON}

    scorers = get_cached_or_fetch(
        f"topscorers_{LEAGUE_ID}_{CURRENT_SEASON}",
        "players/topscorers", params, max_age_hours=12
    )
    assists = get_cached_or_fetch(
        f"topassists_{LEAGUE_ID}_{CURRENT_SEASON}",
        "players/topassists", params, max_age_hours=12
    )

    def parse_top_players(raw, category):
        rows = []
        for entry in (raw or []):
            player = entry["player"]
            for stat in entry.get("statistics", []):
                rows.append({
                    "Category": category,
                    "PlayerID": player["id"],
                    "PlayerName": player["name"],
                    "Age": player.get("age"),
                    "Nationality": player.get("nationality"),
                    "TeamAPI": stat["team"]["name"],
                    "Team": normalize_team(stat["team"]["name"]),
                    "Position": stat.get("games", {}).get("position"),
                    "Appearances": stat.get("games", {}).get("appearences"),
                    "Rating": stat.get("games", {}).get("rating"),
                    "Goals": stat.get("goals", {}).get("total"),
                    "Assists": stat.get("goals", {}).get("assists"),
                    "Shots": stat.get("shots", {}).get("total"),
                    "ShotsOnTarget": stat.get("shots", {}).get("on"),
                    "KeyPasses": stat.get("passes", {}).get("key"),
                    "PassAccuracy": stat.get("passes", {}).get("accuracy"),
                    "Tackles": stat.get("tackles", {}).get("total"),
                    "Dribbles": stat.get("dribbles", {}).get("success"),
                    "YellowCards": stat.get("cards", {}).get("yellow"),
                    "RedCards": stat.get("cards", {}).get("red"),
                })
        return rows

    all_rows = parse_top_players(scorers, "TopScorer") + parse_top_players(assists, "TopAssist")
    df = pd.DataFrame(all_rows)
    df.to_csv(os.path.join(DATA_DIR, "api_top_players.csv"), index=False, sep="\t")
    print(f"  Saved {len(df)} top player records to api_top_players.csv")
    return df


# ---------------------------------------------------------------------------
# Weekly data fetchers
# ---------------------------------------------------------------------------

def fetch_team_statistics(team_ids):
    """Fetch season stats for each team. Cost: 1 request per team."""
    print("\n📈 Fetching team statistics...")
    all_rows = []
    for team_id, team_name in team_ids.items():
        data = get_cached_or_fetch(
            f"teamstats_{LEAGUE_ID}_{CURRENT_SEASON}_{team_id}",
            "teams/statistics",
            {"league": LEAGUE_ID, "season": CURRENT_SEASON, "team": team_id},
            max_age_hours=72,
        )
        if not data:
            continue

        # data is a dict (not list) for this endpoint
        stats = data if isinstance(data, dict) else data[0] if data else {}
        if not stats:
            continue

        fixtures = stats.get("fixtures", {})
        goals = stats.get("goals", {})
        clean = stats.get("clean_sheet", {})
        failed = stats.get("failed_to_score", {})
        penalty = stats.get("penalty", {})
        biggest = stats.get("biggest", {})
        lineups = stats.get("lineups", [])
        cards = stats.get("cards", {})

        top_formation = lineups[0]["formation"] if lineups else ""
        form_str = stats.get("form", "")

        all_rows.append({
            "TeamID": team_id,
            "Team": team_name,
            "FormString": form_str,
            "TotalPlayed": fixtures.get("played", {}).get("total", 0),
            "TotalWins": fixtures.get("wins", {}).get("total", 0),
            "TotalDraws": fixtures.get("draws", {}).get("total", 0),
            "TotalLosses": fixtures.get("loses", {}).get("total", 0),
            "HomeWins": fixtures.get("wins", {}).get("home", 0),
            "AwayWins": fixtures.get("wins", {}).get("away", 0),
            "GoalsForTotal": goals.get("for", {}).get("total", {}).get("total", 0),
            "GoalsAgainstTotal": goals.get("against", {}).get("total", {}).get("total", 0),
            "GoalsForAvgHome": goals.get("for", {}).get("average", {}).get("home", 0),
            "GoalsForAvgAway": goals.get("for", {}).get("average", {}).get("away", 0),
            "GoalsAgainstAvgHome": goals.get("against", {}).get("average", {}).get("home", 0),
            "GoalsAgainstAvgAway": goals.get("against", {}).get("average", {}).get("away", 0),
            # Goals by minute
            "GoalsFor_0_15": goals.get("for", {}).get("minute", {}).get("0-15", {}).get("total"),
            "GoalsFor_16_30": goals.get("for", {}).get("minute", {}).get("16-30", {}).get("total"),
            "GoalsFor_31_45": goals.get("for", {}).get("minute", {}).get("31-45", {}).get("total"),
            "GoalsFor_46_60": goals.get("for", {}).get("minute", {}).get("46-60", {}).get("total"),
            "GoalsFor_61_75": goals.get("for", {}).get("minute", {}).get("61-75", {}).get("total"),
            "GoalsFor_76_90": goals.get("for", {}).get("minute", {}).get("76-90", {}).get("total"),
            "GoalsFor_91_105": goals.get("for", {}).get("minute", {}).get("91-105", {}).get("total"),
            "GoalsAgainst_76_90": goals.get("against", {}).get("minute", {}).get("76-90", {}).get("total"),
            "CleanSheetHome": clean.get("home", 0),
            "CleanSheetAway": clean.get("away", 0),
            "CleanSheetTotal": clean.get("total", 0),
            "FailedToScoreHome": failed.get("home", 0),
            "FailedToScoreAway": failed.get("away", 0),
            "PenaltyScored": penalty.get("scored", {}).get("total", 0),
            "PenaltyMissed": penalty.get("missed", {}).get("total", 0),
            "BiggestWinHome": biggest.get("wins", {}).get("home", ""),
            "BiggestWinAway": biggest.get("wins", {}).get("away", ""),
            "BiggestLossHome": biggest.get("loses", {}).get("home", ""),
            "BiggestStreakWins": biggest.get("streak", {}).get("wins", 0),
            "BiggestStreakDraws": biggest.get("streak", {}).get("draws", 0),
            "BiggestStreakLoses": biggest.get("streak", {}).get("loses", 0),
            "MostUsedFormation": top_formation,
        })
        time.sleep(0.5)  # Rate limiting courtesy

    df = pd.DataFrame(all_rows)
    df.to_csv(os.path.join(DATA_DIR, "api_team_statistics.csv"), index=False, sep="\t")
    print(f"  Saved {len(df)} team stat records to api_team_statistics.csv")
    return df


def fetch_coaches(team_ids):
    """Fetch coach data for each team. Cost: 1 request per team."""
    print("\n🧑‍💼 Fetching coaches...")
    all_rows = []
    for team_id, team_name in team_ids.items():
        data = get_cached_or_fetch(
            f"coach_{team_id}",
            "coachs",
            {"team": team_id},
            max_age_hours=168,  # 1 week
        )
        if not data:
            continue
        for coach in data:
            career = coach.get("career", [])
            current_job = next((c for c in career if c.get("end") is None), {})
            all_rows.append({
                "CoachID": coach["id"],
                "CoachName": coach["name"],
                "FirstName": coach.get("firstname", ""),
                "LastName": coach.get("lastname", ""),
                "Age": coach.get("age"),
                "Nationality": coach.get("nationality"),
                "TeamID": team_id,
                "Team": team_name,
                "CurrentStart": current_job.get("start", ""),
                "TotalCareerClubs": len(career),
                "Photo": coach.get("photo", ""),
            })
        time.sleep(0.5)

    df = pd.DataFrame(all_rows)
    df.to_csv(os.path.join(DATA_DIR, "api_coaches.csv"), index=False, sep="\t")
    print(f"  Saved {len(df)} coach records to api_coaches.csv")
    return df


def fetch_squads(team_ids):
    """Fetch current squad for each team. Cost: 1 request per team."""
    print("\n👥 Fetching squads...")
    all_rows = []
    for team_id, team_name in team_ids.items():
        data = get_cached_or_fetch(
            f"squad_{team_id}",
            "players/squads",
            {"team": team_id},
            max_age_hours=168,
        )
        if not data:
            continue
        for team_data in data:
            for player in team_data.get("players", []):
                all_rows.append({
                    "TeamID": team_id,
                    "Team": team_name,
                    "PlayerID": player["id"],
                    "PlayerName": player["name"],
                    "Age": player.get("age"),
                    "Number": player.get("number"),
                    "Position": player.get("position"),
                    "Photo": player.get("photo", ""),
                })
        time.sleep(0.5)

    df = pd.DataFrame(all_rows)
    df.to_csv(os.path.join(DATA_DIR, "api_squads.csv"), index=False, sep="\t")
    print(f"  Saved {len(df)} player records to api_squads.csv")
    return df


def fetch_transfers(team_ids):
    """Fetch transfers for each team. Cost: 1 request per team."""
    print("\n🔄 Fetching transfers...")
    all_rows = []
    for team_id, team_name in team_ids.items():
        data = get_cached_or_fetch(
            f"transfers_{team_id}",
            "transfers",
            {"team": team_id},
            max_age_hours=168,
        )
        if not data:
            continue
        for player_transfers in data:
            player = player_transfers.get("player", {})
            for t in player_transfers.get("transfers", []):
                all_rows.append({
                    "PlayerID": player.get("id"),
                    "PlayerName": player.get("name"),
                    "Date": t.get("date"),
                    "Type": t.get("type"),
                    "TeamFrom": t.get("teams", {}).get("out", {}).get("name", ""),
                    "TeamTo": t.get("teams", {}).get("in", {}).get("name", ""),
                })
        time.sleep(0.5)

    df = pd.DataFrame(all_rows)
    # Filter to recent transfers (current season)
    if not df.empty and "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        season_start = f"{CURRENT_SEASON}-07-01"
        df = df[df["Date"] >= season_start]
    df.to_csv(os.path.join(DATA_DIR, "api_transfers.csv"), index=False, sep="\t")
    print(f"  Saved {len(df)} transfer records to api_transfers.csv")
    return df


# ---------------------------------------------------------------------------
# Pre-match fetchers
# ---------------------------------------------------------------------------

def fetch_predictions(fixture_ids):
    """Fetch API predictions for upcoming fixtures. Cost: 1 per fixture."""
    print("\n🔮 Fetching predictions...")
    all_rows = []
    for fid in fixture_ids:
        data = get_cached_or_fetch(
            f"prediction_{fid}",
            "predictions",
            {"fixture": fid},
            max_age_hours=6,
        )
        if not data:
            continue
        for pred in data:
            predictions = pred.get("predictions", {})
            teams = pred.get("teams", {})
            h2h = pred.get("h2h", [])
            comparison = pred.get("comparison", {})

            all_rows.append({
                "FixtureID": fid,
                "WinnerID": predictions.get("winner", {}).get("id"),
                "WinnerName": predictions.get("winner", {}).get("name"),
                "WinOrDraw": predictions.get("win_or_draw"),
                "UnderOver": predictions.get("under_over"),
                "GoalsHome": predictions.get("goals", {}).get("home"),
                "GoalsAway": predictions.get("goals", {}).get("away"),
                "Advice": predictions.get("advice", ""),
                "HomeFormPct": predictions.get("percent", {}).get("home", ""),
                "DrawPct": predictions.get("percent", {}).get("draw", ""),
                "AwayFormPct": predictions.get("percent", {}).get("away", ""),
                "H2HTotal": len(h2h),
                # Comparison stats
                "CompForm_Home": comparison.get("form", {}).get("home", ""),
                "CompForm_Away": comparison.get("form", {}).get("away", ""),
                "CompAttack_Home": comparison.get("att", {}).get("home", ""),
                "CompAttack_Away": comparison.get("att", {}).get("away", ""),
                "CompDefense_Home": comparison.get("def", {}).get("home", ""),
                "CompDefense_Away": comparison.get("def", {}).get("away", ""),
                "CompTotal_Home": comparison.get("total", {}).get("home", ""),
                "CompTotal_Away": comparison.get("total", {}).get("away", ""),
            })
        time.sleep(0.5)

    df = pd.DataFrame(all_rows)
    df.to_csv(os.path.join(DATA_DIR, "api_predictions.csv"), index=False, sep="\t")
    print(f"  Saved {len(df)} prediction records to api_predictions.csv")
    return df


def fetch_prematch_odds(fixture_ids):
    """Fetch pre-match odds for upcoming fixtures. Cost: 1 per fixture."""
    print("\n💰 Fetching pre-match odds...")
    all_rows = []
    for fid in fixture_ids:
        data = get_cached_or_fetch(
            f"odds_{fid}",
            "odds",
            {"fixture": fid},
            max_age_hours=6,
        )
        if not data:
            continue
        for fixture_odds in data:
            for bookmaker in fixture_odds.get("bookmakers", []):
                bk_name = bookmaker["name"]
                for bet in bookmaker.get("bets", []):
                    bet_name = bet["name"]
                    for val in bet.get("values", []):
                        all_rows.append({
                            "FixtureID": fid,
                            "Bookmaker": bk_name,
                            "BetType": bet_name,
                            "Value": val.get("value"),
                            "Odd": val.get("odd"),
                        })
        time.sleep(0.5)

    df = pd.DataFrame(all_rows)
    df.to_csv(os.path.join(DATA_DIR, "api_prematch_odds.csv"), index=False, sep="\t")
    print(f"  Saved {len(df)} odds records to api_prematch_odds.csv")
    return df


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def get_team_ids_from_standings():
    """Load team IDs from cached standings, or fetch them."""
    standings_file = os.path.join(DATA_DIR, "api_standings.csv")
    if os.path.exists(standings_file):
        df = pd.read_csv(standings_file, sep="\t")
        return dict(zip(df["TeamID"], df["Team"]))
    # Fetch standings first
    df = fetch_standings()
    if df is not None:
        return dict(zip(df["TeamID"], df["Team"]))
    return {}


def run_daily():
    """Daily refresh: standings, fixtures, injuries, top players. ~6 requests."""
    print("=" * 60)
    print(f"DAILY REFRESH — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)
    check_status()
    fetch_standings()
    fetch_upcoming_fixtures()
    fetch_recent_results()
    fetch_injuries()
    fetch_top_players()
    print("\n✅ Daily refresh complete.")


def run_weekly_monday():
    """Monday: team season statistics. ~20 requests."""
    print("=" * 60)
    print(f"WEEKLY (Mon) — Team Stats — {datetime.now().strftime('%Y-%m-%d')}")
    print("=" * 60)
    check_status()
    team_ids = get_team_ids_from_standings()
    if team_ids:
        fetch_team_statistics(team_ids)
    print("\n✅ Monday weekly refresh complete.")


def run_weekly_tuesday():
    """Tuesday: coach data. ~20 requests."""
    print("=" * 60)
    print(f"WEEKLY (Tue) — Coaches — {datetime.now().strftime('%Y-%m-%d')}")
    print("=" * 60)
    check_status()
    team_ids = get_team_ids_from_standings()
    if team_ids:
        fetch_coaches(team_ids)
    print("\n✅ Tuesday weekly refresh complete.")


def run_weekly_wednesday():
    """Wednesday: squads. ~20 requests."""
    print("=" * 60)
    print(f"WEEKLY (Wed) — Squads — {datetime.now().strftime('%Y-%m-%d')}")
    print("=" * 60)
    check_status()
    team_ids = get_team_ids_from_standings()
    if team_ids:
        fetch_squads(team_ids)
    print("\n✅ Wednesday weekly refresh complete.")


def run_weekly_thursday():
    """Thursday: transfers. ~20 requests."""
    print("=" * 60)
    print(f"WEEKLY (Thu) — Transfers — {datetime.now().strftime('%Y-%m-%d')}")
    print("=" * 60)
    check_status()
    team_ids = get_team_ids_from_standings()
    if team_ids:
        fetch_transfers(team_ids)
    print("\n✅ Thursday weekly refresh complete.")


def run_prematch():
    """Pre-match: predictions + odds for upcoming fixtures. ~16 requests."""
    print("=" * 60)
    print(f"PRE-MATCH — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)
    check_status()
    fixtures_file = os.path.join(DATA_DIR, "api_upcoming_fixtures.csv")
    if not os.path.exists(fixtures_file):
        fetch_upcoming_fixtures()
    df = pd.read_csv(fixtures_file, sep="\t")
    fixture_ids = df["FixtureID"].tolist()[:8]  # Limit to next 8 to conserve budget
    if fixture_ids:
        fetch_predictions(fixture_ids)
        fetch_prematch_odds(fixture_ids)
    print("\n✅ Pre-match data complete.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python fetch_api_football.py [--daily|--weekly-mon|--weekly-tue|--weekly-wed|--weekly-thu|--prematch|--status]")
        sys.exit(1)

    cmd = sys.argv[1]
    if cmd == "--status":
        check_status()
    elif cmd == "--daily":
        run_daily()
    elif cmd == "--weekly-mon":
        run_weekly_monday()
    elif cmd == "--weekly-tue":
        run_weekly_tuesday()
    elif cmd == "--weekly-wed":
        run_weekly_wednesday()
    elif cmd == "--weekly-thu":
        run_weekly_thursday()
    elif cmd == "--prematch":
        run_prematch()
    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)
```

### 9b. Feature Engineering Integration — Additions to `prepare_model_data.py`

```python
# Add these functions to prepare_model_data.py to merge API-Football data
# into the feature engineering pipeline.

def merge_standings_features(df):
    """Merge league standings into match data as features."""
    standings_file = "data_files/api_standings.csv"
    if not os.path.exists(standings_file):
        print("  [SKIP] No standings data found")
        return df

    standings = pd.read_csv(standings_file, sep="\t")

    # Home team standings
    home_standings = standings.rename(columns={
        "Rank": "HomeLeaguePosition",
        "Points": "HomeStandingPoints",
        "GoalDifference": "HomeStandingGD",
        "Form": "HomeFormString",
        "CleanSheetHome": "HomeCleanSheetPct_Standing",
    })[["Team", "HomeLeaguePosition", "HomeStandingPoints", "HomeStandingGD", "HomeFormString"]]
    df = df.merge(home_standings, left_on="HomeTeam", right_on="Team", how="left").drop(columns=["Team"], errors="ignore")

    # Away team standings
    away_standings = standings.rename(columns={
        "Rank": "AwayLeaguePosition",
        "Points": "AwayStandingPoints",
        "GoalDifference": "AwayStandingGD",
        "Form": "AwayFormString",
    })[["Team", "AwayLeaguePosition", "AwayStandingPoints", "AwayStandingGD", "AwayFormString"]]
    df = df.merge(away_standings, left_on="AwayTeam", right_on="Team", how="left").drop(columns=["Team"], errors="ignore")

    # Derived features
    df["PositionDifference"] = df["HomeLeaguePosition"] - df["AwayLeaguePosition"]
    df["PointsDifference"] = df["HomeStandingPoints"] - df["AwayStandingPoints"]

    return df


def merge_team_stats_features(df):
    """Merge team season statistics into match data."""
    stats_file = "data_files/api_team_statistics.csv"
    if not os.path.exists(stats_file):
        print("  [SKIP] No team statistics data found")
        return df

    stats = pd.read_csv(stats_file, sep="\t")

    # Calculate derived metrics
    stats["CleanSheetPct"] = stats["CleanSheetTotal"] / stats["TotalPlayed"].clip(lower=1)
    stats["FailedToScorePct"] = (stats["FailedToScoreHome"] + stats["FailedToScoreAway"]) / stats["TotalPlayed"].clip(lower=1)
    stats["PenaltyConversionRate"] = stats["PenaltyScored"] / (stats["PenaltyScored"] + stats["PenaltyMissed"]).clip(lower=1)
    stats["LateGoalsPct"] = stats["GoalsFor_76_90"].fillna(0) / stats["GoalsForTotal"].clip(lower=1)

    # Home team stats
    home_cols = {
        "CleanSheetPct": "HomeCleanSheetPct",
        "FailedToScorePct": "HomeFailedToScorePct",
        "PenaltyConversionRate": "HomePenaltyConversion",
        "LateGoalsPct": "HomeLateGoalsPct",
        "GoalsForAvgHome": "HomeGoalsForAvgHome",
        "GoalsAgainstAvgHome": "HomeGoalsAgainstAvgHome",
        "MostUsedFormation": "HomeMostUsedFormation",
        "BiggestStreakWins": "HomeBiggestWinStreak",
    }
    home_stats = stats.rename(columns=home_cols)[["Team"] + list(home_cols.values())]
    df = df.merge(home_stats, left_on="HomeTeam", right_on="Team", how="left").drop(columns=["Team"], errors="ignore")

    # Away team stats
    away_cols = {
        "CleanSheetPct": "AwayCleanSheetPct",
        "FailedToScorePct": "AwayFailedToScorePct",
        "PenaltyConversionRate": "AwayPenaltyConversion",
        "LateGoalsPct": "AwayLateGoalsPct",
        "GoalsForAvgAway": "AwayGoalsForAvgAway",
        "GoalsAgainstAvgAway": "AwayGoalsAgainstAvgAway",
        "MostUsedFormation": "AwayMostUsedFormation",
        "BiggestStreakWins": "AwayBiggestWinStreak",
    }
    away_stats = stats.rename(columns=away_cols)[["Team"] + list(away_cols.values())]
    df = df.merge(away_stats, left_on="AwayTeam", right_on="Team", how="left").drop(columns=["Team"], errors="ignore")

    return df


def merge_api_predictions_features(df):
    """Merge API prediction percentages into match data for upcoming fixtures."""
    pred_file = "data_files/api_predictions.csv"
    if not os.path.exists(pred_file):
        print("  [SKIP] No API predictions data found")
        return df

    preds = pd.read_csv(pred_file, sep="\t")
    if preds.empty:
        return df

    pred_features = preds.rename(columns={
        "HomeFormPct": "APIPred_HomePct",
        "DrawPct": "APIPred_DrawPct",
        "AwayFormPct": "APIPred_AwayPct",
        "CompAttack_Home": "APIPred_HomeAttack",
        "CompAttack_Away": "APIPred_AwayAttack",
        "CompDefense_Home": "APIPred_HomeDefense",
        "CompDefense_Away": "APIPred_AwayDefense",
    })[["FixtureID", "APIPred_HomePct", "APIPred_DrawPct", "APIPred_AwayPct",
        "APIPred_HomeAttack", "APIPred_AwayAttack", "APIPred_HomeDefense", "APIPred_AwayDefense"]]

    # Convert percentage strings to numeric
    for col in ["APIPred_HomePct", "APIPred_DrawPct", "APIPred_AwayPct"]:
        pred_features[col] = pred_features[col].str.rstrip("%").astype(float) / 100

    df = df.merge(pred_features, on="FixtureID", how="left")
    return df


def merge_injury_features_api(df):
    """Replace web-scraped injury data with API-Football injury data."""
    inj_file = "data_files/api_injuries.csv"
    if not os.path.exists(inj_file):
        print("  [SKIP] No API injury data found")
        return df

    injuries = pd.read_csv(inj_file, sep="\t")
    if injuries.empty:
        return df

    # Count injuries per team
    team_injury_counts = injuries.groupby("Team").agg(
        InjuryCount=("PlayerID", "nunique"),
        MissingPlayers=("PlayerID", lambda x: x.nunique()),
    ).reset_index()

    # Merge for home team
    home_inj = team_injury_counts.rename(columns={
        "InjuryCount": "HomeInjuryCount_API",
        "MissingPlayers": "HomeMissingPlayers",
    })
    df = df.merge(home_inj[["Team", "HomeInjuryCount_API", "HomeMissingPlayers"]],
                  left_on="HomeTeam", right_on="Team", how="left").drop(columns=["Team"], errors="ignore")

    # Merge for away team
    away_inj = team_injury_counts.rename(columns={
        "InjuryCount": "AwayInjuryCount_API",
        "MissingPlayers": "AwayMissingPlayers",
    })
    df = df.merge(away_inj[["Team", "AwayInjuryCount_API", "AwayMissingPlayers"]],
                  left_on="AwayTeam", right_on="Team", how="left").drop(columns=["Team"], errors="ignore")

    df["HomeInjuryCount_API"] = df["HomeInjuryCount_API"].fillna(0)
    df["AwayInjuryCount_API"] = df["AwayInjuryCount_API"].fillna(0)
    df["InjuryAdvantage_API"] = df["AwayInjuryCount_API"] - df["HomeInjuryCount_API"]

    return df
```

### 9c. Model Feature Addition — Example for `premier-league-predictions.py`

```python
# After loading features, add new API-sourced columns to the feature list:
NEW_API_FEATURES = [
    # Standings
    "HomeLeaguePosition", "AwayLeaguePosition", "PositionDifference", "PointsDifference",
    # Team stats
    "HomeCleanSheetPct", "AwayCleanSheetPct",
    "HomeFailedToScorePct", "AwayFailedToScorePct",
    "HomeLateGoalsPct", "AwayLateGoalsPct",
    "HomePenaltyConversion", "AwayPenaltyConversion",
    # API predictions as features
    "APIPred_HomePct", "APIPred_DrawPct", "APIPred_AwayPct",
    # Injuries from API
    "HomeInjuryCount_API", "AwayInjuryCount_API", "InjuryAdvantage_API",
]

# Add to existing feature selection:
# available_features = [f for f in NEW_API_FEATURES if f in X.columns]
# feature_cols = existing_features + available_features
```

---

## 10. Phased Rollout Plan

### Phase 1: Foundation (Week 1)
**Goal**: Set up API client, replace fragile scrapers
**Requests/day**: ~6-10

| Task | Details |
|------|---------|
| Create `fetch_api_football.py` | Core client with caching (code in §9a) |
| Run `--daily` to test | Verify fixtures, standings, injuries |
| Update `team_name_mapping.py` | Add API-Football team name mappings |
| Replace ESPN fixtures | Switch `upcoming_fixtures.csv` source to API |
| Replace injury scraping | Switch from web scraping to `/injuries` endpoint |

### Phase 2: Team Intelligence (Week 2)
**Goal**: Add team statistics and standings features
**Requests/day**: ~20 on Mon, ~6 other days

| Task | Details |
|------|---------|
| Run `--weekly-mon` | Fetch all 20 team stats |
| Add `merge_standings_features()` | Integrate into `prepare_model_data.py` |
| Add `merge_team_stats_features()` | Clean sheet %, late goals, formation |
| Retrain models | Evaluate impact of new features |

### Phase 3: Players & Predictions (Week 3)
**Goal**: Add player data and API predictions
**Requests/day**: ~22 on match days

| Task | Details |
|------|---------|
| Run `--prematch` before match days | Get odds + predictions |
| Add `merge_api_predictions_features()` | Use API preds as ensemble signal |
| Fetch top scorers/assists daily | Build player importance index |
| Replace hardcoded manager data | Use `/coachs` endpoint |

### Phase 4: Advanced Features (Week 4+)
**Goal**: Formations, transfers, squad strength
**Requests/day**: ~20 on weekly days

| Task | Details |
|------|---------|
| Fetch squads + transfers weekly | Squad composition tracking |
| Build player importance scoring | Rating-weighted injury impact |
| Formation matchup analysis | Historical formation performance |
| Streamlit UI enhancements | New tabs: Team Deep Dive, Match Preview |

---

## Appendix A: Quick Reference — API-Football IDs

### Premier League
- **League ID**: 39
- **Country Code**: GB-ENG
- **Current Season**: 2025

### Key Team IDs (from standings)
Run `python fetch_api_football.py --daily` to populate `api_standings.csv` with all current team IDs.

Common team IDs from API-Football:
| Team | API ID |
|------|--------|
| Arsenal | 42 |
| Aston Villa | 66 |
| Bournemouth | 35 |
| Brentford | 55 |
| Brighton | 51 |
| Chelsea | 49 |
| Crystal Palace | 52 |
| Everton | 45 |
| Fulham | 36 |
| Ipswich | 57 |
| Leicester | 46 |
| Liverpool | 40 |
| Man City | 50 |
| Man United | 33 |
| Newcastle | 34 |
| Nott'm Forest | 65 |
| Southampton | 41 |
| Tottenham | 47 |
| West Ham | 48 |
| Wolves | 39 |

> **Note**: Verify these IDs by running the standings fetch — squad composition changes each season with promotion/relegation.

## Appendix B: Sample API Responses

### Standings Response (key fields)
```json
{
  "rank": 1,
  "team": {"id": 50, "name": "Manchester City"},
  "points": 72,
  "goalsDiff": 45,
  "form": "WWDWW",
  "all": {"played": 30, "win": 22, "draw": 6, "lose": 2, "goals": {"for": 68, "against": 23}},
  "home": {"played": 15, "win": 13, "draw": 2, "lose": 0},
  "away": {"played": 15, "win": 9, "draw": 4, "lose": 2}
}
```

### Team Statistics Response (key fields)
```json
{
  "form": "WDLDWLDLDWLWDDWWDLWWLWLLDWWDWDWWWWDWDW",
  "fixtures": {"played": {"home": 19, "away": 19, "total": 38}, "wins": {"home": 16, "away": 12, "total": 28}},
  "goals": {
    "for": {"total": {"home": 57, "away": 45, "total": 102}, "average": {"home": "3.0", "away": "2.4"}},
    "against": {"total": {"home": 12, "away": 23, "total": 35}},
    "for": {"minute": {"0-15": {"total": 10}, "76-90": {"total": 18}}}
  },
  "clean_sheet": {"home": 12, "away": 6, "total": 18},
  "penalty": {"scored": {"total": 8}, "missed": {"total": 2}},
  "lineups": [{"formation": "4-3-3", "played": 25}]
}
```

### Injury Response (key fields)
```json
{
  "player": {"id": 18846, "name": "Bukayo Saka", "type": "Missing Fixture", "reason": "Hamstring Injury"},
  "team": {"id": 42, "name": "Arsenal"},
  "fixture": {"id": 1035337, "date": "2025-03-22T15:00:00+00:00"}
}
```

### Prediction Response (key fields)
```json
{
  "predictions": {
    "winner": {"id": 50, "name": "Manchester City"},
    "win_or_draw": true,
    "under_over": "-3.5",
    "goals": {"home": "-2.5", "away": "-1.5"},
    "advice": "Manchester City or draw and target -3.5 goals",
    "percent": {"home": "65%", "draw": "20%", "away": "15%"}
  },
  "comparison": {
    "form": {"home": "80%", "away": "60%"},
    "att": {"home": "85%", "away": "70%"},
    "def": {"home": "90%", "away": "65%"},
    "total": {"home": "85%", "away": "65%"}
  }
}
```

---

## Appendix C: python-dotenv Setup

Ensure your `.env` file has:
```
SPORTS_API_KEY=your_key_here
```

And `python-dotenv` is in `requirements.txt`:
```
python-dotenv
```

Load in any script:
```python
from dotenv import load_dotenv
load_dotenv()
API_KEY = os.getenv("SPORTS_API_KEY")
```
