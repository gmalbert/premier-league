"""
Fetch data from API-Football v3.
Replaces ESPN fixtures, web-scraped injuries, and hardcoded manager data.
Adds team stats, standings, player data, odds, and predictions.

Usage:
    python fetch_api_football.py --daily        # Daily data refresh (~6 requests)
    python fetch_api_football.py --weekly-mon   # Monday: team stats (~20 requests)
    python fetch_api_football.py --weekly-tue   # Tuesday: coach data (~20 requests)
    python fetch_api_football.py --weekly-wed   # Wednesday: squads (~20 requests)
    python fetch_api_football.py --weekly-thu   # Thursday: transfers (~20 requests)
    python fetch_api_football.py --prematch     # Pre-match: predictions + odds (~16 requests)
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
from team_name_mapping import TEAM_NAME_MAP

load_dotenv()

API_KEY = os.getenv("SPORTS_API_KEY")
BASE_URL = "https://v3.football.api-sports.io"
LEAGUE_ID = 39
# Free plan allows 2022-2024 only; paid plan can set this to newer seasons.
CURRENT_SEASON = None  # will be detected at runtime
HEADERS = {"x-apisports-key": API_KEY}
CACHE_DIR = "data_files/api_cache"
DATA_DIR = "data_files"

os.makedirs(CACHE_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Team name mapping (API-Football → historical data)
# Extends the existing team_name_mapping with API-Football specific names
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
}
# Merge with existing mapping (prefer API-Football-specific names)
for k, v in TEAM_NAME_MAP.items():
    if k not in API_FOOTBALL_TEAM_MAP:
        API_FOOTBALL_TEAM_MAP[k] = v


def normalize_team(api_name):
    """Map API-Football team name to historical data team name."""
    return API_FOOTBALL_TEAM_MAP.get(api_name, api_name)


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
    print(f"  [API] /{endpoint} -> {results} results  ({remaining} requests left today)")

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


def detect_latest_season(min_season=2022, max_season=None):
    """Detect latest available season from API-Football for the league."""
    if max_season is None:
        max_season = datetime.now().year + 1

    # Keep candidate range sane (API free plan limit).
    max_season = min(max_season, 2026)

    for season in range(max_season, min_season - 1, -1):
        cache_key = f"season_check_{LEAGUE_ID}_{season}"
        print(f"Checking API availability for season {season}...")
        try:
            response = api_get("standings", {"league": LEAGUE_ID, "season": season})
        except Exception as e:
            print(f"  Season check failed for {season}: {e}")
            continue

        if response:
            try:
                # Standings response should be list with league object containing standings lists
                for entry in response:
                    standings = entry.get("league", {}).get("standings", [])
                    if standings and any(len(group) for group in standings):
                        print(f"  Season {season} is available.")
                        return season
            except Exception:
                pass

        time.sleep(2)

    print("Could not auto-detect current season; defaulting to 2024.")
    return 2024


def check_status():
    """Check account status (free call)."""
    url = f"{BASE_URL}/status"
    resp = requests.get(url, headers=HEADERS, timeout=10)
    raw = resp.json().get("response", {})
    # API returns a list for some accounts, a dict for others
    info = raw[0] if isinstance(raw, list) and raw else raw if isinstance(raw, dict) else {}
    req = info.get("requests", {})
    current = req.get("current", 0)
    limit = req.get("limit_day", 100)
    print(f"\n  Plan: {info.get('subscription', {}).get('plan')}")
    print(f"  Requests today: {current} / {limit}")
    print(f"  Remaining: {limit - current}\n")
    return current, limit


# ---------------------------------------------------------------------------
# Daily data fetchers
# ---------------------------------------------------------------------------

def fetch_standings():
    """Fetch current league table. Cost: 1 request."""
    print("\nFetching standings...")
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
    """Fetch next upcoming PL fixtures using date range (free plan compatible). Cost: 1 request."""
    print("\nFetching upcoming fixtures...")
    today = datetime.now().strftime("%Y-%m-%d")
    future = (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d")
    data = get_cached_or_fetch(
        f"fixtures_next_{LEAGUE_ID}_{today}",
        "fixtures",
        {"league": LEAGUE_ID, "season": CURRENT_SEASON, "from": today, "to": future, "status": "NS"},
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
    """Fetch last 10 completed PL fixtures using date range. Cost: 1 request."""
    print("\nFetching recent results...")
    today = datetime.now().strftime("%Y-%m-%d")
    past = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
    data = get_cached_or_fetch(
        f"fixtures_last_{LEAGUE_ID}_{today}",
        "fixtures",
        {"league": LEAGUE_ID, "season": CURRENT_SEASON, "from": past, "to": today, "status": "FT"},
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
    print("\nFetching injuries...")
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
    print("\nFetching top players...")
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
    print("\nFetching team statistics...")
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
        time.sleep(7)  # Free plan: max 10 requests/min

    df = pd.DataFrame(all_rows)
    df.to_csv(os.path.join(DATA_DIR, "api_team_statistics.csv"), index=False, sep="\t")
    print(f"  Saved {len(df)} team stat records to api_team_statistics.csv")
    return df


def fetch_coaches(team_ids):
    """Fetch coach data for each team. Cost: 1 request per team."""
    print("\nFetching coaches...")
    all_rows = []
    for team_id, team_name in team_ids.items():
        data = get_cached_or_fetch(
            f"coach_{team_id}",
            "coachs",
            {"team": team_id},
            max_age_hours=168,
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
            })
        time.sleep(7)  # Free plan: max 10 requests/min

    df = pd.DataFrame(all_rows)
    df.to_csv(os.path.join(DATA_DIR, "api_coaches.csv"), index=False, sep="\t")
    print(f"  Saved {len(df)} coach records to api_coaches.csv")
    return df


def fetch_squads(team_ids):
    """Fetch current squad for each team. Cost: 1 request per team."""
    print("\nFetching squads...")
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
                })
        time.sleep(7)  # Free plan: max 10 requests/min

    df = pd.DataFrame(all_rows)
    df.to_csv(os.path.join(DATA_DIR, "api_squads.csv"), index=False, sep="\t")
    print(f"  Saved {len(df)} player records to api_squads.csv")
    return df


def fetch_transfers(team_ids):
    """Fetch transfers for each team. Cost: 1 request per team."""
    print("\nFetching transfers...")
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
        time.sleep(7)  # Free plan: 10 req/min

    df = pd.DataFrame(all_rows)
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
    print("\nFetching predictions...")
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
            comparison = pred.get("comparison", {})
            h2h = pred.get("h2h", [])

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
                "CompForm_Home": comparison.get("form", {}).get("home", ""),
                "CompForm_Away": comparison.get("form", {}).get("away", ""),
                "CompAttack_Home": comparison.get("att", {}).get("home", ""),
                "CompAttack_Away": comparison.get("att", {}).get("away", ""),
                "CompDefense_Home": comparison.get("def", {}).get("home", ""),
                "CompDefense_Away": comparison.get("def", {}).get("away", ""),
                "CompTotal_Home": comparison.get("total", {}).get("home", ""),
                "CompTotal_Away": comparison.get("total", {}).get("away", ""),
            })
        time.sleep(7)  # Free plan: 10 req/min

    df = pd.DataFrame(all_rows)
    df.to_csv(os.path.join(DATA_DIR, "api_predictions.csv"), index=False, sep="\t")
    print(f"  Saved {len(df)} prediction records to api_predictions.csv")
    return df


def fetch_prematch_odds(fixture_ids):
    """Fetch pre-match odds for upcoming fixtures. Cost: 1 per fixture."""
    print("\nFetching pre-match odds...")
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
        time.sleep(7)  # Free plan: 10 req/min

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
    df = fetch_standings()
    if df is not None and not df.empty:
        return dict(zip(df["TeamID"], df["Team"]))
    return {}


def run_daily():
    """Daily refresh: standings, fixtures, injuries, top players. ~6 requests."""
    print("=" * 60)
    print(f"DAILY REFRESH - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)
    check_status()
    fetch_standings()
    fetch_upcoming_fixtures()
    fetch_recent_results()
    fetch_injuries()
    fetch_top_players()
    used, limit = check_status()
    print(f"\nDaily refresh complete. Used {used}/{limit} requests today.")


def run_weekly_monday():
    """Monday: team season statistics. ~20 requests."""
    print("=" * 60)
    print(f"WEEKLY (Mon) - Team Stats - {datetime.now().strftime('%Y-%m-%d')}")
    print("=" * 60)
    check_status()
    team_ids = get_team_ids_from_standings()
    if team_ids:
        fetch_team_statistics(team_ids)
    used, limit = check_status()
    print(f"\nMonday refresh complete. Used {used}/{limit} requests today.")


def run_weekly_tuesday():
    """Tuesday: coach data. ~20 requests."""
    print("=" * 60)
    print(f"WEEKLY (Tue) - Coaches - {datetime.now().strftime('%Y-%m-%d')}")
    print("=" * 60)
    check_status()
    team_ids = get_team_ids_from_standings()
    if team_ids:
        fetch_coaches(team_ids)
    used, limit = check_status()
    print(f"\nTuesday refresh complete. Used {used}/{limit} requests today.")


def run_weekly_wednesday():
    """Wednesday: squads. ~20 requests."""
    print("=" * 60)
    print(f"WEEKLY (Wed) - Squads - {datetime.now().strftime('%Y-%m-%d')}")
    print("=" * 60)
    check_status()
    team_ids = get_team_ids_from_standings()
    if team_ids:
        fetch_squads(team_ids)
    used, limit = check_status()
    print(f"\nWednesday refresh complete. Used {used}/{limit} requests today.")


def run_weekly_thursday():
    """Thursday: transfers. ~20 requests."""
    print("=" * 60)
    print(f"WEEKLY (Thu) - Transfers - {datetime.now().strftime('%Y-%m-%d')}")
    print("=" * 60)
    check_status()
    team_ids = get_team_ids_from_standings()
    if team_ids:
        fetch_transfers(team_ids)
    used, limit = check_status()
    print(f"\nThursday refresh complete. Used {used}/{limit} requests today.")


def run_prematch():
    """Pre-match: predictions + odds for upcoming fixtures. ~16 requests."""
    print("=" * 60)
    print(f"PRE-MATCH - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)
    check_status()
    fixtures_file = os.path.join(DATA_DIR, "api_upcoming_fixtures.csv")
    if not os.path.exists(fixtures_file):
        fetch_upcoming_fixtures()
    df = pd.read_csv(fixtures_file, sep="\t")
    fixture_ids = df["FixtureID"].tolist()[:8]  # Limit to 8 to conserve budget
    if fixture_ids:
        fetch_predictions(fixture_ids)
        fetch_prematch_odds(fixture_ids)
    used, limit = check_status()
    print(f"\nPre-match data complete. Used {used}/{limit} requests today.")


# Auto-detect the latest season if not explicitly configured.
if CURRENT_SEASON is None:
    CURRENT_SEASON = detect_latest_season()
print(f"Using CURRENT_SEASON={CURRENT_SEASON}")

if __name__ == "__main__":
    if not API_KEY:
        print("ERROR: SPORTS_API_KEY not found in .env file")
        sys.exit(1)

    if len(sys.argv) < 2:
        print("Usage: python fetch_api_football.py [--daily|--weekly-mon|--weekly-tue|--weekly-wed|--weekly-thu|--prematch|--status]")
        sys.exit(1)

    cmd = sys.argv[1]
    commands = {
        "--status": lambda: check_status(),
        "--daily": run_daily,
        "--weekly-mon": run_weekly_monday,
        "--weekly-tue": run_weekly_tuesday,
        "--weekly-wed": run_weekly_wednesday,
        "--weekly-thu": run_weekly_thursday,
        "--prematch": run_prematch,
    }

    if cmd in commands:
        commands[cmd]()
    else:
        print(f"Unknown command: {cmd}")
        print("Available: " + ", ".join(commands.keys()))
        sys.exit(1)
