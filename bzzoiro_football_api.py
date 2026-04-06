"""
bzzoiro_football_api.py
-----------------------
Client for the Bzzoiro Sports Data API (https://sports.bzzoiro.com/api/).

Free, unlimited, no rate limits.
Auth: Authorization: Token <BZZOIRO_KEY>

Key public functions
--------------------
  get_pl_league_id()                          → int
  get_pl_events(date_from, date_to, status)   → list[dict]
  get_event_detail(event_id)                  → dict  (includes shotmap, momentum, avg positions)
  get_live_matches()                          → list[dict]
  get_pl_predictions(date_from, date_to)      → list[dict]
  get_pl_players(team_id)                     → list[dict]
  get_player_stats(player_id, event_id, team_id) → list[dict]

Data collection
---------------
  backfill_shotmap_data(date_from, date_to, output_path)
  backfill_player_stats(output_path)
  fetch_squad_values(output_path)
  fetch_upcoming_predictions(output_path)

Feature engineering
-------------------
  compute_shotmap_features(shotmap_csv)  → DataFrame
  compute_player_stat_features(stats_csv) → DataFrame
"""

from __future__ import annotations

import json
import os
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import numpy as np
import requests
from dotenv import load_dotenv

load_dotenv()

# ── Constants ──────────────────────────────────────────────────────────────────
_BASE = "https://sports.bzzoiro.com/api"
_IMG_BASE = "https://sports.bzzoiro.com/img"
_CACHE_DIR = Path("cache/bzzoiro")

# TTLs in seconds (match server-side cache)
_TTL = {
    "live": 30,
    "events": 120,
    "predictions": 120,
    "players": 300,
    "player_stats": 300,
    "leagues": 300,
    "teams": 300,
}

# Premier League league ID on Bzzoiro (api_id=17)
_PL_LEAGUE_ID = None  # discovered at runtime


# ── Team name mapping ─────────────────────────────────────────────────────────
# Bzzoiro uses full names; our historical data uses abbreviated names.

BZZOIRO_TEAM_MAP = {
    "Manchester United": "Man United",
    "Manchester City": "Man City",
    "Wolverhampton Wanderers": "Wolves",
    "Brighton And Hove Albion": "Brighton",
    "Brighton and Hove Albion": "Brighton",
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
    "Luton Town": "Luton",
    "Burnley FC": "Burnley",
    "Burnley": "Burnley",
}


def normalize_bzzoiro_team(name: str) -> str:
    """Map a Bzzoiro team name to the historical data format."""
    return BZZOIRO_TEAM_MAP.get(name, name)


# ── Key loading ────────────────────────────────────────────────────────────────

def _load_key() -> str:
    """Load BZZOIRO_KEY from environment or .env file."""
    key = os.environ.get("BZZOIRO_KEY", "")
    if key:
        return key

    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("BZZOIRO_KEY="):
                key = line.split("=", 1)[1].strip().strip('"').strip("'")
                if key:
                    return key

    try:
        import streamlit as st
        key = st.secrets.get("BZZOIRO_KEY", "")
        if key:
            return key
    except Exception:
        pass

    raise RuntimeError(
        "BZZOIRO_KEY not found. Add it to .env or set it as an environment variable."
    )


# ── Cache helpers ──────────────────────────────────────────────────────────────

def _cache_path(name: str) -> Path:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return _CACHE_DIR / name


def _load_cached(path: Path, ttl: int):
    if not path.exists():
        return None
    age = time.time() - path.stat().st_mtime
    if age > ttl:
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _save_cache(path: Path, data) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


# ── Core HTTP ──────────────────────────────────────────────────────────────────

def _get(endpoint: str, params: dict | None = None) -> dict:
    """Single authenticated GET; raises on HTTP errors."""
    headers = {"Authorization": f"Token {_load_key()}"}
    url = f"{_BASE}/{endpoint.strip('/')}/"
    resp = requests.get(url, headers=headers, params=params or {}, timeout=15)

    if resp.status_code >= 500:
        raise RuntimeError(
            f"Bzzoiro API temporarily unavailable (HTTP {resp.status_code})."
        )
    if resp.status_code in (401, 403):
        raise RuntimeError(
            f"Bzzoiro API authentication failed (HTTP {resp.status_code}). "
            "Check BZZOIRO_KEY in .env."
        )
    resp.raise_for_status()
    return resp.json()


def _get_all_pages(endpoint: str, params: dict | None = None) -> list:
    """Fetch all pages of a paginated endpoint and return flat results list."""
    params = dict(params or {})
    results: list = []
    page = 1
    while True:
        params["page"] = page
        data = _get(endpoint, params)
        results.extend(data.get("results", []))
        if not data.get("next"):
            break
        page += 1
    return results


# ── League discovery ───────────────────────────────────────────────────────────

def get_pl_league_id() -> int:
    """Return the Bzzoiro internal league ID for the Premier League."""
    global _PL_LEAGUE_ID
    if _PL_LEAGUE_ID is not None:
        return _PL_LEAGUE_ID

    cache_key = "leagues.json"
    cached = _load_cached(_cache_path(cache_key), _TTL["leagues"])
    if cached is None:
        cached = _get_all_pages("leagues")
        _save_cache(_cache_path(cache_key), cached)

    for league in cached:
        if league.get("name") == "Premier League" and league.get("country") == "England":
            _PL_LEAGUE_ID = league["id"]
            return _PL_LEAGUE_ID

    raise RuntimeError("Premier League not found in Bzzoiro leagues list.")


# ── Public API: Events ─────────────────────────────────────────────────────────

def get_pl_events(
    date_from: str | None = None,
    date_to: str | None = None,
    status: str | None = None,
) -> list[dict]:
    """
    Fetch Premier League matches.

    Parameters
    ----------
    date_from : YYYY-MM-DD (default: today)
    date_to   : YYYY-MM-DD (default: today + 7 days)
    status    : notstarted, inprogress, finished, etc. (optional)
    """
    league_id = get_pl_league_id()
    d_from = date_from or date.today().isoformat()
    d_to = date_to or (date.today() + timedelta(days=7)).isoformat()

    cache_key = f"events_{d_from}_{d_to}_{status or 'all'}.json"
    cached = _load_cached(_cache_path(cache_key), _TTL["events"])
    if cached is not None:
        return cached

    params = {"league": league_id, "date_from": d_from, "date_to": d_to}
    if status:
        params["status"] = status
    # Use Europe/London for UK-local date boundaries
    params["tz"] = "Europe/London"

    results = _get_all_pages("events", params)
    _save_cache(_cache_path(cache_key), results)
    return results


def get_event_detail(event_id: int) -> dict:
    """
    Fetch a single event with full spatial data.

    Returns shotmap, momentum, average_positions, incidents (with goal sequences).
    """
    cache_key = f"event_{event_id}.json"
    cached = _load_cached(_cache_path(cache_key), _TTL["events"])
    if cached is not None:
        return cached

    data = _get(f"events/{event_id}")
    _save_cache(_cache_path(cache_key), data)
    return data


# ── Public API: Live ───────────────────────────────────────────────────────────

def get_live_matches() -> list[dict]:
    """Return currently live PL matches (no disk cache — always fresh)."""
    league_id = get_pl_league_id()
    results = _get_all_pages("live")
    return [
        m for m in results
        if (m.get("league") or {}).get("id") == league_id
    ]


# ── Public API: Predictions ───────────────────────────────────────────────────

def get_pl_predictions(
    date_from: str | None = None,
    date_to: str | None = None,
    upcoming_only: bool = True,
) -> list[dict]:
    """
    Return CatBoost ML predictions for PL matches.

    Each item includes:
        prob_home_win, prob_draw, prob_away_win, predicted_result,
        expected_home_goals, expected_away_goals,
        prob_over_15, prob_over_25, prob_over_35, prob_btts_yes,
        confidence, most_likely_score, favorite, favorite_recommend,
        over_25_recommend, btts_recommend, winner_recommend
    """
    league_id = get_pl_league_id()
    d_from = date_from or date.today().isoformat()
    d_to = date_to or (date.today() + timedelta(days=14)).isoformat()

    cache_key = f"predictions_{d_from}_{d_to}.json"
    cached = _load_cached(_cache_path(cache_key), _TTL["predictions"])
    if cached is not None:
        return cached

    params: dict = {"league": league_id, "tz": "Europe/London"}
    if date_from or date_to:
        if date_from:
            params["date_from"] = date_from
        if date_to:
            params["date_to"] = date_to
    else:
        params["upcoming"] = str(upcoming_only).lower()

    results = _get_all_pages("predictions", params)
    _save_cache(_cache_path(cache_key), results)
    return results


# ── Public API: Players ───────────────────────────────────────────────────────

def get_pl_teams() -> list[dict]:
    """Return all teams in the Premier League."""
    league_id = get_pl_league_id()
    cache_key = "pl_teams.json"
    cached = _load_cached(_cache_path(cache_key), _TTL["teams"])
    if cached is not None:
        return cached

    results = _get_all_pages("teams", {"league": league_id})
    _save_cache(_cache_path(cache_key), results)
    return results


def get_pl_players(team_id: int | None = None) -> list[dict]:
    """
    Return players. Filter by team_id for a specific squad.

    Each player includes: name, position (G/D/M/F), jersey_number, height,
    date_of_birth, nationality, market_value, current_team, national_team.
    """
    params: dict = {}
    if team_id:
        params["team"] = team_id
    cache_key = f"players_{team_id or 'all'}.json"
    cached = _load_cached(_cache_path(cache_key), _TTL["players"])
    if cached is not None:
        return cached

    results = _get_all_pages("players", params)
    _save_cache(_cache_path(cache_key), results)
    return results


# ── Public API: Player Stats ──────────────────────────────────────────────────

def get_player_stats(
    player_id: int | None = None,
    event_id: int | None = None,
    team_id: int | None = None,
) -> list[dict]:
    """
    Return per-match player statistics.

    Filter by player, event (match), or team.
    30+ metrics: xG, xA, passes, duels, tackles, rating, etc.
    """
    params: dict = {}
    if player_id:
        params["player"] = player_id
    if event_id:
        params["event"] = event_id
    if team_id:
        params["team"] = team_id

    cache_key = f"player_stats_{player_id or ''}_{event_id or ''}_{team_id or ''}.json"
    cached = _load_cached(_cache_path(cache_key), _TTL["player_stats"])
    if cached is not None:
        return cached

    results = _get_all_pages("player-stats", params)
    _save_cache(_cache_path(cache_key), results)
    return results


# ── Image URLs ─────────────────────────────────────────────────────────────────

def team_logo_url(api_id: int) -> str:
    """Return public URL for a team logo (no auth required)."""
    return f"{_IMG_BASE}/team/{api_id}/"


def player_photo_url(api_id: int) -> str:
    """Return public URL for a player photo (no auth required)."""
    return f"{_IMG_BASE}/player/{api_id}/"


def league_logo_url(api_id: int = 17) -> str:
    """Return public URL for a league logo (PL api_id=17 by default)."""
    return f"{_IMG_BASE}/league/{api_id}/"


# ═══════════════════════════════════════════════════════════════════════════════
# DATA COLLECTION SCRIPTS
# ═══════════════════════════════════════════════════════════════════════════════

def backfill_actual_xg(
    date_from: str = "2024-08-01",
    date_to: str | None = None,
    output_path: str = "data_files/bzzoiro_actual_xg.csv",
) -> pd.DataFrame:
    """
    Quick backfill of actual xG + odds for all finished PL matches.

    Faster than backfill_shotmap_data because it only fetches event detail
    (no shotmap/momentum processing needed). Produces one row per match.

    Columns: event_id, match_date, home_team, away_team, home_score, away_score,
    home_score_ht, away_score_ht, actual_home_xg, actual_away_xg,
    odds_home, odds_draw, odds_away, odds_over_25, odds_btts_yes, round_number
    """
    if date_to is None:
        date_to = date.today().isoformat()

    print(f"Fetching finished PL events from {date_from} to {date_to}...")
    events = get_pl_events(date_from=date_from, date_to=date_to, status="finished")
    print(f"Found {len(events)} finished matches. Fetching xG data...")

    rows = []
    for i, evt in enumerate(events):
        event_id = evt["id"]
        home_team = normalize_bzzoiro_team(evt.get("home_team", ""))
        away_team = normalize_bzzoiro_team(evt.get("away_team", ""))
        match_date = evt.get("event_date", "")[:10]

        detail = get_event_detail(event_id)

        rows.append({
            "event_id": event_id,
            "match_date": match_date,
            "home_team": home_team,
            "away_team": away_team,
            "home_score": evt.get("home_score"),
            "away_score": evt.get("away_score"),
            "home_score_ht": detail.get("home_score_ht"),
            "away_score_ht": detail.get("away_score_ht"),
            "actual_home_xg": detail.get("actual_home_xg"),
            "actual_away_xg": detail.get("actual_away_xg"),
            "odds_home": detail.get("odds_home"),
            "odds_draw": detail.get("odds_draw"),
            "odds_away": detail.get("odds_away"),
            "odds_over_15": detail.get("odds_over_15"),
            "odds_over_25": detail.get("odds_over_25"),
            "odds_over_35": detail.get("odds_over_35"),
            "odds_btts_yes": detail.get("odds_btts_yes"),
            "odds_btts_no": detail.get("odds_btts_no"),
            "round_number": detail.get("round_number"),
        })

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(events)}] processed...")

    df = pd.DataFrame(rows)
    if not df.empty:
        df.to_csv(output_path, index=False, sep="\t")
        print(f"Saved {len(df)} match xG records to {output_path}")
    return df


def backfill_shotmap_data(
    date_from: str = "2024-08-01",
    date_to: str | None = None,
    output_path: str = "data_files/bzzoiro_shotmap.csv",
) -> pd.DataFrame:
    """
    Fetch shotmap data for all finished PL matches in the date range.

    Each row = one shot. Columns: event_id, match_date, home_team, away_team,
    minute, type, situation, body_part, is_home, xg, xgot, pos_x, pos_y,
    gm_y, gm_z, gm_location, player_id.

    Also collects momentum, average_positions, and actual xG into separate files.
    """
    if date_to is None:
        date_to = date.today().isoformat()

    print(f"Fetching finished PL events from {date_from} to {date_to}...")
    events = get_pl_events(date_from=date_from, date_to=date_to, status="finished")
    print(f"Found {len(events)} finished matches.")

    shot_rows = []
    momentum_rows = []
    position_rows = []
    xg_rows = []

    for i, evt in enumerate(events):
        event_id = evt["id"]
        home_team = normalize_bzzoiro_team(evt.get("home_team", ""))
        away_team = normalize_bzzoiro_team(evt.get("away_team", ""))
        match_date = evt.get("event_date", "")[:10]

        print(f"  [{i+1}/{len(events)}] {home_team} vs {away_team} ({match_date})...")
        detail = get_event_detail(event_id)

        # Actual xG (confirmed working for PL matches)
        xg_rows.append({
            "event_id": event_id,
            "match_date": match_date,
            "home_team": home_team,
            "away_team": away_team,
            "home_score": evt.get("home_score"),
            "away_score": evt.get("away_score"),
            "home_score_ht": detail.get("home_score_ht"),
            "away_score_ht": detail.get("away_score_ht"),
            "actual_home_xg": detail.get("actual_home_xg"),
            "actual_away_xg": detail.get("actual_away_xg"),
            "odds_home": detail.get("odds_home"),
            "odds_draw": detail.get("odds_draw"),
            "odds_away": detail.get("odds_away"),
            "odds_over_25": detail.get("odds_over_25"),
            "odds_btts_yes": detail.get("odds_btts_yes"),
            "round_number": detail.get("round_number"),
        })

        # Shotmap
        for shot in (detail.get("shotmap") or []):
            pos = shot.get("pos") or {}
            gm = shot.get("gm") or {}
            shot_rows.append({
                "event_id": event_id,
                "match_date": match_date,
                "home_team": home_team,
                "away_team": away_team,
                "minute": shot.get("min"),
                "type": shot.get("type"),
                "situation": shot.get("sit"),
                "body_part": shot.get("body"),
                "is_home": shot.get("home"),
                "xg": shot.get("xg"),
                "xgot": shot.get("xgot"),
                "pos_x": pos.get("x"),
                "pos_y": pos.get("y"),
                "gm_y": gm.get("y"),
                "gm_z": gm.get("z"),
                "gm_location": shot.get("gml"),
                "player_id": shot.get("pid"),
            })

        # Momentum
        for m in (detail.get("momentum") or []):
            momentum_rows.append({
                "event_id": event_id,
                "match_date": match_date,
                "home_team": home_team,
                "away_team": away_team,
                "minute": m.get("minute"),
                "value": m.get("value"),
            })

        # Average positions
        avg_pos = detail.get("average_positions") or {}
        for side in ("home", "away"):
            team_name = home_team if side == "home" else away_team
            for p in (avg_pos.get(side) or []):
                pos = p.get("pos") or {}
                position_rows.append({
                    "event_id": event_id,
                    "match_date": match_date,
                    "team": team_name,
                    "side": side,
                    "player": p.get("player"),
                    "player_id": p.get("pid"),
                    "number": p.get("number"),
                    "avg_x": pos.get("x"),
                    "avg_y": pos.get("y"),
                })

    # Save shotmap
    df_shots = pd.DataFrame(shot_rows)
    if not df_shots.empty:
        df_shots.to_csv(output_path, index=False, sep="\t")
        print(f"Saved {len(df_shots)} shots to {output_path}")

    # Save momentum
    momentum_path = output_path.replace("shotmap", "momentum")
    df_momentum = pd.DataFrame(momentum_rows)
    if not df_momentum.empty:
        df_momentum.to_csv(momentum_path, index=False, sep="\t")
        print(f"Saved {len(df_momentum)} momentum rows to {momentum_path}")

    # Save positions
    positions_path = output_path.replace("shotmap", "avg_positions")
    df_positions = pd.DataFrame(position_rows)
    if not df_positions.empty:
        df_positions.to_csv(positions_path, index=False, sep="\t")
        print(f"Saved {len(df_positions)} position rows to {positions_path}")

    # Save actual xG + odds
    xg_path = output_path.replace("shotmap", "actual_xg")
    df_xg = pd.DataFrame(xg_rows)
    if not df_xg.empty:
        df_xg.to_csv(xg_path, index=False, sep="\t")
        print(f"Saved {len(df_xg)} match xG records to {xg_path}")

    return df_shots


def backfill_player_stats(
    output_path: str = "data_files/bzzoiro_player_stats.csv",
) -> pd.DataFrame:
    """
    Fetch per-match player stats for all PL teams.

    Each row = one player in one match. 30+ stat columns.
    """
    teams = get_pl_teams()
    print(f"Fetching player stats for {len(teams)} PL teams...")

    all_rows = []
    for i, team in enumerate(teams):
        team_id = team["id"]
        team_name = normalize_bzzoiro_team(team.get("name", ""))
        print(f"  [{i+1}/{len(teams)}] {team_name}...")

        stats = get_player_stats(team_id=team_id)
        for s in stats:
            evt = s.get("event") or {}
            row = {
                "event_id": evt.get("id"),
                "match_date": (evt.get("event_date") or "")[:10],
                "home_team": normalize_bzzoiro_team(evt.get("home_team", "")),
                "away_team": normalize_bzzoiro_team(evt.get("away_team", "")),
                "home_score": evt.get("home_score"),
                "away_score": evt.get("away_score"),
                "team": team_name,
                "minutes_played": s.get("minutes_played"),
                "rating": s.get("rating"),
                "touches": s.get("touches"),
                "goals": s.get("goals"),
                "goal_assist": s.get("goal_assist"),
                "expected_goals": s.get("expected_goals"),
                "expected_assists": s.get("expected_assists"),
                "total_shots": s.get("total_shots"),
                "shots_on_target": s.get("shots_on_target"),
                "total_pass": s.get("total_pass"),
                "accurate_pass": s.get("accurate_pass"),
                "key_pass": s.get("key_pass"),
                "total_cross": s.get("total_cross"),
                "accurate_cross": s.get("accurate_cross"),
                "total_long_balls": s.get("total_long_balls"),
                "accurate_long_balls": s.get("accurate_long_balls"),
                "duel_won": s.get("duel_won"),
                "duel_lost": s.get("duel_lost"),
                "aerial_won": s.get("aerial_won"),
                "aerial_lost": s.get("aerial_lost"),
                "total_tackle": s.get("total_tackle"),
                "won_tackle": s.get("won_tackle"),
                "total_clearance": s.get("total_clearance"),
                "interception": s.get("interception"),
                "ball_recovery": s.get("ball_recovery"),
                "dispossessed": s.get("dispossessed"),
                "possession_lost": s.get("possession_lost"),
                "was_fouled": s.get("was_fouled"),
                "fouls": s.get("fouls"),
                "yellow_card": s.get("yellow_card"),
                "red_card": s.get("red_card"),
                "saves": s.get("saves"),
                "goals_conceded": s.get("goals_conceded"),
            }
            all_rows.append(row)

    df = pd.DataFrame(all_rows)
    if not df.empty:
        df.to_csv(output_path, index=False, sep="\t")
        print(f"Saved {len(df)} player-stat records to {output_path}")
    return df


def fetch_squad_values(
    output_path: str = "data_files/bzzoiro_squad_values.csv",
) -> pd.DataFrame:
    """Fetch market values for all PL team squads."""
    teams = get_pl_teams()
    print(f"Fetching squad values for {len(teams)} PL teams...")

    rows = []
    for team in teams:
        team_id = team["id"]
        team_name = normalize_bzzoiro_team(team.get("name", ""))
        players = get_pl_players(team_id=team_id)

        for p in players:
            rows.append({
                "team": team_name,
                "team_api_id": (p.get("current_team") or {}).get("api_id"),
                "player_id": p.get("id"),
                "player_api_id": p.get("api_id"),
                "name": p.get("name"),
                "short_name": p.get("short_name"),
                "position": p.get("position"),
                "jersey_number": p.get("jersey_number"),
                "height": p.get("height"),
                "date_of_birth": p.get("date_of_birth"),
                "nationality": p.get("nationality"),
                "market_value": p.get("market_value"),
            })

    df = pd.DataFrame(rows)
    if not df.empty:
        df.to_csv(output_path, index=False, sep="\t")
        print(f"Saved {len(df)} player records to {output_path}")
    return df


def fetch_upcoming_predictions(
    output_path: str = "data_files/bzzoiro_predictions.csv",
) -> pd.DataFrame:
    """Fetch Bzzoiro predictions for upcoming PL matches and save as CSV."""
    preds = get_pl_predictions(upcoming_only=True)
    print(f"Fetched {len(preds)} predictions.")

    rows = []
    for p in preds:
        evt = p.get("event") or {}
        rows.append({
            "event_id": evt.get("id"),
            "match_date": (evt.get("event_date") or "")[:10],
            "home_team": normalize_bzzoiro_team(evt.get("home_team", "")),
            "away_team": normalize_bzzoiro_team(evt.get("away_team", "")),
            "round_number": evt.get("round_number"),
            "prob_home_win": p.get("prob_home_win"),
            "prob_draw": p.get("prob_draw"),
            "prob_away_win": p.get("prob_away_win"),
            "predicted_result": p.get("predicted_result"),
            "expected_home_goals": p.get("expected_home_goals"),
            "expected_away_goals": p.get("expected_away_goals"),
            "prob_over_15": p.get("prob_over_15"),
            "prob_over_25": p.get("prob_over_25"),
            "prob_over_35": p.get("prob_over_35"),
            "prob_btts_yes": p.get("prob_btts_yes"),
            "confidence": p.get("confidence"),
            "model_version": p.get("model_version"),
            "most_likely_score": p.get("most_likely_score"),
            "favorite": p.get("favorite"),
            "favorite_prob": p.get("favorite_prob"),
            "favorite_recommend": p.get("favorite_recommend"),
            "over_25_recommend": p.get("over_25_recommend"),
            "btts_recommend": p.get("btts_recommend"),
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df.to_csv(output_path, index=False, sep="\t")
        print(f"Saved {len(df)} predictions to {output_path}")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════════

def compute_shotmap_features(
    shotmap_csv: str = "data_files/bzzoiro_shotmap.csv",
    window: int = 5,
) -> pd.DataFrame:
    """
    From shotmap data, compute per-team rolling features:
      - AvgXgPerShot_L{window}  (shot quality)
      - AvgXGoT_L{window}       (post-shot quality)
      - BigChanceCount_L{window} (shots with xG > 0.3)
      - SetPieceXgPct_L{window}  (% of xG from set pieces)
      - TotalXg_L{window}        (cumulative xG)

    Returns a DataFrame with columns:
      match_date, home_team, away_team,
      Home_AvgXgPerShot_L5, Home_AvgXGoT_L5, Home_BigChanceCount_L5, ...
      Away_AvgXgPerShot_L5, Away_AvgXGoT_L5, Away_BigChanceCount_L5, ...
    """
    if not os.path.exists(shotmap_csv):
        print(f"Shotmap CSV not found: {shotmap_csv}. Run backfill_shotmap_data() first.")
        return pd.DataFrame()

    df = pd.read_csv(shotmap_csv, sep="\t")
    if df.empty:
        return pd.DataFrame()

    df["match_date"] = pd.to_datetime(df["match_date"])

    # Determine team per shot
    df["team"] = np.where(df["is_home"], df["home_team"], df["away_team"])
    df["is_set_piece"] = df["situation"].isin(["corner", "set-piece", "free-kick"])

    # Aggregate per match per team
    match_team = df.groupby(["event_id", "match_date", "team"]).agg(
        total_shots=("xg", "count"),
        total_xg=("xg", "sum"),
        avg_xg_per_shot=("xg", "mean"),
        avg_xgot=("xgot", "mean"),
        big_chances=("xg", lambda x: (x > 0.3).sum()),
        set_piece_xg=("xg", lambda x: x[df.loc[x.index, "is_set_piece"]].sum() if len(x) > 0 else 0),
    ).reset_index()

    match_team["set_piece_xg_pct"] = np.where(
        match_team["total_xg"] > 0,
        match_team["set_piece_xg"] / match_team["total_xg"] * 100,
        0,
    )

    # Sort by date for rolling
    match_team = match_team.sort_values(["team", "match_date"])

    # Rolling averages per team
    for col in ["avg_xg_per_shot", "avg_xgot", "big_chances", "set_piece_xg_pct", "total_xg"]:
        match_team[f"{col}_L{window}"] = (
            match_team.groupby("team")[col]
            .transform(lambda x: x.rolling(window, min_periods=1).mean().shift(1))
        )

    # Now we need to create match-level features (home + away)
    # Reconstruct match-level from the original events
    matches = df.drop_duplicates(subset=["event_id"])[["event_id", "match_date", "home_team", "away_team"]].copy()
    matches["match_date"] = pd.to_datetime(matches["match_date"])

    # Merge home team features
    home_feats = match_team[["event_id", "team",
                             f"avg_xg_per_shot_L{window}", f"avg_xgot_L{window}",
                             f"big_chances_L{window}", f"set_piece_xg_pct_L{window}",
                             f"total_xg_L{window}"]].copy()
    home_feats = home_feats.rename(columns={
        f"avg_xg_per_shot_L{window}": f"Home_AvgXgPerShot_L{window}",
        f"avg_xgot_L{window}": f"Home_AvgXGoT_L{window}",
        f"big_chances_L{window}": f"Home_BigChanceCount_L{window}",
        f"set_piece_xg_pct_L{window}": f"Home_SetPieceXgPct_L{window}",
        f"total_xg_L{window}": f"Home_TotalXg_L{window}",
    })

    result = matches.merge(
        home_feats, left_on=["event_id", "home_team"], right_on=["event_id", "team"], how="left"
    ).drop(columns=["team"])

    # Merge away team features
    away_feats = match_team[["event_id", "team",
                             f"avg_xg_per_shot_L{window}", f"avg_xgot_L{window}",
                             f"big_chances_L{window}", f"set_piece_xg_pct_L{window}",
                             f"total_xg_L{window}"]].copy()
    away_feats = away_feats.rename(columns={
        f"avg_xg_per_shot_L{window}": f"Away_AvgXgPerShot_L{window}",
        f"avg_xgot_L{window}": f"Away_AvgXGoT_L{window}",
        f"big_chances_L{window}": f"Away_BigChanceCount_L{window}",
        f"set_piece_xg_pct_L{window}": f"Away_SetPieceXgPct_L{window}",
        f"total_xg_L{window}": f"Away_TotalXg_L{window}",
    })

    result = result.merge(
        away_feats, left_on=["event_id", "away_team"], right_on=["event_id", "team"], how="left"
    ).drop(columns=["team"])

    # Rename for merging with historical data
    result = result.rename(columns={"home_team": "HomeTeam", "away_team": "AwayTeam", "match_date": "MatchDate"})
    result["MatchDate"] = result["MatchDate"].dt.strftime("%Y-%m-%d")

    return result.drop(columns=["event_id"])


def compute_player_stat_features(
    stats_csv: str = "data_files/bzzoiro_player_stats.csv",
    window: int = 5,
) -> pd.DataFrame:
    """
    From player stats data, compute per-team-per-match rolling features:
      - PassAccuracy_L{window}     (accurate_pass / total_pass)
      - DuelWinRate_L{window}      (duel_won / (duel_won + duel_lost))
      - AerialDuelWinRate_L{window}
      - KeyPassesPerGame_L{window}
      - AvgTeamRating_L{window}
      - AvgTeamXg_L{window}        (sum of player xG)

    Returns a DataFrame with Home_ and Away_ prefixed columns.
    """
    if not os.path.exists(stats_csv):
        print(f"Player stats CSV not found: {stats_csv}. Run backfill_player_stats() first.")
        return pd.DataFrame()

    df = pd.read_csv(stats_csv, sep="\t")
    if df.empty:
        return pd.DataFrame()

    df["match_date"] = pd.to_datetime(df["match_date"])

    # Aggregate to team-match level
    team_match = df.groupby(["event_id", "match_date", "team", "home_team", "away_team"]).agg(
        total_pass_sum=("total_pass", "sum"),
        accurate_pass_sum=("accurate_pass", "sum"),
        key_pass_sum=("key_pass", "sum"),
        duel_won_sum=("duel_won", "sum"),
        duel_lost_sum=("duel_lost", "sum"),
        aerial_won_sum=("aerial_won", "sum"),
        aerial_lost_sum=("aerial_lost", "sum"),
        team_xg=("expected_goals", "sum"),
        avg_rating=("rating", "mean"),
    ).reset_index()

    # Derived ratios
    team_match["pass_accuracy"] = np.where(
        team_match["total_pass_sum"] > 0,
        team_match["accurate_pass_sum"] / team_match["total_pass_sum"] * 100,
        0,
    )
    team_match["duel_win_rate"] = np.where(
        (team_match["duel_won_sum"] + team_match["duel_lost_sum"]) > 0,
        team_match["duel_won_sum"] / (team_match["duel_won_sum"] + team_match["duel_lost_sum"]) * 100,
        50,
    )
    team_match["aerial_win_rate"] = np.where(
        (team_match["aerial_won_sum"] + team_match["aerial_lost_sum"]) > 0,
        team_match["aerial_won_sum"] / (team_match["aerial_won_sum"] + team_match["aerial_lost_sum"]) * 100,
        50,
    )

    team_match = team_match.sort_values(["team", "match_date"])

    # Rolling features
    for col in ["pass_accuracy", "duel_win_rate", "aerial_win_rate", "key_pass_sum", "avg_rating", "team_xg"]:
        team_match[f"{col}_L{window}"] = (
            team_match.groupby("team")[col]
            .transform(lambda x: x.rolling(window, min_periods=1).mean().shift(1))
        )

    # Build match-level output
    matches = team_match.drop_duplicates(subset=["event_id"])[["event_id", "match_date", "home_team", "away_team"]].copy()

    feat_cols = [f"pass_accuracy_L{window}", f"duel_win_rate_L{window}",
                 f"aerial_win_rate_L{window}", f"key_pass_sum_L{window}",
                 f"avg_rating_L{window}", f"team_xg_L{window}"]

    # Home
    home_cols = {c: f"Home_{c.replace(f'_L{window}', '').title().replace('_', '')}_L{window}" for c in feat_cols}
    home_feats = team_match[["event_id", "team"] + feat_cols].rename(columns=home_cols)
    result = matches.merge(home_feats, left_on=["event_id", "home_team"], right_on=["event_id", "team"], how="left")
    result = result.drop(columns=["team"], errors="ignore")

    # Away
    away_cols = {c: f"Away_{c.replace(f'_L{window}', '').title().replace('_', '')}_L{window}" for c in feat_cols}
    away_feats = team_match[["event_id", "team"] + feat_cols].rename(columns=away_cols)
    result = result.merge(away_feats, left_on=["event_id", "away_team"], right_on=["event_id", "team"], how="left")
    result = result.drop(columns=["team"], errors="ignore")

    result = result.rename(columns={"home_team": "HomeTeam", "away_team": "AwayTeam", "match_date": "MatchDate"})
    result["MatchDate"] = result["MatchDate"].dt.strftime("%Y-%m-%d")

    return result.drop(columns=["event_id"])


def compute_actual_xg_features(
    xg_csv: str = "data_files/bzzoiro_actual_xg.csv",
    window: int = 5,
) -> pd.DataFrame:
    """
    From actual xG data, compute per-team rolling features:
      - Home_ActualXg_L{window}           (rolling avg actual xG)
      - Away_ActualXg_L{window}
      - Home_XgOverperformance_L{window}  (goals - xG, finishing quality)
      - Away_XgOverperformance_L{window}

    Returns a DataFrame ready to merge on HomeTeam, AwayTeam, MatchDate.
    """
    if not os.path.exists(xg_csv):
        print(f"Actual xG CSV not found: {xg_csv}. Run backfill_actual_xg() first.")
        return pd.DataFrame()

    df = pd.read_csv(xg_csv, sep="\t")
    if df.empty:
        return pd.DataFrame()

    df["match_date"] = pd.to_datetime(df["match_date"])

    # Build home and away records for rolling calculations
    home_records = df[["event_id", "match_date", "home_team", "actual_home_xg", "home_score"]].copy()
    home_records = home_records.rename(columns={
        "home_team": "team", "actual_home_xg": "actual_xg", "home_score": "goals"
    })
    home_records["side"] = "home"

    away_records = df[["event_id", "match_date", "away_team", "actual_away_xg", "away_score"]].copy()
    away_records = away_records.rename(columns={
        "away_team": "team", "actual_away_xg": "actual_xg", "away_score": "goals"
    })
    away_records["side"] = "away"

    all_records = pd.concat([home_records, away_records], ignore_index=True)
    all_records = all_records.dropna(subset=["actual_xg"])
    all_records["xg_overperformance"] = all_records["goals"] - all_records["actual_xg"]
    all_records = all_records.sort_values(["team", "match_date"])

    # Rolling features per team
    all_records[f"actual_xg_L{window}"] = (
        all_records.groupby("team")["actual_xg"]
        .transform(lambda x: x.rolling(window, min_periods=1).mean().shift(1))
    )
    all_records[f"xg_overperf_L{window}"] = (
        all_records.groupby("team")["xg_overperformance"]
        .transform(lambda x: x.rolling(window, min_periods=1).mean().shift(1))
    )

    # Merge back to match level
    home_feats = all_records[all_records["side"] == "home"][
        ["event_id", f"actual_xg_L{window}", f"xg_overperf_L{window}"]
    ].rename(columns={
        f"actual_xg_L{window}": f"Home_ActualXg_L{window}",
        f"xg_overperf_L{window}": f"Home_XgOverperformance_L{window}",
    })

    away_feats = all_records[all_records["side"] == "away"][
        ["event_id", f"actual_xg_L{window}", f"xg_overperf_L{window}"]
    ].rename(columns={
        f"actual_xg_L{window}": f"Away_ActualXg_L{window}",
        f"xg_overperf_L{window}": f"Away_XgOverperformance_L{window}",
    })

    result = df[["match_date", "home_team", "away_team", "event_id"]].copy()
    result = result.merge(home_feats, on="event_id", how="left")
    result = result.merge(away_feats, on="event_id", how="left")

    result = result.rename(columns={"home_team": "HomeTeam", "away_team": "AwayTeam", "match_date": "MatchDate"})
    result["MatchDate"] = pd.to_datetime(result["MatchDate"]).dt.strftime("%Y-%m-%d")

    return result.drop(columns=["event_id"])


def compute_momentum_features(
    momentum_csv: str = "data_files/bzzoiro_momentum.csv",
    window: int = 5,
) -> pd.DataFrame:
    """
    From momentum data, compute per-match features:
      - MomentumAvg          (avg pressure over full match; + = home dominant)
      - MomentumStd          (consistency)
      - LateMomentum         (avg momentum in min 75+)
      - FirstHalfMomentum    (avg momentum in min 1-45)
      - MomentumSwings       (count of sign changes)

    Then rolling averages per team over last {window} matches.
    """
    if not os.path.exists(momentum_csv):
        print(f"Momentum CSV not found: {momentum_csv}. Run backfill_shotmap_data() first.")
        return pd.DataFrame()

    df = pd.read_csv(momentum_csv, sep="\t")
    if df.empty:
        return pd.DataFrame()

    df["match_date"] = pd.to_datetime(df["match_date"])

    # Per-match aggregation
    def match_momentum_features(group):
        values = group["value"].values
        return pd.Series({
            "momentum_avg": values.mean(),
            "momentum_std": values.std() if len(values) > 1 else 0,
            "late_momentum": values[group["minute"].values >= 75].mean() if (group["minute"].values >= 75).any() else 0,
            "first_half_momentum": values[group["minute"].values <= 45].mean() if (group["minute"].values <= 45).any() else 0,
            "momentum_swings": (np.diff(np.sign(values)) != 0).sum() if len(values) > 1 else 0,
        })

    match_feats = df.groupby(["event_id", "match_date", "home_team", "away_team"]).apply(
        match_momentum_features
    ).reset_index()

    # For home team: positive momentum = good. For away: negative = good.
    match_feats["home_momentum_avg"] = match_feats["momentum_avg"]
    match_feats["away_momentum_avg"] = -match_feats["momentum_avg"]
    match_feats["home_late_momentum"] = match_feats["late_momentum"]
    match_feats["away_late_momentum"] = -match_feats["late_momentum"]

    match_feats = match_feats.sort_values("match_date")

    # Rolling for home team features (using home_team grouping)
    for prefix, team_col in [("Home", "home_team"), ("Away", "away_team")]:
        mom_col = f"{prefix.lower()}_momentum_avg"
        late_col = f"{prefix.lower()}_late_momentum"

        match_feats[f"{prefix}_MomentumAvg_L{window}"] = (
            match_feats.groupby(team_col)[mom_col]
            .transform(lambda x: x.rolling(window, min_periods=1).mean().shift(1))
        )
        match_feats[f"{prefix}_LateMomentum_L{window}"] = (
            match_feats.groupby(team_col)[late_col]
            .transform(lambda x: x.rolling(window, min_periods=1).mean().shift(1))
        )
        match_feats[f"{prefix}_MomentumStd_L{window}"] = (
            match_feats.groupby(team_col)["momentum_std"]
            .transform(lambda x: x.rolling(window, min_periods=1).mean().shift(1))
        )

    result = match_feats[["match_date", "home_team", "away_team",
                           f"Home_MomentumAvg_L{window}", f"Home_LateMomentum_L{window}",
                           f"Home_MomentumStd_L{window}",
                           f"Away_MomentumAvg_L{window}", f"Away_LateMomentum_L{window}",
                           f"Away_MomentumStd_L{window}"]].copy()

    result = result.rename(columns={"home_team": "HomeTeam", "away_team": "AwayTeam", "match_date": "MatchDate"})
    result["MatchDate"] = result["MatchDate"].dt.strftime("%Y-%m-%d")

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    usage = """
Usage:
    python bzzoiro_football_api.py --backfill-all       # Full backfill (xG + shotmap + player stats + squad values)
    python bzzoiro_football_api.py --actual-xg           # Backfill actual xG + odds (fastest, highest priority)
    python bzzoiro_football_api.py --shotmap             # Backfill shotmap/momentum/positions only
    python bzzoiro_football_api.py --player-stats        # Backfill per-match player stats
    python bzzoiro_football_api.py --squad-values        # Fetch squad market values
    python bzzoiro_football_api.py --predictions         # Fetch upcoming predictions
    python bzzoiro_football_api.py --live                # Show live PL matches
    python bzzoiro_football_api.py --test                # Quick connectivity test
    """

    if len(sys.argv) < 2:
        print(usage)
        sys.exit(0)

    cmd = sys.argv[1]

    if cmd == "--test":
        print("Testing Bzzoiro API connectivity...")
        league_id = get_pl_league_id()
        print(f"Premier League ID: {league_id}")
        events = get_pl_events()
        print(f"Upcoming PL events (next 7 days): {len(events)}")
        for e in events[:3]:
            print(f"  {e['home_team']} vs {e['away_team']} — {e['event_date']}")
        preds = get_pl_predictions()
        print(f"Upcoming PL predictions: {len(preds)}")
        print("API connection successful.")

    elif cmd == "--backfill-all":
        backfill_actual_xg()
        backfill_shotmap_data()
        backfill_player_stats()
        fetch_squad_values()
        fetch_upcoming_predictions()

    elif cmd == "--actual-xg":
        date_from = sys.argv[2] if len(sys.argv) > 2 else "2024-08-01"
        backfill_actual_xg(date_from=date_from)

    elif cmd == "--shotmap":
        date_from = sys.argv[2] if len(sys.argv) > 2 else "2024-08-01"
        backfill_shotmap_data(date_from=date_from)

    elif cmd == "--player-stats":
        backfill_player_stats()

    elif cmd == "--squad-values":
        fetch_squad_values()

    elif cmd == "--predictions":
        fetch_upcoming_predictions()

    elif cmd == "--live":
        live = get_live_matches()
        if not live:
            print("No PL matches currently in play.")
        for m in live:
            stats = m.get("live_stats") or {}
            home_s = stats.get("home", {})
            away_s = stats.get("away", {})
            print(
                f"{m['current_minute']}' — {m['home_team']} {m['home_score']}-{m['away_score']} {m['away_team']}"
                f"  |  Poss: {home_s.get('ball_possession', '?')}%-{away_s.get('ball_possession', '?')}%"
                f"  Shots: {home_s.get('total_shots', '?')}-{away_s.get('total_shots', '?')}"
            )

    else:
        print(f"Unknown command: {cmd}")
        print(usage)
        sys.exit(1)
