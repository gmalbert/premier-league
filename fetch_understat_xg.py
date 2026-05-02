"""
Fetch EPL expected-goals (xG) match data from Understat.

Understat loads data via a JSON API endpoint: /getLeagueData/{league}/{season}.
A session cookie obtained from the league page is required for the API call.
Covers EPL seasons 2014/15 onwards. Outputs one row per completed match.

Outputs:
  data_files/understat_xg.csv — match-level HomeXG and AwayXG per season

Run: python fetch_understat_xg.py
"""

import requests
import time
import pandas as pd
from os import path

DATA_DIR = 'data_files/'

# Understat display name  →  historical data team name
UNDERSTAT_TEAM_MAP = {
    'Manchester City':           'Man City',
    'Manchester United':         'Man United',
    'Arsenal':                   'Arsenal',
    'Chelsea':                   'Chelsea',
    'Liverpool':                 'Liverpool',
    'Tottenham':                 'Tottenham',
    'Everton':                   'Everton',
    'Aston Villa':               'Aston Villa',
    'Newcastle United':          'Newcastle',
    'West Ham':                  'West Ham',
    'Crystal Palace':            'Crystal Palace',
    'Southampton':               'Southampton',
    'Leicester':                 'Leicester',
    'Fulham':                    'Fulham',
    'Brentford':                 'Brentford',
    'Brighton':                  'Brighton',
    'Bournemouth':               'Bournemouth',
    'Wolverhampton Wanderers':   'Wolves',
    'Nottingham Forest':         "Nott'm Forest",
    'Burnley':                   'Burnley',
    'Leeds':                     'Leeds',
    'Watford':                   'Watford',
    'Norwich':                   'Norwich',
    'Sheffield Utd':             'Sheffield United',
    'Sunderland':                'Sunderland',
    'Stoke':                     'Stoke',
    'Swansea':                   'Swansea',
    'Hull':                      'Hull',
    'Reading':                   'Reading',
    'QPR':                       'QPR',
    'West Brom':                 'West Brom',
    'Middlesbrough':             'Middlesbrough',
    'Blackburn':                 'Blackburn',
    'Huddersfield':              'Huddersfield',
    'Cardiff':                   'Cardiff',
    'Luton':                     'Luton',
    'Ipswich':                   'Ipswich',
}

BASE_UA = (
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
    'AppleWebKit/537.36 (KHTML, like Gecko) '
    'Chrome/120.0.0.0 Safari/537.36'
)


def _make_session(year):
    """
    Create a requests.Session with a valid cookie by visiting the league page first.
    Understat's API requires the session cookie set on the HTML page.
    """
    session = requests.Session()
    session.headers.update({'User-Agent': BASE_UA})
    league_url = f'https://understat.com/league/EPL/{year}'
    try:
        session.get(league_url, timeout=20)
    except Exception as e:
        print(f'  Warning: could not establish session for season {year}: {e}')
    return session


def fetch_season_xg(year, session=None):
    """
    Fetch all completed-match xG for the EPL season starting in `year`.
    year=2014 → 2014/15 season.
    Returns a list of dicts.
    """
    if session is None:
        session = _make_session(year)

    api_url = f'https://understat.com/getLeagueData/EPL/{year}'
    try:
        resp = session.get(
            api_url,
            headers={
                'X-Requested-With': 'XMLHttpRequest',
                'Referer': f'https://understat.com/league/EPL/{year}',
            },
            timeout=20,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f'  Warning: could not fetch season {year}: {e}')
        return []

    dates = data.get('dates', [])
    if not dates:
        print(f'  Warning: no dates data for season {year}')
        return []

    matches = []
    for item in dates:
        if not item.get('isResult', False):
            continue
        try:
            home_title = item['h']['title']
            away_title = item['a']['title']
            xg_block = item.get('xG', {})
            home_xg = float(xg_block.get('h') or 0)
            away_xg = float(xg_block.get('a') or 0)
            match_dt = pd.to_datetime(item.get('datetime', ''), errors='coerce')
            matches.append({
                'MatchDate':            match_dt,
                'HomeTeam_Understat':   home_title,
                'AwayTeam_Understat':   away_title,
                'HomeTeam':             UNDERSTAT_TEAM_MAP.get(home_title, home_title),
                'AwayTeam':             UNDERSTAT_TEAM_MAP.get(away_title, away_title),
                'HomeXG_Understat':     round(home_xg, 4),
                'AwayXG_Understat':     round(away_xg, 4),
                'Season':               year,
            })
        except (KeyError, ValueError, TypeError):
            continue
    return matches


def main():
    all_matches = []
    # EPL xG data available from 2014/15 season (year=2014)
    seasons = range(2014, 2026)

    for year in seasons:
        print(f'Fetching xG for {year}/{str(year + 1)[2:]} season...')
        # Fresh session per season — visits the league page to obtain a cookie
        session = _make_session(year)
        matches = fetch_season_xg(year, session=session)
        print(f'  {len(matches)} completed matches')
        all_matches.extend(matches)
        time.sleep(1.5)   # polite rate limit

    if not all_matches:
        print('No xG data fetched.')
        return

    df = pd.DataFrame(all_matches)
    df = df.dropna(subset=['MatchDate'])
    df['MatchDate'] = df['MatchDate'].dt.normalize()   # date only, no time
    df = df.sort_values('MatchDate').reset_index(drop=True)

    out = path.join(DATA_DIR, 'understat_xg.csv')
    df.to_csv(out, sep='\t', index=False)

    print(f'\nSaved {len(df)} match xG records to {out}')
    print(f'Seasons covered: {df["Season"].min()}–{df["Season"].max()}')
    print(f'Columns: {list(df.columns)}')


if __name__ == '__main__':
    main()
