"""
Fetch ClubElo ratings for Premier League teams.

Outputs:
  data_files/clubelo_ratings.csv  — full Elo history per team
  data_files/clubelo_fixtures.csv — upcoming matches with H/D/A probabilities

Run: python fetch_clubelo.py
"""

import requests
import pandas as pd
import io
import time
from os import path

DATA_DIR = 'data_files/'

# Mapping: historical data team name  →  ClubElo URL slug
# Verify/update slugs at http://clubelo.com (check ranking page for exact spelling)
CLUBELO_TEAM_MAP = {
    'Man City':          'ManCity',
    'Man United':        'ManUnited',
    'Arsenal':           'Arsenal',
    'Chelsea':           'Chelsea',
    'Liverpool':         'Liverpool',
    'Tottenham':         'Tottenham',
    'Everton':           'Everton',
    'Aston Villa':       'AstonVilla',
    'Newcastle':         'Newcastle',
    'West Ham':          'WestHam',
    'Crystal Palace':    'CrystalPalace',
    'Southampton':       'Southampton',
    'Leicester':         'Leicester',
    'Fulham':            'Fulham',
    'Brentford':         'Brentford',
    'Brighton':          'Brighton',
    'Bournemouth':       'Bournemouth',
    'Wolves':            'Wolves',
    "Nott'm Forest":     'Forest',
    'Burnley':           'Burnley',
    'Leeds':             'Leeds',
    'Watford':           'Watford',
    'Norwich':           'Norwich',
    'Sheffield United':  'SheffieldUnited',
    'Sunderland':        'Sunderland',
    'Stoke':             'Stoke',
    'Swansea':           'Swansea',
    'Hull':              'Hull',
    'Reading':           'Reading',
    'QPR':               'QPR',
    'West Brom':         'WestBrom',
    'Middlesbrough':     'Middlesbrough',
    'Blackburn':         'Blackburn',
    'Wigan':             'Wigan',
    'Bolton':            'Bolton',
    'Ipswich':           'Ipswich',
    'Derby':             'Derby',
    'Birmingham':        'Birmingham',
    'Huddersfield':      'Huddersfield',
    'Cardiff':           'Cardiff',
    'Sheffield Weds':    'SheffieldWeds',
    'Charlton':          'Charlton',
    'Luton':             'Luton',
}


def fetch_team_elo(clubelo_name):
    """Fetch full Elo history for one club from api.clubelo.com/CLUBNAME."""
    url = f'http://api.clubelo.com/{clubelo_name}'
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        if resp.text.strip().startswith('No') or len(resp.text.strip()) < 10:
            return None
        df = pd.read_csv(io.StringIO(resp.text))
        if df.empty or 'Elo' not in df.columns:
            return None
        return df
    except Exception as e:
        print(f'  Warning: failed to fetch {clubelo_name}: {e}')
        return None


def fetch_upcoming_fixtures():
    """
    Fetch upcoming match probabilities from api.clubelo.com/Fixtures.
    Returns a DataFrame with columns: Date, HomeTeam, AwayTeam,
    ClubElo_HomeWinProb, ClubElo_DrawProb, ClubElo_AwayWinProb.
    Probabilities are summed over goal-difference outcomes (>0 home, =0 draw, <0 away).
    """
    url = 'http://api.clubelo.com/Fixtures'
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        raw = pd.read_csv(io.StringIO(resp.text))
    except Exception as e:
        print(f'  Warning: failed to fetch ClubElo fixtures: {e}')
        return pd.DataFrame()

    if raw.empty:
        return pd.DataFrame()

    # ClubElo fixture CSV columns:
    # Date, HomeTeam, AwayTeam, ...prob columns for each GD from <=-6 to >=6
    # Identify goal-difference probability columns (numeric or named like "-5", "0", "5" etc.)
    gd_cols = [c for c in raw.columns if c not in ('Date', 'HomeTeam', 'AwayTeam')]

    rows = []
    for _, row in raw.iterrows():
        home_prob = 0.0
        draw_prob = 0.0
        away_prob = 0.0
        for col in gd_cols:
            try:
                gd = int(col)
                p = float(row[col]) if pd.notna(row[col]) else 0.0
                if gd > 0:
                    home_prob += p
                elif gd == 0:
                    draw_prob += p
                else:
                    away_prob += p
            except (ValueError, TypeError):
                continue

        rows.append({
            'Date': row.get('Date', ''),
            'HomeTeam': row.get('HomeTeam', ''),
            'AwayTeam': row.get('AwayTeam', ''),
            'ClubElo_HomeWinProb': round(home_prob, 4),
            'ClubElo_DrawProb': round(draw_prob, 4),
            'ClubElo_AwayWinProb': round(away_prob, 4),
        })

    return pd.DataFrame(rows)


def main():
    all_frames = []
    success = 0
    fail = 0

    for hist_name, elo_slug in CLUBELO_TEAM_MAP.items():
        print(f'Fetching Elo for {hist_name} ({elo_slug})...')
        df = fetch_team_elo(elo_slug)
        if df is not None:
            df['HistTeam'] = hist_name
            all_frames.append(df)
            print(f'  {len(df)} records')
            success += 1
        else:
            print(f'  No data')
            fail += 1
        time.sleep(0.5)   # polite rate limit

    print(f'\n{success} teams fetched, {fail} not found.')

    if all_frames:
        combined = pd.concat(all_frames, ignore_index=True)
        combined['From'] = pd.to_datetime(combined['From'], errors='coerce')
        combined['To'] = pd.to_datetime(combined['To'], errors='coerce')
        combined['Elo'] = pd.to_numeric(combined['Elo'], errors='coerce')
        out = path.join(DATA_DIR, 'clubelo_ratings.csv')
        combined.to_csv(out, sep='\t', index=False)
        print(f'Saved {len(combined)} Elo records to {out}')
    else:
        print('No Elo data saved.')

    # Fetch upcoming fixture probabilities
    print('\nFetching ClubElo upcoming fixture probabilities...')
    fixtures_df = fetch_upcoming_fixtures()
    if not fixtures_df.empty:
        out_fix = path.join(DATA_DIR, 'clubelo_fixtures.csv')
        fixtures_df.to_csv(out_fix, sep='\t', index=False)
        print(f'Saved {len(fixtures_df)} upcoming fixtures to {out_fix}')
    else:
        print('No upcoming fixture data fetched.')


if __name__ == '__main__':
    main()
