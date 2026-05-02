"""
StatsBomb Open Data — research exploration script.

Dataset: Free event data from StatsBomb covering selected competitions.
Source: https://github.com/statsbomb/open-data
License: Free for research/educational use; StatsBomb attribution required.

NOTE: The free open dataset has very limited Premier League coverage.
Most data is La Liga, Women's football, NWSL, and FA WSL.
For Premier League production use, a StatsBomb commercial licence is required.

This script uses statsbombpy to browse available competitions, load a sample
La Liga match (as a schema/methodology reference), and demonstrate the event
structure — including StatsBomb 360 freeze-frame data where available.

Run: python explore_statsbomb.py
     (requires: pip install statsbombpy)
"""

import sys

try:
    from statsbombpy import sb
except ImportError:
    print('statsbombpy not installed. Run: pip install statsbombpy')
    sys.exit(1)

import pandas as pd
from os import path

DATA_DIR = 'data_files/'

# StatsBomb competition IDs for reference
# 2  = La Liga
# 11 = NWSL
# 37 = FA Women's Super League
# 72 = Men's International (World Cup friendly data)
# 49 = FA Women's Cup

# La Liga season 2015/16 (Messi era — richly covered)
SAMPLE_COMPETITION_ID = 2
SAMPLE_SEASON_ID = 27   # 2015/16


def browse_competitions():
    """List all available competitions in the open dataset."""
    print('Fetching available StatsBomb competitions...')
    try:
        comps = sb.competitions()
    except Exception as e:
        print(f'Failed to fetch competitions: {e}')
        return None

    print(f'\nTotal competitions available: {len(comps)}')
    print(comps[['competition_id', 'competition_name', 'season_id', 'season_name']].to_string(index=False))
    return comps


def load_sample_match(competition_id, season_id):
    """Load a sample match from the given competition/season."""
    print(f'\nFetching matches for competition {competition_id}, season {season_id}...')
    try:
        matches = sb.matches(competition_id=competition_id, season_id=season_id)
    except Exception as e:
        print(f'Failed to fetch matches: {e}')
        return None, None

    if matches.empty:
        print('No matches found.')
        return None, None

    print(f'  {len(matches)} matches found')
    # Pick the first completed match
    sample = matches.iloc[0]
    match_id = sample['match_id']
    print(f'  Sample match: {sample.get("home_team", "")} vs {sample.get("away_team", "")}')
    print(f'  Match ID: {match_id}')

    print(f'\nLoading events for match {match_id}...')
    try:
        events = sb.events(match_id=match_id)
    except Exception as e:
        print(f'Failed to fetch events: {e}')
        return matches, None

    return matches, events


def explore_events(events):
    """Print schema and summary statistics for a match event DataFrame."""
    if events is None or events.empty:
        return

    print(f'\nEvent DataFrame shape: {events.shape}')
    print(f'\nEvent types:')
    print(events['type'].value_counts().head(20))

    print(f'\nKey columns: {list(events.columns[:20])}...')

    # Shots
    shots = events[events['type'] == 'Shot']
    print(f'\nShots in match: {len(shots)}')
    if not shots.empty and 'shot' in shots.columns:
        shot_details = shots['shot'].dropna()
        techniques = shot_details.apply(
            lambda x: x.get('technique', {}).get('name', '') if isinstance(x, dict) else ''
        )
        outcomes = shot_details.apply(
            lambda x: x.get('outcome', {}).get('name', '') if isinstance(x, dict) else ''
        )
        print('Shot techniques:', techniques.value_counts().to_dict())
        print('Shot outcomes:', outcomes.value_counts().to_dict())

    # Passes
    passes = events[events['type'] == 'Pass']
    print(f'\nPasses in match: {len(passes)}')

    # Pressures (advanced metric)
    pressures = events[events['type'] == 'Pressure']
    print(f'Pressure events: {len(pressures)}  (proxy for pressing intensity)')

    return events


def explore_360_data(match_id):
    """
    Attempt to load StatsBomb 360 freeze-frame data for a match.
    Not all matches have 360 data.
    """
    try:
        frames = sb.frames(match_id=match_id)
        if not frames.empty:
            print(f'\nStatsBomb 360 data: {len(frames)} frames for match {match_id}')
            print('Columns:', list(frames.columns))
        else:
            print(f'\nNo 360 data available for match {match_id}')
        return frames
    except Exception as e:
        print(f'\n360 data not available for this match: {e}')
        return pd.DataFrame()


def main():
    print('=== StatsBomb Open Data Research Exploration ===\n')
    print('IMPORTANT: Limited Premier League coverage in free tier.')
    print('Demonstrating schema using La Liga data as a reference.\n')

    # 1. Browse available competitions
    comps = browse_competitions()

    if comps is not None:
        # Highlight PL coverage
        pl = comps[comps['competition_name'].str.contains('Premier League', na=False)]
        if pl.empty:
            print('\nNo Premier League data found in free open dataset (as expected).')
        else:
            print(f'\nPremier League entries found: {len(pl)}')
            print(pl[['competition_id', 'competition_name', 'season_id', 'season_name']].to_string(index=False))

    # 2. Load sample La Liga match
    matches, events = load_sample_match(SAMPLE_COMPETITION_ID, SAMPLE_SEASON_ID)

    if events is not None:
        explore_events(events)

        # Save sample events for schema reference
        out = path.join(DATA_DIR, 'statsbomb_sample_events.csv')
        events.to_csv(out, index=False)
        print(f'\nSaved {len(events)} sample events to {out}')

        # 3. Try 360 data for the same match
        if matches is not None and not matches.empty:
            explore_360_data(matches.iloc[0]['match_id'])

    print('\n--- Research notes ---')
    print('StatsBomb open data is useful for:')
    print('  1. Understanding the richest publicly available event schema')
    print('  2. Prototyping pressure/xG/pass-network features (on La Liga data)')
    print('  3. Validating metrics before buying a PL licence')
    print('\nFor PL production use: https://statsbomb.com/products/data/')
    print('Attribution required: cite StatsBomb and use their logo in any publication.')


if __name__ == '__main__':
    main()
