"""
Wyscout Soccer Match Event Dataset — research exploration script.

Dataset: 380 EPL matches (2017/18 season) with spatio-temporal event data.
Source: https://github.com/koenvo/wyscout-soccer-match-event-dataset
License: CC BY 4.0

This script downloads the dataset index, loads a sample match via kloppy,
and demonstrates the event schema. It is for offline research only — this
dataset is a static 2017/18 snapshot and cannot be used for live predictions.

Typical use: prototype event-derived features (xG from position, press rate,
pass network centrality) before applying the methodology to a live data feed.

Run: python explore_wyscout.py
     (requires: pip install kloppy)
"""

import sys

try:
    from kloppy import datasets
    from kloppy.helpers import to_pandas
except ImportError:
    print('kloppy not installed. Run: pip install kloppy')
    sys.exit(1)

import pandas as pd
from os import path

DATA_DIR = 'data_files/'

# Default Wyscout match in the kloppy datasets API:
# 2499841 — Huddersfield Town vs Manchester City (2017/18 EPL)
DEFAULT_MATCH_ID = 2499841

# Other available EPL match IDs (see the index at
# https://github.com/koenvo/wyscout-soccer-match-event-dataset/blob/main/processed/README.md)
SAMPLE_MATCH_IDS = [
    2499841,   # Huddersfield vs Man City
    2499843,   # Another EPL match
    2499845,
]


def load_and_explore(match_id=DEFAULT_MATCH_ID):
    print(f'Loading Wyscout match {match_id} via kloppy datasets API...')
    try:
        dataset = datasets.load('wyscout', match_id=match_id)
    except Exception as e:
        print(f'Failed to load match {match_id}: {e}')
        return None

    print(f'  Coordinate system: {dataset.metadata.coordinate_system}')
    print(f'  Home team: {dataset.metadata.home_team}')
    print(f'  Away team: {dataset.metadata.away_team}')
    print(f'  Total events: {len(dataset.events)}')

    # Convert to pandas DataFrame for exploration
    df = to_pandas(dataset)
    print(f'\nEvent DataFrame shape: {df.shape}')
    print(f'Columns: {list(df.columns)}\n')
    print('Sample events:')
    print(df[['event_type', 'team_id', 'player_id', 'result', 'period_id', 'timestamp']].head(20))

    # Event type distribution
    if 'event_type' in df.columns:
        print('\nEvent type counts:')
        print(df['event_type'].value_counts().head(15))

    # Shot analysis
    shot_mask = df['event_type'].astype(str).str.lower().str.contains('shot', na=False)
    shots = df[shot_mask]
    if len(shots) > 0:
        print(f'\nShots in match: {len(shots)}')
        if 'coordinates_x' in shots.columns:
            print('Avg shot x-coordinate:', shots['coordinates_x'].mean())
            print('Avg shot y-coordinate:', shots['coordinates_y'].mean())

    return df


def derive_shot_quality_features(df):
    """
    Example of deriving a basic shot-quality proxy from event position.
    In a full implementation this would feed a logistic regression to predict xG.
    """
    shot_mask = df['event_type'].astype(str).str.lower().str.contains('shot', na=False)
    shots = df[shot_mask].copy()

    if shots.empty or 'coordinates_x' not in shots.columns:
        print('No shot coordinates available for feature derivation.')
        return

    # Distance to goal (Wyscout coordinates: 0-100 x, 0-100 y; goal at x=100, y=50)
    shots['dist_to_goal'] = ((100 - shots['coordinates_x']) ** 2 +
                             (50 - shots['coordinates_y']) ** 2) ** 0.5
    # Angle to goal (rough)
    import math
    shots['angle_to_goal'] = shots.apply(
        lambda r: abs(math.degrees(math.atan2(
            abs(r['coordinates_y'] - 50),
            max(100 - r['coordinates_x'], 1)
        ))), axis=1
    )
    print('\nDerived shot quality features (sample):')
    print(shots[['dist_to_goal', 'angle_to_goal', 'result']].head(10))
    print(f'\nMean distance to goal: {shots["dist_to_goal"].mean():.2f} units')
    print(f'Mean angle to goal: {shots["angle_to_goal"].mean():.2f} degrees')


def main():
    print('=== Wyscout Research Exploration ===\n')
    print('Dataset: EPL 2017/18 season (static, CC BY 4.0)\n')

    df = load_and_explore(DEFAULT_MATCH_ID)
    if df is not None:
        derive_shot_quality_features(df)

        # Save sample to CSV for further exploration
        out = path.join(DATA_DIR, 'wyscout_sample_events.csv')
        df.to_csv(out, index=False)
        print(f'\nSaved {len(df)} events to {out}')

    print('\n--- Research notes ---')
    print('This 2017/18 snapshot is useful for:')
    print('  1. Validating xG models using position + body-part + game-state features')
    print('  2. Prototyping press-rate or pass-network features')
    print('  3. Understanding the Wyscout event schema before purchasing live data')
    print('\nTo use in production, a live Wyscout / Opta / StatsBomb subscription is needed.')


if __name__ == '__main__':
    main()
