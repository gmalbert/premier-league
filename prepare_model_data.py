from datetime import date, timedelta
import datetime
import pandas as pd
from os import path
import os
import numpy as np
from scipy import stats
from sklearn.impute import KNNImputer
from scrape_injuries_web import scrape_football_injury_news as scrape_premier_injuries, create_injury_features
from fetch_weather_data import add_weather_features, add_weather_impact_category
from manager_data import get_current_manager, get_manager_stats, calculate_manager_advantage

DATA_DIR = 'data_files/'

# Load the combined raw historical data
historical_data = pd.read_csv(path.join(DATA_DIR, 'combined_historical_data.csv'), sep='\t')

column_rename_map = {
    # General Info
    'Div': 'Division',
    'Date': 'MatchDate',
    'Time': 'KickoffTime',
    'HomeTeam': 'HomeTeam',
    'AwayTeam': 'AwayTeam',
    'Season': 'Season',
    'Round': 'Round',
    'Venue': 'Venue',
    'Referee': 'Referee',
    'Attendance': 'Attendance',

    # Full Time Results
    'FTHG': 'FullTimeHomeGoals',
    'FTAG': 'FullTimeAwayGoals',
    'FTR': 'FullTimeResult',

    # Half Time Results
    'HTHG': 'HalfTimeHomeGoals',
    'HTAG': 'HalfTimeAwayGoals',
    'HTR': 'HalfTimeResult',

    # Shots, Corners, Fouls, Cards
    'HS': 'HomeShots',
    'AS': 'AwayShots',
    'HST': 'HomeShotsOnTarget',
    'AST': 'AwayShotsOnTarget',
    'HHW': 'HomeHitWoodwork',
    'AHW': 'AwayHitWoodwork',
    'HC': 'HomeCorners',
    'AC': 'AwayCorners',
    'HF': 'HomeFouls',
    'AF': 'AwayFouls',
    'HO': 'HomeOffsides',
    'AO': 'AwayOffsides',
    'HY': 'HomeYellowCards',
    'AY': 'AwayYellowCards',
    'HR': 'HomeRedCards',
    'AR': 'AwayRedCards',
    'HBP': 'HomeBookingPoints',
    'ABP': 'AwayBookingPoints',

    # Betting Odds - Bet365
    'B365H': 'Bet365_HomeWinOdds',
    'B365D': 'Bet365_DrawOdds',
    'B365A': 'Bet365_AwayWinOdds',
    'B365>2.5': 'Bet365_Over2_5GoalsOdds',
    'B365<2.5': 'Bet365_Under2_5GoalsOdds',
    'B365AHH': 'Bet365_AH_HomeOdds',
    'B365AHA': 'Bet365_AH_AwayOdds',
    'B365AH': 'Bet365_AH_Handicap',

    # Betting Odds - Other Bookmakers
    'BSH': 'BlueSquare_HomeWinOdds',
    'BSD': 'BlueSquare_DrawOdds',
    'BSA': 'BlueSquare_AwayWinOdds',
    'BWH': 'BetWin_HomeWinOdds',
    'BWD': 'BetWin_DrawOdds',
    'BWA': 'BetWin_AwayWinOdds',
    'GBH': 'Gamebookers_HomeWinOdds',
    'GBD': 'Gamebookers_DrawOdds',
    'GBA': 'Gamebookers_AwayWinOdds',
    'IWH': 'Interwetten_HomeWinOdds',
    'IWD': 'Interwetten_DrawOdds',
    'IWA': 'Interwetten_AwayWinOdds',
    'LBH': 'Ladbrokes_HomeWinOdds',
    'LBD': 'Ladbrokes_DrawOdds',
    'LBA': 'Ladbrokes_AwayWinOdds',
    'PSH': 'Pinnacle_HomeWinOdds',
    'PSD': 'Pinnacle_DrawOdds',
    'PSA': 'Pinnacle_AwayWinOdds',
    'SBH': 'SportingBet_HomeWinOdds',
    'SBD': 'SportingBet_DrawOdds',
    'SBA': 'SportingBet_AwayWinOdds',
    'SJH': 'StanJames_HomeWinOdds',
    'SJD': 'StanJames_DrawOdds',
    'SJA': 'StanJames_AwayWinOdds',
    'SYH': 'Stanleybet_HomeWinOdds',
    'SYD': 'Stanleybet_DrawOdds',
    'SYA': 'Stanleybet_AwayWinOdds',
    'VCH': 'VCBet_HomeWinOdds',
    'VCD': 'VCBet_DrawOdds',
    'VCA': 'VCBet_AwayWinOdds',
    'WHH': 'WilliamHill_HomeWinOdds',
    'WHD': 'WilliamHill_DrawOdds',
    'WHA': 'WilliamHill_AwayWinOdds',

    # Max/Avg Odds
    'MaxH': 'Max_HomeWinOdds',
    'MaxD': 'Max_DrawOdds',
    'MaxA': 'Max_AwayWinOdds',
    'AvgH': 'Avg_HomeWinOdds',
    'AvgD': 'Avg_DrawOdds',
    'AvgA': 'Avg_AwayWinOdds',

    # Betbrain (Bb) Odds
    'Bb1X2': 'Betbrain_NumBookmakers',
    'BbMxH': 'Betbrain_MaxHomeWinOdds',
    'BbAvH': 'Betbrain_AvgHomeWinOdds',
    'BbMxD': 'Betbrain_MaxDrawOdds',
    'BbAvD': 'Betbrain_AvgDrawOdds',
    'BbMxA': 'Betbrain_MaxAwayWinOdds',
    'BbAvA': 'Betbrain_AvgAwayWinOdds',

    # Over/Under Odds
    'BbOU': 'Betbrain_NumBookmakers_OverUnder',
    'BbMx>2.5': 'Betbrain_MaxOver2_5GoalsOdds',
    'BbAv>2.5': 'Betbrain_AvgOver2_5GoalsOdds',
    'BbMx<2.5': 'Betbrain_MaxUnder2_5GoalsOdds',
    'BbAv<2.5': 'Betbrain_AvgUnder2_5GoalsOdds',

    # Asian Handicap Odds
    'BbAH': 'Betbrain_NumBookmakers_AH',
    'BbAHh': 'Betbrain_HandicapSize_Home',
    'BbMxAHH': 'Betbrain_MaxAH_HomeOdds',
    'BbAvAHH': 'Betbrain_AvgAH_HomeOdds',
    'BbMxAHA': 'Betbrain_MaxAH_AwayOdds',
    'BbAvAHA': 'Betbrain_AvgAH_AwayOdds',

    # Corners, Cards, Booking Points Odds (if present)
    'BbMxHC': 'Betbrain_MaxHomeCornersOdds',
    'BbAvHC': 'Betbrain_AvgHomeCornersOdds',
    'BbMxAC': 'Betbrain_MaxAwayCornersOdds',
    'BbAvAC': 'Betbrain_AvgAwayCornersOdds',
    'BbMxHY': 'Betbrain_MaxHomeYellowOdds',
    'BbAvHY': 'Betbrain_AvgHomeYellowOdds',
    'BbMxAY': 'Betbrain_MaxAwayYellowOdds',
    'BbAvAY': 'Betbrain_AvgAwayYellowOdds',
    'BbMxHR': 'Betbrain_MaxHomeRedOdds',
    'BbAvHR': 'Betbrain_AvgHomeRedOdds',
    'BbMxAR': 'Betbrain_MaxAwayRedOdds',
    'BbAvAR': 'Betbrain_AvgAwayRedOdds',

    # Miscellaneous
    'PSCH': 'Pinnacle_ClosingHomeOdds',
    'PSCD': 'Pinnacle_ClosingDrawOdds',
    'PSCA': 'Pinnacle_ClosingAwayOdds',
    'B365CH': 'Bet365_ClosingHomeOdds',
    'B365CD': 'Bet365_ClosingDrawOdds',
    'B365CA': 'Bet365_ClosingAwayOdds',
}

# Example usage:
historical_data.rename(columns=column_rename_map, inplace=True)
historical_data.dropna(axis=1, how='all', inplace=True)
historical_data.drop(columns=['Division'], inplace=True, errors='ignore')

# Parse MatchDate to datetime (already in YYYY-MM-DD format)
historical_data['MatchDate'] = pd.to_datetime(historical_data['MatchDate'], errors='coerce')

# HomeWin, AwayWin, Draw columns
historical_data['HomeWin'] = (historical_data['FullTimeResult'] == 'H').astype(int)
historical_data['AwayWin'] = (historical_data['FullTimeResult'] == 'A').astype(int)
historical_data['Draw'] = (historical_data['FullTimeResult'] == 'D').astype(int)

# WinningTeam column
historical_data['WinningTeam'] = np.where(
    historical_data['FullTimeResult'] == 'H',
    historical_data['HomeTeam'],
    np.where(
        historical_data['FullTimeResult'] == 'A',
        historical_data['AwayTeam'],
        np.nan
    )
)

# Half-time win columns (optional)
historical_data['HalfTimeHomeWin'] = (historical_data['HalfTimeResult'] == 'H').astype(int)
historical_data['HalfTimeAwayWin'] = (historical_data['HalfTimeResult'] == 'A').astype(int)
historical_data['HalfTimeDraw'] = (historical_data['HalfTimeResult'] == 'D').astype(int)

historical_data.to_csv(path.join(DATA_DIR, 'combined_historical_data.csv'), sep='\t', index=False)

home_teams = historical_data['HomeTeam'].unique().tolist()
# print("Home Teams:")
# print(home_teams[:50])
away_teams = historical_data['AwayTeam'].unique().tolist()
# print("Away Teams:")
# print(away_teams[:50])
all_teams = set(home_teams).union(set(away_teams))
# print("All Teams:")
# print(all_teams)
all_teams = pd.DataFrame(all_teams, columns=['Team'])
# print("All Teams DataFrame:")
# print(all_teams)
all_teams['TeamId'] = (
    all_teams['Team']
    .str.replace("'", "", regex=False)   # Remove apostrophes
    .str.replace(' ', '_', regex=False)  # Replace spaces with underscores
).str.lower()

# Calculate stats and merge into all_teams
def add_team_stat(all_teams, historical_data, group_col, value_col, stat_name, func='mean'):
    stat_df = historical_data.groupby(group_col)[value_col].agg(func).reset_index()
    stat_df = stat_df.rename(columns={group_col: 'Team', value_col: stat_name})
    return all_teams.merge(stat_df, on='Team', how='left')

# Averages and totals for goals
all_teams = add_team_stat(all_teams, historical_data, 'HomeTeam', 'FullTimeHomeGoals', 'HomeGoalsAve', 'mean')
all_teams = add_team_stat(all_teams, historical_data, 'AwayTeam', 'FullTimeAwayGoals', 'AwayGoalsAve', 'mean')
all_teams = add_team_stat(all_teams, historical_data, 'HomeTeam', 'FullTimeHomeGoals', 'HomeGoalsTotal', 'sum')
all_teams = add_team_stat(all_teams, historical_data, 'AwayTeam', 'FullTimeAwayGoals', 'AwayGoalsTotal', 'sum')
all_teams = add_team_stat(all_teams, historical_data, 'HomeTeam', 'HalfTimeHomeGoals', 'HomeGoalsHalfAve', 'mean')
all_teams = add_team_stat(all_teams, historical_data, 'AwayTeam', 'HalfTimeAwayGoals', 'AwayGoalsHalfAve', 'mean')
all_teams = add_team_stat(all_teams, historical_data, 'HomeTeam', 'HalfTimeHomeGoals', 'HomeGoalsHalfTotal', 'sum')
all_teams = add_team_stat(all_teams, historical_data, 'AwayTeam', 'HalfTimeAwayGoals', 'AwayGoalsHalfTotal', 'sum')

# Shots and shots on target
all_teams = add_team_stat(all_teams, historical_data, 'HomeTeam', 'HomeShots', 'HomeShotsAve', 'mean')
all_teams = add_team_stat(all_teams, historical_data, 'AwayTeam', 'AwayShots', 'AwayShotsAve', 'mean')
all_teams = add_team_stat(all_teams, historical_data, 'HomeTeam', 'HomeShots', 'HomeShotsTotal', 'sum')
all_teams = add_team_stat(all_teams, historical_data, 'AwayTeam', 'AwayShots', 'AwayShotsTotal', 'sum')
all_teams = add_team_stat(all_teams, historical_data, 'HomeTeam', 'HomeShotsOnTarget', 'HomeShotsOnTargetAve', 'mean')
all_teams = add_team_stat(all_teams, historical_data, 'AwayTeam', 'AwayShotsOnTarget', 'AwayShotsOnTargetAve', 'mean')

# Differentials (these are not per-team stats, but you can calculate averages per team if needed)
# If you want per-team average differential as home/away:
home_diff = historical_data.copy()
home_diff['HomeFirstHalfDifferential'] = home_diff['HalfTimeHomeGoals'] - home_diff['HalfTimeAwayGoals']
home_diff['HomeGameDifferential'] = home_diff['FullTimeHomeGoals'] - home_diff['FullTimeAwayGoals']
home_diff['HomeSecondHalfGoals'] = home_diff['FullTimeHomeGoals'] - home_diff['HalfTimeHomeGoals']
home_diff['HomeFirstToSecondHalfGoalRatio'] = home_diff['HomeFirstHalfDifferential'] / home_diff['HomeSecondHalfGoals'].replace(0, np.nan)

away_diff = historical_data.copy()
away_diff['AwayFirstHalfDifferential'] = away_diff['HalfTimeAwayGoals'] - away_diff['HalfTimeHomeGoals']
away_diff['AwayGameDifferential'] = away_diff['FullTimeAwayGoals'] - away_diff['FullTimeHomeGoals']
away_diff['AwaySecondHalfGoals'] = away_diff['FullTimeAwayGoals'] - away_diff['HalfTimeAwayGoals']
away_diff['AwayFirstToSecondHalfGoalRatio'] = away_diff['AwayFirstHalfDifferential'] / away_diff['AwaySecondHalfGoals'].replace(0, np.nan)

all_teams = add_team_stat(all_teams, home_diff, 'HomeTeam', 'HomeFirstHalfDifferential', 'HomeFirstHalfDifferentialAve', 'mean')
all_teams = add_team_stat(all_teams, away_diff, 'AwayTeam', 'AwayFirstHalfDifferential', 'AwayFirstHalfDifferentialAve', 'mean')
all_teams = add_team_stat(all_teams, home_diff, 'HomeTeam', 'HomeGameDifferential', 'HomeGameDifferentialAve', 'mean')
all_teams = add_team_stat(all_teams, away_diff, 'AwayTeam', 'AwayGameDifferential', 'AwayGameDifferentialAve', 'mean')
all_teams = add_team_stat(all_teams, home_diff, 'HomeTeam', 'HomeFirstToSecondHalfGoalRatio', 'HomeFirstToSecondHalfGoalRatioAve', 'mean')
all_teams = add_team_stat(all_teams, away_diff, 'AwayTeam', 'AwayFirstToSecondHalfGoalRatio', 'AwayFirstToSecondHalfGoalRatioAve', 'mean')


# all_teams['HomeGoalsAve'] = historical_data[historical_data['HomeTeam'].isin(all_teams['Team'])].groupby('HomeTeam')['FullTimeHomeGoals'].mean()
# all_teams['AwayGoalsAve'] = historical_data[historical_data['AwayTeam'].isin(all_teams['Team'])].groupby('AwayTeam')['FullTimeAwayGoals'].mean()
# all_teams['HomeGoalsTotal'] = historical_data[historical_data['HomeTeam'].isin(all_teams['Team'])].groupby('HomeTeam')['FullTimeHomeGoals'].sum()
# all_teams['AwayGoalsTotal'] = historical_data[historical_data['AwayTeam'].isin(all_teams['Team'])].groupby('AwayTeam')['FullTimeAwayGoals'].sum()
# all_teams['HomeGoalsHalfAve'] = historical_data[historical_data['HomeTeam'].isin(all_teams['Team'])].groupby('HomeTeam')['HalfTimeHomeGoals'].mean()
# all_teams['AwayGoalsHalfAve'] = historical_data[historical_data['AwayTeam'].isin(all_teams['Team'])].groupby('AwayTeam')['HalfTimeAwayGoals'].mean()
# all_teams['HomeGoalsHalfTotal'] = historical_data[historical_data['HomeTeam'].isin(all_teams['Team'])].groupby('HomeTeam')['HalfTimeHomeGoals'].sum()
# all_teams['AwayGoalsHalfTotal'] = historical_data[historical_data['AwayTeam'].isin(all_teams['Team'])].groupby('AwayTeam')['HalfTimeAwayGoals'].sum()
# all_teams['HomeShotsAve'] = historical_data[historical_data['HomeTeam'].isin(all_teams['Team'])].groupby('HomeTeam')['HomeShots'].mean()
# all_teams['AwayShotsAve'] = historical_data[historical_data['AwayTeam'].isin(all_teams['Team'])].groupby('AwayTeam')['AwayShots'].mean()
# all_teams['HomeShotsTotal'] = historical_data[historical_data['HomeTeam'].isin(all_teams['Team'])].groupby('HomeTeam')['HomeShots'].sum()
# all_teams['AwayShotsTotal'] = historical_data[historical_data['AwayTeam'].isin(all_teams['Team'])].groupby('AwayTeam')['AwayShots'].sum()
# all_teams['HomeShotsOnTargetAve'] = historical_data[historical_data['HomeTeam'].isin(all_teams['Team'])].groupby('HomeTeam')['HomeShotsOnTarget'].mean()
# all_teams['AwayShotsOnTargetAve'] = historical_data[historical_data['AwayTeam'].isin(all_teams['Team'])].groupby('AwayTeam')['AwayShotsOnTarget'].mean()
# all_teams['HomeFirstHalfDifferential'] = historical_data['HalfTimeHomeGoals'] - historical_data['HalfTimeAwayGoals']
# all_teams['AwayFirstHalfDifferential'] = historical_data['HalfTimeAwayGoals'] - historical_data['HalfTimeHomeGoals']
# all_teams['HomeGameDifferential'] = historical_data['FullTimeHomeGoals'] - historical_data['FullTimeAwayGoals']
# all_teams['AwayGameDifferential'] = historical_data['FullTimeAwayGoals'] - historical_data['FullTimeHomeGoals']

print(all_teams.head(50))

# Calculate points for each match
def get_points(result):
    if result == 'H':
        return 3, 0
    elif result == 'A':
        return 0, 3
    elif result == 'D':
        return 1, 1
    return 0, 0

historical_data['HomePoints'], historical_data['AwayPoints'] = zip(*historical_data['FullTimeResult'].map(get_points))

# Sort by date for rolling calculations
historical_data = historical_data.sort_values('MatchDate')

# Rolling sum of points for last 5 games (recent form)
historical_data['HomeTeamPointsLast5'] = (
    historical_data.groupby('HomeTeam')['HomePoints']
    .rolling(window=5, min_periods=1).sum().reset_index(0, drop=True)
)
historical_data['AwayTeamPointsLast5'] = (
    historical_data.groupby('AwayTeam')['AwayPoints']
    .rolling(window=5, min_periods=1).sum().reset_index(0, drop=True)
)

def calc_h2h(row, df, n=5):
    mask = (
        ((df['HomeTeam'] == row['HomeTeam']) & (df['AwayTeam'] == row['AwayTeam'])) |
        ((df['HomeTeam'] == row['AwayTeam']) & (df['AwayTeam'] == row['HomeTeam']))
    )
    prev_matches = df[mask & (df['MatchDate'] < row['MatchDate'])].sort_values('MatchDate', ascending=False).head(n)
    home_wins = ((prev_matches['HomeTeam'] == row['HomeTeam']) & (prev_matches['FullTimeResult'] == 'H')).sum()
    away_wins = ((prev_matches['AwayTeam'] == row['AwayTeam']) & (prev_matches['FullTimeResult'] == 'A')).sum()
    draws = (prev_matches['FullTimeResult'] == 'D').sum()
    return pd.Series([home_wins, away_wins, draws])

historical_data[['HomeH2HWinLast5', 'AwayH2HWinLast5', 'H2HDrawLast5']] = historical_data.apply(
    lambda row: calc_h2h(row, historical_data), axis=1
)

historical_data['Season'] = historical_data['MatchDate'].dt.year  # Adjust if your season format is different

# Cumulative points up to each match
historical_data['HomeTeamCumulativePoints'] = (
    historical_data.groupby(['Season', 'HomeTeam'])['HomePoints'].cumsum() - historical_data['HomePoints']
)
historical_data['AwayTeamCumulativePoints'] = (
    historical_data.groupby(['Season', 'AwayTeam'])['AwayPoints'].cumsum() - historical_data['AwayPoints']
)

historical_data['MatchDate'] = pd.to_datetime(historical_data['MatchDate'], errors='coerce')

def last_match_gap(df, team_col, date_col):
    df = df.sort_values(date_col)
    last_dates = df.groupby(team_col)[date_col].shift(1)
    return (df[date_col] - last_dates).dt.days

historical_data['HomeRestDays'] = last_match_gap(historical_data, 'HomeTeam', 'MatchDate')
historical_data['AwayRestDays'] = last_match_gap(historical_data, 'AwayTeam', 'MatchDate')

# print(all_teams.columns.tolist())

# Save the combined historical data
all_teams.to_csv(path.join(DATA_DIR, 'all_teams.csv'), sep='\t', index=False)
# print(historical_data.head(50))

# 'Team', 'TeamId', 'HomeGoalsAve', 'AwayGoalsAve', 'HomeGoalsTotal', 'AwayGoalsTotal', 'HomeGoalsHalfAve', 'AwayGoalsHalfAve', 'HomeGoalsHalfTotal', 'AwayGoalsHalfTotal', 'HomeShotsAve', 'AwayShotsAve', 'HomeShotsTotal', 'AwayShotsTotal', 'HomeShotsOnTargetAve', 'AwayShotsOnTargetAve', 'HomeFirstHalfDifferentialAve', 'AwayFirstHalfDifferentialAve', 'HomeGameDifferentialAve', 'AwayGameDifferentialAve'



historical_data_with_calculations = historical_data.copy()
historical_data_with_calculations = pd.merge(
    historical_data_with_calculations,
    all_teams[['Team', 'TeamId', 'HomeGoalsAve', 'HomeGoalsTotal',  'HomeGoalsHalfAve', 'HomeGoalsHalfTotal', 'HomeShotsAve',  
        'HomeShotsTotal', 'HomeShotsOnTargetAve', 'HomeFirstHalfDifferentialAve', 'HomeGameDifferentialAve',  'HomeFirstToSecondHalfGoalRatioAve']],
    left_on='HomeTeam', right_on='Team', how='left', suffixes=('', '_Home')
)
historical_data_with_calculations = pd.merge(
    historical_data_with_calculations,
    all_teams[['Team', 'AwayGoalsAve',  'AwayGoalsTotal',  'AwayGoalsHalfAve', 'AwayGoalsHalfTotal',  
        'AwayShotsAve', 'AwayShotsTotal',  'AwayShotsOnTargetAve', 'AwayFirstHalfDifferentialAve',  'AwayGameDifferentialAve',  'AwayFirstToSecondHalfGoalRatioAve']],
    left_on='AwayTeam', right_on='Team', how='left', suffixes=('', '_Away')
)

# Optionally drop the extra 'Team' columns from the merges
historical_data_with_calculations.drop(columns=['Team', 'Team_Away'], inplace=True, errors='ignore')

def smart_imputation(df):
    """Use KNN imputation for missing values in numeric columns"""
    print("Applying KNN imputation for missing data...")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    # Only impute columns that have at least some non-null values
    cols_to_impute = [col for col in numeric_cols if df[col].notna().sum() > 0]
    
    if len(cols_to_impute) > 0:
        imputer = KNNImputer(n_neighbors=5)
        imputed_array = imputer.fit_transform(df[cols_to_impute])
        imputed_df = pd.DataFrame(imputed_array, columns=cols_to_impute, index=df.index)
        df[cols_to_impute] = imputed_df
    
    # For columns that are all null, fill with 0 or mean if available
    for col in numeric_cols:
        if df[col].isnull().all():
            df[col] = df[col].fillna(0)
        elif df[col].isnull().any():
            # If still has nulls after KNN (shouldn't happen), fill with mean
            df[col] = df[col].fillna(df[col].mean())
    
    print(f"Imputed {len(cols_to_impute)} numeric columns")
    return df

def extract_betting_features(df):
    """
    Extract advanced features from betting odds
    Odds data is already available from football-data.co.uk
    """
    print("Extracting betting market features...")

    # Implied probabilities from Bet365 odds
    if 'Bet365_HomeWinOdds' in df.columns:
        # Convert odds to implied probabilities
        df['ImpliedProb_HomeWin'] = 1 / df['Bet365_HomeWinOdds']
        df['ImpliedProb_Draw'] = 1 / df['Bet365_DrawOdds']
        df['ImpliedProb_AwayWin'] = 1 / df['Bet365_AwayWinOdds']

        # Normalize to sum to 1 (remove bookmaker margin)
        total_prob = (df['ImpliedProb_HomeWin'] + df['ImpliedProb_Draw'] + df['ImpliedProb_AwayWin'])
        df['ImpliedProb_HomeWin_Norm'] = df['ImpliedProb_HomeWin'] / total_prob
        df['ImpliedProb_Draw_Norm'] = df['ImpliedProb_Draw'] / total_prob
        df['ImpliedProb_AwayWin_Norm'] = df['ImpliedProb_AwayWin'] / total_prob

        # Market confidence (how much margin the bookmaker is taking)
        df['Bet365_MarketMargin'] = total_prob - 1

        # Odds movement (compare Bet365 vs William Hill if available)
        if 'WilliamHill_HomeWinOdds' in df.columns:
            df['OddsMovement_Home'] = df['Bet365_HomeWinOdds'] - df['WilliamHill_HomeWinOdds']
            df['OddsMovement_Away'] = df['Bet365_AwayWinOdds'] - df['WilliamHill_AwayWinOdds']
            df['OddsMovement_Draw'] = df['Bet365_DrawOdds'] - df['WilliamHill_DrawOdds']
        else:
            df['OddsMovement_Home'] = 0
            df['OddsMovement_Away'] = 0
            df['OddsMovement_Draw'] = 0

        # Odds value indicators (when odds suggest better value than actual probability)
        # Lower odds number = higher probability
        df['Bet365_Value_Home'] = df['ImpliedProb_HomeWin_Norm'] - (1 / df['Bet365_HomeWinOdds'])
        df['Bet365_Value_Away'] = df['ImpliedProb_AwayWin_Norm'] - (1 / df['Bet365_AwayWinOdds'])
        df['Bet365_Value_Draw'] = df['ImpliedProb_Draw_Norm'] - (1 / df['Bet365_DrawOdds'])

        # Odds ratio features
        df['Bet365_HomeVsDraw_Ratio'] = df['Bet365_HomeWinOdds'] / df['Bet365_DrawOdds']
        df['Bet365_AwayVsDraw_Ratio'] = df['Bet365_AwayWinOdds'] / df['Bet365_DrawOdds']
        df['Bet365_HomeVsAway_Ratio'] = df['Bet365_HomeWinOdds'] / df['Bet365_AwayWinOdds']

    # Over/Under odds features
    if 'Bet365_Over2_5GoalsOdds' in df.columns and 'Bet365_Under2_5GoalsOdds' in df.columns:
        df['Bet365_OverUnder_Margin'] = (1 / df['Bet365_Over2_5GoalsOdds']) + (1 / df['Bet365_Under2_5GoalsOdds']) - 1
        df['Bet365_ExpectedTotalGoals'] = (df['Bet365_Over2_5GoalsOdds'] + df['Bet365_Under2_5GoalsOdds']) / 2

    # Asian Handicap features (if available)
    if 'Bet365_AH_HomeOdds' in df.columns and 'Bet365_AH_AwayOdds' in df.columns:
        ah_total = (1 / df['Bet365_AH_HomeOdds']) + (1 / df['Bet365_AH_AwayOdds'])
        df['Bet365_AH_Margin'] = ah_total - 1

    print(f"Added {len([col for col in df.columns if 'Bet365' in col or 'Implied' in col or 'OddsMovement' in col])} betting features")
    return df

# Extract betting features
historical_data_with_calculations = extract_betting_features(historical_data_with_calculations)

# Add injury data
print("Scraping injury data from PremierInjuries.com...")
injury_df = scrape_premier_injuries()
historical_data_with_calculations = create_injury_features(historical_data_with_calculations, injury_df)

# Add weather data
print("Adding weather data from Open-Meteo (completely free)...")
try:
    historical_data_with_calculations = add_weather_features(historical_data_with_calculations)
    historical_data_with_calculations = add_weather_impact_category(historical_data_with_calculations)
    print("Weather data integration completed")
except Exception as e:
    print(f"Warning: Weather data integration failed: {e}")
    print("Continuing without weather data...")

def calculate_advanced_metrics(df):
    """Calculate advanced team performance metrics from HISTORICAL data only"""
    
    df = df.sort_values('MatchDate').reset_index(drop=True)
    
    # First, calculate match-level metrics (these will be shifted to create historical averages)
    df['xG_Home_Match'] = (df['HomeShotsOnTarget'] * 0.35 + df['HomeShots'] * 0.10)
    df['xG_Away_Match'] = (df['AwayShotsOnTarget'] * 0.35 + df['AwayShots'] * 0.10)
    df['ShootingEff_Home_Match'] = df['FullTimeHomeGoals'] / (df['HomeShots'] + 0.1)
    df['ShootingEff_Away_Match'] = df['FullTimeAwayGoals'] / (df['AwayShots'] + 0.1)
    df['GoalDiff_Home_Match'] = df['FullTimeHomeGoals'] - df['FullTimeAwayGoals']
    df['GoalDiff_Away_Match'] = df['FullTimeAwayGoals'] - df['FullTimeHomeGoals']
    
    # Add Poisson-based goal probability features
    # Calculate probabilities for different goal ranges using Poisson distribution
    df['Home_Poisson_0_Goals'] = stats.poisson.pmf(0, df['xG_Home_Match'])
    df['Home_Poisson_1_Goal'] = stats.poisson.pmf(1, df['xG_Home_Match'])
    df['Home_Poisson_2_3_Goals'] = (stats.poisson.pmf(2, df['xG_Home_Match']) + 
                                   stats.poisson.pmf(3, df['xG_Home_Match']))
    df['Home_Poisson_4Plus_Goals'] = 1 - (df['Home_Poisson_0_Goals'] + df['Home_Poisson_1_Goal'] + 
                                          df['Home_Poisson_2_3_Goals'])
    
    df['Away_Poisson_0_Goals'] = stats.poisson.pmf(0, df['xG_Away_Match'])
    df['Away_Poisson_1_Goal'] = stats.poisson.pmf(1, df['xG_Away_Match'])
    df['Away_Poisson_2_3_Goals'] = (stats.poisson.pmf(2, df['xG_Away_Match']) + 
                                   stats.poisson.pmf(3, df['xG_Away_Match']))
    df['Away_Poisson_4Plus_Goals'] = 1 - (df['Away_Poisson_0_Goals'] + df['Away_Poisson_1_Goal'] + 
                                          df['Away_Poisson_2_3_Goals'])
    
    # Calculate Poisson-based expected points (simplified xPTS)
    # Points = 3 * P(Home win) + 1 * P(Draw) + 0 * P(Away win)
    # Using Poisson to estimate scoreline probabilities
    home_win_prob = 0
    draw_prob = 0
    away_win_prob = 0
    
    for home_goals in range(5):  # Reasonable range for goals
        for away_goals in range(5):
            score_prob = (stats.poisson.pmf(home_goals, df['xG_Home_Match']) * 
                         stats.poisson.pmf(away_goals, df['xG_Away_Match']))
            if home_goals > away_goals:
                home_win_prob += score_prob
            elif home_goals == away_goals:
                draw_prob += score_prob
            else:
                away_win_prob += score_prob
    
    df['Poisson_Home_xPTS'] = 3 * home_win_prob + 1 * draw_prob
    df['Poisson_Away_xPTS'] = 3 * away_win_prob + 1 * draw_prob
    
    # Now create rolling averages from PAST matches only (using shift to exclude current match)
    # Home team metrics
    df['HomexG_Avg_L5'] = df.groupby('HomeTeam')['xG_Home_Match'].shift(1).rolling(5, min_periods=1).mean().reset_index(level=0, drop=True)
    df['HomeShootingEff_Avg_L5'] = df.groupby('HomeTeam')['ShootingEff_Home_Match'].shift(1).rolling(5, min_periods=1).mean().reset_index(level=0, drop=True)
    df['HomeMomentum_L3'] = df.groupby('HomeTeam')['FullTimeHomeGoals'].shift(1).rolling(3, min_periods=1).sum().reset_index(level=0, drop=True)
    df['HomeGoalDiff_Avg_L5'] = df.groupby('HomeTeam')['GoalDiff_Home_Match'].shift(1).rolling(5, min_periods=1).mean().reset_index(level=0, drop=True)
    
    # Away team metrics
    df['AwayxG_Avg_L5'] = df.groupby('AwayTeam')['xG_Away_Match'].shift(1).rolling(5, min_periods=1).mean().reset_index(level=0, drop=True)
    df['AwayShootingEff_Avg_L5'] = df.groupby('AwayTeam')['ShootingEff_Away_Match'].shift(1).rolling(5, min_periods=1).mean().reset_index(level=0, drop=True)
    df['AwayMomentum_L3'] = df.groupby('AwayTeam')['FullTimeAwayGoals'].shift(1).rolling(3, min_periods=1).sum().reset_index(level=0, drop=True)
    df['AwayGoalDiff_Avg_L5'] = df.groupby('AwayTeam')['GoalDiff_Away_Match'].shift(1).rolling(5, min_periods=1).mean().reset_index(level=0, drop=True)
    
    # Poisson-based rolling averages
    df['Home_Poisson_0_Goals_Avg_L5'] = df.groupby('HomeTeam')['Home_Poisson_0_Goals'].shift(1).rolling(5, min_periods=1).mean().reset_index(level=0, drop=True)
    df['Home_Poisson_1_Goal_Avg_L5'] = df.groupby('HomeTeam')['Home_Poisson_1_Goal'].shift(1).rolling(5, min_periods=1).mean().reset_index(level=0, drop=True)
    df['Home_Poisson_2_3_Goals_Avg_L5'] = df.groupby('HomeTeam')['Home_Poisson_2_3_Goals'].shift(1).rolling(5, min_periods=1).mean().reset_index(level=0, drop=True)
    df['Home_Poisson_4Plus_Goals_Avg_L5'] = df.groupby('HomeTeam')['Home_Poisson_4Plus_Goals'].shift(1).rolling(5, min_periods=1).mean().reset_index(level=0, drop=True)
    df['Home_Poisson_xPTS_Avg_L5'] = df.groupby('HomeTeam')['Poisson_Home_xPTS'].shift(1).rolling(5, min_periods=1).mean().reset_index(level=0, drop=True)
    
    df['Away_Poisson_0_Goals_Avg_L5'] = df.groupby('AwayTeam')['Away_Poisson_0_Goals'].shift(1).rolling(5, min_periods=1).mean().reset_index(level=0, drop=True)
    df['Away_Poisson_1_Goal_Avg_L5'] = df.groupby('AwayTeam')['Away_Poisson_1_Goal'].shift(1).rolling(5, min_periods=1).mean().reset_index(level=0, drop=True)
    df['Away_Poisson_2_3_Goals_Avg_L5'] = df.groupby('AwayTeam')['Away_Poisson_2_3_Goals'].shift(1).rolling(5, min_periods=1).mean().reset_index(level=0, drop=True)
    df['Away_Poisson_4Plus_Goals_Avg_L5'] = df.groupby('AwayTeam')['Away_Poisson_4Plus_Goals'].shift(1).rolling(5, min_periods=1).mean().reset_index(level=0, drop=True)
    df['Away_Poisson_xPTS_Avg_L5'] = df.groupby('AwayTeam')['Poisson_Away_xPTS'].shift(1).rolling(5, min_periods=1).mean().reset_index(level=0, drop=True)
    
    # Drop intermediate match-level calculations
    df = df.drop(columns=['xG_Home_Match', 'xG_Away_Match', 'ShootingEff_Home_Match', 
                          'ShootingEff_Away_Match', 'GoalDiff_Home_Match', 'GoalDiff_Away_Match',
                          'Home_Poisson_0_Goals', 'Home_Poisson_1_Goal', 'Home_Poisson_2_3_Goals', 'Home_Poisson_4Plus_Goals',
                          'Away_Poisson_0_Goals', 'Away_Poisson_1_Goal', 'Away_Poisson_2_3_Goals', 'Away_Poisson_4Plus_Goals',
                          'Poisson_Home_xPTS', 'Poisson_Away_xPTS'])
    
    # Fill NaN values for first matches with reasonable defaults
    df['HomexG_Avg_L5'] = df['HomexG_Avg_L5'].fillna(1.5)
    df['AwayxG_Avg_L5'] = df['AwayxG_Avg_L5'].fillna(1.5)
    df['HomeShootingEff_Avg_L5'] = df['HomeShootingEff_Avg_L5'].fillna(0.15)
    df['AwayShootingEff_Avg_L5'] = df['AwayShootingEff_Avg_L5'].fillna(0.15)
    df['HomeMomentum_L3'] = df['HomeMomentum_L3'].fillna(3.0)
    df['AwayMomentum_L3'] = df['AwayMomentum_L3'].fillna(3.0)
    df['HomeGoalDiff_Avg_L5'] = df['HomeGoalDiff_Avg_L5'].fillna(0.0)
    df['AwayGoalDiff_Avg_L5'] = df['AwayGoalDiff_Avg_L5'].fillna(0.0)
    
    # Fill NaN values for Poisson features with reasonable defaults
    df['Home_Poisson_0_Goals_Avg_L5'] = df['Home_Poisson_0_Goals_Avg_L5'].fillna(0.3)  # ~30% chance of 0 goals
    df['Home_Poisson_1_Goal_Avg_L5'] = df['Home_Poisson_1_Goal_Avg_L5'].fillna(0.35)  # ~35% chance of 1 goal
    df['Home_Poisson_2_3_Goals_Avg_L5'] = df['Home_Poisson_2_3_Goals_Avg_L5'].fillna(0.25)  # ~25% chance of 2-3 goals
    df['Home_Poisson_4Plus_Goals_Avg_L5'] = df['Home_Poisson_4Plus_Goals_Avg_L5'].fillna(0.1)  # ~10% chance of 4+ goals
    df['Home_Poisson_xPTS_Avg_L5'] = df['Home_Poisson_xPTS_Avg_L5'].fillna(1.5)  # Average points
    
    df['Away_Poisson_0_Goals_Avg_L5'] = df['Away_Poisson_0_Goals_Avg_L5'].fillna(0.3)
    df['Away_Poisson_1_Goal_Avg_L5'] = df['Away_Poisson_1_Goal_Avg_L5'].fillna(0.35)
    df['Away_Poisson_2_3_Goals_Avg_L5'] = df['Away_Poisson_2_3_Goals_Avg_L5'].fillna(0.25)
    df['Away_Poisson_4Plus_Goals_Avg_L5'] = df['Away_Poisson_4Plus_Goals_Avg_L5'].fillna(0.1)
    df['Away_Poisson_xPTS_Avg_L5'] = df['Away_Poisson_xPTS_Avg_L5'].fillna(1.5)
    return df

def calculate_referee_statistics(df):
    """
    Calculate referee statistics from historical data
    Creates features based on referee disciplinary tendencies
    """
    print("Calculating referee statistics...")

    # Calculate per-referee statistics
    referee_stats = []

    for referee in df['Referee'].unique():
        ref_matches = df[df['Referee'] == referee]

        # Basic counts
        total_matches = len(ref_matches)
        total_yellow_cards = (ref_matches['HomeYellowCards'] + ref_matches['AwayYellowCards']).sum()
        total_red_cards = (ref_matches['HomeRedCards'] + ref_matches['AwayRedCards']).sum()
        total_fouls = (ref_matches['HomeFouls'] + ref_matches['AwayFouls']).sum()

        # Per-game averages
        yellow_cards_per_game = total_yellow_cards / total_matches
        red_cards_per_game = total_red_cards / total_matches
        fouls_per_game = total_fouls / total_matches

        # Home vs Away bias
        home_yellow_cards = ref_matches['HomeYellowCards'].sum()
        away_yellow_cards = ref_matches['AwayYellowCards'].sum()
        home_advantage_yellow = (home_yellow_cards - away_yellow_cards) / total_matches

        # Match outcomes when officiating
        home_wins = (ref_matches['FullTimeResult'] == 'H').sum()
        away_wins = (ref_matches['FullTimeResult'] == 'A').sum()
        draws = (ref_matches['FullTimeResult'] == 'D').sum()

        home_win_rate = home_wins / total_matches
        away_win_rate = away_wins / total_matches
        draw_rate = draws / total_matches

        referee_stats.append({
            'Referee': referee,
            'RefTotalMatches': total_matches,
            'RefYellowCardsPerGame': yellow_cards_per_game,
            'RefRedCardsPerGame': red_cards_per_game,
            'RefFoulsPerGame': fouls_per_game,
            'RefHomeAdvantageYellow': home_advantage_yellow,
            'RefHomeWinRate': home_win_rate,
            'RefAwayWinRate': away_win_rate,
            'RefDrawRate': draw_rate
        })

    # Convert to DataFrame
    ref_stats_df = pd.DataFrame(referee_stats)

    # Merge referee stats back to main dataframe
    df = df.merge(ref_stats_df, on='Referee', how='left')

    # Fill any missing values with league averages
    league_avg_yellow = df['RefYellowCardsPerGame'].mean()
    league_avg_red = df['RefRedCardsPerGame'].mean()
    league_avg_fouls = df['RefFoulsPerGame'].mean()

    df['RefYellowCardsPerGame'] = df['RefYellowCardsPerGame'].fillna(league_avg_yellow)
    df['RefRedCardsPerGame'] = df['RefRedCardsPerGame'].fillna(league_avg_red)
    df['RefFoulsPerGame'] = df['RefFoulsPerGame'].fillna(league_avg_fouls)
    df['RefHomeAdvantageYellow'] = df['RefHomeAdvantageYellow'].fillna(0.0)
    df['RefHomeWinRate'] = df['RefHomeWinRate'].fillna(df['RefHomeWinRate'].mean())
    df['RefAwayWinRate'] = df['RefAwayWinRate'].fillna(df['RefAwayWinRate'].mean())
    df['RefDrawRate'] = df['RefDrawRate'].fillna(df['RefDrawRate'].mean())

    print(f"Added referee statistics for {len(ref_stats_df)} referees")
    return df

def add_manager_features(df):
    """
    Add manager-related features to the dataset.

    Args:
        df (pd.DataFrame): Historical match data

    Returns:
        pd.DataFrame: Data with manager features added
    """
    print("Adding manager features...")

    # Map teams to managers based on match date
    df['HomeManager'] = df.apply(lambda row: get_current_manager(row['HomeTeam'], row['MatchDate']), axis=1)
    df['AwayManager'] = df.apply(lambda row: get_current_manager(row['AwayTeam'], row['MatchDate']), axis=1)

    # Add manager statistics
    manager_features = []
    for _, row in df.iterrows():
        home_manager = row['HomeManager']
        away_manager = row['AwayManager']

        home_stats = get_manager_stats(home_manager) if home_manager else {
            'WinRate': 0.45, 'GoalsPerGame': 1.4, 'PreferredFormation': '4-3-3',
            'TacticalFlexibility': 0.65, 'DefensiveSolidity': 0.70, 'AttackingThreat': 0.65
        }
        away_stats = get_manager_stats(away_manager) if away_manager else {
            'WinRate': 0.45, 'GoalsPerGame': 1.4, 'PreferredFormation': '4-3-3',
            'TacticalFlexibility': 0.65, 'DefensiveSolidity': 0.70, 'AttackingThreat': 0.65
        }

        # Calculate advantages
        advantages = calculate_manager_advantage(home_manager, away_manager)

        manager_features.append({
            'HomeManagerWinRate': home_stats['WinRate'],
            'AwayManagerWinRate': away_stats['WinRate'],
            'HomeManagerGoalsPerGame': home_stats['GoalsPerGame'],
            'AwayManagerGoalsPerGame': away_stats['GoalsPerGame'],
            'HomeManagerDefensiveSolidity': home_stats['DefensiveSolidity'],
            'AwayManagerDefensiveSolidity': away_stats['DefensiveSolidity'],
            'HomeManagerAttackingThreat': home_stats['AttackingThreat'],
            'AwayManagerAttackingThreat': away_stats['AttackingThreat'],
            'HomeManagerTacticalFlexibility': home_stats['TacticalFlexibility'],
            'AwayManagerTacticalFlexibility': away_stats['TacticalFlexibility'],
            'HomeManagerFormation': home_stats['PreferredFormation'],
            'AwayManagerFormation': away_stats['PreferredFormation'],
            'ManagerWinRateDiff': advantages['ManagerWinRateDiff'],
            'ManagerGoalsPerGameDiff': advantages['ManagerGoalsPerGameDiff'],
            'ManagerDefensiveAdvantage': advantages['ManagerDefensiveAdvantage'],
            'ManagerAttackingAdvantage': advantages['ManagerAttackingAdvantage'],
            'ManagerTacticalFlexibilityDiff': advantages['ManagerTacticalFlexibilityDiff']
        })

    manager_df = pd.DataFrame(manager_features)
    df = pd.concat([df, manager_df], axis=1)

    # Fill any missing values with league averages for numeric columns
    manager_cols = manager_df.columns
    for col in manager_cols:
        if col in df.columns:
            if df[col].dtype in ['float64', 'int64']:
                df[col] = df[col].fillna(df[col].mean())
            # For categorical columns like formations, fill with most common
            elif col in ['HomeManagerFormation', 'AwayManagerFormation']:
                df[col] = df[col].fillna('4-3-3')  # Most common formation

    print(f"Added manager features for {len(df)} matches")
    return df

# Apply KNN imputation for missing data
historical_data_with_calculations = smart_imputation(historical_data_with_calculations)

# Calculate advanced metrics
print("Calculating advanced team metrics...")
historical_data_with_calculations = calculate_advanced_metrics(historical_data_with_calculations)

# Calculate referee statistics
historical_data_with_calculations = calculate_referee_statistics(historical_data_with_calculations)

# Add manager features
historical_data_with_calculations = add_manager_features(historical_data_with_calculations)


# ---------------------------------------------------------------------------
# NEW FEATURES: League Position, Clean Sheets, Home/Away Form, API Data
# (Enhancements 3, 4, 6, 7 from api-football-integration-guide.md)
# ---------------------------------------------------------------------------

def calculate_league_position_features(df):
    """
    Enhancement 4: League position and points-per-game features.
    Computes PPG, z-score strength, and GD per season without expensive per-row ranking.
    """
    print("Calculating league position features...")
    df = df.sort_values(['Season', 'MatchDate']).reset_index(drop=True)

    # Match number within season (for PPG denominator)
    df['_HomeMatchNum'] = df.groupby(['Season', 'HomeTeam']).cumcount()
    df['_AwayMatchNum'] = df.groupby(['Season', 'AwayTeam']).cumcount()

    # Points per game up to (but not including) this match
    df['HomePointsPerGame'] = (
        df['HomeTeamCumulativePoints'] / (df['_HomeMatchNum'] + 1).clip(lower=1)
    )
    df['AwayPointsPerGame'] = (
        df['AwayTeamCumulativePoints'] / (df['_AwayMatchNum'] + 1).clip(lower=1)
    )
    df['PointsPerGameDiff'] = df['HomePointsPerGame'] - df['AwayPointsPerGame']

    # Season-normalised strength (z-score within season)
    df['HomePointsZScore'] = df.groupby('Season')['HomeTeamCumulativePoints'].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-9)
    )
    df['AwayPointsZScore'] = df.groupby('Season')['AwayTeamCumulativePoints'].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-9)
    )
    df['PointsZScoreDiff'] = df['HomePointsZScore'] - df['AwayPointsZScore']

    # Cumulative goal difference (home perspective, home matches only — fast proxy)
    home_gd = (
        df.groupby(['Season', 'HomeTeam'])['FullTimeHomeGoals'].shift(1).cumsum()
        - df.groupby(['Season', 'HomeTeam'])['FullTimeAwayGoals'].shift(1).cumsum()
    )
    away_gd = (
        df.groupby(['Season', 'AwayTeam'])['FullTimeAwayGoals'].shift(1).cumsum()
        - df.groupby(['Season', 'AwayTeam'])['FullTimeHomeGoals'].shift(1).cumsum()
    )
    df['HomeGoalDiffSeason'] = home_gd.fillna(0)
    df['AwayGoalDiffSeason'] = away_gd.fillna(0)

    # Fill edge-case NaNs
    for col in ['HomePointsPerGame', 'AwayPointsPerGame', 'PointsPerGameDiff',
                'HomePointsZScore', 'AwayPointsZScore', 'PointsZScoreDiff',
                'HomeGoalDiffSeason', 'AwayGoalDiffSeason']:
        df[col] = df[col].fillna(0.0)

    df.drop(columns=['_HomeMatchNum', '_AwayMatchNum'], inplace=True)
    print(f"  Added league position features (PPG, z-score, season GD)")
    return df


def calculate_clean_sheet_features(df):
    """
    Enhancement 3: Clean-sheet % and failed-to-score % over last 10 home/away matches.
    Uses shift(1) so no data leakage.
    """
    print("Calculating clean sheet and scoring features...")
    df = df.sort_values('MatchDate').reset_index(drop=True)

    # Binary flags for current match (will be shifted away before rolling)
    df['_HomeCS']  = (df['FullTimeAwayGoals'] == 0).astype(int)  # home kept clean sheet
    df['_AwayCS']  = (df['FullTimeHomeGoals'] == 0).astype(int)  # away kept clean sheet
    df['_HomeFTS'] = (df['FullTimeHomeGoals'] == 0).astype(int)  # home failed to score
    df['_AwayFTS'] = (df['FullTimeAwayGoals'] == 0).astype(int)  # away failed to score

    df['HomeCleanSheetPct_L10'] = (
        df.groupby('HomeTeam')['_HomeCS']
        .shift(1).rolling(10, min_periods=1).mean()
        .reset_index(level=0, drop=True)
        .fillna(0.30)
    )
    df['AwayCleanSheetPct_L10'] = (
        df.groupby('AwayTeam')['_AwayCS']
        .shift(1).rolling(10, min_periods=1).mean()
        .reset_index(level=0, drop=True)
        .fillna(0.30)
    )
    df['HomeFailedToScorePct_L10'] = (
        df.groupby('HomeTeam')['_HomeFTS']
        .shift(1).rolling(10, min_periods=1).mean()
        .reset_index(level=0, drop=True)
        .fillna(0.20)
    )
    df['AwayFailedToScorePct_L10'] = (
        df.groupby('AwayTeam')['_AwayFTS']
        .shift(1).rolling(10, min_periods=1).mean()
        .reset_index(level=0, drop=True)
        .fillna(0.20)
    )

    # Derived: defensive vs offensive matchup tension
    df['HomeDefVsAwayAttack'] = df['HomeCleanSheetPct_L10'] - (1 - df['AwayFailedToScorePct_L10'])
    df['AwayDefVsHomeAttack'] = df['AwayCleanSheetPct_L10'] - (1 - df['HomeFailedToScorePct_L10'])

    df.drop(columns=['_HomeCS', '_AwayCS', '_HomeFTS', '_AwayFTS'], inplace=True)
    print(f"  Added clean-sheet and scoring features (8 columns)")
    return df


def calculate_home_away_split_form(df):
    """
    Enhancement 3: Form computed separately for home games (home team) and
    away games (away team) — more predictive than combined form for venue-specific analysis.
    """
    print("Calculating home/away split form...")
    df = df.sort_values('MatchDate').reset_index(drop=True)

    # Home team's form in their last 5 HOME games
    df['HomeTeamPointsLast5_HomeOnly'] = (
        df.groupby('HomeTeam')['HomePoints']
        .shift(1).rolling(5, min_periods=1).sum()
        .reset_index(level=0, drop=True)
        .fillna(5.0)
    )

    # Away team's form in their last 5 AWAY games
    df['AwayTeamPointsLast5_AwayOnly'] = (
        df.groupby('AwayTeam')['AwayPoints']
        .shift(1).rolling(5, min_periods=1).sum()
        .reset_index(level=0, drop=True)
        .fillna(5.0)
    )

    # Venue-adjusted form advantage
    df['VenueAdjustedFormDiff'] = (
        df['HomeTeamPointsLast5_HomeOnly'] - df['AwayTeamPointsLast5_AwayOnly']
    )

    # Recent wins streak (last 5 home matches for home, away for away)
    df['_HomeWin'] = (df['FullTimeResult'] == 'H').astype(int)
    df['_AwayWin'] = (df['FullTimeResult'] == 'A').astype(int)

    df['HomeWinStreakHome_L5'] = (
        df.groupby('HomeTeam')['_HomeWin']
        .shift(1).rolling(5, min_periods=1).sum()
        .reset_index(level=0, drop=True)
        .fillna(1.5)
    )
    df['AwayWinStreakAway_L5'] = (
        df.groupby('AwayTeam')['_AwayWin']
        .shift(1).rolling(5, min_periods=1).sum()
        .reset_index(level=0, drop=True)
        .fillna(1.0)
    )
    df.drop(columns=['_HomeWin', '_AwayWin'], inplace=True)

    print(f"  Added home/away split form features (5 columns)")
    return df


def merge_api_team_stats(df):
    """
    Enhancement 3/6: Merge API-derived season statistics when available.
    Adds clean-sheet counts, formations, goals-by-minute from api_team_statistics.csv.
    Gracefully skips if file is absent (e.g., first run before API fetch).
    """
    stats_file = path.join(DATA_DIR, 'api_team_statistics.csv')
    if not path.exists(stats_file):
        print("  api_team_statistics.csv not found — skipping API team stats merge")
        return df

    print("Merging API team statistics...")
    api_stats = pd.read_csv(stats_file, sep='\t')

    # Select the most useful columns for the model
    cols = [
        'Team', 'MostUsedFormation',
        'CleanSheetTotal', 'FailedToScoreHome', 'FailedToScoreAway',
        'PenaltyScored', 'PenaltyMissed',
        'GoalsForAvgHome', 'GoalsForAvgAway',
        'GoalsAgainstAvgHome', 'GoalsAgainstAvgAway',
        'BiggestStreakWins', 'BiggestStreakLoses',
        'GoalsFor_0_15', 'GoalsFor_76_90',
    ]
    cols = [c for c in cols if c in api_stats.columns]
    api_stats_slim = api_stats[cols].copy()

    # Encode formation categorically
    if 'MostUsedFormation' in api_stats_slim.columns:
        formation_map = {f: i for i, f in enumerate(api_stats_slim['MostUsedFormation'].unique())}
        api_stats_slim['MostUsedFormation_Enc'] = api_stats_slim['MostUsedFormation'].map(formation_map).fillna(-1)
        api_stats_slim.drop(columns=['MostUsedFormation'], inplace=True)

    # Compute penalty conversion rate
    if 'PenaltyScored' in api_stats_slim.columns and 'PenaltyMissed' in api_stats_slim.columns:
        total_pen = api_stats_slim['PenaltyScored'] + api_stats_slim['PenaltyMissed']
        api_stats_slim['PenaltyConversionRate'] = (
            api_stats_slim['PenaltyScored'] / total_pen.replace(0, np.nan)
        ).fillna(0.75)
        api_stats_slim.drop(columns=['PenaltyScored', 'PenaltyMissed'], inplace=True)

    # Rename with prefix for home merge, then merge
    home_api = api_stats_slim.rename(columns={c: f'API_Home_{c}' for c in api_stats_slim.columns if c != 'Team'})
    away_api = api_stats_slim.rename(columns={c: f'API_Away_{c}' for c in api_stats_slim.columns if c != 'Team'})

    df = df.merge(home_api, left_on='HomeTeam', right_on='Team', how='left').drop(columns=['Team'], errors='ignore')
    df = df.merge(away_api, left_on='AwayTeam', right_on='Team', how='left').drop(columns=['Team'], errors='ignore')

    # Fill missing (teams not in 2024 season API data)
    api_new_cols = [c for c in df.columns if c.startswith('API_Home_') or c.startswith('API_Away_')]
    for col in api_new_cols:
        if df[col].dtype in [np.float64, np.int64, 'float64', 'int64']:
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(0)

    print(f"  Merged {len(api_new_cols)} API team statistics columns")
    return df


def merge_api_standings(df):
    """
    Enhancement 4: Merge live standings data as a static team-strength prior.
    Uses api_standings.csv to add current-season rank as a season-end strength signal.
    Gracefully skips if file is absent.
    """
    standings_file = path.join(DATA_DIR, 'api_standings.csv')
    if not path.exists(standings_file):
        print("  api_standings.csv not found — skipping standings merge")
        return df

    print("Merging API standings data...")
    standings = pd.read_csv(standings_file, sep='\t')

    standing_cols = ['Team', 'Rank', 'Points', 'Win', 'Draw', 'Lose',
                     'GoalsFor', 'GoalsAgainst', 'GoalDifference']
    standing_cols = [c for c in standing_cols if c in standings.columns]
    standings_slim = standings[standing_cols].copy()

    # These are 2024-season final standings — useful as team strength prior
    home_st = standings_slim.rename(columns={c: f'API_Home_Standing_{c}'
                                              for c in standing_cols if c != 'Team'})
    away_st = standings_slim.rename(columns={c: f'API_Away_Standing_{c}'
                                              for c in standing_cols if c != 'Team'})

    df = df.merge(home_st, left_on='HomeTeam', right_on='Team', how='left').drop(columns=['Team'], errors='ignore')
    df = df.merge(away_st, left_on='AwayTeam', right_on='Team', how='left').drop(columns=['Team'], errors='ignore')

    # Position difference (lower rank number = stronger team)
    if 'API_Home_Standing_Rank' in df.columns and 'API_Away_Standing_Rank' in df.columns:
        df['API_StandingsRankDiff'] = (
            df['API_Away_Standing_Rank'] - df['API_Home_Standing_Rank']
        )

    st_new_cols = [c for c in df.columns if c.startswith('API_Home_Standing_') or c.startswith('API_Away_Standing_') or c == 'API_StandingsRankDiff']
    for col in st_new_cols:
        if df[col].dtype in [np.float64, np.int64, 'float64', 'int64']:
            df[col] = df[col].fillna(df[col].median())

    print(f"  Merged {len(st_new_cols)} API standings columns")
    return df


# Apply new feature engineering
print("\nApplying new feature engineering (Enhancements 3, 4)...")
historical_data_with_calculations = calculate_league_position_features(historical_data_with_calculations)
historical_data_with_calculations = calculate_clean_sheet_features(historical_data_with_calculations)
historical_data_with_calculations = calculate_home_away_split_form(historical_data_with_calculations)
historical_data_with_calculations = merge_api_team_stats(historical_data_with_calculations)
historical_data_with_calculations = merge_api_standings(historical_data_with_calculations)


# ---------------------------------------------------------------------------
# NEW FEATURES: ClubElo ratings and Understat xG
# Run fetch_clubelo.py and fetch_understat_xg.py to generate the source files.
# Both functions degrade gracefully (no-op) when source files are absent.
# ---------------------------------------------------------------------------

def merge_clubelo_features(df):
    """
    Add ClubElo Elo ratings as features: HomeElo, AwayElo, EloDiff.
    Elo is a date-ranged rating — uses merge_asof to find the rating valid
    on each match date. Source: http://clubelo.com/API
    Requires: data_files/clubelo_ratings.csv (run fetch_clubelo.py first).
    """
    elo_file = path.join(DATA_DIR, 'clubelo_ratings.csv')
    if not path.exists(elo_file):
        print('  clubelo_ratings.csv not found — skipping ClubElo merge (run fetch_clubelo.py)')
        return df

    print('Merging ClubElo Elo ratings...')
    elo_df = pd.read_csv(elo_file, sep='\t')
    elo_df['From'] = pd.to_datetime(elo_df['From'], errors='coerce')
    elo_df['Elo'] = pd.to_numeric(elo_df['Elo'], errors='coerce')
    elo_df = elo_df.dropna(subset=['From', 'Elo', 'HistTeam'])
    elo_df = elo_df.sort_values(['HistTeam', 'From']).reset_index(drop=True)

    # Preserve original index so we can map results back after merge_asof
    df = df.copy()
    df['_orig_idx'] = np.arange(len(df))
    df_sorted = df[['_orig_idx', 'MatchDate', 'HomeTeam', 'AwayTeam']].sort_values('MatchDate')

    # Home Elo: look up the most recent Elo entry at or before MatchDate
    home_elo_ref = elo_df[['From', 'HistTeam', 'Elo']].rename(
        columns={'From': 'MatchDate', 'HistTeam': 'HomeTeam', 'Elo': 'HomeElo'}
    ).sort_values('MatchDate')

    away_elo_ref = elo_df[['From', 'HistTeam', 'Elo']].rename(
        columns={'From': 'MatchDate', 'HistTeam': 'AwayTeam', 'Elo': 'AwayElo'}
    ).sort_values('MatchDate')

    home_merged = pd.merge_asof(
        df_sorted[['_orig_idx', 'MatchDate', 'HomeTeam']],
        home_elo_ref,
        on='MatchDate',
        by='HomeTeam',
        direction='backward'
    ).set_index('_orig_idx')['HomeElo']

    away_merged = pd.merge_asof(
        df_sorted[['_orig_idx', 'MatchDate', 'AwayTeam']],
        away_elo_ref,
        on='MatchDate',
        by='AwayTeam',
        direction='backward'
    ).set_index('_orig_idx')['AwayElo']

    df['HomeElo'] = df['_orig_idx'].map(home_merged)
    df['AwayElo'] = df['_orig_idx'].map(away_merged)
    df['EloDiff'] = df['HomeElo'] - df['AwayElo']

    avg_elo = elo_df['Elo'].mean()
    df['HomeElo'] = df['HomeElo'].fillna(avg_elo)
    df['AwayElo'] = df['AwayElo'].fillna(avg_elo)
    df['EloDiff'] = df['EloDiff'].fillna(0.0)
    df.drop(columns=['_orig_idx'], inplace=True)

    matched = (df['HomeElo'] != avg_elo).sum()
    print(f'  Added ClubElo features: HomeElo, AwayElo, EloDiff ({matched} matches with Elo data)')
    return df


def merge_understat_xg_features(df):
    """
    Add Understat xG rolling averages:
      HomeXG_Understat_L5   — home team's avg xG (attacking) over last 5 home matches
      AwayXG_Understat_L5   — away team's avg xG (attacking) over last 5 away matches
      HomeXGA_Understat_L5  — home team's avg xG conceded over last 5 home matches
      AwayXGA_Understat_L5  — away team's avg xG conceded over last 5 away matches

    Uses shift(1) before rolling to prevent data leakage.
    Falls back to shot-based proxy (HomexG_Avg_L5) for pre-2014/15 rows.
    Source: https://understat.com/
    Requires: data_files/understat_xg.csv (run fetch_understat_xg.py first).
    """
    xg_file = path.join(DATA_DIR, 'understat_xg.csv')
    if not path.exists(xg_file):
        print('  understat_xg.csv not found — skipping Understat xG merge (run fetch_understat_xg.py)')
        return df

    print('Merging Understat xG data...')
    xg_df = pd.read_csv(xg_file, sep='\t')
    xg_df['MatchDate'] = pd.to_datetime(xg_df['MatchDate'], errors='coerce').dt.normalize()
    xg_df = xg_df.dropna(subset=['MatchDate', 'HomeTeam', 'AwayTeam'])

    df = df.copy()
    df['_MatchDateOnly'] = df['MatchDate'].dt.normalize()

    df = df.merge(
        xg_df[['MatchDate', 'HomeTeam', 'AwayTeam', 'HomeXG_Understat', 'AwayXG_Understat']].rename(
            columns={'MatchDate': '_MatchDateOnly'}
        ),
        on=['_MatchDateOnly', 'HomeTeam', 'AwayTeam'],
        how='left'
    )
    df.drop(columns=['_MatchDateOnly'], inplace=True)

    # Compute rolling 5-match averages from past matches only (shift avoids leakage)
    df = df.sort_values('MatchDate').reset_index(drop=True)

    df['HomeXG_Understat_L5'] = (
        df.groupby('HomeTeam')['HomeXG_Understat']
        .shift(1).rolling(5, min_periods=1).mean()
        .reset_index(level=0, drop=True)
    )
    df['AwayXG_Understat_L5'] = (
        df.groupby('AwayTeam')['AwayXG_Understat']
        .shift(1).rolling(5, min_periods=1).mean()
        .reset_index(level=0, drop=True)
    )
    # xG conceded: away xG scored against the home team, and vice versa
    df['HomeXGA_Understat_L5'] = (
        df.groupby('HomeTeam')['AwayXG_Understat']
        .shift(1).rolling(5, min_periods=1).mean()
        .reset_index(level=0, drop=True)
    )
    df['AwayXGA_Understat_L5'] = (
        df.groupby('AwayTeam')['HomeXG_Understat']
        .shift(1).rolling(5, min_periods=1).mean()
        .reset_index(level=0, drop=True)
    )

    # Fill NaNs: prefer existing shot-based proxy; fall back to league default
    default_xg = 1.5
    proxy_home = df.get('HomexG_Avg_L5', pd.Series(dtype=float))
    proxy_away = df.get('AwayxG_Avg_L5', pd.Series(dtype=float))
    df['HomeXG_Understat_L5'] = df['HomeXG_Understat_L5'].fillna(proxy_home).fillna(default_xg)
    df['AwayXG_Understat_L5'] = df['AwayXG_Understat_L5'].fillna(proxy_away).fillna(default_xg)
    df['HomeXGA_Understat_L5'] = df['HomeXGA_Understat_L5'].fillna(default_xg)
    df['AwayXGA_Understat_L5'] = df['AwayXGA_Understat_L5'].fillna(default_xg)

    # Drop per-match raw columns — only rolling averages used as features
    df.drop(columns=['HomeXG_Understat', 'AwayXG_Understat'], inplace=True, errors='ignore')

    matched = df['HomeXG_Understat_L5'].notna().sum()
    print(f'  Added Understat xG features: HomeXG_Understat_L5, AwayXG_Understat_L5, '
          f'HomeXGA_Understat_L5, AwayXGA_Understat_L5 ({matched} rows)')
    return df


# Apply ClubElo and Understat xG feature engineering
print('\nApplying ClubElo and Understat xG feature engineering...')
historical_data_with_calculations = merge_clubelo_features(historical_data_with_calculations)
historical_data_with_calculations = merge_understat_xg_features(historical_data_with_calculations)

historical_data_with_calculations.to_csv(
    path.join(DATA_DIR, 'combined_historical_data_with_calculations_new.csv'),
    sep='\t',
    index=False
)