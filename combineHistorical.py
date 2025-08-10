from datetime import date, timedelta
import datetime
import pandas as pd
from os import path
import os
import numpy as np

DATA_DIR = 'data_files/'

premier_2024 = pd.read_csv(path.join(DATA_DIR, 'pl_2024.csv'))
premier_2023 = pd.read_csv(path.join(DATA_DIR, 'pl_2023.csv'))
premier_2022 = pd.read_csv(path.join(DATA_DIR, 'pl_2022.csv'))
premier_2021 = pd.read_csv(path.join(DATA_DIR, 'pl_2021.csv'))
# print(premier_2022.head(50))
historical_data = pd.concat([premier_2021, premier_2022, premier_2023, premier_2024], ignore_index=True)

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

# Select all columns that are blank, and drop them
historical_data.dropna(axis=1, how='all', inplace=True)
historical_data.drop(columns=['Division'], inplace=True)

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

print(historical_data.head(50))