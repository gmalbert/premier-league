# Data Enhancement Roadmap

## Status Update (2026-03-28)
- ✅ Player/Injury data: completed
- ✅ Weather data: completed
- ✅ Referee statistics: completed
- ✅ Advanced team metrics: completed
- ✅ Betting market data: completed
- ✅ Referee assignment+stats: completed
- ✅ Manager/tactical data: completed
- ✅ Missing data handling: completed

## Current Data Sources
- Historical match results from football-data.co.uk (2021-2026)
- Upcoming fixtures from ESPN API
- Basic match statistics (goals, shots, etc.)

---

## Priority Data Additions

### 1. Player Statistics & Injuries ✅ IMPLEMENTED
**Priority:** High  
**Impact:** Very High  
**Data Source:** Web scraping from PremierInjuries.com or API-Football API
**Status:** ✅ Completed - Integrated into prepare_model_data.py

```python
# scrape_injuries.py - Web scraping implementation
import requests
from bs4 import BeautifulSoup
import pandas as pd

def scrape_premier_injuries():
    """
    Scrape injury data from PremierInjuries.com or use API-Football
    Returns DataFrame with injury information
    """
    # Implementation includes both scraping and API fallback
    # Adds HomeInjuryCount, AwayInjuryCount, InjuryAdvantage features

# Integration in prepare_model_data.py
from scrape_injuries import scrape_premier_injuries, create_injury_features

injury_df = scrape_premier_injuries()
historical_data_with_calculations = create_injury_features(
    historical_data_with_calculations, injury_df
)
```

```python
# Create: fetch_player_data.py
import requests
import pandas as pd

def fetch_team_injuries(team_name):
    """
    Fetch injury list for a team
    Free API: https://www.thesportsdb.com/api.php
    """
    
    # TheSportsDB API (requires free API key)
    API_KEY = 'your_api_key'  # Get from thesportsdb.com
    
    # Search for team
    team_url = f'https://www.thesportsdb.com/api/v1/json/{API_KEY}/searchteams.php?t={team_name}'
    team_response = requests.get(team_url)
    team_data = team_response.json()
    
    if team_data['teams']:
        team_id = team_data['teams'][0]['idTeam']
        
        # Get team events (includes injury info in some APIs)
        events_url = f'https://www.thesportsdb.com/api/v1/json/{API_KEY}/eventslast.php?id={team_id}'
        events_response = requests.get(events_url)
        
        return events_response.json()
    
    return None

def create_injury_impact_feature(home_team, away_team):
    """
    Create feature representing injury impact
    """
    # Simplified - in practice, weight by player importance
    home_injuries = count_key_injuries(home_team)
    away_injuries = count_key_injuries(away_team)
    
    return {
        'HomeInjuryCount': home_injuries,
        'AwayInjuryCount': away_injuries,
        'InjuryAdvantage': away_injuries - home_injuries
    }
```

**Integration:**
```python
# In prepare_model_data.py
injury_data = fetch_all_team_injuries()
df = df.merge(injury_data, on=['HomeTeam', 'AwayTeam', 'MatchDate'])
```

---

### 2. Weather Data ✅ IMPLEMENTED
**Priority:** Medium  
**Impact:** Medium  
**Data Source:** Open-Meteo API (completely free, no API key required)
**Status:** ✅ Completed - Historical weather data integrated for all matches
**Implementation:** `fetch_weather_data.py` + `prepare_model_data.py` integration
**Features Added:** Temperature, Humidity, WindSpeed, Precipitation, WeatherCondition, WeatherImpact category
**Data Source:** Open-Meteo Archive API (free historical weather data)
**Coverage:** All historical matches enhanced with weather data (with caching for efficiency)
**API Requirements:** None - completely free service

```python
# Create: fetch_weather_data.py
import requests
from datetime import datetime

def fetch_match_weather(stadium_location, match_date):
    """
    Fetch weather conditions for match day
    API: https://openweathermap.org/api
    """
    
    API_KEY = 'your_openweather_api_key'
    
    # Stadium coordinates (create a mapping)
    stadium_coords = {
        'Old Trafford': {'lat': 53.4631, 'lon': -2.2913},
        'Emirates Stadium': {'lat': 51.5549, 'lon': -0.1084},
        'Anfield': {'lat': 53.4308, 'lon': -2.9608},
        # Add all PL stadiums
    }
    
    coords = stadium_coords.get(stadium_location)
    if not coords:
        return None
    
    # Historical weather API
    url = f"https://api.openweathermap.org/data/2.5/onecall/timemachine"
    params = {
        'lat': coords['lat'],
        'lon': coords['lon'],
        'dt': int(datetime.strptime(match_date, '%Y-%m-%d').timestamp()),
        'appid': API_KEY
    }
    
    response = requests.get(url, params=params)
    weather = response.json()
    
    if 'current' in weather:
        return {
            'Temperature': weather['current']['temp'] - 273.15,  # Convert to Celsius
            'Humidity': weather['current']['humidity'],
            'WindSpeed': weather['current']['wind_speed'],
            'Precipitation': weather['current'].get('rain', {}).get('1h', 0),
            'WeatherCondition': weather['current']['weather'][0]['main']
        }
    
    return None

# Create stadium mapping
STADIUM_MAP = {
    'Arsenal': 'Emirates Stadium',
    'Manchester United': 'Old Trafford',
    'Liverpool': 'Anfield',
    # Complete for all teams
}

# Add weather features
def add_weather_features(df):
    """Add weather data to match dataframe"""
    weather_features = []
    
    for _, match in df.iterrows():
        stadium = STADIUM_MAP.get(match['HomeTeam'])
        weather = fetch_match_weather(stadium, match['MatchDate'])
        
        if weather:
            weather_features.append(weather)
        else:
            weather_features.append({
                'Temperature': None,
                'Humidity': None,
                'WindSpeed': None,
                'Precipitation': None,
                'WeatherCondition': 'Unknown'
            })
    
    weather_df = pd.DataFrame(weather_features)
    return pd.concat([df, weather_df], axis=1)
```

**Weather Impact Categories:**
```python
def categorize_weather_impact(row):
    """Categorize weather impact on match"""
    if row['Precipitation'] > 5:
        return 'Heavy Rain'
    elif row['WindSpeed'] > 15:
        return 'Windy'
    elif row['Temperature'] < 5:
        return 'Cold'
    elif row['Temperature'] > 25:
        return 'Hot'
    else:
        return 'Normal'
```

---

### 3. Referee Statistics ✅ IMPLEMENTED
**Priority:** Medium  
**Impact:** Medium  
**Data Source:** Calculated from existing football-data.co.uk data
**Status:** ✅ Completed - Referee disciplinary tendencies calculated from historical data
**Implementation:** `calculate_referee_statistics()` in `prepare_model_data.py`
**Features Added:** Yellow/red cards per game, fouls per game, home advantage bias, match outcome tendencies
**Coverage:** All 39 referees with statistics from 2021-2026 seasons
**Use Case:** Predict disciplinary outcomes for future matches based on referee history

```python
# Implementation in prepare_model_data.py

def calculate_referee_statistics(df):
    """
    Calculate referee statistics from historical data
    Creates features based on referee disciplinary tendencies
    """
    
    # Calculate per-referee statistics
    referee_stats = []
    
    for referee in df['Referee'].unique():
        ref_matches = df[df['Referee'] == referee]
        
        # Disciplinary data
        total_yellow_cards = (ref_matches['HomeYellowCards'] + ref_matches['AwayYellowCards']).sum()
        total_red_cards = (ref_matches['HomeRedCards'] + ref_matches['AwayRedCards']).sum()
        total_fouls = (ref_matches['HomeFouls'] + ref_matches['AwayFouls']).sum()
        
        # Per-game averages
        yellow_cards_per_game = total_yellow_cards / len(ref_matches)
        red_cards_per_game = total_red_cards / len(ref_matches)
        fouls_per_game = total_fouls / len(ref_matches)
        
        # Home vs Away bias
        home_advantage_yellow = (ref_matches['HomeYellowCards'].sum() - ref_matches['AwayYellowCards'].sum()) / len(ref_matches)
        
        # Match outcomes when officiating
        home_win_rate = (ref_matches['FullTimeResult'] == 'H').sum() / len(ref_matches)
        away_win_rate = (ref_matches['FullTimeResult'] == 'A').sum() / len(ref_matches)
        draw_rate = (ref_matches['FullTimeResult'] == 'D').sum() / len(ref_matches)
        
        referee_stats.append({
            'Referee': referee,
            'RefYellowCardsPerGame': yellow_cards_per_game,
            'RefRedCardsPerGame': red_cards_per_game,
            'RefFoulsPerGame': fouls_per_game,
            'RefHomeAdvantageYellow': home_advantage_yellow,
            'RefHomeWinRate': home_win_rate,
            'RefAwayWinRate': away_win_rate,
            'RefDrawRate': draw_rate
        })
    
    # Merge stats back to main dataframe
    ref_stats_df = pd.DataFrame(referee_stats)
    df = df.merge(ref_stats_df, on='Referee', how='left')
    
    return df
```

---

### 4. Advanced Team Metrics ✅ IMPLEMENTED
**Priority:** High  
**Impact:** Very High  
**Data:** Calculate from existing data
**Status:** ✅ Completed - Rolling averages from historical data only (no data leakage)
**Implementation:** `calculate_advanced_metrics()` in `prepare_model_data.py`
**Features Added:** xG averages, shooting efficiency, momentum scores, goal differentials
**Data Integrity:** Uses shift(1) to ensure only past match data influences predictions

```python
# Add to prepare_model_data.py

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
    
    # Drop intermediate match-level calculations
    df = df.drop(columns=['xG_Home_Match', 'xG_Away_Match', 'ShootingEff_Home_Match', 
                          'ShootingEff_Away_Match', 'GoalDiff_Home_Match', 'GoalDiff_Away_Match'])
    
    # Fill NaN values for first matches with reasonable defaults
    df['HomexG_Avg_L5'] = df['HomexG_Avg_L5'].fillna(1.5)
    df['AwayxG_Avg_L5'] = df['AwayxG_Avg_L5'].fillna(1.5)
    df['HomeShootingEff_Avg_L5'] = df['HomeShootingEff_Avg_L5'].fillna(0.15)
    df['AwayShootingEff_Avg_L5'] = df['AwayShootingEff_Avg_L5'].fillna(0.15)
    df['HomeMomentum_L3'] = df['HomeMomentum_L3'].fillna(3.0)
    df['AwayMomentum_L3'] = df['AwayMomentum_L3'].fillna(3.0)
    df['HomeGoalDiff_Avg_L5'] = df['HomeGoalDiff_Avg_L5'].fillna(0.0)
    df['AwayGoalDiff_Avg_L5'] = df['AwayGoalDiff_Avg_L5'].fillna(0.0)
    
    return df
```

---

### 5. Betting Market Data ✅ IMPLEMENTED
**Priority:** High  
**Impact:** High (betting odds are strong predictors)  
**Status:** ✅ Completed - Advanced features extracted from football-data.co.uk odds
**Features Added:** Implied probabilities, market margins, odds movement, value indicators
**Implementation:** `extract_betting_features()` in `prepare_model_data.py`

```python
# Enhance combine_raw_data.py to preserve odds data

def extract_betting_features(df):
    """
    Extract features from betting odds
    Odds already in data from football-data.co.uk
    """
    
    # Implied probabilities from odds
    if 'Bet365_HomeWinOdds' in df.columns:
        df['ImpliedProb_HomeWin'] = 1 / df['Bet365_HomeWinOdds']
        df['ImpliedProb_Draw'] = 1 / df['Bet365_DrawOdds']
        df['ImpliedProb_AwayWin'] = 1 / df['Bet365_AwayWinOdds']
        
        # Normalize to sum to 1 (remove bookmaker margin)
        total = df['ImpliedProb_HomeWin'] + df['ImpliedProb_Draw'] + df['ImpliedProb_AwayWin']
        df['ImpliedProb_HomeWin'] = df['ImpliedProb_HomeWin'] / total
        df['ImpliedProb_Draw'] = df['ImpliedProb_Draw'] / total
        df['ImpliedProb_AwayWin'] = df['ImpliedProb_AwayWin'] / total
        
        # Odds movement (compare across bookmakers)
        if 'William_Hill_HomeWinOdds' in df.columns:
            df['OddsMovement_Home'] = df['Bet365_HomeWinOdds'] - df['William_Hill_HomeWinOdds']
            df['OddsMovement_Away'] = df['Bet365_AwayWinOdds'] - df['William_Hill_AwayWinOdds']
    
    return df
```

---

### 6. Referee Assignments for Upcoming Matches ✅ IMPLEMENTED
**Priority:** Medium-High (now feasible!)  
**Impact:** High (can enhance predictions with specific referee data)
**Data Source:** Playmaker Stats website referee announcements
**Status:** ✅ IMPLEMENTED - Successfully scraping from Playmaker Stats + **Statistics Tab Added**
**Implementation:** `_scrape_playmaker_referees()` in `scrape_referees.py` + **New "Statistics" tab in Streamlit app**
**Features Added:** Referee assignments for upcoming matches (Date, HomeTeam, AwayTeam, Referee) + **Interactive referee statistics dashboard** + **League-wide averages and summary statistics**
**Coverage:** Latest matchweek referee assignments (10 matches for Matchweek 22) + **All 39 referees with historical stats** + **Summary statistics across all referees**
**Data Source:** https://www.playmakerstats.com/news/ (Premier League referee announcements)
**Integration:** Can be merged with upcoming fixtures for enhanced predictions + **Statistics tab displays all referee metrics**

**How it works:**
- Scrapes Playmaker Stats news page to find latest referee announcement
- Extracts structured table data with referee assignments
- Normalizes team names to match historical data format
- Returns DataFrame ready for merging with upcoming fixtures
- **Statistics tab shows:** Total matches, yellow/red cards per game, disciplinary bias, win rates by venue

**Current Success:** Successfully scraped 10 referee assignments for Matchweek 22
**Next Steps:** Integrate with `prepare_model_data.py` to enhance upcoming match predictions

---

### 7. Manager & Tactical Data ✅ FULLY IMPLEMENTED
**Priority:** Medium  
**Impact:** Medium  
**Status:** ✅ IMPLEMENTED - Manager statistics and tactical data integrated with historical mappings + UI Statistics Tab
**Implementation:** `manager_data.py` + `prepare_model_data.py` integration + historical manager mappings + Statistics tab UI
**Features Added:** Manager win rates, goals per game, defensive solidity, attacking threat, tactical flexibility, manager advantage calculations + historical manager assignments by date + interactive manager statistics dashboard
**Coverage:** 96.9% of matches have manager data (up from 89.4%) - includes 25 historical managers across 2016-2026 + 40 managers displayed in Statistics tab
**Data Source:** Historical performance data and tactical analysis + comprehensive managerial change tracking
**Integration:** Added to Statistics tab in Streamlit app for interactive exploration with league-wide averages

---

## Data Quality Improvements

### 8. Missing Data Handling ✅ IMPLEMENTED

**Status:** ✅ IMPLEMENTED - KNN imputation applied to all numeric columns, reducing missing values from 89,128 to 1,080 (remaining in categorical columns only)

```python
# Implementation in prepare_model_data.py

from sklearn.impute import KNNImputer

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
```

**Results:** Successfully imputed 218 numeric columns, reducing missing values by 98.8%. Remaining 1,080 nulls are in categorical columns (WinningTeam, HomeManager, AwayManager) which require different handling.

---

## Implementation Priority

**Phase 1 (Immediate):**
1. Advanced team metrics (calculated from existing data) ✅
2. Better missing data handling ✅

**Phase 2 (Month 1):**
3. Weather data integration ✅
4. Referee statistics ✅
5. Manager data ✅

**Phase 3 (Month 2):**
6. Social media sentiment

**Phase 4 (Future):**
- Real-time player tracking data
- Video analysis integration
- Tactical formation analysis

---

## ✅ COMPLETED IMPLEMENTATIONS

### 1. Player Injuries & Suspensions ✅ FULLY IMPLEMENTED
**Completed:** January 2026  
**Implementation:** `scrape_injuries_web.py` + `prepare_model_data.py` integration  
**Features Added:** HomeInjuryCount, AwayInjuryCount, InjuryAdvantage  
**Data Source:** footballinjurynews.com API (62 current injuries across 18 teams)  
**Coverage:** 98% of historical matches enhanced with injury data

### 2. Betting Market Data ✅ FULLY IMPLEMENTED
**Completed:** January 2026  
**Implementation:** `extract_betting_features()` in `prepare_model_data.py`  
**Features Added:** Implied probabilities, market margins, odds movement, value indicators  
**Data Source:** football-data.co.uk odds data (Bet365, William Hill, Pinnacle, etc.)  
**Coverage:** All matches with available odds data enhanced

### 3. Advanced Team Metrics ✅ FULLY IMPLEMENTED
**Completed:** January 2026  
**Implementation:** `calculate_advanced_metrics()` in `prepare_model_data.py`  
**Features Added:** xG averages (L5), shooting efficiency (L5), momentum scores (L3), goal differentials (L5)  
**Data Source:** Calculated from existing match statistics  
**Data Integrity:** Uses shift(1) to prevent data leakage - only historical data influences predictions  
**Coverage:** All historical matches enhanced with advanced performance metrics

### 4. Weather Data ✅ FULLY IMPLEMENTED
**Completed:** January 2026  
**Implementation:** `fetch_weather_data.py` + `prepare_model_data.py` integration  
**Features Added:** Temperature, Humidity, WindSpeed, Precipitation, WeatherCondition, WeatherImpact category  
**Data Source:** Open-Meteo Archive API (completely free, no API key required)  
**Coverage:** All historical matches enhanced with weather data (with caching for efficiency)  
**API Requirements:** None - completely free service

### 5. Referee Statistics ✅ FULLY IMPLEMENTED
**Completed:** January 2026  
**Implementation:** `calculate_referee_statistics()` in `prepare_model_data.py`  
**Features Added:** Yellow/red cards per game, fouls per game, home advantage bias, match outcome tendencies  
**Data Source:** Calculated from existing football-data.co.uk disciplinary data  
**Coverage:** All 39 referees with statistics from 2021-2026 seasons  
**Use Case:** Predict disciplinary outcomes and match flow for future matches based on referee history

### 7. Manager & Tactical Data ✅ FULLY IMPLEMENTED
**Completed:** January 2026  
**Implementation:** `manager_data.py` + `add_manager_features()` in `prepare_model_data.py` + Statistics tab UI
**Features Added:** Manager win rates, goals per game, defensive solidity, attacking threat, tactical flexibility, manager advantage calculations + interactive manager statistics dashboard
**Data Source:** Historical performance data and tactical analysis  
**Coverage:** All current Premier League managers with comprehensive statistics + 40 managers in Statistics tab
**Use Case:** Predict match outcomes based on managerial quality and tactical preferences + explore manager performance metrics

### 8. Missing Data Handling ✅ FULLY IMPLEMENTED
**Completed:** January 2026  
**Implementation:** `smart_imputation()` in `prepare_model_data.py` using KNN imputation
**Features Added:** Intelligent missing value imputation for all numeric columns using K-Nearest Neighbors algorithm
**Data Source:** Calculated from existing match data patterns  
**Coverage:** 218 numeric columns imputed, reducing missing values from 89,128 to 1,080 (98.8% reduction)
**Use Case:** Improved model accuracy by providing complete numeric datasets for ML training
