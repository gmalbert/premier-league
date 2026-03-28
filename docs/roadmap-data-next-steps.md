# Data Enhancement - Next Steps

## Status Update (2026-03-28)
- ✅ Injury/weather/referee/manager/KNN baseline implemented (from prior roadmap)
- ⚪ Social media sentiment: pending
- ⚪ Transfer market activity: pending
- ⚪ Enhanced historical odds coverage: pending

Building on the successful implementations of player injuries, weather data, referee statistics, manager data, and KNN imputation, here are the next actionable steps to enhance data quality and coverage.

---

## 1. Social Media Sentiment Analysis

**Priority:** Medium  
**Effort:** 3-4 hours  
**Expected Impact:** Captures team morale and fan confidence

Integrate Twitter/X sentiment analysis for teams and upcoming matches.

```python
# Create: fetch_sentiment_data.py

import requests
import pandas as pd
from textblob import TextBlob
from datetime import datetime, timedelta

def fetch_twitter_sentiment(team_name, days_back=7):
    """
    Fetch and analyze Twitter sentiment for a team
    
    Uses Twitter API v2 (requires bearer token)
    Alternative: Use newspaper3k to scrape sports news headlines
    """
    
    # Option 1: Twitter API (requires authentication)
    # bearer_token = os.getenv('TWITTER_BEARER_TOKEN')
    
    # Option 2: News headline sentiment (no API key needed)
    from newspaper import Article, news_pool
    import feedparser
    
    sentiment_scores = []
    
    # BBC Sport RSS feed
    feed_url = f'https://feeds.bbci.co.uk/sport/football/rss.xml'
    feed = feedparser.parse(feed_url)
    
    for entry in feed.entries[:20]:  # Last 20 articles
        if team_name.lower() in entry.title.lower() or team_name.lower() in entry.summary.lower():
            # Analyze sentiment of title and summary
            text = f"{entry.title}. {entry.summary}"
            blob = TextBlob(text)
            sentiment_scores.append({
                'Date': entry.published_parsed,
                'Title': entry.title,
                'Sentiment': blob.sentiment.polarity,  # -1 to 1
                'Subjectivity': blob.sentiment.subjectivity
            })
    
    if len(sentiment_scores) == 0:
        return {'SentimentScore': 0, 'MediaCoverage': 0}
    
    df = pd.DataFrame(sentiment_scores)
    
    return {
        'SentimentScore': df['Sentiment'].mean(),
        'MediaCoverage': len(df),
        'PositiveNews': (df['Sentiment'] > 0.1).sum(),
        'NegativeNews': (df['Sentiment'] < -0.1).sum()
    }

def add_sentiment_features(historical_df):
    """Add sentiment features to historical data"""
    
    # For recent matches, fetch sentiment
    recent_cutoff = datetime.now() - timedelta(days=90)
    recent_matches = historical_df[
        pd.to_datetime(historical_df['MatchDate']) > recent_cutoff
    ]
    
    sentiment_data = []
    
    teams = set(recent_matches['HomeTeam'].unique()) | set(recent_matches['AwayTeam'].unique())
    
    for team in teams:
        sentiment = fetch_twitter_sentiment(team)
        sentiment_data.append({
            'Team': team,
            **sentiment
        })
    
    sentiment_df = pd.DataFrame(sentiment_data)
    
    # Merge with historical data
    historical_df = historical_df.merge(
        sentiment_df, left_on='HomeTeam', right_on='Team', 
        how='left', suffixes=('', '_Home')
    ).merge(
        sentiment_df, left_on='AwayTeam', right_on='Team',
        how='left', suffixes=('', '_Away')
    )
    
    # Fill missing sentiment with neutral (0)
    sentiment_cols = ['SentimentScore', 'MediaCoverage', 'PositiveNews', 'NegativeNews']
    for col in sentiment_cols:
        historical_df[f'{col}_Home'] = historical_df[f'{col}_Home'].fillna(0)
        historical_df[f'{col}_Away'] = historical_df[f'{col}_Away'].fillna(0)
    
    # Calculate sentiment advantage
    historical_df['SentimentAdvantage'] = (
        historical_df['SentimentScore_Home'] - historical_df['SentimentScore_Away']
    )
    
    return historical_df

# Add to requirements.txt:
# textblob>=0.17.0
# feedparser>=6.0.0
# newspaper3k>=0.2.8
```

---

## 2. Transfer Market Activity

**Priority:** Medium  
**Effort:** 2-3 hours  
**Expected Impact:** Captures squad strength changes

Track recent transfers and their impact on team performance.

```python
# Create: fetch_transfer_data.py

import requests
import pandas as pd
from datetime import datetime

def fetch_recent_transfers(days_back=30):
    """
    Fetch recent transfer activity for Premier League teams
    
    Data Source: Transfermarkt API (unofficial) or web scraping
    """
    
    # Using web scraping approach (no API key needed)
    from bs4 import BeautifulSoup
    
    url = 'https://www.transfermarkt.com/premier-league/transfers/wettbewerb/GB1'
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    response = requests.get(url, headers=headers, timeout=10)
    
    if response.status_code != 200:
        print(f"Failed to fetch transfer data: {response.status_code}")
        return pd.DataFrame()
    
    soup = BeautifulSoup(response.content, 'html.parser')
    
    transfers = []
    
    # Parse transfer tables (simplified - actual implementation would be more complex)
    transfer_boxes = soup.find_all('div', class_='box')
    
    for box in transfer_boxes[:20]:  # Limit to recent transfers
        team = box.find('h2')
        if team:
            team_name = team.text.strip()
            
            # Find transfer details
            rows = box.find_all('tr')
            for row in rows:
                cols = row.find_all('td')
                if len(cols) >= 4:
                    transfers.append({
                        'Team': team_name,
                        'Player': cols[0].text.strip(),
                        'TransferType': 'In' if 'arrivals' in str(box) else 'Out',
                        'Value': cols[2].text.strip(),
                        'Date': datetime.now()
                    })
    
    return pd.DataFrame(transfers)

def calculate_transfer_impact(team_name, transfers_df):
    """Calculate net transfer spend and activity for a team"""
    
    team_transfers = transfers_df[transfers_df['Team'] == team_name]
    
    if len(team_transfers) == 0:
        return {
            'NetTransferSpend': 0,
            'TransfersIn': 0,
            'TransfersOut': 0,
            'SquadStrengthChange': 0
        }
    
    transfers_in = (team_transfers['TransferType'] == 'In').sum()
    transfers_out = (team_transfers['TransferType'] == 'Out').sum()
    
    # Simplified value parsing (would need more robust implementation)
    def parse_value(val_str):
        """Convert transfer value string to number"""
        if 'Free' in val_str or '-' in val_str:
            return 0
        # Extract numeric value (simplified)
        import re
        numbers = re.findall(r'[\d.]+', val_str)
        return float(numbers[0]) if numbers else 0
    
    spend_in = team_transfers[team_transfers['TransferType'] == 'In']['Value'].apply(parse_value).sum()
    spend_out = team_transfers[team_transfers['TransferType'] == 'Out']['Value'].apply(parse_value).sum()
    
    return {
        'NetTransferSpend': spend_in - spend_out,
        'TransfersIn': transfers_in,
        'TransfersOut': transfers_out,
        'SquadStrengthChange': transfers_in - transfers_out
    }

# Add to prepare_model_data.py
def add_transfer_features(historical_df):
    """Add transfer activity features"""
    
    transfers = fetch_recent_transfers(days_back=90)
    
    if len(transfers) == 0:
        historical_df['NetTransferSpend_Home'] = 0
        historical_df['SquadStrengthChange_Home'] = 0
        historical_df['NetTransferSpend_Away'] = 0
        historical_df['SquadStrengthChange_Away'] = 0
        return historical_df
    
    teams = set(historical_df['HomeTeam'].unique()) | set(historical_df['AwayTeam'].unique())
    
    transfer_impact = []
    for team in teams:
        impact = calculate_transfer_impact(team, transfers)
        transfer_impact.append({'Team': team, **impact})
    
    transfer_df = pd.DataFrame(transfer_impact)
    
    # Merge with historical data
    historical_df = historical_df.merge(
        transfer_df[['Team', 'NetTransferSpend', 'SquadStrengthChange']],
        left_on='HomeTeam', right_on='Team', how='left'
    ).merge(
        transfer_df[['Team', 'NetTransferSpend', 'SquadStrengthChange']],
        left_on='AwayTeam', right_on='Team', how='left',
        suffixes=('_Home', '_Away')
    )
    
    return historical_df

# Add to requirements.txt:
# beautifulsoup4>=4.12.0
```

---

## 3. Enhanced Historical Odds Data

**Priority:** High  
**Effort:** 2 hours  
**Expected Impact:** Better probability calibration

Extend odds coverage to include Asian handicaps and over/under markets.

```python
# Add to prepare_model_data.py

def extract_advanced_odds_features(df):
    """
    Extract advanced betting market features
    
    Adds:
    - Asian handicap odds and implied probabilities
    - Over/Under 2.5 goals market
    - Both teams to score (BTTS) market
    - Odds movement indicators
    """
    
    # Asian Handicap features (if available in data)
    if 'BbAH' in df.columns:  # Bet365 Asian Handicap
        df['AsianHandicap_HomeBackProb'] = 1 / df['BbAH']
        df['AsianHandicap_AwayBackProb'] = 1 / df['BbAHA']
        df['AsianHandicap_Margin'] = (
            df['AsianHandicap_HomeBackProb'] + df['AsianHandicap_AwayBackProb'] - 1
        )
    
    # Over/Under 2.5 Goals
    if 'BbOU' in df.columns:
        df['Over2_5_Prob'] = 1 / df['BbOU']
        df['Under2_5_Prob'] = 1 / df['BbOUA']
        
        # Expected goals based on O/U market
        # Simplified: assumes Poisson distribution
        import numpy as np
        df['ExpectedGoals_FromOdds'] = -np.log(df['Under2_5_Prob']) / 0.5
    
    # Both Teams to Score (if available)
    if 'BTTS_Yes' in df.columns:
        df['BTTS_Prob'] = 1 / df['BTTS_Yes']
        df['BTTS_No_Prob'] = 1 / df['BTTS_No']
    
    # Odds movement (compare opening vs closing odds)
    if 'B365H_Open' in df.columns and 'B365H' in df.columns:
        df['HomeOdds_Movement'] = (df['B365H'] - df['B365H_Open']) / df['B365H_Open']
        df['DrawOdds_Movement'] = (df['B365D'] - df['B365D_Open']) / df['B365D_Open']
        df['AwayOdds_Movement'] = (df['B365A'] - df['B365A_Open']) / df['B365A_Open']
        
        # Detect sharp money (significant odds movement)
        df['SharpMoney_Home'] = (df['HomeOdds_Movement'] < -0.1).astype(int)
        df['SharpMoney_Away'] = (df['AwayOdds_Movement'] < -0.1).astype(int)
    
    # Betting volume indicators (if available)
    # Higher volume = more confident market
    if 'VolH' in df.columns:
        total_vol = df['VolH'] + df['VolD'] + df['VolA']
        df['HomeVolume_Pct'] = df['VolH'] / total_vol
        df['DrawVolume_Pct'] = df['VolD'] / total_vol
        df['AwayVolume_Pct'] = df['VolA'] / total_vol
    
    return df

# Fetch additional odds data from football-data.co.uk
def download_full_odds_data(season='2324'):
    """Download detailed odds data including Asian handicaps and O/U"""
    
    url = f'https://www.football-data.co.uk/mmz4281/{season}/E0.csv'
    
    try:
        df = pd.read_csv(url, encoding='utf-8')
        print(f"Downloaded {len(df)} matches with detailed odds for season {season}")
        return df
    except Exception as e:
        print(f"Error downloading odds data: {e}")
        return None
```

---

## 4. Venue-Specific Data

**Priority:** Medium  
**Effort:** 2 hours  
**Expected Impact:** Captures ground advantage variations

Add stadium-specific features like capacity, pitch dimensions, and historical win rates.

```python
# Create: data_files/stadium_data.csv (manual creation or web scraping)

STADIUM_DATA = {
    'Arsenal': {
        'Stadium': 'Emirates Stadium',
        'Capacity': 60704,
        'PitchLength': 105,
        'PitchWidth': 68,
        'Altitude': 41,  # meters
        'RoofType': 'Open',
        'HistoricalHomeWinRate': 0.61
    },
    'Chelsea': {
        'Stadium': 'Stamford Bridge',
        'Capacity': 40834,
        'PitchLength': 103,
        'PitchWidth': 67,
        'Altitude': 11,
        'RoofType': 'Open',
        'HistoricalHomeWinRate': 0.58
    },
    # ... continue for all 20 teams
}

def add_stadium_features(historical_df):
    """Add stadium-specific features to match data"""
    
    import pandas as pd
    
    # Convert to DataFrame
    stadium_df = pd.DataFrame.from_dict(STADIUM_DATA, orient='index').reset_index()
    stadium_df.columns = ['Team'] + list(stadium_df.columns[1:])
    
    # Merge with match data (only for home team since venue is home)
    historical_df = historical_df.merge(
        stadium_df[['Team', 'Capacity', 'PitchLength', 'PitchWidth', 
                   'HistoricalHomeWinRate']],
        left_on='HomeTeam',
        right_on='Team',
        how='left',
        suffixes=('', '_Stadium')
    )
    
    # Calculate pitch size advantage
    # Smaller pitches favor defensive teams, larger favor attacking teams
    avg_pitch_size = 105 * 68
    historical_df['PitchSizeRatio'] = (
        historical_df['PitchLength'] * historical_df['PitchWidth']
    ) / avg_pitch_size
    
    # Crowd capacity (higher = stronger home advantage)
    historical_df['CrowdPressure'] = historical_df['Capacity'] / 60000  # Normalized
    
    return historical_df

# Scrape stadium data from Wikipedia
def scrape_stadium_data():
    """Scrape Premier League stadium information from Wikipedia"""
    
    import requests
    from bs4 import BeautifulSoup
    import pandas as pd
    
    url = 'https://en.wikipedia.org/wiki/List_of_Premier_League_stadiums'
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Find the stadium table
    table = soup.find('table', {'class': 'wikitable'})
    
    stadium_data = []
    rows = table.find_all('tr')[1:]  # Skip header
    
    for row in rows:
        cols = row.find_all('td')
        if len(cols) >= 4:
            stadium_data.append({
                'Team': cols[0].text.strip(),
                'Stadium': cols[1].text.strip(),
                'Capacity': cols[2].text.strip().replace(',', ''),
                'Location': cols[3].text.strip()
            })
    
    return pd.DataFrame(stadium_data)
```

---

## 5. Match Scheduling Data

**Priority:** Low  
**Effort:** 1 hour  
**Expected Impact:** Captures fixture congestion effects

Add features for rest days, fixture density, and European competition schedule.

```python
# Add to prepare_model_data.py

def calculate_fixture_congestion(df):
    """
    Calculate fixture congestion metrics
    
    Features added:
    - Days since last match (rest days)
    - Matches in last 7/14/30 days
    - European competition involvement
    """
    
    df = df.sort_values(['HomeTeam', 'MatchDate'])
    
    # Calculate rest days for home team
    df['HomeRestDays'] = df.groupby('HomeTeam')['MatchDate'].diff().dt.days
    
    # Calculate rest days for away team
    df_away = df.sort_values(['AwayTeam', 'MatchDate'])
    df_away['AwayRestDays'] = df_away.groupby('AwayTeam')['MatchDate'].diff().dt.days
    
    # Merge back
    df = df.merge(
        df_away[['HomeTeam', 'AwayTeam', 'MatchDate', 'AwayRestDays']],
        on=['HomeTeam', 'AwayTeam', 'MatchDate'],
        how='left'
    )
    
    # Calculate fixture density (matches in rolling windows)
    for days in [7, 14, 30]:
        # Home team
        df[f'HomeMatches_Last{days}Days'] = df.groupby('HomeTeam').rolling(
            window=f'{days}D', on='MatchDate'
        ).size().reset_index(drop=True)
        
        # Away team
        df[f'AwayMatches_Last{days}Days'] = df.groupby('AwayTeam').rolling(
            window=f'{days}D', on='MatchDate'
        ).size().reset_index(drop=True)
    
    # Rest advantage (positive = home team more rested)
    df['RestAdvantage'] = df['HomeRestDays'] - df['AwayRestDays']
    
    # Fixture congestion score (higher = more congested)
    df['HomeCongestion'] = (
        df['HomeMatches_Last7Days'] * 3 +
        df['HomeMatches_Last14Days'] * 2 +
        df['HomeMatches_Last30Days']
    ) / 6
    
    df['AwayCongestion'] = (
        df['AwayMatches_Last7Days'] * 3 +
        df['AwayMatches_Last14Days'] * 2 +
        df['AwayMatches_Last30Days']
    ) / 6
    
    df['CongestionAdvantage'] = df['AwayCongestion'] - df['HomeCongestion']
    
    return df

# Add European competition data
def add_european_competition(df):
    """Add flags for teams playing in European competitions"""
    
    # Manually define teams in European competitions by season
    # Could be automated by scraping UEFA website
    
    EUROPEAN_TEAMS = {
        '2023-24': ['Man City', 'Arsenal', 'Liverpool', 'Chelsea', 
                   'Man United', 'Newcastle', 'Aston Villa'],
        '2024-25': ['Man City', 'Arsenal', 'Liverpool', 'Man United',
                   'Aston Villa', 'Tottenham']
    }
    
    # Determine season from match date
    df['Season'] = df['MatchDate'].dt.year.astype(str) + '-' + \
                   (df['MatchDate'].dt.year + 1).astype(str).str[-2:]
    
    # Add European competition flag
    df['HomeInEurope'] = df.apply(
        lambda row: 1 if row['HomeTeam'] in EUROPEAN_TEAMS.get(row['Season'], []) else 0,
        axis=1
    )
    df['AwayInEurope'] = df.apply(
        lambda row: 1 if row['AwayTeam'] in EUROPEAN_TEAMS.get(row['Season'], []) else 0,
        axis=1
    )
    
    # European schedule advantage (team not in Europe has advantage)
    df['EuropeanAdvantage'] = df['AwayInEurope'] - df['HomeInEurope']
    
    return df
```

---

## Implementation Priority

**Week 1:**
1. Enhanced Historical Odds Data (High Impact, Uses Existing Sources)
2. Match Scheduling Data (Medium Impact, Easy to Calculate)

**Week 2:**
3. Venue-Specific Data (Medium Impact, One-Time Setup)

**Week 3:**
4. Transfer Market Activity (Medium Impact, Captures Squad Changes)

**Week 4:**
5. Social Media Sentiment Analysis (Lower Impact, Experimental)

---

## Success Metrics

- **Enhanced Odds:** Reduce prediction MAE by 0.02-0.05
- **Stadium Data:** Improve home win prediction accuracy by 2-3%
- **Fixture Congestion:** Better prediction of tired team performances
- **Transfer Activity:** Capture mid-season squad strength changes
- **Sentiment:** Correlation >0.3 with actual team performance
