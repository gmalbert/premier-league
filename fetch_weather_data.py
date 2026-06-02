import openmeteo_requests
import datetime as dt
import requests_cache
import numpy as np
import pandas as pd
from retry_requests import retry
from openmeteo_sdk.Variable import Variable
import os
from datetime import datetime

# Setup the Open-Meteo API client with cache and retry on error
cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

# Stadium coordinates mapping
STADIUM_COORDS = {
    'Old Trafford': {'lat': 53.4631, 'lon': -2.2913},
    'Emirates Stadium': {'lat': 51.5549, 'lon': -0.1084},
    'Anfield': {'lat': 53.4308, 'lon': -2.9608},
    'Stamford Bridge': {'lat': 51.4817, 'lon': -0.1910},
    'Tottenham Hotspur Stadium': {'lat': 51.6043, 'lon': -0.0664},
    'Etihad Stadium': {'lat': 53.4831, 'lon': -2.2004},
    'Wembley Stadium': {'lat': 51.5560, 'lon': -0.2795},
    'Villa Park': {'lat': 52.5093, 'lon': -1.8848},
    'Goodison Park': {'lat': 53.4389, 'lon': -2.9663},
    'St. James\' Park': {'lat': 54.9756, 'lon': -1.6217},
    'Carrow Road': {'lat': 52.6221, 'lon': 1.3091},
    'Selhurst Park': {'lat': 51.3983, 'lon': -0.0855},
    'King Power Stadium': {'lat': 52.6203, 'lon': -1.1422},
    'The Hawthorns': {'lat': 52.5090, 'lon': -1.9639},
    'London Stadium': {'lat': 51.5383, 'lon': -0.0164},
    'Turf Moor': {'lat': 53.7890, 'lon': -2.2302},
    'Molineux Stadium': {'lat': 52.5903, 'lon': -2.1304},
    'Bramall Lane': {'lat': 53.3703, 'lon': -1.4708},
    'Riverside Stadium': {'lat': 54.5782, 'lon': -1.2168},
    'Madejski Stadium': {'lat': 51.4223, 'lon': -0.9828},
    'Crest': {'lat': 51.4223, 'lon': -0.9828},  # Reading's Madejski Stadium
    'Falmer Stadium': {'lat': 50.8618, 'lon': -0.0833},
    'Vicarage Road': {'lat': 51.6498, 'lon': -0.4015},
    'Kirklees Stadium': {'lat': 53.6544, 'lon': -1.7683},
    'Deepdale': {'lat': 53.7723, 'lon': -2.6883},
    'Bloomfield Road': {'lat': 53.8044, 'lon': -3.0481},
    'Gigg Lane': {'lat': 53.5806, 'lon': -2.5352},
    'Home Park': {'lat': 50.3882, 'lon': -4.1508},
    'Roots Hall': {'lat': 51.5602, 'lon': 0.5091},
    'Adams Park': {'lat': 51.6306, 'lon': -0.7992},
    'Salford City Stadium': {'lat': 53.4925, 'lon': -2.3333},
    'Brentford Community Stadium': {'lat': 51.4908, 'lon': -0.2887},
    'Vitality Stadium': {'lat': 50.7352, 'lon': -1.8383},
    'Chesil Stadium': {'lat': 50.7352, 'lon': -1.8383},  # AFC Bournemouth's Vitality Stadium
    'Craven Cottage': {'lat': 51.4750, 'lon': -0.2217},  # Fulham
    'Portman Road': {'lat': 52.0544, 'lon': 1.1450},  # Ipswich
    'Elland Road': {'lat': 53.7778, 'lon': -1.5722},  # Leeds
    'Kenilworth Road': {'lat': 51.8842, 'lon': -0.4314},  # Luton
    'City Ground': {'lat': 52.9399, 'lon': -1.1326},  # Nottingham Forest
    'St Mary\'s Stadium': {'lat': 50.9058, 'lon': -1.3911},  # Southampton
    'Stadium of Light': {'lat': 54.9146, 'lon': -1.3884},  # Sunderland
}

# Team to stadium mapping
STADIUM_MAP = {
    'Man United': 'Old Trafford',
    'Arsenal': 'Emirates Stadium',
    'Liverpool': 'Anfield',
    'Chelsea': 'Stamford Bridge',
    'Tottenham': 'Tottenham Hotspur Stadium',
    'Man City': 'Etihad Stadium',
    'Aston Villa': 'Villa Park',
    'Everton': 'Goodison Park',
    'Newcastle': 'St. James\' Park',
    'Norwich': 'Carrow Road',
    'Crystal Palace': 'Selhurst Park',
    'Leicester': 'King Power Stadium',
    'West Brom': 'The Hawthorns',
    'West Ham': 'London Stadium',
    'Burnley': 'Turf Moor',
    'Wolves': 'Molineux Stadium',
    'Sheffield United': 'Bramall Lane',
    'Middlesbrough': 'Riverside Stadium',
    'Reading': 'Madejski Stadium',
    'Brighton': 'Falmer Stadium',
    'Watford': 'Vicarage Road',
    'Huddersfield': 'Kirklees Stadium',
    'Preston': 'Deepdale',
    'Blackpool': 'Bloomfield Road',
    'Bury': 'Gigg Lane',
    'Plymouth': 'Home Park',
    'Southend': 'Roots Hall',
    'Wycombe': 'Adams Park',
    'Salford City': 'Salford City Stadium',
    'Brentford': 'Brentford Community Stadium',
    'Bournemouth': 'Vitality Stadium',
    'Portsmouth': 'Fratton Park',
    'Fulham': 'Craven Cottage',
    'Ipswich': 'Portman Road',
    'Leeds': 'Elland Road',
    'Luton': 'Kenilworth Road',
    'Nott\'m Forest': 'City Ground',
    'Southampton': 'St Mary\'s Stadium',
    'Sunderland': 'Stadium of Light',
}

def fetch_match_weather(stadium_location, match_date, api_key=None):
    """
    Fetch weather conditions for match day using Open-Meteo API (completely free)
    API: https://open-meteo.com/

    Args:
        stadium_location (str): Name of the stadium
        match_date (str or datetime): Date in YYYY-MM-DD format or datetime object
        api_key (str): Not needed for Open-Meteo (ignored)

    Returns:
        dict: Weather data or None if failed
    """

    # Convert datetime to string if needed
    if hasattr(match_date, 'strftime'):
        match_date = match_date.strftime('%Y-%m-%d')
    elif isinstance(match_date, str) and ' ' in match_date:
        # Handle string with timestamp
        match_date = match_date.split(' ')[0]

    coords = STADIUM_COORDS.get(stadium_location)
    if not coords:
        print(f"Warning: No coordinates found for stadium '{stadium_location}'")
        return None

    try:
        # Open-Meteo Historical Weather API (completely free, no API key required)
        url = "https://archive-api.open-meteo.com/v1/archive"

        params = {
            "latitude": coords['lat'],
            "longitude": coords['lon'],
            "start_date": match_date,
            "end_date": match_date,
            "hourly": ["temperature_2m", "precipitation", "relative_humidity_2m", "wind_speed_10m"],
            "daily": ["weathercode"],
            "temperature_unit": "celsius",
            "wind_speed_unit": "ms",
            "timezone": "Europe/London"
        }

        responses = openmeteo.weather_api(url, params=params)
        response = responses[0]

        # Process hourly data. The order of variables needs to be the same as requested.
        hourly = response.Hourly()
        hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
        hourly_precipitation = hourly.Variables(1).ValuesAsNumpy()
        hourly_relative_humidity_2m = hourly.Variables(2).ValuesAsNumpy()
        hourly_wind_speed_10m = hourly.Variables(3).ValuesAsNumpy()

        # Process daily data
        daily = response.Daily()
        daily_weathercode = daily.Variables(0).ValuesAsNumpy()

        # Get data for typical kickoff time (around 16:00/17:00)
        # Find the hour closest to 16:00 (4 PM)
        target_hour = 16
        hours = pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left"
        ).hour

        if len(hours) > 0:
            # Find closest hour to target
            hour_diffs = [abs(h - target_hour) for h in hours]
            closest_idx = hour_diffs.index(min(hour_diffs))

            temperature = float(hourly_temperature_2m[closest_idx]) if len(hourly_temperature_2m) > closest_idx else None
            humidity = float(hourly_relative_humidity_2m[closest_idx]) if len(hourly_relative_humidity_2m) > closest_idx else None
            wind_speed = float(hourly_wind_speed_10m[closest_idx]) if len(hourly_wind_speed_10m) > closest_idx else None
            precipitation = float(hourly_precipitation[closest_idx]) if len(hourly_precipitation) > closest_idx else None

            # Get weather condition from daily data
            weather_code = int(daily_weathercode[0]) if len(daily_weathercode) > 0 else None

            # Convert WMO weather code to description
            weather_descriptions = {
                0: 'Clear sky', 1: 'Mainly clear', 2: 'Partly cloudy', 3: 'Overcast',
                45: 'Fog', 48: 'Depositing rime fog',
                51: 'Light drizzle', 53: 'Moderate drizzle', 55: 'Dense drizzle',
                56: 'Light freezing drizzle', 57: 'Dense freezing drizzle',
                61: 'Slight rain', 63: 'Moderate rain', 65: 'Heavy rain',
                66: 'Light freezing rain', 67: 'Heavy freezing rain',
                71: 'Slight snow fall', 73: 'Moderate snow fall', 75: 'Heavy snow fall',
                77: 'Snow grains', 80: 'Slight rain showers', 81: 'Moderate rain showers',
                82: 'Violent rain showers', 85: 'Slight snow showers', 86: 'Heavy snow showers',
                95: 'Thunderstorm', 96: 'Thunderstorm with slight hail', 99: 'Thunderstorm with heavy hail'
            }

            weather_condition = weather_descriptions.get(weather_code, 'Unknown') if weather_code is not None else 'Unknown'

            return {
                'Temperature': temperature,
                'Humidity': humidity,
                'WindSpeed': wind_speed,
                'Precipitation': precipitation,
                'WeatherCondition': weather_condition,
                'WeatherDescription': weather_condition
            }
        else:
            print(f"Warning: No hourly data available for {match_date} at {stadium_location}")
            return None

    except Exception as e:
        print(f"Error fetching weather for {stadium_location} on {match_date}: {e}")
        return None


def add_weather_features(df, api_key=None, cache_file='weather_cache.csv'):
    """
    Add weather data to match dataframe using batch processing

    Args:
        df (pd.DataFrame): Match dataframe with HomeTeam and MatchDate columns
        api_key (str): Not needed for Open-Meteo (ignored)
        cache_file (str): File to cache weather data

    Returns:
        pd.DataFrame: DataFrame with weather features added
    """

    # Try to load cached weather data
    cache_path = f'data_files/{cache_file}'
    if os.path.exists(cache_path):
        try:
            cached_weather = pd.read_csv(cache_path)
            print(f"Loaded {len(cached_weather)} cached weather records")
        except Exception:
            cached_weather = pd.DataFrame()
    else:
        cached_weather = pd.DataFrame()

    # Prepare data for batch processing
    df_copy = df.copy()
    df_copy['Stadium'] = df_copy['HomeTeam'].map(STADIUM_MAP)

    # Get unique stadium-date combinations that need weather data
    weather_requests = []
    for idx, match in df_copy.iterrows():
        stadium = match['Stadium']
        match_date = match['MatchDate']
        if pd.notna(stadium):
            cache_key = f"{match['HomeTeam']}_{pd.Timestamp(match_date).strftime('%Y-%m-%d')}"
            # Check if we already have this in cache
            if cached_weather.empty or cache_key not in cached_weather['cache_key'].values:
                weather_requests.append({
                    'stadium': stadium,
                    'date': match_date,
                    'cache_key': cache_key,
                    'home_team': match['HomeTeam']
                })

    print(f"Need to fetch weather for {len(weather_requests)} unique stadium-date combinations")

    # Batch process weather requests
    new_weather_data = []
    batch_size = 50  # Process in batches to avoid memory issues

    for i in range(0, len(weather_requests), batch_size):
        batch = weather_requests[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(len(weather_requests)-1)//batch_size + 1}")

        # Group by date for this batch to minimize API calls
        date_groups = {}
        for req in batch:
            date = req['date']
            if date not in date_groups:
                date_groups[date] = []
            date_groups[date].append(req)

        # Process each date
        for date, requests in date_groups.items():
            try:
                # Get all stadiums for this date
                stadiums = list(set(req['stadium'] for req in requests))

                # For each stadium on this date, fetch weather
                for stadium in stadiums:
                    weather = fetch_match_weather(stadium, date)
                    if weather:
                        # Apply to all matches at this stadium on this date
                        for req in requests:
                            if req['stadium'] == stadium:
                                weather_entry = weather.copy()
                                weather_entry['cache_key'] = req['cache_key']
                                weather_entry['stadium'] = stadium
                                new_weather_data.append(weather_entry)

            except Exception as e:
                print(f"Error processing date {date}: {e}")
                continue

    # Save new weather data to cache
    if new_weather_data:
        new_weather_df = pd.DataFrame(new_weather_data)
        if not cached_weather.empty:
            combined_cache = pd.concat([cached_weather, new_weather_df], ignore_index=True)
        else:
            combined_cache = new_weather_df

        # Remove duplicates based on cache_key
        combined_cache = combined_cache.drop_duplicates(subset=['cache_key'])

        os.makedirs('data_files', exist_ok=True)
        combined_cache.to_csv(cache_path, index=False)
        print(f"Cached {len(new_weather_data)} new weather records")

    # Now merge weather data back to the original dataframe
    if not cached_weather.empty or new_weather_data:
        all_weather = pd.concat([cached_weather, pd.DataFrame(new_weather_data)], ignore_index=True) if new_weather_data else cached_weather
        all_weather = all_weather.drop_duplicates(subset=['cache_key'])

        # Create weather features dataframe
        weather_features = []
        for idx, match in df.iterrows():
            cache_key = f"{match['HomeTeam']}_{pd.Timestamp(match['MatchDate']).strftime('%Y-%m-%d')}"
            weather_row = all_weather[all_weather['cache_key'] == cache_key]

            if not weather_row.empty:
                weather = weather_row.iloc[0].to_dict()
                # Remove cache-specific columns
                clean_weather = {k: v for k, v in weather.items() if k not in ['cache_key', 'stadium']}
                weather_features.append(clean_weather)
            else:
                weather_features.append({
                    'Temperature': None,
                    'Humidity': None,
                    'WindSpeed': None,
                    'Precipitation': 0,
                    'WeatherCondition': 'Unknown',
                    'WeatherDescription': 'Unknown'
                })

        weather_df = pd.DataFrame(weather_features)
        result_df = pd.concat([df, weather_df], axis=1)
        return result_df

    # If no weather data available, return original dataframe with empty weather columns
    empty_weather = pd.DataFrame({
        'Temperature': [None] * len(df),
        'Humidity': [None] * len(df),
        'WindSpeed': [None] * len(df),
        'Precipitation': [0] * len(df),
        'WeatherCondition': ['Unknown'] * len(df),
        'WeatherDescription': ['Unknown'] * len(df)
    })
    return pd.concat([df, empty_weather], axis=1)

def categorize_weather_impact(row):
    """
    Categorize weather impact on match

    Args:
        row: DataFrame row with weather columns

    Returns:
        str: Weather impact category
    """
    if pd.isna(row.get('Precipitation', 0)):
        return 'Unknown'

    if row['Precipitation'] > 5:
        return 'Heavy Rain'
    elif pd.notna(row.get('WindSpeed', None)) and row['WindSpeed'] > 15:
        return 'Windy'
    elif pd.notna(row.get('Temperature', None)) and row['Temperature'] < 5:
        return 'Cold'
    elif pd.notna(row.get('Temperature', None)) and row['Temperature'] > 25:
        return 'Hot'
    else:
        return 'Normal'

def add_weather_impact_category(df):
    """
    Add weather impact category column to dataframe

    Args:
        df (pd.DataFrame): DataFrame with weather columns

    Returns:
        pd.DataFrame: DataFrame with WeatherImpact column added
    """
    df = df.copy()
    df['WeatherImpact'] = df.apply(categorize_weather_impact, axis=1)
    return df

if __name__ == "__main__":
    # Test the weather fetching
    print("Testing weather data fetching with Open-Meteo (completely free)...")

    # Test with a recent match
    test_weather = fetch_match_weather('Old Trafford', '2024-12-01')
    if test_weather:
        print("Sample weather data:")
        for key, value in test_weather.items():
            print(f"  {key}: {value}")
    else:
        print("Failed to fetch weather data. Check your internet connection.")