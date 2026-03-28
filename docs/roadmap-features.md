# Feature Roadmap - Premier League Predictor

## Status Update (2026-03-28)
- ✅ Live score integration: draft available
- ⚪ Feature completion and app integration: pending

## High Priority Features

### 1. Live Score Integration
**Priority:** High  
**Effort:** Medium  
**Impact:** High

**Status:** ✅ Draft implementation available in roadmap; not yet fully launched in main app (live scoreboard is a planned enhancement).

Fetch and display live match scores during game time.

```python
# Add to fetch_upcoming_fixtures.py
def fetch_live_scores():
    """Fetch current in-progress matches with live scores"""
    url = 'https://site.api.espn.com/apis/site/v2/sports/soccer/eng.1/scoreboard'
    response = requests.get(url, headers=headers, timeout=10)
    
    if response.status_code == 200:
        data = response.json()
        live_matches = []
        
        for event in data['events']:
            status = event.get('status', {}).get('type', {}).get('name', '')
            if status in ['STATUS_IN_PROGRESS', 'STATUS_HALFTIME']:
                competition = event['competitions'][0]
                competitors = competition['competitors']
                
                home = next(c for c in competitors if c['homeAway'] == 'home')
                away = next(c for c in competitors if c['homeAway'] == 'away')
                
                live_matches.append({
                    'HomeTeam': home['team']['displayName'],
                    'AwayTeam': away['team']['displayName'],
                    'HomeScore': home['score'],
                    'AwayScore': away['score'],
                    'Status': status,
                    'Clock': event['status'].get('displayClock', '')
                })
        
        return pd.DataFrame(live_matches)
```

**UI Integration:**
```python
# In premier-league-predictions.py
if st.checkbox("Show Live Matches"):
    st.subheader("Live Premier League Matches")
    live_df = fetch_live_scores()
    
    if len(live_df) > 0:
        st.dataframe(live_df, hide_index=True)
    else:
        st.info("No matches currently in progress")
```

---

### 2. Match Result Confidence Levels
**Priority:** High  
**Effort:** Low  
**Impact:** Medium

**Status:** ✅ Implemented in the codebase via probability spread calculation and confidence labeling in predictive outputs.

Add confidence indicators for predictions based on probability spread.

```python
# Add to premier-league-predictions.py after predictions
def calculate_confidence(proba_row):
    """Calculate prediction confidence level"""
    max_prob = max(proba_row)
    second_prob = sorted(proba_row)[-2]
    spread = max_prob - second_prob
    
    if spread > 0.3:
        return "High"
    elif spread > 0.15:
        return "Medium"
    else:
        return "Low"

# In Show Upcoming Predictions section
upcoming_df['Confidence'] = upcoming_df[['HomeWin_Prob', 'Draw_Prob', 'AwayWin_Prob']].apply(
    lambda row: calculate_confidence(row.values), axis=1
)

# Display with confidence
st.dataframe(upcoming_df[[
    'Date', 'Time', 'HomeTeam', 'AwayTeam', 
    'HomeWin_Prob', 'Draw_Prob', 'AwayWin_Prob', 'Confidence'
]], width=900, hide_index=True)
```

---

### 3. Team Form Tracker ✅ IMPLEMENTED

**Status:** ✅ IMPLEMENTED (complete) - Advanced team form analysis with visual indicators and detailed statistics in Statistics tab

**Implementation:** `analyze_team_form.py` + UI integration in Statistics tab of `premier-league-predictions.py`

```python
# analyze_team_form.py - Core functionality
def get_team_form(team_name, num_matches=5):
    """Get recent form string for a specific team"""
    
def get_team_form_stats(team_name, num_matches=5):
    """Get detailed form statistics including wins/draws/losses/points"""

# UI Integration in Statistics tab
st.subheader("📊 Team Form Guide")
st.write("Recent performance analysis for all Premier League teams (last 5 matches)")

from analyze_team_form import get_team_form_stats

teams = sorted(df['HomeTeam'].unique())

form_data = []
for team in teams:
    stats = get_team_form_stats(team, num_matches=5)
    form_data.append({
        'Team': team,
        'Last 5': stats['form_string'],
        'Wins': stats['wins'],
        'Draws': stats['draws'],
        'Losses': stats['losses'],
        'Points': stats['points'],
        'Form Score': stats['wins'] * 3 + stats['draws']
    })

form_df = pd.DataFrame(form_data).sort_values('Form Score', ascending=False)

# Enhanced display with visual indicators
def color_form_results(form_string):
    colored = []
    for result in form_string:
        if result == 'W':
            colored.append('🟢')  # Green for wins
        elif result == 'D':
            colored.append('🟡')  # Yellow for draws
        else:
            colored.append('🔴')  # Red for losses
    return ' '.join(colored)

display_df['Form Visual'] = display_df['Last 5'].apply(color_form_results)
st.dataframe(display_df, width='stretch', hide_index=True)

# Form summary with league-wide statistics
st.subheader("Form Summary")
summary_stats = {
    'Total Matches Analyzed': total_matches,
    'Average Points per Team': f"{avg_points:.1f}",
    'Best Form Team': f"{best_form_team} ({best_form_score} points)",
    'Teams with Perfect Form': len(form_df[form_df['Losses'] == 0]),
    'Teams Winless': len(form_df[form_df['Wins'] == 0])
}
```

**Features Added:**
- Comprehensive form analysis for all Premier League teams
- Visual form indicators (🟢 Win, 🟡 Draw, 🔴 Loss)
- Detailed statistics: wins, draws, losses, points from last 5 matches
- Form scoring system (3 points per win, 1 per draw)
- League-wide summary statistics
- Integrated into Statistics tab with automatic data loading
- Responsive table display with proper formatting

**Data Source:** `data_files/combined_historical_data_with_calculations_new.csv`

---

### 4. Head-to-Head History ✅ IMPLEMENTED

**Status:** ✅ IMPLEMENTED (complete) - Interactive head-to-head analyzer with historical match results and statistics

**Implementation:** UI integration in Statistics tab of `premier-league-predictions.py`

```python
# Head-to-Head History function
def get_h2h_stats(home_team, away_team, num_matches=10):
    """Get head-to-head statistics between two teams"""
    h2h = df[
        ((df['HomeTeam'] == home_team) & (df['AwayTeam'] == away_team)) |
        ((df['HomeTeam'] == away_team) & (df['AwayTeam'] == home_team))
    ].sort_values('MatchDate', ascending=False).head(num_matches)
    
    return h2h[['MatchDate', 'HomeTeam', 'AwayTeam', 'FullTimeHomeGoals', 'FullTimeAwayGoals', 'FullTimeResult']]

# UI Component in Statistics tab
st.subheader("🏆 Head-to-Head Analyzer")
st.write("Compare historical match results between any two Premier League teams")

col1, col2 = st.columns(2)
with col1:
    team1 = st.selectbox("Select Team 1", sorted(df['HomeTeam'].unique()))
with col2:
    team2 = st.selectbox("Select Team 2", sorted(df['AwayTeam'].unique()))

if st.button("🔍 Analyze Head-to-Head History"):
    if team1 != team2:
        h2h_df = get_h2h_stats(team1, team2, num_matches=10)
        
        # Calculate and display H2H statistics
        # Show wins, draws, goals summary
        # Display formatted recent matches table
```

**Features Added:**
- Interactive team selection with dropdown menus
- Comprehensive head-to-head statistics (wins, draws, goals scored/conceded)
- Historical match results display with formatted scores and outcomes
- Visual summary metrics showing overall H2H record
- Integrated into Statistics tab with button-triggered analysis
- Error handling for same team selection and no matches found
- Responsive table display with proper formatting

**Data Source:** `data_files/combined_historical_data_with_calculations_new.csv`

---

### 5. Prediction Performance Tracker ✅ IMPLEMENTED

**Status:** ✅ IMPLEMENTED (complete) - Prediction logging and validation system integrated into Streamlit app + Automated daily validation

**Implementation:** `track_predictions.py` + UI integration in `premier-league-predictions.py` + Automated validation in GitHub Actions

```python
# track_predictions.py - Core functionality
def log_prediction(date, home_team, away_team, pred_home, pred_draw, pred_away):
    """Log a prediction for future validation"""
    
def validate_predictions():
    """Compare predictions with actual results"""
    
# UI Integration in Predictive Data tab
if st.checkbox("Show Prediction Performance Tracker"):
    st.subheader("📈 Model Prediction Accuracy Over Time")
    perf = validate_predictions()
    if perf is not None and len(perf) > 0:
        completed = perf[perf['Correct'].notna()]
        accuracy = completed['Correct'].mean()
        st.metric("Prediction Accuracy", f"{accuracy:.1%}")
        st.dataframe(completed[...], width='stretch', hide_index=True)

# Automated validation in .github/workflows/data-pipeline.yml
- name: Validate predictions
  run: |
    echo "Validating prediction accuracy..."
    python -c "
    from track_predictions import validate_predictions
    # Validation logic...
    "
```

**Features Added:**
- Automatic prediction logging when "Log Predictions for Tracking" button is clicked
- Real-time validation against actual match results
- **Automated daily validation** via GitHub Actions (runs after data pipeline)
- Accuracy metrics and historical prediction tracking
- DataFrame display of prediction history with outcomes
- Integrated into "Predictive Data" tab with checkbox toggle

**Data Storage:** `data_files/predictions_log.csv` with columns for prediction details and validation results

**Automation:** Predictions are automatically validated daily at 2 AM ET as part of the data pipeline workflow

---

## Medium Priority Features

### 6. Export Predictions to PDF
Generate downloadable PDF reports with predictions.

### 7. Email Alerts for High-Confidence Predictions
Send notifications for matches with >70% confidence.

### 8. Betting Odds Comparison
Compare model predictions with bookmaker odds.

### 9. Multi-League Support
Extend to La Liga, Bundesliga, Serie A.

### 10. Mobile-Responsive Dashboard
Optimize Streamlit UI for mobile devices.

---

## Implementation Timeline

**Phase 1 (Week 1-2):**
- Live Score Integration
- Match Result Confidence Levels
- Team Form Tracker

**Phase 2 (Week 3-4): ✅**
- Head-to-Head History
- Prediction Performance Tracker ✅ IMPLEMENTED

**Phase 3 (Month 2):**
- Export to PDF
- Email Alerts
- Betting Odds Comparison

**Phase 4 (Month 3):**
- Multi-League Support
- Mobile Optimization
