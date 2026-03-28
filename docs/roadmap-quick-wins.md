# Quick Wins - Immediate Improvements

## Status Update (2026-03-28)
- ✅ Last update timestamp: completed
- ⚪ Add match commentary: pending
- ⚪ Color-code confidence: pending
- ⚪ Sorting in tables: pending
- ⚪ Export CSV: pending
- ⚪ Top features chart: pending

Easy-to-implement enhancements that provide immediate value with minimal effort.

---

## 1. Add Match Commentary
**Effort:** 5 minutes  
**Impact:** Medium

Display predicted winner with natural language.

```python
# Add to premier-league-predictions.py after predictions

def generate_match_commentary(row):
    """Generate natural language prediction"""
    probs = [
        ('Home Win', row['HomeWin_Prob']),
        ('Draw', row['Draw_Prob']),
        ('Away Win', row['AwayWin_Prob'])
    ]
    
    prediction = max(probs, key=lambda x: x[1])
    confidence = prediction[1]
    
    if confidence > 0.6:
        strength = "highly likely"
    elif confidence > 0.45:
        strength = "likely"
    else:
        strength = "possible"
    
    if prediction[0] == 'Home Win':
        return f"**{row['HomeTeam']}** {strength} to win ({confidence:.1%} confidence)"
    elif prediction[0] == 'Draw':
        return f"Match {strength} to end in a draw ({confidence:.1%} confidence)"
    else:
        return f"**{row['AwayTeam']}** {strength} to win ({confidence:.1%} confidence)"

# In Show Upcoming Predictions section
upcoming_df['Prediction'] = upcoming_df.apply(generate_match_commentary, axis=1)

st.subheader("Upcoming Match Predictions")
for _, match in upcoming_df.iterrows():
    with st.expander(f"{match['HomeTeam']} vs {match['AwayTeam']} - {match['Date']}"):
        st.write(match['Prediction'])
        
        # Show probability bars
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Home Win", f"{match['HomeWin_Prob']:.1%}")
        with col2:
            st.metric("Draw", f"{match['Draw_Prob']:.1%}")
        with col3:
            st.metric("Away Win", f"{match['AwayWin_Prob']:.1%}")
```

---

## 2. Color-Code Confidence Levels
**Effort:** 3 minutes  
**Impact:** Medium

Visual indicators for prediction strength.

```python
def color_confidence(val):
    """Color-code predictions by confidence"""
    if val > 0.6:
        return 'background-color: #90EE90'  # Light green
    elif val > 0.45:
        return 'background-color: #FFFFE0'  # Light yellow
    else:
        return 'background-color: #FFB6C1'  # Light red

# Apply styling
styled_df = upcoming_df[['Date', 'Time', 'HomeTeam', 'AwayTeam', 
                         'HomeWin_Prob', 'Draw_Prob', 'AwayWin_Prob']].style.applymap(
    color_confidence, subset=['HomeWin_Prob', 'Draw_Prob', 'AwayWin_Prob']
)

st.dataframe(styled_df, hide_index=True)
```

---

## 3. Add Sorting to Data Tables
**Effort:** 2 minutes  
**Impact:** Low

Allow users to sort predictions by different columns.

```python
# Replace static dataframe with interactive one
st.dataframe(
    upcoming_df,
    height=get_dataframe_height(upcoming_df),
    use_container_width=True,
    hide_index=True,
    column_config={
        "HomeWin_Prob": st.column_config.ProgressColumn(
            "Home Win %",
            format="%.1f%%",
            min_value=0,
            max_value=1,
        ),
        "Draw_Prob": st.column_config.ProgressColumn(
            "Draw %",
            format="%.1f%%",
            min_value=0,
            max_value=1,
        ),
        "AwayWin_Prob": st.column_config.ProgressColumn(
            "Away Win %",
            format="%.1f%%",
            min_value=0,
            max_value=1,
        ),
    }
)
```

---

## 4. Export to CSV Button
**Effort:** 3 minutes  
**Impact:** Medium

Let users download predictions.

```python
import io

# Add download button
@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

csv = convert_df_to_csv(upcoming_df)

st.download_button(
    label="📥 Download Predictions as CSV",
    data=csv,
    file_name=f'pl_predictions_{datetime.now().strftime("%Y%m%d")}.csv',
    mime='text/csv',
)
```

---

## 5. Add Last Update Timestamp ✅ **COMPLETED**
**Effort:** 2 minutes  
**Impact:** Low

Show when data was last refreshed.

```python
import os
from datetime import datetime

# At top of app
fixtures_file = path.join(DATA_DIR, 'upcoming_fixtures.csv')
if path.exists(fixtures_file):
    last_updated = datetime.fromtimestamp(os.path.getmtime(fixtures_file))
    st.caption(f"Last updated: {last_updated.strftime('%Y-%m-%d %I:%M %p ET')}")
```

---

## 6. Filter by Date Range
**Effort:** 5 minutes  
**Impact:** Medium

Allow users to filter upcoming matches by date.

```python
if st.checkbox("Show Upcoming Matches"):
    upcoming_csv = path.join(DATA_DIR, 'upcoming_fixtures.csv')
    if path.exists(upcoming_csv):
        upcoming_df = pd.read_csv(upcoming_csv)
        
        # Date filter
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("From Date", value=pd.to_datetime(upcoming_df['Date'].min()))
        with col2:
            end_date = st.date_input("To Date", value=pd.to_datetime(upcoming_df['Date'].max()))
        
        # Filter dataframe
        filtered_df = upcoming_df[
            (pd.to_datetime(upcoming_df['Date']) >= pd.to_datetime(start_date)) &
            (pd.to_datetime(upcoming_df['Date']) <= pd.to_datetime(end_date))
        ]
        
        st.write(f"Found {len(filtered_df)} matches")
        st.dataframe(filtered_df, hide_index=True)
```

---

## 7. Show Top Features Chart
**Effort:** 5 minutes  
**Impact:** High

Visualize most important features.

```python
import plotly.express as px

if st.checkbox("Show Predictive Data"):
    # ... existing model training code ...
    
    # After importance calculation
    st.subheader("Top 10 Most Important Features")
    
    top_features = importance_df.head(10)
    
    fig = px.bar(
        top_features,
        x='Mean Importance (%)',
        y='Feature',
        orientation='h',
        title='Feature Importance Rankings',
        color='Mean Importance (%)',
        color_continuous_scale='Viridis'
    )
    
    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig, use_container_width=True)
```

**Add to requirements.txt:**
```
plotly
```

---

## 8. Add Model Accuracy Widget ✅ **COMPLETED**
**Effort:** 2 minutes  
**Impact:** Medium

Prominent display of current model performance.

```python
# After model evaluation
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label="Model Accuracy",
        value=f"{acc:.1%}",
        delta=f"{acc - 0.5:.1%} vs random"
    )

with col2:
    st.metric(
        label="Mean Absolute Error",
        value=f"{mae:.3f}",
        delta=f"{0.67 - mae:.3f} vs baseline"
    )

with col3:
    predictions_count = len(y_test)
    st.metric(
        label="Test Predictions",
        value=predictions_count
    )
```

---

## 9. Add Team Filter
**Effort:** 3 minutes  
**Impact:** Medium

Filter predictions for specific teams.

```python
if st.checkbox("Show Upcoming Predictions"):
    # ... load data ...
    
    # Team filter
    all_teams = sorted(set(upcoming_df['HomeTeam'].unique()) | set(upcoming_df['AwayTeam'].unique()))
    selected_team = st.selectbox("Filter by team (optional)", ['All Teams'] + all_teams)
    
    if selected_team != 'All Teams':
        upcoming_df = upcoming_df[
            (upcoming_df['HomeTeam'] == selected_team) | 
            (upcoming_df['AwayTeam'] == selected_team)
        ]
    
    st.write(f"Showing {len(upcoming_df)} matches")
    st.dataframe(upcoming_df, hide_index=True)
```

---

## 10. Improve Page Layout
**Effort:** 5 minutes  
**Impact:** High

Better organization with tabs.

```python
# Replace checkboxes with tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Raw Data", 
    "🔮 Predictions", 
    "📅 Upcoming Matches",
    "📈 Model Performance"
])

with tab1:
    st.subheader("Historical Match Data")
    df_sorted = df.sort_values(by=['MatchDate', 'KickoffTime'], ascending=[False, False])
    st.dataframe(df_sorted, height=get_dataframe_height(df_sorted), hide_index=True)

with tab2:
    # Prediction code
    pass

with tab3:
    # Upcoming matches code
    pass

with tab4:
    # Model performance code
    pass
```

---

## 11. Add Refresh Data Button
**Effort:** 3 minutes  
**Impact:** High

Manual refresh of fixture data.

```python
st.sidebar.subheader("Data Management")

if st.sidebar.button("🔄 Refresh Upcoming Fixtures"):
    with st.spinner("Fetching latest fixtures..."):
        import subprocess
        result = subprocess.run(
            ['python', 'fetch_upcoming_fixtures.py'],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            st.sidebar.success("✅ Fixtures updated!")
            st.rerun()
        else:
            st.sidebar.error("❌ Update failed")
```

---

## 12. Add Match Time Countdown
**Effort:** 5 minutes  
**Impact:** Medium

Show time remaining until next match.

```python
from datetime import datetime, timedelta

def get_next_match_countdown(upcoming_df):
    """Calculate time until next match"""
    if len(upcoming_df) == 0:
        return None
    
    upcoming_df['DateTime'] = pd.to_datetime(
        upcoming_df['Date'] + ' ' + upcoming_df['Time']
    )
    
    next_match = upcoming_df.sort_values('DateTime').iloc[0]
    time_until = next_match['DateTime'] - datetime.now()
    
    if time_until.total_seconds() > 0:
        days = time_until.days
        hours = time_until.seconds // 3600
        minutes = (time_until.seconds % 3600) // 60
        
        return {
            'match': f"{next_match['HomeTeam']} vs {next_match['AwayTeam']}",
            'time': f"{days}d {hours}h {minutes}m",
            'datetime': next_match['DateTime']
        }
    
    return None

# Display in sidebar
countdown = get_next_match_countdown(upcoming_df)
if countdown:
    st.sidebar.info(f"⏱️ Next match in: **{countdown['time']}**")
    st.sidebar.write(countdown['match'])
```

---

## Implementation Checklist

- [ ] Add match commentary (5 min)
- [ ] Color-code confidence (3 min)
- [ ] Export CSV button (3 min)
- [x] Last update timestamp (2 min)
- [x] Model accuracy widget (2 min)
- [ ] Top features chart (5 min)
- [ ] Team filter (3 min)
- [ ] Improve layout with tabs (5 min)
- [ ] Refresh data button (3 min)
- [ ] Match countdown (5 min)

**Total implementation time: ~35 minutes**  
**Combined impact: Very High**
