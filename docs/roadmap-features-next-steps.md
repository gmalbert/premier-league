# Feature Development - Next Steps

## Status Update (2026-03-28)
- ✅ Team form/h2h/prediction tracker: completed
- ⚪ Live match tracker: pending
- ⚪ Odds comparison tool: pending
- ⚪ Interactive commentary generator: pending
- ⚪ PDF report export: pending

Building on successfully implemented features (Team Form Tracker, Head-to-Head History, Prediction Performance Tracker), here are the next actionable enhancements to improve user experience and functionality.

---

## 1. Live Match Tracker with Auto-Refresh

**Priority:** High  
**Effort:** 2-3 hours  
**Expected Impact:** Real-time engagement during match days

Display live scores with automatic updates and prediction validation.

```python
# Create: live_match_tracker.py

import requests
import pandas as pd
from datetime import datetime
import streamlit as st
import time

def fetch_live_matches():
    """
    Fetch currently in-progress Premier League matches
    Returns DataFrame with live scores and match status
    """
    
    url = 'https://site.api.espn.com/apis/site/v2/sports/soccer/eng.1/scoreboard'
    headers = {'User-Agent': 'Mozilla/5.0'}
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code != 200:
            return pd.DataFrame()
        
        data = response.json()
        live_matches = []
        
        for event in data.get('events', []):
            status = event.get('status', {})
            status_type = status.get('type', {}).get('name', '')
            
            # Check if match is live
            if status_type in ['STATUS_IN_PROGRESS', 'STATUS_HALFTIME', 
                              'STATUS_SCHEDULED', 'STATUS_FULL_TIME']:
                
                competition = event['competitions'][0]
                competitors = competition['competitors']
                
                home = next(c for c in competitors if c['homeAway'] == 'home')
                away = next(c for c in competitors if c['homeAway'] == 'away')
                
                # Get live statistics if available
                stats = competition.get('details', [])
                
                live_matches.append({
                    'HomeTeam': home['team']['displayName'],
                    'AwayTeam': away['team']['displayName'],
                    'HomeScore': int(home.get('score', 0)),
                    'AwayScore': int(away.get('score', 0)),
                    'Status': status_type.replace('STATUS_', ''),
                    'Clock': status.get('displayClock', 'N/A'),
                    'Period': status.get('period', 0),
                    'MatchId': event.get('id', '')
                })
        
        return pd.DataFrame(live_matches)
    
    except Exception as e:
        print(f"Error fetching live matches: {e}")
        return pd.DataFrame()

def get_match_statistics(match_id):
    """Fetch detailed statistics for a live match"""
    
    url = f'https://site.api.espn.com/apis/site/v2/sports/soccer/eng.1/summary?event={match_id}'
    headers = {'User-Agent': 'Mozilla/5.0'}
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            
            # Extract statistics
            stats = data.get('boxscore', {}).get('teams', [])
            
            if len(stats) >= 2:
                home_stats = stats[0]['statistics']
                away_stats = stats[1]['statistics']
                
                return {
                    'home': {stat['name']: stat['displayValue'] for stat in home_stats},
                    'away': {stat['name']: stat['displayValue'] for stat in away_stats}
                }
        
        return None
    
    except Exception as e:
        print(f"Error fetching match statistics: {e}")
        return None

# Integration in premier-league-predictions.py
def display_live_matches_tab():
    """Display live matches with auto-refresh"""
    
    st.subheader("⚽ Live Premier League Matches")
    
    # Auto-refresh toggle
    auto_refresh = st.checkbox("Auto-refresh every 30 seconds", value=True)
    
    # Refresh button
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("🔄 Refresh Now"):
            st.rerun()
    
    with col2:
        last_update = datetime.now().strftime('%I:%M:%S %p')
        st.caption(f"Last updated: {last_update}")
    
    # Fetch live matches
    live_df = fetch_live_matches()
    
    if len(live_df) == 0:
        st.info("No matches currently in progress")
        return
    
    # Display each match
    for _, match in live_df.iterrows():
        with st.expander(
            f"{match['HomeTeam']} {match['HomeScore']} - {match['AwayScore']} {match['AwayTeam']} "
            f"({match['Status']}) ⏱️ {match['Clock']}",
            expanded=True
        ):
            # Score display
            col1, col2, col3 = st.columns([2, 1, 2])
            
            with col1:
                st.markdown(f"### {match['HomeTeam']}")
                st.metric("Score", match['HomeScore'])
            
            with col2:
                st.markdown("### vs")
                st.caption(f"{match['Status']}")
                st.caption(f"{match['Clock']}")
            
            with col3:
                st.markdown(f"### {match['AwayTeam']}")
                st.metric("Score", match['AwayScore'])
            
            # Load prediction if exists
            if 'model' in st.session_state:
                st.divider()
                st.subheader("Model Prediction vs Reality")
                
                # Get prediction (would need to match against upcoming_fixtures)
                # Display comparison with actual score
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Predicted Home Win", "45.2%")
                with col2:
                    st.metric("Predicted Draw", "28.5%")
                with col3:
                    st.metric("Predicted Away Win", "26.3%")
            
            # Fetch detailed statistics
            if match['MatchId']:
                stats = get_match_statistics(match['MatchId'])
                
                if stats:
                    st.divider()
                    st.subheader("Match Statistics")
                    
                    # Create comparison table
                    stat_names = list(stats['home'].keys())
                    stat_df = pd.DataFrame({
                        'Statistic': stat_names,
                        match['HomeTeam']: [stats['home'].get(s, 'N/A') for s in stat_names],
                        match['AwayTeam']: [stats['away'].get(s, 'N/A') for s in stat_names]
                    })
                    
                    st.dataframe(stat_df, hide_index=True, use_container_width=True)
    
    # Auto-refresh logic
    if auto_refresh:
        time.sleep(30)
        st.rerun()

# Add to tab structure
tab_live, tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "⚽ Live Matches", "Upcoming Matches", "Predictive Data", 
    "Upcoming Predictions", "Statistics", "Raw Data"
])

with tab_live:
    display_live_matches_tab()
```

---

## 2. Interactive Match Commentary Generator

**Priority:** Medium  
**Effort:** 1-2 hours  
**Expected Impact:** Better user engagement

Generate natural language match previews and predictions.

```python
# Create: match_commentary.py

def generate_match_preview(home_team, away_team, prediction_proba, team_stats, h2h_stats):
    """
    Generate comprehensive match commentary with AI-style insights
    
    Args:
        home_team: Home team name
        away_team: Away team name
        prediction_proba: [home_prob, draw_prob, away_prob]
        team_stats: DataFrame with team statistics
        h2h_stats: Head-to-head historical data
    
    Returns:
        Formatted match preview text
    """
    
    home_prob, draw_prob, away_prob = prediction_proba
    
    # Determine prediction
    max_prob = max(prediction_proba)
    if max_prob == home_prob:
        predicted_outcome = "home win"
        confidence = home_prob
        favorite = home_team
    elif max_prob == away_prob:
        predicted_outcome = "away win"
        confidence = away_prob
        favorite = away_team
    else:
        predicted_outcome = "draw"
        confidence = draw_prob
        favorite = None
    
    # Confidence level
    if confidence > 0.6:
        confidence_text = "highly confident"
    elif confidence > 0.45:
        confidence_text = "moderately confident"
    else:
        confidence_text = "uncertain"
    
    # Build commentary
    commentary = f"## Match Preview: {home_team} vs {away_team}\n\n"
    
    # Opening prediction
    if favorite:
        commentary += f"**Prediction:** {favorite} to win ({confidence:.1%} probability)\n\n"
        commentary += f"Our model is **{confidence_text}** in this prediction. "
    else:
        commentary += f"**Prediction:** Draw ({confidence:.1%} probability)\n\n"
        commentary += f"This match is expected to be tightly contested with a **{confidence_text}** prediction of a draw. "
    
    # Form analysis
    home_form = team_stats[team_stats['Team'] == home_team]['RecentForm'].values[0] if len(team_stats) > 0 else "N/A"
    away_form = team_stats[team_stats['Team'] == away_team]['RecentForm'].values[0] if len(team_stats) > 0 else "N/A"
    
    commentary += f"\n\n### Recent Form\n"
    commentary += f"- **{home_team}:** {home_form}\n"
    commentary += f"- **{away_team}:** {away_form}\n"
    
    # Key stats comparison
    if len(team_stats) > 0:
        home_stats = team_stats[team_stats['Team'] == home_team].iloc[0]
        away_stats = team_stats[team_stats['Team'] == away_team].iloc[0]
        
        commentary += f"\n### Key Statistics\n"
        commentary += f"| Metric | {home_team} | {away_team} |\n"
        commentary += f"|--------|-------------|-------------|\n"
        commentary += f"| Goals Per Game | {home_stats.get('GoalsPerGame', 0):.2f} | {away_stats.get('GoalsPerGame', 0):.2f} |\n"
        commentary += f"| Goals Conceded | {home_stats.get('ConcededPerGame', 0):.2f} | {away_stats.get('ConcededPerGame', 0):.2f} |\n"
        commentary += f"| Win Rate | {home_stats.get('WinRate', 0):.1%} | {away_stats.get('WinRate', 0):.1%} |\n"
    
    # H2H history
    if h2h_stats is not None and len(h2h_stats) > 0:
        home_wins = (h2h_stats['WinningTeam'] == home_team).sum()
        away_wins = (h2h_stats['WinningTeam'] == away_team).sum()
        draws = (h2h_stats['FullTimeResult'] == 'D').sum()
        
        commentary += f"\n### Head-to-Head (Last {len(h2h_stats)} meetings)\n"
        commentary += f"- {home_team} wins: {home_wins}\n"
        commentary += f"- {away_team} wins: {away_wins}\n"
        commentary += f"- Draws: {draws}\n"
        
        if home_wins > away_wins:
            commentary += f"\n{home_team} has dominated recent encounters.\n"
        elif away_wins > home_wins:
            commentary += f"\n{away_team} has the better recent record in this fixture.\n"
        else:
            commentary += f"\nHistorically, this fixture has been evenly matched.\n"
    
    # Betting insights
    commentary += f"\n### Betting Insights\n"
    if home_prob > 0.5:
        commentary += f"Strong value in backing {home_team} at current odds.\n"
    elif away_prob > 0.5:
        commentary += f"Away win looks like good value for {away_team}.\n"
    elif draw_prob > 0.35:
        commentary += f"Draw represents potential value in this tightly matched fixture.\n"
    
    return commentary

# Integration in upcoming predictions tab
for _, match in upcoming_df.iterrows():
    with st.expander(f"📝 {match['HomeTeam']} vs {match['AwayTeam']} - {match['Date']}"):
        
        # Get prediction probabilities
        proba = [match['HomeWin_Prob'], match['Draw_Prob'], match['AwayWin_Prob']]
        
        # Generate commentary
        commentary = generate_match_preview(
            match['HomeTeam'],
            match['AwayTeam'],
            proba,
            team_stats_df,
            get_h2h_stats(match['HomeTeam'], match['AwayTeam'])
        )
        
        st.markdown(commentary)
```

---

## 3. Betting Odds Comparison Tool

**Priority:** High  
**Effort:** 2-3 hours  
**Expected Impact:** Help users find value bets

Compare model predictions with bookmaker odds to identify value opportunities.

```python
# Create: betting_comparison.py

import requests
import pandas as pd

def fetch_current_odds(match_date=None):
    """
    Fetch current betting odds from multiple bookmakers
    
    Data sources:
    - The Odds API (requires free API key, 500 requests/month)
    - Oddsportal scraping (no API key)
    """
    
    # Option 1: The Odds API
    API_KEY = 'your_api_key'  # Get from https://the-odds-api.com/
    
    url = 'https://api.the-odds-api.com/v4/sports/soccer_epl/odds/'
    params = {
        'apiKey': API_KEY,
        'regions': 'uk',
        'markets': 'h2h',  # Head to head (match result)
        'oddsFormat': 'decimal'
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            
            odds_data = []
            
            for game in data:
                home_team = game['home_team']
                away_team = game['away_team']
                
                for bookmaker in game['bookmakers']:
                    market = bookmaker['markets'][0]  # h2h market
                    
                    home_odds = next(o['price'] for o in market['outcomes'] if o['name'] == home_team)
                    away_odds = next(o['price'] for o in market['outcomes'] if o['name'] == away_team)
                    draw_odds = next((o['price'] for o in market['outcomes'] if o['name'] == 'Draw'), None)
                    
                    odds_data.append({
                        'HomeTeam': home_team,
                        'AwayTeam': away_team,
                        'Bookmaker': bookmaker['title'],
                        'HomeOdds': home_odds,
                        'DrawOdds': draw_odds,
                        'AwayOdds': away_odds,
                        'LastUpdate': bookmaker['last_update']
                    })
            
            return pd.DataFrame(odds_data)
        
    except Exception as e:
        print(f"Error fetching odds: {e}")
    
    return pd.DataFrame()

def calculate_value_bets(predictions_df, odds_df):
    """
    Calculate expected value for bets based on model predictions vs bookmaker odds
    
    EV = (Probability * Decimal_Odds) - 1
    Positive EV = value bet
    """
    
    value_bets = []
    
    for _, pred in predictions_df.iterrows():
        # Find matching odds
        match_odds = odds_df[
            (odds_df['HomeTeam'] == pred['HomeTeam']) &
            (odds_df['AwayTeam'] == pred['AwayTeam'])
        ]
        
        if len(match_odds) == 0:
            continue
        
        # Get best odds across bookmakers
        best_home_odds = match_odds['HomeOdds'].max()
        best_draw_odds = match_odds['DrawOdds'].max()
        best_away_odds = match_odds['AwayOdds'].max()
        
        # Calculate EV for each outcome
        home_ev = (pred['HomeWin_Prob'] * best_home_odds) - 1
        draw_ev = (pred['Draw_Prob'] * best_draw_odds) - 1
        away_ev = (pred['AwayWin_Prob'] * best_away_odds) - 1
        
        # Find value bets (EV > 0)
        if home_ev > 0.05:  # At least 5% edge
            value_bets.append({
                'Match': f"{pred['HomeTeam']} vs {pred['AwayTeam']}",
                'Bet': f"{pred['HomeTeam']} to win",
                'Model_Prob': pred['HomeWin_Prob'],
                'Best_Odds': best_home_odds,
                'Expected_Value': home_ev,
                'Recommended_Stake': calculate_kelly_stake(pred['HomeWin_Prob'], best_home_odds)
            })
        
        if draw_ev > 0.05:
            value_bets.append({
                'Match': f"{pred['HomeTeam']} vs {pred['AwayTeam']}",
                'Bet': "Draw",
                'Model_Prob': pred['Draw_Prob'],
                'Best_Odds': best_draw_odds,
                'Expected_Value': draw_ev,
                'Recommended_Stake': calculate_kelly_stake(pred['Draw_Prob'], best_draw_odds)
            })
        
        if away_ev > 0.05:
            value_bets.append({
                'Match': f"{pred['HomeTeam']} vs {pred['AwayTeam']}",
                'Bet': f"{pred['AwayTeam']} to win",
                'Model_Prob': pred['AwayWin_Prob'],
                'Best_Odds': best_away_odds,
                'Expected_Value': away_ev,
                'Recommended_Stake': calculate_kelly_stake(pred['AwayWin_Prob'], best_away_odds)
            })
    
    return pd.DataFrame(value_bets)

def calculate_kelly_stake(probability, decimal_odds):
    """
    Calculate optimal stake using Kelly Criterion
    
    Kelly % = (decimal_odds * probability - 1) / (decimal_odds - 1)
    """
    
    if decimal_odds <= 1:
        return 0
    
    kelly = (decimal_odds * probability - 1) / (decimal_odds - 1)
    
    # Use fractional Kelly (25%) for safety
    fractional_kelly = kelly * 0.25
    
    # Cap at 5% of bankroll
    return min(fractional_kelly, 0.05)

# UI Integration
st.subheader("💰 Value Betting Analysis")

if st.button("Find Value Bets"):
    with st.spinner("Fetching current odds and comparing with model..."):
        
        current_odds = fetch_current_odds()
        
        if len(current_odds) > 0:
            value_bets = calculate_value_bets(upcoming_predictions_df, current_odds)
            
            if len(value_bets) > 0:
                st.success(f"Found {len(value_bets)} value betting opportunities!")
                
                # Style the dataframe
                styled_df = value_bets.style.format({
                    'Model_Prob': '{:.1%}',
                    'Best_Odds': '{:.2f}',
                    'Expected_Value': '{:+.2%}',
                    'Recommended_Stake': '{:.1%}'
                }).background_gradient(subset=['Expected_Value'], cmap='RdYlGn')
                
                st.dataframe(styled_df, hide_index=True, use_container_width=True)
                
                st.warning("⚠️ Betting involves risk. Only bet what you can afford to lose.")
            else:
                st.info("No significant value bets found at current odds.")
        else:
            st.error("Could not fetch current odds. Please try again later.")

# Add to requirements.txt:
# requests>=2.31.0
```

---

## 4. Export Predictions to PDF Report

**Priority:** Medium  
**Effort:** 2 hours  
**Expected Impact:** Professional presentation of predictions

Generate downloadable PDF reports with match predictions and analysis.

```python
# Enhance generate_pdf_report.py

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from datetime import datetime
import io

def generate_weekly_predictions_report(predictions_df, team_stats_df, model_metrics):
    """
    Generate comprehensive weekly predictions PDF report
    
    Args:
        predictions_df: DataFrame with upcoming match predictions
        team_stats_df: DataFrame with team statistics
        model_metrics: Dictionary with model performance metrics
    
    Returns:
        BytesIO object with PDF content
    """
    
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, 
                           rightMargin=72, leftMargin=72,
                           topMargin=72, bottomMargin=18)
    
    # Container for report elements
    elements = []
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#1f4788'),
        spaceAfter=30,
        alignment=1  # Center
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=colors.HexColor('#2c5aa0'),
        spaceAfter=12
    )
    
    # Title
    title = Paragraph("Premier League Weekly Predictions", title_style)
    elements.append(title)
    
    # Report date
    date_text = Paragraph(
        f"Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}",
        styles['Normal']
    )
    elements.append(date_text)
    elements.append(Spacer(1, 12))
    
    # Model Performance Summary
    elements.append(Paragraph("Model Performance Summary", heading_style))
    
    perf_data = [
        ['Metric', 'Value'],
        ['Model Type', 'Ensemble (XGBoost + RF + GB)'],
        ['Accuracy', f"{model_metrics.get('accuracy', 0):.1%}"],
        ['Mean Absolute Error', f"{model_metrics.get('mae', 0):.3f}"],
        ['Predictions', str(model_metrics.get('num_predictions', 0))]
    ]
    
    perf_table = Table(perf_data, colWidths=[3*inch, 3*inch])
    perf_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    elements.append(perf_table)
    elements.append(Spacer(1, 20))
    
    # Match Predictions
    elements.append(Paragraph("Upcoming Match Predictions", heading_style))
    elements.append(Spacer(1, 12))
    
    for _, match in predictions_df.iterrows():
        # Match header
        match_title = Paragraph(
            f"<b>{match['HomeTeam']} vs {match['AwayTeam']}</b>",
            styles['Heading3']
        )
        elements.append(match_title)
        
        match_info = Paragraph(
            f"{match['Date']} at {match['Time']} ET",
            styles['Normal']
        )
        elements.append(match_info)
        elements.append(Spacer(1, 6))
        
        # Prediction table
        pred_data = [
            ['Outcome', 'Probability', 'Confidence'],
            ['Home Win', f"{match['HomeWin_Prob']:.1%}", 
             '🟢 High' if match['HomeWin_Prob'] > 0.5 else '🟡 Medium' if match['HomeWin_Prob'] > 0.35 else '🔴 Low'],
            ['Draw', f"{match['Draw_Prob']:.1%}",
             '🟢 High' if match['Draw_Prob'] > 0.4 else '🟡 Medium' if match['Draw_Prob'] > 0.25 else '🔴 Low'],
            ['Away Win', f"{match['AwayWin_Prob']:.1%}",
             '🟢 High' if match['AwayWin_Prob'] > 0.5 else '🟡 Medium' if match['AwayWin_Prob'] > 0.35 else '🔴 Low']
        ]
        
        pred_table = Table(pred_data, colWidths=[2*inch, 2*inch, 2*inch])
        pred_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c5aa0')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey])
        ]))
        
        elements.append(pred_table)
        elements.append(Spacer(1, 20))
    
    # Build PDF
    doc.build(elements)
    buffer.seek(0)
    
    return buffer

# Integration in Streamlit
st.subheader("📄 Export Predictions Report")

if st.button("Generate PDF Report"):
    with st.spinner("Generating PDF report..."):
        
        model_metrics = {
            'accuracy': acc,
            'mae': mae,
            'num_predictions': len(upcoming_df)
        }
        
        pdf_buffer = generate_weekly_predictions_report(
            upcoming_df, team_stats_df, model_metrics
        )
        
        st.download_button(
            label="📥 Download PDF Report",
            data=pdf_buffer,
            file_name=f"pl_predictions_{datetime.now().strftime('%Y%m%d')}.pdf",
            mime="application/pdf"
        )

# Add to requirements.txt:
# reportlab>=4.0.0
```

---

## 5. Email Alerts for High-Confidence Predictions

**Priority:** Low  
**Effort:** 2 hours  
**Expected Impact:** Automated notifications

Send email notifications for predictions with >70% confidence.

```python
# Create: email_alerts.py

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import os

def send_prediction_alert(recipient_email, high_confidence_predictions):
    """
    Send email alert for high-confidence predictions
    
    Args:
        recipient_email: Recipient's email address
        high_confidence_predictions: DataFrame with predictions >= 70% confidence
    """
    
    # Email configuration
    sender_email = os.getenv('SENDER_EMAIL', 'predictions@premierleague.com')
    sender_password = os.getenv('EMAIL_PASSWORD')
    
    # Create message
    message = MIMEMultipart("alternative")
    message["Subject"] = f"🔔 {len(high_confidence_predictions)} High-Confidence Predictions Alert"
    message["From"] = sender_email
    message["To"] = recipient_email
    
    # HTML content
    html_body = f"""
    <html>
      <body>
        <h2>High-Confidence Premier League Predictions</h2>
        <p>The following matches have predictions with >70% confidence:</p>
        <table border="1" cellpadding="5">
          <tr style="background-color: #2c5aa0; color: white;">
            <th>Match</th>
            <th>Date</th>
            <th>Prediction</th>
            <th>Confidence</th>
          </tr>
    """
    
    for _, pred in high_confidence_predictions.iterrows():
        max_prob = max(pred['HomeWin_Prob'], pred['Draw_Prob'], pred['AwayWin_Prob'])
        
        if max_prob == pred['HomeWin_Prob']:
            outcome = f"{pred['HomeTeam']} to Win"
        elif max_prob == pred['AwayWin_Prob']:
            outcome = f"{pred['AwayTeam']} to Win"
        else:
            outcome = "Draw"
        
        html_body += f"""
          <tr>
            <td>{pred['HomeTeam']} vs {pred['AwayTeam']}</td>
            <td>{pred['Date']} at {pred['Time']}</td>
            <td><b>{outcome}</b></td>
            <td>{max_prob:.1%}</td>
          </tr>
        """
    
    html_body += """
        </table>
        <p><i>This is an automated alert from the Premier League Predictor.</i></p>
      </body>
    </html>
    """
    
    part = MIMEText(html_body, "html")
    message.attach(part)
    
    # Send email
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, recipient_email, message.as_string())
        
        print(f"Alert email sent to {recipient_email}")
        return True
    
    except Exception as e:
        print(f"Failed to send email: {e}")
        return False

# Integration in Streamlit
st.subheader("📧 Email Alerts")

with st.expander("Configure Email Alerts"):
    recipient_email = st.text_input("Your Email Address")
    
    confidence_threshold = st.slider(
        "Confidence Threshold for Alerts",
        min_value=0.5,
        max_value=0.9,
        value=0.7,
        step=0.05,
        format="%.0f%%"
    )
    
    if st.button("Send Test Alert"):
        if recipient_email:
            # Filter high-confidence predictions
            high_conf = upcoming_df[
                (upcoming_df['HomeWin_Prob'] >= confidence_threshold) |
                (upcoming_df['Draw_Prob'] >= confidence_threshold) |
                (upcoming_df['AwayWin_Prob'] >= confidence_threshold)
            ]
            
            if len(high_conf) > 0:
                if send_prediction_alert(recipient_email, high_conf):
                    st.success(f"✅ Alert sent to {recipient_email}!")
                else:
                    st.error("Failed to send email. Check configuration.")
            else:
                st.info(f"No predictions above {confidence_threshold:.0%} confidence threshold.")
        else:
            st.warning("Please enter your email address.")
```

---

## Implementation Priority

**Week 1:**
1. Live Match Tracker (High Impact, Real-time Engagement)
2. Betting Odds Comparison (High Impact, Valuable for Users)

**Week 2:**
3. Interactive Match Commentary (Medium Impact, Better UX)
4. Export to PDF (Medium Impact, Professional Reports)

**Week 3:**
5. Email Alerts (Low Impact, Nice-to-Have Feature)

---

## Success Metrics

- **Live Tracker:** 30+ users engaging during match days
- **Match Commentary:** Improved session duration by 25%
- **Odds Comparison:** Identify 5+ value bets per week
- **PDF Reports:** 50+ downloads per month
- **Email Alerts:** 100+ subscribers with 40% open rate
