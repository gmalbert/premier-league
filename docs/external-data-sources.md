# External Data Sources for Soccer App Enhancement

A review of four free/open data sources and how they can be integrated into this Premier League predictions app.

---

## 1. ClubElo API — http://clubelo.com/API

### What it is
A free, no-auth REST API serving Elo ratings for European club football going back to 1939. Elo is a relative strength rating system that updates after every match based on result, margin, and opponent strength.

### Available endpoints

| Endpoint | Returns |
|---|---|
| `api.clubelo.com/YYYY-MM-DD` | Full ranking snapshot for that date |
| `api.clubelo.com/CLUBNAME` | Full Elo history for a single club |
| `api.clubelo.com/Fixtures` | Upcoming matches with pre-calculated win/draw/loss probabilities |

The `/Fixtures` endpoint returns probabilities for every goal difference from -5 to +5, letting you derive traditional 1X2 odds by summing.

### Relevance to this app
- **Directly applicable.** Elo rating is a compact, market-independent measure of team strength. It can replace or supplement rolling form windows as a feature in the XGBoost model.
- Fetch home/away Elo at match date using `api.clubelo.com/YYYY-MM-DD` and join to the historical data by team name and date.
- The `/Fixtures` endpoint provides baseline win probabilities that can serve as a calibration benchmark or as a feature alongside betting odds.
- No scraping required — clean CSV/JSON responses, easy to integrate into `prepare_model_data.py`.
- **Limitation:** Club name spelling must match ClubElo's exact slug convention. Build a mapping dict; wrong slugs return empty CSVs silently.

### Discovering correct slugs
The safest way to find a team's exact ClubElo slug is to query a recent ranking snapshot and check the `Club` column:
```python
import requests, pandas as pd, io
df = pd.read_csv(io.StringIO(requests.get("http://api.clubelo.com/2026-05-02").text))
print(df[['Club']].to_string())   # shows all slugs currently in the system
```

**Known non-obvious slug corrections (EPL teams):**

| Common name | Wrong assumption | Correct slug |
|---|---|---|
| Wolves (Wolverhampton) | `Wolverhampton` | `Wolves` |
| Nottingham Forest | `NottmForest` | `Forest` |
| Sheffield Wednesday | `SheffieldWednesday` | `SheffieldWeds` |

Slug format rules: no spaces, no apostrophes, CamelCase. When in doubt, check the ranking endpoint.

### Integration effort: Low
```python
import requests, pandas as pd, io

resp = requests.get("http://api.clubelo.com/ManCity", timeout=10)
elo_df = pd.read_csv(io.StringIO(resp.text))
# columns: Rank, Club, Country, Level, Elo, From, To
```

For upcoming fixture probabilities, the `/Fixtures` response contains one row per goal difference (-5 to +5). Sum rows where `ExpHomeGoals - ExpAwayGoals > 0` for home win probability, `== 0` for draw, `< 0` for away win.

---

## 2. Understat — https://understat.com/

### What it is
Understat provides expected goals (xG) and expected assists (xA) data for the top six European leagues including the EPL. Their neural-network model was trained on 100,000+ shots using over 10 parameters per shot (angle, distance, body part, assist type, game state, etc.). Data is available back to the 2014/15 season.

### Data available (EPL)
- **Team-level:** xG, xGA, xPts per season and per match
- **Player-level:** xG, xA, shots, key passes, minutes played per season
- **Match-level:** xG timeline (when each xG event happened in the match)

> ⚠️ **Architecture change (confirmed May 2026):** Understat **no longer embeds data in `<script>` tags**. The old `BeautifulSoup` + `JSON.parse` approach returns no data. They now load data via a JSON REST endpoint that requires an active session cookie.

### Correct API approach (as of 2026)

Understat's frontend JavaScript (`js/league.min.js`) calls:
```
https://understat.com/getLeagueData/{league}/{year}
```
where `league` = `EPL` and `year` = the calendar year the season starts (e.g. `2024` for 2024/25).

**This endpoint returns 404 without a valid session cookie.** You must first perform a GET request to the league HTML page so the server sets a session cookie, then reuse that session for the API call.

```python
import requests, pandas as pd

def make_session(year):
    """Visit the league page to obtain a session cookie."""
    session = requests.Session()
    session.headers.update({'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
    session.get(f'https://understat.com/league/EPL/{year}', timeout=20)
    return session

def fetch_season_xg(year):
    session = make_session(year)
    resp = session.get(
        f'https://understat.com/getLeagueData/EPL/{year}',
        headers={
            'X-Requested-With': 'XMLHttpRequest',
            'Referer': f'https://understat.com/league/EPL/{year}',
        },
        timeout=20,
    )
    data = resp.json()   # top-level keys: 'teams', 'players', 'dates'
    return data['dates'] # list of match objects
```

**Response structure for each item in `dates`:**
```json
{
  "id": "26602",
  "isResult": true,
  "h": {"id": "89", "title": "Manchester United", ...},
  "a": {"id": "228", "title": "Fulham", ...},
  "goals": {"h": "1", "a": "0"},
  "xG":    {"h": "2.04268", "a": "0.418711"},
  "datetime": "2024-08-16 19:00:00",
  "forecast": {"w": "0.8069", "d": "0.1489", "l": "0.0442"}
}
```
- Filter `isResult == true` to exclude upcoming fixtures.
- `xG.h` and `xG.a` are string floats — cast with `float()`.
- `forecast` gives Understat's own win/draw/loss probabilities (useful as a feature or calibration baseline).
- Use `time.sleep(1.5)` between season requests to avoid rate-limiting.
- A fresh session per season is safer than reusing one session across all 12 years.

### Relevance to this app
- **High value for modelling.** xG is widely considered a better predictor of future performance than actual goals because it strips out finishing luck. Using xG-for and xG-against as features alongside actual goals-for/against is a meaningful model improvement.
- xG-based rolling averages (last 5 matches) complement the existing `groupby().rolling(window=5)` pattern already used for goals.
- `forecast` probabilities are a market-independent signal that can sit alongside betting odds features.
- **Limitation:** Session-based; no published SLA or official API contract. May break again if Understat redesigns their backend. Monitor HTTP status codes and response structure.

### Integration effort: Low–Medium
No external packages needed beyond `requests` and `pandas`. The old `beautifulsoup4` / `understat` pip package approach no longer works.

---

## 3. Wyscout Soccer Match Event Dataset — https://github.com/koenvo/wyscout-soccer-match-event-dataset

### What it is
A processed version of the Pappalardo et al. (2019) public Wyscout dataset. It contains spatio-temporal match event data (every pass, shot, foul, dribble, clearance, etc.) for 380 EPL matches from the **2017/18 season**, plus four other major leagues, the 2018 World Cup, and Euro 2016.

### Data structure
- **Events:** every on-ball action with timestamp, player, team, pitch coordinates (x/y), outcome, event subtypes
- **Matches, Players, Teams:** reference JSON files
- Loaded via the `kloppy` library (`pip install kloppy`)
- License: CC BY 4.0 (free to use with attribution)

### Relevance to this app
- **Research / offline use only** — this is a static 2017/18 snapshot, not a live feed. It cannot be used for current-season predictions.
- Useful for **feature engineering research**: build and validate event-derived features (e.g. shot quality from position, pressure index, pass network centrality) before applying them to live data from a paid provider.
- Could be used to train a separate shot quality / xG model as an alternative to Understat's values.
- `kloppy` can also read Opta, StatsBomb, and Wyscout formats, making it a useful abstraction layer if you ever upgrade to a paid data provider.
- **Limitation:** 2017/18 data only. No ongoing updates. EPL team/player names may not match current naming conventions.

### Integration effort: Low (for research) / Not applicable (for production)
```python
from kloppy import datasets
dataset = datasets.load("wyscout")          # loads default EPL match
dataset = datasets.load("wyscout", match_id=2499843)
```

---

## 4. StatsBomb Open Data — https://github.com/statsbomb/open-data

### What it is
StatsBomb's free open data release containing their proprietary event data format — widely regarded as the most detailed publicly available football event data. Includes 360-degree freeze-frame data for selected matches (player positions at moment of each event).

### Data available
- **Events:** every on-ball and off-ball action with sub-event detail (e.g. pass end location, shot technique, carry distance, pressure applied)
- **Lineups:** starting XI and substitutions per match
- **StatsBomb 360:** freeze-frame player coordinates for selected matches
- **Competitions included:** La Liga (multiple Messi seasons), Women's tournaments, NWSL, FA Women's Super League, Champions League (limited), and some other competitions — **Premier League coverage is very limited in the open dataset**
- License: free for research/educational use; StatsBomb attribution (logo/name) required on any published work

### Relevance to this app
- **Limited direct applicability** due to minimal Premier League data in the free tier. The open data is primarily La Liga and women's football.
- **High value as a training/research corpus:** The rich event schema makes it excellent for building and validating advanced features (pressing intensity, pass networks, shot assist chains) that could later be applied to a paid data feed.
- StatsBomb's `statsbombpy` library (`pip install statsbombpy`) provides a clean Python interface for loading competitions, matches, and events.
- **For PL production use, a StatsBomb commercial license would be required.**

### Integration effort: Low (for research)
```python
from statsbombpy import sb

competitions = sb.competitions()
matches = sb.matches(competition_id=2, season_id=27)   # La Liga example
events  = sb.events(match_id=3773369)
```

---

## Summary & Recommendations

| Source | PL Coverage | Real-time | Cost | Integration Effort | Recommended Use |
|---|---|---|---|---|---|
| **ClubElo API** | Full EPL history + upcoming | Yes | Free | Low | Add Elo rating as a model feature; use `/Fixtures` probabilities as calibration baseline |
| **Understat** | EPL 2014/15 → now | Current season | Free (session API) | Low–Medium | xG rolling averages as features; replace/supplement raw goal counts |
| **Wyscout Dataset** | EPL 2017/18 only | No (static) | Free (CC BY 4.0) | Low | Offline feature research; build custom xG model |
| **StatsBomb Open Data** | Minimal PL | No (static) | Free (attribution) | Low | Offline research; schema reference for advanced event features |

### Immediate wins for this app
1. **ClubElo** — integrate first. Fetch team Elo at each match date and add `HomeElo`, `AwayElo`, and `EloDiff` as features in `prepare_model_data.py`. Use `pd.merge_asof` with `direction='backward'` to align Elo at each match date.
2. **Understat xG** — use the session-based JSON API (see above). Add xG-based rolling averages (`HomeXG_Understat_L5`, `AwayXGA_Understat_L5`) via `groupby().rolling(5).mean().shift(1)`. These features are well-correlated with future performance and complement the existing betting-odds features. **Do not use BeautifulSoup or the `understat` pip package — both rely on the old `<script>` tag approach that no longer works.**
3. **Wyscout / StatsBomb** — use offline to prototype event-level features (shot quality, press rate) before investing in a live data feed.

---

## Implementation Notes & Lessons Learned

### ClubElo
- The API base URL is `http://` (not HTTPS) — requests to `https://api.clubelo.com` may fail.
- The `/Fixtures` response has one row per goal-difference bucket, not one row per match. Aggregate by summing probabilities: home win = sum of rows where home margin > 0.
- Use `pd.merge_asof` (sorted by date) to join Elo history onto match data: `pd.merge_asof(matches, elo_df, left_on='Date', right_on='From', direction='backward')`.
- Rate-limit to one request per team with `time.sleep(0.5)` — the API is small and shared.
- Always verify slugs by querying the daily ranking (`api.clubelo.com/YYYY-MM-DD`) and reading the `Club` column — do not guess slug format.

### Understat
- **The `<script>` tag / `datesData` approach is dead.** As of 2026, all four `<script>` blocks on the league page contain inline JS only — no embedded JSON data.
- The real data endpoint is `https://understat.com/getLeagueData/{league}/{year}` (found by inspecting `js/league.min.js`).
- The `X-Requested-With: XMLHttpRequest` header is required alongside the session cookie; without it the server may return HTML instead of JSON.
- The `understat` pip package (`pip install understat`) relies on `datesData` in `<script>` tags and is also broken.
- `resp.json()` works directly — no custom JSON decoding needed (unlike the old unicode-escape dance).
- `forecast` in each match object gives Understat's own probabilistic predictions — this is a free bonus feature.
- Seasons with `year >= 2025` may return partial data (only completed matches have `isResult: true`); this is correct behaviour.
