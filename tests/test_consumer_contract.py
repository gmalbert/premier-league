from config import LEAGUE_CONFIG
from pitch_oracle_core import __version__


def test_consumer_is_epl_configured():
    assert LEAGUE_CONFIG.key == "epl"
    assert LEAGUE_CONFIG.football_data_div == "E0"
    assert LEAGUE_CONFIG.espn_slug == "eng.1"
    assert LEAGUE_CONFIG.sources.api_football_league_id == 39
    assert __version__ == "1.3.0"
