from config import LEAGUE_CONFIG


def test_consumer_is_epl_configured():
    assert LEAGUE_CONFIG.key == "epl"
    assert LEAGUE_CONFIG.football_data_div == "E0"
    assert LEAGUE_CONFIG.espn_slug == "eng.1"
    assert LEAGUE_CONFIG.sources.api_football_league_id == 39

