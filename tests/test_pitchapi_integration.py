from pitch_oracle_core import get_league_config
from pitch_oracle_core.features.families import FeatureFamilyConfig


def test_pitchapi_registered_with_no_implicit_promotion(tmp_path):
    configuration = get_league_config('epl')
    assert configuration.sources.pitchapi and configuration.sources.pitchapi_league_id
    assert FeatureFamilyConfig.load(tmp_path / "absent.json", league_key='epl').enabled_families == ()
