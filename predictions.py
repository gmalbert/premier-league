"""Thin Streamlit entry page for the Premier League deployment."""

from pitch_oracle_core.app_factory import run
from config import LEAGUE_CONFIG

run(LEAGUE_CONFIG)

