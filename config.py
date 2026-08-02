"""Premier League consumer configuration."""

from dataclasses import replace

from pitch_oracle_core import ThemeConfig, get_league_config

LEAGUE_CONFIG = replace(
    get_league_config("epl"),
    theme=ThemeConfig(
        primary="#1554a6",
        primary_dark="#0d2f5f",
        sidebar="#f1f4f8",
        page="#ffffff",
        border="#d9e0e8",
        muted="#64748b",
    ),
)
