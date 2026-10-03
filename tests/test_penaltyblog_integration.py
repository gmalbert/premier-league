import numpy as np


def test_shared_penaltyblog_grid_contract_is_available():
    from pitch_oracle_core.domain.probability_grid import ProbabilityGrid
    from pitch_oracle_core.markets.grid import market_row_from_domain

    mass = np.zeros((3, 3), dtype=float)
    mass[1, 0] = 0.3
    mass[1, 1] = 0.4
    mass[0, 1] = 0.3
    row = market_row_from_domain(ProbabilityGrid(mass, 0.0, 2, 2))
    assert row["home_win"] == 0.3
    assert row["draw"] == 0.4
    assert row["away_win"] == 0.3
