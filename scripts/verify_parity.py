"""Verify the consumer's saved EPL metrics through the shared core facade."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[1]))

from config import LEAGUE_CONFIG
from pitch_oracle_core.training import evaluate


def main() -> None:
    metrics = evaluate(LEAGUE_CONFIG, root=Path(__file__).parents[1])
    required = {"league_avg", "home_mae", "away_mae", "outcome_acc"}
    missing = required - metrics.keys()
    if missing:
        raise SystemExit(f"Missing parity metrics: {sorted(missing)}")
    print("EPL shared-core parity metrics:")
    for key in sorted(required):
        print(f"  {key}: {metrics[key]}")


if __name__ == "__main__":
    main()
