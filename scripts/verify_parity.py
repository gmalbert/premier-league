"""Strict EPL artifact and chronological model-quality gate."""

from __future__ import annotations

import math
from pathlib import Path
import pickle

from pitch_oracle_core import FeatureContract, __version__
from pitch_oracle_core.cache import validate_cache


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    validate_cache(ROOT, expected_league="epl")
    contract = FeatureContract.load(ROOT / "precomputed" / "preprocessed_data.pkl")
    with (ROOT / "models" / "ensemble_model.pkl").open("rb") as stream:
        ensemble = pickle.load(stream)
    width = getattr(ensemble, "n_features_in_", None)
    if width is not None and width != len(contract.feature_names):
        raise SystemExit(
            f"Ensemble width {width} does not match contract width {len(contract.feature_names)}"
        )

    with (ROOT / "models" / "model_performance.pkl").open("rb") as stream:
        performance = pickle.load(stream)
    required = {"xgb_baseline", "ensemble", "optimized_xgb", "poisson"}
    missing = required.difference(performance)
    if missing:
        raise SystemExit(f"Missing model metrics: {sorted(missing)}")
    for name in ("xgb_baseline", "ensemble", "optimized_xgb"):
        accuracy = float(performance[name]["accuracy"])
        log_loss = float(performance[name]["log_loss"])
        if not (0.0 <= accuracy <= 1.0 and math.isfinite(log_loss) and log_loss < 2.0):
            raise SystemExit(f"Implausible chronological metrics for {name}: {performance[name]}")
    poisson_accuracy = float(performance["poisson"]["outcome_acc"])
    if not 0.0 <= poisson_accuracy <= 1.0:
        raise SystemExit(f"Invalid Poisson outcome accuracy: {poisson_accuracy}")

    print(f"EPL artifacts verified with pitch-oracle-core {__version__}")
    print(f"Feature contract width: {len(contract.feature_names)}")
    for name in sorted(required):
        print(f"  {name}: {performance[name]}")


if __name__ == "__main__":
    main()
