"""Generate the strict EPL upcoming-prediction cache."""

from __future__ import annotations

from pathlib import Path
import pickle
import json
import sys

import numpy as np
import pandas as pd

from pitch_oracle_core import (
    FeatureContract,
    build_prediction_frame,
    production_probabilities,
    build_upcoming_feature_matrix,
)


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from config import LEAGUE_CONFIG

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data_files"
MODELS = ROOT / "models"
PRECOMPUTED = ROOT / "precomputed"


def production_candidate() -> str:
    report = json.loads((ROOT / "precomputed/model-audit/model_ablation.json").read_text(encoding="utf-8"))
    gate = report.get("release_gate", {})
    candidate = gate.get("production_candidate")
    if not gate.get("passed") or candidate not in ("no_odds", "poisson"):
        raise RuntimeError("Model audit did not validate a production candidate")
    return candidate


def generate() -> Path:
    historical_path = DATA / "combined_historical_data_with_calculations_new.csv"
    fixtures_path = DATA / "upcoming_fixtures.csv"
    model_path = MODELS / "ensemble_model.pkl"
    contract_path = PRECOMPUTED / "preprocessed_data.pkl"
    missing = [
        str(path)
        for path in (historical_path, fixtures_path, model_path, contract_path)
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError("Missing prediction inputs: " + ", ".join(missing))

    historical = pd.read_csv(historical_path, sep="\t")
    upcoming = pd.read_csv(fixtures_path)
    contract = FeatureContract.load(contract_path)
    with model_path.open("rb") as stream:
        model = pickle.load(stream)
    classes = tuple(int(value) for value in getattr(model, "classes_", (0, 1, 2)))
    if classes != (0, 1, 2):
        raise RuntimeError(f"Ensemble class order is {classes}; expected (0, 1, 2)")

    matrix = build_upcoming_feature_matrix(historical, upcoming, contract)
    probabilities = production_probabilities(historical, upcoming, contract, production_candidate=production_candidate(), models_dir=MODELS, league_key=LEAGUE_CONFIG.key, data_dir=DATA)
    result = build_prediction_frame(upcoming, probabilities)
    output = DATA / "upcoming_predictions.csv"
    result.to_csv(output, index=False)
    return output


if __name__ == "__main__":
    print(f"Wrote {generate()}")
