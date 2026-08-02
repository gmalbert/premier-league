"""Generate the strict EPL upcoming-prediction cache."""

from __future__ import annotations

from pathlib import Path
import pickle

import numpy as np
import pandas as pd

from pitch_oracle_core import (
    FeatureContract,
    build_prediction_frame,
    build_upcoming_feature_matrix,
)


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data_files"
MODELS = ROOT / "models"
PRECOMPUTED = ROOT / "precomputed"


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
    probabilities = np.asarray(model.predict_proba(matrix), dtype=float)
    result = build_prediction_frame(upcoming, probabilities)
    output = DATA / "upcoming_predictions.csv"
    result.to_csv(output, index=False)
    return output


if __name__ == "__main__":
    print(f"Wrote {generate()}")
