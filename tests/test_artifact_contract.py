from pathlib import Path
import pickle

import pandas as pd

from pitch_oracle_core import FeatureContract
from pitch_oracle_core.cache import validate_cache


ROOT = Path(__file__).resolve().parents[1]


def test_runtime_artifacts_match_core_and_league_contract():
    assert validate_cache(ROOT, expected_league="epl") == ()
    contract = FeatureContract.load(ROOT / "precomputed" / "preprocessed_data.pkl")
    with (ROOT / "models" / "ensemble_model.pkl").open("rb") as stream:
        model = pickle.load(stream)
    assert model.n_features_in_ == len(contract.feature_names)


def test_prediction_cache_has_assessment_fields_and_normalized_probabilities():
    predictions = pd.read_csv(ROOT / "data_files" / "upcoming_predictions.csv")
    required = {
        "HomeWin_Prob", "Draw_Prob", "AwayWin_Prob", "PredictedResult",
        "Risk_Score", "Confidence_Score", "Risk_Category", "Recommendation",
        "PredictionGeneratedAt",
    }
    assert required.issubset(predictions.columns)
    probabilities = predictions[["HomeWin_Prob", "Draw_Prob", "AwayWin_Prob"]]
    assert probabilities.notna().all().all()
    assert probabilities.ge(0).all().all()
    assert probabilities.sum(axis=1).between(0.999999, 1.000001).all()
