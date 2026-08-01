"""Generate the upcoming-match prediction cache used by the Streamlit app."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data_files"
MODELS = ROOT / "models"
PRECOMPUTED = ROOT / "precomputed"

DROP_COLS = {
    "FullTimeResult", "FullTimeHomeGoals", "FullTimeAwayGoals",
    "HalfTimeResult", "HalfTimeHomeGoals", "HalfTimeAwayGoals",
    "HomeWin", "AwayWin", "Draw", "WinningTeam", "HomePoints", "AwayPoints",
    "HomeTeamCumulativePoints", "AwayTeamCumulativePoints", "MatchDate",
    "KickoffTime", "Season", "Round", "Venue", "Referee", "HomeTeam",
    "AwayTeam", "Division", "target",
}


def _training_shape(df: pd.DataFrame) -> tuple[list[str], list[str], int]:
    usable = df.drop(columns=[c for c in DROP_COLS if c in df], errors="ignore")
    numeric = list(usable.select_dtypes(include=[np.number]).columns)
    categorical = list(usable.select_dtypes(include=["object"]).columns)
    return numeric, categorical, len(numeric) + len(categorical)


def generate() -> Path:
    historical = DATA / "combined_historical_data_with_calculations_new.csv"
    fixtures = DATA / "upcoming_fixtures.csv"
    model_path = MODELS / "ensemble_model.pkl"
    if not historical.exists() or not fixtures.exists() or not model_path.exists():
        raise FileNotFoundError("Historical data, upcoming fixtures, or ensemble model is missing")

    df = pd.read_csv(historical, sep="\t")
    upcoming = pd.read_csv(fixtures)
    numeric_cols, categorical_cols, feature_count = _training_shape(df)

    home_avgs: dict[str, dict[str, float]] = {}
    for team in df["HomeTeam"].dropna().unique():
        rows = df[df["HomeTeam"] == team].tail(10)
        home_avgs[team] = {
            col: float(rows[col].mean())
            for col in numeric_cols
            if col in rows and pd.notna(rows[col].mean())
        }
    away_avgs: dict[str, dict[str, float]] = {}
    for team in df["AwayTeam"].dropna().unique():
        rows = df[df["AwayTeam"] == team].tail(10)
        away_avgs[team] = {
            col: float(rows[col].mean())
            for col in numeric_cols
            if col in rows and pd.notna(rows[col].mean())
        }
    global_means = {col: float(df[col].mean()) for col in numeric_cols if col in df}

    rows = []
    for _, match in upcoming.iterrows():
        home, away = match["HomeTeam"], match["AwayTeam"]
        values = []
        for col in numeric_cols:
            if col.startswith("Home") and home in home_avgs and col in home_avgs[home]:
                values.append(home_avgs[home][col])
            elif col.startswith("Away") and away in away_avgs and col in away_avgs[away]:
                values.append(away_avgs[away][col])
            else:
                values.append(global_means.get(col, 0.0))
        values.extend([0.0] * len(categorical_cols))
        rows.append(values)

    features = np.asarray(rows, dtype=np.float32)
    if features.shape[1] < feature_count:
        features = np.hstack([features, np.zeros((len(features), feature_count - features.shape[1]))])
    elif features.shape[1] > feature_count:
        features = features[:, :feature_count]

    with model_path.open("rb") as stream:
        model = pickle.load(stream)
    probabilities = model.predict_proba(features)
    result = upcoming.copy()
    result["HomeWin_Prob"] = probabilities[:, 0]
    result["Draw_Prob"] = probabilities[:, 1]
    result["AwayWin_Prob"] = probabilities[:, 2]
    labels = np.asarray(["Home Win", "Draw", "Away Win"])
    result["PredictedResult"] = labels[np.argmax(probabilities, axis=1)]
    result["PredictionGeneratedAt"] = pd.Timestamp.utcnow().isoformat()

    output = DATA / "upcoming_predictions.csv"
    result.to_csv(output, index=False)
    return output


if __name__ == "__main__":
    print(f"Wrote {generate()}")
