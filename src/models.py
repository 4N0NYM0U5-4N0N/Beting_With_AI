from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import RESULTS, TEST_SEASONS, TRAIN_SEASONS, VALIDATION_SEASONS


@dataclass
class SplitFrames:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def chronological_splits(frame: pd.DataFrame) -> SplitFrames:
    seasons = set(frame["season"].astype(str))
    expected = set(TRAIN_SEASONS + VALIDATION_SEASONS + TEST_SEASONS)
    unknown = sorted(seasons.difference(expected))
    if unknown:
        raise ValueError(f"Unexpected seasons in chronological split: {unknown}")
    output = SplitFrames(
        train=frame[frame["season"].isin(TRAIN_SEASONS)].copy(),
        validation=frame[frame["season"].isin(VALIDATION_SEASONS)].copy(),
        test=frame[frame["season"].isin(TEST_SEASONS)].copy(),
    )
    if min(output.validation["match_date"]) <= max(output.train["match_date"]):
        raise ValueError("Validation dates overlap or precede training dates.")
    if min(output.test["match_date"]) <= max(output.validation["match_date"]):
        raise ValueError("Test dates overlap or precede validation dates.")
    return output


def split_summary(splits: SplitFrames) -> list[dict[str, Any]]:
    return [
        {
            "split": name,
            "rows": len(frame),
            "seasons": ", ".join(sorted(frame["season"].unique())),
            "min_date": str(frame["match_date"].min().date()),
            "max_date": str(frame["match_date"].max().date()),
        }
        for name, frame in (("TRAINING", splits.train), ("VALIDATION", splits.validation), ("OUT-OF-SAMPLE", splits.test))
    ]


def _probability_frame(probabilities: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(probabilities, columns=["prob_H", "prob_D", "prob_A"])


def _aligned_probabilities(model: Any, X: pd.DataFrame) -> np.ndarray:
    raw = model.predict_proba(X)
    classes = list(model.classes_)
    output = np.zeros((len(X), len(RESULTS)))
    for index, label in enumerate(RESULTS):
        if label in classes:
            output[:, index] = raw[:, classes.index(label)]
    return output


def _bookmaker_probability(frame: pd.DataFrame) -> np.ndarray:
    return frame[["bookmaker_prob_home", "bookmaker_prob_draw", "bookmaker_prob_away"]].to_numpy(dtype=float)


def _historical_frequency(reference: pd.DataFrame, target_length: int) -> np.ndarray:
    counts = reference["full_time_result"].value_counts()
    values = np.array([counts.get(label, 0) for label in RESULTS], dtype=float)
    return np.tile(values / values.sum(), (target_length, 1))


def _make_pipeline(kind: str) -> Pipeline:
    if kind == "logistic_regression":
        estimator = LogisticRegression(max_iter=600, random_state=42)
        return Pipeline([("imputer", SimpleImputer(strategy="median", add_indicator=True)), ("scale", StandardScaler()), ("model", estimator)])
    if kind == "random_forest":
        estimator = RandomForestClassifier(
            n_estimators=220,
            max_depth=7,
            min_samples_leaf=8,
            random_state=42,
            n_jobs=-1,
        )
        return Pipeline([("imputer", SimpleImputer(strategy="median", add_indicator=True)), ("model", estimator)])
    if kind == "gradient_boosting":
        estimator = GradientBoostingClassifier(
            n_estimators=140,
            learning_rate=0.04,
            max_depth=2,
            min_samples_leaf=10,
            random_state=42,
        )
        return Pipeline([("imputer", SimpleImputer(strategy="median", add_indicator=True)), ("model", estimator)])
    raise ValueError(f"Unknown model kind: {kind}")


def fit_predictions(
    splits: SplitFrames,
    predictor_columns: list[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    train = splits.train
    all_predictions: list[pd.DataFrame] = []
    models: dict[str, Any] = {}
    data_by_split = [("TRAINING", train), ("VALIDATION", splits.validation), ("OUT-OF-SAMPLE", splits.test)]

    baselines: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for split_name, frame in data_by_split:
        if frame.empty:
            continue
        baselines[f"{split_name}:historical_frequency"] = (
            _historical_frequency(train, len(frame)),
            frame["full_time_result"].to_numpy(),
        )
        baselines[f"{split_name}:bookmaker_probability"] = (
            _bookmaker_probability(frame),
            frame["full_time_result"].to_numpy(),
        )
        for model_name, fit_frame in (
            ("logistic_regression", train),
            ("random_forest", train),
            ("gradient_boosting", train),
        ):
            model = models.get(model_name)
            if model is None:
                model = _make_pipeline(model_name)
                model.fit(fit_frame[predictor_columns], fit_frame["full_time_result"])
                models[model_name] = model
            probabilities = _aligned_probabilities(model, frame[predictor_columns])
            pred = _probability_frame(probabilities)
            pred.insert(0, "model", model_name)
            pred.insert(0, "split", split_name)
            pred.insert(2, "match_id", frame["match_id"].to_numpy())
            pred["actual_result"] = frame["full_time_result"].to_numpy()
            pred["home_odds"] = frame["home_odds"].to_numpy()
            pred["draw_odds"] = frame["draw_odds"].to_numpy()
            pred["away_odds"] = frame["away_odds"].to_numpy()
            pred["season"] = frame["season"].to_numpy()
            all_predictions.append(pred)
        for name, probabilities in (
            ("historical_frequency", baselines[f"{split_name}:historical_frequency"][0]),
            ("bookmaker_probability", baselines[f"{split_name}:bookmaker_probability"][0]),
        ):
            pred = _probability_frame(probabilities)
            pred.insert(0, "model", name)
            pred.insert(0, "split", split_name)
            pred.insert(2, "match_id", frame["match_id"].to_numpy())
            pred["actual_result"] = frame["full_time_result"].to_numpy()
            pred["home_odds"] = frame["home_odds"].to_numpy()
            pred["draw_odds"] = frame["draw_odds"].to_numpy()
            pred["away_odds"] = frame["away_odds"].to_numpy()
            pred["season"] = frame["season"].to_numpy()
            all_predictions.append(pred)

    # For the final test period, also offer the conventional frozen-parameter
    # refit on train + validation. The strategy is never tuned on test.
    expanded = pd.concat([splits.train, splits.validation], ignore_index=True)
    for model_name in ("logistic_regression", "random_forest", "gradient_boosting"):
        refit = _make_pipeline(model_name)
        refit.fit(expanded[predictor_columns], expanded["full_time_result"])
        models[f"{model_name}_train_validation"] = refit
        probabilities = _aligned_probabilities(refit, splits.test[predictor_columns])
        pred = _probability_frame(probabilities)
        pred.insert(0, "model", f"{model_name}_train_validation")
        pred.insert(0, "split", "OUT-OF-SAMPLE_REFIT")
        pred.insert(2, "match_id", splits.test["match_id"].to_numpy())
        pred["actual_result"] = splits.test["full_time_result"].to_numpy()
        pred["home_odds"] = splits.test["home_odds"].to_numpy()
        pred["draw_odds"] = splits.test["draw_odds"].to_numpy()
        pred["away_odds"] = splits.test["away_odds"].to_numpy()
        pred["season"] = splits.test["season"].to_numpy()
        all_predictions.append(pred)

    return pd.concat(all_predictions, ignore_index=True), models


def brier_score(probabilities: np.ndarray, actual: Iterable[str]) -> float:
    labels = list(actual)
    actual_matrix = np.zeros_like(probabilities)
    for i, label in enumerate(labels):
        actual_matrix[i, RESULTS.index(label)] = 1
    return float(np.mean(np.sum((probabilities - actual_matrix) ** 2, axis=1)))


def calibration_table(probabilities: np.ndarray, actual: Iterable[str], bins: int = 10) -> pd.DataFrame:
    labels = list(actual)
    actual_matrix = np.zeros_like(probabilities)
    for i, label in enumerate(labels):
        actual_matrix[i, RESULTS.index(label)] = 1
    confidence = probabilities.max(axis=1)
    correct = (probabilities.argmax(axis=1) == actual_matrix.argmax(axis=1)).astype(float)
    bucket = np.minimum((confidence * bins).astype(int), bins - 1)
    rows = []
    for index in range(bins):
        mask = bucket == index
        rows.append(
            {
                "probability_range": f"{index / bins:.1f}-{(index + 1) / bins:.1f}",
                "count": int(mask.sum()),
                "mean_predicted_probability": float(confidence[mask].mean()) if mask.any() else np.nan,
                "observed_accuracy": float(correct[mask].mean()) if mask.any() else np.nan,
            }
        )
    return pd.DataFrame(rows)


def evaluate_predictions(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics: list[dict[str, Any]] = []
    calibration: list[pd.DataFrame] = []
    for (split, model), group in predictions.groupby(["split", "model"], sort=True):
        probabilities = group[["prob_H", "prob_D", "prob_A"]].to_numpy(dtype=float)
        actual = group["actual_result"].tolist()
        actual_indices = np.array([RESULTS.index(label) for label in actual])
        clipped = np.clip(probabilities[np.arange(len(actual)), actual_indices], 1e-15, 1.0)
        metrics.append(
            {
                "split": split,
                "model": model,
                "rows": len(group),
                "log_loss": float(-np.mean(np.log(clipped))),
                "brier_score": brier_score(probabilities, actual),
                "accuracy": float((probabilities.argmax(axis=1) == np.array([RESULTS.index(x) for x in actual])).mean()),
            }
        )
        table = calibration_table(probabilities, actual)
        table.insert(0, "model", model)
        table.insert(0, "split", split)
        calibration.append(table)
    return pd.DataFrame(metrics), pd.concat(calibration, ignore_index=True)
