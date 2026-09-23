from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import (
    EXPORTS_DIR,
    HISTORICAL_PATH,
    PROCESSED_DIR,
    RESULTS,
    REPORTS_DIR,
    TEAM_ALIASES_PATH,
)
from .data_loader import load_and_clean, parse_date_strict
from .features import build_current_features, build_features, predictor_columns
from .leakage import audit_features
from .models import _aligned_probabilities, _make_pipeline


REQUIRED_COLUMNS = ["date", "home_team", "away_team", "market", "selection", "odds"]
SELECTION_ALIASES = {
    "H": "H",
    "D": "D",
    "A": "A",
    "HOME": "H",
    "DRAW": "D",
    "AWAY": "A",
}
ANALYSIS_COLUMNS = [
    "date",
    "home_team",
    "away_team",
    "market",
    "selection",
    "odds",
    "raw_implied_probability",
    "market_overround",
    "normalized_market_probability",
    "logistic_probability",
    "random_forest_probability",
    "gradient_boosting_probability",
    "logistic_probability_difference",
    "random_forest_probability_difference",
    "gradient_boosting_probability_difference",
    "logistic_theoretical_ev",
    "random_forest_theoretical_ev",
    "gradient_boosting_theoretical_ev",
    "model_agreement",
    "market_most_likely_selection",
    "data_quality_status",
    "calibration_status",
]
MODEL_NAMES = ("logistic_regression", "random_forest", "gradient_boosting")
MODEL_LABELS = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "gradient_boosting": "Gradient Boosting",
}


@dataclass
class CurrentOddsValidation:
    raw: pd.DataFrame
    normalized: pd.DataFrame
    row_errors: dict[int, list[str]] = field(default_factory=dict)
    fixture_errors: dict[str, list[str]] = field(default_factory=dict)
    warnings: dict[str, list[str]] = field(default_factory=dict)
    overrounds: pd.DataFrame = field(default_factory=pd.DataFrame)
    fatal_error: str | None = None

    @property
    def invalid_rows(self) -> int:
        return sum(bool(errors) for errors in self.row_errors.values())

    @property
    def valid_fixtures(self) -> int:
        return int(self.normalized["fixture_key"].nunique()) if not self.normalized.empty else 0

    def _fixture_source_rows(self) -> dict[str, list[int]]:
        if self.raw.empty or "fixture_key" not in self.raw:
            return {}
        output: dict[str, list[int]] = {}
        for key, group in self.raw.dropna(subset=["fixture_key"]).groupby("fixture_key", sort=False):
            output[str(key)] = [int(value) for value in group["_source_row"]]
        return output


def _fixture_key(date_value: pd.Timestamp, home: str, away: str, market: str) -> str:
    return f"{date_value.date()}|{home}|{away}|{market}"


def _load_aliases(path: str | Path = TEAM_ALIASES_PATH) -> dict[str, str]:
    aliases = pd.read_csv(path, dtype=str, keep_default_na=False)
    expected = {"input_name", "canonical_name"}
    missing = expected.difference(aliases.columns)
    if missing:
        raise ValueError(f"Team alias file is missing required columns: {sorted(missing)}")
    mapping: dict[str, str] = {}
    for _, row in aliases.iterrows():
        input_name = str(row["input_name"]).strip()
        canonical_name = str(row["canonical_name"]).strip()
        if not input_name or not canonical_name:
            raise ValueError("Team alias file contains a blank input or canonical name.")
        key = input_name.casefold()
        if key in mapping and mapping[key] != canonical_name:
            raise ValueError(f"Team alias maps to multiple canonical names: {input_name}")
        mapping[key] = canonical_name
    return mapping


def validate_current_odds(
    path: str | Path,
    aliases_path: str | Path = TEAM_ALIASES_PATH,
) -> CurrentOddsValidation:
    raw = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = [column for column in REQUIRED_COLUMNS if column not in raw.columns]
    if missing:
        return CurrentOddsValidation(
            raw=raw,
            normalized=pd.DataFrame(),
            fatal_error=f"Current odds file is missing required columns: {missing}",
        )

    raw = raw[REQUIRED_COLUMNS].copy()
    raw["_source_row"] = np.arange(2, len(raw) + 2)
    aliases = _load_aliases(aliases_path)
    records: list[dict[str, Any]] = []
    row_errors: dict[int, list[str]] = {}

    def add_error(source_row: int, detail: str) -> None:
        row_errors.setdefault(source_row, []).append(detail)

    for _, row in raw.iterrows():
        source_row = int(row["_source_row"])
        date_value: pd.Timestamp | pd.NaT = pd.NaT
        try:
            date_value = parse_date_strict(row["date"])
        except ValueError as exc:
            add_error(source_row, f"invalid date: {exc}")

        home_input = str(row["home_team"]).strip()
        away_input = str(row["away_team"]).strip()
        home = aliases.get(home_input.casefold())
        away = aliases.get(away_input.casefold())
        if not home_input:
            add_error(source_row, "home_team is blank.")
        elif home is None:
            add_error(source_row, f"unknown home team {home_input!r}.")
        if not away_input:
            add_error(source_row, "away_team is blank.")
        elif away is None:
            add_error(source_row, f"unknown away team {away_input!r}.")
        if home is not None and away is not None and home == away:
            add_error(source_row, "home_team and away_team resolve to the same team.")

        market = str(row["market"]).strip().upper().replace(" ", "")
        if market != "1X2":
            add_error(source_row, f"unsupported market {row['market']!r}; only 1X2 is supported.")

        selection_input = str(row["selection"]).strip().upper()
        selection = SELECTION_ALIASES.get(selection_input)
        if selection is None:
            add_error(source_row, f"invalid selection {row['selection']!r}; expected H, D, or A.")

        odds = pd.to_numeric(row["odds"], errors="coerce")
        if pd.isna(odds) or not np.isfinite(float(odds)) or float(odds) <= 1:
            add_error(source_row, "odds must be a finite decimal number greater than 1.")

        fixture_key = (
            _fixture_key(date_value, home, away, market)
            if pd.notna(date_value) and home is not None and away is not None
            else ""
        )
        records.append(
            {
                "_source_row": source_row,
                "date": date_value.strftime("%Y-%m-%d") if pd.notna(date_value) else str(row["date"]).strip(),
                "match_date": date_value,
                "home_team": home or home_input,
                "away_team": away or away_input,
                "market": market,
                "selection": selection or selection_input,
                "odds": float(odds) if pd.notna(odds) and np.isfinite(float(odds)) else np.nan,
                "fixture_key": fixture_key,
            }
        )

    candidate = pd.DataFrame(records)
    fixture_errors: dict[str, list[str]] = {}
    warnings: dict[str, list[str]] = {}
    if not candidate.empty:
        for key, group in candidate[candidate["fixture_key"] != ""].groupby("fixture_key", sort=False):
            group_errors: list[str] = []
            selections = group["selection"].tolist()
            duplicate = len(selections) != len(set(selections))
            if duplicate:
                group_errors.append("duplicate fixture/market/selection rows.")
            missing_selections = sorted(set(RESULTS).difference(selections))
            if missing_selections:
                group_errors.append(f"missing selections: {', '.join(missing_selections)}.")
            if group_errors:
                fixture_errors[str(key)] = group_errors
            numeric_odds = group["odds"].dropna().astype(float)
            if len(numeric_odds) == 3 and not duplicate and not missing_selections:
                overround = float((1 / numeric_odds).sum())
                fixture_warnings: list[str] = []
                if overround < 1.0 or overround > 1.20:
                    fixture_warnings.append(f"unusual bookmaker overround {overround:.6f}.")
                if (numeric_odds > 20).any():
                    fixture_warnings.append("extreme odds above 20.0.")
                if fixture_warnings:
                    warnings[str(key)] = fixture_warnings

    valid_mask = []
    for _, row in candidate.iterrows():
        source_row = int(row["_source_row"])
        errors = list(row_errors.get(source_row, []))
        if row["fixture_key"] in fixture_errors:
            errors.extend(fixture_errors[row["fixture_key"]])
        row_errors[source_row] = list(dict.fromkeys(errors))
        valid_mask.append(not errors)
    candidate["_valid"] = valid_mask
    normalized = candidate[candidate["_valid"]].copy()

    if not normalized.empty:
        normalized["raw_implied_probability"] = 1 / normalized["odds"].astype(float)
        overround = normalized.groupby("fixture_key")["raw_implied_probability"].transform("sum")
        normalized["market_overround"] = overround
        normalized["normalized_market_probability"] = normalized["raw_implied_probability"] / overround
        overrounds = (
            normalized.groupby(["fixture_key", "date", "home_team", "away_team", "market"], as_index=False)
            .agg(
                odds_rows=("selection", "size"),
                overround=("market_overround", "first"),
            )
        )
    else:
        overrounds = pd.DataFrame(
            columns=["fixture_key", "date", "home_team", "away_team", "market", "odds_rows", "overround"]
        )
    return CurrentOddsValidation(
        raw=candidate,
        normalized=normalized,
        row_errors=row_errors,
        fixture_errors=fixture_errors,
        warnings=warnings,
        overrounds=overrounds,
    )


def write_current_odds_validation_report(
    validation: CurrentOddsValidation,
    path: str | Path = EXPORTS_DIR / "current_odds_validation.md",
) -> None:
    lines = [
        "# Current SportyBet Odds Validation",
        "",
        f"- Input rows: {len(validation.raw)}",
        f"- Fixtures with valid H/D/A odds: {validation.valid_fixtures}",
        f"- Invalid rows: {sum(bool(errors) for errors in validation.row_errors.values())}",
        f"- Duplicate fixture/market/selection groups: "
        f"{sum('duplicate fixture/market/selection rows.' in errors for errors in validation.fixture_errors.values())}",
        "",
    ]
    if validation.fatal_error:
        lines.extend(["## Fatal input error", "", f"- {validation.fatal_error}", ""])
        Path(path).write_text("\n".join(lines), encoding="utf-8")
        return

    lines.extend(["## Invalid rows", ""])
    invalid_lines = [
        f"- Source row {row}: " + "; ".join(errors)
        for row, errors in sorted(validation.row_errors.items())
        if errors
    ]
    lines.extend(invalid_lines or ["No invalid rows."])
    lines.extend(["", "## Missing selections", ""])
    missing = [
        f"- {key}: {', '.join(error.split(': ', 1)[1][:-1].split(', '))}"
        for key, errors in validation.fixture_errors.items()
        for error in errors
        if error.startswith("missing selections:")
    ]
    lines.extend(missing or ["No missing H/D/A selections."])
    lines.extend(["", "## Unknown teams", ""])
    unknown = [
        f"- Source row {row}: " + "; ".join(error for error in errors if "unknown" in error)
        for row, errors in sorted(validation.row_errors.items())
        if any("unknown" in error for error in errors)
    ]
    lines.extend(unknown or ["No unknown teams."])
    lines.extend(["", "## Duplicate selections", ""])
    duplicates = [
        f"- {key}: " + "; ".join(errors)
        for key, errors in validation.fixture_errors.items()
        if any("duplicate" in error for error in errors)
    ]
    lines.extend(duplicates or ["No duplicate selections."])

    numeric_odds = validation.raw["odds"].dropna().astype(float) if not validation.raw.empty else pd.Series(dtype=float)
    lines.extend(
        [
            "",
            "## Odds range",
            "",
            f"- Minimum odds: {numeric_odds.min() if not numeric_odds.empty else 'not available'}",
            f"- Maximum odds: {numeric_odds.max() if not numeric_odds.empty else 'not available'}",
            "",
            "## Bookmaker overround by fixture",
            "",
            "| Date | Home | Away | Market | Odds rows | Overround |",
            "|---|---|---|---|---:|---:|",
        ]
    )
    if validation.overrounds.empty:
        lines.append("| No complete fixtures | | | | | |")
    else:
        for _, row in validation.overrounds.iterrows():
            lines.append(
                f"| {row['date']} | {row['home_team']} | {row['away_team']} | {row['market']} | "
                f"{int(row['odds_rows'])} | {float(row['overround']):.6f} |"
            )
    lines.extend(
        [
            "",
            "Team aliases are applied only from the controlled `data/processed/team_aliases.csv` file. "
            "Unknown names are reported and are not analyzed.",
            "A complete fixture must contain exactly one H, D, and A row. Invalid fixtures remain visible "
            "in this report and are not silently repaired.",
        ]
    )
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _calibration_status(
    calibration: pd.DataFrame,
    model: str,
    probabilities: np.ndarray,
) -> tuple[str, dict[str, Any] | None]:
    if calibration.empty:
        return "Insufficient historical calibration evidence for this probability range.", None
    rows = calibration[(calibration["split"] == "OUT-OF-SAMPLE") & (calibration["model"] == model)].copy()
    if rows.empty:
        rows = calibration[(calibration["split"] == "VALIDATION") & (calibration["model"] == model)].copy()
    if rows.empty:
        return "Insufficient historical calibration evidence for this probability range.", None
    confidence = float(np.max(probabilities))
    bucket = min(int(confidence * 10), 9)
    bucket_rows = rows[rows["probability_range"] == f"{bucket / 10:.1f}-{(bucket + 1) / 10:.1f}"]
    if bucket_rows.empty or int(bucket_rows.iloc[0]["count"]) == 0:
        return "Insufficient historical calibration evidence for this probability range.", None
    row = bucket_rows.iloc[0]
    split_label = "OOS" if rows["split"].iloc[0] == "OUT-OF-SAMPLE" else "validation"
    status = (
        f"{split_label} confidence bucket {row['probability_range']}; "
        f"observed accuracy {float(row['observed_accuracy']):.4f} from {int(row['count'])} matches"
    )
    return status, row.to_dict()


def _quality_warnings(
    feature_row: pd.Series,
    model_probabilities: dict[str, np.ndarray],
    validation_warnings: list[str],
) -> list[str]:
    warnings = list(validation_warnings)
    for prefix in ("home_team", "away_team"):
        if float(feature_row.get(f"{prefix}_overall_matches_played", 0)) < 5:
            warnings.append(f"insufficient historical matches for {prefix.replace('_team', '')} team")
        if float(feature_row.get(f"{prefix}_rolling5_matches_available", 0)) < 5:
            warnings.append(f"insufficient rolling-5 sample for {prefix.replace('_team', '')} team")
    if float(feature_row.get("h2h_matches_played", 0)) == 0:
        warnings.append("missing H2H history")
    missing_features = int(feature_row.isna().sum())
    if missing_features:
        warnings.append(f"{missing_features} feature values require model imputation")
    for model, probabilities in model_probabilities.items():
        if len(probabilities) != 3 or not np.isfinite(probabilities).all() or not np.isclose(probabilities.sum(), 1.0, atol=1e-8):
            warnings.append(f"model probability anomaly: {model}")
    return list(dict.fromkeys(warnings))


def _agreement(probabilities: dict[str, np.ndarray]) -> tuple[str, dict[str, str]]:
    selections = {
        model: RESULTS[int(np.argmax(values))]
        for model, values in probabilities.items()
    }
    unique = set(selections.values())
    if len(unique) == 1:
        status = "AGREEMENT"
    elif len(unique) == 2:
        status = "PARTIAL_AGREEMENT"
    else:
        status = "DISAGREEMENT"
    return status, selections


def _invalid_output_rows(validation: CurrentOddsValidation) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for _, item in validation.raw.iterrows():
        errors = validation.row_errors.get(int(item["_source_row"]), [])
        if not errors:
            continue
        raw_probability = (
            float(1 / item["odds"])
            if pd.notna(item["odds"]) and float(item["odds"]) > 1
            else np.nan
        )
        rows.append(
            {
                "date": item["date"],
                "home_team": item["home_team"],
                "away_team": item["away_team"],
                "market": item["market"],
                "selection": item["selection"],
                "odds": item["odds"],
                "raw_implied_probability": raw_probability,
                "market_overround": np.nan,
                "normalized_market_probability": np.nan,
                "model_agreement": "NOT_ANALYZED",
                "market_most_likely_selection": "NOT_ANALYZED",
                "data_quality_status": "INVALID: " + "; ".join(errors),
                "calibration_status": "Not analyzed because the fixture is invalid.",
            }
        )
    return rows


def run_current_analysis(
    path: str | Path,
    *,
    historical_path: str | Path = HISTORICAL_PATH,
    aliases_path: str | Path = TEAM_ALIASES_PATH,
) -> dict[str, Any]:
    validation = validate_current_odds(path, aliases_path)
    validation_path = EXPORTS_DIR / "current_odds_validation.md"
    analysis_path = EXPORTS_DIR / "current_sportybet_analysis.csv"
    report_path = REPORTS_DIR / "current_sportybet_analysis.md"
    EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    write_current_odds_validation_report(validation, validation_path)
    if validation.fatal_error:
        pd.DataFrame(columns=ANALYSIS_COLUMNS).to_csv(analysis_path, index=False)
        Path(report_path).write_text(
            "# Current SportyBet Odds Analysis\n\n"
            f"Analysis could not run: {validation.fatal_error}\n",
            encoding="utf-8",
        )
        return {
            "validation": validation,
            "rows": 0,
            "fixtures": 0,
            "warnings": len(validation.row_errors),
            "analysis_path": analysis_path,
            "validation_path": validation_path,
            "report_path": report_path,
        }

    output_rows = _invalid_output_rows(validation)
    report_fixtures: list[dict[str, Any]] = []
    historical = load_and_clean(historical_path).frame
    historical_features = build_features(historical)
    feature_frame = historical_features.frame.copy()
    feature_frame["match_date"] = pd.to_datetime(feature_frame["match_date"], errors="raise")
    predictors = predictor_columns(feature_frame)
    feature_audit = audit_features(feature_frame, historical_features.feature_dictionary, predictors)
    safe_predictors = feature_audit.loc[feature_audit["classification"] == "SAFE", "feature_name"].tolist()
    if not safe_predictors:
        raise RuntimeError("No SAFE predictors remain for current inference.")

    calibration_path = EXPORTS_DIR / "calibration.csv"
    calibration = pd.read_csv(calibration_path) if calibration_path.exists() else pd.DataFrame()
    models = {}
    for model_name in MODEL_NAMES:
        model = _make_pipeline(model_name)
        model.fit(feature_frame[safe_predictors], feature_frame["full_time_result"])
        models[model_name] = model

    for fixture_key, fixture in validation.normalized.groupby("fixture_key", sort=False):
        fixture = fixture.sort_values("selection")
        first = fixture.iloc[0]
        current = pd.DataFrame(
            [
                {
                    "match_id": f"CURRENT|{fixture_key}",
                    "season": "CURRENT",
                    "match_date": first["match_date"],
                    "date": first["date"],
                    "home_team": first["home_team"],
                    "away_team": first["away_team"],
                    "home_odds": float(fixture.loc[fixture["selection"] == "H", "odds"].iloc[0]),
                    "draw_odds": float(fixture.loc[fixture["selection"] == "D", "odds"].iloc[0]),
                    "away_odds": float(fixture.loc[fixture["selection"] == "A", "odds"].iloc[0]),
                }
            ]
        )
        built_current = build_current_features(historical, current)
        feature_row = built_current.frame.iloc[0]
        model_probabilities = {
            model_name: _aligned_probabilities(model, built_current.frame[safe_predictors])[0]
            for model_name, model in models.items()
        }
        agreement, model_selections = _agreement(model_probabilities)
        market_probs = fixture.set_index("selection")["normalized_market_probability"].to_dict()
        market_most_likely = max(market_probs, key=market_probs.get)
        fixture_warnings = _quality_warnings(
            feature_row,
            model_probabilities,
            validation.warnings.get(str(fixture_key), []),
        )
        calibration_details: dict[str, Any] = {}
        for model_name, probabilities in model_probabilities.items():
            status, detail = _calibration_status(calibration, model_name, probabilities)
            calibration_details[model_name] = {"status": status, "detail": detail}

        for _, odds_row in fixture.iterrows():
            selection = odds_row["selection"]
            outcome_index = RESULTS.index(selection)
            output = {
                "date": odds_row["date"],
                "home_team": odds_row["home_team"],
                "away_team": odds_row["away_team"],
                "market": odds_row["market"],
                "selection": selection,
                "odds": float(odds_row["odds"]),
                "raw_implied_probability": float(odds_row["raw_implied_probability"]),
                "market_overround": float(odds_row["market_overround"]),
                "normalized_market_probability": float(odds_row["normalized_market_probability"]),
                "model_agreement": agreement,
                "market_most_likely_selection": market_most_likely,
                "data_quality_status": "VALID_WITH_WARNINGS" if fixture_warnings else "VALID",
                "calibration_status": "; ".join(
                    f"{MODEL_LABELS[model]}: {calibration_details[model]['status']}"
                    for model in MODEL_NAMES
                ),
            }
            for model_name in MODEL_NAMES:
                probability = float(model_probabilities[model_name][outcome_index])
                prefix = model_name.removesuffix("_regression")
                output[f"{prefix}_probability"] = probability
                output[f"{prefix}_probability_difference"] = probability - float(
                    odds_row["normalized_market_probability"]
                )
                output[f"{prefix}_theoretical_ev"] = probability * float(odds_row["odds"]) - 1
            output_rows.append(output)

        report_fixtures.append(
            {
                "fixture_key": fixture_key,
                "date": first["date"],
                "home_team": first["home_team"],
                "away_team": first["away_team"],
                "market_overround": float(first["market_overround"]),
                "fixture": fixture,
                "model_probabilities": model_probabilities,
                "model_selections": model_selections,
                "agreement": agreement,
                "market_probs": market_probs,
                "market_most_likely": market_most_likely,
                "warnings": fixture_warnings,
                "calibration": calibration_details,
                "feature_row": feature_row,
            }
        )

    analysis = pd.DataFrame(output_rows)
    for column in ANALYSIS_COLUMNS:
        if column not in analysis:
            analysis[column] = np.nan
    analysis = analysis[ANALYSIS_COLUMNS]
    analysis.to_csv(analysis_path, index=False)
    _write_analysis_report(report_fixtures, validation, report_path)
    return {
        "validation": validation,
        "rows": len(analysis),
        "analyzed_rows": len(validation.normalized),
        "fixtures": len(report_fixtures),
        "warnings": sum(len(item["warnings"]) for item in report_fixtures) + sum(
            bool(errors) for errors in validation.row_errors.values()
        ),
        "analysis_path": analysis_path,
        "validation_path": validation_path,
        "report_path": report_path,
    }


def _write_analysis_report(
    fixtures: list[dict[str, Any]],
    validation: CurrentOddsValidation,
    path: str | Path,
) -> None:
    lines = [
        "# Current SportyBet Odds Analysis",
        "",
        "This is a descriptive research analysis. It applies the existing historical "
        "feature definitions and fixed model procedures to manually supplied current odds.",
        "It does not log in, scrape, place, submit, or automate bets.",
        "",
        f"- Fixtures analyzed: {len(fixtures)}",
        f"- Input rows: {len(validation.raw)}",
        f"- Invalid input rows: {sum(bool(errors) for errors in validation.row_errors.values())}",
        "- Historical methodology changed: NO",
        "- Current fixtures used for model training: NO",
        "- Same-date historical results used: NO",
        "",
    ]
    if not fixtures:
        lines.append("No valid complete fixtures were available for model inference.")
    invalid_rows = [
        f"- Source row {row}: " + "; ".join(errors)
        for row, errors in sorted(validation.row_errors.items())
        if errors
    ]
    if invalid_rows:
        lines.extend(["", "## Invalid input rows", "", *invalid_rows])
    for item in fixtures:
        fixture = item["fixture"]
        lines.extend(
            [
                f"## {item['home_team']} vs {item['away_team']}",
                "",
                f"- Date: {item['date']}",
                "- Market: 1X2",
                f"- Bookmaker overround: {item['market_overround']:.6f}",
                "",
                "### Market probabilities",
                "",
                "| Selection | Odds | Raw implied probability | Normalized market probability |",
                "|---|---:|---:|---:|",
            ]
        )
        for _, row in fixture.sort_values("selection").iterrows():
            lines.append(
                f"| {row['selection']} | {float(row['odds']):.4f} | "
                f"{float(row['raw_implied_probability']):.6f} | {float(row['normalized_market_probability']):.6f} |"
            )
        lines.extend(["", "### Model probabilities and theoretical comparison", ""])
        lines.extend(
            [
                "| Model | H | D | A | Most probable | Calibration information |",
                "|---|---:|---:|---:|---|---|",
            ]
        )
        for model_name in MODEL_NAMES:
            probabilities = item["model_probabilities"][model_name]
            lines.append(
                f"| {MODEL_LABELS[model_name]} | {probabilities[0]:.6f} | {probabilities[1]:.6f} | "
                f"{probabilities[2]:.6f} | {item['model_selections'][model_name]} | "
                f"{item['calibration'][model_name]['status']} |"
            )
        lines.extend(
            [
                "",
                "| Selection | Odds | Model | Probability | Probability difference | THEORETICAL MODEL EV |",
                "|---|---:|---|---:|---:|---:|",
            ]
        )
        for _, row in fixture.sort_values("selection").iterrows():
            selection = row["selection"]
            index = RESULTS.index(selection)
            for model_name in MODEL_NAMES:
                probability = float(item["model_probabilities"][model_name][index])
                market_probability = float(row["normalized_market_probability"])
                lines.append(
                    f"| {selection} | {float(row['odds']):.4f} | {MODEL_LABELS[model_name]} | "
                    f"{probability:.6f} | {probability - market_probability:.6f} | "
                    f"{probability * float(row['odds']) - 1:.6f} |"
                )
        lines.extend(
            [
                "",
                "### Model agreement and market comparison",
                "",
                f"- Logistic Regression most probable outcome: {item['model_selections']['logistic_regression']}",
                f"- Random Forest most probable outcome: {item['model_selections']['random_forest']}",
                f"- Gradient Boosting most probable outcome: {item['model_selections']['gradient_boosting']}",
                f"- Agreement status: {item['agreement']}",
                f"- Market highest normalized probability: {item['market_most_likely']}",
                "",
                "### Data-quality warnings",
                "",
            ]
        )
        lines.extend(f"- {warning}" for warning in item["warnings"] or ["None."])
        lines.extend(
            [
                "",
                "Calibration uses the existing validation/OOS confidence-bucket results. "
                "It is historical context, not a confidence interval or guarantee.",
                "",
            ]
        )
    lines.extend(
        [
            "## Interpretation limits",
            "",
            "A positive model-market probability difference or THEORETICAL MODEL EV is conditional "
            "on the model and market assumptions. It is not a guaranteed return, recommendation, "
            "or evidence that a fixture will win.",
        ]
    )
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")