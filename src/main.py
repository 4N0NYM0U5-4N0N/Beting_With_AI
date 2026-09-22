from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .backtest import run_backtest
from .config import (
    AUDITS_DIR,
    BACKTESTS_DIR,
    CURRENT_ODDS_PATH,
    EXPORTS_DIR,
    HISTORICAL_PATH,
    PROCESSED_DIR,
    REPORTS_DIR,
)
from .data_loader import load_and_clean, load_current_odds
from .features import build_current_features, build_features, predictor_columns
from .leakage import audit_features, write_leakage_report
from .models import chronological_splits, evaluate_predictions, fit_predictions, split_summary
from .odds import validate_current_market
from .report import (
    write_cleaning_report,
    write_final_report,
    write_manual_validation_report,
)


def _ensure_directories() -> None:
    for directory in (PROCESSED_DIR, REPORTS_DIR, AUDITS_DIR, BACKTESTS_DIR, EXPORTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def _write_dataset_v2(frame: pd.DataFrame, safe_predictors: list[str]) -> None:
    identifiers = [
        "match_id",
        "season",
        "match_date",
        "date",
        "home_team",
        "away_team",
    ]
    targets = ["full_time_result", "full_time_home_goals", "full_time_away_goals"]
    odds = ["home_odds", "draw_odds", "away_odds"]
    columns = [column for column in identifiers + targets + odds + safe_predictors if column in frame.columns]
    export = frame[columns].copy()
    export.to_csv(EXPORTS_DIR / "epl_pre_match_features_v2.csv", index=False)
    export.to_csv(PROCESSED_DIR / "epl_pre_match_features_v2.csv", index=False)


def _run_pipeline() -> dict[str, object]:
    _ensure_directories()
    source_frame = pd.read_csv(HISTORICAL_PATH, dtype=str, keep_default_na=False)
    cleaning = load_and_clean(HISTORICAL_PATH)
    write_cleaning_report(
        cleaning.decisions,
        cleaning.anomalies,
        len(source_frame),
        len(cleaning.frame),
        EXPORTS_DIR / "cleaning_report.md",
    )
    cleaning.frame.to_csv(PROCESSED_DIR / "historical_cleaned.csv", index=False)

    built = build_features(cleaning.frame)
    feature_frame = built.frame
    feature_frame["match_date"] = pd.to_datetime(feature_frame["match_date"], errors="raise")
    predictors = predictor_columns(feature_frame)
    audit = audit_features(feature_frame, built.feature_dictionary, predictors)
    audit.to_csv(EXPORTS_DIR / "feature_dictionary.csv", index=False)
    write_leakage_report(audit, EXPORTS_DIR / "leakage_audit.md")
    safe_predictors = audit.loc[audit["classification"] == "SAFE", "feature_name"].tolist()
    if not safe_predictors:
        raise RuntimeError("No SAFE predictors remain after the leakage audit.")
    _write_dataset_v2(feature_frame, safe_predictors)
    write_manual_validation_report(built.validation_traces, EXPORTS_DIR / "manual_feature_validation.md")

    splits = chronological_splits(feature_frame)
    predictions, _ = fit_predictions(splits, safe_predictors)
    metrics, calibration = evaluate_predictions(predictions)
    predictions.to_csv(EXPORTS_DIR / "model_predictions.csv", index=False)
    metrics.to_csv(EXPORTS_DIR / "model_performance.csv", index=False)
    calibration.to_csv(EXPORTS_DIR / "calibration.csv", index=False)
    summary, details, selections = run_backtest(predictions, BACKTESTS_DIR)

    write_final_report(
        cleaned=cleaning.frame,
        anomalies=cleaning.anomalies,
        audit=audit,
        splits=split_summary(splits),
        metrics=metrics,
        calibration=calibration,
        backtest_summary=summary,
        path=REPORTS_DIR / "final_research_report.md",
    )
    summary.to_csv(EXPORTS_DIR / "backtest_summary.csv", index=False)
    split_rows = split_summary(splits)
    Path(EXPORTS_DIR / "chronological_split.json").write_text(
        json.dumps(split_rows, indent=2), encoding="utf-8"
    )
    return {
        "historical_matches": len(cleaning.frame),
        "features": len(safe_predictors),
        "safe_features": int((audit["classification"] == "SAFE").sum()),
        "leakage_features": int((audit["classification"] == "LEAKAGE").sum()),
        "unknown_features": int((audit["classification"] == "UNKNOWN").sum()),
        "training_matches": len(splits.train),
        "validation_matches": len(splits.validation),
        "out_of_sample_matches": len(splits.test),
        "metrics": metrics,
        "bookmaker_metrics": metrics[metrics["model"] == "bookmaker_probability"],
        "out_of_sample_backtest": summary[summary["split"].isin(["OUT-OF-SAMPLE", "OUT-OF-SAMPLE_REFIT"])],
        "report_paths": [
            str(EXPORTS_DIR / "cleaning_report.md"),
            str(EXPORTS_DIR / "leakage_audit.md"),
            str(EXPORTS_DIR / "manual_feature_validation.md"),
            str(REPORTS_DIR / "final_research_report.md"),
        ],
    }


def _print_summary(result: dict[str, object]) -> None:
    print(f"number of historical matches: {result['historical_matches']}")
    print(f"number of features: {result['features']}")
    print(f"number of SAFE features: {result['safe_features']}")
    print(f"number of LEAKAGE features: {result['leakage_features']}")
    print(f"number of UNKNOWN features: {result['unknown_features']}")
    print(f"training matches: {result['training_matches']}")
    print(f"validation matches: {result['validation_matches']}")
    print(f"out-of-sample matches: {result['out_of_sample_matches']}")
    print("\nmodel performance:")
    print(result["metrics"].to_string(index=False))
    print("\nbookmaker baseline performance:")
    print(result["bookmaker_metrics"].to_string(index=False))
    print("\nout-of-sample backtest results:")
    print(result["out_of_sample_backtest"].to_string(index=False))
    print("\ngenerated report paths:")
    for path in result["report_paths"]:
        print(path)


def analyze_current(path: str | Path) -> None:
    _ensure_directories()
    cleaning = load_and_clean(HISTORICAL_PATH)
    odds = validate_current_market(load_current_odds(path))
    odds["selection_key"] = (
        odds["selection"]
        .str.upper()
        .str.replace(" ", "", regex=False)
        .map({"HOME": "H", "DRAW": "D", "AWAY": "A", "H": "H", "D": "D", "A": "A"})
    )
    one_x_two = odds[odds["market"] == "1X2"].copy()
    odds_pivot = one_x_two.pivot_table(
        index=["match_date", "home_team", "away_team"],
        columns="selection_key",
        values="odds",
        aggfunc="first",
    ).reset_index()
    odds_pivot = odds_pivot.rename(columns={"H": "home_odds", "D": "draw_odds", "A": "away_odds"})
    for col in ("home_odds", "draw_odds", "away_odds"):
        if col not in odds_pivot:
            odds_pivot[col] = np.nan
    fixtures = odds[["match_date", "home_team", "away_team"]].drop_duplicates().merge(
        odds_pivot,
        on=["match_date", "home_team", "away_team"],
        how="left",
    )
    fixtures["season"] = "CURRENT"
    fixtures["date"] = fixtures["match_date"].dt.strftime("%Y-%m-%d")
    fixtures["match_id"] = (
        "CURRENT|"
        + fixtures["date"]
        + "|"
        + fixtures["home_team"]
        + "|"
        + fixtures["away_team"]
    )
    feature_result = build_current_features(cleaning.frame, fixtures)
    historical_features = build_features(cleaning.frame).frame
    historical_features["match_date"] = pd.to_datetime(historical_features["match_date"], errors="raise")
    historical_predictors = predictor_columns(historical_features)
    historical_built = build_features(cleaning.frame)
    audit = audit_features(historical_features, historical_built.feature_dictionary, historical_predictors)
    safe_predictors = audit.loc[audit["classification"] == "SAFE", "feature_name"].tolist()
    train_frame = historical_features
    from .models import _make_pipeline, _aligned_probabilities

    expanded = _make_pipeline("logistic_regression")
    expanded.fit(train_frame[safe_predictors], train_frame["full_time_result"])
    probabilities = _aligned_probabilities(expanded, feature_result.frame[safe_predictors])
    model_rows = pd.DataFrame(
        probabilities,
        columns=["model_probability_home", "model_probability_draw", "model_probability_away"],
    )
    model_rows.insert(0, "match_id", feature_result.frame["match_id"])
    probability_map = {
        "Home": "model_probability_home",
        "Draw": "model_probability_draw",
        "Away": "model_probability_away",
        "H": "model_probability_home",
        "D": "model_probability_draw",
        "A": "model_probability_away",
    }
    fixture_key = fixtures.set_index("match_id")[["match_date", "home_team", "away_team"]]
    output_rows = []
    for _, market_row in odds.iterrows():
        key = (
            market_row["match_date"],
            market_row["home_team"],
            market_row["away_team"],
        )
        matches = fixture_key[
            (fixture_key["match_date"] == key[0])
            & (fixture_key["home_team"] == key[1])
            & (fixture_key["away_team"] == key[2])
        ]
        if matches.empty:
            continue
        match_id = matches.index[0]
        probability_row = model_rows[model_rows["match_id"] == match_id].iloc[0]
        model_probability = (
            float(probability_row[probability_map.get(market_row["selection"], "model_probability_home")])
            if market_row["market"] == "1X2"
            else np.nan
        )
        market_probability = float(market_row["market_normalized_probability"])
        output_rows.append(
            {
                "date": market_row["match_date"].strftime("%Y-%m-%d"),
                "home_team": market_row["home_team"],
                "away_team": market_row["away_team"],
                "market": market_row["market"],
                "selection": market_row["selection"],
                "odds": market_row["odds"],
                "market_implied_probability": market_row["market_implied_probability"],
                "market_normalized_probability": market_probability,
                "model_probability": model_probability,
                "model_based_theoretical_ev": (model_probability * market_row["odds"] - 1) if pd.notna(model_probability) else np.nan,
                "classification": (
                    "MODEL PROBABILITY ABOVE MARKET IMPLIED PROBABILITY"
                    if pd.notna(model_probability) and model_probability > market_probability
                    else "MODEL PROBABILITY BELOW MARKET IMPLIED PROBABILITY"
                    if pd.notna(model_probability)
                    else "UNSUPPORTED MARKET — NO MODEL PROBABILITY"
                ),
                "market_overround": market_row["market_overround"],
            }
        )
    output = pd.DataFrame(output_rows)
    output.to_csv(EXPORTS_DIR / "current_match_analysis.csv", index=False)
    print("Current odds analysis is descriptive and theoretical only; no recommendation or bet is placed.")
    print(output.to_string(index=False))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Leakage-controlled EPL quantitative research engine.")
    parser.add_argument(
        "command",
        choices=("audit", "build_features", "leakage_audit", "validate_features", "backtest", "full_analysis", "analyze_current"),
    )
    parser.add_argument("path", nargs="?", default=str(CURRENT_ODDS_PATH))
    args = parser.parse_args(argv)
    if args.command == "full_analysis":
        _print_summary(_run_pipeline())
        return 0
    if args.command == "audit":
        cleaning = load_and_clean(HISTORICAL_PATH)
        _ensure_directories()
        write_cleaning_report(cleaning.decisions, cleaning.anomalies, len(pd.read_csv(HISTORICAL_PATH)), len(cleaning.frame), EXPORTS_DIR / "cleaning_report.md")
        print(f"Clean historical matches: {len(cleaning.frame)}")
        print(f"Flagged anomalies: {len(cleaning.anomalies)}")
        print(EXPORTS_DIR / "cleaning_report.md")
        return 0
    if args.command == "build_features":
        _ensure_directories()
        cleaning = load_and_clean(HISTORICAL_PATH)
        built = build_features(cleaning.frame)
        built.frame.to_csv(EXPORTS_DIR / "epl_pre_match_features_v2.csv", index=False)
        write_manual_validation_report(built.validation_traces, EXPORTS_DIR / "manual_feature_validation.md")
        print(f"Built {len(built.frame)} feature rows and {len(predictor_columns(built.frame))} numeric predictors.")
        return 0
    if args.command == "leakage_audit":
        _ensure_directories()
        cleaning = load_and_clean(HISTORICAL_PATH)
        built = build_features(cleaning.frame)
        predictors = predictor_columns(built.frame)
        audit = audit_features(built.frame, built.feature_dictionary, predictors)
        audit.to_csv(EXPORTS_DIR / "feature_dictionary.csv", index=False)
        write_leakage_report(audit, EXPORTS_DIR / "leakage_audit.md")
        print(audit["classification"].value_counts().to_string())
        return 0
    if args.command == "validate_features":
        _ensure_directories()
        cleaning = load_and_clean(HISTORICAL_PATH)
        built = build_features(cleaning.frame)
        write_manual_validation_report(built.validation_traces, EXPORTS_DIR / "manual_feature_validation.md")
        print(f"Wrote {len(built.validation_traces)} manual traces.")
        return 0
    if args.command in {"backtest", "full_analysis"}:
        _print_summary(_run_pipeline())
        return 0
    if args.command == "analyze_current":
        analyze_current(args.path)
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
