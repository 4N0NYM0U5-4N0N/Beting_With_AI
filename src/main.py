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
from .current_odds import run_current_analysis
from .data_loader import load_and_clean
from .features import build_current_features, build_features, predictor_columns
from .leakage import audit_features, write_leakage_report
from .models import chronological_splits, evaluate_predictions, fit_predictions, split_summary
from .oos_audit import create_oos_audit
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Leakage-controlled EPL quantitative research engine.")
    parser.add_argument(
        "command",
        choices=("audit", "build_features", "leakage_audit", "validate_features", "backtest", "full_analysis", "analyze_current", "oos_audit"),
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
        result = run_current_analysis(args.path)
        validation = result["validation"]
        overrounds = validation.overrounds["overround"] if not validation.overrounds.empty else pd.Series(dtype=float)
        print("Current EPL analysis complete.")
        print("")
        print(f"Fixtures analyzed:\n{result['fixtures']}")
        print("")
        print(f"Valid fixtures:\n{validation.valid_fixtures}")
        print("")
        print(f"Invalid fixtures:\n{sum(bool(errors) for errors in validation.row_errors.values())}")
        print("")
        print("Models:")
        print("Logistic Regression")
        print("Random Forest")
        print("Gradient Boosting")
        print("")
        print(
            f"Market overround range:\n"
            f"{overrounds.min():.6f} – {overrounds.max():.6f}"
            if not overrounds.empty
            else "Market overround range:\nnot available"
        )
        print("")
        print(f"Model/market probability comparisons generated:\n{result['fixtures'] * 9}")
        print("")
        print(f"Theoretical EV calculations generated:\n{result['fixtures'] * 9}")
        print("")
        print(f"Warnings:\n{result['warnings']}")
        print("")
        print("Reports:")
        print(result["validation_path"])
        print(result["analysis_path"])
        print(result["report_path"])
        return 0 if not validation.fatal_error else 1
    if args.command == "oos_audit":
        audit, reconciliation, errors = create_oos_audit()
        print("OOS audit export created:")
        print("exports/oos_backtest_audit.csv")
        print("")
        print("Reconciliation report:")
        print("exports/oos_audit_reconciliation.md")
        print("")
        print(f"Rows exported:\n{len(audit)}")
        print("")
        print(f"Models included:\n{', '.join(sorted(audit['model'].unique()))}")
        print("")
        parsed_dates = pd.to_datetime(audit["date"], dayfirst=True)
        print(f"OOS date range:\n{parsed_dates.min().date()} to {parsed_dates.max().date()}")
        print("")
        print(f"Reconciliation status:\n{'PASS' if not errors and (reconciliation['match'] == 'YES').all() else 'FAIL'}")
        if errors:
            print("\nValidation issues:")
            print("\n".join(f"- {error}" for error in errors))
        return 0 if not errors and (reconciliation["match"] == "YES").all() else 1
    return 1


if __name__ == "__main__":
    sys.exit(main())
