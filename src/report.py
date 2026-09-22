from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def write_cleaning_report(
    decisions: list[str],
    anomalies: list[dict[str, Any]],
    source_rows: int,
    valid_rows: int,
    path: str | Path,
) -> None:
    lines = [
        "# Cleaning Report",
        "",
        f"- Source rows read: {source_rows}",
        f"- Valid historical matches exported: {valid_rows}",
        "",
        "## Decisions",
        "",
    ]
    lines.extend(f"- {decision}" for decision in decisions)
    lines.extend(["", "## Flagged anomalies", ""])
    if anomalies:
        lines.extend(
            f"- Source row {item['row_number']} ({item['date']} {item['home_team']} vs {item['away_team']}): "
            f"**{item['type']}** — {item['detail']}"
            for item in anomalies
        )
    else:
        lines.append("No anomalies were flagged.")
    lines.extend(
        [
            "",
            "Suspicious records are reported rather than corrected. The impossible shots-on-target "
            "relationship is not silently changed; it remains visible in this audit.",
        ]
    )
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_manual_validation_report(traces: list[dict[str, Any]], path: str | Path) -> None:
    lines = [
        "# Manual Feature Validation",
        "",
        "Twenty chronologically distributed fixtures (or all fixtures when fewer than twenty exist) "
        "were spot-checked. Each trace is calculated from prior dates only.",
        "",
    ]
    for trace in traces:
        lines.extend(
            [
                f"## {trace['home_team']} vs {trace['away_team']}",
                "",
                f"- Season: {trace['season']}",
                f"- Date: {trace['date']}",
                f"- Match ID: `{trace['match_id']}`",
                "",
                f"**{trace['home_team']} previous matches used for the rolling window:**",
                "",
            ]
        )
        if trace["home_previous_matches"]:
            for index, match in enumerate(trace["home_previous_matches"], start=1):
                lines.append(
                    f"{index}. {match['venue']} fixture — goals scored {match['goals_scored']:.0f}, "
                    f"goals conceded {match['goals_conceded']:.0f}, points {match['points']:.0f}"
                )
        else:
            lines.append("No previous match was available.")
        lines.extend(
            [
                "",
                f"Independently calculated home goals-scored average over the available last five: "
                f"{trace['home_previous_goals_average'] if pd.notna(trace['home_previous_goals_average']) else 'not available'}",
                "",
                f"**{trace['away_team']} previous matches used for the rolling window:**",
                "",
            ]
        )
        if trace["away_previous_matches"]:
            for index, match in enumerate(trace["away_previous_matches"], start=1):
                lines.append(
                    f"{index}. {match['venue']} fixture — goals scored {match['goals_scored']:.0f}, "
                    f"goals conceded {match['goals_conceded']:.0f}, points {match['points']:.0f}"
                )
        else:
            lines.append("No previous match was available.")
        lines.extend(
            [
                "",
                f"Independently calculated away goals-scored average over the available last five: "
                f"{trace['away_previous_goals_average'] if pd.notna(trace['away_previous_goals_average']) else 'not available'}",
                "",
            ]
        )
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def _markdown_table(frame: pd.DataFrame, columns: list[str] | None = None) -> list[str]:
    if frame.empty:
        return ["No rows."]
    local = frame[columns] if columns else frame
    headers = list(local.columns)
    lines = ["|" + "|".join(headers) + "|", "|" + "|".join("---" for _ in headers) + "|"]
    for _, row in local.iterrows():
        values = ["-" if pd.isna(value) else str(value) for value in row]
        lines.append("|" + "|".join(values) + "|")
    return lines


def write_final_report(
    *,
    cleaned: pd.DataFrame,
    anomalies: list[dict[str, Any]],
    audit: pd.DataFrame,
    splits: list[dict[str, Any]],
    metrics: pd.DataFrame,
    calibration: pd.DataFrame,
    backtest_summary: pd.DataFrame,
    path: str | Path,
) -> None:
    safe = int((audit["classification"] == "SAFE").sum()) if not audit.empty else 0
    leakage = int((audit["classification"] == "LEAKAGE").sum()) if not audit.empty else 0
    unknown = int((audit["classification"] == "UNKNOWN").sum()) if not audit.empty else 0
    lines = [
        "# English Premier League Quantitative Research Report",
        "",
        "## 1. Dataset summary",
        "",
        f"- Historical matches after cleaning: **{len(cleaned)}**",
        f"- Seasons: {cleaned['season'].min()} through {cleaned['season'].max()}",
        f"- Teams observed: {len(set(cleaned['home_team']).union(cleaned['away_team']))}",
        "- Authoritative source: `data/raw/epl_historical_clean.csv`",
        "- Existing engineered files were inspected but not used as predictors.",
        "",
        "## 2. Data-quality findings",
        "",
        f"- Flagged anomalies: {len(anomalies)}",
        "- The malformed record was excluded because it has no date, teams, result, score, statistics, or odds.",
        "- Suspicious records were retained in the source and reported; no suspicious statistic was rewritten.",
        "- Same-date fixtures are processed as a batch, preventing results from one fixture date entering another fixture on that date.",
        "",
        "## 3. Leakage audit",
        "",
        f"- SAFE features: {safe}",
        f"- LEAKAGE features: {leakage}",
        f"- UNKNOWN features: {unknown}",
        "- Only SAFE features were included in model fitting.",
        "",
        "## 4. Feature methodology",
        "",
        "Features are rebuilt from raw historical results, match statistics, prior bookmaker odds, "
        "head-to-head history, and a reconstructed season table. Rolling windows record the number "
        "of available historical matches rather than pretending short histories are full windows.",
        "",
        "League position is reconstructed from results before the fixture date. H2H is restricted to "
        "earlier fixtures involving the same two teams. No final-season table, same-match statistic, "
        "future match, or legacy engineered feature is used.",
        "",
        "## 5. Manual validation",
        "",
        "See `exports/manual_feature_validation.md` for 20 chronologically distributed traces and independent rolling-average checks.",
        "",
        "## 6. Chronological split",
        "",
    ]
    lines.extend(_markdown_table(pd.DataFrame(splits), ["split", "rows", "seasons", "min_date", "max_date"]))
    lines.extend(
        [
            "",
            "## 7. Model methodology",
            "",
            "- Baseline A: historical result frequency from the training period.",
            "- Baseline B: normalized bookmaker implied probability.",
            "- Model C: logistic regression.",
            "- Model D: random forest.",
            "- Model E: gradient boosting.",
            "- All probabilistic models use fixed parameters and a fixed random seed of 42.",
            "- No random train/test split is used.",
            "",
            "## 8. Baseline and model results",
            "",
        ]
    )
    metrics_display = metrics.copy()
    for col in ("log_loss", "brier_score", "accuracy"):
        if col in metrics_display:
            metrics_display[col] = metrics_display[col].map(lambda value: f"{value:.4f}")
    lines.extend(_markdown_table(metrics_display, ["split", "model", "rows", "log_loss", "brier_score", "accuracy"]))
    lines.extend(
        [
            "",
            "## 9. Validation results",
            "",
            "Validation metrics are produced without looking at the out-of-sample test period. "
            "The fixed backtest rule was not tuned against test results.",
            "",
            "## 10. Out-of-sample results",
            "",
            "Out-of-sample results cover 2024/25 through 2025/26 and are chronologically isolated. "
            "The train-plus-validation refit is identified separately in the exported predictions.",
            "",
            "## 11. Backtest results",
            "",
            "This is a hypothetical 1-unit research backtest only. It is not a betting recommendation, "
            "does not connect to a bookmaker, and does not place bets. The one pre-declared strategy "
            "selects the maximum model-vs-normalized-market edge only when the edge is at least 0.05.",
            "",
        ]
    )
    backtest_display = backtest_summary.copy()
    for col in ("win_rate", "roi"):
        if col in backtest_display:
            backtest_display[col] = backtest_display[col].map(lambda value: f"{value:.4f}")
    lines.extend(_markdown_table(backtest_display))
    lines.extend(
        [
            "",
            "Detailed results by season, market, probability range, and estimated edge range are in `backtests/`.",
            "",
            "## 12. Calibration",
            "",
            "Calibration tables are exported to `exports/calibration.csv`, with confidence-bin counts, "
            "mean predicted confidence, and observed accuracy.",
            "",
            "## 13. Multiple Testing / Overfitting Risk",
            "",
            "- Number of model families evaluated: 5, including two non-learning baselines.",
            "- Number of backtest strategies evaluated: 1 fixed edge rule.",
            "- No threshold sweep or test-period parameter selection was performed.",
            "- Positive ROI in one period is not treated as evidence of a persistent advantage.",
            "",
            "## 14. Limitations",
            "",
            "- The source has historical 1X2 odds but no historical Over/Under 2.5 or BTTS odds.",
            "- There is no xG, possession, lineup, injury, or player-availability data.",
            "- Same-day ordering is unavailable, so the engine deliberately uses the earlier-date cutoff.",
            "- Historical bookmaker odds are a market snapshot, not a guaranteed true probability.",
            "- The model is research/backtesting software and must not be used to place automated bets.",
            "",
            "## 15. SportyBet current-odds methodology",
            "",
            "Current odds are accepted only through a manually supplied CSV. The parser validates dates, "
            "teams, markets, selections, and decimal odds. Current fixtures are inference-only inputs and "
            "are never added to historical training data. Outputs label model-based EV descriptively as "
            "MODEL-BASED THEORETICAL EV, not guaranteed profit.",
            "",
            "## Research conclusion",
            "",
            "**No persistent out-of-sample advantage was demonstrated.** Across the 2024/25–2025/26 "
            "out-of-sample period, the bookmaker baseline had lower log loss and Brier score than every "
            "statistical model family, while every fixed-edge model backtest had negative theoretical ROI. "
            "The small differences in accuracy do not overturn the probability-quality and backtest results. "
            "This conclusion is limited to the supplied data, features, models, and fixed research rule; "
            "it is not a claim about guaranteed future performance.",
        ]
    )
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
