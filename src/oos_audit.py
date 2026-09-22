from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import BACKTESTS_DIR, EXPORTS_DIR, PROCESSED_DIR


AUDIT_COLUMNS = [
    "date",
    "season",
    "home_team",
    "away_team",
    "actual_result",
    "model",
    "split",
    "selected_outcome",
    "model_probability",
    "market_probability",
    "odds",
    "edge",
    "profit",
]

OOS_SPLIT = "OUT-OF-SAMPLE"


def _max_drawdown(profits: pd.Series) -> float:
    cumulative = profits.cumsum()
    return float((cumulative - cumulative.cummax()).min()) if not profits.empty else 0.0


def _longest_losing_streak(profits: pd.Series) -> int:
    longest = current = 0
    for profit in profits:
        if profit < 0:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def _aggregate(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model, group in frame.groupby("model", sort=True):
        wins = group["actual_result"].eq(group["selected_outcome"])
        profit = group["profit"].astype(float)
        rows.append(
            {
                "model": model,
                "selections": int(len(group)),
                "wins": int(wins.sum()),
                "losses": int((~wins).sum()),
                "win_rate": float(wins.mean()),
                "total_profit": float(profit.sum()),
                "roi": float(profit.sum() / len(group)),
                "average_odds": float(group["odds"].astype(float).mean()),
                "maximum_drawdown": _max_drawdown(profit),
                "longest_losing_streak": _longest_losing_streak(profit),
            }
        )
    return pd.DataFrame(rows)


def _validate_export(
    audit: pd.DataFrame,
    source_selections: pd.DataFrame,
    source_summary: pd.DataFrame,
) -> list[str]:
    errors: list[str] = []
    if list(audit.columns) != AUDIT_COLUMNS:
        errors.append(f"Columns do not match the required order: {list(audit.columns)}")
    if set(audit["split"].unique()) != {OOS_SPLIT}:
        errors.append("Export contains a split other than OUT-OF-SAMPLE.")
    if not set(audit["season"].unique()).issubset({"2024/25", "2025/26"}):
        errors.append("Export contains a season outside 2024/25 and 2025/26.")
    duplicate_columns = ["season", "date", "home_team", "away_team", "model", "selected_outcome"]
    if audit.duplicated(duplicate_columns).any():
        errors.append("Duplicate fixture/model/selection combinations exist.")
    for column in ("actual_result", "selected_outcome"):
        if not audit[column].isin({"H", "D", "A"}).all():
            errors.append(f"{column} contains values outside H/D/A.")
    for column in ("model_probability", "market_probability"):
        values = audit[column].astype(float)
        if not values.between(0, 1).all():
            errors.append(f"{column} contains probabilities outside [0, 1].")
    if not (audit["odds"].astype(float) > 1).all():
        errors.append("Odds must be greater than 1.")

    source = source_selections[
        (source_selections["split"] == OOS_SPLIT)
        & source_selections["model"].isin(set(audit["model"]))
    ].reset_index(drop=True)
    if len(source) != len(audit):
        errors.append("Export row count does not match existing OUT-OF-SAMPLE selections.")
    if not np.allclose(
        audit["edge"].astype(float).to_numpy(),
        audit["model_probability"].astype(float).to_numpy()
        - audit["market_probability"].astype(float).to_numpy(),
        atol=1e-12,
    ):
        errors.append("Edge values do not equal the existing model-minus-market values.")
    expected_profit = np.where(
        audit["actual_result"].eq(audit["selected_outcome"]),
        audit["odds"].astype(float) - 1,
        -1,
    )
    if not np.allclose(audit["profit"].astype(float).to_numpy(), expected_profit, atol=1e-12):
        errors.append("Profit values do not match the existing 1-unit profit rule.")

    actual_models = set(audit["model"])
    expected_models = set(source_summary["model"])
    if actual_models != expected_models:
        errors.append(f"Model set differs from existing OOS summary: {actual_models} vs {expected_models}.")
    return errors


def create_oos_audit(
    selections_path: str | Path = BACKTESTS_DIR / "backtest_selections.csv",
    summary_path: str | Path = BACKTESTS_DIR / "backtest_summary.csv",
    match_path: str | Path = PROCESSED_DIR / "historical_cleaned.csv",
    output_path: str | Path = EXPORTS_DIR / "oos_backtest_audit.csv",
    reconciliation_path: str | Path = EXPORTS_DIR / "oos_audit_reconciliation.md",
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Export only already-selected OOS rows; never retrains or reselects."""
    selections = pd.read_csv(selections_path)
    summary = pd.read_csv(summary_path)
    matches = pd.read_csv(match_path, usecols=["match_id", "date", "home_team", "away_team"])

    existing_oos_summary = summary[summary["split"] == OOS_SPLIT].copy()
    existing_oos_selections = selections[selections["split"] == OOS_SPLIT].copy()
    if existing_oos_selections.empty:
        raise ValueError("Existing backtest output contains no OUT-OF-SAMPLE selections.")

    audit = existing_oos_selections.merge(
        matches,
        on="match_id",
        how="left",
        validate="many_to_one",
        sort=False,
    )
    if audit[["date", "season", "home_team", "away_team"]].isna().any().any():
        raise ValueError("Could not map every existing OOS selection to its historical fixture.")
    audit = audit.rename(
        columns={
            "selection": "selected_outcome",
            "estimated_edge": "edge",
        }
    )[AUDIT_COLUMNS]
    audit.to_csv(output_path, index=False)

    export_summary = _aggregate(audit)
    existing_by_model = existing_oos_summary.set_index("model")
    export_by_model = export_summary.set_index("model")
    reconciliation_rows: list[dict[str, object]] = []
    discrepancy_lines: list[str] = []
    numeric_columns = [
        "selections",
        "wins",
        "losses",
        "win_rate",
        "total_profit",
        "roi",
        "average_odds",
        "maximum_drawdown",
        "longest_losing_streak",
    ]
    for model in sorted(set(existing_by_model.index).union(export_by_model.index)):
        existing = existing_by_model.loc[model] if model in existing_by_model.index else None
        exported = export_by_model.loc[model] if model in export_by_model.index else None
        row: dict[str, object] = {"model": model}
        matches_model = existing is not None and exported is not None
        differences: list[str] = []
        for column in numeric_columns:
            existing_value = (
                existing["profit"] if column == "total_profit" and existing is not None
                else existing[column] if existing is not None else np.nan
            )
            export_value = exported[column] if exported is not None else np.nan
            row[f"existing_{column}"] = existing_value
            row[f"export_{column}"] = export_value
            if existing is None or exported is None or not np.isclose(float(existing_value), float(export_value), atol=1e-9):
                matches_model = False
                differences.append(
                    f"{column}: existing={existing_value}, export={export_value}"
                )
        row["match"] = "YES" if matches_model else "NO"
        reconciliation_rows.append(row)
        if differences:
            discrepancy_lines.append(f"- **{model}** — " + "; ".join(differences))

    reconciliation = pd.DataFrame(reconciliation_rows)
    validation_errors = _validate_export(audit, selections, existing_oos_summary)
    if validation_errors:
        discrepancy_lines.extend(f"- Validation error: {error}" for error in validation_errors)

    display_columns = [
        "model",
        "existing_selections",
        "export_selections",
        "existing_roi",
        "export_roi",
        "match",
    ]
    report_lines = [
        "# OOS Audit Reconciliation",
        "",
        "This report compares the compact audit export with the already-generated "
        "`backtests/backtest_summary.csv`. No models were retrained and no selections were regenerated.",
        "",
        "| Model | Existing selections | Export selections | Existing ROI | Export ROI | Match |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for _, row in reconciliation[display_columns].iterrows():
        report_lines.append(
            f"| {row['model']} | {row['existing_selections']} | {row['export_selections']} | "
            f"{float(row['existing_roi']):.12f} | {float(row['export_roi']):.12f} | {row['match']} |"
        )
    report_lines.extend(
        [
            "",
            "## Full metric comparison",
            "",
            "| Model | Metric | Existing | Export | Match |",
            "|---|---|---:|---:|---|",
        ]
    )
    for _, row in reconciliation.iterrows():
        for column in numeric_columns:
            report_lines.append(
                f"| {row['model']} | {column} | {row[f'existing_{column}']} | "
                f"{row[f'export_{column}']} | "
                f"{'YES' if row['match'] == 'YES' else 'SEE DISCREPANCY'} |"
            )
    report_lines.extend(
        [
            "",
            "## Validation",
            "",
            f"- Export rows: {len(audit)}",
            f"- Models: {', '.join(sorted(audit['model'].unique()))}",
            f"- Date range: {pd.to_datetime(audit['date'], dayfirst=True).min().date()} to {pd.to_datetime(audit['date'], dayfirst=True).max().date()}",
            f"- Duplicate fixture/model/selection combinations: {int(audit.duplicated(['season', 'date', 'home_team', 'away_team', 'model', 'selected_outcome']).sum())}",
            f"- Validation status: {'PASS' if not validation_errors else 'FAIL'}",
            "",
            "## Discrepancies",
            "",
        ]
    )
    report_lines.extend(discrepancy_lines or ["No discrepancies. The export reproduces the existing OOS backtest metrics."])
    Path(reconciliation_path).write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    # Read both artifacts back so the command verifies they are usable files.
    pd.read_csv(output_path)
    Path(reconciliation_path).read_text(encoding="utf-8")
    return audit, reconciliation, validation_errors
