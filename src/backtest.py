from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import RESULTS


EDGE_THRESHOLD = 0.05
STRATEGY_NAME = "fixed_edge_0.05"


def _selection_rows(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in predictions.iterrows():
        probs = {"H": float(row["prob_H"]), "D": float(row["prob_D"]), "A": float(row["prob_A"])}
        odds = {"H": float(row["home_odds"]), "D": float(row["draw_odds"]), "A": float(row["away_odds"])}
        market = {
            "H": 1 / odds["H"],
            "D": 1 / odds["D"],
            "A": 1 / odds["A"],
        }
        total = sum(market.values())
        market = {key: value / total for key, value in market.items()}
        selection = max(RESULTS, key=lambda label: probs[label] - market[label])
        edge = probs[selection] - market[selection]
        if edge < EDGE_THRESHOLD:
            continue
        won = row["actual_result"] == selection
        profit = odds[selection] - 1 if won else -1
        rows.append(
            {
                "split": row["split"],
                "model": row["model"],
                "strategy": STRATEGY_NAME,
                "match_id": row["match_id"],
                "season": row["season"],
                "selection": selection,
                "model_probability": probs[selection],
                "market_probability": market[selection],
                "estimated_edge": edge,
                "odds": odds[selection],
                "actual_result": row["actual_result"],
                "won": won,
                "profit": profit,
            }
        )
    return pd.DataFrame(rows)


def _max_drawdown(profits: pd.Series) -> float:
    if profits.empty:
        return 0.0
    cumulative = profits.cumsum()
    peak = cumulative.cummax()
    return float((cumulative - peak).min())


def _longest_losing_streak(profits: pd.Series) -> int:
    longest = current = 0
    for value in profits:
        if value < 0:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def summarize_backtest(selections: pd.DataFrame, predictions: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    if selections.empty:
        empty = pd.DataFrame(
            columns=[
                "split", "model", "strategy", "selections", "wins", "losses",
                "win_rate", "average_odds", "profit", "roi", "maximum_drawdown",
                "longest_losing_streak",
            ]
        )
        return empty, {"by_season": empty.copy(), "by_market": empty.copy(), "by_probability": empty.copy(), "by_edge": empty.copy()}

    summary_rows: list[dict[str, Any]] = []
    for (split, model, strategy), group in selections.groupby(["split", "model", "strategy"]):
        summary_rows.append(
            {
                "split": split,
                "model": model,
                "strategy": strategy,
                "selections": len(group),
                "wins": int(group["won"].sum()),
                "losses": int((~group["won"]).sum()),
                "win_rate": float(group["won"].mean()),
                "average_odds": float(group["odds"].mean()),
                "profit": float(group["profit"].sum()),
                "roi": float(group["profit"].sum() / len(group)),
                "maximum_drawdown": _max_drawdown(group["profit"]),
                "longest_losing_streak": _longest_losing_streak(group["profit"]),
            }
        )
    summary = pd.DataFrame(summary_rows)

    def grouped(column: str, labels: pd.Series) -> pd.DataFrame:
        local = selections.copy()
        local["_bucket"] = labels.to_numpy()
        rows = []
        for (split, model, bucket), group in local.groupby(["split", "model", "_bucket"], dropna=False):
            rows.append(
                {
                    "split": split,
                    "model": model,
                    "group": str(bucket),
                    "selections": len(group),
                    "wins": int(group["won"].sum()),
                    "profit": float(group["profit"].sum()),
                    "roi": float(group["profit"].sum() / len(group)),
                }
            )
        return pd.DataFrame(rows)

    details = {
        "by_season": grouped("season", selections["season"]),
        "by_market": grouped("selection", selections["selection"]),
        "by_probability": grouped(
            "probability",
            pd.cut(
                selections["model_probability"],
                bins=[0, 0.4, 0.5, 0.6, 0.7, 1.0],
                include_lowest=True,
            ),
        ),
        "by_edge": grouped(
            "edge",
            pd.cut(
                selections["estimated_edge"],
                bins=[0.05, 0.10, 0.15, 0.25, 1.0],
                include_lowest=True,
            ),
        ),
    }
    return summary, details


def run_backtest(predictions: pd.DataFrame, output_dir: str | Path) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], pd.DataFrame]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    modeled = predictions[predictions["model"].str.contains("logistic|forest|boosting", regex=True)].copy()
    selections = _selection_rows(modeled)
    summary, details = summarize_backtest(selections, predictions)
    selections.to_csv(output_dir / "backtest_selections.csv", index=False)
    summary.to_csv(output_dir / "backtest_summary.csv", index=False)
    for name, frame in details.items():
        frame.to_csv(output_dir / f"{name}.csv", index=False)
    return summary, details, selections
