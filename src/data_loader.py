from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .config import RAW_REQUIRED_COLUMNS, SAME_MATCH_STAT_COLUMNS


NUMERIC_COLUMNS = (
    "full_time_home_goals",
    "full_time_away_goals",
    "half_time_home_goals",
    "half_time_away_goals",
    "home_shots",
    "away_shots",
    "home_shots_on_target",
    "away_shots_on_target",
    "home_corners",
    "away_corners",
    "home_fouls",
    "away_fouls",
    "home_yellow_cards",
    "away_yellow_cards",
    "home_red_cards",
    "away_red_cards",
    "home_odds",
    "draw_odds",
    "away_odds",
)


@dataclass
class CleaningResult:
    frame: pd.DataFrame
    decisions: list[str] = field(default_factory=list)
    anomalies: list[dict[str, Any]] = field(default_factory=list)


def parse_date_strict(value: str | date | datetime) -> pd.Timestamp:
    """Parse only explicitly supported date formats.

    Historical Football-Data files use day-first dates. Current odds accept
    ISO dates as well as the same explicit day-first formats.
    """
    if isinstance(value, (date, datetime)):
        return pd.Timestamp(value).normalize()
    text = str(value).strip()
    if not text:
        raise ValueError("empty date")
    formats = ("%d/%m/%Y", "%d/%m/%y", "%Y-%m-%d")
    matches: list[pd.Timestamp] = []
    for fmt in formats:
        try:
            matches.append(pd.Timestamp(datetime.strptime(text, fmt)).normalize())
        except ValueError:
            continue
    unique = {item.value for item in matches}
    if len(unique) != 1:
        if not matches:
            raise ValueError(f"unsupported date format: {text!r}")
        raise ValueError(f"ambiguous date format: {text!r}")
    return matches[0]


def _is_blank_record(row: pd.Series) -> bool:
    return all(pd.isna(value) or str(value).strip() == "" for value in row)


def _audit_anomalies(frame: pd.DataFrame) -> list[dict[str, Any]]:
    anomalies: list[dict[str, Any]] = []
    for idx, row in frame.iterrows():
        def add(kind: str, detail: str) -> None:
            anomalies.append(
                {
                    "row_number": int(idx) + 2,
                    "match_id": row.get("match_id", ""),
                    "date": row.get("date", ""),
                    "home_team": row.get("home_team", ""),
                    "away_team": row.get("away_team", ""),
                    "type": kind,
                    "detail": detail,
                }
            )

        if any(row[col] < 0 for col in ("full_time_home_goals", "full_time_away_goals")):
            add("impossible_score", "Full-time goals cannot be negative.")
        if row["full_time_result"] not in {"H", "D", "A"}:
            add("invalid_result", f"Unexpected full-time result {row['full_time_result']!r}.")
        if row["full_time_result"] == "H" and row["full_time_home_goals"] <= row["full_time_away_goals"]:
            add("result_score_mismatch", "Result H does not agree with the full-time score.")
        if row["full_time_result"] == "D" and row["full_time_home_goals"] != row["full_time_away_goals"]:
            add("result_score_mismatch", "Result D does not agree with the full-time score.")
        if row["full_time_result"] == "A" and row["full_time_home_goals"] >= row["full_time_away_goals"]:
            add("result_score_mismatch", "Result A does not agree with the full-time score.")

        for odds_col in ("home_odds", "draw_odds", "away_odds"):
            if row[odds_col] <= 1:
                add("invalid_odds", f"{odds_col} must be greater than 1.")

        for shots, shots_on_target in (
            ("home_shots", "home_shots_on_target"),
            ("away_shots", "away_shots_on_target"),
        ):
            if row[shots_on_target] > row[shots]:
                add(
                    "impossible_statistical_relationship",
                    f"{shots_on_target} ({row[shots_on_target]}) exceeds {shots} ({row[shots]}).",
                )
        for col in SAME_MATCH_STAT_COLUMNS:
            if row[col] < 0:
                add("negative_match_statistic", f"{col} cannot be negative.")
    return anomalies


def load_and_clean(path: str | Path) -> CleaningResult:
    path = Path(path)
    raw = pd.read_csv(path, dtype=str, keep_default_na=False)
    required = set(RAW_REQUIRED_COLUMNS)
    missing_columns = sorted(required.difference(raw.columns))
    if missing_columns:
        raise ValueError(f"Historical source is missing required columns: {missing_columns}")

    decisions = [
        f"Loaded authoritative source {path}.",
        "Blank fields are not imputed. Rows missing core match identifiers are excluded.",
        "Historical dates are parsed with explicit day-first formats; no locale-inferred parsing is used.",
        "Suspicious statistical records are retained and reported, not silently corrected.",
        "Features will use only information from dates strictly before the fixture date.",
    ]
    raw = raw.apply(lambda col: col.str.strip() if col.dtype == "object" else col)
    blank_mask = raw.apply(_is_blank_record, axis=1)
    if blank_mask.any():
        decisions.append(f"Removed {int(blank_mask.sum())} fully blank row(s).")
        raw = raw.loc[~blank_mask].copy()

    core_missing = raw[list(RAW_REQUIRED_COLUMNS)].apply(
        lambda col: col.isna() | col.eq("")
    ).any(axis=1)
    if core_missing.any():
        decisions.append(
            f"Excluded {int(core_missing.sum())} malformed row(s) with missing core fields; "
            "the original values remain in the source file."
        )
        raw = raw.loc[~core_missing].copy()

    parsed_dates: list[pd.Timestamp] = []
    date_errors: list[str] = []
    for value in raw["date"]:
        try:
            parsed_dates.append(parse_date_strict(value))
        except ValueError as exc:
            parsed_dates.append(pd.NaT)
            date_errors.append(str(exc))
    if date_errors:
        raise ValueError(f"Could not parse historical dates: {date_errors[:5]}")
    raw["match_date"] = parsed_dates

    for col in NUMERIC_COLUMNS:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")
    numeric_missing = raw[list(NUMERIC_COLUMNS)].isna().any(axis=1)
    if numeric_missing.any():
        decisions.append(
            f"Excluded {int(numeric_missing.sum())} row(s) with non-numeric required values."
        )
        raw = raw.loc[~numeric_missing].copy()

    raw["match_id"] = (
        raw["season"].astype(str)
        + "|"
        + raw["match_date"].dt.strftime("%Y-%m-%d")
        + "|"
        + raw["home_team"]
        + "|"
        + raw["away_team"]
    )
    duplicate_mask = raw["match_id"].duplicated(keep=False)
    if duplicate_mask.any():
        decisions.append(
            f"Found {int(duplicate_mask.sum())} records participating in duplicate match keys; "
            "duplicates are retained for review and excluded from model exports."
        )
    else:
        decisions.append("No duplicate match keys were found.")

    raw = raw.sort_values(["match_date", "season", "match_id"]).reset_index(drop=True)
    anomalies = _audit_anomalies(raw)
    decisions.append(f"Retained {len(raw)} valid historical matches after structural cleaning.")
    decisions.append(
        f"Flagged {len(anomalies)} suspicious conditions without modifying their source values."
    )
    return CleaningResult(frame=raw, decisions=decisions, anomalies=anomalies)


def load_current_odds(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    expected = ["date", "home_team", "away_team", "market", "selection", "odds"]
    missing = [col for col in expected if col not in frame.columns]
    if missing:
        raise ValueError(f"Current odds file is missing required columns: {missing}")
    frame = frame[expected].copy()
    for col in ("home_team", "away_team", "market", "selection"):
        frame[col] = frame[col].astype(str).str.strip()
    parsed: list[pd.Timestamp] = []
    for value in frame["date"]:
        parsed.append(parse_date_strict(value))
    frame["match_date"] = parsed
    frame["odds"] = pd.to_numeric(frame["odds"], errors="coerce")
    if frame["odds"].isna().any() or (frame["odds"] <= 1).any():
        raise ValueError("Current odds contain missing or invalid decimal odds.")
    if frame[["home_team", "away_team", "market", "selection"]].eq("").any().any():
        raise ValueError("Current odds contain blank fixture or market fields.")
    return frame
