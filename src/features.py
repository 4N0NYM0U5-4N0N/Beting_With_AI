from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .config import RESULTS
from .odds import add_historical_odds_features


ROLLING_WINDOWS = (5, 10)
ROLLING_METRICS = (
    "goals_scored",
    "goals_conceded",
    "goal_difference",
    "points",
    "shots",
    "shots_on_target",
    "corners",
    "yellow_cards",
)


@dataclass
class FeatureBuildResult:
    frame: pd.DataFrame
    feature_dictionary: list[dict[str, str]]
    validation_traces: list[dict[str, Any]]


def _empty_team_state() -> dict[str, list[dict[str, float]]]:
    return {}


def _team_history(state: dict[str, list[dict[str, float]]], team: str) -> list[dict[str, float]]:
    return state.setdefault(team, [])


def _mean(history: list[dict[str, float]], metric: str, window: int | None = None) -> float:
    values = [item[metric] for item in history[-window:] if metric in item] if window else [
        item[metric] for item in history if metric in item
    ]
    return float(np.mean(values)) if values else np.nan


def _count(history: list[dict[str, float]], window: int | None = None) -> int:
    return len(history[-window:]) if window else len(history)


def _result_counts(history: list[dict[str, float]]) -> tuple[int, int, int]:
    wins = sum(item["result"] == 3 for item in history)
    draws = sum(item["result"] == 1 for item in history)
    losses = sum(item["result"] == 0 for item in history)
    return wins, draws, losses


def _state_features(
    state: dict[str, list[dict[str, float]]],
    team: str,
    prefix: str,
    dictionary: list[dict[str, str]],
) -> dict[str, float]:
    history = _team_history(state, team)
    output: dict[str, float] = {}
    count = _count(history)
    wins, draws, losses = _result_counts(history)
    aggregates = {
        "matches_played": count,
        "points_per_match": _mean(history, "points"),
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "goals_scored": sum(item["goals_scored"] for item in history),
        "goals_conceded": sum(item["goals_conceded"] for item in history),
        "goal_difference": sum(item["goal_difference"] for item in history),
    }
    for name, value in aggregates.items():
        key = f"{prefix}_overall_{name}"
        output[key] = float(value) if value is not None else np.nan
        dictionary.append(_metadata(key, "historical match results and statistics", "cumulative aggregate"))

    for window in ROLLING_WINDOWS:
        n_key = f"{prefix}_rolling{window}_matches_available"
        output[n_key] = float(_count(history, window))
        dictionary.append(_metadata(n_key, "historical matches before fixture", f"count of last {window} available matches"))
        for metric in ROLLING_METRICS:
            key = f"{prefix}_rolling{window}_{metric}"
            output[key] = _mean(history, metric, window)
            dictionary.append(_metadata(key, "historical match statistics", f"mean of last {window} available matches"))
    return output


def _venue_features(
    state: dict[str, list[dict[str, float]]],
    team: str,
    venue: str,
    prefix: str,
    dictionary: list[dict[str, str]],
) -> dict[str, float]:
    history = [item for item in _team_history(state, team) if item["venue"] == venue]
    output: dict[str, float] = {
        f"{prefix}_{venue}_matches_played": float(len(history)),
        f"{prefix}_{venue}_points_per_match": _mean(history, "points"),
        f"{prefix}_{venue}_goals_scored": _mean(history, "goals_scored"),
        f"{prefix}_{venue}_goals_conceded": _mean(history, "goals_conceded"),
        f"{prefix}_{venue}_goal_difference": _mean(history, "goal_difference"),
    }
    for key in output:
        dictionary.append(_metadata(key, f"prior {venue}-only matches for team", "venue-restricted cumulative mean"))
    return output


def _metadata(name: str, source: str, method: str) -> dict[str, str]:
    return {
        "feature_name": name,
        "source_columns": source,
        "calculation_method": method,
        "information_cutoff": "strictly before current fixture date",
        "same_match_information_can_enter": "NO",
        "future_matches_can_enter": "NO",
        "classification": "SAFE",
    }


def _h2h_features(
    h2h: dict[tuple[str, str], list[dict[str, float]]],
    home: str,
    away: str,
    dictionary: list[dict[str, str]],
) -> dict[str, float]:
    matches = h2h.get(tuple(sorted((home, away))), [])
    home_results = [item for item in matches if item["home_team"] == home]
    output = {
        "h2h_matches_played": float(len(matches)),
        "h2h_home_win_rate": float(np.mean([item["result"] == "H" for item in home_results]))
        if home_results
        else np.nan,
        "h2h_draw_rate": float(np.mean([item["result"] == "D" for item in home_results]))
        if home_results
        else np.nan,
        "h2h_away_win_rate": float(np.mean([item["result"] == "A" for item in home_results]))
        if home_results
        else np.nan,
        "h2h_home_perspective_goal_difference": float(
            np.mean([item["goal_difference"] if item["home_team"] == home else -item["goal_difference"] for item in matches])
        )
        if matches
        else np.nan,
    }
    for key in output:
        dictionary.append(_metadata(key, "prior head-to-head results", "mean/rate over prior same-team fixtures"))
    return output


def _league_positions(
    table: dict[str, dict[str, dict[str, float]]],
    season: str,
    home: str,
    away: str,
    dictionary: list[dict[str, str]],
) -> dict[str, float]:
    standings = table.setdefault(season, {})
    ranking = sorted(
        standings.items(),
        key=lambda item: (
            -item[1]["points"],
            -(item[1]["goals_for"] - item[1]["goals_against"]),
            -item[1]["goals_for"],
            item[0],
        ),
    )
    positions = {team: float(index + 1) for index, (team, _) in enumerate(ranking)}
    output = {
        "home_league_position_before_match": positions.get(home, np.nan),
        "away_league_position_before_match": positions.get(away, np.nan),
        "league_position_difference_before_match": (
            positions.get(away, np.nan) - positions.get(home, np.nan)
            if home in positions and away in positions
            else np.nan
        ),
    }
    for key in output:
        dictionary.append(_metadata(key, "prior results in current season", "table sorted by points, goal difference, goals scored"))
    return output


def _summary(row: pd.Series, perspective: str) -> dict[str, float]:
    is_home = perspective == "home"
    gf = float(row["full_time_home_goals"] if is_home else row["full_time_away_goals"])
    ga = float(row["full_time_away_goals"] if is_home else row["full_time_home_goals"])
    result = row["full_time_result"]
    points = 3.0 if (result == "H" and is_home) or (result == "A" and not is_home) else 1.0 if result == "D" else 0.0
    return {
        "goals_scored": gf,
        "goals_conceded": ga,
        "goal_difference": gf - ga,
        "points": points,
        "result": points,
        "shots": float(row["home_shots"] if is_home else row["away_shots"]),
        "shots_on_target": float(row["home_shots_on_target"] if is_home else row["away_shots_on_target"]),
        "corners": float(row["home_corners"] if is_home else row["away_corners"]),
        "yellow_cards": float(row["home_yellow_cards"] if is_home else row["away_yellow_cards"]),
        "venue": perspective,
    }


def _update_table(
    table: dict[str, dict[str, dict[str, float]]], row: pd.Series
) -> None:
    season = str(row["season"])
    season_table = table.setdefault(season, {})
    home = season_table.setdefault(row["home_team"], {"points": 0.0, "goals_for": 0.0, "goals_against": 0.0})
    away = season_table.setdefault(row["away_team"], {"points": 0.0, "goals_for": 0.0, "goals_against": 0.0})
    hg, ag = float(row["full_time_home_goals"]), float(row["full_time_away_goals"])
    home["goals_for"] += hg
    home["goals_against"] += ag
    away["goals_for"] += ag
    away["goals_against"] += hg
    if row["full_time_result"] == "H":
        home["points"] += 3
    elif row["full_time_result"] == "A":
        away["points"] += 3
    else:
        home["points"] += 1
        away["points"] += 1


def _update_h2h(
    h2h: dict[tuple[str, str], list[dict[str, float]]], row: pd.Series
) -> None:
    key = tuple(sorted((row["home_team"], row["away_team"])))
    h2h.setdefault(key, []).append(
        {
            "home_team": row["home_team"],
            "away_team": row["away_team"],
            "result": row["full_time_result"],
            "goal_difference": float(row["full_time_home_goals"]) - float(row["full_time_away_goals"]),
        }
    )


def _build(
    rows: pd.DataFrame,
    include_current: bool = False,
) -> FeatureBuildResult:
    state = _empty_team_state()
    table: dict[str, dict[str, dict[str, float]]] = {}
    h2h: dict[tuple[str, str], list[dict[str, float]]] = {}
    output_rows: list[dict[str, Any]] = []
    feature_dictionary: list[dict[str, str]] = []
    traces: list[dict[str, Any]] = []

    work = rows.copy()
    if "_is_current" not in work:
        work["_is_current"] = False
    work = work.sort_values(["match_date", "season", "match_id"]).reset_index(drop=True)
    selected_trace_positions = set(np.linspace(0, len(work) - 1, min(20, len(work)), dtype=int)) if len(work) else set()

    for position, (_, row) in enumerate(work.iterrows()):
        dictionary: list[dict[str, str]] = []
        home_prefix, away_prefix = "home_team", "away_team"
        feature_row: dict[str, Any] = {
            "match_id": row["match_id"],
            "season": row["season"],
            "match_date": row["match_date"].strftime("%Y-%m-%d"),
            "date": row.get("date", row["match_date"].strftime("%Y-%m-%d")),
            "home_team": row["home_team"],
            "away_team": row["away_team"],
            "_is_current": bool(row["_is_current"]),
        }
        if not row["_is_current"]:
            feature_row.update(
                {
                    "full_time_result": row["full_time_result"],
                    "full_time_home_goals": float(row["full_time_home_goals"]),
                    "full_time_away_goals": float(row["full_time_away_goals"]),
                    "home_odds": float(row["home_odds"]),
                    "draw_odds": float(row["draw_odds"]),
                    "away_odds": float(row["away_odds"]),
                }
            )
        feature_row.update(_state_features(state, row["home_team"], home_prefix, dictionary))
        feature_row.update(_state_features(state, row["away_team"], away_prefix, dictionary))
        feature_row.update(_venue_features(state, row["home_team"], "home", home_prefix, dictionary))
        feature_row.update(_venue_features(state, row["away_team"], "away", away_prefix, dictionary))
        feature_row.update(_h2h_features(h2h, row["home_team"], row["away_team"], dictionary))
        feature_row.update(_league_positions(table, str(row["season"]), row["home_team"], row["away_team"], dictionary))

        if "home_odds" in row and pd.notna(row.get("home_odds")) and float(row["home_odds"]) > 1:
            odds_frame = pd.DataFrame(
                [{"home_odds": row["home_odds"], "draw_odds": row["draw_odds"], "away_odds": row["away_odds"]}]
            )
            odds_features = add_historical_odds_features(odds_frame).iloc[0].to_dict()
            feature_row.update(odds_features)
            for name in odds_features:
                dictionary.append(_metadata(name, "historical bookmaker odds", "decimal odds converted to implied and normalized probabilities"))

        if position in selected_trace_positions:
            home_history = _team_history(state, row["home_team"])
            away_history = _team_history(state, row["away_team"])
            traces.append(
                {
                    "match_id": row["match_id"],
                    "season": row["season"],
                    "date": row["match_date"].strftime("%Y-%m-%d"),
                    "home_team": row["home_team"],
                    "away_team": row["away_team"],
                    "home_previous_matches": home_history[-5:],
                    "away_previous_matches": away_history[-5:],
                    "home_previous_goals_average": _mean(home_history, "goals_scored", 5),
                    "away_previous_goals_average": _mean(away_history, "goals_scored", 5),
                }
            )

        output_rows.append(feature_row)
        if not row["_is_current"]:
            state.setdefault(row["home_team"], []).append(_summary(row, "home"))
            state.setdefault(row["away_team"], []).append(_summary(row, "away"))
            _update_table(table, row)
            _update_h2h(h2h, row)
        feature_dictionary.extend(dictionary)

    frame = pd.DataFrame(output_rows)
    if not frame.empty:
        frame = frame.drop(columns=["_is_current"], errors="ignore")
    unique_dictionary = {item["feature_name"]: item for item in feature_dictionary}
    return FeatureBuildResult(frame=frame, feature_dictionary=list(unique_dictionary.values()), validation_traces=traces)


def build_features(frame: pd.DataFrame) -> FeatureBuildResult:
    return _build(frame, include_current=False)


def build_current_features(historical: pd.DataFrame, current: pd.DataFrame) -> FeatureBuildResult:
    # Current fixtures are inference-only. Build each fixture against a
    # historical prefix strictly earlier than its date so a historical result
    # on the same calendar date cannot enter the current fixture's features.
    # This deliberately leaves build_features() and the historical methodology
    # unchanged.
    historic = historical.copy()
    historic["match_date"] = pd.to_datetime(historic["match_date"], errors="raise")
    current_rows = current.copy()
    current_rows["match_date"] = pd.to_datetime(current_rows["match_date"], errors="raise")
    all_frames: list[pd.DataFrame] = []
    all_dictionary: dict[str, dict[str, str]] = {}
    all_traces: list[dict[str, Any]] = []

    for _, current_row in current_rows.iterrows():
        prefix = historic[historic["match_date"] < current_row["match_date"]].copy()
        one_current = current_row.to_frame().T.copy()
        one_current["_is_current"] = True
        for col in (
            "full_time_result",
            "full_time_home_goals",
            "full_time_away_goals",
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
        ):
            if col not in one_current:
                one_current[col] = np.nan
        prefix["_is_current"] = False
        result = _build(pd.concat([prefix, one_current], ignore_index=True, sort=False), include_current=True)
        current_id = current_row["match_id"]
        current_frame = result.frame[result.frame["match_id"] == current_id]
        all_frames.append(current_frame)
        all_traces.extend(result.validation_traces)
        for item in result.feature_dictionary:
            all_dictionary[item["feature_name"]] = item

    frame = pd.concat(all_frames, ignore_index=True) if all_frames else pd.DataFrame()
    return FeatureBuildResult(
        frame=frame,
        feature_dictionary=list(all_dictionary.values()),
        validation_traces=all_traces,
    )


def predictor_columns(frame: pd.DataFrame) -> list[str]:
    excluded = {
        "match_id",
        "season",
        "match_date",
        "date",
        "home_team",
        "away_team",
        "full_time_result",
        "full_time_home_goals",
        "full_time_away_goals",
    }
    return [
        column
        for column in frame.columns
        if column not in excluded and pd.api.types.is_numeric_dtype(frame[column])
    ]
