from __future__ import annotations

import pandas as pd


def add_historical_odds_features(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    odds = output[["home_odds", "draw_odds", "away_odds"]].astype(float)
    implied = 1 / odds
    output["bookmaker_implied_home"] = implied["home_odds"]
    output["bookmaker_implied_draw"] = implied["draw_odds"]
    output["bookmaker_implied_away"] = implied["away_odds"]
    output["bookmaker_overround"] = implied.sum(axis=1)
    normalized = implied.div(implied.sum(axis=1), axis=0)
    output["bookmaker_prob_home"] = normalized["home_odds"]
    output["bookmaker_prob_draw"] = normalized["draw_odds"]
    output["bookmaker_prob_away"] = normalized["away_odds"]
    return output


def odds_probability_columns() -> list[str]:
    return [
        "bookmaker_implied_home",
        "bookmaker_implied_draw",
        "bookmaker_implied_away",
        "bookmaker_overround",
        "bookmaker_prob_home",
        "bookmaker_prob_draw",
        "bookmaker_prob_away",
    ]


def validate_current_market(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["market"] = output["market"].str.upper().str.replace(" ", "", regex=False)
    output["selection"] = output["selection"].str.strip()
    supported = {"1X2"}
    output["market_supported"] = output["market"].isin(supported)
    group_cols = ["match_date", "home_team", "away_team", "market"]
    output["market_implied_probability"] = 1 / output["odds"]
    overround = output.groupby(group_cols)["market_implied_probability"].transform("sum")
    output["market_overround"] = overround
    output["market_normalized_probability"] = (
        output["market_implied_probability"] / output["market_overround"]
    )
    return output
