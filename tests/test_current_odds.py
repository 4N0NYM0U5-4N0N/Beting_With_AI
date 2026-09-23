import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.current_odds import (
    ANALYSIS_COLUMNS,
    _quality_warnings,
    run_current_analysis,
    validate_current_odds,
)
from src.features import build_current_features


def _write_odds(path: Path, rows: list[dict[str, object]]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


class CurrentOddsTests(unittest.TestCase):
    def _complete_rows(self) -> list[dict[str, object]]:
        return [
            {
                "date": "2026-09-25",
                "home_team": "Manchester United",
                "away_team": "Arsenal",
                "market": "1X2",
                "selection": selection,
                "odds": odds,
            }
            for selection, odds in (("H", 2.45), ("D", 3.60), ("A", 2.85))
        ]

    def test_alias_mapping_and_market_math(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "odds.csv"
            _write_odds(path, self._complete_rows())
            validation = validate_current_odds(path)

        self.assertEqual(validation.valid_fixtures, 1)
        self.assertEqual(set(validation.normalized["home_team"]), {"Man United"})
        self.assertEqual(set(validation.normalized["away_team"]), {"Arsenal"})
        fixture = validation.normalized
        self.assertAlmostEqual(float(fixture["raw_implied_probability"].sum()), 1.0368182360663563)
        self.assertAlmostEqual(float(fixture["normalized_market_probability"].sum()), 1.0)

    def test_duplicate_missing_and_unknown_teams_are_reported(self):
        rows = self._complete_rows()
        rows[1]["selection"] = "H"
        rows[2]["home_team"] = "Unknown FC"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "odds.csv"
            _write_odds(path, rows)
            validation = validate_current_odds(path)

        self.assertEqual(validation.valid_fixtures, 0)
        self.assertTrue(any("duplicate" in error for errors in validation.fixture_errors.values() for error in errors))
        self.assertTrue(any("unknown home team" in error for errors in validation.row_errors.values() for error in errors))

    def test_missing_selection_and_invalid_odds_are_not_silently_repaired(self):
        rows = self._complete_rows()[:2]
        rows[1]["odds"] = 1
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "odds.csv"
            _write_odds(path, rows)
            validation = validate_current_odds(path)

        self.assertEqual(validation.valid_fixtures, 0)
        self.assertTrue(any("odds must be" in error for errors in validation.row_errors.values() for error in errors))
        self.assertTrue(any("missing selections" in error for errors in validation.fixture_errors.values() for error in errors))

    def test_current_features_exclude_same_date_historical_results(self):
        historical = pd.DataFrame(
            [
                {
                    "match_id": "old",
                    "season": "2025/26",
                    "match_date": pd.Timestamp("2026-09-24"),
                    "date": "2026-09-24",
                    "home_team": "Man United",
                    "away_team": "Chelsea",
                    "full_time_result": "H",
                    "full_time_home_goals": 1,
                    "full_time_away_goals": 0,
                    "home_odds": 2.0,
                    "draw_odds": 3.0,
                    "away_odds": 4.0,
                    "home_shots": 10,
                    "away_shots": 5,
                    "home_shots_on_target": 4,
                    "away_shots_on_target": 2,
                    "home_corners": 5,
                    "away_corners": 2,
                    "home_fouls": 10,
                    "away_fouls": 10,
                    "home_yellow_cards": 1,
                    "away_yellow_cards": 1,
                    "home_red_cards": 0,
                    "away_red_cards": 0,
                },
                {
                    "match_id": "same-date",
                    "season": "2025/26",
                    "match_date": pd.Timestamp("2026-09-25"),
                    "date": "2026-09-25",
                    "home_team": "Man United",
                    "away_team": "Arsenal",
                    "full_time_result": "A",
                    "full_time_home_goals": 0,
                    "full_time_away_goals": 2,
                    "home_odds": 2.0,
                    "draw_odds": 3.0,
                    "away_odds": 4.0,
                    "home_shots": 8,
                    "away_shots": 8,
                    "home_shots_on_target": 2,
                    "away_shots_on_target": 4,
                    "home_corners": 3,
                    "away_corners": 4,
                    "home_fouls": 10,
                    "away_fouls": 9,
                    "home_yellow_cards": 1,
                    "away_yellow_cards": 2,
                    "home_red_cards": 0,
                    "away_red_cards": 0,
                },
            ]
        )
        current = pd.DataFrame(
            [
                {
                    "match_id": "current",
                    "season": "CURRENT",
                    "match_date": pd.Timestamp("2026-09-25"),
                    "date": "2026-09-25",
                    "home_team": "Man United",
                    "away_team": "Arsenal",
                    "home_odds": 2.4,
                    "draw_odds": 3.5,
                    "away_odds": 2.9,
                }
            ]
        )
        built = build_current_features(historical, current).frame.iloc[0]
        self.assertEqual(float(built["home_team_overall_matches_played"]), 1.0)

    def test_target_fields_do_not_count_as_predictor_imputation(self):
        feature_row = pd.Series(
            {
                "full_time_result": np.nan,
                "full_time_home_goals": np.nan,
                "full_time_away_goals": np.nan,
                "home_league_position_before_match": np.nan,
                "away_league_position_before_match": np.nan,
                "league_position_difference_before_match": np.nan,
                "home_team_overall_matches_played": 5.0,
                "away_team_overall_matches_played": 5.0,
                "home_team_rolling5_matches_available": 5.0,
                "away_team_rolling5_matches_available": 5.0,
                "h2h_matches_played": 1.0,
            }
        )
        warnings = _quality_warnings(
            feature_row,
            {
                "logistic_regression": np.array([0.4, 0.3, 0.3]),
                "random_forest": np.array([0.4, 0.3, 0.3]),
                "gradient_boosting": np.array([0.4, 0.3, 0.3]),
            },
            [],
            [
                "home_league_position_before_match",
                "away_league_position_before_match",
                "league_position_difference_before_match",
            ],
        )
        self.assertIn("3 predictor values require model imputation", warnings)
        self.assertNotIn("6 predictor values require model imputation", warnings)

    def test_analysis_outputs_all_models_probability_sums_and_ev(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "odds.csv"
            _write_odds(path, self._complete_rows())
            result = run_current_analysis(path)
            output = pd.read_csv(result["analysis_path"])

        self.assertEqual(list(output.columns), ANALYSIS_COLUMNS)
        self.assertEqual(len(output), 3)
        for prefix in ("logistic", "random_forest", "gradient_boosting"):
            probabilities = output[f"{prefix}_probability"].astype(float)
            self.assertTrue((probabilities >= 0).all())
            self.assertAlmostEqual(float(probabilities.sum()), 1.0, places=8)
        first = output.iloc[0]
        for prefix in ("logistic", "random_forest", "gradient_boosting"):
            probability = float(first[f"{prefix}_probability"])
            market = float(first["normalized_market_probability"])
            odds = float(first["odds"])
            self.assertAlmostEqual(float(first[f"{prefix}_probability_difference"]), probability - market)
            self.assertAlmostEqual(float(first[f"{prefix}_theoretical_ev"]), probability * odds - 1)


if __name__ == "__main__":
    unittest.main()