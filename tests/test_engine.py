import unittest

import pandas as pd

from src.data_loader import parse_date_strict
from src.odds import add_historical_odds_features


class EngineTests(unittest.TestCase):
    def test_date_parsing_is_explicit(self):
        self.assertEqual(parse_date_strict("13/08/11"), pd.Timestamp("2011-08-13"))
        self.assertEqual(parse_date_strict("2026-09-25"), pd.Timestamp("2026-09-25"))
        with self.assertRaises(ValueError):
            parse_date_strict("not-a-date")

    def test_odds_overround_and_normalization(self):
        frame = add_historical_odds_features(
            pd.DataFrame([{"home_odds": 2.0, "draw_odds": 4.0, "away_odds": 4.0}])
        )
        self.assertAlmostEqual(frame.loc[0, "bookmaker_overround"], 1.0)
        self.assertAlmostEqual(
            frame.loc[0, "bookmaker_prob_home"]
            + frame.loc[0, "bookmaker_prob_draw"]
            + frame.loc[0, "bookmaker_prob_away"],
            1.0,
        )


if __name__ == "__main__":
    unittest.main()
