from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = ROOT / "reports"
AUDITS_DIR = REPORTS_DIR / "audits"
BACKTESTS_DIR = ROOT / "backtests"
MODELS_DIR = ROOT / "models"
EXPORTS_DIR = ROOT / "exports"

HISTORICAL_PATH = RAW_DIR / "epl_historical_clean.csv"
CURRENT_ODDS_PATH = ROOT / "current_sportybet_odds.csv"

TRAIN_SEASONS = tuple(f"{year}/{str(year + 1)[-2:]}" for year in range(2011, 2022))
VALIDATION_SEASONS = ("2022/23", "2023/24")
TEST_SEASONS = ("2024/25", "2025/26")

RESULTS = ("H", "D", "A")
TARGET_COLUMNS = (
    "full_time_result",
    "full_time_home_goals",
    "full_time_away_goals",
)
SAME_MATCH_STAT_COLUMNS = (
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
)

RAW_REQUIRED_COLUMNS = (
    "season",
    "date",
    "home_team",
    "away_team",
    "full_time_home_goals",
    "full_time_away_goals",
    "full_time_result",
    "home_odds",
    "draw_odds",
    "away_odds",
)

STAT_PAIRS = (
    ("home_shots", "away_shots"),
    ("home_shots_on_target", "away_shots_on_target"),
    ("home_corners", "away_corners"),
    ("home_fouls", "away_fouls"),
    ("home_yellow_cards", "away_yellow_cards"),
    ("home_red_cards", "away_red_cards"),
)
