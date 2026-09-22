# EPL Quantitative Research Engine

Python-only, reproducible research software for testing whether historical English Premier League
statistics contain pre-match signal that survives strict chronological out-of-sample evaluation.

This project is research/backtesting only. It does not scrape SportyBet, connect to a betting account,
place bets, or claim guaranteed winnings.

## Setup

```bash
uv sync
```

Or install the pinned scientific dependencies from `requirements.txt`.

The authoritative input is:

```text
data/raw/epl_historical_clean.csv
```

The legacy engineered files in `data/raw/` are retained for audit context only and are never used as
predictors.

## Commands

```bash
python -m src.main audit
python -m src.main build_features
python -m src.main leakage_audit
python -m src.main validate_features
python -m src.main backtest
python -m src.main full_analysis
python -m src.main analyze_current current_sportybet_odds.csv
```

Run the full reproducible pipeline with:

```bash
python -m src.main full_analysis
```

It cleans the source, rebuilds pre-match features, audits leakage, creates manual spot checks,
trains fixed-parameter baseline models, evaluates chronological validation and out-of-sample periods,
runs a fixed one-unit theoretical backtest, and writes reports and datasets.

## Chronological design

- Training: 2011/12 through 2021/22
- Validation: 2022/23 through 2023/24
- Out-of-sample: 2024/25 through 2025/26

Features use only matches on dates strictly before each fixture. Since kickoff times are unavailable,
all results on the same date are withheld from every other fixture on that date.

## Outputs

- `exports/cleaning_report.md`
- `exports/leakage_audit.md`
- `exports/feature_dictionary.csv`
- `exports/manual_feature_validation.md`
- `exports/epl_pre_match_features_v2.csv`
- `exports/model_predictions.csv`
- `exports/model_performance.csv`
- `exports/calibration.csv`
- `backtests/`
- `reports/final_research_report.md`

Current odds are manually supplied using `current_sportybet_odds.csv.example`. Current fixtures remain
inference-only and are never mixed into historical training data.
