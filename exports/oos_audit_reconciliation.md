# OOS Audit Reconciliation

This report compares the compact audit export with the already-generated `backtests/backtest_summary.csv`. No models were retrained and no selections were regenerated.

| Model | Existing selections | Export selections | Existing ROI | Export ROI | Match |
|---|---:|---:|---:|---:|---|
| gradient_boosting | 547 | 547.0 | -0.197970749543 | -0.197970749543 | YES |
| logistic_regression | 553 | 553.0 | -0.126618444846 | -0.126618444846 | YES |
| random_forest | 168 | 168.0 | -0.193869047619 | -0.193869047619 | YES |

## Full metric comparison

| Model | Metric | Existing | Export | Match |
|---|---|---:|---:|---|
| gradient_boosting | selections | 547 | 547.0 | YES |
| gradient_boosting | wins | 203 | 203.0 | YES |
| gradient_boosting | losses | 344 | 344.0 | YES |
| gradient_boosting | win_rate | 0.3711151736745887 | 0.3711151736745887 | YES |
| gradient_boosting | total_profit | -108.29 | -108.28999999999999 | YES |
| gradient_boosting | roi | -0.1979707495429616 | -0.1979707495429616 | YES |
| gradient_boosting | average_odds | 2.986745886654479 | 2.986745886654479 | YES |
| gradient_boosting | maximum_drawdown | -118.65000000000006 | -118.65000000000006 | YES |
| gradient_boosting | longest_losing_streak | 11 | 11.0 | YES |
| logistic_regression | selections | 553 | 553.0 | YES |
| logistic_regression | wins | 224 | 224.0 | YES |
| logistic_regression | losses | 329 | 329.0 | YES |
| logistic_regression | win_rate | 0.4050632911392405 | 0.4050632911392405 | YES |
| logistic_regression | total_profit | -70.02000000000001 | -70.02 | YES |
| logistic_regression | roi | -0.1266184448462929 | -0.12661844484629295 | YES |
| logistic_regression | average_odds | 2.7905063291139243 | 2.7905063291139243 | YES |
| logistic_regression | maximum_drawdown | -84.99000000000001 | -84.99000000000001 | YES |
| logistic_regression | longest_losing_streak | 8 | 8.0 | YES |
| random_forest | selections | 168 | 168.0 | YES |
| random_forest | wins | 44 | 44.0 | YES |
| random_forest | losses | 124 | 124.0 | YES |
| random_forest | win_rate | 0.2619047619047619 | 0.2619047619047619 | YES |
| random_forest | total_profit | -32.56999999999999 | -32.56999999999999 | YES |
| random_forest | roi | -0.1938690476190475 | -0.19386904761904758 | YES |
| random_forest | average_odds | 4.644583333333333 | 4.644583333333333 | YES |
| random_forest | maximum_drawdown | -31.570000000000007 | -31.570000000000007 | YES |
| random_forest | longest_losing_streak | 13 | 13.0 | YES |

## Validation

- Export rows: 1268
- Models: gradient_boosting, logistic_regression, random_forest
- Date range: 2024-08-16 to 2026-05-24
- Duplicate fixture/model/selection combinations: 0
- Validation status: PASS

## Discrepancies

No discrepancies. The export reproduces the existing OOS backtest metrics.
