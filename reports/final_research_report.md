# English Premier League Quantitative Research Report

## 1. Dataset summary

- Historical matches after cleaning: **5700**
- Seasons: 2011/12 through 2025/26
- Teams observed: 39
- Authoritative source: `data/raw/epl_historical_clean.csv`
- Existing engineered files were inspected but not used as predictors.

## 2. Data-quality findings

- Flagged anomalies: 1
- The malformed record was excluded because it has no date, teams, result, score, statistics, or odds.
- Suspicious records were retained in the source and reported; no suspicious statistic was rewritten.
- Same-date fixtures are processed as a batch, preventing results from one fixture date entering another fixture on that date.

## 3. Leakage audit

- SAFE features: 80
- LEAKAGE features: 0
- UNKNOWN features: 0
- Only SAFE features were included in model fitting.

## 4. Feature methodology

Features are rebuilt from raw historical results, match statistics, prior bookmaker odds, head-to-head history, and a reconstructed season table. Rolling windows record the number of available historical matches rather than pretending short histories are full windows.

League position is reconstructed from results before the fixture date. H2H is restricted to earlier fixtures involving the same two teams. No final-season table, same-match statistic, future match, or legacy engineered feature is used.

## 5. Manual validation

See `exports/manual_feature_validation.md` for 20 chronologically distributed traces and independent rolling-average checks.

## 6. Chronological split

|split|rows|seasons|min_date|max_date|
|---|---|---|---|---|
|TRAINING|4180|2011/12, 2012/13, 2013/14, 2014/15, 2015/16, 2016/17, 2017/18, 2018/19, 2019/20, 2020/21, 2021/22|2011-08-13|2022-05-22|
|VALIDATION|760|2022/23, 2023/24|2022-08-05|2024-05-19|
|OUT-OF-SAMPLE|760|2024/25, 2025/26|2024-08-16|2026-05-24|

## 7. Model methodology

- Baseline A: historical result frequency from the training period.
- Baseline B: normalized bookmaker implied probability.
- Model C: logistic regression.
- Model D: random forest.
- Model E: gradient boosting.
- All probabilistic models use fixed parameters and a fixed random seed of 42.
- No random train/test split is used.

## 8. Baseline and model results

|split|model|rows|log_loss|brier_score|accuracy|
|---|---|---|---|---|---|
|OUT-OF-SAMPLE|bookmaker_probability|760|0.9947|0.5951|0.5145|
|OUT-OF-SAMPLE|gradient_boosting|760|1.0479|0.6280|0.4882|
|OUT-OF-SAMPLE|historical_frequency|760|1.0818|0.6554|0.4171|
|OUT-OF-SAMPLE|logistic_regression|760|1.0398|0.6175|0.5039|
|OUT-OF-SAMPLE|random_forest|760|1.0041|0.6012|0.5158|
|OUT-OF-SAMPLE_REFIT|gradient_boosting_train_validation|760|1.0362|0.6209|0.4855|
|OUT-OF-SAMPLE_REFIT|logistic_regression_train_validation|760|1.0328|0.6164|0.5000|
|OUT-OF-SAMPLE_REFIT|random_forest_train_validation|760|1.0066|0.6031|0.5105|
|TRAINING|bookmaker_probability|4180|0.9588|0.5672|0.5493|
|TRAINING|gradient_boosting|4180|0.9124|0.5381|0.5773|
|TRAINING|historical_frequency|4180|1.0658|0.6446|0.4462|
|TRAINING|logistic_regression|4180|0.9425|0.5579|0.5548|
|TRAINING|random_forest|4180|0.8493|0.4981|0.6091|
|VALIDATION|bookmaker_probability|760|0.9377|0.5533|0.5776|
|VALIDATION|gradient_boosting|760|0.9553|0.5615|0.5579|
|VALIDATION|historical_frequency|760|1.0523|0.6353|0.4724|
|VALIDATION|logistic_regression|760|0.9455|0.5576|0.5684|
|VALIDATION|random_forest|760|0.9408|0.5549|0.5605|

## 9. Validation results

Validation metrics are produced without looking at the out-of-sample test period. The fixed backtest rule was not tuned against test results.

## 10. Out-of-sample results

Out-of-sample results cover 2024/25 through 2025/26 and are chronologically isolated. The train-plus-validation refit is identified separately in the exported predictions.

## 11. Backtest results

This is a hypothetical 1-unit research backtest only. It is not a betting recommendation, does not connect to a bookmaker, and does not place bets. The one pre-declared strategy selects the maximum model-vs-normalized-market edge only when the edge is at least 0.05.

|split|model|strategy|selections|wins|losses|win_rate|average_odds|profit|roi|maximum_drawdown|longest_losing_streak|
|---|---|---|---|---|---|---|---|---|---|---|---|
|OUT-OF-SAMPLE|gradient_boosting|fixed_edge_0.05|547|203|344|0.3711|2.986745886654479|-108.28999999999999|-0.1980|-118.65000000000006|11|
|OUT-OF-SAMPLE|logistic_regression|fixed_edge_0.05|553|224|329|0.4051|2.7905063291139243|-70.02000000000001|-0.1266|-84.99000000000001|8|
|OUT-OF-SAMPLE|random_forest|fixed_edge_0.05|168|44|124|0.2619|4.644583333333333|-32.56999999999999|-0.1939|-31.570000000000007|13|
|OUT-OF-SAMPLE_REFIT|gradient_boosting_train_validation|fixed_edge_0.05|421|151|270|0.3587|2.8956532066508314|-72.58|-0.1724|-74.78000000000003|10|
|OUT-OF-SAMPLE_REFIT|logistic_regression_train_validation|fixed_edge_0.05|548|230|318|0.4197|2.6499452554744525|-77.91999999999999|-0.1422|-93.56|10|
|OUT-OF-SAMPLE_REFIT|random_forest_train_validation|fixed_edge_0.05|219|77|142|0.3516|3.36392694063927|-29.380000000000003|-0.1342|-43.61999999999999|11|
|TRAINING|gradient_boosting|fixed_edge_0.05|1320|798|522|0.6045|4.45234090909091|1140.65|0.8641|-10.970000000000027|7|
|TRAINING|logistic_regression|fixed_edge_0.05|1927|923|1004|0.4790|3.1861338868707834|460.37|0.2389|-28.33000000000002|10|
|TRAINING|random_forest|fixed_edge_0.05|1600|1473|127|0.9206|3.3152874999999993|2975.3|1.8596|-2.0|2|
|VALIDATION|gradient_boosting|fixed_edge_0.05|380|159|221|0.4184|3.838184210526315|-3.179999999999998|-0.0084|-25.150000000000013|9|
|VALIDATION|logistic_regression|fixed_edge_0.05|424|191|233|0.4505|2.892075471698113|-4.510000000000005|-0.0106|-24.799999999999997|9|
|VALIDATION|random_forest|fixed_edge_0.05|174|69|105|0.3966|4.3549425287356325|12.160000000000002|0.0699|-12.209999999999997|5|

Detailed results by season, market, probability range, and estimated edge range are in `backtests/`.

## 12. Calibration

Calibration tables are exported to `exports/calibration.csv`, with confidence-bin counts, mean predicted confidence, and observed accuracy.

## 13. Multiple Testing / Overfitting Risk

- Number of model families evaluated: 5, including two non-learning baselines.
- Number of backtest strategies evaluated: 1 fixed edge rule.
- No threshold sweep or test-period parameter selection was performed.
- Positive ROI in one period is not treated as evidence of a persistent advantage.

## 14. Limitations

- The source has historical 1X2 odds but no historical Over/Under 2.5 or BTTS odds.
- There is no xG, possession, lineup, injury, or player-availability data.
- Same-day ordering is unavailable, so the engine deliberately uses the earlier-date cutoff.
- Historical bookmaker odds are a market snapshot, not a guaranteed true probability.
- The model is research/backtesting software and must not be used to place automated bets.

## 15. SportyBet current-odds methodology

Current odds are accepted only through a manually supplied CSV. The parser validates dates, teams, markets, selections, and decimal odds. Current fixtures are inference-only inputs and are never added to historical training data. Outputs label model-based EV descriptively as MODEL-BASED THEORETICAL EV, not guaranteed profit.

## Research conclusion

**No persistent out-of-sample advantage was demonstrated.** Across the 2024/25–2025/26 out-of-sample period, the bookmaker baseline had lower log loss and Brier score than every statistical model family, while every fixed-edge model backtest had negative theoretical ROI. The small differences in accuracy do not overturn the probability-quality and backtest results. This conclusion is limited to the supplied data, features, models, and fixed research rule; it is not a claim about guaranteed future performance.
