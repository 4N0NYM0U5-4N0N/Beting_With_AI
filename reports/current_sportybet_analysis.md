# Current SportyBet Odds Analysis

This is a descriptive research analysis. It applies the existing historical feature definitions and fixed model procedures to manually supplied current odds.
It does not log in, scrape, place, submit, or automate bets.

- Fixtures analyzed: 1
- Input rows: 3
- Invalid input rows: 0
- Historical methodology changed: NO
- Current fixtures used for model training: NO
- Same-date historical results used: NO

## Man United vs Arsenal

- Date: 2026-09-25
- Market: 1X2
- Bookmaker overround: 1.036818

### Market probabilities

| Selection | Odds | Raw implied probability | Normalized market probability |
|---|---:|---:|---:|
| A | 2.8500 | 0.350877 | 0.338417 |
| D | 3.6000 | 0.277778 | 0.267914 |
| H | 2.4500 | 0.408163 | 0.393669 |

### Model probabilities and theoretical comparison

| Model | H | D | A | Most probable | Calibration information |
|---|---:|---:|---:|---|---|
| Logistic Regression | 0.345304 | 0.244300 | 0.410396 | A | OOS confidence bucket 0.4-0.5; observed accuracy 0.3828 from 209 matches |
| Random Forest | 0.390425 | 0.337901 | 0.271674 | H | OOS confidence bucket 0.3-0.4; observed accuracy 0.4622 from 119 matches |
| Gradient Boosting | 0.308124 | 0.432202 | 0.259674 | D | OOS confidence bucket 0.4-0.5; observed accuracy 0.4433 from 194 matches |

| Selection | Odds | Model | Probability | Probability difference | THEORETICAL MODEL EV |
|---|---:|---|---:|---:|---:|
| A | 2.8500 | Logistic Regression | 0.410396 | 0.071979 | 0.169629 |
| A | 2.8500 | Random Forest | 0.271674 | -0.066743 | -0.225729 |
| A | 2.8500 | Gradient Boosting | 0.259674 | -0.078743 | -0.259930 |
| D | 3.6000 | Logistic Regression | 0.244300 | -0.023614 | -0.120520 |
| D | 3.6000 | Random Forest | 0.337901 | 0.069988 | 0.216444 |
| D | 3.6000 | Gradient Boosting | 0.432202 | 0.164288 | 0.555928 |
| H | 2.4500 | Logistic Regression | 0.345304 | -0.048365 | -0.154005 |
| H | 2.4500 | Random Forest | 0.390425 | -0.003244 | -0.043459 |
| H | 2.4500 | Gradient Boosting | 0.308124 | -0.085545 | -0.245096 |

### Model agreement and market comparison

- Logistic Regression most probable outcome: A
- Random Forest most probable outcome: H
- Gradient Boosting most probable outcome: D
- Agreement status: DISAGREEMENT
- Market highest normalized probability: H

### Data-quality warnings

- 6 feature values require model imputation

Calibration uses the existing validation/OOS confidence-bucket results. It is historical context, not a confidence interval or guarantee.

## Interpretation limits

A positive model-market probability difference or THEORETICAL MODEL EV is conditional on the model and market assumptions. It is not a guaranteed return, recommendation, or evidence that a fixture will win.
