# Current Analysis Imputation Audit

## Scope

This audit covers the synthetic fixture analyzed from `current_sportybet_odds.csv`:

- Date: `2026-09-25`
- Fixture: `Man United` vs `Arsenal`
- Current feature-row missing values: **6**
- Actual missing model predictors reported by the diagnostic: **3**

The current feature row contains six `NaN` values. Three are current-match target
fields that are intentionally absent and are not model predictors. The diagnostic
excludes those fields and reports only the three missing league-position predictors.
Those predictors are absent because the current fixture is assigned the `CURRENT`
season and no results have been processed into a `CURRENT` season table.

| feature | missing_reason | imputation_method | imputed_value | training_consistency | leakage_risk |
|---|---|---|---:|---|---|
| `full_time_result` | The current fixture has no known final result. Current fixtures are inference-only. | Not imputed; excluded from `predictor_columns` and used only as a historical training target. | N/A | Consistent. Historical rows provide this label; it is never a predictor input. | None. No current or future result is used. |
| `full_time_home_goals` | The current fixture has no known final home-goal total. | Not imputed; excluded from `predictor_columns`. | N/A | Consistent. Historical goal targets are excluded from model inputs. | None. No current or future score is used. |
| `full_time_away_goals` | The current fixture has no known final away-goal total. | Not imputed; excluded from `predictor_columns`. | N/A | Consistent. Historical goal targets are excluded from model inputs. | None. No current or future score is used. |
| `home_league_position_before_match` | The current row uses season `CURRENT`; its season table has no prior processed results, so the home position is undefined. | `SimpleImputer(strategy="median", add_indicator=True)` in the existing model pipeline. | `11.0` | Consistent. Historical model training uses the same median imputer. The chronological training-split median for this feature is also `11.0`. | None identified. The median is derived from historical training data only. |
| `away_league_position_before_match` | The current row uses season `CURRENT`; its season table has no prior processed results, so the away position is undefined. | `SimpleImputer(strategy="median", add_indicator=True)` in the existing model pipeline. | `11.0` | Consistent. Historical model training uses the same median imputer. The chronological training-split median for this feature is also `11.0`. | None identified. The median is derived from historical training data only. |
| `league_position_difference_before_match` | It is calculated only when both current-season league positions exist; both positions are undefined, so the difference is undefined. | `SimpleImputer(strategy="median", add_indicator=True)` in the existing model pipeline. | `-1.0` | Consistent. Historical model training uses the same median imputer. The chronological training-split median for this feature is also `-1.0`. | None identified. The median is derived from historical training data only. |

## Training and leakage confirmation

- The historical training path uses the same `SimpleImputer(strategy="median", add_indicator=True)` through the existing model pipelines.
- Logistic Regression, Random Forest, and Gradient Boosting all use that pipeline behavior.
- The current-analysis models also use that existing pipeline behavior; no current fixture result, future result, or current odds outcome is used to calculate the imputed values.
- The three league-position medians in the chronological training split are `11.0`, `11.0`, and `-1.0`, matching the values used for the synthetic current fixture.
- The three missing `full_time_*` fields are not imputed at all because they are target/non-predictor fields.

## Classification

**EXPECTED — after diagnostic correction.**

The underlying imputation of the three missing league-position predictors is
**EXPECTED** for a newly analyzed fixture whose current-season table has no
historical prefix. The diagnostic now reports **3 predictor values require model
imputation**. It separately states that `full_time_result`,
`full_time_home_goals`, and `full_time_away_goals` are unavailable because the
fixture has not occurred yet; these fields are not predictors and are not counted
as imputation warnings. No methodology, model, feature definition, historical
output, or current analysis calculation was changed as part of this correction.