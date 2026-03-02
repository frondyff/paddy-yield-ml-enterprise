# CAROB Country-Error Diagnosis (Benin and Tanzania)

Date: 2026-03-02  
Purpose: document country-level error concentration findings used to guide tuning expectations.

## Current Pipeline Context
- Primary tuning pipeline: `src/paddy_yield_ml/pipelines/carob_model_tune_top2.py`
- Primary scenario: `modifiable_plus_context` (17 features)
- Current locked test results from `outputs/carob_model_tune_top2/model_winners.csv`:
  - `ExtraTrees`: `R2=0.4792`, `RMSE=1002.02`, `MAE=749.09`
  - `CatBoost`: `R2=0.4792`, `RMSE=1002.08`, `MAE=740.98`

Interpretation:
- Under the current train/validation/test protocol, both contenders land near RMSE ~1002.
- This diagnosis explains why pushing substantially below that range is difficult.

## Diagnostic Track Used
The detailed country-error root-cause artifacts were produced in dedicated stress-test runs:
- `outputs/carob_rmse_feasibility_diagnostics/*`
- `outputs/carob_country_residual_correction/*`
- `outputs/carob_country_root_cause/*`

These diagnostics are supplemental to model selection, not a replacement for the locked test protocol.

## Main Findings
- Error concentration is high:
  - top 10% predictions explain `53.9%` of SSE
  - top 20% predictions explain `74.6%` of SSE
- Country error concentration is extreme:
  - Benin SSE share: `44.56%`
  - Tanzania SSE share: `41.34%`
  - Combined: about `85.0%` of SSE
- Country residual correction helped, but only modestly:
  - baseline RMSE mean: `1083.40`
  - country residual correction RMSE mean: `1064.01`
  - delta: `-19.38`
- Signal limitations in top-error countries:
  - Benin modifiable constancy: `4/7` constant (`57.14%`)
  - Tanzania modifiable constancy: `5/7` constant (`71.43%`)
  - Tanzania has severe missingness in key context fields (`soil_P`, `soil_pH`, etc.)
  - Benin shows duplicate-feature inconsistency (same feature vector, wide yield spread)

## Practical Implication
- Remaining error is not only a tuning issue; much of it appears tied to sparse or inconsistent country-level signal.
- Incremental tuning may still help, but large RMSE step-downs are unlikely without additional reliable country-level predictors or cleaner context coverage.

## Supporting Artifacts
- `outputs/carob_rmse_feasibility_diagnostics/champion_seed_metrics.csv`
- `outputs/carob_rmse_feasibility_diagnostics/error_concentration_summary.csv`
- `outputs/carob_rmse_feasibility_diagnostics/error_by_country.csv`
- `outputs/carob_country_residual_correction/summary.csv`
- `outputs/carob_country_residual_correction/country_summary.csv`
- `outputs/carob_country_root_cause/country_target_profile.csv`
- `outputs/carob_country_root_cause/focus_trial_target_profile.csv`
- `outputs/carob_country_root_cause/focus_constancy_summary_by_role.csv`
- `outputs/carob_country_root_cause/country_feature_missingness_constancy.csv`
- `outputs/carob_country_root_cause/country_duplicate_feature_summary.csv`
- `outputs/carob_country_root_cause/focus_trial_treatment_profile.csv`
- `outputs/carob_country_root_cause/focus_trial_treatment_diff.csv`
