# CAROB Fine-Tuning Diagnosis: Benin and Tanzania

Date: 2026-02-26  
Context: This note documents why RMSE ~900 was not feasible under the locked CAROB fine-tuning setup.

## Locked Setup
- Model: CatBoost champion from `outputs/carob_model_tune/model_winners.csv`
- Scenario: `modifiable_plus_context` (17 features)
- Split: trial-aware 80/20
- Seeds: `42, 52, 62`

## Main Findings
- Error concentration is high. Top 10% of predictions explain 53.9% of SSE and top 20% explain 74.6%.
- Country error is highly concentrated. Benin and Tanzania together explain about 85.0% of SSE.
- Country-specific residual correction helped but was not enough. RMSE improved from 1083.40 to 1064.01 (delta -19.38), still far from 900.
- Benin and Tanzania have high within-country and within-trial target spread.
- Actionable levers are sparse in those countries because many modifiable features are constant.
- Tanzania has major context missingness (for example soil variables), which likely hides key yield drivers.
- Benin shows duplicate-feature inconsistency: same observed feature vector maps to very different yields.

## Quantitative Evidence
- Baseline overall: `R2=0.6076`, `RMSE=1083.40`, `MAE=798.96`
- Country residual correction overall: `R2=0.6201`, `RMSE=1064.01`, `MAE=788.72`
- Benin error share: `44.56%` SSE
- Tanzania error share: `41.34%` SSE
- Benin modifiable constancy: `4/7` constant (`57.14%`)
- Tanzania modifiable constancy: `5/7` constant (`71.43%`)
- Tanzania missingness examples: `soil_P=100%`, `soil_pH=72.16%`, `N_fertilizer=45.36%`, `K_fertilizer=45.36%`, `row_spacing=50%`
- Benin duplicate feature combos: `12` combos, `36` rows (`6.56%` of Benin rows), max yield range inside same combo: `7642`

## Interpretation
- The model is not only under-tuned. The data signal appears limited in the two highest-error countries.
- Large outcome spread remains after controlling for currently available features.
- For Benin/Tanzania, several modifiable variables do not vary enough to explain trial-level yield differences.
- Missing or unobserved drivers are likely significant, especially in Tanzania.
- This is consistent with regression-to-mean behavior and persistent tail errors.

## What This Means for Feasibility
- RMSE ~900 is unlikely under the current locked feature set and split.
- Country-specific residual correction is directionally useful but only closes a small part of the gap.
- Additional gains likely require better country-specific signal, stronger trial-level context, or revised evaluation targets by country segment.

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
