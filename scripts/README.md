# Scripts Guide (Student-Friendly)

This folder has simple "runner" scripts.

Think of them like shortcut buttons:
- each one runs one full pipeline stage
- you do not need to remember long module paths
- everyone on the team runs the same steps in the same way

## Why we wrote these scripts
- Reproducibility: same command, same process, fewer accidental differences.
- Team collaboration: easier handoff between teammates.
- Debugging: if something breaks, we know exactly which stage caused it.
- Presentation readiness: every stage writes outputs to a clear folder.

## How to run
From project root:

```bash
uv run python scripts/run_baseline.py
uv run python scripts/run_feature_prepare.py
uv run python scripts/run_model_compare.py
uv run python scripts/run_model_select_tune.py --run-tag dual_eval
uv run python scripts/run_ablation_eval.py --run-tag weather_location_ablation
uv run python scripts/run_interpretability_report.py --run-tag milestone_interpretability_v1
```

## Script-by-script: what each one does

### `run_baseline.py`
- Runs first-pass baseline pipeline.
- Purpose: quick sanity check and starting benchmark.

### `run_feature_prepare.py`
- Cleans and prepares data for modeling.
- Builds/uses feature roles (modifiable, context, proxy).
- Flags leakage/proxy risks.
- Creates hybrid feature candidate list.

### `run_model_compare.py`
- Compares multiple models and feature-set combinations.
- Uses group-aware validation.
- Helps pick a strong model family before heavy tuning.

### `run_model_select_tune.py`
- Performs stronger tuning and scenario testing.
- Uses strict `LOGO` as primary validation.
- Also reports secondary `GroupShuffle` and `RandomShuffle` for context.

### `run_ablation_eval.py`
- Tests hypothesis-driven feature scenarios.
- In our case: checked whether adding weather/water and location helps.

### `run_interpretability_report.py`
- Generates SHAP global and local explanations.
- Extracts modifiable-only decision rules.
- Generates recommendation draft and 1-page milestone summary.

## What results we got (latest snapshot)

These are from the latest output folders in this repo.

### 1) Model comparison stage
From `outputs/model_compare/model_comparison_summary.csv`:
- Best row: `hybrid_with_review + CatBoost`
- R2: `0.4991`
- RMSE: `204.22`
- MAE: `161.03`

### 2) Tuning + strict evaluation stage
From `outputs/model_select_tune/dual_eval/`:
- Best config: `full_review + CatBoost`
- LOGO: `R2=0.5063`, `RMSE=202.69`, `MAE=159.56`
- GroupShuffle: `R2=0.5077`
- RandomShuffle: `R2=0.5066`

Interpretation:
- Performance is stable across strict and secondary splits.
- Current realistic ceiling with this data is around `R2 ~ 0.51`.

### 3) Ablation stage (hypothesis test)
From `outputs/ablation_eval/weather_location_ablation/ablation_combined_summary.csv`:
- Conservative base feature set performed best.
- Adding weather/water features reduced R2.
- Adding weather/water plus location did not recover enough.

Interpretation:
- We revoked the "add weather/water/location to improve model" hypothesis.

### 4) Interpretability stage
From `outputs/interpretability/milestone_interpretability_v1/`:
- Global SHAP top features:
  - `Pest_60Day(in ml)`
  - `Seedrate(in Kg)`
  - `Micronutrients_70Days`
  - `Weed28D_thiobencarb`
  - `Potassh_50Days`
- Exported `5` decision rules using only modifiable numeric features.
- Example strong rule:
  - `Weed28D_thiobencarb > 7 AND LP_nurseryarea(in Tonnes) <= 5.5 AND Micronutrients_70Days > 67.5`
  - Support: about `29.3%` of rows
  - Positive yield lift in this dataset

Interpretation:
- Model explanations are mainly tied to modifiable inputs, which is useful for recommendations.

## Important caution
- SHAP explains model behavior, not true cause-and-effect.
- Current dataset has no explicit date/season field.
- So these findings are strong predictive signals, but causal claims need extra data or experiments.

## Quick output map
- Baseline and prep: `outputs/feature_prepare/`
- Model comparison: `outputs/model_compare/`
- Tuning and dual evaluation: `outputs/model_select_tune/dual_eval/`
- Hypothesis ablation: `outputs/ablation_eval/weather_location_ablation/`
- Interpretability package: `outputs/interpretability/milestone_interpretability_v1/`
