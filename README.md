# Paddy Yield Decision Support Using Enterprise Machine Learning

McGill MMA Enterprise Analytics Group Project

## Enterprise context
Agricultural organizations need reliable yield forecasting and resource planning under uncertain weather and field conditions. This project builds an enterprise-ready, reproducible ML workflow for paddy yield analysis and decision support.

## Goals
- Identify key yield drivers with interpretable analysis.
- Prepare a robust feature set for hybrid feature selection.
- Support model comparison, strict/secondary validation, and actionable recommendations with reproducible pipelines and quality checks.

## Core pipelines
- `src/paddy_yield_ml/pipelines/baseline.py`
- `src/paddy_yield_ml/pipelines/feature_prepare.py`
- `src/paddy_yield_ml/pipelines/model_compare.py`
- `src/paddy_yield_ml/pipelines/model_select_tune.py`
- `src/paddy_yield_ml/pipelines/ablation_eval.py`
- `src/paddy_yield_ml/pipelines/interpretability_report.py`

## Data and outputs
- Inputs:
  - `data/input/paddydataset.csv`
  - `data/metadata/data_dictionary_paddy.csv`
- Outputs:
  - `outputs/baseline/`
  - `outputs/feature_prepare/`
  - `outputs/model_compare/`
  - `outputs/model_select_tune/`
  - `outputs/ablation_eval/`
  - `outputs/interpretability/`

## Validation philosophy
- Primary metric: `Leave-One-Agriblock-Out (LOGO)` for cross-agriblock generalization.
- Secondary metrics: `GroupShuffle` and `RandomShuffle` for operational context.
- Proxy/leakage controls are strict; interpretability findings are associative (not causal).

## Current milestone snapshot
- Best model family so far: `CatBoost`.
- Best strict result (LOGO): around `R2 ~ 0.506` on `full_review` feature set.
- Weather/water+location ablation did not improve over conservative base feature set.
- Interpretability package delivered:
  - global SHAP importance,
  - local SHAP examples,
  - modifiable-only decision rules,
  - recommendation draft and 1-page milestone summary.

## Project layout
```text
paddy-yield-ml-enterprise/
  data/
    input/
    metadata/
  outputs/
    baseline/
    feature_prepare/
    model_compare/
    model_select_tune/
    ablation_eval/
    interpretability/
  scripts/
  src/paddy_yield_ml/pipelines/
  tests/
  Makefile
  pyproject.toml
```

## Setup and run
```bash
uv sync --all-groups
uv run python src/paddy_yield_ml/pipelines/baseline.py
uv run python src/paddy_yield_ml/pipelines/feature_prepare.py
uv run python src/paddy_yield_ml/pipelines/model_compare.py
uv run python src/paddy_yield_ml/pipelines/model_select_tune.py --run-tag dual_eval
uv run python src/paddy_yield_ml/pipelines/ablation_eval.py --run-tag weather_location_ablation
uv run python src/paddy_yield_ml/pipelines/interpretability_report.py --run-tag milestone_interpretability_v1
```

Wrappers:
```bash
python scripts/run_baseline.py
python scripts/run_feature_prepare.py
python scripts/run_model_compare.py
python scripts/run_model_select_tune.py --run-tag dual_eval
python scripts/run_ablation_eval.py --run-tag weather_location_ablation
python scripts/run_interpretability_report.py --run-tag milestone_interpretability_v1
```

For a walkthrough of script intent and results, see:
- `scripts/README.md`

## Quality and automation
```bash
make lint
make format
make typecheck
make test
make verify
```

## Team
- Abdelaziz Ahmed
- Frondy Ferdianto
- Hazel Guan
- Muhammad Hydarali
- Simmi Agnihotram
