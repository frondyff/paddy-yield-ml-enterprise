# CAROB Rice Decision Support Using Interpretable Enterprise ML

McGill MMA Enterprise Analytics Group Project

## Project direction
This repository now uses **CAROB (doi:10.7910/DVN/AMAZXA)** as the primary case for the decision-support framework.
The former paddy dataset pipelines are kept only as legacy reference and are no longer the core analysis path.

## Enterprise objective
Build a reproducible decision-support workflow that can:
- predict rice yield under trial/site variability,
- explain predictions with transparent feature logic,
- produce actionable recommendations with confidence/caveat labels,
- support causal stretch-goal diagnostics where data structure allows.

## CAROB core pipelines
- `src/paddy_yield_ml/pipelines/carob_baseline.py`
- `src/paddy_yield_ml/pipelines/carob_feature_prepare.py`
- `src/paddy_yield_ml/pipelines/carob_model_compare.py`
- `src/paddy_yield_ml/pipelines/carob_interpretability.py`
- `src/paddy_yield_ml/pipelines/carob_rule_causal_aipw.py`

## Inputs (CAROB)
- `data/input/carob_amazxa.csv`
- `data/metadata/carob_amazxa_meta.csv`
- `data/metadata/data_dictionary_carob_amazxa.csv`

## Outputs (CAROB)
- `outputs/carob_baseline/`
- `outputs/carob_feature_prepare/`
- `outputs/carob_model_compare/`
- `outputs/carob_interpretability/`
- `outputs/carob_rule_causal_aipw/rule_aipw_v2/`

## Validation philosophy (CAROB)
- Predictive robustness: Leave-One-Trial-Out grouped evaluation.
- Feature governance: role-based screening (modifiable/context/proxy) + redundancy review.
- Causal stretch goal: rule-as-treatment AIPW diagnostics with trial-aware uncertainty checks.

## Setup
```bash
uv sync --all-groups
```

## Run CAROB pipelines
```bash
uv run python src/paddy_yield_ml/pipelines/carob_baseline.py
uv run python src/paddy_yield_ml/pipelines/carob_feature_prepare.py
uv run python src/paddy_yield_ml/pipelines/carob_model_compare.py
uv run python src/paddy_yield_ml/pipelines/carob_interpretability.py --run-tag iter3_defensible_v5
uv run python src/paddy_yield_ml/pipelines/carob_rule_causal_aipw.py --run-tag rule_aipw_v2
```

Wrappers:
```bash
python scripts/run_carob_baseline.py
python scripts/run_carob_feature_prepare.py
python scripts/run_carob_model_compare.py
python scripts/run_carob_interpretability.py --run-tag iter3_defensible_v5
python scripts/run_carob_rule_causal_aipw.py --run-tag rule_aipw_v2
```

Make targets:
```bash
make run-carob-baseline
make run-carob-feature-prepare
make run-carob-model-compare
make run-carob-interpretability
make run-carob-rule-causal-aipw
make help
```

## Legacy (paddy) scope
Paddy pipelines remain in the repository as legacy reference and compatibility paths.
They are not the primary analysis flow and should not be used as default project entrypoints.

Legacy wrappers and pipelines include:
- `scripts/run_baseline.py`
- `scripts/run_feature_prepare.py`
- `scripts/run_model_compare.py`
- `scripts/run_model_select_tune.py`
- `scripts/run_ablation_eval.py`
- `scripts/run_interpretability_report.py`
- `scripts/run_causal_pilot.py`

## Quality checks
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
