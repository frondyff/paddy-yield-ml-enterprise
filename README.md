# Paddy Yield Decision Support Using Enterprise Machine Learning

McGill MMA Enterprise Analytics Group Project

## Enterprise context
Agricultural organizations need reliable yield forecasting and resource planning under uncertain weather and field conditions. This project builds an enterprise-ready, reproducible ML workflow for paddy yield analysis and decision support.

## Goals
- Identify key yield drivers with interpretable analysis.
- Prepare a robust feature set for hybrid feature selection.
- Support model comparison with reproducible pipelines and quality checks.

## Core pipelines
- `src/paddy_yield_ml/pipelines/baseline.py`

- `src/paddy_yield_ml/pipelines/feature_prepare.py`

- `src/paddy_yield_ml/pipelines/model_compare.py`


## Data and outputs
- Inputs:
  - `data/input/paddydataset.csv`
  - `data/metadata/data_dictionary_paddy.csv`
- Outputs:
  - `outputs/baseline/`
  - `outputs/feature_prepare/`

## Project layout
```text
paddy-yield-ml-enterprise/
  data/
    input/
    metadata/
  outputs/
    baseline/
    feature_prepare/
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
```

Wrappers:
```bash
python scripts/run_baseline.py
python scripts/run_feature_prepare.py
python scripts/run_model_compare.py
```

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
