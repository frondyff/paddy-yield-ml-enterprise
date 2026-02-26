# Scripts Guide (CAROB-first)

These runner scripts are shortcut buttons for the CAROB workflow.

## Primary run order
From project root:

```bash
uv run python scripts/run_carob_baseline.py
uv run python scripts/run_carob_feature_prepare.py
uv run python scripts/run_carob_model_compare.py
uv run python scripts/run_carob_causal_pilot.py --run-tag v1
```

## What each script does

### `run_carob_baseline.py`
- Loads CAROB AMAZXA data.
- Produces first-pass EDA, trial profile, and baseline grouped metrics.

### `run_carob_feature_prepare.py`
- Performs sectioned EDA for trial/country/treatment structure.
- Builds actionability roles (modifiable/context/proxy).
- Generates `hybrid_selection_candidates.csv` for modeling.

### `run_carob_model_compare.py`
- Compares models across feature scenarios.
- Uses Leave-One-Trial-Out validation for generalization across trials.

### `run_carob_causal_pilot.py`
- Estimates +P vs -P treatment effect with trial-aware meta-analysis.
- Produces heterogeneity diagnostics and action-playbook payloads.

## Output folders
- `outputs/carob_baseline/`
- `outputs/carob_feature_prepare/`
- `outputs/carob_model_compare/`
- `outputs/carob_causal_pilot/`

## Notes
- Legacy paddy scripts are still present but are no longer the project's primary path.
- Causal outputs are assumption-based and should always be interpreted with caveats.

## Legacy wrappers (paddy reference)
- `run_baseline.py`
- `run_feature_prepare.py`
- `run_model_compare.py`
- `run_model_select_tune.py`
- `run_ablation_eval.py`
- `run_interpretability_report.py`
- `run_causal_pilot.py`
