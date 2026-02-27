# Scripts Guide (CAROB-first)

These runner scripts are shortcut buttons for the CAROB workflow.

## Primary run order
From project root:

```bash
uv run python scripts/run_carob_baseline.py
uv run python scripts/run_carob_feature_prepare.py
uv run python scripts/run_carob_model_compare.py
uv run python scripts/run_carob_interpretability.py --run-tag iter3_defensible_v5
uv run python scripts/run_carob_rule_causal_aipw.py --run-tag rule_aipw_v2
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

### `run_carob_interpretability.py`
- Runs the iterative interpretability workflow (SHAP, permutation, rules, country diagnostics).
- Produces a defensible rule set and causal handoff outputs.

### `run_carob_rule_causal_aipw.py`
- Estimates rule-as-treatment effects using AIPW with trial-aware diagnostics.
- Produces pair diagnostics, scorecards, and playbook overlay outputs.

### `run_carob_interpretability_report.py` (optional)
- Runs a lightweight SHAP report pipeline for quick summaries.

## Output folders
- `outputs/carob_baseline/`
- `outputs/carob_feature_prepare/`
- `outputs/carob_model_compare/`
- `outputs/carob_interpretability/`
- `outputs/carob_rule_causal_aipw/rule_aipw_v2/`

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
