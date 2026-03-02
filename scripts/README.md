# Scripts Guide (CAROB-first)

These runner scripts are shortcut buttons for the CAROB workflow.

## Primary run order
From project root:

```bash
uv run python scripts/run_carob_baseline.py
uv run python scripts/run_carob_feature_prepare.py
uv run python scripts/run_carob_model_compare.py
uv run python scripts/run_carob_model_tune_top2.py
uv run python scripts/run_carob_interpretability.py --run-tag iter5_extratrees_shap_only_v1
uv run python scripts/run_carob_rule_causal_aipw.py --run-tag rule_aipw_v4_extratrees_shap_only --interp-dir outputs/carob_interpretability/iter5_extratrees_shap_only_v1
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
- Uses trial-aware train/validation/test evaluation.

### `run_carob_model_tune_top2.py`
- Tunes the top two contenders under the same trial-aware train/validation/test protocol.
- Produces the locked candidate winner set for interpretability and causal handoff.

### `run_carob_interpretability.py`
- Runs the iterative interpretability workflow (ExtraTrees TreeSHAP, permutation, rules, country diagnostics).
- Produces a defensible rule set and causal handoff outputs.

### `run_carob_rule_causal_aipw.py`
- Estimates rule-as-treatment effects using AIPW with trial-aware diagnostics.
- Enforces interpretability provenance (expects ExtraTrees SHAP-only artifacts by default).
- Produces pair diagnostics, scorecards, and playbook overlay outputs.

## Output folders
- `outputs/carob_baseline/`
- `outputs/carob_feature_prepare/`
- `outputs/carob_model_compare/`
- `outputs/carob_interpretability/`
- `outputs/carob_rule_causal_aipw/rule_aipw_v4_extratrees_shap_only/`

## Notes
- Legacy paddy scripts are still present but are no longer the project's primary path.
- Causal outputs are assumption-based and should always be interpreted with caveats.

## UI (Streamlit)
The interactive app is `streamlit_app.py` at repo root (not a wrapper script).

Run:

```bash
uv run streamlit run streamlit_app.py --server.port 8501
```

Current UI contract:
- Title: `Rice What-If Yield Advisor`
- Numeric context and modifiable levers use sliders
- Rule table includes rule-level yield lift (`kg/ha`)
- Causal evidence is shown only in Scientific mode

## Legacy wrappers (paddy reference)
- `run_baseline.py`
- `run_feature_prepare.py`
- `run_model_compare.py`
- `run_model_select_tune.py`
- `run_ablation_eval.py`
- `run_interpretability_report.py`
- `run_causal_pilot.py`
