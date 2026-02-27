# Experiment Manifest

Date: February 27, 2026  
Repository: `paddy-yield-ml-enterprise`  
Git commit at manifest update: `a8cacd4`  
Working tree state at update: clean

## 1) Objective
Deliver a CAROB-first, interpretable yield decision-support workflow with:
- leakage-aware feature governance,
- trial-aware predictive evaluation,
- rule-level interpretability outputs,
- and rule-as-treatment causal diagnostics.

## 2) Data Contract
- Input dataset: `data/input/carob_amazxa.csv`
- Data dictionary: `data/metadata/data_dictionary_carob_amazxa.csv`
- Metadata: `data/metadata/carob_amazxa_meta.csv`
- Target: `yield`
- Grouping key: `trial_id`
- Treatment field in raw data: `treatment` (`+P`, `-P`)  
  Note: causal pipeline uses **rule-as-treatment**, not raw treatment labels.

## 3) Feature-Prepare Contract
Pipeline: `src/paddy_yield_ml/pipelines/carob_feature_prepare.py`  
Runner: `scripts/run_carob_feature_prepare.py`

### 3.1 Rules and Gates
- Pipeline version: `carob-fe-v3`
- Proxy policy: `preserve_role`
- Country constancy gate: enabled (`threshold=0.8`, `min_rows=20`)
- Trial full-missing soil gate: enabled (`soil_P`, `soil_pH`)

### 3.2 Population Summary
- Rows raw: `1202`
- Rows after country gate: `1108`
- Rows after trial gate: `830`
- Countries raw -> after gate: `8 -> 6`
- Trials raw -> after country gate -> after trial gate: `19 -> 17 -> 13`

Source:
- `outputs/carob_feature_prepare/modeling_population_summary.csv`

### 3.3 Feature Status Summary
- `candidate_modifiable=7`
- `reserve_context=10`
- `excluded=10`

Sources:
- `outputs/carob_feature_prepare/hybrid_selection_candidates.csv`
- `outputs/carob_feature_prepare/feature_prepare_runlog.txt`

## 4) Model-Compare Contract
Pipeline: `src/paddy_yield_ml/pipelines/carob_model_compare.py`  
Runner: `scripts/run_carob_model_compare.py`

### 4.1 Evaluation Settings
- Scenario: `modifiable_plus_context`
- Features used: `17`
- Split: trial-aware 80/20 holdout
- Models compared: ExtraTrees, CatBoost, LightGBM, XGBoost, RandomForest

### 4.2 Current Best (Model Compare)
- Model: `ExtraTrees`
- `R2=0.5161`, `RMSE=965.95`, `MAE=733.76`
- `n_train=670`, `n_test=160`, `n_trials_in_test=13`

Sources:
- `outputs/carob_model_compare/model_comparison_summary.csv`
- `outputs/carob_model_compare/scenario_feature_sets.csv`

## 5) Top-2 Fine-Tuning Contract
Pipeline: `src/paddy_yield_ml/pipelines/carob_model_tune_top2.py`  
Runner: `scripts/run_carob_model_tune_top2.py`

### 5.1 Winner
- Model: `CatBoost`
- `R2_mean=0.5389`
- `RMSE_mean=922.90`
- `MAE_mean=671.81`
- `RMSE_worst=971.49`

### 5.2 Runner-up
- Model: `ExtraTrees`
- `R2_mean=0.5029`
- `RMSE_mean=957.70`

Source:
- `outputs/carob_model_tune_top2/model_winners.csv`

## 6) Interpretability Contract
Pipeline: `src/paddy_yield_ml/pipelines/carob_interpretability.py`  
Runner: `scripts/run_carob_interpretability.py`  
Run tag: `iter3_defensible_v5`

### 6.1 Outputs in Use
- Final rules: `5`
- Rule-country status counts:
  - `works_here=4`
  - `unstable_or_small_effect=3`
  - `conflicts_here=2`
  - `insufficient_evidence=16`

Sources:
- `outputs/carob_interpretability/iter3_defensible_v5/iteration3_rules_final.csv`
- `outputs/carob_interpretability/iter3_defensible_v5/iteration3_rule_country_generalization.csv`
- `outputs/carob_interpretability/iter3_defensible_v5/iteration3_action_playbook.md`

## 7) Rule-Causal Contract (AIPW)
Pipeline: `src/paddy_yield_ml/pipelines/carob_rule_causal_aipw.py`  
Runner: `scripts/run_carob_rule_causal_aipw.py`  
Run tag: `rule_aipw_v2`

### 7.1 Recommendation Summary
- Scorecard rows: `7`
- `Pilot-only=4`
- `Do-not-recommend=3`
- `Recommend=0`

Sources:
- `outputs/carob_rule_causal_aipw/rule_aipw_v2/causal_rule_scorecard.csv`
- `outputs/carob_rule_causal_aipw/rule_aipw_v2/pair_aipw_summary.csv`
- `outputs/carob_rule_causal_aipw/rule_aipw_v2/causal_rule_playbook_overlay.md`

## 8) Reproducibility Commands
Run in this order from repo root:

```powershell
uv run python scripts/run_carob_feature_prepare.py
uv run python scripts/run_carob_model_compare.py
uv run python scripts/run_carob_model_tune_top2.py
uv run python scripts/run_carob_interpretability.py --run-tag iter3_defensible_v5
uv run python scripts/run_carob_rule_causal_aipw.py --run-tag rule_aipw_v2
```

## 9) Known Limits
- Causal outputs remain assumption-dependent and should not be framed as policy-grade guarantees.
- Several rule-country pairs remain `Pilot-only` due to diagnostic bottlenecks (balance/overlap/trial diversity).
- CAROB outputs are tracked for reproducibility, but generated artifacts should still be interpreted with run tags.
