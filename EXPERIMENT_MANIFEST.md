# Experiment Manifest

Date: 2026-02-26 14:10:36 -05:00  
Repository: `paddy-yield-ml-enterprise`  
Git commit: `f8d6a8b2d7587e14a8decb9d341c9dd3e395b590`  
Working tree state: dirty (local edits + untracked files present)

## 1) Objective
Build an interpretable yield decision-support workflow using CAROB data with leakage-aware feature governance and trial-aware evaluation.

## 2) Data Contract
- Input dataset: `data/input/carob_amazxa.csv`
- Data dictionary: `data/metadata/data_dictionary_carob_amazxa.csv`
- Metadata: `data/metadata/carob_amazxa_meta.csv`
- Target: `yield`
- Grouping key: `trial_id`
- Treatment column: `treatment` with levels `+P` and `-P`

## 3) Feature-Prepare Contract (Selection First, Row Gates After)
Pipeline file: `src/paddy_yield_ml/pipelines/carob_feature_prepare.py`  
Run script: `scripts/run_carob_feature_prepare.py`

### 3.1 Rule Settings
- `pipeline_version=carob-fe-v3`
- `data_proxy_policy=preserve_role`
- Country constancy gate:
  - enabled: `True`
  - ratio threshold: `0.8`
  - min rows: `20`
- Trial full-missing soil gate:
  - enabled: `True`
  - features: `soil_P`, `soil_pH`
  - drop trial if any listed feature is 100% missing in that trial

### 3.2 Population Counts
- Rows raw: `1202`
- Rows used for feature selection base: `1202`
- Rows after country gate: `1108`
- Rows after trial gate: `830`
- Rows removed by country gate: `94`
- Rows removed by trial gate: `278`
- Countries raw: `8`
- Countries after country gate: `6`
- Trials raw: `19`
- Trials after country gate: `17`
- Trials after trial gate: `13`

Source: `outputs/carob_feature_prepare/modeling_population_summary.csv`

### 3.3 Excluded Groups
- Countries dropped by country gate: `Sri Lanka`, `Gambia`
  - Source: `outputs/carob_feature_prepare/country_exclusion_audit.csv`
- Trials dropped by trial gate: `14`, `15`, `19`, `7`
  - Source: `outputs/carob_feature_prepare/trial_exclusion_audit.csv`

### 3.4 Feature Status (Post-Selection)
- `candidate_modifiable=7`
- `candidate_redundant_review=0`
- `reserve_context=10`
- `excluded=10`

Source: `outputs/carob_feature_prepare/feature_prepare_runlog.txt`

## 4) Model-Compare Contract
Pipeline file: `src/paddy_yield_ml/pipelines/carob_model_compare.py`  
Run script: `scripts/run_carob_model_compare.py`

### 4.1 Evaluation Settings
- Scenario: `modifiable_plus_context`
- Features used: `17`
- Split: trial-aware 80/20 (`test_size=0.2`)
- Random state: `42`
- Numeric missing imputation: train-only trial median with global train-median fallback
- Categorical imputation: most-frequent + one-hot encoding (`handle_unknown=ignore`)
- Models compared: RandomForest, ExtraTrees, CatBoost, XGBoost, LightGBM

### 4.2 Scenario Features
`variety`, `flooded`, `irrigated`, `N_fertilizer`, `P_fertilizer`, `row_spacing`, `K_fertilizer`, `soil_P`, `plot_area`, `latitude`, `soil_pH`, `location`, `country`, `longitude`, `rep`, `season`, `planting_date`

Source: `outputs/carob_model_compare_post_feature_selection_gates/scenario_feature_sets.csv`

### 4.3 Model-Compare Result (Current)
- Best: `ExtraTrees` (param set 3)
- Params: `{"max_depth": 12, "min_samples_leaf": 2, "n_estimators": 500}`
- `R2=0.5161`
- `RMSE=965.95`
- `MAE=733.76`
- `n_train=670`, `n_test=160`, `n_trials_in_test=13`

Source: `outputs/carob_model_compare_post_feature_selection_gates/model_comparison_summary.csv`

## 5) Top-2 Fine-Tuning Contract
Tuned contenders: `ExtraTrees` and `CatBoost`  
Scenario: `modifiable_plus_context`  
Rows: `830`  
Features: `17`  
Seeds for stability: `42, 52, 62`

### 5.1 Winner
- Model: `CatBoost`
- Params: `{"bagging_temperature": 1.0, "depth": 8, "l2_leaf_reg": 7.0, "learning_rate": 0.05, "n_estimators": 500, "random_strength": 0.5}`
- `R2_mean=0.5389`
- `RMSE_mean=922.90`
- `MAE_mean=671.81`
- `RMSE_worst=971.49`

### 5.2 Runner-Up
- Model: `ExtraTrees`
- `R2_mean=0.5029`
- `RMSE_mean=957.70`
- `MAE_mean=713.88`
- `RMSE_worst=1013.22`

Sources:
- `outputs/carob_model_tune_top2/model_winners.csv`
- `outputs/carob_model_tune_top2/final_model_decision.md`

## 6) Reproducibility Commands
Run in order:

```powershell
.venv\Scripts\python.exe scripts/run_carob_feature_prepare.py
.venv\Scripts\python.exe scripts/run_carob_model_compare.py --out-dir outputs/carob_model_compare_post_feature_selection_gates
.venv\Scripts\python.exe scripts/run_carob_model_tune_top2.py --out-dir outputs/carob_model_tune_top2
```

## 7) Known Limits
- Results remain sensitive to trial/country composition.
- Missing soil fields required trial exclusions for reliability.
- Working tree is currently dirty; commit and tag this manifest with code state before final reporting.
