# Pipelines Index

This package contains two tracks:

## Active track (CAROB)
- `carob_common.py`
- `carob_baseline.py`
- `carob_feature_prepare.py`
- `carob_model_compare.py`
- `carob_model_compare_excl_benin.py`
- `carob_model_tune.py`
- `carob_model_tune_top2.py`
- `carob_interpretability.py`
- `carob_rule_causal_aipw.py`

Recommended run order:
1. `carob_baseline.py`
2. `carob_feature_prepare.py`
3. `carob_model_compare.py`
4. `carob_interpretability.py`
5. `carob_rule_causal_aipw.py`

## Legacy track (paddy reference)
- `baseline.py`
- `feature_prepare.py`
- `model_compare.py`
- `model_select_tune.py`
- `ablation_eval.py`
- `interpretability_report.py`
- `causal_pilot.py`

Legacy modules remain for reference and compatibility. The default project path is CAROB.
