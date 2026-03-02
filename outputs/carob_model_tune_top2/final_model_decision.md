# Final Model Decision (Top-2 Contenders Tune)

- Scenario: `modifiable_plus_context`
- Models tuned: `['extratrees', 'catboost']`
- Split: trial-aware train/validation/test = `0.60` / `0.20` / `0.20`
- Primary seed: `42`
- Stability seeds: `[42, 52, 62]`
- Trial-median imputation: `True`
- Learning-curve enabled: `True`
- Candidate frame rows: `830`
- Features used: `17`

## Winner Ranking (Final Locked Test)
1. ExtraTrees | test_r2=0.4792 | test_rmse=1002.02 | test_mae=749.09 | val_r2_mean_stability=0.5072
2. CatBoost | test_r2=0.4792 | test_rmse=1002.08 | test_mae=740.98 | val_r2_mean_stability=0.4755

## Selected Configuration
- Model: `ExtraTrees`
- Params: `{"max_depth": 18, "max_features": "log2", "min_samples_leaf": 1, "min_samples_split": 8, "n_estimators": 800}`

## Best-Model Learning Curve Artifacts
- `learning_curve_best_model_metrics.csv`
- `learning_curve_best_model_summary.csv`
- `learning_curve_best_model.png`