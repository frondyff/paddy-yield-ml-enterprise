# CAROB Top Feature Explanation

- Scenario: `modifiable_plus_context`
- SHAP model: `CatBoost`
- Params source: `catboost_scenario_best`
- Params used: `{"depth": 6, "l2_leaf_reg": 3.0, "learning_rate": 0.05, "n_estimators": 300}`
- Expected value (mean baseline prediction): `2672.1764`

## Top Global Driver
- Feature: `location`
- Role: `context`
- Mean |SHAP|: `263.797094`

## Top 10 Features
1. `location` | role=`context` | mean_|SHAP|=`263.797094`
2. `variety` | role=`modifiable` | mean_|SHAP|=`187.096865`
3. `country` | role=`context` | mean_|SHAP|=`158.804801`
4. `soil_pH` | role=`context` | mean_|SHAP|=`147.825486`
5. `soil_P` | role=`context` | mean_|SHAP|=`140.115061`
6. `P_fertilizer` | role=`modifiable` | mean_|SHAP|=`122.714825`
7. `plot_area` | role=`context` | mean_|SHAP|=`95.321626`
8. `latitude` | role=`context` | mean_|SHAP|=`59.666007`
9. `flooded` | role=`modifiable` | mean_|SHAP|=`58.189458`
10. `planting_date` | role=`context` | mean_|SHAP|=`27.010917`