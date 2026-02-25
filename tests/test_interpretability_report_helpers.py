import pandas as pd

from paddy_yield_ml.pipelines import interpretability_report as ir


def test_select_local_case_indices_returns_key_cases() -> None:
    y_true = pd.Series([100.0, 110.0, 90.0, 120.0, 105.0, 95.0])
    y_pred = pd.Series([102.0, 108.0, 130.0, 80.0, 106.0, 96.0])
    out = ir.select_local_case_indices(y_true=y_true, y_pred=y_pred)
    assert {"highest_prediction", "lowest_prediction", "largest_overprediction", "largest_underprediction"}.issubset(
        set(out.keys())
    )
    assert len(set(out.values())) == len(out)


def test_pick_modifiable_numeric_features_filters_roles_and_types() -> None:
    x = pd.DataFrame(
        {
            "A": [1.0, 2.0],
            "B": ["x", "y"],
            "C": [3.0, 4.0],
        }
    )
    role_map = pd.DataFrame(
        [
            {"column_name": "A", "final_role": "modifiable", "feature_group": "Input"},
            {"column_name": "B", "final_role": "modifiable", "feature_group": "Input"},
            {"column_name": "C", "final_role": "context", "feature_group": "Profile"},
        ]
    )
    out = ir.pick_modifiable_numeric_features(
        x=x,
        model_features=["A", "B", "C"],
        role_map_df=role_map,
    )
    assert out == ["A"]
