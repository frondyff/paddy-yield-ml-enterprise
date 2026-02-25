from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer

from paddy_yield_ml.pipelines.baseline_train import (
    RAW_TARGET_COL,
    TARGET_COL,
    add_per_hectare_features,
    build_preprocessor,
    clean_columns,
    train_and_evaluate,
)


def test_clean_columns_normalizes_whitespace() -> None:
    cols = ["  A   B  ", " C\tD ", "E"]
    assert clean_columns(cols) == ["A B", "C D", "E"]


def test_add_per_hectare_features_creates_target_and_drops_raw_totals() -> None:
    df = pd.DataFrame(
        {
            "Hectares": [2.0, 4.0],
            RAW_TARGET_COL: [100.0, 200.0],
            "LP_nurseryarea(in Tonnes)": [10.0, 20.0],
            "Variety": ["v1", "v2"],
        }
    )

    out = add_per_hectare_features(df)

    assert TARGET_COL in out.columns
    assert np.allclose(out[TARGET_COL].values, [50.0, 50.0])
    assert RAW_TARGET_COL not in out.columns
    assert "LP_nurseryarea(in Tonnes)" not in out.columns
    assert "LP_nurseryarea(in Tonnes)_per_hectare" in out.columns


def test_build_preprocessor_fits_small_mixed_dataframe() -> None:
    X = pd.DataFrame(
        {
            "num1": [1.0, 2.0, 3.0],
            "cat1": ["a", "b", "a"],
            "num2": [3, 2, 1],
        }
    )
    preprocessor = build_preprocessor(X)

    assert isinstance(preprocessor, ColumnTransformer)
    Xt = preprocessor.fit_transform(X)
    assert Xt.shape[0] == len(X)


def test_train_and_evaluate_end_to_end_with_group_eval() -> None:
    rows = []
    for i in range(30):
        rows.append(
            {
                "Hectares": float(1 + (i % 5)),
                RAW_TARGET_COL: float(100 + i * 3),
                "LP_nurseryarea(in Tonnes)": float(5 + i),
                "Rainfall": float(50 + i),
                "Variety": "A" if i % 2 == 0 else "B",
                "Agriblock": f"G{i % 3}",
            }
        )
    df = pd.DataFrame(rows)
    featured = add_per_hectare_features(df)

    result = train_and_evaluate(
        featured,
        seed=7,
        test_size=0.2,
        group_col="Agriblock",
        run_group_eval=True,
    )

    assert "metrics" in result
    assert "random_split" in result["metrics"]
    assert set(result["metrics"]["random_split"].keys()) == {"mae", "rmse", "r2"}
    assert "group_split" in result["metrics"]
