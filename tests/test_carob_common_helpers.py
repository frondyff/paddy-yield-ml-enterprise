import pandas as pd

from paddy_yield_ml.pipelines import carob_common as cc


def test_eta_squared_nonnegative_and_signal() -> None:
    y = pd.Series([1, 2, 3, 10, 11, 12], dtype=float)
    g = pd.Series(["A", "A", "A", "B", "B", "B"])
    eta = cc.eta_squared(y, g)
    assert eta >= 0.0
    assert eta <= 1.0
    assert eta > 0.8


def test_safe_corr_handles_low_variance() -> None:
    x = pd.Series([1, 1, 1, 1, 1], dtype=float)
    y = pd.Series([1, 2, 3, 4, 5], dtype=float)
    out = cc.safe_corr(x, y, method="spearman")
    assert pd.isna(out)
