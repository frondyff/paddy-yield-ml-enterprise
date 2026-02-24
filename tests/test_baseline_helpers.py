from paddy_yield_ml.pipelines import baseline


def test_clean_columns_trims_and_compacts_double_spaces() -> None:
    cols = ["  A  ", "B", "C   D"]
    out = baseline.clean_columns(cols)
    assert out == ["A", "B", "C D"]


def test_baseline_paths_point_to_clean_project_layout() -> None:
    assert baseline.DATA_PATH.as_posix().endswith("data/input/paddydataset.csv")
    assert baseline.OUT_DIR.as_posix().endswith("outputs/baseline")
