from paddy_yield_ml.pipelines import baseline_train
from paddy_yield_ml.pipelines.paths import project_root, resolve_data_path


def test_clean_columns_trims_and_compacts_double_spaces() -> None:
    cols = ["  A  ", "B", "C   D"]
    out = baseline_train.clean_columns(cols)
    assert out == ["A", "B", "C D"]


def test_baseline_paths_point_to_clean_project_layout() -> None:
    assert resolve_data_path(None).as_posix().endswith("data/input/paddydataset.csv")
    assert (project_root() / "outputs" / "baseline").as_posix().endswith("outputs/baseline")
