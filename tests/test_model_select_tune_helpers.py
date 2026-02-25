import pandas as pd

from paddy_yield_ml.pipelines import model_select_tune as mst


def test_parse_csv_list_and_int_list() -> None:
    assert mst.parse_csv_list("a, b ,a") == ["a", "b"]
    assert mst.parse_int_list("1, 2,3") == [1, 2, 3]


def test_core_and_review_feature_split() -> None:
    df = pd.DataFrame(
        [
            {"feature": "A", "status": "candidate_modifiable"},
            {"feature": "B", "status": "reserve_context"},
            {"feature": "C", "status": "candidate_redundant_review"},
            {"feature": "D", "status": "excluded"},
        ]
    )
    core, review = mst.core_and_review_features(df)
    assert core == ["A", "B"]
    assert review == ["C"]


def test_build_stability_table_marks_threshold() -> None:
    runs = [
        {"selected_features": ["A", "B"]},
        {"selected_features": ["A"]},
        {"selected_features": ["A", "C"]},
    ]
    table, stable = mst.build_stability_table(runs, ["A", "B", "C"], threshold=0.66)
    assert "A" in stable
    assert "B" not in stable
    assert "C" not in stable
    assert table.shape[0] == 3
