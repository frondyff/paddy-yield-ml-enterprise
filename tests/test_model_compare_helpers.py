import pandas as pd

from paddy_yield_ml.pipelines import model_compare as mc


def test_build_feature_sets_groups_candidate_statuses() -> None:
    candidates_df = pd.DataFrame(
        [
            {"feature": "A", "status": "candidate_modifiable", "hybrid_priority_score": 0.9},
            {"feature": "B", "status": "reserve_context", "hybrid_priority_score": 0.8},
            {"feature": "C", "status": "candidate_redundant_review", "hybrid_priority_score": 0.7},
            {"feature": "D", "status": "excluded", "hybrid_priority_score": 0.1},
        ]
    )

    feature_sets = mc.build_feature_sets(candidates_df)

    assert feature_sets["modifiable_only"] == ["A"]
    assert feature_sets["modifiable_plus_context"] == ["A", "B"]
    assert feature_sets["hybrid_with_review"] == ["A", "B", "C"]


def test_filter_available_features_drops_reserved_and_missing() -> None:
    frame = pd.DataFrame(
        {
            "A": [1, 2],
            "B": [3, 4],
            mc.RAW_TARGET_COL: [10, 11],
            mc.TARGET_COL: [5.0, 5.5],
            mc.GROUP_COL: ["x", "y"],
        }
    )

    filtered = mc.filter_available_features(["A", "missing", mc.GROUP_COL, "B"], frame)
    assert filtered == ["A", "B"]
