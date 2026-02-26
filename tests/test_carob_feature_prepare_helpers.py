import pandas as pd

from paddy_yield_ml.pipelines import carob_feature_prepare as fp


def test_group_constancy_gate_drops_high_constancy_group(tmp_path) -> None:
    prev_out = fp.OUT_DIR
    fp.OUT_DIR = tmp_path
    try:
        g1 = pd.DataFrame(
            {
                "trial_id": ["A"] * 20,
                "yield": [100 + i for i in range(20)],
                "f1": [1] * 20,
                "f2": [2] * 20,
                "f3": [3] * 20,
            }
        )
        g2 = pd.DataFrame(
            {
                "trial_id": ["B"] * 20,
                "yield": [120 + i for i in range(20)],
                "f1": list(range(20)),
                "f2": [i % 2 for i in range(20)],
                "f3": [3] * 20,
            }
        )
        df = pd.concat([g1, g2], ignore_index=True)

        filtered, audit = fp.apply_group_constancy_gate(
            df=df,
            group_col="trial_id",
            enabled=True,
            ratio_threshold=0.8,
            min_rows=20,
            audit_filename="test_group_exclusion_audit.csv",
        )

        kept_groups = set(filtered["trial_id"].astype(str).unique().tolist())
        dropped_groups = set(audit.loc[audit["drop_group"], "group_value"].astype(str).tolist())
        assert kept_groups == {"B"}
        assert dropped_groups == {"A"}
    finally:
        fp.OUT_DIR = prev_out


def test_group_constancy_gate_excludes_global_constants_from_ratio(tmp_path) -> None:
    prev_out = fp.OUT_DIR
    fp.OUT_DIR = tmp_path
    try:
        g1 = pd.DataFrame(
            {
                "trial_id": ["A"] * 20,
                "yield": [100 + i for i in range(20)],
                "global_const": [7] * 20,
                "var_feat": list(range(20)),
            }
        )
        g2 = pd.DataFrame(
            {
                "trial_id": ["B"] * 20,
                "yield": [130 + i for i in range(20)],
                "global_const": [7] * 20,
                "var_feat": list(range(20, 40)),
            }
        )
        df = pd.concat([g1, g2], ignore_index=True)

        filtered, audit = fp.apply_group_constancy_gate(
            df=df,
            group_col="trial_id",
            enabled=True,
            ratio_threshold=0.4,
            min_rows=20,
            audit_filename="test_group_exclusion_audit.csv",
        )

        assert len(filtered) == len(df)
        assert int(audit["drop_group"].sum()) == 0
        assert int(audit["n_features_evaluated"].iloc[0]) == 1
        assert int(audit["global_non_varying_feature_count"].iloc[0]) >= 1
    finally:
        fp.OUT_DIR = prev_out


def test_trial_full_missing_gate_drops_trials_with_full_missing_soil(tmp_path) -> None:
    prev_out = fp.OUT_DIR
    fp.OUT_DIR = tmp_path
    try:
        df = pd.DataFrame(
            {
                "trial_id": ["1", "1", "2", "2", "3", "3"],
                "yield": [10, 11, 12, 13, 14, 15],
                "soil_P": [None, None, 1.0, 1.1, 2.0, 2.1],
                "soil_pH": [6.0, 6.1, None, None, 5.8, 5.9],
            }
        )

        filtered, audit = fp.apply_trial_full_missing_gate(
            df=df,
            group_col="trial_id",
            features=("soil_P", "soil_pH"),
            enabled=True,
            audit_filename="test_trial_exclusion_audit.csv",
        )

        kept_trials = set(filtered["trial_id"].astype(str).unique().tolist())
        dropped_trials = set(audit.loc[audit["drop_group"], "group_value"].astype(str).tolist())
        assert kept_trials == {"3"}
        assert dropped_trials == {"1", "2"}
    finally:
        fp.OUT_DIR = prev_out


def test_trial_full_missing_gate_keeps_trials_when_not_fully_missing(tmp_path) -> None:
    prev_out = fp.OUT_DIR
    fp.OUT_DIR = tmp_path
    try:
        df = pd.DataFrame(
            {
                "trial_id": ["A", "A", "B", "B"],
                "yield": [1, 2, 3, 4],
                "soil_P": [None, 1.0, 2.0, None],
                "soil_pH": [6.0, None, 5.9, 6.1],
            }
        )

        filtered, audit = fp.apply_trial_full_missing_gate(
            df=df,
            group_col="trial_id",
            features=("soil_P", "soil_pH"),
            enabled=True,
            audit_filename="test_trial_exclusion_audit.csv",
        )

        assert len(filtered) == len(df)
        assert int(audit["drop_group"].sum()) == 0
    finally:
        fp.OUT_DIR = prev_out
