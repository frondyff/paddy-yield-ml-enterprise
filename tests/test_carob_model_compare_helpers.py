import pandas as pd

from paddy_yield_ml.pipelines import carob_model_compare as cm


def test_build_trial_aware_split_indices_has_train_and_test_for_each_group_with_n_ge_2() -> None:
    groups = pd.Series(["A", "A", "A", "B", "B", "C", "C", "D"])
    tr, te = cm.build_trial_aware_split_indices(groups=groups, test_size=0.2, random_state=42)

    tr_set = set(tr.tolist())
    te_set = set(te.tolist())
    assert tr_set.isdisjoint(te_set)
    assert len(te_set) > 0

    # A, B, C each have >=2 rows, so each should appear in both train and test.
    group_by_idx = groups.to_dict()
    for g in ["A", "B", "C"]:
        train_has = any(group_by_idx[i] == g for i in tr_set)
        test_has = any(group_by_idx[i] == g for i in te_set)
        assert train_has
        assert test_has


def test_impute_numeric_by_group_uses_train_group_median_and_global_fallback() -> None:
    x_train = pd.DataFrame(
        {
            "f_num": [1.0, 3.0, None, 9.0],
            "f_cat": ["a", "b", "c", "d"],
        }
    )
    g_train = pd.Series(["A", "A", "B", "B"])

    x_test = pd.DataFrame(
        {
            "f_num": [None, None, 5.0],
            "f_cat": ["x", "y", "z"],
        }
    )
    g_test = pd.Series(["A", "C", "B"])  # C is unseen in train

    xtr, xte = cm.impute_numeric_by_group(
        x_train=x_train,
        x_test=x_test,
        groups_train=g_train,
        groups_test=g_test,
    )

    # Train B missing filled with B median (9.0).
    assert float(xtr.loc[2, "f_num"]) == 9.0
    # Test A missing filled with A median ((1+3)/2 = 2.0).
    assert float(xte.loc[0, "f_num"]) == 2.0
    # Test unseen C missing falls back to train global median (median of [1,3,9] = 3.0).
    assert float(xte.loc[1, "f_num"]) == 3.0
    # Non-missing remains unchanged.
    assert float(xte.loc[2, "f_num"]) == 5.0
    # Non-numeric untouched.
    assert xte["f_cat"].tolist() == ["x", "y", "z"]
