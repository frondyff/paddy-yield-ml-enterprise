import pandas as pd

from paddy_yield_ml.pipelines import causal_pilot as cp


def test_parse_rule_conditions_and_mask() -> None:
    rule = "A > 2.0 AND B <= 5.0"
    parsed = cp.parse_rule_conditions(rule)
    assert parsed == [("A", ">", 2.0), ("B", "<=", 5.0)]

    df = pd.DataFrame({"A": [1, 3, 4], "B": [4, 6, 5]})
    mask = cp.eval_rule_mask(df, rule)
    assert mask.tolist() == [False, False, True]
    assert cp.rule_mentions_feature(rule, "A") is True
    assert cp.rule_mentions_feature(rule, "AA") is False


def test_parse_treatment_specs_default_and_custom() -> None:
    default_specs = cp.parse_treatment_specs("")
    assert len(default_specs) == 3
    assert default_specs[0].feature == "Weed28D_thiobencarb"

    custom = cp.parse_treatment_specs("x|FeatA|>|1.5,y|FeatB|<=|7,z|FeatC|==|wet")
    assert [s.name for s in custom] == ["x", "y", "z"]
    assert custom[1].op == "<="
    assert custom[1].value == "7"
    assert custom[2].op == "=="


def test_apply_binary_treatment() -> None:
    s = pd.Series([1, 5, 8, None])
    hi = cp.apply_binary_treatment(s, ">", "4")
    lo = cp.apply_binary_treatment(s, "<=", "4")
    assert hi.tolist()[:3] == [0.0, 1.0, 1.0]
    assert lo.tolist()[:3] == [1.0, 0.0, 0.0]
    assert pd.isna(hi.iloc[3])

    cat = pd.Series(["wet", "dry", "wet", None])
    eq = cp.apply_binary_treatment(cat, "==", "wet")
    ne = cp.apply_binary_treatment(cat, "!=", "wet")
    assert eq.tolist()[:3] == [1.0, 0.0, 1.0]
    assert ne.tolist()[:3] == [0.0, 1.0, 0.0]
    assert pd.isna(eq.iloc[3])


def test_detect_hectare_coupled_numeric_features() -> None:
    df = pd.DataFrame(
        {
            "Hectares": [2, 2, 4, 4, 6, 6],
            "DoseLinear": [20, 20, 40, 40, 60, 60],
            "DoseVariable": [10, 12, 22, 20, 31, 33],
        }
    )
    out = cp.detect_hectare_coupled_numeric_features(
        frame=df,
        candidate_features=["DoseLinear", "DoseVariable", "Hectares"],
        corr_threshold=0.999,
    )
    flags = dict(zip(out["feature"], out["size_coupled_flag"], strict=False))
    assert flags["DoseLinear"] is True
    assert flags["DoseVariable"] is False
