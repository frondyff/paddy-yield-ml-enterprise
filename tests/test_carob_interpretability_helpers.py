import pandas as pd

from paddy_yield_ml.pipelines import carob_interpretability as ci


def test_parse_rule_conditions_extracts_conditions() -> None:
    out = ci.parse_rule_conditions("N_fertilizer <= 40.5 AND irrigated > 0.5")
    assert out == [("N_fertilizer", "<=", 40.5), ("irrigated", ">", 0.5)]


def test_apply_rule_conditions_filters_rows() -> None:
    x = pd.DataFrame(
        {
            "N_fertilizer": [20.0, 50.0, 40.0],
            "irrigated": [1, 1, 0],
        }
    )
    mask = ci.apply_rule_conditions(x, "N_fertilizer <= 40 AND irrigated > 0.5")
    assert mask.tolist() == [True, False, False]


def test_select_rules_with_backoff_uses_threshold_pass_when_possible() -> None:
    rules = pd.DataFrame(
        {
            "rule_id": ["R1", "R2", "R3"],
            "support_mean": [0.12, 0.09, 0.02],
            "lift_mean": [120.0, 70.0, 200.0],
            "sign_match_rate": [0.95, 0.80, 0.20],
        }
    )
    selected, meta = ci.select_rules_with_backoff(
        rules,
        min_rules=2,
        support_thresholds=[0.08],
        abs_lift_thresholds=[60.0],
        sign_thresholds=[0.75],
    )
    assert len(selected) == 2
    assert meta["mode"] == "threshold_pass"
    assert set(selected["rule_id"].tolist()) == {"R1", "R2"}


def test_condition_to_english_handles_bool_like_feature() -> None:
    role_map = pd.DataFrame(
        [
            {"column_name": "irrigated", "meaning": "Field irrigation status", "unit_or_levels": "0/1"},
        ]
    )
    txt = ci.condition_to_english("irrigated", ">", 0.5, role_map_df=role_map, bool_like_cols={"irrigated"})
    assert txt == "Field irrigation status is Yes/1"


def test_summarize_rule_country_generalization_labels_statuses() -> None:
    def row(
        *,
        seed: int,
        country: str,
        support_count: int,
        support_pct: float,
        lift: float,
    ) -> dict[str, float | int | str]:
        return {
            "rule_id": "R1",
            "seed": seed,
            "country": country,
            "support_count": support_count,
            "support_pct": support_pct,
            "actual_lift_vs_country_baseline": lift,
        }

    seed_rows = pd.DataFrame(
        [
            # R1 in Benin: consistent positive, enough support => works_here
            row(seed=1, country="Benin", support_count=5, support_pct=0.20, lift=120.0),
            row(seed=2, country="Benin", support_count=4, support_pct=0.18, lift=140.0),
            # R1 in Japan: enough support but opposite sign => conflicts_here
            row(seed=1, country="Japan", support_count=5, support_pct=0.20, lift=-130.0),
            row(seed=2, country="Japan", support_count=4, support_pct=0.18, lift=-110.0),
            # R1 in Nigeria: no support => insufficient_evidence
            row(seed=1, country="Nigeria", support_count=0, support_pct=0.00, lift=float("nan")),
            row(seed=2, country="Nigeria", support_count=0, support_pct=0.00, lift=float("nan")),
        ]
    )
    final_rules = pd.DataFrame([{"rule_id": "R1", "lift_score": 200.0}])
    out = ci.summarize_rule_country_generalization(seed_rows, final_rules, min_abs_lift=100.0)
    status_map = dict(zip(out["country"], out["generalization_status"], strict=False))
    assert status_map["Benin"] == "works_here"
    assert status_map["Japan"] == "conflicts_here"
    assert status_map["Nigeria"] == "insufficient_evidence"
