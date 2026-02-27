import pandas as pd

from paddy_yield_ml.pipelines import carob_rule_causal_aipw as cr


def test_parse_seed_list_dedupes_and_parses_ints() -> None:
    out = cr.parse_seed_list("42, 52,42, 62")
    assert out == [42, 52, 62]


def test_summarize_pair_estimates_sign_match_rate() -> None:
    seed_df = pd.DataFrame(
        [
            {
                "rule_id": "R1",
                "country": "Benin",
                "generalization_status": "works_here",
                "analysis_tier": "primary",
                "seed": 42,
                "ate_aipw": 100.0,
                "ci_low": 20.0,
                "ci_high": 180.0,
                "ci_excludes_zero": True,
                "overlap_fraction_common_support": 0.9,
                "max_abs_smd_after": 0.1,
                "ess_treated": 40.0,
                "ess_control": 50.0,
                "p_value_bootstrap_two_sided": 0.04,
            },
            {
                "rule_id": "R1",
                "country": "Benin",
                "generalization_status": "works_here",
                "analysis_tier": "primary",
                "seed": 52,
                "ate_aipw": 120.0,
                "ci_low": 10.0,
                "ci_high": 220.0,
                "ci_excludes_zero": True,
                "overlap_fraction_common_support": 0.85,
                "max_abs_smd_after": 0.15,
                "ess_treated": 38.0,
                "ess_control": 49.0,
                "p_value_bootstrap_two_sided": 0.03,
            },
            {
                "rule_id": "R1",
                "country": "Benin",
                "generalization_status": "works_here",
                "analysis_tier": "primary",
                "seed": 62,
                "ate_aipw": -10.0,
                "ci_low": -120.0,
                "ci_high": 80.0,
                "ci_excludes_zero": False,
                "overlap_fraction_common_support": 0.88,
                "max_abs_smd_after": 0.2,
                "ess_treated": 39.0,
                "ess_control": 48.0,
                "p_value_bootstrap_two_sided": 0.52,
            },
        ]
    )
    out = cr.summarize_pair_estimates(seed_df)
    assert len(out) == 1
    # Reference sign is positive (seed 42). 2 out of 3 seeds match.
    assert float(out.loc[0, "sign_match_rate"]) == 2.0 / 3.0


def test_build_recommendation_handles_conflict_status() -> None:
    row = pd.Series(
        {
            "generalization_status": "conflicts_here",
            "ci_excludes_zero_ref_seed": True,
            "sign_match_rate": 1.0,
            "overlap_fraction_common_support_ref_seed": 0.9,
            "max_abs_smd_after_ref_seed": 0.1,
            "ate_mean_across_seeds": 200.0,
        }
    )
    rec, reason = cr.build_recommendation(row)
    assert rec == "Do-not-recommend"
    assert "conflicts" in reason

