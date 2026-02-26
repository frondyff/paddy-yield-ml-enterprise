import pandas as pd

from paddy_yield_ml.pipelines import carob_causal_pilot as cp


def test_confidence_label_thresholds() -> None:
    label, caveats = cp.confidence_label((10.0, 20.0), i2=30.0, n_trials_both=15)
    assert label == "High"
    assert "assumption_based" in caveats

    label2, caveats2 = cp.confidence_label((-5.0, 7.0), i2=80.0, n_trials_both=8)
    assert label2 == "Low"
    assert "ci_crosses_zero" in caveats2


def test_meta_effect_returns_valid_fixed_effect() -> None:
    trial_df = pd.DataFrame(
        {
            "trial_id": ["1", "2", "3"],
            "has_both_arms": [True, True, True],
            "effect_plusP_minusP": [100.0, 120.0, 80.0],
            "se2": [25.0, 16.0, 36.0],
        }
    )
    out = cp.meta_effect(trial_df)
    assert out["n_trials"] == 3
    assert out["fixed_effect"] > 0
    assert out["fixed_ci_high"] > out["fixed_ci_low"]
