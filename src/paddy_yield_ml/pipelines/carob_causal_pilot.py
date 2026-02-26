"""Trial-aware causal pilot for CAROB AMAZXA (+P vs -P)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from paddy_yield_ml.pipelines import carob_common as cc

try:
    project_root = Path(__file__).resolve().parents[3]
except NameError:
    project_root = Path.cwd()

OUT_ROOT = project_root / "outputs" / "carob_causal_pilot"


def mode_value_or_nan(g: pd.DataFrame, col: str) -> str | float:
    if col not in g.columns:
        return float("nan")
    vals = g[col].dropna().astype(str)
    if len(vals) == 0:
        return float("nan")
    return vals.mode().iat[0]


def soil_p_mean_or_nan(g: pd.DataFrame) -> float:
    if "soil_P" not in g.columns:
        return float("nan")
    return float(pd.to_numeric(g["soil_P"], errors="coerce").mean())


def confidence_label(
    fixed_ci: tuple[float, float],
    i2: float,
    n_trials_both: int,
) -> tuple[str, str]:
    caveats: list[str] = ["assumption_based"]
    ci_excludes_zero = bool(fixed_ci[0] > 0 or fixed_ci[1] < 0)

    if n_trials_both < 10:
        caveats.append("low_trial_support")
    if i2 >= 75:
        caveats.append("high_heterogeneity")
    elif i2 >= 50:
        caveats.append("moderate_heterogeneity")

    if not ci_excludes_zero:
        caveats.append("ci_crosses_zero")

    if n_trials_both >= 14 and ci_excludes_zero and i2 < 50:
        return "High", ",".join(caveats)
    if n_trials_both >= 10 and ci_excludes_zero:
        return "Medium", ",".join(caveats)
    return "Low", ",".join(caveats)


def build_trial_effects(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for trial_id, g in df.groupby(cc.GROUP_COL, dropna=False):
        t = pd.to_numeric(g.loc[g["treat_p"] == 1, cc.TARGET_COL], errors="coerce").dropna().to_numpy(dtype=float)
        c = pd.to_numeric(g.loc[g["treat_p"] == 0, cc.TARGET_COL], errors="coerce").dropna().to_numpy(dtype=float)

        n_t = int(len(t))
        n_c = int(len(c))
        if n_t == 0 or n_c == 0:
            rows.append(
                {
                    "trial_id": trial_id,
                    "n_rows": int(len(g)),
                    "n_treated": n_t,
                    "n_control": n_c,
                    "has_both_arms": False,
                    "effect_plusP_minusP": np.nan,
                    "se2": np.nan,
                    "country": mode_value_or_nan(g, "country"),
                    "location": mode_value_or_nan(g, "location"),
                    "soil_P_mean": soil_p_mean_or_nan(g),
                }
            )
            continue

        mt, mc = float(np.mean(t)), float(np.mean(c))
        vt = float(np.var(t, ddof=1)) if n_t > 1 else np.nan
        vc = float(np.var(c, ddof=1)) if n_c > 1 else np.nan
        se2 = float((vt / n_t) + (vc / n_c)) if np.isfinite(vt) and np.isfinite(vc) else np.nan

        rows.append(
            {
                "trial_id": trial_id,
                "n_rows": int(len(g)),
                "n_treated": n_t,
                "n_control": n_c,
                "has_both_arms": True,
                "effect_plusP_minusP": float(mt - mc),
                "se2": se2,
                "country": mode_value_or_nan(g, "country"),
                "location": mode_value_or_nan(g, "location"),
                "soil_P_mean": soil_p_mean_or_nan(g),
            }
        )

    return pd.DataFrame(rows).sort_values("trial_id")


def meta_effect(trial_df: pd.DataFrame) -> dict[str, float]:
    valid = trial_df[trial_df["has_both_arms"]].copy()
    valid = valid[valid["se2"].notna() & (valid["se2"] > 0)].copy()
    if len(valid) < 3:
        return {
            "n_trials": int(len(valid)),
            "fixed_effect": np.nan,
            "fixed_se": np.nan,
            "fixed_ci_low": np.nan,
            "fixed_ci_high": np.nan,
            "random_effect": np.nan,
            "random_se": np.nan,
            "random_ci_low": np.nan,
            "random_ci_high": np.nan,
            "q": np.nan,
            "i2": np.nan,
            "tau2": np.nan,
        }

    d = valid["effect_plusP_minusP"].to_numpy(dtype=float)
    se2 = valid["se2"].to_numpy(dtype=float)
    w = 1.0 / se2

    fixed = float(np.sum(w * d) / np.sum(w))
    fixed_se = float(np.sqrt(1.0 / np.sum(w)))
    fixed_low = float(fixed - 1.96 * fixed_se)
    fixed_high = float(fixed + 1.96 * fixed_se)

    q = float(np.sum(w * (d - fixed) ** 2))
    df_q = max(len(valid) - 1, 1)
    i2 = float(max((q - df_q) / q, 0.0) * 100.0) if q > 0 else 0.0

    sum_w = float(np.sum(w))
    sum_w2 = float(np.sum(w**2))
    c_val = sum_w - (sum_w2 / sum_w) if sum_w > 0 else 0.0
    tau2 = float(max((q - df_q) / c_val, 0.0)) if c_val > 0 else 0.0

    w_re = 1.0 / (se2 + tau2)
    random_eff = float(np.sum(w_re * d) / np.sum(w_re))
    random_se = float(np.sqrt(1.0 / np.sum(w_re)))
    random_low = float(random_eff - 1.96 * random_se)
    random_high = float(random_eff + 1.96 * random_se)

    return {
        "n_trials": int(len(valid)),
        "fixed_effect": fixed,
        "fixed_se": fixed_se,
        "fixed_ci_low": fixed_low,
        "fixed_ci_high": fixed_high,
        "random_effect": random_eff,
        "random_se": random_se,
        "random_ci_low": random_low,
        "random_ci_high": random_high,
        "q": q,
        "i2": i2,
        "tau2": tau2,
    }


def heterogeneity_tables(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows_country: list[dict[str, object]] = []
    if "country" in df.columns:
        for country, g in df.groupby("country", dropna=False):
            n_t = int((g["treat_p"] == 1).sum())
            n_c = int((g["treat_p"] == 0).sum())
            if n_t == 0 or n_c == 0:
                continue
            eff = float(
                pd.to_numeric(g.loc[g["treat_p"] == 1, cc.TARGET_COL], errors="coerce").mean()
                - pd.to_numeric(g.loc[g["treat_p"] == 0, cc.TARGET_COL], errors="coerce").mean()
            )
            rows_country.append(
                {
                    "country": country,
                    "n_rows": int(len(g)),
                    "n_trials": int(g[cc.GROUP_COL].astype(str).nunique()),
                    "n_treated": n_t,
                    "n_control": n_c,
                    "naive_effect_plusP_minusP": eff,
                }
            )

    rows_soil: list[dict[str, object]] = []
    if "soil_P" in df.columns:
        sdf = df.copy()
        sdf["soil_P"] = pd.to_numeric(sdf["soil_P"], errors="coerce")
        sdf = sdf[sdf["soil_P"].notna()].copy()
        if len(sdf) >= 40 and int(sdf["soil_P"].nunique()) >= 4:
            sdf["soil_P_bin"] = pd.qcut(sdf["soil_P"], q=4, duplicates="drop")
            for b, g in sdf.groupby("soil_P_bin", dropna=False):
                n_t = int((g["treat_p"] == 1).sum())
                n_c = int((g["treat_p"] == 0).sum())
                if n_t == 0 or n_c == 0:
                    continue
                eff = float(
                    pd.to_numeric(g.loc[g["treat_p"] == 1, cc.TARGET_COL], errors="coerce").mean()
                    - pd.to_numeric(g.loc[g["treat_p"] == 0, cc.TARGET_COL], errors="coerce").mean()
                )
                rows_soil.append(
                    {
                        "soil_P_bin": str(b),
                        "n_rows": int(len(g)),
                        "n_trials": int(g[cc.GROUP_COL].astype(str).nunique()),
                        "n_treated": n_t,
                        "n_control": n_c,
                        "naive_effect_plusP_minusP": eff,
                        "soil_P_min": float(g["soil_P"].min()),
                        "soil_P_max": float(g["soil_P"].max()),
                    }
                )

    return pd.DataFrame(rows_country), pd.DataFrame(rows_soil)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CAROB trial-aware causal pilot")
    parser.add_argument("--run-tag", type=str, default="latest")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = OUT_ROOT / args.run_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    df = cc.load_analysis_frame(require_treatment=True)
    df["treat_p"] = (df[cc.TREATMENT_COL].astype(str).str.strip() == cc.TREATMENT_POS).astype(int)

    trial_df = build_trial_effects(df)
    trial_df.to_csv(out_dir / "trial_effects.csv", index=False)

    meta = meta_effect(trial_df)
    n_trials_both = int(trial_df["has_both_arms"].sum()) if not trial_df.empty else 0
    conf, caveats = confidence_label(
        fixed_ci=(float(meta.get("fixed_ci_low", np.nan)), float(meta.get("fixed_ci_high", np.nan))),
        i2=float(meta.get("i2", np.nan)) if pd.notna(meta.get("i2", np.nan)) else 999.0,
        n_trials_both=n_trials_both,
    )

    overall = {
        "run_tag": args.run_tag,
        "dataset": "CAROB AMAZXA",
        "treatment": "+P vs -P",
        "n_rows": int(len(df)),
        "n_trials_total": int(df[cc.GROUP_COL].astype(str).nunique()),
        "n_trials_with_both_arms": n_trials_both,
        "confidence_level": conf,
        "caveats": caveats,
        **meta,
    }
    pd.DataFrame([overall]).to_csv(out_dir / "causal_overall_summary.csv", index=False)

    country_df, soil_df = heterogeneity_tables(df)
    country_df.to_csv(out_dir / "causal_heterogeneity_country.csv", index=False)
    soil_df.to_csv(out_dir / "causal_heterogeneity_soilP_bins.csv", index=False)

    action = {
        "what_to_change": [
            (
                "Prioritize phosphorus treatment (+P) where baseline soil-P is low/moderate "
                "and trial analogs are comparable."
            ),
            "Use variety-specific responses; do not assume one universal lift across all varieties.",
        ],
        "what_to_control_for": [
            "Trial/site context (country, location, year/season)",
            "Soil baseline (soil_P, soil_pH)",
        ],
        "what_to_avoid": [
            "Single global recommendation without heterogeneity checks",
            "Interpreting post-harvest outcomes (grain_P/residue_P) as pre-harvest predictors",
        ],
        "confidence_level": conf,
        "caveats": caveats,
    }
    (out_dir / "action_playbook.json").write_text(json.dumps(action, indent=2), encoding="utf-8")

    payload = {
        "overall": overall,
        "trial_effects": trial_df.to_dict(orient="records"),
        "heterogeneity_country": country_df.to_dict(orient="records"),
        "heterogeneity_soilP_bins": soil_df.to_dict(orient="records"),
        "action_playbook": action,
    }
    (out_dir / "ui_causal_payload.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    runlog = [
        f"run_tag={args.run_tag}",
        f"rows={len(df)}",
        f"n_trials_total={int(df[cc.GROUP_COL].astype(str).nunique())}",
        f"n_trials_with_both_arms={n_trials_both}",
        f"fixed_effect={meta.get('fixed_effect', np.nan)}",
        f"fixed_ci=({meta.get('fixed_ci_low', np.nan)}, {meta.get('fixed_ci_high', np.nan)})",
        f"random_effect={meta.get('random_effect', np.nan)}",
        f"random_ci=({meta.get('random_ci_low', np.nan)}, {meta.get('random_ci_high', np.nan)})",
        f"i2={meta.get('i2', np.nan)}",
        f"confidence_level={conf}",
        f"caveats={caveats}",
    ]
    (out_dir / "causal_runlog.txt").write_text("\n".join(runlog), encoding="utf-8")

    print(f"Saved CAROB causal outputs to: {out_dir}")
    print(f"Fixed effect (+P vs -P): {meta.get('fixed_effect', np.nan):.2f} kg/ha")
    print(f"95% CI: ({meta.get('fixed_ci_low', np.nan):.2f}, {meta.get('fixed_ci_high', np.nan):.2f})")
    print(f"Heterogeneity I2: {meta.get('i2', np.nan):.2f}% | confidence={conf}")


if __name__ == "__main__":
    main()
