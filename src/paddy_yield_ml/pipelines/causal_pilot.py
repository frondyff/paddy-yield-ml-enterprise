"""
Rule-linked agriblock-aware causal pilot (assumption-based).

This pipeline estimates treatment effects for a few modifiable levers, then links
those effects to decision-rule segments from interpretability outputs.

Run:
  python src/paddy_yield_ml/pipelines/causal_pilot.py --run-tag latest

Outputs (under ./outputs/causal_pilot/<run-tag>/):
  - treatment_definitions.csv
  - treatment_covariates.csv
  - size_coupled_feature_diagnostics.csv
  - causal_effect_estimates.csv
  - causal_overall_summary.csv
  - causal_rule_segment_summary.csv
  - causal_balance_diagnostics.csv
  - ui_causal_payload.json
  - causal_runlog.txt
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline

from paddy_yield_ml.pipelines import feature_prepare as fp
from paddy_yield_ml.pipelines import model_compare as mc

try:
    project_root = Path(__file__).resolve().parents[3]
except NameError:
    project_root = Path.cwd()

OUT_ROOT = project_root / "outputs" / "causal_pilot"
DEFAULT_RULES_PATH = (
    project_root / "outputs" / "interpretability" / "milestone_interpretability_v1" / "modifiable_decision_rules.csv"
)
DEFAULT_SCENARIO_PATH = project_root / "outputs" / "model_select_tune" / "dual_eval" / "scenario_feature_sets.csv"

TREATMENT_SPECS_DEFAULT: list[tuple[str, str, str, str]] = [
    ("weed_28d_high", "Weed28D_thiobencarb", ">", "7.0"),
    ("micronutrients_70d_high", "Micronutrients_70Days", ">", "67.5"),
    ("nursery_lp_moderate_or_low", "LP_nurseryarea(in Tonnes)", "<=", "5.5"),
]

FEATURE_DAY_MAP = {
    "DAP_20days": 20,
    "Weed28D_thiobencarb": 28,
    "Urea_40Days": 40,
    "Potassh_50Days": 50,
    "Pest_60Day(in ml)": 60,
    "Micronutrients_70Days": 70,
}

BASELINE_EARLY_FEATURES = {
    "Variety",
    "Nursery",
    "Hectares",
    "Soil Types",
    "Seedrate(in Kg)",
    "LP_Mainfield(in Tonnes)",
    "Nursery area (Cents)",
}


@dataclass(frozen=True)
class TreatmentSpec:
    name: str
    feature: str
    op: str
    value: str


def maybe_float(value: str) -> float | None:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def load_feature_set(path: Path, scenario_name: str) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Missing scenario file: {path}")
    df = pd.read_csv(path)
    required = {"scenario", "feature"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Scenario file missing required columns: {sorted(missing)}")
    features = df.loc[df["scenario"].astype(str) == scenario_name, "feature"].astype(str).dropna().tolist()
    if not features:
        options = sorted(df["scenario"].astype(str).unique().tolist())
        raise ValueError(f"Scenario '{scenario_name}' not found. Available: {options}")
    return mc.dedupe_keep_order(features)


def load_rules(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing rules file: {path}")
    df = pd.read_csv(path)
    required = {"rule_id", "rule_conditions", "support_pct", "actual_lift_vs_global"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Rules file missing required columns: {sorted(missing)}")
    return df.copy()


def parse_treatment_specs(raw: str) -> list[TreatmentSpec]:
    if not raw.strip():
        return [
            TreatmentSpec(name=name, feature=feature, op=op, value=value)
            for name, feature, op, value in TREATMENT_SPECS_DEFAULT
        ]

    specs: list[TreatmentSpec] = []
    chunks = [c.strip() for c in raw.split(",") if c.strip()]
    for chunk in chunks:
        parts = [p.strip() for p in chunk.split("|")]
        if len(parts) != 4:
            raise ValueError("Treatment chunk must have 4 pipe-separated fields: name|feature|op|value")
        name, feature, op, value_raw = parts
        if op not in {">", "<=", "==", "!="}:
            raise ValueError(f"Unsupported op '{op}' in treatment chunk: {chunk}")
        if op in {">", "<="} and maybe_float(value_raw) is None:
            raise ValueError(f"Treatment chunk has non-numeric cutoff for numeric operator '{op}': {chunk}")
        specs.append(TreatmentSpec(name=name, feature=feature, op=op, value=value_raw))
    return specs


def treatment_day(feature: str) -> int:
    return int(FEATURE_DAY_MAP.get(feature, 0))


def select_covariates_for_treatment(
    model_features: list[str],
    treatment_feature: str,
    excluded_covariates: set[str] | None = None,
) -> list[str]:
    day = treatment_day(treatment_feature)
    if day <= 0:
        covariates = [f for f in model_features if f != treatment_feature and f in BASELINE_EARLY_FEATURES]
    else:
        covariates = [f for f in model_features if f != treatment_feature and treatment_day(f) < day]
    if excluded_covariates:
        covariates = [f for f in covariates if f not in excluded_covariates]
    covariates = mc.dedupe_keep_order(covariates)
    if not covariates:
        raise ValueError(f"No covariates selected for treatment feature: {treatment_feature}")
    return covariates


def apply_binary_treatment(series: pd.Series, op: str, value: str) -> pd.Series:
    vals = pd.to_numeric(series, errors="coerce")
    numeric_value = maybe_float(value)
    if op == ">":
        if numeric_value is None:
            raise ValueError(f"Operator '>' requires numeric value, got: {value}")
        out = (vals > numeric_value).astype(float)
    elif op == "<=":
        if numeric_value is None:
            raise ValueError(f"Operator '<=' requires numeric value, got: {value}")
        out = (vals <= numeric_value).astype(float)
    elif op == "==":
        if numeric_value is not None and vals.notna().sum() > 0:
            out = (vals == numeric_value).astype(float)
            out = out.where(vals.notna(), np.nan)
            return out
        series_str = series.astype(str).str.strip().str.lower()
        target = str(value).strip().lower()
        out = (series_str == target).astype(float)
    elif op == "!=":
        if numeric_value is not None and vals.notna().sum() > 0:
            out = (vals != numeric_value).astype(float)
            out = out.where(vals.notna(), np.nan)
            return out
        series_str = series.astype(str).str.strip().str.lower()
        target = str(value).strip().lower()
        out = (series_str != target).astype(float)
    else:
        raise ValueError(f"Unsupported treatment operator: {op}")
    if op in {">", "<="}:
        out = out.where(vals.notna(), np.nan)
    else:
        out = out.where(series.notna(), np.nan)
    return out


def detect_hectare_coupled_numeric_features(
    frame: pd.DataFrame,
    candidate_features: list[str],
    corr_threshold: float,
) -> pd.DataFrame:
    if fp.SIZE_COL not in frame.columns:
        return pd.DataFrame(columns=["feature", "within_hectare_max_unique", "corr_with_hectares", "size_coupled_flag"])

    hectares = pd.to_numeric(frame[fp.SIZE_COL], errors="coerce")
    rows: list[dict[str, Any]] = []
    for feature in mc.dedupe_keep_order(candidate_features):
        if feature not in frame.columns or feature == fp.SIZE_COL:
            continue
        vals = pd.to_numeric(frame[feature], errors="coerce")
        if vals.notna().sum() < 3:
            continue
        pair = pd.DataFrame({"hectares": hectares, "value": vals}).dropna()
        if len(pair) < 3:
            continue
        within_counts = pair.groupby("hectares", dropna=False)["value"].nunique(dropna=False).to_numpy(dtype=float)
        within_unique = int(np.max(within_counts)) if len(within_counts) > 0 else 0
        corr = float(pair["value"].corr(pair["hectares"]))
        coupled = bool(within_unique <= 1 and np.isfinite(corr) and abs(corr) >= corr_threshold)
        rows.append(
            {
                "feature": feature,
                "within_hectare_max_unique": within_unique,
                "corr_with_hectares": corr,
                "size_coupled_flag": coupled,
            }
        )
    return pd.DataFrame(rows)


def parse_rule_conditions(rule: str) -> list[tuple[str, str, float]]:
    out: list[tuple[str, str, float]] = []
    for part in str(rule).split(" AND "):
        part = part.strip()
        if " <= " in part:
            f, v = part.split(" <= ", maxsplit=1)
            out.append((f.strip(), "<=", float(v.strip())))
        elif " > " in part:
            f, v = part.split(" > ", maxsplit=1)
            out.append((f.strip(), ">", float(v.strip())))
        else:
            raise ValueError(f"Unsupported rule token: {part}")
    return out


def condition_to_text(condition: tuple[str, str, float]) -> str:
    feat, op, threshold = condition
    return f"{feat} {op} {threshold:.3f}"


def eval_conditions_mask(frame: pd.DataFrame, conditions: list[tuple[str, str, float]]) -> pd.Series:
    mask = pd.Series(True, index=frame.index)
    for feat, op, threshold in conditions:
        if feat not in frame.columns:
            return pd.Series(False, index=frame.index)
        vals = pd.to_numeric(frame[feat], errors="coerce")
        if op == ">":
            cond = vals > threshold
        else:
            cond = vals <= threshold
        mask &= cond.fillna(False)
    return mask


def eval_rule_mask(frame: pd.DataFrame, rule_conditions: str) -> pd.Series:
    return eval_conditions_mask(frame, parse_rule_conditions(rule_conditions))


def rule_mentions_feature(rule_conditions: str, feature: str) -> bool:
    try:
        conditions = parse_rule_conditions(rule_conditions)
    except Exception:
        return False
    return any(cond[0] == feature for cond in conditions)


def prepare_estimation_frame(
    frame: pd.DataFrame,
    treatment_feature: str,
    treatment_op: str,
    treatment_value: str,
    covariates: list[str],
) -> pd.DataFrame:
    required = mc.dedupe_keep_order(covariates + [treatment_feature, mc.TARGET_COL, mc.GROUP_COL])
    subset = frame[required].copy()
    subset[mc.TARGET_COL] = pd.to_numeric(subset[mc.TARGET_COL], errors="coerce")
    subset["treatment"] = apply_binary_treatment(subset[treatment_feature], op=treatment_op, value=treatment_value)
    subset = subset[subset[mc.TARGET_COL].notna() & subset["treatment"].notna() & subset[mc.GROUP_COL].notna()].copy()
    subset["treatment"] = subset["treatment"].astype(int)
    subset[mc.GROUP_COL] = subset[mc.GROUP_COL].astype(str)
    return subset.reset_index(drop=True)


def fit_crossfit_nuisance(
    df: pd.DataFrame,
    covariates: list[str],
    n_splits: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, str]:
    x = df[covariates].copy()
    y = pd.to_numeric(df[mc.TARGET_COL], errors="coerce").to_numpy(dtype=float)
    t = df["treatment"].to_numpy(dtype=int)
    groups = df[mc.GROUP_COL].astype(str)

    n = len(df)
    e_hat = np.full(n, np.nan, dtype=float)
    m0_hat = np.full(n, np.nan, dtype=float)

    n_groups = groups.nunique()
    if n_groups < 2:
        return e_hat, m0_hat, "fail_insufficient_groups"
    splits = min(n_splits, n_groups)
    cv = GroupKFold(n_splits=splits)

    for train_idx, test_idx in cv.split(x, t, groups):
        x_train = x.iloc[train_idx].copy()
        x_test = x.iloc[test_idx].copy()
        t_train = t[train_idx]
        y_train = y[train_idx]

        # Propensity model P(T=1|X), agriblock-aware via covariates that include group.
        prop_pipe = Pipeline(
            [
                ("preprocess", mc.make_preprocessor(x_train)),
                (
                    "model",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                ),
            ]
        )
        try:
            prop_pipe.fit(x_train, t_train)
            e_hat[test_idx] = prop_pipe.predict_proba(x_test)[:, 1]
        except Exception:
            e_hat[test_idx] = float(np.mean(t_train))

        # Outcome model for untreated potential outcome m0(X)=E[Y|T=0,X].
        control_mask = t_train == 0
        if int(np.sum(control_mask)) < 30:
            if int(np.sum(control_mask)) > 0:
                fill_m0 = float(np.mean(y_train[control_mask]))
            else:
                fill_m0 = float(np.mean(y_train))
            m0_hat[test_idx] = fill_m0
        else:
            x_control = x_train.loc[control_mask].copy()
            y_control = y_train[control_mask]
            m0_pipe = Pipeline(
                [
                    ("preprocess", mc.make_preprocessor(x_control)),
                    (
                        "model",
                        RandomForestRegressor(
                            n_estimators=200,
                            min_samples_leaf=8,
                            random_state=random_state,
                            n_jobs=-1,
                        ),
                    ),
                ]
            )
            m0_pipe.fit(x_control, y_control)
            m0_hat[test_idx] = m0_pipe.predict(x_test)

    if np.isnan(e_hat).any():
        e_hat[np.isnan(e_hat)] = float(np.nanmean(e_hat))
    if np.isnan(m0_hat).any():
        m0_hat[np.isnan(m0_hat)] = float(np.nanmean(m0_hat))
    return e_hat, m0_hat, "ok"


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    denom = float(np.sum(weights))
    if denom <= 0:
        return float(np.nan)
    return float(np.sum(values * weights) / denom)


def weighted_var(values: np.ndarray, weights: np.ndarray) -> float:
    mu = weighted_mean(values, weights)
    if np.isnan(mu):
        return float(np.nan)
    denom = float(np.sum(weights))
    if denom <= 0:
        return float(np.nan)
    return float(np.sum(weights * (values - mu) ** 2) / denom)


def compute_balance_table(
    df: pd.DataFrame,
    covariates: list[str],
    control_weights: np.ndarray,
) -> pd.DataFrame:
    t = df["treatment"].to_numpy(dtype=int)
    rows: list[dict[str, Any]] = []
    for col in covariates:
        vals = pd.to_numeric(df[col], errors="coerce")
        mask = vals.notna()
        if int(mask.sum()) == 0:
            continue
        x = vals[mask].to_numpy(dtype=float)
        t_sub = t[mask.to_numpy()]
        w_sub = control_weights[mask.to_numpy()]
        tr = t_sub == 1
        co = t_sub == 0
        if int(np.sum(tr)) < 2 or int(np.sum(co)) < 2:
            continue

        x_tr = x[tr]
        x_co = x[co]
        mean_tr = float(np.mean(x_tr))
        mean_co = float(np.mean(x_co))
        var_tr = float(np.var(x_tr, ddof=1))
        var_co = float(np.var(x_co, ddof=1))
        pooled = float(np.sqrt(max((var_tr + var_co) / 2.0, 0.0)))
        smd_pre = float((mean_tr - mean_co) / pooled) if pooled >= 1e-6 else float("nan")

        w_co = w_sub[co]
        w_co = w_co / np.sum(w_co) if np.sum(w_co) > 0 else np.ones_like(w_co) / max(len(w_co), 1)
        mean_co_w = weighted_mean(x_co, w_co)
        var_co_w = weighted_var(x_co, w_co)
        pooled_post = float(np.sqrt(max((var_tr + var_co_w) / 2.0, 0.0)))
        smd_post = float((mean_tr - mean_co_w) / pooled_post) if pooled_post >= 1e-6 else float("nan")

        rows.append(
            {
                "covariate": col,
                "smd_pre": float(smd_pre),
                "smd_post": float(smd_post),
                "abs_smd_pre": float(abs(smd_pre)),
                "abs_smd_post": float(abs(smd_post)),
            }
        )
    return pd.DataFrame(rows)


def estimate_att(
    df: pd.DataFrame,
    e_hat: np.ndarray,
    m0_hat: np.ndarray,
    propensity_clip: float,
) -> tuple[dict[str, Any], np.ndarray]:
    y = pd.to_numeric(df[mc.TARGET_COL], errors="coerce").to_numpy(dtype=float)
    t = df["treatment"].to_numpy(dtype=int)
    n = len(df)
    n_t = int(np.sum(t == 1))
    n_c = int(np.sum(t == 0))
    if n_t == 0 or n_c == 0:
        return {
            "status": "fail_no_overlap",
            "n_rows": n,
            "n_treated": n_t,
            "n_control": n_c,
        }, np.zeros(n, dtype=float)

    e = np.clip(e_hat, propensity_clip, 1.0 - propensity_clip)
    control_weights = np.where(t == 0, e / (1.0 - e), 0.0)

    p1 = float(np.mean(t))
    psi = t * (y - m0_hat) - (1 - t) * control_weights * (y - m0_hat)
    phi = psi / max(p1, 1e-8)
    att = float(np.mean(phi))
    se = float(np.std(phi, ddof=1) / math.sqrt(max(n, 1)))
    if se <= 0 or np.isnan(se):
        z = float("nan")
        p_value = float("nan")
    else:
        z = float(att / se)
        p_value = float(2.0 * (1.0 - NormalDist().cdf(abs(z))))
    ci_low = float(att - 1.96 * se) if np.isfinite(se) else float("nan")
    ci_high = float(att + 1.96 * se) if np.isfinite(se) else float("nan")

    treated_e = e[t == 1]
    control_e = e[t == 0]
    overlap_mid = float(np.mean((e >= 0.1) & (e <= 0.9)))
    clip_frac = float(np.mean((e_hat <= propensity_clip) | (e_hat >= 1.0 - propensity_clip)))
    control_w = control_weights[t == 0]
    if np.sum(control_w**2) > 0:
        ess_control = float((np.sum(control_w) ** 2) / np.sum(control_w**2))
    else:
        ess_control = float("nan")

    result = {
        "status": "ok",
        "n_rows": n,
        "n_treated": n_t,
        "n_control": n_c,
        "treated_rate": p1,
        "att_kg_per_hectare": att,
        "se": se,
        "z_score": z,
        "p_value": p_value,
        "ci_low_95": ci_low,
        "ci_high_95": ci_high,
        "propensity_min_treated": float(np.min(treated_e)),
        "propensity_max_treated": float(np.max(treated_e)),
        "propensity_min_control": float(np.min(control_e)),
        "propensity_max_control": float(np.max(control_e)),
        "overlap_mid_0p1_0p9_share": overlap_mid,
        "clipped_propensity_share": clip_frac,
        "ess_control": ess_control,
    }
    return result, control_weights


def confidence_and_caveats(row: dict[str, Any], max_abs_smd_post: float) -> tuple[str, str]:
    caveats: list[str] = ["assumption_based"]
    if row.get("status") != "ok":
        caveats.append("estimation_failed")
        return "Low", ",".join(caveats)

    overlap = float(row.get("overlap_mid_0p1_0p9_share", np.nan))
    ess = float(row.get("ess_control", np.nan))
    pval = float(row.get("p_value", np.nan))
    tr_rate = float(row.get("treated_rate", np.nan))

    if overlap < 0.6:
        caveats.append("low_overlap")
    if max_abs_smd_post > 0.1:
        caveats.append("residual_imbalance")
    if ess < 80:
        caveats.append("low_effective_sample")
    if tr_rate < 0.15 or tr_rate > 0.85:
        caveats.append("treatment_rate_extreme")

    if overlap >= 0.7 and max_abs_smd_post <= 0.1 and ess >= 120 and pval < 0.05:
        return "High", ",".join(caveats)
    if overlap >= 0.5 and max_abs_smd_post <= 0.2 and ess >= 60 and pval < 0.2:
        return "Medium", ",".join(caveats)
    return "Low", ",".join(caveats)


def run(args: argparse.Namespace) -> pd.DataFrame:
    out_dir = OUT_ROOT / args.run_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    model_features = load_feature_set(Path(args.feature_set_path), args.feature_set_name)
    rules_df = load_rules(Path(args.rules_path))
    treatment_specs = parse_treatment_specs(args.treatments)
    frame = mc.load_analysis_frame()
    diagnostic_features = model_features + [spec.feature for spec in treatment_specs]
    size_coupled_df = detect_hectare_coupled_numeric_features(
        frame=frame,
        candidate_features=diagnostic_features,
        corr_threshold=args.size_coupled_corr_threshold,
    )
    size_coupled_set = set(size_coupled_df.loc[size_coupled_df["size_coupled_flag"], "feature"].astype(str).tolist())

    treatment_rows: list[dict[str, Any]] = []
    covariate_rows: list[dict[str, Any]] = []
    effect_rows: list[dict[str, Any]] = []
    balance_rows: list[dict[str, Any]] = []

    for spec in treatment_specs:
        if spec.feature not in frame.columns:
            effect_rows.append(
                {
                    "treatment_name": spec.name,
                    "treatment_feature": spec.feature,
                    "subgroup_type": "overall",
                    "subgroup_id": "overall",
                    "subgroup_rule": "(all rows)",
                    "status": "fail_missing_treatment_feature",
                }
            )
            continue

        numeric_cutoff_treatment = spec.op in {">", "<="}
        if numeric_cutoff_treatment and spec.feature in size_coupled_set:
            effect_rows.append(
                {
                    "treatment_name": spec.name,
                    "treatment_feature": spec.feature,
                    "subgroup_type": "overall",
                    "subgroup_id": "overall",
                    "subgroup_rule": "(all rows)",
                    "status": "fail_non_identifiable_size_coupled",
                }
            )
            treatment_rows.append(
                {
                    "treatment_name": spec.name,
                    "treatment_feature": spec.feature,
                    "op": spec.op,
                    "value": spec.value,
                    "treatment_day": treatment_day(spec.feature),
                    "n_rows_prepared": 0,
                    "treated_rate_prepared": float("nan"),
                    "size_coupled_treatment_flag": True,
                }
            )
            continue

        excluded_covariates = size_coupled_set if args.drop_size_coupled_covariates else set()
        covariates = select_covariates_for_treatment(
            model_features=model_features,
            treatment_feature=spec.feature,
            excluded_covariates=excluded_covariates,
        )
        prepared = prepare_estimation_frame(
            frame=frame,
            treatment_feature=spec.feature,
            treatment_op=spec.op,
            treatment_value=spec.value,
            covariates=covariates,
        )

        treatment_rows.append(
            {
                "treatment_name": spec.name,
                "treatment_feature": spec.feature,
                "op": spec.op,
                "value": spec.value,
                "treatment_day": treatment_day(spec.feature),
                "n_rows_prepared": int(len(prepared)),
                "treated_rate_prepared": float(prepared["treatment"].mean()) if len(prepared) > 0 else float("nan"),
                "size_coupled_treatment_flag": spec.feature in size_coupled_set,
            }
        )
        for rank, cov in enumerate(covariates, start=1):
            covariate_rows.append(
                {
                    "treatment_name": spec.name,
                    "treatment_feature": spec.feature,
                    "covariate_rank": rank,
                    "covariate": cov,
                }
            )

        # Rule linkage uses residual segments: rule conditions excluding treatment condition.
        # This avoids deterministic all-treated/all-control subgroups.
        feature_name = spec.feature
        relevant_rules = rules_df[
            rules_df["rule_conditions"].astype(str).map(
                lambda rule_text, f=feature_name: rule_mentions_feature(rule_text, f)
            )
        ].copy()
        subgroup_defs: list[tuple[str, str, str, pd.Series]] = [
            ("overall", "overall", "(all rows)", pd.Series(True, index=prepared.index))
        ]
        for _, rr in relevant_rules.iterrows():
            rid = str(rr["rule_id"])
            all_conditions = parse_rule_conditions(str(rr["rule_conditions"]))
            residual_conditions = [c for c in all_conditions if c[0] != spec.feature]
            if not residual_conditions:
                continue
            residual_text = " AND ".join(condition_to_text(c) for c in residual_conditions)
            residual_mask = eval_conditions_mask(prepared, residual_conditions)
            subgroup_defs.append(("rule_residual_active", rid, residual_text, residual_mask))
            if args.include_inactive_rule_segments:
                subgroup_defs.append(("rule_residual_inactive", rid, residual_text, ~residual_mask))

        for subgroup_type, subgroup_id, subgroup_rule, subgroup_mask in subgroup_defs:
            sub = prepared[subgroup_mask].copy().reset_index(drop=True)
            n_sub = int(len(sub))
            n_t = int(np.sum(sub["treatment"] == 1))
            n_c = int(np.sum(sub["treatment"] == 0))
            n_groups = int(sub[mc.GROUP_COL].nunique()) if n_sub > 0 else 0
            treated_share = (n_t / n_sub) if n_sub > 0 else float("nan")

            if (
                n_sub < args.min_subgroup_rows
                or n_t < args.min_treated_rows
                or n_c < args.min_control_rows
                or n_groups < 2
                or treated_share < args.min_treated_share
                or treated_share > args.max_treated_share
            ):
                effect_rows.append(
                    {
                        "treatment_name": spec.name,
                        "treatment_feature": spec.feature,
                        "subgroup_type": subgroup_type,
                        "subgroup_id": subgroup_id,
                        "subgroup_rule": subgroup_rule,
                        "status": "fail_insufficient_support",
                        "n_rows": n_sub,
                        "n_treated": n_t,
                        "n_control": n_c,
                        "n_groups": n_groups,
                        "treated_rate": treated_share,
                    }
                )
                continue

            e_hat, m0_hat, nuisance_status = fit_crossfit_nuisance(
                df=sub,
                covariates=covariates + [mc.GROUP_COL],
                n_splits=args.crossfit_splits,
                random_state=args.random_state,
            )
            if nuisance_status != "ok":
                effect_rows.append(
                    {
                        "treatment_name": spec.name,
                        "treatment_feature": spec.feature,
                        "subgroup_type": subgroup_type,
                        "subgroup_id": subgroup_id,
                        "subgroup_rule": subgroup_rule,
                        "status": nuisance_status,
                        "n_rows": n_sub,
                        "n_treated": n_t,
                        "n_control": n_c,
                        "n_groups": n_groups,
                    }
                )
                continue

            est, control_weights = estimate_att(
                df=sub,
                e_hat=e_hat,
                m0_hat=m0_hat,
                propensity_clip=args.propensity_clip,
            )

            balance_df = compute_balance_table(
                df=sub,
                covariates=covariates,
                control_weights=control_weights,
            )
            max_abs_smd_post = float(balance_df["abs_smd_post"].max()) if not balance_df.empty else float("nan")
            conf, caveats = confidence_and_caveats(
                est, max_abs_smd_post=max_abs_smd_post if np.isfinite(max_abs_smd_post) else 999.0
            )

            effect_row = {
                "treatment_name": spec.name,
                "treatment_feature": spec.feature,
                "subgroup_type": subgroup_type,
                "subgroup_id": subgroup_id,
                "subgroup_rule": subgroup_rule,
                "confidence_level": conf,
                "caveats": caveats,
                "max_abs_smd_post": max_abs_smd_post,
                "balance_covariate_count": int(len(balance_df)),
                **est,
            }
            effect_rows.append(effect_row)

            if not balance_df.empty:
                for _, br in balance_df.iterrows():
                    balance_rows.append(
                        {
                            "treatment_name": spec.name,
                            "subgroup_type": subgroup_type,
                            "subgroup_id": subgroup_id,
                            "covariate": br["covariate"],
                            "smd_pre": float(br["smd_pre"]),
                            "smd_post": float(br["smd_post"]),
                            "abs_smd_pre": float(br["abs_smd_pre"]),
                            "abs_smd_post": float(br["abs_smd_post"]),
                        }
                    )

    treatment_df = pd.DataFrame(treatment_rows)
    covariate_df = pd.DataFrame(covariate_rows)
    effects_df = pd.DataFrame(effect_rows)
    balance_out_df = pd.DataFrame(balance_rows)

    treatment_df.to_csv(out_dir / "treatment_definitions.csv", index=False)
    covariate_df.to_csv(out_dir / "treatment_covariates.csv", index=False)
    effects_df.to_csv(out_dir / "causal_effect_estimates.csv", index=False)
    balance_out_df.to_csv(out_dir / "causal_balance_diagnostics.csv", index=False)
    size_coupled_df.to_csv(out_dir / "size_coupled_feature_diagnostics.csv", index=False)

    overall_df = effects_df[effects_df["subgroup_type"] == "overall"].copy()
    overall_df.to_csv(out_dir / "causal_overall_summary.csv", index=False)
    rule_df = effects_df[effects_df["subgroup_type"] != "overall"].copy()
    rule_df.to_csv(out_dir / "causal_rule_segment_summary.csv", index=False)

    payload = {
        "run_tag": args.run_tag,
        "feature_set_name": args.feature_set_name,
        "assumption_note": (
            "Causal estimates are assumption-based (ignorability, overlap, consistency, SUTVA) "
            "with agriblock-aware nuisance models."
        ),
        "overall_effects": overall_df.to_dict(orient="records"),
        "rule_segment_effects": rule_df.to_dict(orient="records"),
    }
    (out_dir / "ui_causal_payload.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    runlog = [
        f"run_tag={args.run_tag}",
        f"feature_set={args.feature_set_name}",
        f"n_treatments={len(treatment_specs)}",
        f"min_subgroup_rows={args.min_subgroup_rows}",
        f"min_treated_rows={args.min_treated_rows}",
        f"min_control_rows={args.min_control_rows}",
        f"min_treated_share={args.min_treated_share}",
        f"max_treated_share={args.max_treated_share}",
        f"propensity_clip={args.propensity_clip}",
        f"size_coupled_corr_threshold={args.size_coupled_corr_threshold}",
        f"drop_size_coupled_covariates={args.drop_size_coupled_covariates}",
        f"include_inactive_rule_segments={args.include_inactive_rule_segments}",
        f"effects_rows={len(effects_df)}",
        f"overall_rows={len(overall_df)}",
        f"rule_rows={len(rule_df)}",
        f"size_coupled_features={len(size_coupled_set)}",
    ]
    (out_dir / "causal_runlog.txt").write_text("\n".join(runlog), encoding="utf-8")

    print(f"Saved causal pilot outputs to: {out_dir}")
    if not overall_df.empty:
        ok = overall_df[overall_df["status"] == "ok"].copy()
        if not ok.empty:
            top = ok.sort_values("att_kg_per_hectare", ascending=False).iloc[0]
            print(
                "Top overall ATT:"
                f" {top['treatment_name']} | ATT={top['att_kg_per_hectare']:.2f} kg/ha"
                f" | CI=({top['ci_low_95']:.2f}, {top['ci_high_95']:.2f})"
                f" | confidence={top['confidence_level']}"
            )
    return effects_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rule-linked agriblock-aware causal pilot.")
    parser.add_argument("--run-tag", type=str, default="latest")
    parser.add_argument("--feature-set-path", type=str, default=str(DEFAULT_SCENARIO_PATH))
    parser.add_argument("--feature-set-name", type=str, default="full_review")
    parser.add_argument("--rules-path", type=str, default=str(DEFAULT_RULES_PATH))
    parser.add_argument(
        "--treatments",
        type=str,
        default="",
        help="Optional CSV list of name|feature|op|value specs. Operators: >, <=, ==, !=.",
    )
    parser.add_argument("--crossfit-splits", type=int, default=3)
    parser.add_argument("--propensity-clip", type=float, default=0.03)
    parser.add_argument("--min-subgroup-rows", type=int, default=160)
    parser.add_argument("--min-treated-rows", type=int, default=40)
    parser.add_argument("--min-control-rows", type=int, default=40)
    parser.add_argument("--min-treated-share", type=float, default=0.15)
    parser.add_argument("--max-treated-share", type=float, default=0.85)
    parser.add_argument("--size-coupled-corr-threshold", type=float, default=0.999)
    parser.add_argument(
        "--drop-size-coupled-covariates",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--include-inactive-rule-segments", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
