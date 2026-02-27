"""Rule-as-treatment causal inference for CAROB using AIPW."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold
from sklearn.pipeline import Pipeline

from paddy_yield_ml.pipelines import carob_common as cc
from paddy_yield_ml.pipelines import carob_interpretability as ci
from paddy_yield_ml.pipelines import carob_model_compare as cm

try:
    project_root = Path(__file__).resolve().parents[3]
except NameError:
    project_root = Path.cwd()

OUT_ROOT = project_root / "outputs" / "carob_rule_causal_aipw"
DEFAULT_INTERP_DIR = project_root / "outputs" / "carob_interpretability" / "iter3_defensible_v5"
DEFAULT_SEEDS = "42,52,62"
DEFAULT_PRIMARY_STATUSES = "works_here"
DEFAULT_SECONDARY_STATUSES = "conflicts_here,unstable_or_small_effect"


@dataclass(frozen=True)
class PairSpec:
    rule_id: str
    country: str
    generalization_status: str
    analysis_tier: str
    rule_conditions: str


@dataclass
class PairData:
    pair: PairSpec
    frame: pd.DataFrame
    x: pd.DataFrame
    y: pd.Series
    a: pd.Series
    trials: pd.Series
    covariates: list[str]
    rule_features: list[str]


def parse_seed_list(raw: str) -> list[int]:
    out = [int(x.strip()) for x in str(raw).split(",") if x.strip()]
    if not out:
        raise ValueError("At least one seed is required.")
    return list(dict.fromkeys(out))


def parse_status_list(raw: str) -> list[str]:
    out = [x.strip() for x in str(raw).split(",") if x.strip()]
    if not out:
        raise ValueError("At least one status is required.")
    return cm.dedupe_keep_order(out)


def load_scenario_features(interp_dir: Path) -> list[str]:
    path = interp_dir / "scenario_feature_set.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing scenario feature set file: {path}")
    sdf = pd.read_csv(path)
    required = {"feature"}
    if not required.issubset(sdf.columns):
        raise ValueError(f"Scenario feature set missing columns: {sorted(required)}")
    features = sdf["feature"].astype(str).dropna().tolist()
    return cm.dedupe_keep_order(features)


def load_pair_catalog(
    interp_dir: Path,
    *,
    primary_statuses: list[str],
    secondary_statuses: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rules_path = interp_dir / "iteration3_rules_final.csv"
    country_path = interp_dir / "iteration3_rule_country_generalization.csv"
    if not rules_path.exists():
        raise FileNotFoundError(f"Missing rules file: {rules_path}")
    if not country_path.exists():
        raise FileNotFoundError(f"Missing country generalization file: {country_path}")

    rules = pd.read_csv(rules_path)
    ctry = pd.read_csv(country_path)
    req_rules = {"rule_id", "rule_conditions"}
    req_ctry = {"rule_id", "country", "generalization_status"}
    if not req_rules.issubset(rules.columns):
        raise ValueError(f"Rules file missing columns: {sorted(req_rules)}")
    if not req_ctry.issubset(ctry.columns):
        raise ValueError(f"Country generalization file missing columns: {sorted(req_ctry)}")

    final_rule_ids = set(rules["rule_id"].astype(str).tolist())
    ctry = ctry.copy()
    ctry["rule_id"] = ctry["rule_id"].astype(str)
    ctry["country"] = ctry["country"].astype(str)
    ctry["generalization_status"] = ctry["generalization_status"].astype(str)
    ctry = ctry[ctry["rule_id"].isin(final_rule_ids)].copy()

    target_statuses = set(primary_statuses + secondary_statuses)
    ctry = ctry[ctry["generalization_status"].isin(target_statuses)].copy()
    if ctry.empty:
        raise ValueError("No pair rows left after status filtering. Check status configuration.")

    tier_map: dict[str, str] = {}
    for s in primary_statuses:
        tier_map[s] = "primary"
    for s in secondary_statuses:
        tier_map[s] = "secondary"
    ctry["analysis_tier"] = ctry["generalization_status"].map(tier_map).fillna("secondary")

    merge = pd.merge(
        ctry,
        rules[["rule_id", "rule_conditions"]],
        on="rule_id",
        how="left",
    ).drop_duplicates(subset=["rule_id", "country", "generalization_status"], keep="first")
    merge = merge.sort_values(["analysis_tier", "rule_id", "country"]).reset_index(drop=True)
    return merge, rules


def impute_numeric_by_trial(df: pd.DataFrame, trial_col: str) -> pd.DataFrame:
    out = df.copy()
    num_cols = out.select_dtypes(include=[np.number]).columns.tolist()
    if not num_cols:
        return out

    g = out[trial_col].astype(str)
    med_group = out[num_cols].groupby(g).median(numeric_only=True)
    med_global = out[num_cols].median(numeric_only=True)

    for col in num_cols:
        fill_group = g.map(med_group[col]) if col in med_group.columns else np.nan
        out[col] = (
            pd.to_numeric(out[col], errors="coerce")
            .fillna(fill_group)
            .fillna(float(med_global.get(col, np.nan)))
        )
    return out


def select_covariates(
    x: pd.DataFrame,
    *,
    rule_features: list[str],
) -> list[str]:
    blocked = set(rule_features)
    out: list[str] = []
    for c in x.columns:
        if c in blocked:
            continue
        s = x[c]
        nunique = int(pd.Series(s).nunique(dropna=True))
        if nunique < 2:
            continue
        out.append(c)
    return out


def build_pair_data(
    frame: pd.DataFrame,
    *,
    features: list[str],
    pair: PairSpec,
) -> PairData:
    cols = list(dict.fromkeys(features + [cc.TARGET_COL, cc.GROUP_COL, "country"]))
    missing = [c for c in cols if c not in frame.columns]
    if missing:
        raise ValueError(f"Frame missing required columns for pair prep: {missing}")

    sub = frame[cols].copy()
    sub = sub.loc[:, ~sub.columns.duplicated()].copy()
    sub = sub[sub["country"].astype(str) == pair.country].copy()
    sub[cc.TARGET_COL] = pd.to_numeric(sub[cc.TARGET_COL], errors="coerce")
    sub = sub[sub[cc.TARGET_COL].notna() & sub[cc.GROUP_COL].notna()].reset_index(drop=True)
    if sub.empty:
        raise ValueError(f"No rows left for pair {pair.rule_id}-{pair.country}.")

    x = sub[features].copy()
    x[cc.GROUP_COL] = sub[cc.GROUP_COL].astype(str)
    x = impute_numeric_by_trial(x, trial_col=cc.GROUP_COL)

    rule_features = sorted({f for f, _op, _thr in ci.parse_rule_conditions(pair.rule_conditions)})
    a_mask = ci.apply_rule_conditions(x, pair.rule_conditions).astype(bool)
    a = a_mask.astype(int)

    y = pd.to_numeric(sub[cc.TARGET_COL], errors="coerce").astype(float)
    trials = sub[cc.GROUP_COL].astype(str)

    x_model = x.copy()
    x_model["__trial_context__"] = trials.astype(str)
    if cc.GROUP_COL in x_model.columns:
        x_model = x_model.drop(columns=[cc.GROUP_COL])

    covars = select_covariates(x_model, rule_features=rule_features)
    if not covars:
        raise ValueError(f"No covariates left after excluding rule features for {pair.rule_id}-{pair.country}.")

    return PairData(
        pair=pair,
        frame=sub,
        x=x_model[covars].copy(),
        y=y.reset_index(drop=True),
        a=a.reset_index(drop=True),
        trials=trials.reset_index(drop=True),
        covariates=covars,
        rule_features=rule_features,
    )


def pair_readiness_row(
    pdata: PairData,
    *,
    min_total: int,
    min_arm: int,
    min_trials_per_arm: int,
) -> dict[str, Any]:
    n_total = int(len(pdata.y))
    n_treated = int((pdata.a == 1).sum())
    n_control = int((pdata.a == 0).sum())

    tmp = pd.DataFrame({"trial_id": pdata.trials, "a": pdata.a})
    treat_trials = int(tmp.loc[tmp["a"] == 1, "trial_id"].nunique())
    control_trials = int(tmp.loc[tmp["a"] == 0, "trial_id"].nunique())

    ready = bool(
        (n_total >= int(min_total))
        and (n_treated >= int(min_arm))
        and (n_control >= int(min_arm))
        and (treat_trials >= int(min_trials_per_arm))
        and (control_trials >= int(min_trials_per_arm))
    )

    return {
        "rule_id": pdata.pair.rule_id,
        "country": pdata.pair.country,
        "generalization_status": pdata.pair.generalization_status,
        "analysis_tier": pdata.pair.analysis_tier,
        "n_total": n_total,
        "n_treated": n_treated,
        "n_control": n_control,
        "treated_rate": float(n_treated / n_total) if n_total > 0 else np.nan,
        "n_trials": int(pd.Series(pdata.trials).nunique()),
        "n_trials_treated": treat_trials,
        "n_trials_control": control_trials,
        "estimable_by_gates": ready,
    }


def build_cv_splits(
    groups: pd.Series,
    a: pd.Series,
    *,
    n_splits: int,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    g = groups.astype(str).reset_index(drop=True)
    av = pd.to_numeric(a, errors="coerce").fillna(0).astype(int).reset_index(drop=True)
    n = len(g)
    idx = np.arange(n, dtype=int)

    unique_groups = int(g.nunique())
    if unique_groups >= 2 and n_splits >= 2 and int(av.nunique()) >= 2:
        k = min(int(n_splits), unique_groups)
        cv = GroupKFold(n_splits=k)
        candidate = list(cv.split(idx, groups=g))
        group_ok = True
        for tr_idx, _te_idx in candidate:
            if int(pd.Series(av.iloc[tr_idx]).nunique()) < 2:
                group_ok = False
                break
        if group_ok:
            return candidate

    if int(av.nunique()) >= 2:
        k = max(2, min(int(n_splits), n, int(av.value_counts().min())))
        cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
        return list(cv.split(idx, av))

    k = max(2, min(int(n_splits), n))
    cv = KFold(n_splits=k, shuffle=True, random_state=seed)
    return list(cv.split(idx))


def fit_predict_nuisance_crossfit(
    x: pd.DataFrame,
    y: pd.Series,
    a: pd.Series,
    trials: pd.Series,
    *,
    seed: int,
    n_splits: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(y)
    e_hat = np.full(n, np.nan, dtype=float)
    m1_hat = np.full(n, np.nan, dtype=float)
    m0_hat = np.full(n, np.nan, dtype=float)

    splits = build_cv_splits(trials, a, n_splits=n_splits, seed=seed)
    for tr_idx, te_idx in splits:
        xtr = x.iloc[tr_idx].copy()
        xte = x.iloc[te_idx].copy()
        ytr = y.iloc[tr_idx].copy()
        atr = a.iloc[tr_idx].copy()

        if int(pd.Series(atr).nunique()) < 2:
            raise ValueError("Training fold has only one treatment class; cannot fit propensity model.")

        prop = Pipeline(
            [
                ("preprocess", cm.make_preprocessor(xtr)),
                ("logit", LogisticRegression(max_iter=3000, solver="lbfgs", class_weight="balanced")),
            ]
        )
        prop.fit(xtr, atr)
        e_hat[te_idx] = prop.predict_proba(xte)[:, 1]

        arm1_idx = np.where(atr.to_numpy(dtype=int) == 1)[0]
        arm0_idx = np.where(atr.to_numpy(dtype=int) == 0)[0]
        if len(arm1_idx) < 8 or len(arm0_idx) < 8:
            raise ValueError("Training fold has too few treated/control rows for outcome modeling.")

        reg1 = Pipeline(
            [
                ("preprocess", cm.make_preprocessor(xtr.iloc[arm1_idx])),
                (
                    "rf",
                    RandomForestRegressor(
                        n_estimators=400,
                        min_samples_leaf=5,
                        random_state=seed + 17,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
        reg0 = Pipeline(
            [
                ("preprocess", cm.make_preprocessor(xtr.iloc[arm0_idx])),
                (
                    "rf",
                    RandomForestRegressor(
                        n_estimators=400,
                        min_samples_leaf=5,
                        random_state=seed + 29,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
        reg1.fit(xtr.iloc[arm1_idx], ytr.iloc[arm1_idx])
        reg0.fit(xtr.iloc[arm0_idx], ytr.iloc[arm0_idx])
        m1_hat[te_idx] = reg1.predict(xte)
        m0_hat[te_idx] = reg0.predict(xte)

    if np.isnan(e_hat).any() or np.isnan(m1_hat).any() or np.isnan(m0_hat).any():
        raise RuntimeError("Cross-fit nuisance prediction produced NaNs.")
    return e_hat, m1_hat, m0_hat


def aipw_from_nuisance(
    y: pd.Series,
    a: pd.Series,
    e_hat: np.ndarray,
    m1_hat: np.ndarray,
    m0_hat: np.ndarray,
    *,
    propensity_clip: float,
) -> tuple[np.ndarray, float]:
    yv = pd.to_numeric(y, errors="coerce").to_numpy(dtype=float)
    av = pd.to_numeric(a, errors="coerce").to_numpy(dtype=float)
    lo = float(propensity_clip)
    hi = float(1.0 - propensity_clip)
    e = np.clip(np.asarray(e_hat, dtype=float), lo, hi)
    m1 = np.asarray(m1_hat, dtype=float)
    m0 = np.asarray(m0_hat, dtype=float)

    phi = m1 - m0 + (av * (yv - m1) / e) - ((1.0 - av) * (yv - m0) / (1.0 - e))
    ate = float(np.mean(phi))
    return phi, ate


def standardized_mean_diff(
    x: pd.Series,
    a: pd.Series,
    *,
    weights: np.ndarray | None = None,
) -> float:
    xv = pd.to_numeric(x, errors="coerce")
    av = pd.to_numeric(a, errors="coerce")
    valid = xv.notna() & av.notna()
    xv = xv[valid]
    av = av[valid].astype(int)
    if len(xv) == 0:
        return float("nan")

    treat = xv[av == 1].to_numpy(dtype=float)
    ctrl = xv[av == 0].to_numpy(dtype=float)
    if len(treat) == 0 or len(ctrl) == 0:
        return float("nan")

    if weights is None:
        mt = float(np.mean(treat))
        mc = float(np.mean(ctrl))
        vt = float(np.var(treat, ddof=1)) if len(treat) > 1 else 0.0
        vc = float(np.var(ctrl, ddof=1)) if len(ctrl) > 1 else 0.0
    else:
        w = np.asarray(weights, dtype=float)[valid.to_numpy()]
        wt = w[av.to_numpy() == 1]
        wc = w[av.to_numpy() == 0]
        if np.sum(wt) <= 0 or np.sum(wc) <= 0:
            return float("nan")
        mt = float(np.sum(wt * treat) / np.sum(wt))
        mc = float(np.sum(wc * ctrl) / np.sum(wc))
        vt = float(np.sum(wt * (treat - mt) ** 2) / np.sum(wt))
        vc = float(np.sum(wc * (ctrl - mc) ** 2) / np.sum(wc))

    denom = np.sqrt(max((vt + vc) / 2.0, 1e-12))
    return float((mt - mc) / denom)


def balance_table(
    x: pd.DataFrame,
    a: pd.Series,
    e_hat: np.ndarray,
    *,
    clip: float,
) -> pd.DataFrame:
    e = np.clip(np.asarray(e_hat, dtype=float), clip, 1.0 - clip)
    av = pd.to_numeric(a, errors="coerce").to_numpy(dtype=float)
    w = (av / e) + ((1.0 - av) / (1.0 - e))

    num_cols = x.select_dtypes(include=[np.number]).columns.tolist()
    rows: list[dict[str, Any]] = []
    for col in num_cols:
        smd_b = standardized_mean_diff(x[col], a, weights=None)
        smd_a = standardized_mean_diff(x[col], a, weights=w)
        rows.append(
            {
                "feature": col,
                "smd_before": smd_b,
                "smd_after": smd_a,
                "abs_smd_before": abs(smd_b) if pd.notna(smd_b) else np.nan,
                "abs_smd_after": abs(smd_a) if pd.notna(smd_a) else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("abs_smd_after", ascending=False).reset_index(drop=True)


def effective_sample_size(weights: np.ndarray) -> float:
    w = np.asarray(weights, dtype=float)
    denom = float(np.sum(w**2))
    if denom <= 0:
        return float("nan")
    num = float(np.sum(w) ** 2)
    return float(num / denom)


def cluster_bootstrap_ci(
    phi: np.ndarray,
    clusters: pd.Series,
    *,
    n_bootstrap: int,
    seed: int,
) -> tuple[float, float, float]:
    ph = np.asarray(phi, dtype=float)
    cl = clusters.astype(str).reset_index(drop=True)
    if len(ph) != len(cl):
        raise ValueError("phi and clusters length mismatch")

    unique_cl = cl.unique().tolist()
    if len(unique_cl) < 2:
        m = float(np.mean(ph))
        return m, m, np.nan

    rng = np.random.default_rng(seed)
    idx_by_cluster: dict[str, np.ndarray] = {
        c: np.where(cl.to_numpy(dtype=object) == c)[0] for c in unique_cl
    }

    boot: list[float] = []
    k = len(unique_cl)
    for _ in range(int(n_bootstrap)):
        sampled = rng.choice(unique_cl, size=k, replace=True)
        parts = [idx_by_cluster[str(c)] for c in sampled]
        if not parts:
            continue
        idx = np.concatenate(parts)
        boot.append(float(np.mean(ph[idx])))

    if not boot:
        m = float(np.mean(ph))
        return m, m, np.nan

    arr = np.asarray(boot, dtype=float)
    low = float(np.quantile(arr, 0.025))
    high = float(np.quantile(arr, 0.975))
    p_two = float(2.0 * min(np.mean(arr <= 0.0), np.mean(arr >= 0.0)))
    return low, high, p_two


def estimate_pair_seed(
    pdata: PairData,
    *,
    seed: int,
    n_folds: int,
    propensity_clip: float,
    n_bootstrap: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    e_hat, m1_hat, m0_hat = fit_predict_nuisance_crossfit(
        x=pdata.x,
        y=pdata.y,
        a=pdata.a,
        trials=pdata.trials,
        seed=seed,
        n_splits=n_folds,
    )
    phi, ate = aipw_from_nuisance(
        y=pdata.y,
        a=pdata.a,
        e_hat=e_hat,
        m1_hat=m1_hat,
        m0_hat=m0_hat,
        propensity_clip=propensity_clip,
    )

    e = np.clip(e_hat, propensity_clip, 1.0 - propensity_clip)
    a = pdata.a.to_numpy(dtype=int)
    wt_t = 1.0 / e[a == 1]
    wt_c = 1.0 / (1.0 - e[a == 0])

    e_t = e[a == 1]
    e_c = e[a == 0]
    common_low = float(max(np.min(e_t), np.min(e_c)))
    common_high = float(min(np.max(e_t), np.max(e_c)))
    overlap_frac = float(np.mean((e >= common_low) & (e <= common_high)))

    bal = balance_table(pdata.x, pdata.a, e_hat=e_hat, clip=propensity_clip)
    max_abs_smd_before = float(pd.to_numeric(bal["abs_smd_before"], errors="coerce").max()) if not bal.empty else np.nan
    max_abs_smd_after = float(pd.to_numeric(bal["abs_smd_after"], errors="coerce").max()) if not bal.empty else np.nan

    ci_low, ci_high, p_two = cluster_bootstrap_ci(
        phi=phi,
        clusters=pdata.trials,
        n_bootstrap=n_bootstrap,
        seed=seed + 999,
    )
    ci_excludes_zero = bool((ci_low > 0 and ci_high > 0) or (ci_low < 0 and ci_high < 0))

    diag = {
        "rule_id": pdata.pair.rule_id,
        "country": pdata.pair.country,
        "generalization_status": pdata.pair.generalization_status,
        "analysis_tier": pdata.pair.analysis_tier,
        "seed": int(seed),
        "n": int(len(pdata.y)),
        "n_treated": int((pdata.a == 1).sum()),
        "n_control": int((pdata.a == 0).sum()),
        "ate_aipw": ate,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "ci_excludes_zero": ci_excludes_zero,
        "p_value_bootstrap_two_sided": p_two,
        "propensity_min": float(np.min(e)),
        "propensity_max": float(np.max(e)),
        "propensity_common_low": common_low,
        "propensity_common_high": common_high,
        "overlap_fraction_common_support": overlap_frac,
        "ess_treated": effective_sample_size(wt_t),
        "ess_control": effective_sample_size(wt_c),
        "max_abs_smd_before": max_abs_smd_before,
        "max_abs_smd_after": max_abs_smd_after,
    }
    bal.insert(0, "seed", int(seed))
    bal.insert(0, "country", pdata.pair.country)
    bal.insert(0, "rule_id", pdata.pair.rule_id)
    return diag, bal


def summarize_pair_estimates(seed_df: pd.DataFrame) -> pd.DataFrame:
    if seed_df.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for pair_key, grp in seed_df.groupby(["rule_id", "country"], dropna=False):
        if isinstance(pair_key, tuple) and len(pair_key) == 2:
            rule_id, country = pair_key
        else:
            rule_id, country = pair_key, ""
        grp = grp.sort_values("seed").reset_index(drop=True)
        ref = grp.iloc[0]
        ate_vals = pd.to_numeric(grp["ate_aipw"], errors="coerce")
        sign_ref = float(np.sign(float(ref["ate_aipw"])))
        sign_match = float(np.mean(np.sign(ate_vals.replace(0.0, np.nan)).fillna(0.0) == sign_ref))
        rows.append(
            {
                "rule_id": str(rule_id),
                "country": str(country),
                "generalization_status": str(ref["generalization_status"]),
                "analysis_tier": str(ref["analysis_tier"]),
                "n_seed_runs": int(len(grp)),
                "ate_ref_seed": float(ref["ate_aipw"]),
                "ci_low_ref_seed": float(ref["ci_low"]),
                "ci_high_ref_seed": float(ref["ci_high"]),
                "ci_excludes_zero_ref_seed": bool(ref["ci_excludes_zero"]),
                "ate_mean_across_seeds": float(np.mean(ate_vals)),
                "ate_std_across_seeds": float(np.std(ate_vals)),
                "sign_match_rate": sign_match,
                "overlap_fraction_common_support_ref_seed": float(ref["overlap_fraction_common_support"]),
                "max_abs_smd_after_ref_seed": float(ref["max_abs_smd_after"]),
                "ess_treated_ref_seed": float(ref["ess_treated"]),
                "ess_control_ref_seed": float(ref["ess_control"]),
                "p_value_bootstrap_two_sided_ref_seed": float(ref["p_value_bootstrap_two_sided"]),
            }
        )
    return pd.DataFrame(rows).sort_values(["analysis_tier", "rule_id", "country"]).reset_index(drop=True)


def build_recommendation(score_row: pd.Series) -> tuple[str, str]:
    status = str(score_row["generalization_status"])
    ci_ok = bool(score_row["ci_excludes_zero_ref_seed"])
    sign_ok = float(score_row["sign_match_rate"]) >= 0.67
    overlap_ok = float(score_row["overlap_fraction_common_support_ref_seed"]) >= 0.60
    balance_ok = float(score_row["max_abs_smd_after_ref_seed"]) <= 0.20
    effect_meaningful = abs(float(score_row["ate_mean_across_seeds"])) >= 100.0
    trials_t = int(score_row.get("n_trials_treated", 0))
    trials_c = int(score_row.get("n_trials_control", 0))
    trial_diversity_ok = min(trials_t, trials_c) >= 2

    if status == "works_here":
        if ci_ok and sign_ok and overlap_ok and balance_ok and effect_meaningful and trial_diversity_ok:
            return "Recommend", "passes_effect_significance_stability_overlap_balance"
        if ci_ok and sign_ok and overlap_ok and balance_ok and effect_meaningful and not trial_diversity_ok:
            return "Pilot-only", "promising_signal_but_low_trial_diversity_in_one_arm"
        return "Pilot-only", "works_here_but_some_causal_diagnostics_not_fully_met"

    if status == "conflicts_here":
        return "Do-not-recommend", "predictive_rule_direction_conflicts_in_country"

    if status == "unstable_or_small_effect":
        if ci_ok and sign_ok and overlap_ok and balance_ok:
            return "Pilot-only", "causal_signal_present_but_predictive_transfer_was_unstable"
        return "Do-not-recommend", "unstable_or_small_effect_and_causal_diagnostics_weak"

    return "Do-not-recommend", "unsupported_status"


def build_scorecard(summary_df: pd.DataFrame, readiness_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df.empty:
        return pd.DataFrame()
    read_cols = [
        "rule_id",
        "country",
        "n_total",
        "n_treated",
        "n_control",
        "treated_rate",
        "n_trials",
        "n_trials_treated",
        "n_trials_control",
        "estimable_by_gates",
    ]
    merged = pd.merge(summary_df, readiness_df[read_cols], on=["rule_id", "country"], how="left")
    recs: list[str] = []
    notes: list[str] = []
    for _, row in merged.iterrows():
        rec, note = build_recommendation(row)
        recs.append(rec)
        notes.append(note)
    merged["recommendation"] = recs
    merged["decision_reason"] = notes
    return merged.sort_values(["analysis_tier", "rule_id", "country"]).reset_index(drop=True)


def write_playbook_overlay(
    out_path: Path,
    *,
    scorecard_df: pd.DataFrame,
) -> None:
    lines: list[str] = [
        "# Causal Overlay for Rule-Based Recommendations (AIPW)",
        "",
        "This overlay evaluates rule-as-treatment effects by country.",
        "Primary claims are for `works_here` pairs; secondary pairs are diagnostics-only.",
        "",
    ]
    if scorecard_df.empty:
        lines.append("- No estimable pairs.")
        out_path.write_text("\n".join(lines), encoding="utf-8")
        return

    for tier in ["primary", "secondary"]:
        part = scorecard_df[scorecard_df["analysis_tier"] == tier].copy()
        if part.empty:
            continue
        lines += [f"## {tier.capitalize()} Pairs", ""]
        for _, r in part.iterrows():
            lines.append(
                f"- `{r['rule_id']}-{r['country']}` [{r['generalization_status']}] -> "
                f"ATE={float(r['ate_ref_seed']):+.1f} kg/ha "
                f"(95% CI {float(r['ci_low_ref_seed']):+.1f}, {float(r['ci_high_ref_seed']):+.1f}), "
                f"sign_match={float(r['sign_match_rate']):.2f}, "
                f"overlap={float(r['overlap_fraction_common_support_ref_seed']):.2f}, "
                f"max|SMD|={float(r['max_abs_smd_after_ref_seed']):.2f}, "
                f"decision=`{r['recommendation']}` ({r['decision_reason']})."
            )
        lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rule-as-treatment causal inference with AIPW.")
    parser.add_argument("--run-tag", type=str, default="latest")
    parser.add_argument("--interp-dir", type=str, default=str(DEFAULT_INTERP_DIR))
    parser.add_argument("--seeds", type=str, default=DEFAULT_SEEDS)
    parser.add_argument("--primary-statuses", type=str, default=DEFAULT_PRIMARY_STATUSES)
    parser.add_argument("--secondary-statuses", type=str, default=DEFAULT_SECONDARY_STATUSES)
    parser.add_argument("--min-total", type=int, default=80)
    parser.add_argument("--min-arm", type=int, default=25)
    parser.add_argument("--min-trials-per-arm", type=int, default=1)
    parser.add_argument("--n-folds", type=int, default=3)
    parser.add_argument("--propensity-clip", type=float, default=0.05)
    parser.add_argument("--n-bootstrap", type=int, default=400)
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = OUT_ROOT / args.run_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    interp_dir = Path(args.interp_dir)
    seeds = parse_seed_list(args.seeds)
    primary_statuses = parse_status_list(args.primary_statuses)
    secondary_statuses = parse_status_list(args.secondary_statuses)

    pair_catalog_df, _rules_df = load_pair_catalog(
        interp_dir=interp_dir,
        primary_statuses=primary_statuses,
        secondary_statuses=secondary_statuses,
    )
    pair_catalog_df.to_csv(out_dir / "pair_catalog.csv", index=False)

    features = load_scenario_features(interp_dir)
    frame = cc.load_analysis_frame(require_treatment=False)

    readiness_rows: list[dict[str, Any]] = []
    seed_rows: list[dict[str, Any]] = []
    balance_rows: list[pd.DataFrame] = []
    skip_rows: list[dict[str, Any]] = []

    for _, row in pair_catalog_df.iterrows():
        pair = PairSpec(
            rule_id=str(row["rule_id"]),
            country=str(row["country"]),
            generalization_status=str(row["generalization_status"]),
            analysis_tier=str(row["analysis_tier"]),
            rule_conditions=str(row["rule_conditions"]),
        )
        try:
            pdata = build_pair_data(frame=frame, features=features, pair=pair)
            ready = pair_readiness_row(
                pdata=pdata,
                min_total=int(args.min_total),
                min_arm=int(args.min_arm),
                min_trials_per_arm=int(args.min_trials_per_arm),
            )
            readiness_rows.append(ready)
            if not bool(ready["estimable_by_gates"]):
                skip_rows.append(
                    {
                        "rule_id": pair.rule_id,
                        "country": pair.country,
                        "reason": "failed_readiness_gates",
                    }
                )
                continue

            for seed in seeds:
                try:
                    diag, bal = estimate_pair_seed(
                        pdata=pdata,
                        seed=int(seed),
                        n_folds=int(args.n_folds),
                        propensity_clip=float(args.propensity_clip),
                        n_bootstrap=int(args.n_bootstrap),
                    )
                except Exception as exc:
                    skip_rows.append(
                        {
                            "rule_id": pair.rule_id,
                            "country": pair.country,
                            "reason": f"seed_{seed}_failed: {type(exc).__name__}: {exc}",
                        }
                    )
                    continue
                seed_rows.append(diag)
                balance_rows.append(bal)
        except Exception as exc:
            readiness_rows.append(
                {
                    "rule_id": pair.rule_id,
                    "country": pair.country,
                    "generalization_status": pair.generalization_status,
                    "analysis_tier": pair.analysis_tier,
                    "n_total": 0,
                    "n_treated": 0,
                    "n_control": 0,
                    "treated_rate": np.nan,
                    "n_trials": 0,
                    "n_trials_treated": 0,
                    "n_trials_control": 0,
                    "estimable_by_gates": False,
                }
            )
            skip_rows.append(
                {
                    "rule_id": pair.rule_id,
                    "country": pair.country,
                    "reason": f"pair_build_failed: {type(exc).__name__}: {exc}",
                }
            )

    readiness_df = pd.DataFrame(readiness_rows).drop_duplicates(subset=["rule_id", "country"], keep="last")
    readiness_df.to_csv(out_dir / "pair_readiness.csv", index=False)

    seed_df = pd.DataFrame(seed_rows)
    if seed_df.empty:
        raise RuntimeError("No estimable pair-seed results were produced.")
    seed_df = seed_df.sort_values(["analysis_tier", "rule_id", "country", "seed"]).reset_index(drop=True)
    seed_df.to_csv(out_dir / "pair_aipw_seed_estimates.csv", index=False)
    seed_df.to_csv(out_dir / "pair_diagnostics.csv", index=False)

    balance_df = pd.concat(balance_rows, ignore_index=True) if balance_rows else pd.DataFrame()
    if not balance_df.empty:
        balance_df.to_csv(out_dir / "pair_balance_smd.csv", index=False)
    else:
        pd.DataFrame(columns=["rule_id", "country", "seed", "feature", "smd_before", "smd_after"]).to_csv(
            out_dir / "pair_balance_smd.csv",
            index=False,
        )

    summary_df = summarize_pair_estimates(seed_df)
    summary_df.to_csv(out_dir / "pair_aipw_summary.csv", index=False)
    scorecard_df = build_scorecard(summary_df, readiness_df)
    scorecard_df.to_csv(out_dir / "causal_rule_scorecard.csv", index=False)

    write_playbook_overlay(
        out_path=out_dir / "causal_rule_playbook_overlay.md",
        scorecard_df=scorecard_df,
    )

    if skip_rows:
        pd.DataFrame(skip_rows).to_csv(out_dir / "pair_skips.csv", index=False)
    else:
        pd.DataFrame(columns=["rule_id", "country", "reason"]).to_csv(out_dir / "pair_skips.csv", index=False)

    runlog = [
        f"run_tag={args.run_tag}",
        f"interp_dir={interp_dir}",
        f"seeds={seeds}",
        f"primary_statuses={primary_statuses}",
        f"secondary_statuses={secondary_statuses}",
        f"pairs_cataloged={len(pair_catalog_df)}",
        f"pairs_ready={int(readiness_df['estimable_by_gates'].sum()) if not readiness_df.empty else 0}",
        f"pair_seed_results={len(seed_df)}",
        f"scorecard_rows={len(scorecard_df)}",
    ]
    (out_dir / "causal_runlog.txt").write_text("\n".join(runlog), encoding="utf-8")

    print(f"Saved causal AIPW outputs to: {out_dir}")
    print(f"Pairs cataloged: {len(pair_catalog_df)} | ready: {int(readiness_df['estimable_by_gates'].sum())}")
    print(f"Pair-seed estimates: {len(seed_df)} | scorecard rows: {len(scorecard_df)}")

    return {
        "out_dir": out_dir,
        "pair_catalog_df": pair_catalog_df,
        "readiness_df": readiness_df,
        "scorecard_df": scorecard_df,
    }


def main() -> None:
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
