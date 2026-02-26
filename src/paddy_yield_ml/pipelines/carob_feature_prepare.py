"""CAROB in-depth EDA + hybrid-selection preparation for decision support."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from paddy_yield_ml.pipelines import carob_common as cc

try:
    project_root = Path(__file__).resolve().parents[3]
except NameError:
    project_root = Path.cwd()

OUT_DIR = project_root / "outputs" / "carob_feature_prepare"
PIPELINE_VERSION = "carob-fe-v3"
DATA_PROXY_POLICY = "preserve_role"
COUNTRY_CONSTANCY_EXCLUSION_ENABLED = True
COUNTRY_CONSTANCY_RATIO_THRESHOLD = 0.80
COUNTRY_CONSTANCY_MIN_ROWS = 20
TRIAL_FULL_MISSING_EXCLUSION_ENABLED = True
TRIAL_FULL_MISSING_FEATURES = ("soil_P", "soil_pH")


def numeric_variance_audit(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    y = pd.to_numeric(df[cc.TARGET_COL], errors="coerce")
    num_cols = df.select_dtypes(include=[np.number, "boolean"]).columns.tolist()

    for col in num_cols:
        s = pd.to_numeric(df[col], errors="coerce")
        non_null = s.dropna()
        nunique = int(non_null.nunique(dropna=True)) if len(non_null) else 0
        dominant = float(non_null.value_counts(normalize=True).iloc[0]) if len(non_null) else np.nan
        corr = 1.0 if col == cc.TARGET_COL else cc.safe_corr(s, y, method="spearman")
        rows.append(
            {
                "feature": col,
                "missing_pct": float(s.isna().mean() * 100),
                "n_unique": nunique,
                "dominant_value_ratio": dominant,
                "std": float(non_null.std(ddof=0)) if len(non_null) else np.nan,
                "variance": float(non_null.var(ddof=0)) if len(non_null) else np.nan,
                "min": float(non_null.min()) if len(non_null) else np.nan,
                "max": float(non_null.max()) if len(non_null) else np.nan,
                "abs_spearman_to_target": abs(corr) if pd.notna(corr) else np.nan,
                "constant_flag": bool(nunique <= 1),
                "near_constant_flag": bool(nunique <= 1 or (pd.notna(dominant) and dominant >= 0.995)),
            }
        )

    out = pd.DataFrame(rows).sort_values(
        ["constant_flag", "near_constant_flag", "feature"], ascending=[False, False, True]
    )
    out.to_csv(OUT_DIR / "numeric_variance_audit.csv", index=False)
    return out


def categorical_quality_overview(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    cat_cols = df.select_dtypes(exclude=[np.number, "boolean"]).columns.tolist()
    y = pd.to_numeric(df[cc.TARGET_COL], errors="coerce")

    for col in cat_cols:
        s = df[col].astype("string")
        card = int(s.nunique(dropna=True))
        top_ratio = float(s.fillna("<MISSING>").value_counts(normalize=True).iloc[0]) if len(s) else np.nan
        eta = cc.eta_squared(y, s)
        rows.append(
            {
                "feature": col,
                "missing_pct": float(s.isna().mean() * 100),
                "cardinality": card,
                "top_level_ratio": top_ratio,
                "eta_squared_to_target": eta,
                "constant_flag": bool(card <= 1),
                "near_constant_flag": bool(card <= 1 or (pd.notna(top_ratio) and top_ratio >= 0.995)),
            }
        )

    out = pd.DataFrame(rows).sort_values(
        ["constant_flag", "near_constant_flag", "feature"], ascending=[False, False, True]
    )
    out.to_csv(OUT_DIR / "categorical_quality_overview.csv", index=False)
    return out


def within_group_numeric_signal(df: pd.DataFrame) -> pd.DataFrame:
    out_rows: list[dict[str, object]] = []

    num_cols = [c for c in df.select_dtypes(include=[np.number, "boolean"]).columns if c != cc.TARGET_COL]
    for col in num_cols:
        weighted_corrs: list[float] = []
        weights: list[int] = []
        for _, g in df.groupby(cc.GROUP_COL, dropna=False):
            if len(g) < 5:
                continue
            corr = cc.safe_corr(
                pd.to_numeric(g[col], errors="coerce"),
                pd.to_numeric(g[cc.TARGET_COL], errors="coerce"),
                method="spearman",
            )
            if pd.notna(corr):
                weighted_corrs.append(float(corr))
                weights.append(int(len(g)))

        if weights and np.sum(weights) > 0:
            weighted = float(np.average(weighted_corrs, weights=weights))
        else:
            weighted = float("nan")

        out_rows.append(
            {
                "feature": col,
                "abs_weighted_spearman": abs(weighted) if pd.notna(weighted) else np.nan,
                "n_groups_used": int(len(weights)),
            }
        )

    out = pd.DataFrame(out_rows).sort_values("abs_weighted_spearman", ascending=False, na_position="last")
    out.to_csv(OUT_DIR / "within_trial_numeric_signal.csv", index=False)
    return out


def categorical_target_effects(df: pd.DataFrame) -> pd.DataFrame:
    y = pd.to_numeric(df[cc.TARGET_COL], errors="coerce")
    rows: list[dict[str, object]] = []

    cat_cols = [c for c in df.select_dtypes(exclude=[np.number, "boolean"]).columns if c != cc.GROUP_COL]
    for col in cat_cols:
        eta_global = cc.eta_squared(y, df[col].astype("string"))
        eta_group_vals: list[float] = []
        group_weights: list[int] = []
        for _, g in df.groupby(cc.GROUP_COL, dropna=False):
            if len(g) < 8:
                continue
            eta_g = cc.eta_squared(pd.to_numeric(g[cc.TARGET_COL], errors="coerce"), g[col].astype("string"))
            if pd.notna(eta_g):
                eta_group_vals.append(float(eta_g))
                group_weights.append(int(len(g)))

        eta_within = (
            float(np.average(eta_group_vals, weights=group_weights))
            if group_weights and np.sum(group_weights) > 0
            else np.nan
        )
        rows.append(
            {
                "feature": col,
                "global_eta_squared": eta_global,
                "within_group_eta_squared_weighted": eta_within,
            }
        )

    out = pd.DataFrame(rows).sort_values("global_eta_squared", ascending=False, na_position="last")
    out.to_csv(OUT_DIR / "categorical_target_effects.csv", index=False)
    return out


def proxy_leakage_audit(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    groups = df[cc.GROUP_COL].astype(str)

    for col in df.columns:
        if col in {cc.TARGET_COL, cc.GROUP_COL}:
            continue

        if pd.api.types.is_numeric_dtype(df[col]) or str(df[col].dtype) == "boolean":
            s = pd.to_numeric(df[col], errors="coerce")
            if s.notna().sum() < 10:
                continue

            overall_std = float(s.std(ddof=0)) if s.notna().sum() > 1 else np.nan
            within_std = float(s.groupby(groups).std(ddof=0).mean()) if groups.nunique() > 1 else np.nan
            within_ratio = (
                (within_std / overall_std)
                if pd.notna(overall_std) and overall_std > 1e-12 and pd.notna(within_std)
                else np.nan
            )

            valid = s.notna() & groups.notna()
            if int(valid.sum()) >= 10:
                mean_total = float(s[valid].mean())
                ss_total = float(((s[valid] - mean_total) ** 2).sum())
                ss_between = float(
                    sum(len(g) * (g.mean() - mean_total) ** 2 for _, g in s[valid].groupby(groups[valid]))
                )
                eta2 = (ss_between / ss_total) if ss_total > 1e-12 else np.nan
            else:
                eta2 = np.nan

            proxy_flag = bool(pd.notna(eta2) and eta2 >= 0.85 and pd.notna(within_ratio) and within_ratio <= 0.25)

            rows.append(
                {
                    "feature": col,
                    "feature_type": "numeric",
                    "between_group_eta2": eta2,
                    "within_group_std_ratio": within_ratio,
                    "proxy_flag": proxy_flag,
                }
            )
        else:
            s = df[col].astype("string")
            purity = (
                float(s.groupby(groups).apply(lambda g: g.value_counts(normalize=True).iloc[0]).mean())
                if len(s)
                else np.nan
            )
            proxy_flag = bool(pd.notna(purity) and purity >= 0.95)
            rows.append(
                {
                    "feature": col,
                    "feature_type": "categorical",
                    "between_group_eta2": np.nan,
                    "within_group_std_ratio": np.nan,
                    "group_purity_mean": purity,
                    "proxy_flag": proxy_flag,
                }
            )

    out = pd.DataFrame(rows).sort_values(["proxy_flag", "feature"], ascending=[False, True])
    out.to_csv(OUT_DIR / "proxy_leakage_audit.csv", index=False)
    return out


def high_correlation_pairs(df: pd.DataFrame, threshold: float = 0.97) -> pd.DataFrame:
    num_cols = [c for c in df.select_dtypes(include=[np.number, "boolean"]).columns if c != cc.TARGET_COL]
    if len(num_cols) < 2:
        out = pd.DataFrame(columns=["feature_1", "feature_2", "corr", "high_corr_flag", "drop_candidate"])
        out.to_csv(OUT_DIR / "high_correlation_pairs.csv", index=False)
        return out

    corr = df[num_cols].corr(numeric_only=True)
    target = pd.to_numeric(df[cc.TARGET_COL], errors="coerce")
    rows: list[dict[str, object]] = []

    for i in range(len(num_cols)):
        for j in range(i + 1, len(num_cols)):
            f1 = num_cols[i]
            f2 = num_cols[j]
            cv = float(corr.loc[f1, f2]) if pd.notna(corr.loc[f1, f2]) else np.nan
            high = bool(pd.notna(cv) and abs(cv) >= threshold)
            drop = ""
            if high:
                a1 = abs(cc.safe_corr(pd.to_numeric(df[f1], errors="coerce"), target, method="spearman"))
                a2 = abs(cc.safe_corr(pd.to_numeric(df[f2], errors="coerce"), target, method="spearman"))
                if pd.notna(a1) and pd.notna(a2):
                    drop = f1 if a1 < a2 else f2
            rows.append(
                {
                    "feature_1": f1,
                    "feature_2": f2,
                    "corr": cv,
                    "high_corr_flag": high,
                    "drop_candidate": drop,
                }
            )

    out = pd.DataFrame(rows).sort_values(["high_corr_flag", "corr"], ascending=[False, False], na_position="last")
    out.to_csv(OUT_DIR / "high_correlation_pairs.csv", index=False)
    return out


def apply_group_constancy_gate(
    df: pd.DataFrame,
    *,
    group_col: str = "country",
    enabled: bool = COUNTRY_CONSTANCY_EXCLUSION_ENABLED,
    ratio_threshold: float = COUNTRY_CONSTANCY_RATIO_THRESHOLD,
    min_rows: int = COUNTRY_CONSTANCY_MIN_ROWS,
    audit_filename: str = "country_exclusion_audit.csv",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if group_col not in df.columns:
        raise ValueError(f"Constancy gate group column not found: {group_col}")

    reserved = {cc.TARGET_COL, cc.GROUP_COL} | cc.IDENTIFIER_COLS | cc.POST_OUTCOME_COLS
    candidate_features = [c for c in df.columns if c not in reserved]

    # Exclude globally non-varying features so groups are not penalized for dataset-wide constants.
    eval_features: list[str] = []
    global_non_varying: list[str] = []
    for feat in candidate_features:
        non_missing = df[feat].dropna()
        nunique = int(non_missing.nunique(dropna=True)) if len(non_missing) else 0
        if nunique <= 1:
            global_non_varying.append(feat)
        else:
            eval_features.append(feat)

    rows: list[dict[str, object]] = []
    for grp, gdf in df.groupby(group_col, dropna=False):
        group_id = str(grp)
        n_rows = int(len(gdf))
        constant_feats: list[str] = []
        empty_feats: list[str] = []
        for feat in eval_features:
            non_missing = gdf[feat].dropna()
            if len(non_missing) == 0:
                empty_feats.append(feat)
                constant_feats.append(feat)
                continue
            if int(non_missing.nunique(dropna=True)) <= 1:
                constant_feats.append(feat)

        denom = int(len(eval_features))
        const_count = int(len(constant_feats))
        empty_count = int(len(empty_feats))
        ratio = float(const_count / denom) if denom > 0 else np.nan
        drop_candidate = bool(
            denom > 0 and n_rows >= int(min_rows) and pd.notna(ratio) and float(ratio) >= float(ratio_threshold)
        )
        drop_group = bool(enabled and drop_candidate)
        reason = (
            "rule_disabled"
            if not enabled
            else "below_min_rows"
            if n_rows < int(min_rows)
            else "ratio_ge_threshold"
            if drop_group
            else "kept"
        )

        rows.append(
            {
                "group_column": group_col,
                "group_value": group_id,
                "n_rows": n_rows,
                "n_features_evaluated": denom,
                "non_varying_feature_count": const_count,
                "empty_feature_count": empty_count,
                "non_varying_feature_ratio": ratio,
                "drop_candidate_by_ratio": drop_candidate,
                "drop_group": drop_group,
                "decision_reason": reason,
                "ratio_threshold": float(ratio_threshold),
                "min_rows_threshold": int(min_rows),
                "global_non_varying_feature_count": int(len(global_non_varying)),
                "global_non_varying_features": ";".join(global_non_varying),
                "non_varying_features_in_group": ";".join(constant_feats),
            }
        )

    audit = pd.DataFrame(rows).sort_values(
        ["drop_group", "non_varying_feature_ratio", "n_rows"],
        ascending=[False, False, False],
        na_position="last",
    )
    audit.to_csv(OUT_DIR / audit_filename, index=False)

    if not enabled:
        return df.copy(), audit

    dropped = set(audit.loc[audit["drop_group"], "group_value"].astype(str))
    if not dropped:
        return df.copy(), audit

    filtered = df[~df[group_col].astype(str).isin(dropped)].reset_index(drop=True)
    if filtered.empty:
        raise ValueError("Group constancy gate dropped all rows. Lower threshold or disable the rule.")
    return filtered, audit


def apply_trial_full_missing_gate(
    df: pd.DataFrame,
    *,
    group_col: str = cc.GROUP_COL,
    features: tuple[str, ...] = TRIAL_FULL_MISSING_FEATURES,
    enabled: bool = TRIAL_FULL_MISSING_EXCLUSION_ENABLED,
    audit_filename: str = "trial_exclusion_audit.csv",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if group_col not in df.columns:
        raise ValueError(f"Trial full-missing gate group column not found: {group_col}")

    work = df.copy()
    work[group_col] = work[group_col].astype(str)
    rows: list[dict[str, object]] = []

    for group_value, gdf in work.groupby(group_col, dropna=False):
        full_missing_features: list[str] = []
        row: dict[str, object] = {
            "group_column": group_col,
            "group_value": str(group_value),
            "n_rows": int(len(gdf)),
        }

        for feat in features:
            miss_col = f"{feat}_missing_pct"
            full_col = f"{feat}_all_missing"
            if feat not in work.columns:
                row[miss_col] = float("nan")
                row[full_col] = False
                continue

            missing_pct = float(gdf[feat].isna().mean())
            is_full = bool(missing_pct >= 1.0)
            row[miss_col] = missing_pct
            row[full_col] = is_full
            if is_full:
                full_missing_features.append(feat)

        drop_group = bool(enabled and len(full_missing_features) > 0)
        row["drop_group"] = drop_group
        row["decision_reason"] = "full_missing:" + ";".join(full_missing_features) if drop_group else "kept"
        rows.append(row)

    audit = pd.DataFrame(rows).sort_values(["drop_group", "group_value"], ascending=[False, True], na_position="last")
    audit.to_csv(OUT_DIR / audit_filename, index=False)

    if not enabled:
        return work, audit

    dropped = set(audit.loc[audit["drop_group"].astype(bool), "group_value"].astype(str))
    if not dropped:
        return work.reset_index(drop=True), audit

    filtered = work[~work[group_col].astype(str).isin(dropped)].reset_index(drop=True)
    if filtered.empty:
        raise ValueError("Trial full-missing gate dropped all rows. Disable the rule or review input data.")
    return filtered, audit


def build_actionability_split(role_map: pd.DataFrame, proxy_df: pd.DataFrame) -> pd.DataFrame:
    out = role_map.copy()
    proxy_map = proxy_df.set_index("feature")["proxy_flag"].to_dict() if not proxy_df.empty else {}

    out["proxy_flag_from_data"] = out["column_name"].map(lambda c: bool(proxy_map.get(c, False)))
    out["final_role"] = out["final_role"].astype(str).str.strip().str.lower()
    if DATA_PROXY_POLICY == "strict":
        out.loc[out["proxy_flag_from_data"], "final_role"] = "proxy"
    elif DATA_PROXY_POLICY != "preserve_role":
        raise ValueError(f"Unsupported DATA_PROXY_POLICY: {DATA_PROXY_POLICY}")
    out.loc[out["leakage_risk"].astype(str).str.contains("high", case=False, na=False), "final_role"] = "proxy"
    out.loc[out["modeling_recommendation"].astype(str).str.startswith("exclude"), "final_role"] = "proxy"

    out.to_csv(OUT_DIR / "actionability_role_map.csv", index=False)
    summary = (
        out.groupby(["feature_group", "final_role"], dropna=False)
        .size()
        .reset_index(name="n_features")
        .sort_values(["feature_group", "final_role"])
    )
    summary.to_csv(OUT_DIR / "actionability_role_summary.csv", index=False)
    return out


def build_hybrid_candidates(
    df: pd.DataFrame,
    role_map: pd.DataFrame,
    num_audit: pd.DataFrame,
    cat_audit: pd.DataFrame,
    within_num: pd.DataFrame,
    cat_effects: pd.DataFrame,
    corr_pairs: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    num_map = num_audit.set_index("feature").to_dict(orient="index") if not num_audit.empty else {}
    cat_map = cat_audit.set_index("feature").to_dict(orient="index") if not cat_audit.empty else {}
    within_map = within_num.set_index("feature").to_dict(orient="index") if not within_num.empty else {}
    cat_eff_map = cat_effects.set_index("feature").to_dict(orient="index") if not cat_effects.empty else {}

    high_pairs = corr_pairs[corr_pairs["high_corr_flag"]] if not corr_pairs.empty else pd.DataFrame()
    corr_count: dict[str, int] = {}
    corr_drop_count: dict[str, int] = {}
    if not high_pairs.empty:
        for _, r in high_pairs.iterrows():
            f1 = str(r["feature_1"])
            f2 = str(r["feature_2"])
            dc = str(r.get("drop_candidate", "")).strip()
            corr_count[f1] = corr_count.get(f1, 0) + 1
            corr_count[f2] = corr_count.get(f2, 0) + 1
            if dc:
                corr_drop_count[dc] = corr_drop_count.get(dc, 0) + 1

    rm = role_map.set_index("column_name")
    rows: list[dict[str, object]] = []

    for feature in df.columns:
        if feature in {cc.TARGET_COL, cc.GROUP_COL} | cc.IDENTIFIER_COLS:
            continue

        role_info = rm.loc[feature] if feature in rm.index else None
        final_role = str(role_info["final_role"]).lower() if role_info is not None else "context"
        modeling_rec = str(role_info["modeling_recommendation"]).strip().lower() if role_info is not None else "review"
        feature_group = str(role_info["feature_group"]) if role_info is not None else "Unmapped"

        is_num = bool(pd.api.types.is_numeric_dtype(df[feature]) or str(df[feature].dtype) == "boolean")
        if is_num:
            info = num_map.get(feature, {})
            global_assoc = info.get("abs_spearman_to_target", np.nan)
            within_assoc = within_map.get(feature, {}).get("abs_weighted_spearman", np.nan)
            constant_flag = bool(info.get("constant_flag", False))
            near_constant_flag = bool(info.get("near_constant_flag", False))
        else:
            info = cat_map.get(feature, {})
            global_assoc = cat_eff_map.get(feature, {}).get("global_eta_squared", np.nan)
            within_assoc = cat_eff_map.get(feature, {}).get("within_group_eta_squared_weighted", np.nan)
            constant_flag = bool(info.get("constant_flag", False))
            near_constant_flag = bool(info.get("near_constant_flag", False))

        hi_drop = int(corr_drop_count.get(feature, 0))
        hi_count = int(corr_count.get(feature, 0))

        reasons: list[str] = []
        if final_role == "proxy":
            reasons.append("proxy_or_leakage")
        if modeling_rec.startswith("exclude"):
            reasons.append("dictionary_exclude")
        if constant_flag:
            reasons.append("constant")
        elif near_constant_flag:
            reasons.append("near_constant")
        if hi_drop > 0:
            reasons.append(f"redundant_drop_candidate={hi_drop}")

        if final_role == "proxy" or modeling_rec.startswith("exclude") or constant_flag:
            status = "excluded"
        elif near_constant_flag and (pd.isna(global_assoc) or float(global_assoc) < 0.02):
            status = "excluded"
        elif final_role == "modifiable":
            status = "candidate_modifiable" if hi_drop == 0 else "candidate_redundant_review"
        elif final_role == "context":
            status = "reserve_context"
        else:
            status = "excluded"

        assoc_term = 0.0 if pd.isna(global_assoc) else float(global_assoc)
        within_term = 0.0 if pd.isna(within_assoc) else float(within_assoc)
        role_bonus = 1.0 if final_role == "modifiable" else 0.25 if final_role == "context" else -0.5
        score = role_bonus + (0.8 * assoc_term) + (0.5 * within_term) - (0.1 * hi_drop)

        rows.append(
            {
                "feature": feature,
                "feature_group": feature_group,
                "final_role": final_role,
                "status": status,
                "global_association": global_assoc,
                "within_group_association": within_assoc,
                "high_corr_pair_count": hi_count,
                "high_corr_drop_candidate_count": hi_drop,
                "hybrid_priority_score": score,
                "reasons": ";".join(reasons),
            }
        )

    out = pd.DataFrame(rows)
    status_rank = {
        "candidate_modifiable": 0,
        "candidate_redundant_review": 1,
        "reserve_context": 2,
        "excluded": 3,
    }
    out["status_rank"] = out["status"].map(status_rank).fillna(99)
    out = out.sort_values(
        ["status_rank", "hybrid_priority_score"],
        ascending=[True, False],
    ).drop(columns=["status_rank"])

    screening = out.copy()
    screening["recommendation"] = screening["status"].map(
        {
            "candidate_modifiable": "keep",
            "candidate_redundant_review": "review",
            "reserve_context": "keep",
            "excluded": "drop",
        }
    )
    screening.to_csv(OUT_DIR / "feature_screening_summary.csv", index=False)

    out.to_csv(OUT_DIR / "hybrid_selection_candidates.csv", index=False)
    return screening, out


def sectioned_profiles(df: pd.DataFrame) -> None:
    trial_profile = (
        df.groupby(cc.GROUP_COL)
        .agg(
            n_rows=(cc.TARGET_COL, "size"),
            yield_mean=(cc.TARGET_COL, "mean"),
            yield_std=(cc.TARGET_COL, "std"),
            plus_share=(cc.TREATMENT_COL, lambda s: float((s == cc.TREATMENT_POS).mean())),
            n_varieties=("variety", "nunique") if "variety" in df.columns else (cc.TARGET_COL, "size"),
        )
        .reset_index()
        .sort_values(cc.GROUP_COL)
    )
    trial_profile.to_csv(OUT_DIR / "section_trial_profile.csv", index=False)

    if "country" in df.columns:
        country_profile = (
            df.groupby("country")
            .agg(
                n_rows=(cc.TARGET_COL, "size"),
                n_trials=(cc.GROUP_COL, "nunique"),
                yield_mean=(cc.TARGET_COL, "mean"),
                plus_share=(cc.TREATMENT_COL, lambda s: float((s == cc.TREATMENT_POS).mean())),
            )
            .reset_index()
            .sort_values("country")
        )
        country_profile.to_csv(OUT_DIR / "section_country_profile.csv", index=False)

    rows: list[dict[str, object]] = []
    for feat in ["treatment", "P_fertilizer", "N_fertilizer", "K_fertilizer", "variety", "irrigated", "row_spacing"]:
        if feat not in df.columns:
            continue
        nun = df.groupby(cc.GROUP_COL)[feat].nunique(dropna=True)
        nun_values = nun.to_numpy(dtype=float)
        min_unique = int(np.min(nun_values)) if len(nun_values) > 0 else 0
        median_unique = float(np.median(nun_values)) if len(nun_values) > 0 else float("nan")
        max_unique = int(np.max(nun_values)) if len(nun_values) > 0 else 0
        rows.append(
            {
                "feature": feat,
                "min_unique_within_trial": min_unique,
                "median_unique_within_trial": median_unique,
                "max_unique_within_trial": max_unique,
                "trials_with_2plus_levels": int((nun >= 2).sum()),
                "n_trials_total": int(len(nun)),
            }
        )
    pd.DataFrame(rows).sort_values("feature").to_csv(OUT_DIR / "section_within_trial_variation.csv", index=False)


def plot_artifacts(df: pd.DataFrame, missing_df: pd.DataFrame, within_df: pd.DataFrame) -> None:
    plt.figure(figsize=(8, 5))
    pd.to_numeric(df[cc.TARGET_COL], errors="coerce").hist(bins=35)
    plt.title("CAROB target distribution (yield)")
    plt.xlabel(cc.TARGET_COL)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "target_distribution.png", dpi=150)
    plt.close()

    if not missing_df.empty:
        top = missing_df.sort_values("missing_pct", ascending=False).head(20)
        plt.figure(figsize=(10, 6))
        plt.barh(top["feature"], top["missing_pct"])
        plt.gca().invert_yaxis()
        plt.title("Top missingness features")
        plt.xlabel("Missing %")
        plt.tight_layout()
        plt.savefig(OUT_DIR / "missingness_top20.png", dpi=150)
        plt.close()

    if not within_df.empty:
        top = within_df.sort_values("abs_weighted_spearman", ascending=False).head(12)
        plt.figure(figsize=(10, 6))
        plt.barh(top["feature"], top["abs_weighted_spearman"])
        plt.gca().invert_yaxis()
        plt.title("Top within-trial numeric signal")
        plt.xlabel("|Weighted Spearman|")
        plt.tight_layout()
        plt.savefig(OUT_DIR / "top_within_trial_numeric_signal.png", dpi=150)
        plt.close()


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    raw_df = cc.load_analysis_frame(require_treatment=True)
    feature_df = raw_df.copy()
    role_map_raw = cc.load_role_map(frame_cols=feature_df.columns.tolist())

    print(f"CAROB feature_prepare version: {PIPELINE_VERSION}")
    print("Shape (raw):", raw_df.shape)
    print("Shape (feature-selection base):", feature_df.shape)

    missing_df = pd.DataFrame(
        {
            "feature": feature_df.columns,
            "missing_count": [int(feature_df[c].isna().sum()) for c in feature_df.columns],
            "missing_pct": [float(feature_df[c].isna().mean() * 100) for c in feature_df.columns],
        }
    ).sort_values("missing_pct", ascending=False)
    missing_df.to_csv(OUT_DIR / "missingness_audit.csv", index=False)

    num_audit = numeric_variance_audit(feature_df)
    cat_audit = categorical_quality_overview(feature_df)
    within_num = within_group_numeric_signal(feature_df)
    cat_effects = categorical_target_effects(feature_df)
    proxy_df = proxy_leakage_audit(feature_df)
    corr_pairs = high_correlation_pairs(feature_df)

    role_map = build_actionability_split(role_map_raw, proxy_df)
    screening_df, hybrid_df = build_hybrid_candidates(
        df=feature_df,
        role_map=role_map,
        num_audit=num_audit,
        cat_audit=cat_audit,
        within_num=within_num,
        cat_effects=cat_effects,
        corr_pairs=corr_pairs,
    )

    # Apply row exclusions after feature selection artifacts are computed.
    df_country, country_gate_audit = apply_group_constancy_gate(raw_df, group_col="country")
    df_model, trial_gate_audit = apply_trial_full_missing_gate(df_country, group_col=cc.GROUP_COL)
    sectioned_profiles(df_model)
    plot_artifacts(feature_df, missing_df, within_num)

    population_summary = pd.DataFrame(
        [
            {
                "rows_raw": int(len(raw_df)),
                "rows_feature_selection_base": int(len(feature_df)),
                "rows_after_country_gate": int(len(df_country)),
                "rows_after_trial_gate": int(len(df_model)),
                "rows_removed_country_gate": int(len(raw_df) - len(df_country)),
                "rows_removed_trial_gate": int(len(df_country) - len(df_model)),
                "countries_raw": int(raw_df["country"].nunique()) if "country" in raw_df.columns else 0,
                "countries_after_country_gate": (
                    int(df_country["country"].nunique()) if "country" in df_country.columns else 0
                ),
                "trials_raw": int(raw_df[cc.GROUP_COL].nunique()) if cc.GROUP_COL in raw_df.columns else 0,
                "trials_after_country_gate": int(df_country[cc.GROUP_COL].nunique())
                if cc.GROUP_COL in df_country.columns
                else 0,
                "trials_after_trial_gate": (
                    int(df_model[cc.GROUP_COL].nunique()) if cc.GROUP_COL in df_model.columns else 0
                ),
                "countries_dropped_by_gate": int(country_gate_audit["drop_group"].sum())
                if not country_gate_audit.empty
                else 0,
                "trials_dropped_by_gate": (
                    int(trial_gate_audit["drop_group"].sum()) if not trial_gate_audit.empty else 0
                ),
            }
        ]
    )
    population_summary.to_csv(OUT_DIR / "modeling_population_summary.csv", index=False)

    print("Shape (after country gate):", df_country.shape)
    print("Shape (after trial full-missing gate):", df_model.shape)
    print(
        "Countries dropped by constancy gate:",
        int(country_gate_audit["drop_group"].sum()) if not country_gate_audit.empty else 0,
    )
    print(
        "Trials dropped by full-missing gate:",
        int(trial_gate_audit["drop_group"].sum()) if not trial_gate_audit.empty else 0,
    )

    runlog = [
        f"pipeline_version={PIPELINE_VERSION}",
        f"data_proxy_policy={DATA_PROXY_POLICY}",
        f"country_constancy_exclusion_enabled={COUNTRY_CONSTANCY_EXCLUSION_ENABLED}",
        f"country_constancy_ratio_threshold={COUNTRY_CONSTANCY_RATIO_THRESHOLD}",
        f"country_constancy_min_rows={COUNTRY_CONSTANCY_MIN_ROWS}",
        f"trial_full_missing_exclusion_enabled={TRIAL_FULL_MISSING_EXCLUSION_ENABLED}",
        f"trial_full_missing_features={';'.join(TRIAL_FULL_MISSING_FEATURES)}",
        f"countries_raw={int(raw_df['country'].nunique()) if 'country' in raw_df.columns else 0}",
        (
            "countries_feature_selection_base="
            f"{int(feature_df['country'].nunique()) if 'country' in feature_df.columns else 0}"
        ),
        f"countries_after_gate={int(df_country['country'].nunique()) if 'country' in df_country.columns else 0}",
        (
            "countries_dropped_by_gate="
            f"{int(country_gate_audit['drop_group'].sum()) if not country_gate_audit.empty else 0}"
        ),
        f"groups_raw={int(raw_df[cc.GROUP_COL].nunique()) if cc.GROUP_COL in raw_df.columns else 0}",
        (
            "groups_feature_selection_base="
            f"{int(feature_df[cc.GROUP_COL].nunique()) if cc.GROUP_COL in feature_df.columns else 0}"
        ),
        (
            "groups_after_country_gate="
            f"{int(df_country[cc.GROUP_COL].nunique()) if cc.GROUP_COL in df_country.columns else 0}"
        ),
        (
            "groups_after_trial_gate="
            f"{int(df_model[cc.GROUP_COL].nunique()) if cc.GROUP_COL in df_model.columns else 0}"
        ),
        (
            "trials_dropped_by_gate="
            f"{int(trial_gate_audit['drop_group'].sum()) if not trial_gate_audit.empty else 0}"
        ),
        f"rows_feature_selection_base={len(feature_df)}",
        f"rows_modeling_after_gates={len(df_model)}",
        f"cols={len(feature_df.columns)}",
        f"candidate_modifiable={int((hybrid_df['status'] == 'candidate_modifiable').sum())}",
        f"candidate_redundant_review={int((hybrid_df['status'] == 'candidate_redundant_review').sum())}",
        f"reserve_context={int((hybrid_df['status'] == 'reserve_context').sum())}",
        f"excluded={int((hybrid_df['status'] == 'excluded').sum())}",
        f"drop_recommendations={int((screening_df['recommendation'] == 'drop').sum())}",
    ]
    (OUT_DIR / "feature_prepare_runlog.txt").write_text("\n".join(runlog), encoding="utf-8")

    print("\nHybrid candidate status counts:")
    print(hybrid_df["status"].value_counts().to_string())
    print(f"\nSaved CAROB feature-prepare outputs to: {OUT_DIR}")


if __name__ == "__main__":
    main()
