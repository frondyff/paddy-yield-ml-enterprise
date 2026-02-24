"""
Dictionary-driven in-depth EDA + hybrid-selection preparation for paddydataset.csv.

Run:
  python src/paddy_yield_ml/pipelines/feature_prepare.py

Outputs (under ./outputs/feature_prepare):
  - all core v2 quality/redundancy/proxy audits
  - sectioned EDA using data_dictionary_paddy.csv feature groups
  - domain validation checks (ranges, windows, impossible combinations)
  - actionability split (modifiable/context/proxy)
  - stage-based diagnostics across crop windows
  - hybrid-selection-ready candidate list
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    # Resolve project root from src/paddy_yield_ml/pipelines/feature_prepare.py
    project_root = Path(__file__).resolve().parents[3]
except NameError:
    project_root = Path.cwd()

DATA_PATH = project_root / "data" / "input" / "paddydataset.csv"
DICT_PATH = project_root / "data" / "metadata" / "data_dictionary_paddy.csv"
OUT_DIR = project_root / "outputs" / "feature_prepare"
PIPELINE_VERSION = "fe-pipeline-minor-1"

RAW_TARGET_COL = "Paddy yield(in Kg)"
TARGET_COL = "Paddy yield_per_hectare(in Kg)"
GROUP_COL = "Agriblock"
SIZE_COL = "Hectares"

SIZE_SCALED_COLS = [
    "LP_nurseryarea(in Tonnes)",
    "Micronutrients_70Days",
    "Weed28D_thiobencarb",
    "Urea_40Days",
    "DAP_20days",
    "Nursery area (Cents)",
    "Pest_60Day(in ml)",
    "LP_Mainfield(in Tonnes)",
    "Seedrate(in Kg)",
    "Potassh_50Days",
]

EXPECTED_DICT_COLUMNS = [
    "column_name",
    "feature_group",
    "data_type",
    "unit_or_levels",
    "meaning",
    "confidence",
    "evidence_basis",
    "actionability",
    "leakage_risk",
    "modeling_recommendation",
]


def clean_columns(cols: Iterable[str]) -> list[str]:
    return [" ".join(str(c).strip().split()) for c in cols]


def load_data_dictionary(path: Path, df_columns: list[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing data dictionary: {path}")

    dd = pd.read_csv(path)
    missing_dict_cols = [c for c in EXPECTED_DICT_COLUMNS if c not in dd.columns]
    if missing_dict_cols:
        raise ValueError(
            f"Data dictionary missing required columns: {missing_dict_cols}"
        )

    dd = dd.copy()
    dd["column_name"] = dd["column_name"].astype(str).map(lambda x: " ".join(x.strip().split()))
    dd = dd.drop_duplicates(subset=["column_name"], keep="first")

    data_cols = set(df_columns)
    dict_cols = set(dd["column_name"].tolist())
    missing_from_dict = sorted(data_cols - dict_cols)
    missing_from_data = sorted(dict_cols - data_cols)
    if missing_from_dict or missing_from_data:
        raise ValueError(
            "Dictionary/data column mismatch. "
            f"Missing from dictionary: {missing_from_dict}; "
            f"Missing from data: {missing_from_data}"
        )

    dd = dd.set_index("column_name").loc[df_columns].reset_index()
    return dd


def infer_base_role(row: pd.Series) -> str:
    leakage_risk = str(row.get("leakage_risk", "")).strip().lower()
    feature_group = str(row.get("feature_group", "")).strip().lower()
    actionability = str(row.get("actionability", "")).strip().lower()

    if "leak" in leakage_risk:
        return "proxy"
    if actionability == "not_modifiable_for_prediction":
        return "proxy"

    modifiable_groups = {
        "input practice",
        "fertilizer schedule",
        "crop protection schedule",
        "crop choice",
        "field management",
    }
    if feature_group in modifiable_groups:
        return "modifiable"
    return "context"


def parse_window_from_feature(name: str) -> tuple[int, int] | tuple[None, None]:
    # Patterns like D1_D30, D31_D60
    m = re.search(r"D(\d+)_D(\d+)", name)
    if m:
        return int(m.group(1)), int(m.group(2))

    # Patterns like 30_50 (used in rain/irrigation columns)
    m = re.search(r"(\d+)_(\d+)", name)
    if m:
        return int(m.group(1)), int(m.group(2))

    # Patterns like 30D..., interpreted as day 1-30 bucket in this schema.
    m = re.match(r"(\d+)D", name)
    if m:
        end = int(m.group(1))
        return 1, end

    return None, None


def stage_label(start_day: int | None, end_day: int | None) -> str:
    if start_day is None or end_day is None:
        return "UNSTAGED"
    return f"D{start_day}_D{end_day}"


def add_stage_and_role_metadata(df: pd.DataFrame, dd: pd.DataFrame) -> pd.DataFrame:
    out = dd.copy()
    starts = []
    ends = []
    labels = []
    for col in out["column_name"]:
        s, e = parse_window_from_feature(col)
        starts.append(s)
        ends.append(e)
        labels.append(stage_label(s, e))
    out["stage_start_day"] = starts
    out["stage_end_day"] = ends
    out["stage_label"] = labels
    out["base_role"] = out.apply(infer_base_role, axis=1)
    out["is_numeric"] = out["column_name"].map(
        lambda c: bool(pd.api.types.is_numeric_dtype(df[c]))
    )
    return out


def safe_corr(x: pd.Series, y: pd.Series, method: str = "pearson") -> float:
    valid = x.notna() & y.notna()
    if valid.sum() < 3:
        return np.nan
    xv = pd.to_numeric(x[valid], errors="coerce")
    yv = pd.to_numeric(y[valid], errors="coerce")
    valid_xy = xv.notna() & yv.notna()
    if valid_xy.sum() < 3:
        return np.nan
    xv = xv[valid_xy]
    yv = yv[valid_xy]
    if xv.nunique(dropna=True) < 2 or yv.nunique(dropna=True) < 2:
        return np.nan
    return float(xv.corr(yv, method=method))


def eta_squared(y: pd.Series, categories: pd.Series) -> float:
    valid = y.notna() & categories.notna()
    if valid.sum() < 3:
        return np.nan
    yv = pd.to_numeric(y[valid], errors="coerce")
    cv = categories[valid].astype(str)
    valid2 = yv.notna() & cv.notna()
    yv = yv[valid2]
    cv = cv[valid2]
    if len(yv) < 3 or cv.nunique(dropna=True) < 2:
        return np.nan

    mean_total = float(yv.mean())
    ss_total = float(((yv - mean_total) ** 2).sum())
    if ss_total == 0:
        return np.nan

    grouped = yv.groupby(cv)
    ss_between = 0.0
    for _, grp in grouped:
        ss_between += len(grp) * float((grp.mean() - mean_total) ** 2)
    return ss_between / ss_total


def numeric_group_proxy_scores(series: pd.Series, groups: pd.Series) -> tuple[float, float]:
    valid = series.notna() & groups.notna()
    if valid.sum() < 3:
        return np.nan, np.nan
    s = pd.to_numeric(series[valid], errors="coerce")
    g = groups[valid]
    valid2 = s.notna()
    s = s[valid2]
    g = g[valid2]
    if len(s) < 3 or s.nunique(dropna=True) < 2:
        return np.nan, np.nan

    total_std = float(s.std(ddof=0))
    if total_std == 0:
        return np.nan, np.nan

    weighted_within_var = 0.0
    for _, grp in s.groupby(g):
        if len(grp) > 1:
            weighted_within_var += (len(grp) / len(s)) * float(grp.var(ddof=0))
    within_std_ratio = np.sqrt(weighted_within_var) / total_std

    overall_mean = float(s.mean())
    grouped = s.groupby(g)
    counts = grouped.size().astype(float)
    means = grouped.mean()
    ss_between = float((((means - overall_mean) ** 2) * counts).sum())
    ss_total = float(((s - overall_mean) ** 2).sum())
    between_eta2 = ss_between / ss_total if ss_total > 0 else np.nan

    return within_std_ratio, between_eta2


def categorical_group_purity(series: pd.Series, groups: pd.Series) -> tuple[float, float]:
    valid = series.notna() & groups.notna()
    if valid.sum() == 0:
        return np.nan, np.nan
    s = series[valid].astype(str)
    g = groups[valid].astype(str)
    if s.nunique(dropna=True) < 2:
        return np.nan, np.nan

    ct = pd.crosstab(s, g)
    if ct.empty:
        return np.nan, np.nan

    per_level_total = ct.sum(axis=1).astype(float)
    per_level_purity = ct.max(axis=1) / per_level_total
    weighted_purity = float((per_level_purity * (per_level_total / per_level_total.sum())).sum())
    perfect_level_ratio = float((per_level_purity == 1.0).mean())
    return weighted_purity, perfect_level_ratio


def save_missingness_audit(df: pd.DataFrame) -> pd.DataFrame:
    miss = df.isna().sum().rename("missing_count").to_frame().reset_index()
    miss = miss.rename(columns={"index": "column"})
    miss["missing_pct"] = (miss["missing_count"] / len(df)) * 100
    miss = miss.sort_values(["missing_count", "column"], ascending=[False, True]).reset_index(drop=True)
    miss.to_csv(OUT_DIR / "missingness_audit.csv", index=False)
    return miss


def save_duplicate_audit(df: pd.DataFrame) -> pd.DataFrame:
    exact_dup_count = int(df.duplicated().sum())
    summary = pd.DataFrame(
        [
            {
                "rows_before_dedup": len(df),
                "exact_duplicate_rows": exact_dup_count,
                "exact_duplicate_pct": (exact_dup_count / len(df)) * 100 if len(df) else np.nan,
            }
        ]
    )
    summary.to_csv(OUT_DIR / "duplicate_audit_summary.csv", index=False)

    feature_cols = [c for c in df.columns if c != RAW_TARGET_COL]
    conflict_df = (
        df.groupby(feature_cols, dropna=False)[RAW_TARGET_COL]
        .agg(row_count="size", target_nunique="nunique", target_min="min", target_max="max")
        .reset_index()
    )
    conflict_df = conflict_df[conflict_df["target_nunique"] > 1].sort_values(
        ["target_nunique", "row_count"], ascending=[False, False]
    )
    conflict_df.to_csv(OUT_DIR / "duplicate_target_conflicts.csv", index=False)
    return summary


def normalize_per_hectare(
    df: pd.DataFrame,
    drop_original: bool = True,
    create_input_scaled: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    if SIZE_COL not in df.columns:
        raise ValueError(f"Size column not found: {SIZE_COL}")

    out = df.copy()
    hectares = pd.to_numeric(out[SIZE_COL], errors="coerce")
    out[TARGET_COL] = pd.to_numeric(out[RAW_TARGET_COL], errors="coerce") / hectares

    created_cols: list[str] = []
    if create_input_scaled:
        for col in SIZE_SCALED_COLS:
            if col in out.columns:
                new_col = f"{col}_per_hectare"
                out[new_col] = pd.to_numeric(out[col], errors="coerce") / hectares
                if drop_original:
                    out = out.drop(columns=[col])
                created_cols.append(new_col)
    return out, created_cols


def numeric_variance_audit(df: pd.DataFrame) -> pd.DataFrame:
    records: list[dict] = []
    if TARGET_COL not in df.columns:
        raise ValueError(f"Missing target column for audit: {TARGET_COL}")

    target = pd.to_numeric(df[TARGET_COL], errors="coerce")
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    for col in num_cols:
        s = pd.to_numeric(df[col], errors="coerce")
        non_null = s.dropna()
        n_non_null = len(non_null)
        nunique = int(non_null.nunique(dropna=True)) if n_non_null else 0
        if n_non_null:
            top_freq = int(non_null.value_counts(dropna=False).iloc[0])
            dominant_ratio = top_freq / n_non_null
            std = float(non_null.std(ddof=0))
            var = float(non_null.var(ddof=0))
            min_v = float(non_null.min())
            max_v = float(non_null.max())
        else:
            dominant_ratio = np.nan
            std = np.nan
            var = np.nan
            min_v = np.nan
            max_v = np.nan

        pearson = safe_corr(s, target, method="pearson") if col != TARGET_COL else 1.0
        spearman = safe_corr(s, target, method="spearman") if col != TARGET_COL else 1.0

        constant_flag = nunique <= 1
        near_constant_flag = constant_flag or (
            pd.notna(dominant_ratio) and dominant_ratio >= 0.995
        ) or (pd.notna(std) and std < 1e-8)

        records.append(
            {
                "feature": col,
                "missing_count": int(s.isna().sum()),
                "missing_pct": float((s.isna().sum() / len(df)) * 100),
                "n_non_null": n_non_null,
                "n_unique": nunique,
                "dominant_value_ratio": dominant_ratio,
                "std": std,
                "variance": var,
                "min": min_v,
                "max": max_v,
                "pearson_to_target": pearson,
                "spearman_to_target": spearman,
                "abs_spearman_to_target": abs(spearman) if pd.notna(spearman) else np.nan,
                "constant_flag": constant_flag,
                "near_constant_flag": near_constant_flag,
            }
        )

    out = pd.DataFrame(records).sort_values(
        ["constant_flag", "near_constant_flag", "abs_spearman_to_target", "feature"],
        ascending=[False, False, False, True],
    )
    out.to_csv(OUT_DIR / "numeric_variance_audit.csv", index=False)
    return out


def categorical_quality_audit(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    overview_records: list[dict] = []
    level_records: list[dict] = []

    n = len(df)
    rare_threshold = max(5, int(np.ceil(0.01 * n))) if n else 5

    for col in cat_cols:
        s = df[col].astype("string")
        value_counts = s.fillna("<MISSING>").value_counts(dropna=False)

        cardinality = int(s.nunique(dropna=True))
        missing_count = int(s.isna().sum())
        missing_pct = (missing_count / n) * 100 if n else np.nan
        top_level = str(value_counts.index[0]) if len(value_counts) else ""
        top_count = int(value_counts.iloc[0]) if len(value_counts) else 0
        top_pct = (top_count / n) * 100 if n else np.nan

        rare_levels = value_counts[value_counts < rare_threshold]
        rare_level_count = int(len(rare_levels))
        rare_row_count = int(rare_levels.sum()) if rare_level_count else 0
        near_constant_flag = top_pct >= 99.5 if pd.notna(top_pct) else False
        constant_flag = cardinality <= 1

        overview_records.append(
            {
                "feature": col,
                "missing_count": missing_count,
                "missing_pct": missing_pct,
                "cardinality": cardinality,
                "top_level": top_level,
                "top_level_count": top_count,
                "top_level_pct": top_pct,
                "rare_threshold_count": rare_threshold,
                "rare_level_count": rare_level_count,
                "rare_row_count": rare_row_count,
                "constant_flag": constant_flag,
                "near_constant_flag": near_constant_flag,
            }
        )

        for level, count in value_counts.items():
            level_records.append(
                {
                    "feature": col,
                    "level": str(level),
                    "count": int(count),
                    "pct": (int(count) / n) * 100 if n else np.nan,
                    "rare_level_flag": bool(int(count) < rare_threshold),
                }
            )

    overview_df = pd.DataFrame(overview_records).sort_values(
        ["cardinality", "feature"], ascending=[False, True]
    )
    levels_df = pd.DataFrame(level_records).sort_values(
        ["feature", "count", "level"], ascending=[True, False, True]
    )
    overview_df.to_csv(OUT_DIR / "categorical_quality_overview.csv", index=False)
    levels_df.to_csv(OUT_DIR / "categorical_level_profile.csv", index=False)
    return overview_df, levels_df


def within_group_numeric_signal(df: pd.DataFrame) -> pd.DataFrame:
    if GROUP_COL not in df.columns or TARGET_COL not in df.columns:
        return pd.DataFrame()

    records: list[dict] = []
    y = pd.to_numeric(df[TARGET_COL], errors="coerce")
    num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != TARGET_COL]

    for col in num_cols:
        weighted_corr_parts: list[tuple[float, int]] = []
        group_corr_records: list[dict] = []
        for grp_name, gdf in df.groupby(GROUP_COL, dropna=False):
            xg = pd.to_numeric(gdf[col], errors="coerce")
            yg = pd.to_numeric(gdf[TARGET_COL], errors="coerce")
            valid = xg.notna() & yg.notna()
            if valid.sum() < 8:
                continue
            xgv = xg[valid]
            ygv = yg[valid]
            if xgv.nunique(dropna=True) < 2 or ygv.nunique(dropna=True) < 2:
                continue
            corr = xgv.corr(ygv, method="spearman")
            if pd.notna(corr):
                weighted_corr_parts.append((float(corr), int(valid.sum())))
                group_corr_records.append(
                    {"feature": col, "group": grp_name, "n_rows": int(valid.sum()), "spearman": float(corr)}
                )

        if weighted_corr_parts:
            corrs = np.array([c for c, _ in weighted_corr_parts], dtype=float)
            weights = np.array([w for _, w in weighted_corr_parts], dtype=float)
            weighted_corr = float(np.average(corrs, weights=weights))
            median_abs_group_corr = float(np.median(np.abs(corrs)))
            rows_used = int(weights.sum())
            groups_used = int(len(weights))
        else:
            weighted_corr = np.nan
            median_abs_group_corr = np.nan
            rows_used = 0
            groups_used = 0

        records.append(
            {
                "feature": col,
                "global_spearman_to_target": safe_corr(pd.to_numeric(df[col], errors="coerce"), y, method="spearman"),
                "weighted_spearman_within_group": weighted_corr,
                "abs_weighted_spearman": abs(weighted_corr) if pd.notna(weighted_corr) else np.nan,
                "median_abs_group_spearman": median_abs_group_corr,
                "groups_used": groups_used,
                "rows_used": rows_used,
            }
        )

        if group_corr_records:
            pd.DataFrame(group_corr_records).to_csv(
                OUT_DIR / f"within_group_corr_details__{col.replace('/', '_').replace(':', '_')}.csv",
                index=False,
            )

    out = pd.DataFrame(records).sort_values("abs_weighted_spearman", ascending=False)
    out.to_csv(OUT_DIR / "within_agriblock_numeric_signal.csv", index=False)
    return out


def categorical_target_effects(df: pd.DataFrame) -> pd.DataFrame:
    if TARGET_COL not in df.columns:
        return pd.DataFrame()

    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    records: list[dict] = []
    target = pd.to_numeric(df[TARGET_COL], errors="coerce")

    for col in cat_cols:
        global_eta2 = eta_squared(target, df[col])

        within_vals: list[float] = []
        within_weights: list[int] = []
        if GROUP_COL in df.columns:
            for _, gdf in df.groupby(GROUP_COL, dropna=False):
                eta2_local = eta_squared(
                    pd.to_numeric(gdf[TARGET_COL], errors="coerce"),
                    gdf[col],
                )
                if pd.notna(eta2_local):
                    within_vals.append(float(eta2_local))
                    within_weights.append(int(len(gdf)))

        weighted_within_eta2 = (
            float(np.average(within_vals, weights=within_weights))
            if within_vals
            else np.nan
        )

        records.append(
            {
                "feature": col,
                "cardinality": int(df[col].nunique(dropna=True)),
                "global_eta_squared": global_eta2,
                "within_group_eta_squared_weighted": weighted_within_eta2,
            }
        )

    out = pd.DataFrame(records).sort_values("global_eta_squared", ascending=False)
    out.to_csv(OUT_DIR / "categorical_target_effects.csv", index=False)
    return out


def proxy_leakage_audit(df: pd.DataFrame) -> pd.DataFrame:
    if GROUP_COL not in df.columns:
        return pd.DataFrame()

    records: list[dict] = []
    groups = df[GROUP_COL]

    for col in df.columns:
        if col == GROUP_COL:
            continue
        s = df[col]
        if pd.api.types.is_numeric_dtype(s):
            within_std_ratio, between_eta2 = numeric_group_proxy_scores(
                pd.to_numeric(s, errors="coerce"), groups
            )
            proxy_flag = bool(
                (pd.notna(between_eta2) and between_eta2 >= 0.95)
                or (pd.notna(within_std_ratio) and within_std_ratio <= 0.15)
            )
            records.append(
                {
                    "feature": col,
                    "feature_type": "numeric",
                    "group_purity_weighted": np.nan,
                    "perfect_level_ratio": np.nan,
                    "within_std_ratio": within_std_ratio,
                    "between_group_eta_squared": between_eta2,
                    "proxy_flag": proxy_flag,
                }
            )
        else:
            weighted_purity, perfect_level_ratio = categorical_group_purity(s, groups)
            proxy_flag = bool(
                (pd.notna(weighted_purity) and weighted_purity >= 0.95)
                or (pd.notna(perfect_level_ratio) and perfect_level_ratio >= 0.80)
            )
            records.append(
                {
                    "feature": col,
                    "feature_type": "categorical",
                    "group_purity_weighted": weighted_purity,
                    "perfect_level_ratio": perfect_level_ratio,
                    "within_std_ratio": np.nan,
                    "between_group_eta_squared": np.nan,
                    "proxy_flag": proxy_flag,
                }
            )

    out = pd.DataFrame(records).sort_values(
        ["proxy_flag", "feature_type", "feature"], ascending=[False, True, True]
    )
    out.to_csv(OUT_DIR / "proxy_leakage_audit.csv", index=False)
    return out


def correlation_pair_audit(df: pd.DataFrame, target_corr_map: dict[str, float]) -> pd.DataFrame:
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(num_cols) < 2:
        return pd.DataFrame()

    corr_signed = df[num_cols].corr(numeric_only=True)
    corr_abs = corr_signed.abs()

    mask = np.triu(np.ones(corr_abs.shape), k=1).astype(bool)
    pair_df = corr_abs.where(mask).stack().reset_index()
    pair_df = pair_df.rename(columns={"level_0": "feature_1", "level_1": "feature_2", 0: "abs_corr"})
    pair_df["signed_corr"] = pair_df.apply(
        lambda r: corr_signed.loc[r["feature_1"], r["feature_2"]], axis=1
    )
    pair_df["high_corr_flag"] = pair_df["abs_corr"] >= 0.98

    def choose_drop(f1: str, f2: str) -> str:
        if TARGET_COL in {f1, f2}:
            return ""
        if RAW_TARGET_COL in {f1, f2}:
            return ""
        c1 = abs(target_corr_map.get(f1, np.nan))
        c2 = abs(target_corr_map.get(f2, np.nan))
        if pd.isna(c1) and pd.isna(c2):
            return f2
        if pd.isna(c1):
            return f1
        if pd.isna(c2):
            return f2
        return f1 if c1 < c2 else f2

    pair_df["drop_candidate"] = pair_df.apply(
        lambda r: choose_drop(r["feature_1"], r["feature_2"]) if r["high_corr_flag"] else "",
        axis=1,
    )
    pair_df = pair_df.sort_values("abs_corr", ascending=False).reset_index(drop=True)
    pair_df.to_csv(OUT_DIR / "high_correlation_pairs.csv", index=False)
    return pair_df


def sectioned_eda_by_feature_group(
    df: pd.DataFrame,
    meta_df: pd.DataFrame,
    target_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    records: list[dict] = []
    feature_records: list[dict] = []

    y = pd.to_numeric(df[target_col], errors="coerce")

    for _, row in meta_df.iterrows():
        feature = row["column_name"]
        if feature == target_col or feature == RAW_TARGET_COL:
            continue
        if feature not in df.columns:
            continue

        group = row["feature_group"]
        role = row.get("final_role", row.get("base_role", "context"))
        missing_pct = float(df[feature].isna().mean() * 100)
        if pd.api.types.is_numeric_dtype(df[feature]):
            assoc = safe_corr(pd.to_numeric(df[feature], errors="coerce"), y, method="spearman")
            assoc_metric = "abs_spearman"
        else:
            assoc = eta_squared(y, df[feature])
            assoc_metric = "eta_squared"
        feature_records.append(
            {
                "feature_group": group,
                "feature": feature,
                "role": role,
                "is_numeric": bool(pd.api.types.is_numeric_dtype(df[feature])),
                "missing_pct": missing_pct,
                "association_metric": assoc_metric,
                "association_value": assoc,
                "abs_association": abs(float(assoc)) if pd.notna(assoc) else np.nan,
            }
        )

    features_df = pd.DataFrame(feature_records)
    features_df.to_csv(OUT_DIR / "sectioned_feature_level_stats.csv", index=False)

    for grp, gdf in features_df.groupby("feature_group", dropna=False):
        top_row = (
            gdf.dropna(subset=["abs_association"])
            .sort_values("abs_association", ascending=False)
            .head(1)
        )
        top_feature = str(top_row["feature"].iloc[0]) if not top_row.empty else ""
        top_assoc = float(top_row["abs_association"].iloc[0]) if not top_row.empty else np.nan

        records.append(
            {
                "feature_group": grp,
                "n_features": int(len(gdf)),
                "n_numeric": int(gdf["is_numeric"].sum()),
                "n_categorical": int((~gdf["is_numeric"]).sum()),
                "mean_missing_pct": float(gdf["missing_pct"].mean()),
                "modifiable_features": int((gdf["role"] == "modifiable").sum()),
                "context_features": int((gdf["role"] == "context").sum()),
                "proxy_features": int((gdf["role"] == "proxy").sum()),
                "top_feature_by_association": top_feature,
                "top_abs_association": top_assoc,
            }
        )

    summary_df = pd.DataFrame(records).sort_values("feature_group").reset_index(drop=True)
    summary_df.to_csv(OUT_DIR / "sectioned_group_summary.csv", index=False)
    return summary_df, features_df


def domain_validation_checks(
    df: pd.DataFrame,
    meta_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rule_rows: list[dict] = []
    issue_rows: list[dict] = []

    def record_rule(
        rule_name: str,
        mask: pd.Series,
        severity: str,
        check_type: str,
        columns: str,
        description: str,
    ) -> None:
        viol_idx = df.index[mask].tolist()
        n_viol = len(viol_idx)
        pct = (n_viol / len(df)) * 100 if len(df) else np.nan
        rule_rows.append(
            {
                "rule_name": rule_name,
                "check_type": check_type,
                "severity": severity,
                "columns": columns,
                "violations": n_viol,
                "violation_pct": pct,
                "description": description,
            }
        )
        for i in viol_idx[:30]:
            issue_rows.append(
                {
                    "rule_name": rule_name,
                    "row_index": int(i),
                    "columns": columns,
                }
            )

    # Non-negative plausibility checks for numeric input-like fields.
    nn_keywords = [
        "kg",
        "tonnes",
        "mm",
        "knots",
        "percent",
        "bundles",
        "cents",
        "hectares",
        "ml",
        "application amount",
    ]
    for _, row in meta_df.iterrows():
        col = row["column_name"]
        if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
            continue
        if col == TARGET_COL:
            continue
        unit_text = str(row.get("unit_or_levels", "")).lower()
        if any(k in unit_text for k in nn_keywords):
            s = pd.to_numeric(df[col], errors="coerce")
            record_rule(
                rule_name=f"non_negative__{col}",
                mask=s < 0,
                severity="high",
                check_type="unit_range_plausibility",
                columns=col,
                description="Expected non-negative values based on unit semantics.",
            )

    # Relative humidity bounds 0-100.
    rh_cols = [c for c in df.columns if c.startswith("Relative Humidity_")]
    for col in rh_cols:
        s = pd.to_numeric(df[col], errors="coerce")
        record_rule(
            rule_name=f"humidity_bounds__{col}",
            mask=(s < 0) | (s > 100),
            severity="high",
            check_type="unit_range_plausibility",
            columns=col,
            description="Relative humidity should lie in [0, 100].",
        )

    # Temperature plausible bounds.
    temp_cols = [c for c in df.columns if c.startswith("Min temp_") or c.startswith("Max temp_")]
    for col in temp_cols:
        s = pd.to_numeric(df[col], errors="coerce")
        record_rule(
            rule_name=f"temperature_bounds__{col}",
            mask=(s < -10) | (s > 60),
            severity="medium",
            check_type="unit_range_plausibility",
            columns=col,
            description="Temperature outside plausible agronomic bounds (-10 to 60).",
        )

    # Wind direction valid compass notation.
    wind_dir_cols = [c for c in df.columns if c.startswith("Wind Direction_")]
    valid_dirs = {
        "N",
        "NNE",
        "NE",
        "ENE",
        "E",
        "ESE",
        "SE",
        "SSE",
        "S",
        "SSW",
        "SW",
        "WSW",
        "W",
        "WNW",
        "NW",
        "NNW",
    }
    for col in wind_dir_cols:
        s = df[col].astype(str).str.upper().str.strip()
        record_rule(
            rule_name=f"wind_direction_vocab__{col}",
            mask=~s.isin(valid_dirs),
            severity="medium",
            check_type="unit_range_plausibility",
            columns=col,
            description="Wind direction should match 16-point compass codes.",
        )

    # Day-window consistency: min <= max for matching windows.
    pairs = [
        ("Min temp_D1_D30", "Max temp_D1_D30"),
        ("Min temp_D31_D60", "Max temp_D31_D60"),
        ("Min temp_D61_D90", "Max temp_D61_D90"),
        ("Min temp_D91_D120", "Max temp_D91_D120"),
    ]
    for min_col, max_col in pairs:
        if min_col in df.columns and max_col in df.columns:
            mn = pd.to_numeric(df[min_col], errors="coerce")
            mx = pd.to_numeric(df[max_col], errors="coerce")
            record_rule(
                rule_name=f"min_le_max__{min_col}__{max_col}",
                mask=mn > mx,
                severity="high",
                check_type="day_window_consistency",
                columns=f"{min_col}|{max_col}",
                description="Minimum temperature cannot exceed maximum temperature in same window.",
            )

    # Impossible combinations.
    if SIZE_COL in df.columns:
        s = pd.to_numeric(df[SIZE_COL], errors="coerce")
        record_rule(
            rule_name="hectares_positive",
            mask=s <= 0,
            severity="critical",
            check_type="impossible_combination",
            columns=SIZE_COL,
            description="Cultivated area must be strictly positive.",
        )

    if RAW_TARGET_COL in df.columns:
        y = pd.to_numeric(df[RAW_TARGET_COL], errors="coerce")
        record_rule(
            rule_name="yield_non_negative",
            mask=y < 0,
            severity="critical",
            check_type="impossible_combination",
            columns=RAW_TARGET_COL,
            description="Yield cannot be negative.",
        )

    if "Nursery area (Cents)" in df.columns and SIZE_COL in df.columns:
        nursery = pd.to_numeric(df["Nursery area (Cents)"], errors="coerce")
        hectares = pd.to_numeric(df[SIZE_COL], errors="coerce")
        max_cents = hectares * 247.105
        record_rule(
            rule_name="nursery_area_le_field_area",
            mask=nursery > max_cents,
            severity="medium",
            check_type="impossible_combination",
            columns="Nursery area (Cents)|Hectares",
            description="Nursery area should not exceed total cultivated area converted to cents.",
        )

    summary_df = pd.DataFrame(rule_rows).sort_values(
        ["violations", "severity", "rule_name"], ascending=[False, True, True]
    )
    issues_df = pd.DataFrame(issue_rows)

    summary_df.to_csv(OUT_DIR / "domain_validation_summary.csv", index=False)
    issues_df.to_csv(OUT_DIR / "domain_validation_issue_rows.csv", index=False)
    return summary_df, issues_df


def build_actionability_split(
    meta_df: pd.DataFrame,
    proxy_df: pd.DataFrame,
) -> pd.DataFrame:
    proxy_map = (
        proxy_df.set_index("feature")["proxy_flag"].to_dict() if not proxy_df.empty else {}
    )

    out = meta_df.copy()
    out["proxy_flag_from_data"] = out["column_name"].map(lambda c: bool(proxy_map.get(c, False)))
    out["final_role"] = out["base_role"]
    out.loc[out["proxy_flag_from_data"], "final_role"] = "proxy"
    out.loc[out["leakage_risk"].astype(str).str.contains("leak", case=False, na=False), "final_role"] = "proxy"
    out.to_csv(OUT_DIR / "actionability_role_map.csv", index=False)

    role_summary = (
        out.groupby(["feature_group", "final_role"], dropna=False)["column_name"]
        .count()
        .reset_index(name="n_features")
        .sort_values(["feature_group", "final_role"])
    )
    role_summary.to_csv(OUT_DIR / "actionability_role_summary.csv", index=False)
    return out


def stage_based_analysis(
    df: pd.DataFrame,
    meta_df: pd.DataFrame,
    target_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    y = pd.to_numeric(df[target_col], errors="coerce")
    per_feature_rows: list[dict] = []

    for _, row in meta_df.iterrows():
        col = row["column_name"]
        if col not in df.columns:
            continue
        if col in {target_col, RAW_TARGET_COL, GROUP_COL}:
            continue
        stage = row["stage_label"]
        if stage == "UNSTAGED":
            continue

        if pd.api.types.is_numeric_dtype(df[col]):
            assoc = safe_corr(pd.to_numeric(df[col], errors="coerce"), y, method="spearman")
            assoc_metric = "abs_spearman"
        else:
            assoc = eta_squared(y, df[col])
            assoc_metric = "eta_squared"

        per_feature_rows.append(
            {
                "stage_label": stage,
                "stage_start_day": row["stage_start_day"],
                "stage_end_day": row["stage_end_day"],
                "feature": col,
                "feature_group": row["feature_group"],
                "final_role": row["final_role"],
                "is_numeric": bool(pd.api.types.is_numeric_dtype(df[col])),
                "association_metric": assoc_metric,
                "association_value": assoc,
                "abs_association": abs(float(assoc)) if pd.notna(assoc) else np.nan,
            }
        )

    per_feature_df = pd.DataFrame(per_feature_rows).sort_values(
        ["stage_start_day", "abs_association"], ascending=[True, False]
    )
    per_feature_df.to_csv(OUT_DIR / "stage_feature_stats.csv", index=False)

    summary_rows: list[dict] = []
    for stage, gdf in per_feature_df.groupby("stage_label", dropna=False):
        top_row = (
            gdf.dropna(subset=["abs_association"])
            .sort_values("abs_association", ascending=False)
            .head(1)
        )
        summary_rows.append(
            {
                "stage_label": stage,
                "stage_start_day": gdf["stage_start_day"].iloc[0],
                "stage_end_day": gdf["stage_end_day"].iloc[0],
                "n_features": int(len(gdf)),
                "n_modifiable": int((gdf["final_role"] == "modifiable").sum()),
                "n_context": int((gdf["final_role"] == "context").sum()),
                "n_proxy": int((gdf["final_role"] == "proxy").sum()),
                "top_feature": str(top_row["feature"].iloc[0]) if not top_row.empty else "",
                "top_abs_association": float(top_row["abs_association"].iloc[0]) if not top_row.empty else np.nan,
                "mean_abs_assoc_modifiable": float(
                    gdf.loc[gdf["final_role"] == "modifiable", "abs_association"].mean()
                ),
                "mean_abs_assoc_context": float(
                    gdf.loc[gdf["final_role"] == "context", "abs_association"].mean()
                ),
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values("stage_start_day")
    summary_df.to_csv(OUT_DIR / "stage_summary.csv", index=False)
    return summary_df, per_feature_df


def build_hybrid_selection_candidates(
    df: pd.DataFrame,
    meta_df: pd.DataFrame,
    numeric_audit_df: pd.DataFrame,
    within_group_df: pd.DataFrame,
    categorical_effect_df: pd.DataFrame,
    corr_pairs_df: pd.DataFrame,
) -> pd.DataFrame:
    num_map = (
        numeric_audit_df.set_index("feature").to_dict(orient="index")
        if not numeric_audit_df.empty
        else {}
    )
    within_map = (
        within_group_df.set_index("feature").to_dict(orient="index")
        if not within_group_df.empty
        else {}
    )
    cat_map = (
        categorical_effect_df.set_index("feature").to_dict(orient="index")
        if not categorical_effect_df.empty
        else {}
    )

    high_corr_pairs = corr_pairs_df[corr_pairs_df["high_corr_flag"]] if not corr_pairs_df.empty else pd.DataFrame()
    corr_count: dict[str, int] = {}
    corr_drop_count: dict[str, int] = {}
    if not high_corr_pairs.empty:
        for _, r in high_corr_pairs.iterrows():
            f1 = r["feature_1"]
            f2 = r["feature_2"]
            drop_f = r.get("drop_candidate", "")
            corr_count[f1] = corr_count.get(f1, 0) + 1
            corr_count[f2] = corr_count.get(f2, 0) + 1
            if isinstance(drop_f, str) and drop_f:
                corr_drop_count[drop_f] = corr_drop_count.get(drop_f, 0) + 1

    rows: list[dict] = []
    for _, r in meta_df.iterrows():
        feature = r["column_name"]
        if feature not in df.columns:
            continue
        if feature in {RAW_TARGET_COL, TARGET_COL, GROUP_COL}:
            continue

        is_num = bool(pd.api.types.is_numeric_dtype(df[feature]))
        missing_pct = float(df[feature].isna().mean() * 100)
        final_role = r["final_role"]
        modeling_rec = str(r.get("modeling_recommendation", "")).strip().lower()

        if is_num:
            global_assoc = num_map.get(feature, {}).get("abs_spearman_to_target", np.nan)
            within_assoc = within_map.get(feature, {}).get("abs_weighted_spearman", np.nan)
            constant_flag = bool(num_map.get(feature, {}).get("constant_flag", False))
            near_constant_flag = bool(num_map.get(feature, {}).get("near_constant_flag", False))
        else:
            global_assoc = cat_map.get(feature, {}).get("global_eta_squared", np.nan)
            within_assoc = cat_map.get(feature, {}).get("within_group_eta_squared_weighted", np.nan)
            constant_flag = bool(df[feature].nunique(dropna=True) <= 1)
            top_ratio = float(df[feature].astype("string").fillna("<MISSING>").value_counts(normalize=True).iloc[0])
            near_constant_flag = bool(top_ratio >= 0.995)

        hi_corr = int(corr_count.get(feature, 0))
        hi_corr_drop = int(corr_drop_count.get(feature, 0))

        reasons: list[str] = []
        if final_role == "proxy":
            reasons.append("proxy_or_leakage")
        if modeling_rec == "exclude_from_modeling":
            reasons.append("dictionary_exclude")
        if constant_flag:
            reasons.append("constant")
        elif near_constant_flag:
            reasons.append("near_constant")
        if hi_corr_drop > 0:
            reasons.append(f"redundant_drop_candidate={hi_corr_drop}")

        if final_role == "proxy" or modeling_rec == "exclude_from_modeling" or constant_flag:
            status = "excluded"
        elif near_constant_flag and (pd.isna(global_assoc) or global_assoc < 0.03):
            status = "excluded"
        elif hi_corr_drop > 0 and (pd.isna(global_assoc) or global_assoc < 0.05):
            status = "excluded"
        elif final_role == "modifiable":
            status = "candidate_modifiable" if hi_corr_drop == 0 else "candidate_redundant_review"
        elif final_role == "context":
            status = "reserve_context"
        else:
            status = "excluded"

        assoc_term = 0.0 if pd.isna(global_assoc) else float(global_assoc)
        within_term = 0.0 if pd.isna(within_assoc) else float(within_assoc)
        role_bonus = 1.0 if final_role == "modifiable" else 0.25 if final_role == "context" else -0.5
        redundancy_penalty = 0.1 * hi_corr_drop
        score = role_bonus + 0.8 * assoc_term + 0.5 * within_term - redundancy_penalty

        rows.append(
            {
                "feature": feature,
                "final_role": final_role,
                "feature_group": r["feature_group"],
                "status": status,
                "missing_pct": missing_pct,
                "global_association": global_assoc,
                "within_group_association": within_assoc,
                "high_corr_pair_count": hi_corr,
                "high_corr_drop_candidate_count": hi_corr_drop,
                "hybrid_priority_score": score,
                "reasons": ";".join(reasons),
            }
        )

    out = pd.DataFrame(rows)
    status_order = {
        "candidate_modifiable": 0,
        "candidate_redundant_review": 1,
        "reserve_context": 2,
        "excluded": 3,
    }
    out["status_rank"] = out["status"].map(status_order).fillna(99)
    out = out.sort_values(["status_rank", "hybrid_priority_score"], ascending=[True, False])
    out = out.drop(columns=["status_rank"])
    out.to_csv(OUT_DIR / "hybrid_selection_candidates.csv", index=False)
    return out


def cluster_analysis(df: pd.DataFrame, proxy_df: pd.DataFrame) -> None:
    drop_cols = {TARGET_COL, RAW_TARGET_COL, GROUP_COL}
    if not proxy_df.empty:
        proxy_cols = set(proxy_df.loc[proxy_df["proxy_flag"], "feature"].tolist())
        drop_cols |= proxy_cols

    X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    if X.empty:
        print("\nCluster analysis skipped: no usable features after proxy filtering.")
        return

    sample_size = min(1000, len(X))
    sample_idx = X.sample(n=sample_size, random_state=42).index
    Xs = X.loc[sample_idx]
    sample_df = df.loc[sample_idx].copy()

    cat_cols = Xs.select_dtypes(exclude=[np.number]).columns.tolist()
    num_cols = Xs.select_dtypes(include=[np.number]).columns.tolist()
    if not cat_cols and not num_cols:
        print("\nCluster analysis skipped: no categorical or numeric columns available.")
        return

    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", StandardScaler(), num_cols),
        ]
    )
    Xenc = pre.fit_transform(Xs)

    if sample_size < 10:
        print("\nCluster analysis skipped: insufficient rows in sample.")
        return

    max_k = min(8, sample_size - 1)
    if max_k < 2:
        print("\nCluster analysis skipped: not enough rows for KMeans.")
        return

    sil_records = []
    best_k = None
    best_sil = -1.0
    best_labels = None

    print("\nCluster analysis (proxy-filtered features):")
    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(Xenc)
        sil = float(silhouette_score(Xenc, labels))
        sil_records.append({"k": k, "silhouette": sil})
        print(f"  k={k}: silhouette={sil:.4f}")
        if sil > best_sil:
            best_sil = sil
            best_k = k
            best_labels = labels

    pd.DataFrame(sil_records).to_csv(OUT_DIR / "cluster_silhouette_scores.csv", index=False)
    if best_labels is None or best_k is None:
        print("Cluster analysis skipped: best labels could not be computed.")
        return

    sample_df = sample_df.copy()
    sample_df["cluster"] = best_labels
    counts = sample_df["cluster"].value_counts().sort_index()
    print(f"Best k by silhouette: {best_k} (score {best_sil:.4f})")
    for cid, count in counts.items():
        print(f"  cluster {cid}: {int(count)} rows")

    num_profile = sample_df.select_dtypes(include=[np.number]).copy()
    if "cluster" in num_profile.columns:
        variances = num_profile.drop(columns=["cluster"]).var().sort_values(ascending=False)
        top_num = [c for c in variances.index if c != TARGET_COL][:9]
        profile_cols = [TARGET_COL] + top_num if TARGET_COL in num_profile.columns else top_num[:10]
        profile_cols = [c for c in profile_cols if c in num_profile.columns]
        profile = sample_df.groupby("cluster")[profile_cols].mean().round(3)
        profile.to_csv(OUT_DIR / "cluster_profile_top_variance.csv")

    Xdense = Xenc.toarray() if hasattr(Xenc, "toarray") else Xenc
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(Xdense)

    plt.figure(figsize=(7, 5))
    plt.scatter(coords[:, 0], coords[:, 1], c=best_labels, cmap="tab10", s=14)
    plt.title(f"Cluster PCA View (best k={best_k})")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "cluster_pca_best_k.png", dpi=150)
    plt.close()


def plot_artifacts(
    df: pd.DataFrame,
    missing_df: pd.DataFrame,
    within_group_df: pd.DataFrame,
) -> None:
    if TARGET_COL in df.columns:
        plt.figure(figsize=(8, 5))
        pd.to_numeric(df[TARGET_COL], errors="coerce").hist(bins=30)
        plt.title("Target Distribution (per-hectare yield)")
        plt.xlabel(TARGET_COL)
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(OUT_DIR / "target_distribution.png", dpi=150)
        plt.close()

    if not missing_df.empty:
        top_missing = missing_df.sort_values("missing_pct", ascending=False).head(20)
        plt.figure(figsize=(9, 6))
        plt.barh(top_missing["column"][::-1], top_missing["missing_pct"][::-1], color="#3A7CA5")
        plt.title("Top 20 Missingness Percentages")
        plt.xlabel("Missing %")
        plt.ylabel("Feature")
        plt.tight_layout()
        plt.savefig(OUT_DIR / "missingness_top20.png", dpi=150)
        plt.close()

    if GROUP_COL in df.columns and TARGET_COL in df.columns:
        labels = sorted(df[GROUP_COL].dropna().astype(str).unique().tolist())
        if labels:
            box_data = [
                pd.to_numeric(df.loc[df[GROUP_COL].astype(str) == g, TARGET_COL], errors="coerce").dropna()
                for g in labels
            ]
            plt.figure(figsize=(9, 5))
            plt.boxplot(box_data, tick_labels=labels, showfliers=False)
            plt.title("Per-hectare Yield by Agriblock")
            plt.ylabel(TARGET_COL)
            plt.xticks(rotation=25, ha="right")
            plt.tight_layout()
            plt.savefig(OUT_DIR / "target_by_agriblock_boxplot.png", dpi=150)
            plt.close()

    num_df = df.select_dtypes(include=[np.number])
    if num_df.shape[1] >= 2:
        var_order = num_df.var().sort_values(ascending=False).index.tolist()
        top_cols = [c for c in var_order if c != TARGET_COL][:19]
        if TARGET_COL in num_df.columns:
            top_cols = [TARGET_COL] + top_cols
        top_cols = top_cols[:20]
        corr = num_df[top_cols].corr(numeric_only=True)
        plt.figure(figsize=(10, 8))
        plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
        plt.colorbar()
        plt.xticks(range(len(top_cols)), top_cols, rotation=90)
        plt.yticks(range(len(top_cols)), top_cols)
        plt.title("Numeric Correlation Heatmap (Top-Variance Features)")
        plt.tight_layout()
        plt.savefig(OUT_DIR / "numeric_corr_heatmap_top20.png", dpi=150)
        plt.close()

    if not within_group_df.empty:
        top = (
            within_group_df.dropna(subset=["abs_weighted_spearman"])
            .sort_values("abs_weighted_spearman", ascending=False)
            .head(12)
        )
        if not top.empty:
            plt.figure(figsize=(10, 6))
            plt.barh(top["feature"][::-1], top["abs_weighted_spearman"][::-1], color="#0B6E4F")
            plt.title("Top Within-Agriblock Numeric Signal (|Weighted Spearman|)")
            plt.xlabel("|Weighted Spearman|")
            plt.ylabel("Feature")
            plt.tight_layout()
            plt.savefig(OUT_DIR / "top_within_group_numeric_signal.png", dpi=150)
            plt.close()


def build_feature_screening_summary(
    df: pd.DataFrame,
    numeric_audit_df: pd.DataFrame,
    categorical_overview_df: pd.DataFrame,
    within_group_df: pd.DataFrame,
    categorical_effect_df: pd.DataFrame,
    proxy_df: pd.DataFrame,
    corr_pairs_df: pd.DataFrame,
    role_map_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    feature_cols = [c for c in df.columns if c not in {TARGET_COL, RAW_TARGET_COL}]

    numeric_map = (
        numeric_audit_df.set_index("feature").to_dict(orient="index")
        if not numeric_audit_df.empty
        else {}
    )
    cat_map = (
        categorical_overview_df.set_index("feature").to_dict(orient="index")
        if not categorical_overview_df.empty
        else {}
    )
    within_map = (
        within_group_df.set_index("feature").to_dict(orient="index")
        if not within_group_df.empty
        else {}
    )
    cat_eff_map = (
        categorical_effect_df.set_index("feature").to_dict(orient="index")
        if not categorical_effect_df.empty
        else {}
    )
    proxy_map = (
        proxy_df.set_index("feature").to_dict(orient="index")
        if not proxy_df.empty
        else {}
    )
    role_map = (
        role_map_df.set_index("column_name").to_dict(orient="index")
        if role_map_df is not None
        and not role_map_df.empty
        and "column_name" in role_map_df.columns
        else {}
    )

    high_corr_pairs = corr_pairs_df[corr_pairs_df["high_corr_flag"]] if not corr_pairs_df.empty else pd.DataFrame()
    high_corr_count_map: dict[str, int] = {}
    high_corr_drop_count_map: dict[str, int] = {}
    if not high_corr_pairs.empty:
        for _, row in high_corr_pairs.iterrows():
            f1 = row["feature_1"]
            f2 = row["feature_2"]
            drop_f = row.get("drop_candidate", "")
            high_corr_count_map[f1] = high_corr_count_map.get(f1, 0) + 1
            high_corr_count_map[f2] = high_corr_count_map.get(f2, 0) + 1
            if isinstance(drop_f, str) and drop_f:
                high_corr_drop_count_map[drop_f] = high_corr_drop_count_map.get(drop_f, 0) + 1

    records: list[dict] = []
    for col in feature_cols:
        is_numeric = pd.api.types.is_numeric_dtype(df[col])
        missing_pct = float(df[col].isna().mean() * 100)

        constant_flag = False
        near_constant_flag = False
        global_assoc = np.nan
        within_assoc = np.nan
        assoc_metric = "n/a"

        if is_numeric and col in numeric_map:
            info = numeric_map[col]
            constant_flag = bool(info.get("constant_flag", False))
            near_constant_flag = bool(info.get("near_constant_flag", False))
            global_assoc = info.get("abs_spearman_to_target", np.nan)
            assoc_metric = "abs_spearman"
            within_assoc = (
                within_map.get(col, {}).get("abs_weighted_spearman", np.nan)
                if within_map
                else np.nan
            )
        elif (not is_numeric) and col in cat_map:
            cinfo = cat_map[col]
            constant_flag = bool(cinfo.get("constant_flag", False))
            near_constant_flag = bool(cinfo.get("near_constant_flag", False))
            eff = cat_eff_map.get(col, {})
            global_assoc = eff.get("global_eta_squared", np.nan)
            within_assoc = eff.get("within_group_eta_squared_weighted", np.nan)
            assoc_metric = "eta_squared"

        proxy_flag = bool(proxy_map.get(col, {}).get("proxy_flag", False))
        role_info = role_map.get(col, {})
        dict_role = str(role_info.get("final_role", "")).strip().lower()
        dict_leak = str(role_info.get("leakage_risk", "")).strip().lower()
        dict_model_rec = str(role_info.get("modeling_recommendation", "")).strip().lower()
        dictionary_proxy_or_leak = (
            dict_role == "proxy"
            or ("leak" in dict_leak)
            or ("exclude" in dict_model_rec)
        )
        high_corr_count = int(high_corr_count_map.get(col, 0))
        high_corr_drop_count = int(high_corr_drop_count_map.get(col, 0))

        reasons: list[str] = []
        if constant_flag:
            reasons.append("constant")
        elif near_constant_flag:
            reasons.append("near_constant")
        if proxy_flag:
            reasons.append("proxy_to_group")
        if dictionary_proxy_or_leak:
            reasons.append("dictionary_proxy_or_leakage")
        if high_corr_count > 0:
            reasons.append(f"high_corr_pairs={high_corr_count}")

        if dictionary_proxy_or_leak:
            recommendation = "drop"
        elif constant_flag:
            recommendation = "drop"
        elif near_constant_flag and (pd.isna(global_assoc) or global_assoc < 0.02):
            recommendation = "drop"
        elif proxy_flag and (pd.isna(within_assoc) or within_assoc < 0.03):
            recommendation = "drop"
        elif high_corr_drop_count > 0 and (pd.isna(global_assoc) or global_assoc < 0.05):
            recommendation = "drop"
        elif high_corr_count > 0 or proxy_flag:
            recommendation = "review"
        else:
            recommendation = "keep"

        records.append(
            {
                "feature": col,
                "feature_type": "numeric" if is_numeric else "categorical",
                "missing_pct": missing_pct,
                "constant_flag": constant_flag,
                "near_constant_flag": near_constant_flag,
                "proxy_flag": proxy_flag,
                "dictionary_proxy_or_leakage": dictionary_proxy_or_leak,
                "high_corr_pair_count": high_corr_count,
                "high_corr_drop_candidate_count": high_corr_drop_count,
                "association_metric": assoc_metric,
                "global_association": global_assoc,
                "within_group_association": within_assoc,
                "recommendation": recommendation,
                "reasons": ";".join(reasons),
            }
        )

    out = pd.DataFrame(records).sort_values(
        ["recommendation", "feature_type", "global_association"],
        ascending=[True, True, False],
    )
    out.to_csv(OUT_DIR / "feature_screening_summary.csv", index=False)
    return out


def print_compact_summary(
    raw_df: pd.DataFrame,
    dedup_df: pd.DataFrame,
    per_hectare_cols: list[str],
    dictionary_df: pd.DataFrame,
    role_map_df: pd.DataFrame,
    sectioned_group_df: pd.DataFrame,
    domain_validation_df: pd.DataFrame,
    stage_summary_df: pd.DataFrame,
    hybrid_candidates_df: pd.DataFrame,
    numeric_audit_df: pd.DataFrame,
    categorical_overview_df: pd.DataFrame,
    proxy_df: pd.DataFrame,
    corr_pairs_df: pd.DataFrame,
    within_group_df: pd.DataFrame,
    screening_df: pd.DataFrame,
) -> None:
    print("\nData overview")
    print(f"  Raw rows: {len(raw_df)}")
    print(f"  Deduped rows: {len(dedup_df)} (removed {len(raw_df) - len(dedup_df)})")
    print(f"  Columns after feature engineering: {dedup_df.shape[1]}")
    print(f"  Per-hectare columns created: {len(per_hectare_cols)}")
    print(f"  Dictionary-controlled features: {len(dictionary_df)}")

    if TARGET_COL in dedup_df.columns:
        print("\nTarget summary (per-hectare)")
        print(pd.to_numeric(dedup_df[TARGET_COL], errors="coerce").describe().round(3).to_string())

    if not numeric_audit_df.empty:
        n_const = int(numeric_audit_df["constant_flag"].sum())
        n_near = int(numeric_audit_df["near_constant_flag"].sum())
        print("\nNumeric variance audit")
        print(f"  Constant numeric features: {n_const}")
        print(f"  Near-constant numeric features: {n_near}")
        print("  Top numeric signals (abs Spearman to target):")
        top_num = numeric_audit_df[
            numeric_audit_df["feature"] != TARGET_COL
        ].sort_values("abs_spearman_to_target", ascending=False).head(8)
        for _, row in top_num.iterrows():
            print(f"    {row['feature']}: {row['abs_spearman_to_target']:.4f}")

    if not categorical_overview_df.empty:
        n_cat_const = int(categorical_overview_df["constant_flag"].sum())
        n_cat_rare = int((categorical_overview_df["rare_level_count"] > 0).sum())
        print("\nCategorical quality audit")
        print(f"  Constant categorical features: {n_cat_const}")
        print(f"  Features with rare levels: {n_cat_rare}")

    if not sectioned_group_df.empty:
        print("\nSectioned EDA by feature group")
        for _, row in sectioned_group_df.sort_values("feature_group").iterrows():
            print(
                f"  {row['feature_group']}: n={int(row['n_features'])} "
                f"(modifiable={int(row['modifiable_features'])}, "
                f"context={int(row['context_features'])}, proxy={int(row['proxy_features'])}) "
                f"| top={row['top_feature_by_association']}"
            )

    if not role_map_df.empty:
        print("\nActionability split")
        rc = role_map_df["final_role"].value_counts().to_dict()
        print(
            "  "
            + ", ".join(
                f"{k}={v}" for k, v in sorted(rc.items(), key=lambda x: x[0])
            )
        )

    if not domain_validation_df.empty:
        total_issues = int(domain_validation_df["violations"].sum())
        print("\nDomain validation checks")
        print(f"  Total violations across checks: {total_issues}")
        for _, row in domain_validation_df[domain_validation_df["violations"] > 0].head(8).iterrows():
            print(
                f"    {row['rule_name']}: {int(row['violations'])} rows ({row['violation_pct']:.2f}%)"
            )

    if not proxy_df.empty:
        proxy_count = int(proxy_df["proxy_flag"].sum())
        print("\nProxy/leakage audit vs Agriblock")
        print(f"  Features flagged as group proxies: {proxy_count}")
        top_proxy = proxy_df[proxy_df["proxy_flag"]].head(10)
        for _, row in top_proxy.iterrows():
            if row["feature_type"] == "categorical":
                score = row["group_purity_weighted"]
                print(f"    {row['feature']}: categorical purity={score:.4f}")
            else:
                score = row["between_group_eta_squared"]
                print(f"    {row['feature']}: between-group eta^2={score:.4f}")

    if not corr_pairs_df.empty:
        high_corr_count = int(corr_pairs_df["high_corr_flag"].sum())
        print("\nRedundancy audit")
        print(f"  High-correlation pairs (|r| >= 0.98): {high_corr_count}")
        for _, row in corr_pairs_df[corr_pairs_df["high_corr_flag"]].head(10).iterrows():
            drop_note = f" | drop_candidate={row['drop_candidate']}" if row["drop_candidate"] else ""
            print(
                f"    {row['feature_1']} <-> {row['feature_2']}: "
                f"r={row['signed_corr']:.4f}{drop_note}"
            )

    if not within_group_df.empty:
        print("\nWithin-Agriblock numeric signal")
        top_within = within_group_df.sort_values("abs_weighted_spearman", ascending=False).head(8)
        for _, row in top_within.iterrows():
            print(f"    {row['feature']}: |weighted spearman|={row['abs_weighted_spearman']:.4f}")

    if not stage_summary_df.empty:
        print("\nStage-based analysis")
        for _, row in stage_summary_df.iterrows():
            print(
                f"  {row['stage_label']}: n={int(row['n_features'])}, "
                f"modifiable={int(row['n_modifiable'])}, proxy={int(row['n_proxy'])}, "
                f"top={row['top_feature']}"
            )

    if not screening_df.empty:
        print("\nFeature screening recommendations")
        rec_counts = screening_df["recommendation"].value_counts().to_dict()
        print(
            "  "
            + ", ".join(
                f"{k}={v}" for k, v in sorted(rec_counts.items(), key=lambda x: x[0])
            )
        )
        print("  Example drop candidates:")
        for _, row in screening_df[screening_df["recommendation"] == "drop"].head(12).iterrows():
            print(f"    {row['feature']} ({row['reasons']})")

    if not hybrid_candidates_df.empty:
        print("\nHybrid-selection-ready candidates")
        hc = hybrid_candidates_df["status"].value_counts().to_dict()
        print(
            "  "
            + ", ".join(
                f"{k}={v}" for k, v in sorted(hc.items(), key=lambda x: x[0])
            )
        )
        top = hybrid_candidates_df[
            hybrid_candidates_df["status"] == "candidate_modifiable"
        ].head(12)
        for _, row in top.iterrows():
            print(
                f"    {row['feature']} | score={row['hybrid_priority_score']:.3f} "
                f"| assoc={row['global_association']:.3f}"
            )
        secondary = hybrid_candidates_df[
            hybrid_candidates_df["status"] == "candidate_redundant_review"
        ].head(8)
        if not secondary.empty:
            print("  Redundant-but-modifiable (review):")
            for _, row in secondary.iterrows():
                print(
                    f"    {row['feature']} | score={row['hybrid_priority_score']:.3f} "
                    f"| high_corr_drop={int(row['high_corr_drop_candidate_count'])}"
                )


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing dataset: {DATA_PATH}")
    if not DICT_PATH.exists():
        raise FileNotFoundError(f"Missing data dictionary: {DICT_PATH}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Running feature_prepare pipeline ({PIPELINE_VERSION})")

    raw_df = pd.read_csv(DATA_PATH)
    raw_df.columns = clean_columns(raw_df.columns)

    if RAW_TARGET_COL not in raw_df.columns:
        raise ValueError(f"Target column not found: {RAW_TARGET_COL}")
    if GROUP_COL not in raw_df.columns:
        raise ValueError(f"Group column not found: {GROUP_COL}")

    dict_df = load_data_dictionary(DICT_PATH, raw_df.columns.tolist())

    print("Shape:", raw_df.shape)
    print("\nColumns:", raw_df.columns.tolist())
    print("\nDtypes:\n", raw_df.dtypes)

    missing_df = save_missingness_audit(raw_df)
    save_duplicate_audit(raw_df)

    dedup_df = raw_df.drop_duplicates().reset_index(drop=True)
    analysis_df, per_hectare_cols = normalize_per_hectare(
        dedup_df,
        drop_original=False,
        create_input_scaled=False,
    )

    # Build dictionary metadata aligned to the deduplicated analysis frame.
    dict_meta_df = add_stage_and_role_metadata(analysis_df, dict_df)

    numeric_audit_df = numeric_variance_audit(analysis_df)
    categorical_overview_df, _ = categorical_quality_audit(analysis_df)
    within_group_df = within_group_numeric_signal(analysis_df)
    categorical_effect_df = categorical_target_effects(analysis_df)
    proxy_df = proxy_leakage_audit(analysis_df)

    role_map_df = build_actionability_split(dict_meta_df, proxy_df)
    sectioned_group_df, _ = sectioned_eda_by_feature_group(
        analysis_df,
        role_map_df,
        TARGET_COL,
    )
    domain_validation_df, _ = domain_validation_checks(dedup_df, role_map_df)
    stage_summary_df, _ = stage_based_analysis(analysis_df, role_map_df, TARGET_COL)

    target_corr_map = {
        str(r["feature"]): float(r["abs_spearman_to_target"])
        for _, r in numeric_audit_df.iterrows()
        if pd.notna(r["abs_spearman_to_target"])
    }
    corr_pairs_df = correlation_pair_audit(analysis_df, target_corr_map)

    screening_df = build_feature_screening_summary(
        df=analysis_df,
        numeric_audit_df=numeric_audit_df,
        categorical_overview_df=categorical_overview_df,
        within_group_df=within_group_df,
        categorical_effect_df=categorical_effect_df,
        proxy_df=proxy_df,
        corr_pairs_df=corr_pairs_df,
        role_map_df=role_map_df,
    )

    hybrid_candidates_df = build_hybrid_selection_candidates(
        df=analysis_df,
        meta_df=role_map_df,
        numeric_audit_df=numeric_audit_df,
        within_group_df=within_group_df,
        categorical_effect_df=categorical_effect_df,
        corr_pairs_df=corr_pairs_df,
    )

    plot_artifacts(analysis_df, missing_df, within_group_df)
    cluster_analysis(analysis_df, proxy_df)

    print_compact_summary(
        raw_df=raw_df,
        dedup_df=analysis_df,
        per_hectare_cols=per_hectare_cols,
        dictionary_df=role_map_df,
        role_map_df=role_map_df,
        sectioned_group_df=sectioned_group_df,
        domain_validation_df=domain_validation_df,
        stage_summary_df=stage_summary_df,
        hybrid_candidates_df=hybrid_candidates_df,
        numeric_audit_df=numeric_audit_df,
        categorical_overview_df=categorical_overview_df,
        proxy_df=proxy_df,
        corr_pairs_df=corr_pairs_df,
        within_group_df=within_group_df,
        screening_df=screening_df,
    )

    print("\nSaved artifacts to:", OUT_DIR)


if __name__ == "__main__":
    main()
