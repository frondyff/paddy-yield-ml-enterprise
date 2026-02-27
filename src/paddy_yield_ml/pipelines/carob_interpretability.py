"""CAROB interpretability pipeline with a 3-iteration defensibility loop.

This pipeline is designed to be causal-ready (associative outputs with explicit caveats):
1) Iteration 1: SHAP + permutation + local examples + initial modifiable-only rules.
2) Iteration 2: cross-seed stability diagnostics for feature ranks and rule effects.
3) Iteration 3: final defensible rule set with confidence labels and action playbook.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor

from paddy_yield_ml.pipelines import carob_common as cc
from paddy_yield_ml.pipelines import carob_model_compare as cm

try:
    project_root = Path(__file__).resolve().parents[3]
except NameError:
    project_root = Path.cwd()

OUT_ROOT = project_root / "outputs" / "carob_interpretability"
DEFAULT_CANDIDATES_PATH = project_root / "outputs" / "carob_feature_prepare" / "hybrid_selection_candidates.csv"
DEFAULT_ROLE_MAP_PATH = project_root / "outputs" / "carob_feature_prepare" / "actionability_role_map.csv"
DEFAULT_WINNERS_PATH = project_root / "outputs" / "carob_model_tune_top2" / "model_winners.csv"
DEFAULT_SCENARIO = "modifiable_plus_context"
TREE_LEAF = -1
COND_RE = re.compile(r"^(?P<feature>.+?)\s*(?P<op><=|>=|<|>)\s*(?P<threshold>-?\d+(?:\.\d+)?)$")


@dataclass
class HoldoutBundle:
    x_train: pd.DataFrame
    x_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    groups_train: pd.Series
    groups_test: pd.Series
    countries_train: pd.Series
    countries_test: pd.Series
    cat_cols: list[str]
    bool_like_cols: set[str]


def parse_seed_list(raw: str) -> list[int]:
    out = [int(x.strip()) for x in str(raw).split(",") if x.strip()]
    if not out:
        raise ValueError("At least one seed is required.")
    return list(dict.fromkeys(out))


def load_role_map(path: Path, frame_cols: list[str]) -> pd.DataFrame:
    if path.exists():
        role_df = pd.read_csv(path)
    else:
        role_df = cc.load_role_map(frame_cols=frame_cols)
    required = {"column_name", "final_role", "feature_group", "meaning", "unit_or_levels"}
    missing = required - set(role_df.columns)
    if missing:
        raise ValueError(f"Role map missing columns: {sorted(missing)}")
    role_df = role_df.copy()
    role_df["column_name"] = role_df["column_name"].astype(str)
    role_df = role_df.drop_duplicates(subset=["column_name"], keep="first")
    return role_df


def load_best_catboost_params(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing tuned winner file: {path}")
    df = pd.read_csv(path)
    required = {"model_key", "params_json"}
    if not required.issubset(df.columns):
        raise ValueError(f"Winner file missing columns: {sorted(required)}")

    mask = df["model_key"].astype(str).str.lower() == "catboost"
    if not mask.any():
        raise ValueError("No CatBoost row found in winners file.")
    row = df.loc[mask].sort_values("r2_mean", ascending=False).iloc[0]
    raw = str(row["params_json"])
    try:
        params = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Could not parse params_json: {raw}") from exc
    if not isinstance(params, dict):
        raise ValueError("Parsed CatBoost params are not a JSON object.")
    if "n_estimators" in params and "iterations" not in params:
        params["iterations"] = int(params.pop("n_estimators"))
    return params


def load_frame_and_features(candidates_path: Path, scenario: str) -> tuple[pd.DataFrame, list[str]]:
    frame = cm.load_frame_with_country_gate(candidates_path)
    candidates_df = cm.load_hybrid_candidates(candidates_path)
    feature_sets = cm.build_feature_sets(candidates_df)
    if scenario not in feature_sets:
        raise ValueError(f"Scenario '{scenario}' not found. Available: {sorted(feature_sets)}")
    features = cm.filter_available_features(feature_sets[scenario], frame)
    return frame, features


def _coerce_bool_to_int(x: pd.DataFrame) -> tuple[pd.DataFrame, set[str]]:
    out = x.copy()
    bool_cols = out.select_dtypes(include=["bool"]).columns.tolist()
    for col in bool_cols:
        out[col] = out[col].astype(int)
    return out, set(bool_cols)


def build_holdout_bundle(
    frame: pd.DataFrame,
    features: list[str],
    *,
    seed: int,
    test_size: float,
    trial_median_impute: bool,
) -> HoldoutBundle:
    extra = ["country"] if "country" in frame.columns else []
    required = list(dict.fromkeys(features + [cc.TARGET_COL, cc.GROUP_COL] + extra))
    subset = frame[required].copy()
    subset = subset.loc[:, ~subset.columns.duplicated()].copy()
    subset[cc.TARGET_COL] = pd.to_numeric(subset[cc.TARGET_COL], errors="coerce")
    subset = subset[subset[cc.TARGET_COL].notna() & subset[cc.GROUP_COL].notna()].reset_index(drop=True)

    x = subset[features].copy()
    y = subset[cc.TARGET_COL].copy()
    groups = subset[cc.GROUP_COL].astype(str)
    if "country" in subset.columns:
        countries = subset["country"].astype(str).fillna("unknown")
    else:
        countries = pd.Series(["unknown"] * len(subset), dtype="string")

    tr_idx, te_idx = cm.build_trial_aware_split_indices(groups=groups, test_size=test_size, random_state=seed)
    x_train, x_test = x.iloc[tr_idx].copy(), x.iloc[te_idx].copy()
    y_train, y_test = y.iloc[tr_idx].copy(), y.iloc[te_idx].copy()
    g_train, g_test = groups.iloc[tr_idx].copy(), groups.iloc[te_idx].copy()
    c_train, c_test = countries.iloc[tr_idx].copy(), countries.iloc[te_idx].copy()

    if trial_median_impute:
        x_train, x_test = cm.impute_numeric_by_group(
            x_train=x_train,
            x_test=x_test,
            groups_train=g_train,
            groups_test=g_test,
        )

    x_train, train_bool = _coerce_bool_to_int(x_train)
    x_test, test_bool = _coerce_bool_to_int(x_test)
    bool_like_cols = train_bool | test_bool

    numeric_cols = x_train.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        med = x_train[numeric_cols].apply(pd.to_numeric, errors="coerce").median(numeric_only=True)
        for col in numeric_cols:
            x_train[col] = pd.to_numeric(x_train[col], errors="coerce").fillna(float(med.get(col, np.nan)))
            x_test[col] = pd.to_numeric(x_test[col], errors="coerce").fillna(float(med.get(col, np.nan)))

    cat_cols = [c for c in x_train.columns if c not in numeric_cols]
    for col in cat_cols:
        x_train[col] = x_train[col].astype(str).fillna("MISSING")
        x_test[col] = x_test[col].astype(str).fillna("MISSING")

    x_train = x_train.reset_index(drop=True)
    x_test = x_test.reset_index(drop=True)

    return HoldoutBundle(
        x_train=x_train,
        x_test=x_test,
        y_train=y_train.reset_index(drop=True),
        y_test=y_test.reset_index(drop=True),
        groups_train=g_train.reset_index(drop=True),
        groups_test=g_test.reset_index(drop=True),
        countries_train=c_train.reset_index(drop=True),
        countries_test=c_test.reset_index(drop=True),
        cat_cols=cat_cols,
        bool_like_cols=bool_like_cols,
    )


def build_catboost(params: dict[str, Any], seed: int) -> CatBoostRegressor:
    base: dict[str, Any] = {
        "iterations": 500,
        "learning_rate": 0.05,
        "depth": 6,
        "l2_leaf_reg": 3.0,
        "loss_function": "RMSE",
        "random_seed": seed,
        "verbose": False,
        "allow_writing_files": False,
    }
    tuned = dict(params)
    tuned.pop("n_estimators", None)
    base.update(tuned)
    return CatBoostRegressor(**base)


def fit_and_explain(
    bundle: HoldoutBundle,
    params: dict[str, Any],
    *,
    seed: int,
) -> tuple[CatBoostRegressor, pd.Series, pd.Series, pd.DataFrame, float, dict[str, float]]:
    model = build_catboost(params=params, seed=seed)
    cat_idx = [bundle.x_train.columns.get_loc(c) for c in bundle.cat_cols]
    train_pool = Pool(bundle.x_train, label=bundle.y_train, cat_features=cat_idx)
    test_pool = Pool(bundle.x_test, label=bundle.y_test, cat_features=cat_idx)

    model.fit(train_pool)
    pred_train = pd.Series(model.predict(train_pool), index=np.arange(len(bundle.x_train)))
    pred_test = pd.Series(model.predict(test_pool), index=np.arange(len(bundle.x_test)))

    shap_raw = model.get_feature_importance(data=test_pool, type="ShapValues")
    if shap_raw.ndim != 2 or shap_raw.shape[1] != bundle.x_test.shape[1] + 1:
        raise ValueError("Unexpected SHAP shape from CatBoost.")
    shap_df = pd.DataFrame(shap_raw[:, :-1], columns=bundle.x_test.columns, index=bundle.x_test.index)
    expected_value = float(np.mean(shap_raw[:, -1]))

    metrics = {
        "r2_test": float(r2_score(bundle.y_test, pred_test)),
        "rmse_test": float(np.sqrt(mean_squared_error(bundle.y_test, pred_test))),
        "mae_test": float(mean_absolute_error(bundle.y_test, pred_test)),
        "r2_train": float(r2_score(bundle.y_train, pred_train)),
        "rmse_train": float(np.sqrt(mean_squared_error(bundle.y_train, pred_train))),
        "mae_train": float(mean_absolute_error(bundle.y_train, pred_train)),
    }
    return model, pred_train, pred_test, shap_df, expected_value, metrics


def make_global_shap_table(shap_df: pd.DataFrame, role_map_df: pd.DataFrame) -> pd.DataFrame:
    role_lookup = role_map_df.set_index("column_name")["final_role"].to_dict()
    out = (
        shap_df.abs()
        .mean(axis=0)
        .sort_values(ascending=False)
        .rename_axis("feature")
        .reset_index(name="mean_abs_shap")
    )
    out["shap_rank"] = np.arange(1, len(out) + 1)
    out["final_role"] = out["feature"].map(role_lookup).fillna("unknown")
    return out


def compute_permutation_importance(
    model: CatBoostRegressor,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    cat_cols: list[str],
    *,
    seed: int,
    n_repeats: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cat_idx = [x_test.columns.get_loc(c) for c in cat_cols]

    base_pred = pd.Series(model.predict(Pool(x_test, cat_features=cat_idx)))
    base_rmse = float(np.sqrt(mean_squared_error(y_test, base_pred)))
    base_r2 = float(r2_score(y_test, base_pred))

    rows: list[dict[str, float | str]] = []
    for feature in x_test.columns.tolist():
        deltas_rmse: list[float] = []
        deltas_r2: list[float] = []
        source = x_test[feature].to_numpy(copy=True)
        for _ in range(int(n_repeats)):
            x_perm = x_test.copy()
            x_perm[feature] = rng.permutation(source)
            pred_perm = pd.Series(model.predict(Pool(x_perm, cat_features=cat_idx)))
            rmse_perm = float(np.sqrt(mean_squared_error(y_test, pred_perm)))
            r2_perm = float(r2_score(y_test, pred_perm))
            deltas_rmse.append(rmse_perm - base_rmse)
            deltas_r2.append(base_r2 - r2_perm)

        rows.append(
            {
                "feature": feature,
                "delta_rmse_mean": float(np.mean(deltas_rmse)),
                "delta_rmse_std": float(np.std(deltas_rmse)),
                "delta_r2_mean": float(np.mean(deltas_r2)),
                "delta_r2_std": float(np.std(deltas_r2)),
            }
        )

    out = pd.DataFrame(rows).sort_values("delta_rmse_mean", ascending=False).reset_index(drop=True)
    out["perm_rank"] = np.arange(1, len(out) + 1)
    return out


def plot_top_bar(
    df: pd.DataFrame,
    *,
    feature_col: str,
    value_col: str,
    role_col: str | None,
    top_n: int,
    out_path: Path,
    title: str,
    x_label: str,
) -> None:
    top = df.head(int(top_n)).iloc[::-1]
    colors = ["#1b9e77"] * len(top)
    if role_col is not None and role_col in top.columns:
        palette = {"modifiable": "#1b9e77", "context": "#4e79a7", "proxy": "#e15759", "unknown": "#888888"}
        colors = [palette.get(str(v), "#888888") for v in top[role_col]]

    plt.figure(figsize=(10, 6))
    plt.barh(top[feature_col], top[value_col], color=colors)
    plt.title(title)
    plt.xlabel(x_label)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def select_local_cases(y_true: pd.Series, y_pred: pd.Series) -> dict[str, int]:
    frame = pd.DataFrame({"y_true": y_true, "y_pred": y_pred}).copy()
    frame["residual"] = frame["y_pred"] - frame["y_true"]
    frame["abs_residual"] = frame["residual"].abs()

    used: set[int] = set()
    out: dict[str, int] = {}

    def pick(name: str, order: list[int]) -> None:
        for idx in order:
            if idx not in used:
                used.add(idx)
                out[name] = idx
                return

    pick("highest_pred", frame.sort_values("y_pred", ascending=False).index.tolist())
    pick("lowest_pred", frame.sort_values("y_pred", ascending=True).index.tolist())
    pick("largest_overpred", frame.sort_values("residual", ascending=False).index.tolist())
    pick("largest_underpred", frame.sort_values("residual", ascending=True).index.tolist())
    pick("largest_abs_error", frame.sort_values("abs_residual", ascending=False).index.tolist())
    return out


def save_local_outputs(
    x_test: pd.DataFrame,
    y_test: pd.Series,
    groups_test: pd.Series,
    pred_test: pd.Series,
    shap_df: pd.DataFrame,
    *,
    case_map: dict[str, int],
    top_n_features: int,
    out_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    overview_rows: list[dict[str, Any]] = []
    detail_rows: list[dict[str, Any]] = []

    for case_name, idx in case_map.items():
        actual = float(y_test.iloc[idx])
        pred = float(pred_test.iloc[idx])
        residual = pred - actual
        overview_rows.append(
            {
                "case_name": case_name,
                "row_index": int(idx),
                "trial_id": str(groups_test.iloc[idx]),
                "actual_yield": actual,
                "predicted_yield": pred,
                "residual_pred_minus_actual": residual,
                "abs_residual": abs(residual),
            }
        )

        row_shap = shap_df.iloc[idx].copy()
        row_feat = x_test.iloc[idx]
        top_feats = row_shap.abs().sort_values(ascending=False).head(int(top_n_features)).index.tolist()

        vals: list[float] = []
        labels: list[str] = []
        cols: list[str] = []
        for rank, feat in enumerate(top_feats, start=1):
            sv = float(row_shap.loc[feat])
            fv = row_feat.loc[feat]
            detail_rows.append(
                {
                    "case_name": case_name,
                    "row_index": int(idx),
                    "feature_rank": rank,
                    "feature": feat,
                    "feature_value": fv,
                    "shap_value": sv,
                    "abs_shap_value": abs(sv),
                    "direction": "positive" if sv >= 0 else "negative",
                }
            )
            vals.append(sv)
            labels.append(feat)
            cols.append("#1b9e77" if sv >= 0 else "#e15759")

        order = np.argsort(np.abs(np.array(vals)))
        vals = [vals[i] for i in order]
        labels = [labels[i] for i in order]
        cols = [cols[i] for i in order]

        plt.figure(figsize=(10, 6))
        plt.barh(labels, vals, color=cols)
        plt.axvline(0.0, color="#666666", linewidth=1)
        plt.title(f"Local SHAP Contributions: {case_name}")
        plt.xlabel("SHAP contribution to prediction")
        plt.tight_layout()
        plt.savefig(out_dir / f"iteration1_local_shap_{case_name}.png", dpi=180)
        plt.close()

    overview_df = pd.DataFrame(overview_rows)
    detail_df = pd.DataFrame(detail_rows)
    overview_df.to_csv(out_dir / "iteration1_local_cases_overview.csv", index=False)
    detail_df.to_csv(out_dir / "iteration1_local_shap_details.csv", index=False)
    return overview_df, detail_df


def build_crosscheck_table(
    global_shap_df: pd.DataFrame,
    perm_df: pd.DataFrame,
    role_map_df: pd.DataFrame,
) -> pd.DataFrame:
    role_lookup = role_map_df.set_index("column_name")["final_role"].to_dict()
    merged = pd.merge(
        global_shap_df[["feature", "mean_abs_shap", "shap_rank"]],
        perm_df[["feature", "delta_rmse_mean", "delta_r2_mean", "perm_rank"]],
        on="feature",
        how="outer",
    )
    merged["final_role"] = merged["feature"].map(role_lookup).fillna("unknown")
    merged["shap_rank"] = pd.to_numeric(merged["shap_rank"], errors="coerce")
    merged["perm_rank"] = pd.to_numeric(merged["perm_rank"], errors="coerce")
    merged["rank_gap_abs"] = (merged["shap_rank"] - merged["perm_rank"]).abs()
    merged["consensus_top10"] = (merged["shap_rank"] <= 10) & (merged["perm_rank"] <= 10)
    merged = merged.sort_values(["consensus_top10", "shap_rank", "perm_rank"], ascending=[False, True, True])
    return merged.reset_index(drop=True)


def numeric_grid(values: pd.Series, n_points: int, low_q: float = 0.05, high_q: float = 0.95) -> np.ndarray:
    arr = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if len(arr) < 10:
        return np.array([])
    qs = np.linspace(low_q, high_q, int(n_points))
    grid = np.unique(np.quantile(arr, qs))
    return grid.astype(float)


def compute_pdp_curve(
    model: CatBoostRegressor,
    x_ref: pd.DataFrame,
    cat_cols: list[str],
    feature: str,
    *,
    n_points: int,
) -> pd.DataFrame:
    grid = numeric_grid(x_ref[feature], n_points=n_points)
    if len(grid) < 2:
        return pd.DataFrame(columns=["x", "pdp_centered"])
    cat_idx = [x_ref.columns.get_loc(c) for c in cat_cols]
    means: list[float] = []
    for val in grid:
        x_mod = x_ref.copy()
        x_mod[feature] = float(val)
        pred = model.predict(Pool(x_mod, cat_features=cat_idx))
        means.append(float(np.mean(pred)))
    arr = np.array(means, dtype=float)
    centered = arr - float(np.mean(arr))
    return pd.DataFrame({"x": grid, "pdp_centered": centered})


def compute_ale_curve(
    model: CatBoostRegressor,
    x_ref: pd.DataFrame,
    cat_cols: list[str],
    feature: str,
    *,
    n_bins: int,
) -> pd.DataFrame:
    vals = pd.to_numeric(x_ref[feature], errors="coerce")
    vals = vals.dropna()
    if len(vals) < 20:
        return pd.DataFrame(columns=["x", "ale_centered", "support_count"])
    edges = np.unique(np.quantile(vals.to_numpy(dtype=float), np.linspace(0.0, 1.0, int(n_bins) + 1)))
    if len(edges) < 3:
        return pd.DataFrame(columns=["x", "ale_centered", "support_count"])

    cat_idx = [x_ref.columns.get_loc(c) for c in cat_cols]
    effects: list[float] = []
    centers: list[float] = []
    counts: list[int] = []

    for i in range(len(edges) - 1):
        low = float(edges[i])
        high = float(edges[i + 1])
        if i == len(edges) - 2:
            mask = (pd.to_numeric(x_ref[feature], errors="coerce") >= low) & (
                pd.to_numeric(x_ref[feature], errors="coerce") <= high
            )
        else:
            mask = (pd.to_numeric(x_ref[feature], errors="coerce") >= low) & (
                pd.to_numeric(x_ref[feature], errors="coerce") < high
            )
        idx = np.where(mask.to_numpy())[0]
        if len(idx) == 0:
            continue

        x_bin = x_ref.iloc[idx].copy()
        x_low = x_bin.copy()
        x_high = x_bin.copy()
        x_low[feature] = low
        x_high[feature] = high
        pred_low = model.predict(Pool(x_low, cat_features=cat_idx))
        pred_high = model.predict(Pool(x_high, cat_features=cat_idx))
        effects.append(float(np.mean(pred_high - pred_low)))
        centers.append((low + high) / 2.0)
        counts.append(int(len(idx)))

    if not effects:
        return pd.DataFrame(columns=["x", "ale_centered", "support_count"])

    ale_vals = np.cumsum(np.array(effects, dtype=float))
    w = np.array(counts, dtype=float)
    center_term = float(np.sum(ale_vals * w) / np.sum(w))
    ale_centered = ale_vals - center_term
    return pd.DataFrame({"x": centers, "ale_centered": ale_centered, "support_count": counts})


def plot_ale_pdp(
    ale_df: pd.DataFrame,
    pdp_df: pd.DataFrame,
    *,
    feature: str,
    out_path: Path,
) -> None:
    if ale_df.empty and pdp_df.empty:
        return
    plt.figure(figsize=(10, 6))
    if not ale_df.empty:
        plt.plot(ale_df["x"], ale_df["ale_centered"], marker="o", label="ALE (centered)", color="#1b9e77")
    if not pdp_df.empty:
        plt.plot(pdp_df["x"], pdp_df["pdp_centered"], marker="x", label="PDP (centered)", color="#4e79a7")
    plt.axhline(0.0, color="#666666", linewidth=1)
    plt.title(f"Iteration 1: ALE/PDP for {feature}")
    plt.xlabel(feature)
    plt.ylabel("Centered effect on predicted yield")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def _leaf_condition_strings(tree: DecisionTreeRegressor, feature_names: list[str]) -> dict[int, list[str]]:
    tree_ = tree.tree_
    out: dict[int, list[str]] = {}

    def walk(node_id: int, conds: list[str]) -> None:
        if tree_.children_left[node_id] == TREE_LEAF:
            out[node_id] = conds.copy()
            return
        feat_idx = int(tree_.feature[node_id])
        feat_name = feature_names[feat_idx]
        thr = float(tree_.threshold[node_id])
        walk(int(tree_.children_left[node_id]), conds + [f"{feat_name} <= {thr:.4f}"])
        walk(int(tree_.children_right[node_id]), conds + [f"{feat_name} > {thr:.4f}"])

    walk(0, [])
    return out


def parse_rule_conditions(rule_conditions: str) -> list[tuple[str, str, float]]:
    parts = [p.strip() for p in str(rule_conditions).split("AND") if p.strip()]
    out: list[tuple[str, str, float]] = []
    for part in parts:
        m = COND_RE.match(part)
        if not m:
            continue
        feat = m.group("feature").strip()
        op = m.group("op").strip()
        thr = float(m.group("threshold"))
        out.append((feat, op, thr))
    return out


def apply_rule_conditions(x: pd.DataFrame, rule_conditions: str) -> pd.Series:
    parsed = parse_rule_conditions(rule_conditions)
    if not parsed:
        return pd.Series([True] * len(x), index=x.index)
    mask = pd.Series([True] * len(x), index=x.index)
    for feat, op, thr in parsed:
        if feat not in x.columns:
            return pd.Series([False] * len(x), index=x.index)
        values = pd.to_numeric(x[feat], errors="coerce")
        if op == "<=":
            mask = mask & (values <= thr)
        elif op == ">":
            mask = mask & (values > thr)
        elif op == "<":
            mask = mask & (values < thr)
        elif op == ">=":
            mask = mask & (values >= thr)
    return mask.fillna(False)


def extract_modifiable_rules(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    pred_train: pd.Series,
    pred_test: pd.Series,
    *,
    modifiable_numeric_features: list[str],
    n_rules: int,
    min_support_pct: float,
    random_state: int,
) -> pd.DataFrame:
    if not modifiable_numeric_features:
        raise ValueError("No numeric modifiable features available for rule extraction.")

    xr_train = x_train[modifiable_numeric_features].copy()
    xr_test = x_test[modifiable_numeric_features].copy()
    for col in xr_train.columns:
        xr_train[col] = pd.to_numeric(xr_train[col], errors="coerce")
        xr_test[col] = pd.to_numeric(xr_test[col], errors="coerce")
        med = float(xr_train[col].median())
        xr_train[col] = xr_train[col].fillna(med)
        xr_test[col] = xr_test[col].fillna(med)

    n_samples = len(xr_train)
    min_leaf = max(20, int(np.ceil(n_samples * float(min_support_pct))))

    best_tree: DecisionTreeRegressor | None = None
    for depth in range(2, 8):
        tree = DecisionTreeRegressor(max_depth=depth, min_samples_leaf=min_leaf, random_state=random_state)
        tree.fit(xr_train, pred_train)
        leaf_count = int(np.sum(tree.tree_.children_left == TREE_LEAF))
        best_tree = tree
        if leaf_count >= int(n_rules):
            break
    if best_tree is None:
        raise RuntimeError("Could not fit surrogate tree for rule extraction.")

    leaf_ids_train = best_tree.apply(xr_train)
    unique_leaves = sorted(np.unique(leaf_ids_train).tolist())
    leaf_rules = _leaf_condition_strings(best_tree, xr_train.columns.tolist())

    global_actual = float(np.mean(y_test))
    global_pred = float(np.mean(pred_test))
    rows: list[dict[str, Any]] = []

    for leaf_id in unique_leaves:
        conds = leaf_rules.get(int(leaf_id), [])
        rule_cond = " AND ".join(conds) if conds else "(all rows)"
        mask_test = apply_rule_conditions(xr_test, rule_cond)
        support = int(mask_test.sum())
        if support == 0:
            continue

        pred_mean = float(np.mean(pred_test[mask_test]))
        actual_mean = float(np.mean(y_test[mask_test]))
        rows.append(
            {
                "rule_conditions": rule_cond,
                "features_used": "|".join(sorted({f for f, _, _ in parse_rule_conditions(rule_cond)})),
                "support_count": support,
                "support_pct": float(support / len(xr_test)),
                "predicted_yield_mean": pred_mean,
                "actual_yield_mean": actual_mean,
                "predicted_lift_vs_global": float(pred_mean - global_pred),
                "actual_lift_vs_global": float(actual_mean - global_actual),
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError("No qualifying rules found.")
    out = out.sort_values(["support_pct", "actual_lift_vs_global"], ascending=[False, False]).reset_index(drop=True)
    out.insert(0, "rule_id", [f"R{i}" for i in range(1, len(out) + 1)])
    return out.head(max(8, int(n_rules))).copy()


def condition_to_english(
    feature: str,
    op: str,
    threshold: float,
    role_map_df: pd.DataFrame,
    bool_like_cols: set[str],
) -> str:
    row = role_map_df.loc[role_map_df["column_name"].astype(str) == feature]
    meaning = str(row["meaning"].iloc[0]) if not row.empty else feature
    unit = str(row["unit_or_levels"].iloc[0]) if not row.empty else ""
    unit_part = f" ({unit})" if unit and unit.lower() != "unknown" else ""

    if feature in bool_like_cols:
        if op in {"<=", "<"} and threshold <= 0.5:
            return f"{meaning} is No/0"
        if op in {">", ">="} and threshold >= 0.5:
            return f"{meaning} is Yes/1"

    thr_txt = f"{threshold:.2f}"
    if op == "<=":
        return f"{meaning}{unit_part} is at or below {thr_txt}"
    if op == ">":
        return f"{meaning}{unit_part} is above {thr_txt}"
    if op == "<":
        return f"{meaning}{unit_part} is below {thr_txt}"
    if op == ">=":
        return f"{meaning}{unit_part} is at or above {thr_txt}"
    return f"{meaning}{unit_part} has condition {op} {thr_txt}"


def rules_to_english_markdown(
    rules_df: pd.DataFrame,
    role_map_df: pd.DataFrame,
    bool_like_cols: set[str],
    out_path: Path,
    *,
    title: str,
) -> None:
    lines = [f"# {title}", ""]
    for _, row in rules_df.iterrows():
        parsed = parse_rule_conditions(str(row["rule_conditions"]))
        conds = [condition_to_english(f, op, thr, role_map_df, bool_like_cols) for f, op, thr in parsed]
        cond_text = "; and ".join(conds) if conds else "all rows"
        lift = float(row.get("actual_lift_vs_global", row.get("lift_mean", 0.0)))
        support = float(row.get("support_pct", row.get("support_mean", 0.0))) * 100.0
        direction = "higher" if lift >= 0 else "lower"
        lines.append(
            f"- **{row['rule_id']}**: If {cond_text}, this segment shows **{direction} yield by {lift:+.1f} kg/ha** "
            f"vs baseline (support: {support:.1f}%)."
        )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def seed_snapshot(
    frame: pd.DataFrame,
    features: list[str],
    role_map_df: pd.DataFrame,
    params: dict[str, Any],
    *,
    seed: int,
    test_size: float,
    trial_median_impute: bool,
    perm_repeats: int,
) -> dict[str, Any]:
    bundle = build_holdout_bundle(
        frame=frame,
        features=features,
        seed=seed,
        test_size=test_size,
        trial_median_impute=trial_median_impute,
    )
    model, pred_train, pred_test, shap_df, expected_value, metrics = fit_and_explain(bundle, params=params, seed=seed)
    global_shap = make_global_shap_table(shap_df, role_map_df)
    perm = compute_permutation_importance(
        model=model,
        x_test=bundle.x_test,
        y_test=bundle.y_test,
        cat_cols=bundle.cat_cols,
        seed=seed + 777,
        n_repeats=perm_repeats,
    )
    cross = build_crosscheck_table(global_shap, perm, role_map_df)
    return {
        "bundle": bundle,
        "model": model,
        "pred_train": pred_train,
        "pred_test": pred_test,
        "shap_df": shap_df,
        "global_shap": global_shap,
        "perm": perm,
        "cross": cross,
        "metrics": metrics,
        "expected_value": expected_value,
    }


def aggregate_feature_stability(seed_rows: list[dict[str, Any]], role_map_df: pd.DataFrame) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for row in seed_rows:
        seed = int(row["seed"])
        shap = row["global_shap"][["feature", "shap_rank", "mean_abs_shap"]].copy()
        perm = row["perm"][["feature", "perm_rank", "delta_rmse_mean"]].copy()
        merged = pd.merge(shap, perm, on="feature", how="outer")
        merged["seed"] = seed
        pieces.append(merged)
    all_df = pd.concat(pieces, ignore_index=True)
    agg = (
        all_df.groupby("feature", as_index=False)
        .agg(
            shap_rank_mean=("shap_rank", "mean"),
            shap_rank_std=("shap_rank", "std"),
            perm_rank_mean=("perm_rank", "mean"),
            perm_rank_std=("perm_rank", "std"),
            shap_top10_rate=("shap_rank", lambda s: float(np.mean(pd.to_numeric(s, errors="coerce") <= 10))),
            perm_top10_rate=("perm_rank", lambda s: float(np.mean(pd.to_numeric(s, errors="coerce") <= 10))),
            mean_abs_shap_mean=("mean_abs_shap", "mean"),
            delta_rmse_mean=("delta_rmse_mean", "mean"),
        )
        .sort_values(["shap_rank_mean", "perm_rank_mean"], ascending=[True, True])
        .reset_index(drop=True)
    )
    role_lookup = role_map_df.set_index("column_name")["final_role"].to_dict()
    agg["final_role"] = agg["feature"].map(role_lookup).fillna("unknown")
    agg["joint_top10_rate"] = np.minimum(agg["shap_top10_rate"], agg["perm_top10_rate"])
    return agg


def evaluate_rules_over_seeds(
    rules_df: pd.DataFrame,
    frame: pd.DataFrame,
    features: list[str],
    params: dict[str, Any],
    *,
    seeds: list[int],
    test_size: float,
    trial_median_impute: bool,
) -> pd.DataFrame:
    all_rows: list[dict[str, Any]] = []
    for seed in seeds:
        snap = seed_snapshot(
            frame=frame,
            features=features,
            role_map_df=pd.DataFrame({"column_name": features, "final_role": "unknown"}),
            params=params,
            seed=int(seed),
            test_size=test_size,
            trial_median_impute=trial_median_impute,
            perm_repeats=2,
        )
        bundle = snap["bundle"]
        pred_test = snap["pred_test"]
        y_test = bundle.y_test
        global_actual = float(np.mean(y_test))
        global_pred = float(np.mean(pred_test))
        x_eval = bundle.x_test.copy()

        for _, r in rules_df.iterrows():
            mask = apply_rule_conditions(x_eval, str(r["rule_conditions"]))
            support = int(mask.sum())
            if support == 0:
                all_rows.append(
                    {
                        "rule_id": str(r["rule_id"]),
                        "seed": int(seed),
                        "support_count": 0,
                        "support_pct": 0.0,
                        "actual_lift_vs_global": np.nan,
                        "predicted_lift_vs_global": np.nan,
                    }
                )
                continue
            pred_mean = float(np.mean(pred_test[mask]))
            actual_mean = float(np.mean(y_test[mask]))
            all_rows.append(
                {
                    "rule_id": str(r["rule_id"]),
                    "seed": int(seed),
                    "support_count": support,
                    "support_pct": float(support / len(x_eval)),
                    "actual_lift_vs_global": float(actual_mean - global_actual),
                    "predicted_lift_vs_global": float(pred_mean - global_pred),
                }
            )
    return pd.DataFrame(all_rows)


def evaluate_rules_by_country_over_seeds(
    rules_df: pd.DataFrame,
    frame: pd.DataFrame,
    features: list[str],
    params: dict[str, Any],
    *,
    seeds: list[int],
    test_size: float,
    trial_median_impute: bool,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        snap = seed_snapshot(
            frame=frame,
            features=features,
            role_map_df=pd.DataFrame({"column_name": features, "final_role": "unknown"}),
            params=params,
            seed=int(seed),
            test_size=test_size,
            trial_median_impute=trial_median_impute,
            perm_repeats=2,
        )
        bundle = snap["bundle"]
        x_eval = bundle.x_test.copy()
        y_eval = bundle.y_test.reset_index(drop=True)
        pred_eval = snap["pred_test"].reset_index(drop=True)
        country_eval = bundle.countries_test.astype(str).fillna("unknown").reset_index(drop=True)

        global_actual_country = pd.DataFrame({"country": country_eval, "yield": y_eval}).groupby(
            "country", as_index=False
        )["yield"].mean()
        global_pred_country = pd.DataFrame({"country": country_eval, "pred": pred_eval}).groupby(
            "country", as_index=False
        )["pred"].mean()
        actual_map = dict(zip(global_actual_country["country"], global_actual_country["yield"], strict=False))
        pred_map = dict(zip(global_pred_country["country"], global_pred_country["pred"], strict=False))

        for _, r in rules_df.iterrows():
            rid = str(r["rule_id"])
            mask = apply_rule_conditions(x_eval, str(r["rule_conditions"])).reset_index(drop=True)
            tmp = pd.DataFrame(
                {
                    "country": country_eval,
                    "yield": y_eval,
                    "pred": pred_eval,
                    "match": mask,
                }
            )
            for country, grp in tmp.groupby("country", dropna=False):
                support_count = int(grp["match"].sum())
                n_country = int(len(grp))
                support_pct = float(support_count / n_country) if n_country > 0 else 0.0
                if support_count == 0:
                    act_lift = np.nan
                    pred_lift = np.nan
                else:
                    act_lift = float(grp.loc[grp["match"], "yield"].mean() - actual_map[str(country)])
                    pred_lift = float(grp.loc[grp["match"], "pred"].mean() - pred_map[str(country)])
                rows.append(
                    {
                        "rule_id": rid,
                        "seed": int(seed),
                        "country": str(country),
                        "n_country_test": n_country,
                        "support_count": support_count,
                        "support_pct": support_pct,
                        "actual_lift_vs_country_baseline": act_lift,
                        "predicted_lift_vs_country_baseline": pred_lift,
                    }
                )
    return pd.DataFrame(rows)


def summarize_rule_country_generalization(
    rule_country_seed_df: pd.DataFrame,
    final_rules_df: pd.DataFrame,
    *,
    stable_sign_threshold: float = 0.8,
    support_seed_rate_threshold: float = 0.6,
    min_abs_lift: float = 100.0,
) -> pd.DataFrame:
    if rule_country_seed_df.empty:
        return pd.DataFrame()
    allowed_rule_ids = set(final_rules_df["rule_id"].astype(str).tolist())

    grouped = (
        rule_country_seed_df.groupby(["rule_id", "country"], as_index=False)
        .agg(
            n_seeds=("seed", "nunique"),
            support_mean=("support_pct", "mean"),
            support_std=("support_pct", "std"),
            support_nonzero_seeds=("support_count", lambda s: int((pd.to_numeric(s, errors="coerce") > 0).sum())),
            lift_mean=("actual_lift_vs_country_baseline", "mean"),
            lift_std=("actual_lift_vs_country_baseline", "std"),
            pos_lift_seeds=(
                "actual_lift_vs_country_baseline",
                lambda s: int((pd.to_numeric(s, errors="coerce") > 0).sum()),
            ),
            neg_lift_seeds=(
                "actual_lift_vs_country_baseline",
                lambda s: int((pd.to_numeric(s, errors="coerce") < 0).sum()),
            ),
        )
        .reset_index(drop=True)
    )
    grouped["rule_id"] = grouped["rule_id"].astype(str)
    grouped = grouped[grouped["rule_id"].isin(allowed_rule_ids)].copy()
    if grouped.empty:
        return grouped

    grouped["support_seed_rate"] = grouped["support_nonzero_seeds"] / grouped["n_seeds"].replace(0, np.nan)
    grouped["support_seed_rate"] = grouped["support_seed_rate"].fillna(0.0)
    denom = (grouped["pos_lift_seeds"] + grouped["neg_lift_seeds"]).replace(0, np.nan)
    grouped["same_sign_rate"] = np.maximum(grouped["pos_lift_seeds"], grouped["neg_lift_seeds"]) / denom
    grouped["same_sign_rate"] = grouped["same_sign_rate"].fillna(0.0)
    grouped["stable_sign"] = grouped["same_sign_rate"] >= float(stable_sign_threshold)

    sign_source_col = "lift_score" if "lift_score" in final_rules_df.columns else "actual_lift_vs_global"
    expected_sign = (
        final_rules_df.set_index("rule_id")[sign_source_col]
        .astype(float)
        .apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
        .to_dict()
    )
    expected_sign_series = grouped["rule_id"].map(expected_sign).fillna(0).astype(int)
    observed_sign_series = grouped["lift_mean"].apply(
        lambda x: 1 if pd.notna(x) and float(x) > 0 else (-1 if pd.notna(x) and float(x) < 0 else 0)
    )
    grouped["expected_sign"] = expected_sign_series
    grouped["observed_sign"] = observed_sign_series

    enough_support = grouped["support_seed_rate"] >= float(support_seed_rate_threshold)
    meaningful_lift = grouped["lift_mean"].abs() >= float(min_abs_lift)
    sign_aligned = grouped["observed_sign"] == grouped["expected_sign"]

    grouped["generalization_status"] = np.select(
        [
            enough_support & grouped["stable_sign"] & meaningful_lift & sign_aligned,
            enough_support & grouped["stable_sign"] & (~sign_aligned),
            ~enough_support,
        ],
        [
            "works_here",
            "conflicts_here",
            "insufficient_evidence",
        ],
        default="unstable_or_small_effect",
    )
    return grouped.sort_values(["rule_id", "generalization_status", "country"]).reset_index(drop=True)


def summarize_rule_country_counts(rule_country_summary_df: pd.DataFrame) -> pd.DataFrame:
    if rule_country_summary_df.empty:
        return pd.DataFrame()
    return (
        rule_country_summary_df.groupby("rule_id", as_index=False)
        .agg(
            countries_seen=("country", "nunique"),
            countries_works=("generalization_status", lambda s: int((pd.Series(s) == "works_here").sum())),
            countries_conflicts=("generalization_status", lambda s: int((pd.Series(s) == "conflicts_here").sum())),
            countries_insufficient=(
                "generalization_status",
                lambda s: int((pd.Series(s) == "insufficient_evidence").sum()),
            ),
            countries_unstable_or_small_effect=(
                "generalization_status",
                lambda s: int((pd.Series(s) == "unstable_or_small_effect").sum()),
            ),
        )
        .sort_values("rule_id")
        .reset_index(drop=True)
    )


def summarize_rule_stability(rule_seed_df: pd.DataFrame, base_rules_df: pd.DataFrame) -> pd.DataFrame:
    primary_sign = (
        base_rules_df.set_index("rule_id")["actual_lift_vs_global"]
        .apply(lambda x: float(np.sign(float(x))))
        .to_dict()
    )

    rows: list[dict[str, Any]] = []
    for rid, grp in rule_seed_df.groupby("rule_id", dropna=False):
        lifts = pd.to_numeric(grp["actual_lift_vs_global"], errors="coerce")
        supports = pd.to_numeric(grp["support_pct"], errors="coerce")
        sign_ref = primary_sign.get(str(rid), 0.0)
        valid_sign = np.sign(lifts.replace(0.0, np.nan)).dropna()
        if sign_ref == 0.0:
            sign_match = float(np.mean(valid_sign == np.sign(valid_sign))) if len(valid_sign) else 0.0
        else:
            sign_match = float(np.mean(valid_sign == sign_ref)) if len(valid_sign) else 0.0
        rows.append(
            {
                "rule_id": str(rid),
                "support_mean": float(np.nanmean(supports)) if len(supports) else 0.0,
                "support_std": float(np.nanstd(supports)) if len(supports) else 0.0,
                "lift_mean": float(np.nanmean(lifts)) if len(lifts) else 0.0,
                "lift_std": float(np.nanstd(lifts)) if len(lifts) else 0.0,
                "sign_match_rate": sign_match,
                "n_seed_evals": int(len(grp)),
            }
        )
    return pd.DataFrame(rows)


def _selection_view(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "support_mean" in out.columns:
        out["support_score"] = pd.to_numeric(out["support_mean"], errors="coerce")
    else:
        out["support_score"] = pd.to_numeric(out.get("support_pct", 0.0), errors="coerce")

    if "lift_mean" in out.columns:
        out["lift_score"] = pd.to_numeric(out["lift_mean"], errors="coerce")
    else:
        out["lift_score"] = pd.to_numeric(out.get("actual_lift_vs_global", 0.0), errors="coerce")

    if "sign_match_rate" not in out.columns:
        out["sign_match_rate"] = 1.0
    out["sign_match_rate"] = pd.to_numeric(out["sign_match_rate"], errors="coerce").fillna(0.0)
    out["rule_score"] = (
        out["sign_match_rate"] * 100.0 + out["support_score"] * 120.0 + out["lift_score"].abs() / 8.0
    )
    return out


def select_rules_with_backoff(
    rules_df: pd.DataFrame,
    *,
    min_rules: int,
    support_thresholds: list[float],
    abs_lift_thresholds: list[float],
    sign_thresholds: list[float],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    view = _selection_view(rules_df)
    best_meta: dict[str, Any] = {"mode": "fallback"}

    for support_min in support_thresholds:
        for lift_min in abs_lift_thresholds:
            for sign_min in sign_thresholds:
                cand = view[
                    (view["support_score"] >= support_min)
                    & (view["lift_score"].abs() >= lift_min)
                    & (view["sign_match_rate"] >= sign_min)
                ].copy()
                cand = cand.sort_values("rule_score", ascending=False).reset_index(drop=True)
                if len(cand) >= int(min_rules):
                    out = cand.head(int(min_rules)).copy()
                    best_meta = {
                        "mode": "threshold_pass",
                        "support_min": support_min,
                        "abs_lift_min": lift_min,
                        "sign_min": sign_min,
                    }
                    return out, best_meta

    out = view.sort_values("rule_score", ascending=False).head(int(min_rules)).copy()
    best_meta = {
        "mode": "fallback_top_score",
        "support_min": float(min(support_thresholds)) if support_thresholds else 0.0,
        "abs_lift_min": float(min(abs_lift_thresholds)) if abs_lift_thresholds else 0.0,
        "sign_min": float(min(sign_thresholds)) if sign_thresholds else 0.0,
    }
    return out, best_meta


def add_confidence_labels(rules_df: pd.DataFrame) -> pd.DataFrame:
    out = rules_df.copy()
    confidence: list[str] = []
    caveats: list[str] = []
    for _, row in out.iterrows():
        support = float(row.get("support_score", row.get("support_pct", 0.0)))
        lift = float(row.get("lift_score", row.get("actual_lift_vs_global", 0.0)))
        sign = float(row.get("sign_match_rate", 1.0))

        if sign >= 0.9 and support >= 0.10 and abs(lift) >= 100:
            conf = "High"
        elif sign >= 0.67 and support >= 0.05 and abs(lift) >= 50:
            conf = "Medium"
        else:
            conf = "Low"
        confidence.append(conf)

        labels = ["associative_only", "requires_causal_validation"]
        if sign < 0.8:
            labels.append("stability_risk")
        if support < 0.08:
            labels.append("low_support")
        if abs(lift) < 60:
            labels.append("small_effect")
        caveats.append(",".join(labels))

    out["confidence_level"] = confidence
    out["caveat_labels"] = caveats
    return out


def write_action_playbook(
    out_path: Path,
    *,
    final_rules_df: pd.DataFrame,
    feature_stability_df: pd.DataFrame,
    rule_country_summary_df: pd.DataFrame,
    role_map_df: pd.DataFrame,
    model_features: list[str],
) -> None:
    role_map = role_map_df.copy()
    role_map["column_name"] = role_map["column_name"].astype(str)
    feature_set = set(model_features)

    top_modifiable = (
        feature_stability_df[
            (feature_stability_df["final_role"] == "modifiable")
            & (feature_stability_df["feature"].astype(str).isin(feature_set))
        ]
        .sort_values(["joint_top10_rate", "shap_rank_mean"], ascending=[False, True])
        .head(8)
    )
    top_modifiable_strong = top_modifiable[top_modifiable["joint_top10_rate"] >= 0.6].copy()
    top_modifiable_exploratory = top_modifiable[top_modifiable["joint_top10_rate"] < 0.6].copy()
    context_feats = (
        role_map[(role_map["final_role"] == "context") & (role_map["column_name"].isin(feature_set))]
        .sort_values("column_name")["column_name"]
        .astype(str)
        .tolist()
    )
    proxy_feats = (
        role_map[(role_map["final_role"] == "proxy") | (role_map["leakage_risk"].astype(str) == "post_outcome")]
        .sort_values("column_name")["column_name"]
        .astype(str)
        .tolist()
    )

    lines: list[str] = [
        "# Actionable Recommendation Playbook (CAROB)",
        "",
        "Built from iteration-3 explainability outputs (SHAP + permutation + cross-seed rule stability).",
        "",
        "## 1) What to change (modifiable levers)",
    ]
    if top_modifiable_strong.empty:
        lines.append("- No modifiable lever passed the cross-seed consensus threshold.")
    else:
        for _, r in top_modifiable_strong.iterrows():
            joint = float(r["joint_top10_rate"]) * 100.0
            shap_rank = float(r["shap_rank_mean"])
            perm_rank = float(r["perm_rank_mean"])
            lines.append(
                f"- `{r['feature']}`: joint stability={joint:.1f}%, "
                f"mean SHAP rank={shap_rank:.1f}, mean permutation rank={perm_rank:.1f}."
            )
    if not top_modifiable_exploratory.empty:
        lines += [
            "",
            "Exploratory (low-consensus) modifiable features to validate before acting:",
        ]
        for _, r in top_modifiable_exploratory.iterrows():
            lines.append(
                f"- `{r['feature']}`: joint stability={float(r['joint_top10_rate']) * 100.0:.1f}% "
                "(keep as hypothesis, not as primary action)."
            )

    lines += ["", "Rule-backed segments (plain language):"]
    for _, r in final_rules_df.iterrows():
        lift = float(r.get("lift_score", r.get("actual_lift_vs_global", 0.0)))
        support = float(r.get("support_score", r.get("support_pct", 0.0))) * 100.0
        rid = str(r["rule_id"])
        by_country = rule_country_summary_df[rule_country_summary_df["rule_id"].astype(str) == rid].copy()
        works = by_country.loc[by_country["generalization_status"] == "works_here", "country"].astype(str).tolist()
        conflicts = (
            by_country.loc[by_country["generalization_status"] == "conflicts_here", "country"].astype(str).tolist()
        )
        insufficient = (
            by_country.loc[by_country["generalization_status"] == "insufficient_evidence", "country"]
            .astype(str)
            .tolist()
        )
        unstable = (
            by_country.loc[by_country["generalization_status"] == "unstable_or_small_effect", "country"]
            .astype(str)
            .tolist()
        )

        works_txt = ", ".join(works) if works else "none"
        conflicts_txt = ", ".join(conflicts) if conflicts else "none"
        insufficient_txt = ", ".join(insufficient) if insufficient else "none"
        unstable_txt = ", ".join(unstable) if unstable else "none"
        lines.append(
            f"- `{r['rule_id']}`: expected lift={lift:+.1f} kg/ha, "
            f"support={support:.1f}%, "
            f"confidence={r['confidence_level']} (caveats: {r['caveat_labels']})."
        )
        lines.append(
            f"  Where it works: {works_txt} | Conflicts: {conflicts_txt} | "
            f"Insufficient evidence: {insufficient_txt} | Unstable/small effect: {unstable_txt}."
        )

    lines += ["", "## 2) What to control for (context features)"]
    if context_feats:
        for feat in context_feats:
            lines.append(f"- `{feat}`")
    else:
        lines.append("- No context features included in current feature scenario.")

    lines += ["", "## 3) What to avoid (proxy/leakage features)"]
    if proxy_feats:
        for feat in proxy_feats[:14]:
            lines.append(f"- `{feat}`")
        if len(proxy_feats) > 14:
            lines.append(f"- ... and {len(proxy_feats) - 14} additional proxy/leakage flagged variables.")
    else:
        lines.append("- No proxy features listed in role map.")

    lines += [
        "",
        "## 4) Confidence and caveats",
        "- Outputs are predictive associations, not causal effects.",
        "- Prioritize high-confidence rules first and validate with trial-level or quasi-experimental analysis.",
        "- Keep treatment assignment logic explicit in the causal step to avoid post-treatment adjustment bias.",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def write_causal_handoff(
    out_path: Path,
    *,
    final_rules_df: pd.DataFrame,
    feature_stability_df: pd.DataFrame,
) -> None:
    top_modifiable = (
        feature_stability_df[feature_stability_df["final_role"] == "modifiable"]
        .sort_values(["joint_top10_rate", "shap_rank_mean"], ascending=[False, True])
        .head(5)["feature"]
        .astype(str)
        .tolist()
    )
    lines = [
        "# Causal Handoff Notes",
        "",
        "This package is causal-ready but not causal by itself.",
        "",
        "## Recommended transition to causal analysis",
        "1. Use the final rule set as hypothesis generators for treatment contrasts.",
        "2. Define treatment and control cohorts before modeling outcomes.",
        "3. Avoid conditioning on post-treatment variables and known leakage proxies.",
        "4. Use trial-aware or country-aware adjustment sets based on DAG assumptions.",
        "",
        "## Candidate modifiable levers to prioritize",
    ]
    for feat in top_modifiable:
        lines.append(f"- `{feat}`")
    if not top_modifiable:
        lines.append("- No stable modifiable features found.")
    lines += ["", "## Final rule IDs for causal follow-up"]
    for rid in final_rules_df["rule_id"].astype(str).tolist():
        lines.append(f"- `{rid}`")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CAROB interpretability (3-iteration defensibility pipeline).")
    parser.add_argument("--run-tag", type=str, default="latest")
    parser.add_argument("--scenario", type=str, default=DEFAULT_SCENARIO)
    parser.add_argument("--candidates-path", type=str, default=str(DEFAULT_CANDIDATES_PATH))
    parser.add_argument("--role-map-path", type=str, default=str(DEFAULT_ROLE_MAP_PATH))
    parser.add_argument("--winners-path", type=str, default=str(DEFAULT_WINNERS_PATH))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stability-seeds", type=str, default="42,52,62")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--trial-median-impute", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--perm-repeats", type=int, default=8)
    parser.add_argument("--global-top-n", type=int, default=15)
    parser.add_argument("--local-top-n", type=int, default=10)
    parser.add_argument("--ale-top-n", type=int, default=3)
    parser.add_argument("--n-rules", type=int, default=8)
    parser.add_argument("--min-final-rules", type=int, default=5)
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = OUT_ROOT / args.run_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    candidates_path = Path(args.candidates_path)
    role_map_path = Path(args.role_map_path)
    winners_path = Path(args.winners_path)

    frame, features = load_frame_and_features(candidates_path, args.scenario)
    role_map_df = load_role_map(role_map_path, frame_cols=frame.columns.tolist())
    params = load_best_catboost_params(winners_path)
    stability_seeds = parse_seed_list(args.stability_seeds)
    if int(args.seed) not in stability_seeds:
        stability_seeds = list(dict.fromkeys([int(args.seed)] + stability_seeds))

    feature_set_df = pd.DataFrame(
        [{"scenario": args.scenario, "rank_in_scenario": i, "feature": f} for i, f in enumerate(features, start=1)]
    )
    feature_set_df.to_csv(out_dir / "scenario_feature_set.csv", index=False)

    # Iteration 1: baseline explainability and initial rules.
    it1 = seed_snapshot(
        frame=frame,
        features=features,
        role_map_df=role_map_df,
        params=params,
        seed=int(args.seed),
        test_size=float(args.test_size),
        trial_median_impute=bool(args.trial_median_impute),
        perm_repeats=int(args.perm_repeats),
    )

    bundle = it1["bundle"]
    global_shap_df = it1["global_shap"]
    perm_df = it1["perm"]
    cross_df = it1["cross"]
    pred_train = it1["pred_train"]
    pred_test = it1["pred_test"]
    shap_df = it1["shap_df"]
    model = it1["model"]
    metrics = it1["metrics"]

    global_shap_df.to_csv(out_dir / "iteration1_global_shap_importance.csv", index=False)
    perm_df.to_csv(out_dir / "iteration1_permutation_importance.csv", index=False)
    cross_df.to_csv(out_dir / "iteration1_feature_crosscheck.csv", index=False)
    pd.DataFrame([metrics]).to_csv(out_dir / "iteration1_model_metrics.csv", index=False)

    plot_top_bar(
        global_shap_df,
        feature_col="feature",
        value_col="mean_abs_shap",
        role_col="final_role",
        top_n=int(args.global_top_n),
        out_path=out_dir / "iteration1_global_shap_top.png",
        title=f"Iteration 1 Global SHAP (Top {int(args.global_top_n)})",
        x_label="Mean |SHAP| on holdout",
    )
    perm_plot_df = pd.merge(
        perm_df,
        role_map_df[["column_name", "final_role"]],
        left_on="feature",
        right_on="column_name",
        how="left",
    ).drop(columns=["column_name"])
    plot_top_bar(
        perm_plot_df,
        feature_col="feature",
        value_col="delta_rmse_mean",
        role_col="final_role",
        top_n=int(args.global_top_n),
        out_path=out_dir / "iteration1_permutation_top.png",
        title=f"Iteration 1 Permutation Importance (Top {int(args.global_top_n)})",
        x_label="RMSE increase when feature is permuted",
    )

    case_map = select_local_cases(bundle.y_test, pred_test)
    local_overview_df, _local_detail_df = save_local_outputs(
        x_test=bundle.x_test,
        y_test=bundle.y_test,
        groups_test=bundle.groups_test,
        pred_test=pred_test,
        shap_df=shap_df,
        case_map=case_map,
        top_n_features=int(args.local_top_n),
        out_dir=out_dir,
    )

    modifiable_features = role_map_df.loc[
        role_map_df["final_role"].astype(str) == "modifiable", "column_name"
    ].astype(str)
    modifiable_numeric = [
        f
        for f in features
        if f in set(modifiable_features) and pd.api.types.is_numeric_dtype(bundle.x_train[f]) and f in bundle.x_train
    ]
    rules_raw_df = extract_modifiable_rules(
        x_train=bundle.x_train,
        x_test=bundle.x_test,
        y_test=bundle.y_test,
        pred_train=pred_train,
        pred_test=pred_test,
        modifiable_numeric_features=modifiable_numeric,
        n_rules=int(args.n_rules),
        min_support_pct=0.03,
        random_state=int(args.seed),
    )
    rules_raw_df.to_csv(out_dir / "iteration1_rules_raw.csv", index=False)
    rules_to_english_markdown(
        rules_raw_df,
        role_map_df=role_map_df,
        bool_like_cols=bundle.bool_like_cols,
        out_path=out_dir / "iteration1_rules_english.md",
        title="Iteration 1 Rule Explanations (Plain English)",
    )

    actionable_numeric_for_curves = (
        cross_df[
            (cross_df["final_role"] == "modifiable")
            & (cross_df["feature"].isin(modifiable_numeric))
            & (cross_df["consensus_top10"])
        ]["feature"]
        .astype(str)
        .head(int(args.ale_top_n))
        .tolist()
    )
    if not actionable_numeric_for_curves:
        actionable_numeric_for_curves = modifiable_numeric[: int(args.ale_top_n)]

    curve_rows: list[dict[str, Any]] = []
    for feature in actionable_numeric_for_curves:
        ale_df = compute_ale_curve(model, bundle.x_test, bundle.cat_cols, feature, n_bins=10)
        pdp_df = compute_pdp_curve(model, bundle.x_test, bundle.cat_cols, feature, n_points=12)
        ale_df.to_csv(out_dir / f"iteration1_ale_{feature}.csv", index=False)
        pdp_df.to_csv(out_dir / f"iteration1_pdp_{feature}.csv", index=False)
        plot_ale_pdp(
            ale_df=ale_df,
            pdp_df=pdp_df,
            feature=feature,
            out_path=out_dir / f"iteration1_ale_pdp_{feature}.png",
        )
        curve_rows.append(
            {
                "feature": feature,
                "ale_points": int(len(ale_df)),
                "pdp_points": int(len(pdp_df)),
            }
        )
    pd.DataFrame(curve_rows).to_csv(out_dir / "iteration1_curve_inventory.csv", index=False)

    # Iteration 2: cross-seed stability diagnostics and rule refinement.
    seed_rows: list[dict[str, Any]] = []
    for seed in stability_seeds:
        snap = seed_snapshot(
            frame=frame,
            features=features,
            role_map_df=role_map_df,
            params=params,
            seed=int(seed),
            test_size=float(args.test_size),
            trial_median_impute=bool(args.trial_median_impute),
            perm_repeats=max(4, int(args.perm_repeats) // 2),
        )
        seed_rows.append({"seed": int(seed), **snap})

    feature_stability_df = aggregate_feature_stability(seed_rows, role_map_df=role_map_df)
    feature_stability_df.to_csv(out_dir / "iteration2_feature_stability.csv", index=False)

    rule_seed_df = evaluate_rules_over_seeds(
        rules_df=rules_raw_df,
        frame=frame,
        features=features,
        params=params,
        seeds=stability_seeds,
        test_size=float(args.test_size),
        trial_median_impute=bool(args.trial_median_impute),
    )
    rule_seed_df.to_csv(out_dir / "iteration2_rule_seed_evaluations.csv", index=False)
    rule_stability_df = summarize_rule_stability(rule_seed_df, rules_raw_df)
    rule_stability_df.to_csv(out_dir / "iteration2_rule_stability.csv", index=False)

    rule_country_seed_df = evaluate_rules_by_country_over_seeds(
        rules_df=rules_raw_df,
        frame=frame,
        features=features,
        params=params,
        seeds=stability_seeds,
        test_size=float(args.test_size),
        trial_median_impute=bool(args.trial_median_impute),
    )
    rule_country_seed_df.to_csv(out_dir / "iteration2_rule_country_seed_evaluations.csv", index=False)
    rule_country_summary_df = summarize_rule_country_generalization(
        rule_country_seed_df,
        rules_raw_df,
    )
    rule_country_summary_df.to_csv(out_dir / "iteration2_rule_country_generalization.csv", index=False)
    summarize_rule_country_counts(rule_country_summary_df).to_csv(
        out_dir / "iteration2_rule_country_summary.csv",
        index=False,
    )

    rules_iter2_df = pd.merge(rules_raw_df, rule_stability_df, on="rule_id", how="left")
    selected_iter2_df, iter2_meta = select_rules_with_backoff(
        rules_iter2_df,
        min_rules=max(int(args.min_final_rules), 5),
        support_thresholds=[0.05, 0.04, 0.03],
        abs_lift_thresholds=[100.0, 75.0, 50.0, 25.0],
        sign_thresholds=[0.8, 0.67, 0.5],
    )
    selected_iter2_df.to_csv(out_dir / "iteration2_rules_refined.csv", index=False)

    # Iteration 3: final defensible rules + confidence labels + playbook.
    selected_iter3_df, iter3_meta = select_rules_with_backoff(
        selected_iter2_df,
        min_rules=max(int(args.min_final_rules), 5),
        support_thresholds=[0.08, 0.06, 0.05],
        abs_lift_thresholds=[120.0, 100.0, 75.0, 50.0],
        sign_thresholds=[0.9, 0.8, 0.67],
    )
    selected_iter3_df = add_confidence_labels(selected_iter3_df)
    selected_iter3_df.to_csv(out_dir / "iteration3_rules_final.csv", index=False)

    # Recompute country generalization against final selected rules.
    rule_country_summary_final_df = summarize_rule_country_generalization(
        rule_country_seed_df,
        selected_iter3_df,
    )
    rule_country_summary_final_df.to_csv(out_dir / "iteration3_rule_country_generalization.csv", index=False)
    summarize_rule_country_counts(rule_country_summary_final_df).to_csv(
        out_dir / "iteration3_rule_country_summary.csv",
        index=False,
    )

    rules_to_english_markdown(
        selected_iter3_df,
        role_map_df=role_map_df,
        bool_like_cols=bundle.bool_like_cols,
        out_path=out_dir / "iteration3_rules_final_english.md",
        title="Iteration 3 Final Rule Explanations (Plain English)",
    )

    write_action_playbook(
        out_path=out_dir / "iteration3_action_playbook.md",
        final_rules_df=selected_iter3_df,
        feature_stability_df=feature_stability_df,
        rule_country_summary_df=rule_country_summary_final_df,
        role_map_df=role_map_df,
        model_features=features,
    )
    write_causal_handoff(
        out_path=out_dir / "iteration3_causal_handoff.md",
        final_rules_df=selected_iter3_df,
        feature_stability_df=feature_stability_df,
    )

    # UI-friendly exports.
    ui_top_features = feature_stability_df.sort_values(
        ["joint_top10_rate", "shap_rank_mean"], ascending=[False, True]
    ).reset_index(drop=True)
    ui_top_features.to_csv(out_dir / "ui_top_features.csv", index=False)
    local_overview_df.to_csv(out_dir / "ui_local_examples.csv", index=False)
    selected_iter3_df.to_csv(out_dir / "ui_rules.csv", index=False)
    selected_iter3_df[["rule_id", "confidence_level", "caveat_labels"]].to_csv(
        out_dir / "ui_confidence_labels.csv", index=False
    )

    top10_shap = set(global_shap_df.head(10)["feature"].astype(str).tolist())
    top10_perm = set(perm_df.head(10)["feature"].astype(str).tolist())
    overlap_top10 = len(top10_shap & top10_perm) / max(1, len(top10_shap))
    avg_rule_sign_match = float(pd.to_numeric(selected_iter3_df["sign_match_rate"], errors="coerce").mean())
    defensible = bool(
        (len(selected_iter3_df) >= max(int(args.min_final_rules), 5))
        and (overlap_top10 >= 0.4)
        and (avg_rule_sign_match >= 0.67)
    )

    summary_df = pd.DataFrame(
        [
            {
                "iteration": "iteration1",
                "r2_test": metrics["r2_test"],
                "rmse_test": metrics["rmse_test"],
                "mae_test": metrics["mae_test"],
                "rules_count": int(len(rules_raw_df)),
                "top10_shap_perm_overlap_ratio": overlap_top10,
                "notes": "Baseline explainability package generated.",
            },
            {
                "iteration": "iteration2",
                "r2_test": np.nan,
                "rmse_test": np.nan,
                "mae_test": np.nan,
                "rules_count": int(len(selected_iter2_df)),
                "top10_shap_perm_overlap_ratio": overlap_top10,
                "notes": json.dumps(iter2_meta, sort_keys=True),
            },
            {
                "iteration": "iteration3",
                "r2_test": np.nan,
                "rmse_test": np.nan,
                "mae_test": np.nan,
                "rules_count": int(len(selected_iter3_df)),
                "top10_shap_perm_overlap_ratio": overlap_top10,
                "notes": json.dumps(iter3_meta, sort_keys=True),
            },
        ]
    )
    summary_df.to_csv(out_dir / "iteration_summary.csv", index=False)

    runlog_lines = [
        f"run_tag={args.run_tag}",
        f"scenario={args.scenario}",
        f"seed={int(args.seed)}",
        f"stability_seeds={stability_seeds}",
        f"rows_total={len(frame)}",
        f"rows_train={len(bundle.x_train)}",
        f"rows_test={len(bundle.x_test)}",
        f"features_used={len(features)}",
        f"modifiable_numeric_rule_features={len(modifiable_numeric)}",
        f"iteration1_r2_test={metrics['r2_test']:.6f}",
        f"iteration1_rmse_test={metrics['rmse_test']:.6f}",
        f"iteration1_mae_test={metrics['mae_test']:.6f}",
        f"top10_shap_perm_overlap_ratio={overlap_top10:.4f}",
        f"iteration3_rules_count={len(selected_iter3_df)}",
        f"iteration3_avg_rule_sign_match={avg_rule_sign_match:.4f}",
        (
            "iteration3_countries_with_working_rules="
            f"{int(rule_country_summary_final_df['generalization_status'].eq('works_here').sum())}"
        ),
        f"defensible_case={defensible}",
    ]
    (out_dir / "interpretability_runlog.txt").write_text("\n".join(runlog_lines), encoding="utf-8")

    print(f"Saved CAROB interpretability outputs to: {out_dir}")
    print(f"Iteration 1 test metrics: R2={metrics['r2_test']:.4f} | RMSE={metrics['rmse_test']:.2f}")
    print(f"Final rule count: {len(selected_iter3_df)} | defensible_case={defensible}")

    return {
        "out_dir": out_dir,
        "summary_df": summary_df,
        "rules_final_df": selected_iter3_df,
        "feature_stability_df": feature_stability_df,
        "defensible_case": defensible,
    }


def main() -> None:
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
