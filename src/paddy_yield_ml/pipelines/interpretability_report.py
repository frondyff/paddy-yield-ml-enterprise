"""
Interpretability package for the current conservative CatBoost model.

Run:
  python src/paddy_yield_ml/pipelines/interpretability_report.py --run-tag latest

Outputs (under ./outputs/interpretability/<run-tag>/):
  - global_shap_importance.csv
  - global_shap_top_features.png
  - local_shap_examples.csv
  - local_examples_overview.csv
  - local_shap_case_<case_name>.png
  - modifiable_decision_rules.csv
  - modifiable_decision_rules.md
  - actionable_recommendations_draft.md
  - professor_milestone_one_page.md
  - interpretability_runlog.txt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.tree import DecisionTreeRegressor

from paddy_yield_ml.pipelines import model_compare as mc

try:
    project_root = Path(__file__).resolve().parents[3]
except NameError:
    project_root = Path.cwd()

OUT_ROOT = project_root / "outputs" / "interpretability"
DEFAULT_FEATURE_SET_PATH = project_root / "outputs" / "model_select_tune" / "dual_eval" / "scenario_feature_sets.csv"
DEFAULT_PARAMS_PATH = project_root / "outputs" / "model_select_tune" / "dual_eval" / "logo_summary.csv"
DEFAULT_ROLE_MAP_PATH = project_root / "outputs" / "feature_prepare" / "actionability_role_map.csv"
DEFAULT_SECONDARY_PATH = project_root / "outputs" / "model_select_tune" / "dual_eval" / "secondary_split_summary.csv"
DEFAULT_LOGO_PATH = project_root / "outputs" / "model_select_tune" / "dual_eval" / "logo_summary.csv"
TREE_LEAF = -1


def load_feature_set(path: Path, scenario_name: str) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Missing feature set file: {path}")
    df = pd.read_csv(path)
    required = {"scenario", "feature"}
    if not required.issubset(df.columns):
        raise ValueError(f"Feature set file missing columns: {sorted(required)}")
    out = df.loc[df["scenario"].astype(str) == scenario_name, "feature"].astype(str).dropna().tolist()
    if not out:
        available = sorted(df["scenario"].astype(str).unique().tolist())
        raise ValueError(f"Scenario '{scenario_name}' not found. Available: {available}")
    return mc.dedupe_keep_order(out)


def load_best_params(path: Path, feature_set_name: str, model_name: str) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing params source file: {path}")
    df = pd.read_csv(path)
    required = {"feature_set", "model", "params_json"}
    if not required.issubset(df.columns):
        raise ValueError(f"Params file missing columns: {sorted(required)}")

    match = df[
        (df["feature_set"].astype(str) == feature_set_name)
        & (df["model"].astype(str).str.lower() == model_name.lower())
    ]
    if match.empty:
        match = df[df["model"].astype(str).str.lower() == model_name.lower()]
    if match.empty:
        raise ValueError(f"No params found for model '{model_name}' in {path}")

    row = match.sort_values("r2_mean", ascending=False).iloc[0]
    raw = str(row.get("params_json", "{}"))
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Could not parse params_json: {raw}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("params_json did not decode to an object.")
    return parsed


def load_role_map(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing role map file: {path}")
    df = pd.read_csv(path)
    required = {"column_name", "final_role", "feature_group"}
    if not required.issubset(df.columns):
        raise ValueError(f"Role map file missing columns: {sorted(required)}")
    return df


def prepare_training_data(
    frame: pd.DataFrame,
    features: list[str],
) -> tuple[pd.DataFrame, pd.Series, pd.Series, list[str]]:
    required = features + [mc.TARGET_COL, mc.GROUP_COL]
    subset = frame[required].copy()
    subset[mc.TARGET_COL] = pd.to_numeric(subset[mc.TARGET_COL], errors="coerce")
    subset = subset[subset[mc.TARGET_COL].notna() & subset[mc.GROUP_COL].notna()].reset_index(drop=True)

    x = subset[features].copy()
    y = subset[mc.TARGET_COL].copy()
    groups = subset[mc.GROUP_COL].astype(str).copy()

    cat_cols = x.select_dtypes(exclude=[np.number]).columns.tolist()
    for col in cat_cols:
        x[col] = x[col].astype(str).fillna("MISSING")
    return x, y, groups, cat_cols


def build_catboost(params: dict[str, Any], random_state: int) -> CatBoostRegressor:
    base: dict[str, Any] = {
        "iterations": 500,
        "learning_rate": 0.05,
        "depth": 6,
        "l2_leaf_reg": 3,
        "loss_function": "RMSE",
        "random_seed": random_state,
        "verbose": False,
        "allow_writing_files": False,
    }
    base.update(params)
    return CatBoostRegressor(**base)


def compute_shap_dataframe(
    model: CatBoostRegressor,
    x: pd.DataFrame,
    y: pd.Series,
    cat_cols: list[str],
) -> tuple[pd.DataFrame, float]:
    cat_idx = [x.columns.get_loc(col) for col in cat_cols]
    pool = Pool(x, label=y, cat_features=cat_idx)
    shap_raw = model.get_feature_importance(data=pool, type="ShapValues")
    if shap_raw.ndim != 2 or shap_raw.shape[1] != x.shape[1] + 1:
        raise ValueError("Unexpected SHAP shape returned by CatBoost.")
    shap_vals = shap_raw[:, :-1]
    expected_value = float(np.mean(shap_raw[:, -1]))
    shap_df = pd.DataFrame(shap_vals, columns=x.columns, index=x.index)
    return shap_df, expected_value


def save_global_shap(
    shap_df: pd.DataFrame,
    role_map_df: pd.DataFrame,
    out_dir: Path,
    top_n: int,
) -> pd.DataFrame:
    role_lookup = role_map_df.set_index("column_name")["final_role"].to_dict()
    importance = (
        shap_df.abs().mean(axis=0).sort_values(ascending=False).rename_axis("feature").reset_index(name="mean_abs_shap")
    )
    importance["final_role"] = importance["feature"].map(role_lookup).fillna("unknown")
    importance.to_csv(out_dir / "global_shap_importance.csv", index=False)

    top = importance.head(top_n).iloc[::-1]
    role_colors = {"modifiable": "#1b9e77", "context": "#7570b3", "proxy": "#d95f02", "unknown": "#666666"}
    colors = [role_colors.get(str(r), "#666666") for r in top["final_role"]]
    plt.figure(figsize=(10, 6))
    plt.barh(top["feature"], top["mean_abs_shap"], color=colors)
    plt.xlabel("Mean |SHAP value|")
    plt.title(f"Global SHAP Importance (Top {len(top)})")
    plt.tight_layout()
    plt.savefig(out_dir / "global_shap_top_features.png", dpi=180)
    plt.close()
    return importance


def select_local_case_indices(y_true: pd.Series, y_pred: pd.Series) -> dict[str, int]:
    frame = pd.DataFrame(
        {
            "y_true": pd.to_numeric(y_true, errors="coerce"),
            "y_pred": pd.to_numeric(y_pred, errors="coerce"),
        }
    ).dropna()
    frame["residual"] = frame["y_pred"] - frame["y_true"]
    frame["abs_residual"] = frame["residual"].abs()

    used: set[int] = set()
    out: dict[str, int] = {}

    def pick(label: str, ordered_idx: list[int]) -> None:
        for idx in ordered_idx:
            if idx not in used:
                used.add(idx)
                out[label] = idx
                return

    pick("highest_prediction", frame.sort_values("y_pred", ascending=False).index.tolist())
    pick("lowest_prediction", frame.sort_values("y_pred", ascending=True).index.tolist())
    pick("largest_overprediction", frame.sort_values("residual", ascending=False).index.tolist())
    pick("largest_underprediction", frame.sort_values("residual", ascending=True).index.tolist())
    median_pred = float(frame["y_pred"].median())
    pick("median_prediction", (frame["y_pred"] - median_pred).abs().sort_values().index.tolist())

    if len(out) < 5:
        pick("high_error_backup", frame.sort_values("abs_residual", ascending=False).index.tolist())
    return out


def save_local_outputs(
    x: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    y_pred: pd.Series,
    shap_df: pd.DataFrame,
    case_map: dict[str, int],
    out_dir: Path,
    top_n_features: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    overview_rows: list[dict[str, Any]] = []
    detail_rows: list[dict[str, Any]] = []

    for case_name, idx in case_map.items():
        actual = float(y.iloc[idx])
        pred = float(y_pred.iloc[idx])
        residual = pred - actual
        overview_rows.append(
            {
                "case_name": case_name,
                "row_index": int(idx),
                "agriblock": str(groups.iloc[idx]),
                "actual_yield_per_hectare": actual,
                "predicted_yield_per_hectare": pred,
                "residual_pred_minus_actual": residual,
                "abs_residual": abs(residual),
            }
        )

        row_shap = shap_df.iloc[idx]
        row_vals = x.iloc[idx]
        top_feats = row_shap.abs().sort_values(ascending=False).head(top_n_features).index.tolist()

        plot_vals: list[float] = []
        plot_labels: list[str] = []
        colors: list[str] = []
        for rank, feat in enumerate(top_feats, start=1):
            sv = float(row_shap.loc[feat])
            fv = row_vals.loc[feat]
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
            plot_vals.append(sv)
            plot_labels.append(feat)
            colors.append("#1b9e77" if sv >= 0 else "#d95f02")

        order = np.argsort(np.abs(np.asarray(plot_vals)))
        sorted_vals = [plot_vals[i] for i in order]
        sorted_labels = [plot_labels[i] for i in order]
        sorted_colors = [colors[i] for i in order]
        plt.figure(figsize=(10, 6))
        plt.barh(sorted_labels, sorted_vals, color=sorted_colors)
        plt.axvline(0.0, color="#555555", linewidth=1)
        plt.xlabel("SHAP contribution")
        plt.title(f"Local SHAP: {case_name}")
        plt.tight_layout()
        plt.savefig(out_dir / f"local_shap_case_{case_name}.png", dpi=180)
        plt.close()

    overview_df = pd.DataFrame(overview_rows)
    details_df = pd.DataFrame(detail_rows)
    overview_df.to_csv(out_dir / "local_examples_overview.csv", index=False)
    details_df.to_csv(out_dir / "local_shap_examples.csv", index=False)
    return overview_df, details_df


def pick_modifiable_numeric_features(
    x: pd.DataFrame,
    model_features: list[str],
    role_map_df: pd.DataFrame,
) -> list[str]:
    modifiable = set(
        role_map_df.loc[role_map_df["final_role"].astype(str) == "modifiable", "column_name"].astype(str).tolist()
    )
    numeric_cols = set(x.select_dtypes(include=[np.number]).columns.tolist())
    return [f for f in model_features if f in modifiable and f in numeric_cols]


def _leaf_condition_strings(
    tree: DecisionTreeRegressor,
    feature_names: list[str],
) -> dict[int, list[str]]:
    tree_ = tree.tree_
    out: dict[int, list[str]] = {}

    def walk(node_id: int, conditions: list[str]) -> None:
        if tree_.children_left[node_id] == TREE_LEAF:
            out[node_id] = conditions.copy()
            return
        feat_idx = int(tree_.feature[node_id])
        feat_name = feature_names[feat_idx]
        threshold = float(tree_.threshold[node_id])
        left_cond = conditions + [f"{feat_name} <= {threshold:.3f}"]
        right_cond = conditions + [f"{feat_name} > {threshold:.3f}"]
        walk(int(tree_.children_left[node_id]), left_cond)
        walk(int(tree_.children_right[node_id]), right_cond)

    walk(0, [])
    return out


def extract_modifiable_rules(
    x: pd.DataFrame,
    y_actual: pd.Series,
    y_pred: pd.Series,
    modifiable_numeric_features: list[str],
    n_rules: int,
    min_support_pct: float,
    random_state: int,
) -> pd.DataFrame:
    if not modifiable_numeric_features:
        raise ValueError("No numeric modifiable features available for rule extraction.")

    xr = x[modifiable_numeric_features].copy()
    for col in xr.columns:
        xr[col] = pd.to_numeric(xr[col], errors="coerce")
        xr[col] = xr[col].fillna(float(xr[col].median()))

    n_samples = len(xr)
    min_leaf = max(30, int(n_samples * min_support_pct))
    best_tree: DecisionTreeRegressor | None = None
    for depth in range(2, 8):
        tree = DecisionTreeRegressor(max_depth=depth, min_samples_leaf=min_leaf, random_state=random_state)
        tree.fit(xr, y_pred)
        leaf_count = int(np.sum(tree.tree_.children_left == TREE_LEAF))
        best_tree = tree
        if leaf_count >= n_rules:
            break
    if best_tree is None:
        raise RuntimeError("Could not fit surrogate tree for rule extraction.")

    leaf_ids = best_tree.apply(xr)
    unique_leaf_ids = sorted(np.unique(leaf_ids).tolist())
    leaf_rules = _leaf_condition_strings(best_tree, xr.columns.tolist())

    global_actual = float(pd.to_numeric(y_actual, errors="coerce").mean())
    global_pred = float(pd.to_numeric(y_pred, errors="coerce").mean())

    rule_rows: list[dict[str, Any]] = []
    for leaf_id in unique_leaf_ids:
        idx = np.where(leaf_ids == leaf_id)[0]
        support = len(idx)
        if support < min_leaf:
            continue
        conds = leaf_rules.get(int(leaf_id), [])
        features_used = sorted({c.split(" <= ")[0].split(" > ")[0] for c in conds if c})
        pred_mean = float(np.mean(y_pred.iloc[idx]))
        actual_mean = float(np.mean(pd.to_numeric(y_actual.iloc[idx], errors="coerce")))
        rule_rows.append(
            {
                "leaf_id": int(leaf_id),
                "rule_conditions": " AND ".join(conds) if conds else "(all rows)",
                "n_conditions": int(len(conds)),
                "features_used": "|".join(features_used),
                "support_count": int(support),
                "support_pct": float(support / n_samples),
                "predicted_yield_mean": pred_mean,
                "actual_yield_mean": actual_mean,
                "predicted_lift_vs_global": float(pred_mean - global_pred),
                "actual_lift_vs_global": float(actual_mean - global_actual),
                "all_features_modifiable": bool(set(features_used).issubset(set(modifiable_numeric_features))),
            }
        )

    rules_df = pd.DataFrame(rule_rows)
    if rules_df.empty:
        raise RuntimeError("No qualifying surrogate rules found.")

    rules_df = rules_df.sort_values(
        ["predicted_lift_vs_global", "support_count"],
        ascending=[False, False],
    ).reset_index(drop=True)
    rules_df.insert(0, "rule_id", [f"R{i}" for i in range(1, len(rules_df) + 1)])
    return rules_df.head(max(n_rules, 5)).copy()


def write_rules_markdown(rules_df: pd.DataFrame, out_path: Path) -> None:
    lines: list[str] = [
        "# Modifiable-Only Decision Rules",
        "",
        (
            "Rules are extracted from a surrogate tree trained on model predictions "
            "using only numeric modifiable features."
        ),
        "",
    ]
    for _, row in rules_df.iterrows():
        lines.append(f"- **{row['rule_id']}**: `{row['rule_conditions']}`")
        lines.append(
            "  Support: "
            f"{int(row['support_count'])} rows ({float(row['support_pct']) * 100:.1f}%), "
            f"Pred lift: {float(row['predicted_lift_vs_global']):+.2f}, "
            f"Actual lift: {float(row['actual_lift_vs_global']):+.2f}"
        )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def write_actionable_recommendations(
    global_shap_df: pd.DataFrame,
    rules_df: pd.DataFrame,
    role_map_df: pd.DataFrame,
    model_features: list[str],
    out_path: Path,
) -> None:
    model_feature_set = set(model_features)
    top_modifiable = (
        global_shap_df[
            (global_shap_df["final_role"] == "modifiable") & (global_shap_df["feature"].isin(model_feature_set))
        ]
        .head(8)["feature"]
        .astype(str)
        .tolist()
    )
    context_features = (
        role_map_df[
            (role_map_df["final_role"].astype(str) == "context")
            & (role_map_df["column_name"].astype(str).isin(model_feature_set))
        ]["column_name"]
        .astype(str)
        .tolist()
    )
    proxy_features = role_map_df[role_map_df["final_role"].astype(str) == "proxy"]["column_name"].astype(str).tolist()

    lines: list[str] = [
        "# Actionable Recommendations (Draft)",
        "",
        "## 1) What to change (modifiable levers)",
    ]
    for feat in top_modifiable:
        lines.append(f"- Prioritize `{feat}` in agronomic planning and optimization tests.")
    if not top_modifiable:
        lines.append("- No modifiable feature appeared in the model importance ranking.")

    lines += [
        "",
        "Use the following high-yield rule candidates as operational hypotheses (associative, not causal):",
    ]
    for _, row in rules_df.head(5).iterrows():
        lines.append(
            f"- `{row['rule_conditions']}` "
            f"(support {float(row['support_pct']) * 100:.1f}%, pred lift {float(row['predicted_lift_vs_global']):+.2f})"
        )

    lines += [
        "",
        "## 2) What to control for (context features)",
    ]
    for feat in context_features:
        lines.append(f"- Keep `{feat}` as a control/context variable in analysis and validation.")
    if not context_features:
        lines.append("- No context features were present in the selected model.")

    lines += [
        "",
        "## 3) What to avoid (proxy/leakage features)",
        "- Exclude proxy/leakage variables from optimization and reporting metrics.",
        "- Examples from current dictionary and audits:",
    ]
    for feat in proxy_features[:12]:
        lines.append(f"- `{feat}`")
    lines += [
        "",
        "Reason: these variables can encode location or post-outcome information and inflate apparent performance.",
        "All recommendations here are predictive-association based; causal conclusions require additional design.",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def load_metric_row(path: Path, feature_set_name: str, model_name: str) -> pd.Series | None:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if {"feature_set", "model", "r2_mean", "rmse_mean", "mae_mean"}.issubset(df.columns):
        rows = df[
            (df["feature_set"].astype(str) == feature_set_name)
            & (df["model"].astype(str).str.lower() == model_name.lower())
        ]
        if not rows.empty:
            return rows.iloc[0]
    return None


def write_professor_one_pager(
    out_path: Path,
    feature_set_name: str,
    model_name: str,
    logo_row: pd.Series | None,
    group_shuffle_row: pd.Series | None,
    global_shap_df: pd.DataFrame,
    rules_df: pd.DataFrame,
) -> None:
    top_modifiable = (
        global_shap_df[global_shap_df["final_role"] == "modifiable"]["feature"].astype(str).head(5).tolist()
    )
    lines: list[str] = [
        "# Milestone 1-Page Summary",
        "",
        "## Model and Evaluation",
        f"- Final candidate: `{model_name}` on `{feature_set_name}` feature set.",
    ]
    if logo_row is not None:
        lines.append(
            f"- Primary validation (LOGO): R^2={float(logo_row['r2_mean']):.4f}, "
            f"RMSE={float(logo_row['rmse_mean']):.2f}, MAE={float(logo_row['mae_mean']):.2f}."
        )
    if group_shuffle_row is not None:
        lines.append(
            f"- Secondary validation (GroupShuffle): R^2={float(group_shuffle_row['r2_mean']):.4f}, "
            f"RMSE={float(group_shuffle_row['rmse_mean']):.2f}, MAE={float(group_shuffle_row['mae_mean']):.2f}."
        )

    lines += [
        "",
        "## Interpretability Outputs Delivered",
        "- Global SHAP importance generated and ranked by role (modifiable/context/proxy).",
        "- Local SHAP examples generated for high/low prediction and major error cases.",
        "- Surrogate decision rules extracted using only numeric modifiable features.",
        "",
        "## Top Modifiable Levers (Current Model)",
    ]
    for feat in top_modifiable:
        lines.append(f"- `{feat}`")

    lines += [
        "",
        "## Rule-Based Operational Hypotheses",
    ]
    for _, row in rules_df.head(5).iterrows():
        lines.append(
            f"- `{row['rule_conditions']}` "
            f"(support {float(row['support_pct']) * 100:.1f}%, actual lift {float(row['actual_lift_vs_global']):+.2f})"
        )

    lines += [
        "",
        "## Limitations",
        "- No explicit date/season variable is available, limiting temporal alignment and causal interpretation.",
        "- Agriblock-linked confounding remains possible; results are associative, not causal.",
        "- Findings should be validated through controlled field experiments or time-aware data collection.",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = OUT_ROOT / args.run_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    frame = mc.load_analysis_frame()
    features = load_feature_set(Path(args.feature_set_path), args.feature_set_name)
    features = mc.filter_available_features(features, frame)
    params = load_best_params(Path(args.params_path), args.feature_set_name, args.model)
    role_map_df = load_role_map(Path(args.role_map_path))

    x, y, groups, cat_cols = prepare_training_data(frame, features)
    model = build_catboost(params=params, random_state=args.random_state)
    cat_idx = [x.columns.get_loc(col) for col in cat_cols]
    pool = Pool(x, label=y, cat_features=cat_idx)
    model.fit(pool)
    pred = pd.Series(model.predict(x), index=x.index, name="prediction")

    shap_df, expected_value = compute_shap_dataframe(model=model, x=x, y=y, cat_cols=cat_cols)
    global_shap_df = save_global_shap(
        shap_df=shap_df,
        role_map_df=role_map_df,
        out_dir=out_dir,
        top_n=args.global_top_n,
    )

    case_map = select_local_case_indices(y_true=y, y_pred=pred)
    local_overview_df, _ = save_local_outputs(
        x=x,
        y=y,
        groups=groups,
        y_pred=pred,
        shap_df=shap_df,
        case_map=case_map,
        out_dir=out_dir,
        top_n_features=args.local_top_n,
    )

    modifiable_numeric = pick_modifiable_numeric_features(
        x=x,
        model_features=features,
        role_map_df=role_map_df,
    )
    rules_df = extract_modifiable_rules(
        x=x,
        y_actual=y,
        y_pred=pred,
        modifiable_numeric_features=modifiable_numeric,
        n_rules=args.n_rules,
        min_support_pct=args.min_rule_support_pct,
        random_state=args.random_state,
    )
    rules_df.to_csv(out_dir / "modifiable_decision_rules.csv", index=False)
    write_rules_markdown(rules_df, out_dir / "modifiable_decision_rules.md")

    write_actionable_recommendations(
        global_shap_df=global_shap_df,
        rules_df=rules_df,
        role_map_df=role_map_df,
        model_features=features,
        out_path=out_dir / "actionable_recommendations_draft.md",
    )

    logo_row = load_metric_row(Path(args.logo_summary_path), args.feature_set_name, args.model)
    group_shuffle_row = None
    secondary_path = Path(args.secondary_summary_path)
    if secondary_path.exists():
        sec = pd.read_csv(secondary_path)
        sec_rows = sec[
            (sec["evaluation_scheme"].astype(str) == "group_shuffle")
            & (sec["feature_set"].astype(str) == args.feature_set_name)
            & (sec["model"].astype(str).str.lower() == args.model.lower())
        ]
        if not sec_rows.empty:
            group_shuffle_row = sec_rows.iloc[0]

    write_professor_one_pager(
        out_path=out_dir / "professor_milestone_one_page.md",
        feature_set_name=args.feature_set_name,
        model_name=args.model,
        logo_row=logo_row,
        group_shuffle_row=group_shuffle_row,
        global_shap_df=global_shap_df,
        rules_df=rules_df,
    )

    summary_lines = [
        f"run_tag={args.run_tag}",
        f"feature_set={args.feature_set_name}",
        f"model={args.model}",
        f"n_rows={len(x)}",
        f"n_features={len(features)}",
        f"n_modifiable_numeric_for_rules={len(modifiable_numeric)}",
        f"expected_value={expected_value:.6f}",
        f"local_cases={json.dumps(case_map, sort_keys=True)}",
        f"rules_exported={len(rules_df)}",
        f"logo_r2={float(logo_row['r2_mean']):.6f}" if logo_row is not None else "logo_r2=NA",
        (
            f"group_shuffle_r2={float(group_shuffle_row['r2_mean']):.6f}"
            if group_shuffle_row is not None
            else "group_shuffle_r2=NA"
        ),
    ]
    (out_dir / "interpretability_runlog.txt").write_text("\n".join(summary_lines), encoding="utf-8")
    print(f"Saved interpretability artifacts to: {out_dir}")
    print(f"Top global feature: {global_shap_df.iloc[0]['feature']}")
    print(f"Rules exported: {len(rules_df)}")

    return {
        "out_dir": out_dir,
        "global_shap_df": global_shap_df,
        "local_overview_df": local_overview_df,
        "rules_df": rules_df,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SHAP and modifiable-rule interpretability package.")
    parser.add_argument("--run-tag", type=str, default="latest")
    parser.add_argument("--feature-set-path", type=str, default=str(DEFAULT_FEATURE_SET_PATH))
    parser.add_argument("--feature-set-name", type=str, default="full_review")
    parser.add_argument("--params-path", type=str, default=str(DEFAULT_PARAMS_PATH))
    parser.add_argument("--role-map-path", type=str, default=str(DEFAULT_ROLE_MAP_PATH))
    parser.add_argument("--logo-summary-path", type=str, default=str(DEFAULT_LOGO_PATH))
    parser.add_argument("--secondary-summary-path", type=str, default=str(DEFAULT_SECONDARY_PATH))
    parser.add_argument("--model", type=str, default="catboost")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--global-top-n", type=int, default=12)
    parser.add_argument("--local-top-n", type=int, default=10)
    parser.add_argument("--n-rules", type=int, default=5)
    parser.add_argument("--min-rule-support-pct", type=float, default=0.03)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
