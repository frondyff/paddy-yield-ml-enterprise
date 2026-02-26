"""CAROB SHAP interpretability report (CatBoost) under outputs/carob_interpretability."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool

from paddy_yield_ml.pipelines import carob_common as cc
from paddy_yield_ml.pipelines import carob_model_compare as cm

try:
    project_root = Path(__file__).resolve().parents[3]
except NameError:
    project_root = Path.cwd()

OUT_ROOT = project_root / "outputs" / "carob_interpretability"
DEFAULT_SCENARIO_PATH = project_root / "outputs" / "carob_model_compare" / "scenario_feature_sets.csv"
DEFAULT_SUMMARY_PATH = project_root / "outputs" / "carob_model_compare" / "model_comparison_summary.csv"
DEFAULT_CANDIDATES_PATH = project_root / "outputs" / "carob_feature_prepare" / "hybrid_selection_candidates.csv"
DEFAULT_ROLE_MAP_PATH = project_root / "outputs" / "carob_feature_prepare" / "actionability_role_map.csv"
DEFAULT_SCENARIO = "modifiable_plus_context"


def load_scenario_features(path: Path, scenario: str) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Missing scenario feature file: {path}")
    sdf = pd.read_csv(path)
    required = {"scenario", "feature"}
    if not required.issubset(sdf.columns):
        raise ValueError(f"Scenario feature file missing columns: {sorted(required)}")

    features = (
        sdf.loc[sdf["scenario"].astype(str) == scenario, "feature"].astype(str).dropna().tolist()
    )
    if not features:
        available = sorted(sdf["scenario"].astype(str).unique().tolist())
        raise ValueError(f"Scenario '{scenario}' not found. Available: {available}")
    return cm.dedupe_keep_order(features)


def load_best_catboost_params(path: Path, scenario: str) -> tuple[dict[str, Any], str]:
    if not path.exists():
        raise FileNotFoundError(f"Missing model summary file: {path}")
    mdf = pd.read_csv(path)
    required = {"model_key", "scenario", "params_json", "r2"}
    if not required.issubset(mdf.columns):
        raise ValueError(f"Model summary missing columns: {sorted(required)}")

    mdf["model_key"] = mdf["model_key"].astype(str).str.lower()
    mdf["scenario"] = mdf["scenario"].astype(str)

    cat = mdf[(mdf["model_key"] == "catboost") & (mdf["scenario"] == scenario)].copy()
    source_note = "catboost_scenario_best"
    if cat.empty:
        cat = mdf[mdf["model_key"] == "catboost"].copy()
        source_note = "catboost_global_best"

    if cat.empty:
        return {
            "n_estimators": 500,
            "learning_rate": 0.05,
            "depth": 6,
            "l2_leaf_reg": 3.0,
        }, "default_fallback"

    row = cat.sort_values("r2", ascending=False).iloc[0]
    parsed = json.loads(str(row["params_json"]))
    if not isinstance(parsed, dict):
        raise ValueError("CatBoost params_json must decode to object.")
    return parsed, source_note


def prepare_frame(candidates_path: Path, features: list[str]) -> tuple[pd.DataFrame, list[str]]:
    frame = cm.load_frame_with_country_gate(candidates_path)
    usable = cm.filter_available_features(features, frame)

    subset = frame[usable + [cc.TARGET_COL, cc.GROUP_COL]].copy()
    subset[cc.TARGET_COL] = pd.to_numeric(subset[cc.TARGET_COL], errors="coerce")
    subset = subset[subset[cc.TARGET_COL].notna() & subset[cc.GROUP_COL].notna()].reset_index(drop=True)
    return subset, usable


def fit_model(
    frame: pd.DataFrame,
    features: list[str],
    params: dict[str, Any],
    random_state: int,
    test_size: float,
) -> tuple[CatBoostRegressor, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    x = frame[features].copy()
    y = frame[cc.TARGET_COL].copy()
    groups = frame[cc.GROUP_COL].astype(str)

    tr_idx, te_idx = cm.build_trial_aware_split_indices(groups=groups, test_size=test_size, random_state=random_state)
    xtr, xte = x.iloc[tr_idx].copy(), x.iloc[te_idx].copy()
    ytr, yte = y.iloc[tr_idx], y.iloc[te_idx]
    gtr, gte = groups.iloc[tr_idx].astype(str), groups.iloc[te_idx].astype(str)

    xtr, xte = cm.impute_numeric_by_group(
        x_train=xtr,
        x_test=xte,
        groups_train=gtr,
        groups_test=gte,
    )

    cat_cols = xtr.select_dtypes(exclude=[np.number]).columns.tolist()
    for col in cat_cols:
        xtr[col] = xtr[col].astype(str).fillna("MISSING")
        xte[col] = xte[col].astype(str).fillna("MISSING")

    cat_idx = [xtr.columns.get_loc(c) for c in cat_cols]
    model = CatBoostRegressor(
        n_estimators=int(params.get("n_estimators", 500)),
        learning_rate=float(params.get("learning_rate", 0.05)),
        depth=int(params.get("depth", 6)),
        l2_leaf_reg=float(params.get("l2_leaf_reg", 3.0)),
        loss_function="RMSE",
        random_seed=random_state,
        verbose=0,
        allow_writing_files=False,
    )

    train_pool = Pool(xtr, label=ytr, cat_features=cat_idx)
    model.fit(train_pool)

    ypred = pd.Series(model.predict(xte), index=xte.index)
    return model, xte, yte, ypred, gte


def compute_shap(model: CatBoostRegressor, x: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, float]:
    cat_cols = x.select_dtypes(exclude=[np.number]).columns.tolist()
    cat_idx = [x.columns.get_loc(c) for c in cat_cols]
    pool = Pool(x, label=y, cat_features=cat_idx)

    shap_raw = model.get_feature_importance(data=pool, type="ShapValues")
    if shap_raw.ndim != 2 or shap_raw.shape[1] != x.shape[1] + 1:
        raise ValueError("Unexpected SHAP output shape from CatBoost.")

    shap_df = pd.DataFrame(shap_raw[:, :-1], columns=x.columns, index=x.index)
    expected_value = float(np.mean(shap_raw[:, -1]))
    return shap_df, expected_value


def save_outputs(
    out_dir: Path,
    shap_df: pd.DataFrame,
    x: pd.DataFrame,
    y_true: pd.Series,
    y_pred: pd.Series,
    groups: pd.Series,
    role_map_path: Path,
    scenario: str,
    params: dict[str, Any],
    param_source: str,
    expected_value: float,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    role_df = (
        pd.read_csv(role_map_path)
        if role_map_path.exists()
        else pd.DataFrame(columns=["column_name", "final_role"])
    )
    role_lookup = role_df.set_index("column_name")["final_role"].to_dict() if not role_df.empty else {}

    global_df = (
        shap_df.abs().mean(axis=0).sort_values(ascending=False).rename_axis("feature").reset_index(name="mean_abs_shap")
    )
    global_df["final_role"] = global_df["feature"].map(role_lookup).fillna("unknown")
    global_df.to_csv(out_dir / "global_shap_importance.csv", index=False)

    top = global_df.head(15).iloc[::-1]
    colors = [
        "#1b9e77"
        if r == "modifiable"
        else "#7570b3"
        if r == "context"
        else "#d95f02"
        if r == "proxy"
        else "#666666"
        for r in top["final_role"]
    ]
    plt.figure(figsize=(10, 6))
    plt.barh(top["feature"], top["mean_abs_shap"], color=colors)
    plt.xlabel("Mean |SHAP value|")
    plt.title("CAROB Global SHAP Importance (Top 15)")
    plt.tight_layout()
    plt.savefig(out_dir / "global_shap_top_features.png", dpi=180)
    plt.close()

    details = pd.DataFrame({
        "row_index": y_true.index.astype(int),
        "trial_id": groups.values,
        "actual_yield": y_true.values,
        "predicted_yield": y_pred.values,
        "residual_pred_minus_actual": (y_pred - y_true).values,
    })
    details.to_csv(out_dir / "local_examples_overview.csv", index=False)

    top_feature = str(global_df.iloc[0]["feature"]) if not global_df.empty else "n/a"
    top_role = str(global_df.iloc[0]["final_role"]) if not global_df.empty else "unknown"
    top_value = float(global_df.iloc[0]["mean_abs_shap"]) if not global_df.empty else float("nan")

    explanation_lines = [
        "# CAROB Top Feature Explanation",
        "",
        f"- Scenario: `{scenario}`",
        "- SHAP model: `CatBoost`",
        f"- Params source: `{param_source}`",
        f"- Params used: `{json.dumps(params, sort_keys=True)}`",
        f"- Expected value (mean baseline prediction): `{expected_value:.4f}`",
        "",
        "## Top Global Driver",
        f"- Feature: `{top_feature}`",
        f"- Role: `{top_role}`",
        f"- Mean |SHAP|: `{top_value:.6f}`",
        "",
        "## Top 10 Features",
    ]
    for i, row in global_df.head(10).reset_index(drop=True).iterrows():
        rank = int(i) + 1
        explanation_lines.append(
            f"{rank}. `{row['feature']}` | role=`{row['final_role']}` "
            f"| mean_|SHAP|=`{float(row['mean_abs_shap']):.6f}`"
        )
    (out_dir / "top_feature_explanation.md").write_text("\n".join(explanation_lines), encoding="utf-8")

    runlog = [
        f"scenario={scenario}",
        f"rows_explained={len(x)}",
        f"features_used={len(x.columns)}",
        f"params_source={param_source}",
        f"top_feature={top_feature}",
        f"expected_value={expected_value}",
    ]
    (out_dir / "carob_interpretability_runlog.txt").write_text("\n".join(runlog), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CAROB SHAP report with top feature explanation.")
    parser.add_argument("--run-tag", type=str, default="latest")
    parser.add_argument("--scenario", type=str, default=DEFAULT_SCENARIO)
    parser.add_argument("--scenario-path", type=str, default=str(DEFAULT_SCENARIO_PATH))
    parser.add_argument("--summary-path", type=str, default=str(DEFAULT_SUMMARY_PATH))
    parser.add_argument("--candidates-path", type=str, default=str(DEFAULT_CANDIDATES_PATH))
    parser.add_argument("--role-map-path", type=str, default=str(DEFAULT_ROLE_MAP_PATH))
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    out_dir = OUT_ROOT / str(args.run_tag)
    features = load_scenario_features(Path(args.scenario_path), scenario=str(args.scenario))
    params, source = load_best_catboost_params(Path(args.summary_path), scenario=str(args.scenario))
    frame, usable = prepare_frame(Path(args.candidates_path), features)

    model, xte, yte, ypred, gte = fit_model(
        frame=frame,
        features=usable,
        params=params,
        random_state=int(args.random_state),
        test_size=float(args.test_size),
    )
    shap_df, expected_value = compute_shap(model, xte, yte)

    save_outputs(
        out_dir=out_dir,
        shap_df=shap_df,
        x=xte,
        y_true=yte,
        y_pred=ypred,
        groups=gte,
        role_map_path=Path(args.role_map_path),
        scenario=str(args.scenario),
        params=params,
        param_source=source,
        expected_value=expected_value,
    )

    print(f"Saved CAROB interpretability outputs to: {out_dir}")
    top = shap_df.abs().mean(axis=0).sort_values(ascending=False)
    if not top.empty:
        print(f"Top SHAP feature: {top.index[0]} | mean_abs_shap={float(top.iloc[0]):.6f}")


if __name__ == "__main__":
    main()
