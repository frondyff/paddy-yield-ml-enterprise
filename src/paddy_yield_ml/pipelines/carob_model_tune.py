"""Focused fine-tuning for CAROB on a locked scenario and trial-aware split."""

from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from paddy_yield_ml.pipelines import carob_common as cc
from paddy_yield_ml.pipelines import carob_model_compare as cm

try:
    project_root = Path(__file__).resolve().parents[3]
except NameError:
    project_root = Path.cwd()

OUT_DIR = project_root / "outputs" / "carob_model_tune"
CANDIDATES_PATH = project_root / "outputs" / "carob_feature_prepare" / "hybrid_selection_candidates.csv"
DEFAULT_SCENARIO = "modifiable_plus_context"


def as_int(value: object, *, name: str) -> int:
    try:
        return int(cast("int | float | str", value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Expected integer-like value for {name}, got {value!r}") from exc


def as_float(value: object, *, name: str) -> float:
    try:
        return float(cast("int | float | str", value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Expected float-like value for {name}, got {value!r}") from exc


def parse_seeds(raw: str) -> list[int]:
    out = [int(x.strip()) for x in raw.split(",") if x.strip()]
    if not out:
        raise ValueError("At least one seed is required.")
    return out


def build_xgb_grid_coarse() -> list[dict[str, object]]:
    grid: list[dict[str, object]] = []
    for n_estimators, learning_rate, max_depth, min_child_weight in itertools.product(
        [300, 500],
        [0.03, 0.05],
        [4, 6],
        [1, 3],
    ):
        grid.append(
            {
                "n_estimators": n_estimators,
                "learning_rate": learning_rate,
                "max_depth": max_depth,
                "min_child_weight": min_child_weight,
                "subsample": 0.9,
                "colsample_bytree": 0.9,
                "reg_lambda": 1.0,
            }
        )
    return grid


def build_cat_grid_coarse() -> list[dict[str, object]]:
    grid: list[dict[str, object]] = []
    for n_estimators, learning_rate, depth, l2_leaf_reg in itertools.product(
        [300, 500],
        [0.03, 0.05],
        [4, 6],
        [3.0, 5.0],
    ):
        grid.append(
            {
                "n_estimators": n_estimators,
                "learning_rate": learning_rate,
                "depth": depth,
                "l2_leaf_reg": l2_leaf_reg,
                "random_strength": 1.0,
                "bagging_temperature": 0.0,
            }
        )
    return grid


def refine_xgb_grid(best: dict[str, object]) -> list[dict[str, object]]:
    b = best.copy()
    lr = as_float(b["learning_rate"], name="learning_rate")
    depth = as_int(b["max_depth"], name="max_depth")
    mcw = as_int(b["min_child_weight"], name="min_child_weight")
    reg = as_float(b.get("reg_lambda", 1.0), name="reg_lambda")
    n_est = as_int(b["n_estimators"], name="n_estimators")

    candidates = [
        b,
        {**b, "learning_rate": max(0.01, lr * 0.8)},
        {**b, "learning_rate": min(0.2, lr * 1.2)},
        {**b, "max_depth": max(3, depth - 1)},
        {**b, "max_depth": min(10, depth + 1)},
        {**b, "min_child_weight": max(1, mcw - 1)},
        {**b, "min_child_weight": mcw + 1},
        {**b, "reg_lambda": max(0.1, reg * 0.6)},
        {**b, "reg_lambda": reg * 1.6},
        {**b, "n_estimators": max(200, n_est - 100)},
        {**b, "n_estimators": min(900, n_est + 100)},
    ]
    dedup: dict[str, dict[str, object]] = {json.dumps(c, sort_keys=True): c for c in candidates}
    return list(dedup.values())


def refine_cat_grid(best: dict[str, object]) -> list[dict[str, object]]:
    b = best.copy()
    lr = as_float(b["learning_rate"], name="learning_rate")
    depth = as_int(b["depth"], name="depth")
    l2 = as_float(b["l2_leaf_reg"], name="l2_leaf_reg")
    rs = as_float(b.get("random_strength", 1.0), name="random_strength")
    bt = as_float(b.get("bagging_temperature", 0.0), name="bagging_temperature")
    n_est = as_int(b["n_estimators"], name="n_estimators")

    candidates = [
        b,
        {**b, "learning_rate": max(0.01, lr * 0.8)},
        {**b, "learning_rate": min(0.2, lr * 1.2)},
        {**b, "depth": max(3, depth - 1)},
        {**b, "depth": min(10, depth + 1)},
        {**b, "l2_leaf_reg": max(1.0, l2 * 0.7)},
        {**b, "l2_leaf_reg": l2 * 1.5},
        {**b, "random_strength": max(0.1, rs * 0.7)},
        {**b, "random_strength": rs * 1.5},
        {**b, "bagging_temperature": max(0.0, bt - 0.5)},
        {**b, "bagging_temperature": bt + 0.5},
        {**b, "n_estimators": max(200, n_est - 100)},
        {**b, "n_estimators": min(900, n_est + 100)},
    ]
    dedup: dict[str, dict[str, object]] = {json.dumps(c, sort_keys=True): c for c in candidates}
    return list(dedup.values())


def build_model(model_key: str, params: dict[str, object], seed: int) -> object:
    if model_key == "xgboost":
        return XGBRegressor(
            n_estimators=as_int(params["n_estimators"], name="n_estimators"),
            learning_rate=as_float(params["learning_rate"], name="learning_rate"),
            max_depth=as_int(params["max_depth"], name="max_depth"),
            min_child_weight=as_int(params["min_child_weight"], name="min_child_weight"),
            subsample=as_float(params["subsample"], name="subsample"),
            colsample_bytree=as_float(params["colsample_bytree"], name="colsample_bytree"),
            reg_lambda=as_float(params.get("reg_lambda", 1.0), name="reg_lambda"),
            objective="reg:squarederror",
            random_state=seed,
            n_jobs=-1,
            tree_method="hist",
        )
    if model_key == "catboost":
        return CatBoostRegressor(
            n_estimators=as_int(params["n_estimators"], name="n_estimators"),
            learning_rate=as_float(params["learning_rate"], name="learning_rate"),
            depth=as_int(params["depth"], name="depth"),
            l2_leaf_reg=as_float(params["l2_leaf_reg"], name="l2_leaf_reg"),
            random_strength=as_float(params.get("random_strength", 1.0), name="random_strength"),
            bagging_temperature=as_float(params.get("bagging_temperature", 0.0), name="bagging_temperature"),
            loss_function="RMSE",
            random_seed=seed,
            verbose=0,
        )
    raise ValueError(f"Unsupported model_key: {model_key}")


def evaluate_single_seed(
    frame: pd.DataFrame,
    features: list[str],
    params: dict[str, object],
    model_key: str,
    *,
    test_size: float,
    seed: int,
    trial_median_impute: bool = True,
) -> dict[str, object]:
    subset = frame[features + [cc.TARGET_COL, cc.GROUP_COL]].copy()
    subset[cc.TARGET_COL] = pd.to_numeric(subset[cc.TARGET_COL], errors="coerce")
    subset = subset[subset[cc.TARGET_COL].notna() & subset[cc.GROUP_COL].notna()].reset_index(drop=True)

    x = subset[features].copy()
    y = subset[cc.TARGET_COL].copy()
    groups = subset[cc.GROUP_COL].astype(str)

    tr_idx, te_idx = cm.build_trial_aware_split_indices(groups=groups, test_size=test_size, random_state=seed)
    xtr, xte = x.iloc[tr_idx].copy(), x.iloc[te_idx].copy()
    ytr, yte = y.iloc[tr_idx], y.iloc[te_idx]
    gtr, gte = groups.iloc[tr_idx].astype(str), groups.iloc[te_idx].astype(str)

    if trial_median_impute:
        xtr, xte = cm.impute_numeric_by_group(
            x_train=xtr,
            x_test=xte,
            groups_train=gtr,
            groups_test=gte,
        )

    pipeline = Pipeline(
        [
            ("preprocess", cm.make_preprocessor(xtr)),
            ("model", build_model(model_key=model_key, params=params, seed=seed)),
        ]
    )
    t0 = time.perf_counter()
    pipeline.fit(xtr, ytr)
    fit_seconds = float(time.perf_counter() - t0)
    pred = pipeline.predict(xte)

    return {
        "seed": seed,
        "n_train": int(len(tr_idx)),
        "n_test": int(len(te_idx)),
        "n_trials_in_test": int(gte.nunique()),
        "mae": float(mean_absolute_error(yte, pred)),
        "rmse": float(np.sqrt(mean_squared_error(yte, pred))),
        "r2": float(r2_score(yte, pred)),
        "fit_seconds": fit_seconds,
    }


def run_grid_on_seed(
    frame: pd.DataFrame,
    features: list[str],
    model_key: str,
    grid: list[dict[str, object]],
    *,
    test_size: float,
    seed: int,
    stage_name: str,
    trial_median_impute: bool,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for i, params in enumerate(grid, start=1):
        m = evaluate_single_seed(
            frame=frame,
            features=features,
            params=params,
            model_key=model_key,
            test_size=test_size,
            seed=seed,
            trial_median_impute=trial_median_impute,
        )
        rows.append(
            {
                "model_key": model_key,
                "stage": stage_name,
                "param_set": i,
                "params_json": json.dumps(params, sort_keys=True),
                "r2": m["r2"],
                "rmse": m["rmse"],
                "mae": m["mae"],
                "fit_seconds": m["fit_seconds"],
                "seed": seed,
            }
        )
    return pd.DataFrame(rows).sort_values(["r2", "rmse"], ascending=[False, True]).reset_index(drop=True)


def stability_check(
    frame: pd.DataFrame,
    features: list[str],
    model_key: str,
    top_configs: pd.DataFrame,
    *,
    seeds: list[int],
    test_size: float,
    trial_median_impute: bool,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, cfg in top_configs.iterrows():
        params = json.loads(str(cfg["params_json"]))
        cfg_id = int(cfg["config_rank"])
        for seed in seeds:
            m = evaluate_single_seed(
                frame=frame,
                features=features,
                params=params,
                model_key=model_key,
                test_size=test_size,
                seed=int(seed),
                trial_median_impute=trial_median_impute,
            )
            rows.append(
                {
                    "model_key": model_key,
                    "config_rank": cfg_id,
                    "seed": int(seed),
                    "params_json": json.dumps(params, sort_keys=True),
                    "r2": m["r2"],
                    "rmse": m["rmse"],
                    "mae": m["mae"],
                    "fit_seconds": m["fit_seconds"],
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    summary = (
        out.groupby(["model_key", "config_rank", "params_json"], as_index=False)
        .agg(
            r2_mean=("r2", "mean"),
            rmse_mean=("rmse", "mean"),
            mae_mean=("mae", "mean"),
            rmse_worst=("rmse", "max"),
            fit_seconds_mean=("fit_seconds", "mean"),
        )
        .sort_values(["r2_mean", "rmse_mean"], ascending=[False, True])
        .reset_index(drop=True)
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune XGBoost and CatBoost for CAROB trial-aware evaluation.")
    parser.add_argument("--scenario", type=str, default=DEFAULT_SCENARIO)
    parser.add_argument("--candidates-path", type=str, default=str(CANDIDATES_PATH))
    parser.add_argument("--out-dir", type=str, default=str(OUT_DIR))
    parser.add_argument("--seed", type=int, default=42, help="Primary tuning seed.")
    parser.add_argument("--stability-seeds", type=str, default="42,52,62")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--trial-median-impute", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    candidates_path = Path(args.candidates_path)

    frame = cm.load_frame_with_country_gate(candidates_path)
    candidates = cm.load_hybrid_candidates(candidates_path)
    feature_sets = cm.build_feature_sets(candidates)
    if args.scenario not in feature_sets:
        raise ValueError(f"Scenario '{args.scenario}' not found. Available: {sorted(feature_sets)}")
    features = cm.filter_available_features(feature_sets[args.scenario], frame)

    feature_df = pd.DataFrame(
        [{"scenario": args.scenario, "rank_in_scenario": i, "feature": f} for i, f in enumerate(features, start=1)]
    )
    feature_df.to_csv(out_dir / "scenario_feature_set.csv", index=False)

    print(f"Scenario: {args.scenario} | features={len(features)} | rows={len(frame)}")
    print("Stage 1: coarse search")

    xgb_coarse = run_grid_on_seed(
        frame=frame,
        features=features,
        model_key="xgboost",
        grid=build_xgb_grid_coarse(),
        test_size=float(args.test_size),
        seed=int(args.seed),
        stage_name="coarse",
        trial_median_impute=bool(args.trial_median_impute),
    )
    cat_coarse = run_grid_on_seed(
        frame=frame,
        features=features,
        model_key="catboost",
        grid=build_cat_grid_coarse(),
        test_size=float(args.test_size),
        seed=int(args.seed),
        stage_name="coarse",
        trial_median_impute=bool(args.trial_median_impute),
    )
    xgb_coarse.to_csv(out_dir / "xgboost_stage1_coarse.csv", index=False)
    cat_coarse.to_csv(out_dir / "catboost_stage1_coarse.csv", index=False)

    best_xgb_params = json.loads(str(xgb_coarse.iloc[0]["params_json"]))
    best_cat_params = json.loads(str(cat_coarse.iloc[0]["params_json"]))

    print(
        f"  XGBoost coarse best: R2={xgb_coarse.iloc[0]['r2']:.4f} | RMSE={xgb_coarse.iloc[0]['rmse']:.2f}"
        f" | params={best_xgb_params}"
    )
    print(
        f"  CatBoost coarse best: R2={cat_coarse.iloc[0]['r2']:.4f} | RMSE={cat_coarse.iloc[0]['rmse']:.2f}"
        f" | params={best_cat_params}"
    )

    print("Stage 2: local refinement")
    xgb_refine = run_grid_on_seed(
        frame=frame,
        features=features,
        model_key="xgboost",
        grid=refine_xgb_grid(best_xgb_params),
        test_size=float(args.test_size),
        seed=int(args.seed),
        stage_name="refine",
        trial_median_impute=bool(args.trial_median_impute),
    )
    cat_refine = run_grid_on_seed(
        frame=frame,
        features=features,
        model_key="catboost",
        grid=refine_cat_grid(best_cat_params),
        test_size=float(args.test_size),
        seed=int(args.seed),
        stage_name="refine",
        trial_median_impute=bool(args.trial_median_impute),
    )
    xgb_refine.to_csv(out_dir / "xgboost_stage2_refine.csv", index=False)
    cat_refine.to_csv(out_dir / "catboost_stage2_refine.csv", index=False)

    xgb_all = pd.concat([xgb_coarse, xgb_refine], ignore_index=True).drop_duplicates(subset=["params_json"])
    cat_all = pd.concat([cat_coarse, cat_refine], ignore_index=True).drop_duplicates(subset=["params_json"])

    xgb_top = xgb_all.sort_values(["r2", "rmse"], ascending=[False, True]).head(int(args.top_k)).copy()
    cat_top = cat_all.sort_values(["r2", "rmse"], ascending=[False, True]).head(int(args.top_k)).copy()
    xgb_top["config_rank"] = range(1, len(xgb_top) + 1)
    cat_top["config_rank"] = range(1, len(cat_top) + 1)
    xgb_top.to_csv(out_dir / "xgboost_top_configs.csv", index=False)
    cat_top.to_csv(out_dir / "catboost_top_configs.csv", index=False)

    print("Stage 3: stability check (top configs, multiple seeds)")
    stability_seeds = parse_seeds(args.stability_seeds)
    xgb_stab = stability_check(
        frame=frame,
        features=features,
        model_key="xgboost",
        top_configs=xgb_top[["config_rank", "params_json"]],
        seeds=stability_seeds,
        test_size=float(args.test_size),
        trial_median_impute=bool(args.trial_median_impute),
    )
    cat_stab = stability_check(
        frame=frame,
        features=features,
        model_key="catboost",
        top_configs=cat_top[["config_rank", "params_json"]],
        seeds=stability_seeds,
        test_size=float(args.test_size),
        trial_median_impute=bool(args.trial_median_impute),
    )
    xgb_stab.to_csv(out_dir / "xgboost_stability_summary.csv", index=False)
    cat_stab.to_csv(out_dir / "catboost_stability_summary.csv", index=False)

    winners: list[dict[str, object]] = []
    if not xgb_stab.empty:
        wx = xgb_stab.sort_values(["r2_mean", "rmse_mean"], ascending=[False, True]).iloc[0].to_dict()
        wx["model"] = "XGBoost"
        winners.append(wx)
    if not cat_stab.empty:
        wc = cat_stab.sort_values(["r2_mean", "rmse_mean"], ascending=[False, True]).iloc[0].to_dict()
        wc["model"] = "CatBoost"
        winners.append(wc)

    winner_df = pd.DataFrame(winners).sort_values(["r2_mean", "rmse_mean"], ascending=[False, True]).reset_index(
        drop=True
    )
    winner_df.to_csv(out_dir / "model_winners.csv", index=False)

    decision_lines: list[str] = [
        "# Final Model Decision",
        "",
        f"- Scenario: `{args.scenario}`",
        f"- Split: trial-aware `{args.test_size:.2f}` (seed set: `{stability_seeds}`)",
        f"- Trial-median imputation: `{bool(args.trial_median_impute)}`",
        "",
        "## Winner Ranking",
    ]
    for i, row in winner_df.iterrows():
        rank = as_int(i, name="winner_rank") + 1
        decision_lines.append(
            f"{rank}. {row['model']} | r2_mean={row['r2_mean']:.4f} | rmse_mean={row['rmse_mean']:.2f} "
            f"| mae_mean={row['mae_mean']:.2f} | rmse_worst={row['rmse_worst']:.2f}"
        )
    if not winner_df.empty:
        top = winner_df.iloc[0]
        decision_lines.extend(
            [
                "",
                "## Selected Configuration",
                f"- Model: `{top['model']}`",
                f"- Params: `{top['params_json']}`",
            ]
        )
    (out_dir / "final_model_decision.md").write_text("\n".join(decision_lines), encoding="utf-8")

    print("\nModel winner summary:")
    if winner_df.empty:
        print("No winners computed.")
    else:
        print(winner_df[["model", "r2_mean", "rmse_mean", "mae_mean", "rmse_worst"]].to_string(index=False))
    print(f"\nSaved tuning artifacts to: {out_dir}")


if __name__ == "__main__":
    main()
