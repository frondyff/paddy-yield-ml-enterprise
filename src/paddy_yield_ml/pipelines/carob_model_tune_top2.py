"""Focused fine-tuning for current CAROB top-2 contenders (ExtraTrees + CatBoost)."""

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
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

from paddy_yield_ml.pipelines import carob_common as cc
from paddy_yield_ml.pipelines import carob_model_compare as cm

try:
    project_root = Path(__file__).resolve().parents[3]
except NameError:
    project_root = Path.cwd()

OUT_DIR = project_root / "outputs" / "carob_model_tune_top2"
CANDIDATES_PATH = project_root / "outputs" / "carob_feature_prepare" / "hybrid_selection_candidates.csv"
DEFAULT_SCENARIO = "modifiable_plus_context"
DEFAULT_MODELS = "extratrees,catboost"


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


def parse_models(raw: str) -> list[str]:
    allowed = {"extratrees", "catboost"}
    models = [m.strip().lower() for m in raw.split(",") if m.strip()]
    models = cm.dedupe_keep_order(models)
    if not models:
        raise ValueError("At least one model must be selected.")
    unsupported = [m for m in models if m not in allowed]
    if unsupported:
        raise ValueError(f"Unsupported models for top2 tuner: {unsupported}. Allowed: {sorted(allowed)}")
    return models


def sample_grid(grid: list[dict[str, object]], max_size: int, random_state: int) -> list[dict[str, object]]:
    if len(grid) <= max_size:
        return grid
    rng = np.random.default_rng(random_state)
    idx = rng.choice(len(grid), size=max_size, replace=False)
    return [grid[int(i)] for i in idx]


def build_et_grid_coarse(max_size: int, random_state: int) -> list[dict[str, object]]:
    base: list[dict[str, object]] = []
    for n_estimators, max_depth, min_samples_leaf, min_samples_split, max_features in itertools.product(
        [300, 500, 800],
        [None, 8, 12, 16],
        [1, 2, 4],
        [2, 4, 8],
        ["sqrt", "log2", 0.8],
    ):
        base.append(
            {
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "min_samples_leaf": min_samples_leaf,
                "min_samples_split": min_samples_split,
                "max_features": max_features,
            }
        )
    return sample_grid(base, max_size=max_size, random_state=random_state)


def refine_et_grid(best: dict[str, object]) -> list[dict[str, object]]:
    b = best.copy()
    depth = b["max_depth"]
    candidates = [
        b,
        {**b, "n_estimators": max(200, as_int(b["n_estimators"], name="n_estimators") - 200)},
        {**b, "n_estimators": min(1200, as_int(b["n_estimators"], name="n_estimators") + 200)},
        {**b, "max_depth": None if depth is not None else 16},
        {**b, "max_depth": 12 if depth is None else max(6, as_int(depth, name="max_depth") - 2)},
        {**b, "max_depth": 16 if depth is None else min(24, as_int(depth, name="max_depth") + 2)},
        {**b, "min_samples_leaf": max(1, as_int(b["min_samples_leaf"], name="min_samples_leaf") - 1)},
        {**b, "min_samples_leaf": min(8, as_int(b["min_samples_leaf"], name="min_samples_leaf") + 1)},
        {**b, "min_samples_split": max(2, as_int(b["min_samples_split"], name="min_samples_split") - 2)},
        {**b, "min_samples_split": min(16, as_int(b["min_samples_split"], name="min_samples_split") + 2)},
        {**b, "max_features": "sqrt"},
        {**b, "max_features": "log2"},
        {**b, "max_features": 0.8},
    ]
    dedup = {json.dumps(c, sort_keys=True): c for c in candidates}
    return list(dedup.values())


def build_cat_grid_coarse(max_size: int, random_state: int) -> list[dict[str, object]]:
    base: list[dict[str, object]] = []
    for n_estimators, learning_rate, depth, l2_leaf_reg, random_strength, bagging_temperature in itertools.product(
        [300, 500, 700],
        [0.02, 0.03, 0.05, 0.08],
        [4, 6, 8],
        [2.0, 3.0, 5.0, 7.0],
        [0.5, 1.0, 2.0],
        [0.0, 0.5, 1.0],
    ):
        base.append(
            {
                "n_estimators": n_estimators,
                "learning_rate": learning_rate,
                "depth": depth,
                "l2_leaf_reg": l2_leaf_reg,
                "random_strength": random_strength,
                "bagging_temperature": bagging_temperature,
            }
        )
    return sample_grid(base, max_size=max_size, random_state=random_state)


def refine_cat_grid(best: dict[str, object]) -> list[dict[str, object]]:
    b = best.copy()
    candidates = [
        b,
        {**b, "n_estimators": max(200, as_int(b["n_estimators"], name="n_estimators") - 100)},
        {**b, "n_estimators": min(1200, as_int(b["n_estimators"], name="n_estimators") + 100)},
        {**b, "learning_rate": max(0.01, as_float(b["learning_rate"], name="learning_rate") * 0.8)},
        {**b, "learning_rate": min(0.2, as_float(b["learning_rate"], name="learning_rate") * 1.2)},
        {**b, "depth": max(3, as_int(b["depth"], name="depth") - 1)},
        {**b, "depth": min(10, as_int(b["depth"], name="depth") + 1)},
        {**b, "l2_leaf_reg": max(1.0, as_float(b["l2_leaf_reg"], name="l2_leaf_reg") * 0.7)},
        {**b, "l2_leaf_reg": as_float(b["l2_leaf_reg"], name="l2_leaf_reg") * 1.5},
        {**b, "random_strength": max(0.1, as_float(b["random_strength"], name="random_strength") * 0.7)},
        {**b, "random_strength": as_float(b["random_strength"], name="random_strength") * 1.5},
        {
            **b,
            "bagging_temperature": max(0.0, as_float(b["bagging_temperature"], name="bagging_temperature") - 0.5),
        },
        {
            **b,
            "bagging_temperature": min(3.0, as_float(b["bagging_temperature"], name="bagging_temperature") + 0.5),
        },
    ]
    dedup = {json.dumps(c, sort_keys=True): c for c in candidates}
    return list(dedup.values())


def build_model(model_key: str, params: dict[str, object], seed: int) -> object:
    if model_key == "extratrees":
        max_depth_raw = params["max_depth"]
        return ExtraTreesRegressor(
            n_estimators=as_int(params["n_estimators"], name="n_estimators"),
            max_depth=(None if max_depth_raw is None else as_int(max_depth_raw, name="max_depth")),
            min_samples_leaf=as_int(params["min_samples_leaf"], name="min_samples_leaf"),
            min_samples_split=as_int(params["min_samples_split"], name="min_samples_split"),
            max_features=params["max_features"],
            random_state=seed,
            n_jobs=-1,
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
    parser = argparse.ArgumentParser(description="Tune top-2 CAROB contenders (ExtraTrees + CatBoost).")
    parser.add_argument("--scenario", type=str, default=DEFAULT_SCENARIO)
    parser.add_argument("--models", type=str, default=DEFAULT_MODELS)
    parser.add_argument("--candidates-path", type=str, default=str(CANDIDATES_PATH))
    parser.add_argument("--out-dir", type=str, default=str(OUT_DIR))
    parser.add_argument("--seed", type=int, default=42, help="Primary tuning seed.")
    parser.add_argument("--stability-seeds", type=str, default="42,52,62")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--trial-median-impute", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--et-coarse-max", type=int, default=42)
    parser.add_argument("--cat-coarse-max", type=int, default=48)
    return parser.parse_args()


def tune_single_model(
    *,
    frame: pd.DataFrame,
    features: list[str],
    model_key: str,
    seed: int,
    top_k: int,
    test_size: float,
    trial_median_impute: bool,
    coarse_sample_max: int,
    out_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if model_key == "extratrees":
        coarse_grid = build_et_grid_coarse(max_size=coarse_sample_max, random_state=11 + seed)
        coarse = run_grid_on_seed(
            frame=frame,
            features=features,
            model_key=model_key,
            grid=coarse_grid,
            test_size=test_size,
            seed=seed,
            stage_name="coarse",
            trial_median_impute=trial_median_impute,
        )
        best_params = json.loads(str(coarse.iloc[0]["params_json"]))
        refine_grid = refine_et_grid(best_params)
    else:
        coarse_grid = build_cat_grid_coarse(max_size=coarse_sample_max, random_state=23 + seed)
        coarse = run_grid_on_seed(
            frame=frame,
            features=features,
            model_key=model_key,
            grid=coarse_grid,
            test_size=test_size,
            seed=seed,
            stage_name="coarse",
            trial_median_impute=trial_median_impute,
        )
        best_params = json.loads(str(coarse.iloc[0]["params_json"]))
        refine_grid = refine_cat_grid(best_params)

    coarse.to_csv(out_dir / f"{model_key}_stage1_coarse.csv", index=False)
    refine = run_grid_on_seed(
        frame=frame,
        features=features,
        model_key=model_key,
        grid=refine_grid,
        test_size=test_size,
        seed=seed,
        stage_name="refine",
        trial_median_impute=trial_median_impute,
    )
    refine.to_csv(out_dir / f"{model_key}_stage2_refine.csv", index=False)

    all_df = pd.concat([coarse, refine], ignore_index=True).drop_duplicates(subset=["params_json"])
    top_df = all_df.sort_values(["r2", "rmse"], ascending=[False, True]).head(int(top_k)).copy()
    top_df["config_rank"] = range(1, len(top_df) + 1)
    top_df.to_csv(out_dir / f"{model_key}_top_configs.csv", index=False)
    return coarse, top_df


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
    models = parse_models(args.models)
    stability_seeds = parse_seeds(args.stability_seeds)

    feature_df = pd.DataFrame(
        [{"scenario": args.scenario, "rank_in_scenario": i, "feature": f} for i, f in enumerate(features, start=1)]
    )
    feature_df.to_csv(out_dir / "scenario_feature_set.csv", index=False)

    print(f"Scenario: {args.scenario} | features={len(features)} | rows={len(frame)}")
    print(f"Models: {models}")

    top_by_model: dict[str, pd.DataFrame] = {}
    for model_key in models:
        print(f"\nStage 1-2: {model_key}")
        coarse_max = int(args.et_coarse_max) if model_key == "extratrees" else int(args.cat_coarse_max)
        coarse, top_df = tune_single_model(
            frame=frame,
            features=features,
            model_key=model_key,
            seed=int(args.seed),
            top_k=int(args.top_k),
            test_size=float(args.test_size),
            trial_median_impute=bool(args.trial_median_impute),
            coarse_sample_max=coarse_max,
            out_dir=out_dir,
        )
        top_by_model[model_key] = top_df[["config_rank", "params_json"]].copy()
        print(
            f"  best coarse: R2={coarse.iloc[0]['r2']:.4f} | RMSE={coarse.iloc[0]['rmse']:.2f}"
            f" | params={json.loads(str(coarse.iloc[0]['params_json']))}"
        )

    print("\nStage 3: stability check")
    winners: list[dict[str, object]] = []
    model_label = {"extratrees": "ExtraTrees", "catboost": "CatBoost"}
    for model_key in models:
        stab = stability_check(
            frame=frame,
            features=features,
            model_key=model_key,
            top_configs=top_by_model[model_key],
            seeds=stability_seeds,
            test_size=float(args.test_size),
            trial_median_impute=bool(args.trial_median_impute),
        )
        stab.to_csv(out_dir / f"{model_key}_stability_summary.csv", index=False)
        if not stab.empty:
            best = stab.sort_values(["r2_mean", "rmse_mean"], ascending=[False, True]).iloc[0].to_dict()
            best["model"] = model_label.get(model_key, model_key)
            winners.append(best)

    winner_df = (
        pd.DataFrame(winners).sort_values(["r2_mean", "rmse_mean"], ascending=[False, True]).reset_index(drop=True)
    )
    winner_df.to_csv(out_dir / "model_winners.csv", index=False)

    lines: list[str] = [
        "# Final Model Decision (Top-2 Contenders Tune)",
        "",
        f"- Scenario: `{args.scenario}`",
        f"- Models tuned: `{models}`",
        f"- Split: trial-aware `{args.test_size:.2f}` (seed set: `{stability_seeds}`)",
        f"- Trial-median imputation: `{bool(args.trial_median_impute)}`",
        f"- Candidate frame rows: `{len(frame)}`",
        f"- Features used: `{len(features)}`",
        "",
        "## Winner Ranking",
    ]
    for i, row in winner_df.iterrows():
        rank = as_int(i, name="winner_rank") + 1
        lines.append(
            f"{rank}. {row['model']} | r2_mean={row['r2_mean']:.4f} | rmse_mean={row['rmse_mean']:.2f} "
            f"| mae_mean={row['mae_mean']:.2f} | rmse_worst={row['rmse_worst']:.2f}"
        )
    if not winner_df.empty:
        top = winner_df.iloc[0]
        lines.extend(
            [
                "",
                "## Selected Configuration",
                f"- Model: `{top['model']}`",
                f"- Params: `{top['params_json']}`",
            ]
        )
    (out_dir / "final_model_decision.md").write_text("\n".join(lines), encoding="utf-8")

    print("\nWinner summary:")
    if winner_df.empty:
        print("No winners computed.")
    else:
        print(winner_df[["model", "r2_mean", "rmse_mean", "mae_mean", "rmse_worst"]].to_string(index=False))
    print(f"\nSaved tuning outputs to: {out_dir}")


if __name__ == "__main__":
    main()
