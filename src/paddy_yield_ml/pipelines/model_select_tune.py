"""
Step 3 pipeline: wrapper/stability selection + stronger tuning with group-aware validation.

Run:
  python src/paddy_yield_ml/pipelines/model_select_tune.py --run-tag strict

Outputs (under ./outputs/model_select_tune/<run-tag>/):
  - wrapper_runs.csv
  - stability_selection_frequency.csv
  - scenario_feature_sets.csv
  - tuning_trials.csv
  - tuned_groupkfold_summary.csv
  - logo_fold_metrics.csv
  - logo_summary.csv
  - secondary_split_fold_metrics.csv
  - secondary_split_summary.csv
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import (
    GroupKFold,
    GroupShuffleSplit,
    LeaveOneGroupOut,
    ParameterSampler,
    ShuffleSplit,
)
from sklearn.pipeline import Pipeline

from paddy_yield_ml.pipelines import model_compare as mc

try:
    project_root = Path(__file__).resolve().parents[3]
except NameError:
    project_root = Path.cwd()

OUT_ROOT = project_root / "outputs" / "model_select_tune"
RAW_TARGET_COL = mc.RAW_TARGET_COL
TARGET_COL = mc.TARGET_COL
GROUP_COL = mc.GROUP_COL


def parse_csv_list(raw: str) -> list[str]:
    values = [v.strip() for v in raw.split(",") if v.strip()]
    if not values:
        raise ValueError("Expected at least one comma-separated value.")
    return mc.dedupe_keep_order(values)


def parse_int_list(raw: str) -> list[int]:
    values = [int(v.strip()) for v in raw.split(",") if v.strip()]
    if not values:
        raise ValueError("Expected at least one comma-separated integer.")
    return values


def core_and_review_features(candidates_df: pd.DataFrame) -> tuple[list[str], list[str]]:
    core = (
        candidates_df.loc[
            candidates_df["status"].astype(str).isin(["candidate_modifiable", "reserve_context"]),
            "feature",
        ]
        .astype(str)
        .tolist()
    )
    review = (
        candidates_df.loc[
            candidates_df["status"].astype(str) == "candidate_redundant_review",
            "feature",
        ]
        .astype(str)
        .tolist()
    )
    core = mc.dedupe_keep_order(core)
    review = [f for f in mc.dedupe_keep_order(review) if f not in core]
    if not core:
        raise ValueError("No core features found from hybrid selection candidates.")
    return core, review


def _build_model(model_key: str, params: dict[str, Any], random_state: int) -> Any:
    if model_key == "random_forest":
        base: dict[str, Any] = {
            "n_estimators": 300,
            "min_samples_leaf": 2,
            "n_jobs": -1,
            "random_state": random_state,
        }
        base.update(params)
        return RandomForestRegressor(**base)

    if model_key == "lightgbm":
        from lightgbm import LGBMRegressor

        base: dict[str, Any] = {
            "n_estimators": 400,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_child_samples": 20,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "random_state": random_state,
            "verbosity": -1,
        }
        base.update(params)
        return LGBMRegressor(**base)

    if model_key == "xgboost":
        from xgboost import XGBRegressor

        base: dict[str, Any] = {
            "n_estimators": 400,
            "learning_rate": 0.05,
            "max_depth": 6,
            "min_child_weight": 1,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "objective": "reg:squarederror",
            "random_state": random_state,
            "n_jobs": -1,
        }
        base.update(params)
        return XGBRegressor(**base)

    if model_key == "catboost":
        from catboost import CatBoostRegressor

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

    raise ValueError(f"Unsupported model key: {model_key}")


def _prepare_xy(frame: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    subset = frame[features + [TARGET_COL, GROUP_COL]].copy()
    subset[TARGET_COL] = pd.to_numeric(subset[TARGET_COL], errors="coerce")
    subset[GROUP_COL] = subset[GROUP_COL].astype("string")
    subset = subset[subset[TARGET_COL].notna() & subset[GROUP_COL].notna()].reset_index(drop=True)
    return subset[features].copy(), subset[TARGET_COL].copy(), subset[GROUP_COL].astype(str).copy()


def evaluate_groupkfold(
    frame: pd.DataFrame,
    features: list[str],
    model_key: str,
    params: dict[str, Any],
    random_state: int,
    n_splits: int,
) -> dict:
    x, y, groups = _prepare_xy(frame, features)
    n_groups = groups.nunique()
    if n_groups < 2:
        raise ValueError("Need at least 2 groups for group-aware CV.")
    splits = min(n_splits, n_groups)
    cv = GroupKFold(n_splits=splits)

    maes: list[float] = []
    rmses: list[float] = []
    r2s: list[float] = []
    runtimes: list[float] = []

    for train_idx, test_idx in cv.split(x, y, groups):
        x_train = x.iloc[train_idx].copy()
        x_test = x.iloc[test_idx].copy()
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        pipeline = Pipeline(
            [
                ("preprocess", mc.make_preprocessor(x_train)),
                ("model", _build_model(model_key, params=params, random_state=random_state)),
            ]
        )

        t0 = time.perf_counter()
        pipeline.fit(x_train, y_train)
        preds = pipeline.predict(x_test)
        runtimes.append(float(time.perf_counter() - t0))

        maes.append(float(mean_absolute_error(y_test, preds)))
        rmses.append(float(np.sqrt(mean_squared_error(y_test, preds))))
        r2s.append(float(r2_score(y_test, preds)))

    return {
        "mae_mean": float(np.mean(maes)),
        "mae_std": float(np.std(maes)),
        "rmse_mean": float(np.mean(rmses)),
        "rmse_std": float(np.std(rmses)),
        "r2_mean": float(np.mean(r2s)),
        "r2_std": float(np.std(r2s)),
        "fit_predict_seconds_mean": float(np.mean(runtimes)),
        "n_folds": int(splits),
    }


def evaluate_logo(
    frame: pd.DataFrame,
    feature_set_name: str,
    features: list[str],
    model_key: str,
    params: dict[str, Any],
    random_state: int,
) -> tuple[pd.DataFrame, dict]:
    x, y, groups = _prepare_xy(frame, features)
    logo = LeaveOneGroupOut()

    fold_rows: list[dict] = []
    maes: list[float] = []
    rmses: list[float] = []
    r2s: list[float] = []
    runtimes: list[float] = []

    for fold_idx, (train_idx, test_idx) in enumerate(logo.split(x, y, groups), start=1):
        x_train = x.iloc[train_idx].copy()
        x_test = x.iloc[test_idx].copy()
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]
        held_out = groups.iloc[test_idx].iloc[0]

        pipeline = Pipeline(
            [
                ("preprocess", mc.make_preprocessor(x_train)),
                ("model", _build_model(model_key, params=params, random_state=random_state)),
            ]
        )

        t0 = time.perf_counter()
        pipeline.fit(x_train, y_train)
        preds = pipeline.predict(x_test)
        runtime_s = float(time.perf_counter() - t0)

        mae = float(mean_absolute_error(y_test, preds))
        rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
        r2 = float(r2_score(y_test, preds))

        maes.append(mae)
        rmses.append(rmse)
        r2s.append(r2)
        runtimes.append(runtime_s)

        fold_rows.append(
            {
                "feature_set": feature_set_name,
                "model": model_key,
                "fold": fold_idx,
                "held_out_agriblock": held_out,
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                "mae": mae,
                "rmse": rmse,
                "r2": r2,
                "fit_predict_seconds": runtime_s,
            }
        )

    summary = {
        "feature_set": feature_set_name,
        "model": model_key,
        "n_features": int(len(features)),
        "n_folds": int(len(maes)),
        "mae_mean": float(np.mean(maes)),
        "mae_std": float(np.std(maes)),
        "rmse_mean": float(np.mean(rmses)),
        "rmse_std": float(np.std(rmses)),
        "r2_mean": float(np.mean(r2s)),
        "r2_std": float(np.std(r2s)),
        "fit_predict_seconds_mean": float(np.mean(runtimes)),
    }
    return pd.DataFrame(fold_rows), summary


def evaluate_secondary_splits(
    frame: pd.DataFrame,
    feature_set_name: str,
    features: list[str],
    model_key: str,
    params: dict[str, Any],
    random_state: int,
    group_shuffle_splits: int,
    random_shuffle_splits: int,
    test_size: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    x, y, groups = _prepare_xy(frame, features)
    if len(x) < 10:
        raise ValueError("Need at least 10 rows for secondary split benchmarks.")

    fold_rows: list[dict] = []
    summary_rows: list[dict] = []

    # Secondary benchmark 1: seen-group generalization (group-aware shuffled holdouts).
    gss_splits = min(group_shuffle_splits, max(2, groups.nunique()))
    gss = GroupShuffleSplit(
        n_splits=gss_splits,
        test_size=test_size,
        random_state=random_state,
    )
    maes_g: list[float] = []
    rmses_g: list[float] = []
    r2s_g: list[float] = []
    runtimes_g: list[float] = []
    for fold_idx, (train_idx, test_idx) in enumerate(gss.split(x, y, groups), start=1):
        x_train = x.iloc[train_idx].copy()
        x_test = x.iloc[test_idx].copy()
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        pipeline = Pipeline(
            [
                ("preprocess", mc.make_preprocessor(x_train)),
                ("model", _build_model(model_key, params=params, random_state=random_state)),
            ]
        )
        t0 = time.perf_counter()
        pipeline.fit(x_train, y_train)
        preds = pipeline.predict(x_test)
        runtime_s = float(time.perf_counter() - t0)

        mae = float(mean_absolute_error(y_test, preds))
        rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
        r2 = float(r2_score(y_test, preds))

        maes_g.append(mae)
        rmses_g.append(rmse)
        r2s_g.append(r2)
        runtimes_g.append(runtime_s)
        fold_rows.append(
            {
                "evaluation_scheme": "group_shuffle",
                "feature_set": feature_set_name,
                "model": model_key,
                "fold": fold_idx,
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                "mae": mae,
                "rmse": rmse,
                "r2": r2,
                "fit_predict_seconds": runtime_s,
            }
        )

    summary_rows.append(
        {
            "evaluation_scheme": "group_shuffle",
            "feature_set": feature_set_name,
            "model": model_key,
            "n_features": int(len(features)),
            "n_folds": int(len(maes_g)),
            "mae_mean": float(np.mean(maes_g)),
            "mae_std": float(np.std(maes_g)),
            "rmse_mean": float(np.mean(rmses_g)),
            "rmse_std": float(np.std(rmses_g)),
            "r2_mean": float(np.mean(r2s_g)),
            "r2_std": float(np.std(r2s_g)),
            "fit_predict_seconds_mean": float(np.mean(runtimes_g)),
        }
    )

    # Secondary benchmark 2: standard random row-level shuffled holdouts.
    rss = ShuffleSplit(
        n_splits=random_shuffle_splits,
        test_size=test_size,
        random_state=random_state,
    )
    maes_r: list[float] = []
    rmses_r: list[float] = []
    r2s_r: list[float] = []
    runtimes_r: list[float] = []
    for fold_idx, (train_idx, test_idx) in enumerate(rss.split(x, y), start=1):
        x_train = x.iloc[train_idx].copy()
        x_test = x.iloc[test_idx].copy()
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        pipeline = Pipeline(
            [
                ("preprocess", mc.make_preprocessor(x_train)),
                ("model", _build_model(model_key, params=params, random_state=random_state)),
            ]
        )
        t0 = time.perf_counter()
        pipeline.fit(x_train, y_train)
        preds = pipeline.predict(x_test)
        runtime_s = float(time.perf_counter() - t0)

        mae = float(mean_absolute_error(y_test, preds))
        rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
        r2 = float(r2_score(y_test, preds))

        maes_r.append(mae)
        rmses_r.append(rmse)
        r2s_r.append(r2)
        runtimes_r.append(runtime_s)
        fold_rows.append(
            {
                "evaluation_scheme": "random_shuffle",
                "feature_set": feature_set_name,
                "model": model_key,
                "fold": fold_idx,
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                "mae": mae,
                "rmse": rmse,
                "r2": r2,
                "fit_predict_seconds": runtime_s,
            }
        )

    summary_rows.append(
        {
            "evaluation_scheme": "random_shuffle",
            "feature_set": feature_set_name,
            "model": model_key,
            "n_features": int(len(features)),
            "n_folds": int(len(maes_r)),
            "mae_mean": float(np.mean(maes_r)),
            "mae_std": float(np.std(maes_r)),
            "rmse_mean": float(np.mean(rmses_r)),
            "rmse_std": float(np.std(rmses_r)),
            "r2_mean": float(np.mean(r2s_r)),
            "r2_std": float(np.std(r2s_r)),
            "fit_predict_seconds_mean": float(np.mean(runtimes_r)),
        }
    )

    return pd.DataFrame(fold_rows), pd.DataFrame(summary_rows)


def wrapper_forward_selection(
    frame: pd.DataFrame,
    core_features: list[str],
    review_features: list[str],
    model_key: str,
    seed: int,
    min_gain: float,
    max_added: int,
    n_splits: int,
) -> dict:
    selected = list(core_features)
    remaining = [f for f in review_features if f not in selected]
    base_metrics = evaluate_groupkfold(
        frame=frame,
        features=selected,
        model_key=model_key,
        params={},
        random_state=seed,
        n_splits=n_splits,
    )
    best_r2 = float(base_metrics["r2_mean"])
    added: list[str] = []

    while remaining and len(added) < max_added:
        best_feature: str | None = None
        best_metrics: dict | None = None
        candidate_best = best_r2

        for feature in remaining:
            candidate_features = selected + [feature]
            metrics = evaluate_groupkfold(
                frame=frame,
                features=candidate_features,
                model_key=model_key,
                params={},
                random_state=seed,
                n_splits=n_splits,
            )
            if float(metrics["r2_mean"]) > candidate_best:
                candidate_best = float(metrics["r2_mean"])
                best_feature = feature
                best_metrics = metrics

        if best_feature is None:
            break
        gain = candidate_best - best_r2
        if gain < min_gain:
            break

        selected.append(best_feature)
        remaining.remove(best_feature)
        added.append(best_feature)
        best_r2 = candidate_best
        if best_metrics is not None:
            base_metrics = best_metrics

    return {
        "model": model_key,
        "seed": seed,
        "selected_features": selected,
        "added_features": added,
        "base_feature_count": len(core_features),
        "added_count": len(added),
        "final_feature_count": len(selected),
        "final_cv_r2_mean": float(base_metrics["r2_mean"]),
        "final_cv_rmse_mean": float(base_metrics["rmse_mean"]),
        "final_cv_mae_mean": float(base_metrics["mae_mean"]),
    }


def build_stability_table(
    wrapper_runs: list[dict],
    all_candidate_features: list[str],
    threshold: float,
) -> tuple[pd.DataFrame, list[str]]:
    total_runs = len(wrapper_runs)
    if total_runs == 0:
        raise ValueError("No wrapper runs available for stability aggregation.")

    counts = {f: 0 for f in all_candidate_features}
    for run in wrapper_runs:
        for feature in run["selected_features"]:
            if feature in counts:
                counts[feature] += 1

    rows: list[dict] = []
    stable_features: list[str] = []
    for feature in all_candidate_features:
        selected_count = counts.get(feature, 0)
        rate = selected_count / total_runs
        stable_flag = rate >= threshold
        if stable_flag:
            stable_features.append(feature)
        rows.append(
            {
                "feature": feature,
                "selected_count": selected_count,
                "total_runs": total_runs,
                "selection_rate": rate,
                "stable_flag": stable_flag,
            }
        )

    out = pd.DataFrame(rows).sort_values(["selection_rate", "feature"], ascending=[False, True])
    return out, mc.dedupe_keep_order(stable_features)


def build_scenarios(
    frame: pd.DataFrame,
    core_features: list[str],
    stable_features: list[str],
    review_features: list[str],
    include_upper_bound: bool,
) -> dict[str, list[str]]:
    scenarios = {
        "core": mc.filter_available_features(core_features, frame),
        "stable": mc.filter_available_features(stable_features, frame),
        "full_review": mc.filter_available_features(core_features + review_features, frame),
    }
    if include_upper_bound:
        reserved = {RAW_TARGET_COL, TARGET_COL, GROUP_COL}
        upper = [c for c in frame.columns if c not in reserved]
        scenarios["upper_bound_all_non_target"] = mc.filter_available_features(upper, frame)
    return scenarios


def _param_space(model_key: str) -> dict:
    if model_key == "random_forest":
        return {
            "n_estimators": [250, 400, 600, 800],
            "max_depth": [None, 6, 10, 14],
            "min_samples_leaf": [1, 2, 4, 8],
            "max_features": ["sqrt", 0.7, 1.0],
        }
    if model_key == "lightgbm":
        return {
            "n_estimators": [250, 400, 600, 800],
            "learning_rate": [0.02, 0.04, 0.06, 0.1],
            "num_leaves": [15, 31, 63, 127],
            "min_child_samples": [5, 10, 20, 30],
            "subsample": [0.7, 0.85, 1.0],
            "colsample_bytree": [0.7, 0.85, 1.0],
            "reg_lambda": [0.0, 0.5, 1.0, 2.0],
            "verbosity": [-1],
        }
    if model_key == "xgboost":
        return {
            "n_estimators": [250, 400, 600, 800],
            "learning_rate": [0.02, 0.04, 0.06, 0.1],
            "max_depth": [3, 4, 6, 8],
            "min_child_weight": [1, 3, 5],
            "subsample": [0.7, 0.85, 1.0],
            "colsample_bytree": [0.7, 0.85, 1.0],
            "reg_lambda": [0.5, 1.0, 2.0, 5.0],
        }
    if model_key == "catboost":
        return {
            "iterations": [300, 500, 700, 900],
            "learning_rate": [0.02, 0.04, 0.06, 0.1],
            "depth": [4, 6, 8, 10],
            "l2_leaf_reg": [1, 3, 5, 7, 9],
            "bagging_temperature": [0.0, 0.5, 1.0, 2.0],
            "random_strength": [0.0, 1.0, 2.0],
        }
    raise ValueError(f"Unsupported model key: {model_key}")


def tune_model_on_scenario(
    frame: pd.DataFrame,
    scenario_name: str,
    features: list[str],
    model_key: str,
    n_iter: int,
    seed: int,
    n_splits: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    sampler = ParameterSampler(_param_space(model_key), n_iter=n_iter, random_state=seed)
    trial_rows: list[dict] = []
    best: dict[str, Any] | None = None

    for idx, params in enumerate(sampler, start=1):
        metrics = evaluate_groupkfold(
            frame=frame,
            features=features,
            model_key=model_key,
            params=params,
            random_state=seed,
            n_splits=n_splits,
        )
        row = {
            "scenario": scenario_name,
            "model": model_key,
            "trial": idx,
            "n_features": len(features),
            "params_json": json.dumps(params, sort_keys=True),
            **metrics,
        }
        trial_rows.append(row)
        if best is None:
            best = {**row, "params": params}
            continue
        row_r2 = float(row["r2_mean"])
        row_rmse = float(row["rmse_mean"])
        best_r2 = float(best["r2_mean"])
        best_rmse = float(best["rmse_mean"])
        if row_r2 > best_r2 or (np.isclose(row_r2, best_r2) and row_rmse < best_rmse):
            best = {**row, "params": params}

    if best is None:
        raise RuntimeError("No tuning trials executed.")

    return pd.DataFrame(trial_rows), best


def run(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    run_dir = OUT_ROOT / args.run_tag
    run_dir.mkdir(parents=True, exist_ok=True)

    frame = mc.load_analysis_frame()
    candidates_df = mc.load_hybrid_candidates(auto_run_feature_prepare=args.auto_run_feature_prepare)
    core_features, review_features = core_and_review_features(candidates_df)
    core_features = mc.filter_available_features(core_features, frame)
    review_features = mc.filter_available_features(review_features, frame)

    wrapper_models = parse_csv_list(args.wrapper_models)
    wrapper_seeds = parse_int_list(args.wrapper_seeds)
    tune_models = parse_csv_list(args.tune_models)

    wrapper_runs: list[dict] = []
    print("\nWrapper selection runs")
    for model_key in wrapper_models:
        for seed in wrapper_seeds:
            result = wrapper_forward_selection(
                frame=frame,
                core_features=core_features,
                review_features=review_features,
                model_key=model_key,
                seed=seed,
                min_gain=args.wrapper_min_gain,
                max_added=args.wrapper_max_added,
                n_splits=args.groupkfold_splits,
            )
            wrapper_runs.append(result)
            print(
                f"  {model_key} seed={seed}: "
                f"final_r2={result['final_cv_r2_mean']:.4f} "
                f"| selected={result['final_feature_count']} "
                f"| added={result['added_count']}"
            )

    wrapper_df = pd.DataFrame(
        [
            {
                "model": r["model"],
                "seed": r["seed"],
                "base_feature_count": r["base_feature_count"],
                "added_count": r["added_count"],
                "final_feature_count": r["final_feature_count"],
                "final_cv_r2_mean": r["final_cv_r2_mean"],
                "final_cv_rmse_mean": r["final_cv_rmse_mean"],
                "final_cv_mae_mean": r["final_cv_mae_mean"],
                "selected_features": "|".join(r["selected_features"]),
                "added_features": "|".join(r["added_features"]),
            }
            for r in wrapper_runs
        ]
    )
    wrapper_df.to_csv(run_dir / "wrapper_runs.csv", index=False)

    candidate_pool = mc.dedupe_keep_order(core_features + review_features)
    stability_df, stable_features = build_stability_table(
        wrapper_runs=wrapper_runs,
        all_candidate_features=candidate_pool,
        threshold=args.stability_threshold,
    )
    stability_df.to_csv(run_dir / "stability_selection_frequency.csv", index=False)
    stable_features = mc.filter_available_features(stable_features, frame)

    scenarios = build_scenarios(
        frame=frame,
        core_features=core_features,
        stable_features=stable_features,
        review_features=review_features,
        include_upper_bound=args.include_upper_bound_scenario,
    )
    scenario_rows: list[dict] = []
    for scenario_name, feature_list in scenarios.items():
        for rank, feature in enumerate(feature_list, start=1):
            scenario_rows.append(
                {
                    "scenario": scenario_name,
                    "rank_in_scenario": rank,
                    "feature": feature,
                }
            )
    scenario_df = pd.DataFrame(scenario_rows)
    scenario_df.to_csv(run_dir / "scenario_feature_sets.csv", index=False)

    trial_tables: list[pd.DataFrame] = []
    best_rows: list[dict] = []

    print("\nTuning runs (GroupKFold)")
    for scenario_name, feature_list in scenarios.items():
        for model_key in tune_models:
            trials_df, best = tune_model_on_scenario(
                frame=frame,
                scenario_name=scenario_name,
                features=feature_list,
                model_key=model_key,
                n_iter=args.tune_n_iter,
                seed=args.random_state,
                n_splits=args.groupkfold_splits,
            )
            trial_tables.append(trials_df)
            best_rows.append(best)
            print(f"  {scenario_name} | {model_key}: best_r2={best['r2_mean']:.4f} rmse={best['rmse_mean']:.2f}")

    trials_out = pd.concat(trial_tables, ignore_index=True)
    best_df = pd.DataFrame(best_rows).sort_values(["r2_mean", "rmse_mean"], ascending=[False, True])
    trials_out.to_csv(run_dir / "tuning_trials.csv", index=False)
    best_df.drop(columns=["params"], errors="ignore").to_csv(run_dir / "tuned_groupkfold_summary.csv", index=False)

    logo_fold_tables: list[pd.DataFrame] = []
    logo_summary_rows: list[dict] = []
    print("\nFinal LOGO evaluation (best tuned configs)")
    for _, row in best_df.iterrows():
        fold_df, summary = evaluate_logo(
            frame=frame,
            feature_set_name=row["scenario"],
            features=scenarios[row["scenario"]],
            model_key=row["model"],
            params=row["params"],
            random_state=args.random_state,
        )
        summary["params_json"] = json.dumps(row["params"], sort_keys=True)
        logo_fold_tables.append(fold_df)
        logo_summary_rows.append(summary)
        print(f"  {row['scenario']} | {row['model']}: LOGO r2={summary['r2_mean']:.4f} rmse={summary['rmse_mean']:.2f}")

    logo_folds_df = pd.concat(logo_fold_tables, ignore_index=True)
    logo_summary_df = pd.DataFrame(logo_summary_rows).sort_values(["r2_mean", "rmse_mean"], ascending=[False, True])
    logo_folds_df.to_csv(run_dir / "logo_fold_metrics.csv", index=False)
    logo_summary_df.to_csv(run_dir / "logo_summary.csv", index=False)

    # Secondary metrics (less strict) for operational reporting; LOGO remains primary.
    secondary_fold_tables: list[pd.DataFrame] = []
    secondary_summary_tables: list[pd.DataFrame] = []
    print("\nSecondary evaluation (GroupShuffle + RandomShuffle)")
    for _, row in best_df.iterrows():
        sec_folds_df, sec_summary_df = evaluate_secondary_splits(
            frame=frame,
            feature_set_name=row["scenario"],
            features=scenarios[row["scenario"]],
            model_key=row["model"],
            params=row["params"],
            random_state=args.random_state,
            group_shuffle_splits=args.secondary_group_shuffle_splits,
            random_shuffle_splits=args.secondary_random_shuffle_splits,
            test_size=args.secondary_test_size,
        )
        secondary_fold_tables.append(sec_folds_df)
        secondary_summary_tables.append(sec_summary_df)

    secondary_folds_df = pd.concat(secondary_fold_tables, ignore_index=True)
    secondary_summary_df = pd.concat(secondary_summary_tables, ignore_index=True)
    secondary_summary_df = secondary_summary_df.sort_values(
        ["evaluation_scheme", "r2_mean", "rmse_mean"],
        ascending=[True, False, True],
    )
    secondary_folds_df.to_csv(run_dir / "secondary_split_fold_metrics.csv", index=False)
    secondary_summary_df.to_csv(run_dir / "secondary_split_summary.csv", index=False)
    for scheme in ["group_shuffle", "random_shuffle"]:
        top = secondary_summary_df[secondary_summary_df["evaluation_scheme"] == scheme].iloc[0]
        print(
            f"  {scheme}: best={top['feature_set']} | {top['model']} "
            f"| r2={top['r2_mean']:.4f} | rmse={top['rmse_mean']:.2f}"
        )

    best_logo = logo_summary_df.iloc[0]
    print("\nBest Step 3 result")
    print(
        f"  scenario={best_logo['feature_set']} | model={best_logo['model']} "
        f"| r2={best_logo['r2_mean']:.4f} | rmse={best_logo['rmse_mean']:.2f} "
        f"| mae={best_logo['mae_mean']:.2f}"
    )
    print(f"  target_r2={args.target_r2:.2f} | gap={args.target_r2 - best_logo['r2_mean']:.4f}")
    print(f"Saved artifacts to: {run_dir}")

    return best_df, logo_summary_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step 3 wrapper/stability/tuning pipeline.")
    parser.add_argument(
        "--run-tag", type=str, default="latest", help="Output subfolder name under outputs/model_select_tune."
    )
    parser.add_argument("--auto-run-feature-prepare", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--wrapper-models", type=str, default="lightgbm,random_forest")
    parser.add_argument("--wrapper-seeds", type=str, default="42,7,123")
    parser.add_argument("--wrapper-min-gain", type=float, default=0.0015)
    parser.add_argument("--wrapper-max-added", type=int, default=6)
    parser.add_argument("--stability-threshold", type=float, default=0.60)
    parser.add_argument("--include-upper-bound-scenario", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--tune-models", type=str, default="catboost,lightgbm,xgboost,random_forest")
    parser.add_argument("--tune-n-iter", type=int, default=12)
    parser.add_argument("--groupkfold-splits", type=int, default=3)
    parser.add_argument("--target-r2", type=float, default=0.70)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--secondary-group-shuffle-splits", type=int, default=8)
    parser.add_argument("--secondary-random-shuffle-splits", type=int, default=8)
    parser.add_argument("--secondary-test-size", type=float, default=0.2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
