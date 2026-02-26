"""Trial-aware model comparison on CAROB using a locked feature scenario."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from paddy_yield_ml.pipelines import carob_common as cc

try:
    project_root = Path(__file__).resolve().parents[3]
except NameError:
    project_root = Path.cwd()

OUT_DIR = project_root / "outputs" / "carob_model_compare"
CANDIDATES_PATH = project_root / "outputs" / "carob_feature_prepare" / "hybrid_selection_candidates.csv"
DEFAULT_SCENARIO = "modifiable_plus_context"
DEFAULT_MODELS = "random_forest,extra_trees,catboost,xgboost,lightgbm"
DEFAULT_TEST_SIZE = 0.2


@dataclass(frozen=True)
class ModelSpec:
    name: str
    key: str
    param_grid: list[dict[str, object]]
    build: Callable[[dict[str, object]], object]


def dedupe_keep_order(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            out.append(item)
            seen.add(item)
    return out


def make_preprocessor(x_train: pd.DataFrame) -> ColumnTransformer:
    num_cols = x_train.select_dtypes(include=[np.number, "boolean"]).columns.tolist()
    cat_cols = [c for c in x_train.columns if c not in num_cols]

    transformers: list[tuple[str, object, list[str]]] = []
    if num_cols:
        transformers.append(("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), num_cols))
    if cat_cols:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                cat_cols,
            )
        )
    if not transformers:
        raise ValueError("No valid numeric/categorical columns for preprocessing.")
    return ColumnTransformer(transformers=transformers, remainder="drop")


def load_hybrid_candidates(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing candidate file: {path}. Run carob_feature_prepare first.")
    cdf = pd.read_csv(path)
    req = {"feature", "status"}
    missing = req - set(cdf.columns)
    if missing:
        raise ValueError(f"Candidate file missing columns: {sorted(missing)}")
    return cdf


def build_feature_sets(candidates_df: pd.DataFrame) -> dict[str, list[str]]:
    cdf = candidates_df.copy()
    if "hybrid_priority_score" in cdf.columns:
        cdf = cdf.sort_values("hybrid_priority_score", ascending=False, na_position="last")

    modifiable = cdf.loc[cdf["status"] == "candidate_modifiable", "feature"].astype(str).tolist()
    redundant = cdf.loc[cdf["status"] == "candidate_redundant_review", "feature"].astype(str).tolist()
    context = cdf.loc[cdf["status"] == "reserve_context", "feature"].astype(str).tolist()

    modifiable = dedupe_keep_order(modifiable)
    redundant = dedupe_keep_order(redundant)
    context = dedupe_keep_order(context)

    if not modifiable:
        raise ValueError("No candidate_modifiable features found.")

    return {
        "modifiable_only": modifiable,
        "modifiable_plus_context": dedupe_keep_order(modifiable + context),
        "hybrid_with_review": dedupe_keep_order(modifiable + context + redundant),
    }


def filter_available_features(features: Iterable[str], frame: pd.DataFrame) -> list[str]:
    reserved = {cc.TARGET_COL, cc.GROUP_COL}
    reserved |= cc.IDENTIFIER_COLS
    reserved |= cc.POST_OUTCOME_COLS
    valid = [f for f in dedupe_keep_order(features) if f in frame.columns and f not in reserved]
    if not valid:
        raise ValueError("No valid features after filtering.")
    return valid


def parse_model_list(raw: str) -> list[str]:
    models = [m.strip().lower() for m in raw.split(",") if m.strip()]
    if not models:
        raise ValueError("At least one model must be selected.")
    return dedupe_keep_order(models)


def load_frame_with_country_gate(candidates_path: Path) -> pd.DataFrame:
    frame = cc.load_analysis_frame(require_treatment=True)
    audit_path = candidates_path.parent / "country_exclusion_audit.csv"
    if audit_path.exists():
        audit = pd.read_csv(audit_path)
        if "group_value" in audit.columns and "drop_group" in audit.columns:
            if "group_column" in audit.columns:
                audit = audit[audit["group_column"].astype(str) == "country"].copy()
            dropped = set(audit.loc[audit["drop_group"].astype(bool), "group_value"].astype(str))
            if dropped and "country" in frame.columns:
                frame = frame[~frame["country"].astype(str).isin(dropped)].reset_index(drop=True)

    trial_audit_path = candidates_path.parent / "trial_exclusion_audit.csv"
    if not trial_audit_path.exists():
        return frame

    trial_audit = pd.read_csv(trial_audit_path)
    if "group_value" not in trial_audit.columns or "drop_group" not in trial_audit.columns:
        return frame

    if "group_column" in trial_audit.columns:
        trial_audit = trial_audit[trial_audit["group_column"].astype(str) == cc.GROUP_COL].copy()

    dropped_trials = set(trial_audit.loc[trial_audit["drop_group"].astype(bool), "group_value"].astype(str))
    if not dropped_trials or cc.GROUP_COL not in frame.columns:
        return frame
    return frame[~frame[cc.GROUP_COL].astype(str).isin(dropped_trials)].reset_index(drop=True)


def build_model_specs(model_names: list[str], random_state: int) -> list[ModelSpec]:
    specs: list[ModelSpec] = []
    for name in model_names:
        if name == "random_forest":
            grid = [
                {"n_estimators": 300, "max_depth": None, "min_samples_leaf": 2},
                {"n_estimators": 500, "max_depth": None, "min_samples_leaf": 1},
                {"n_estimators": 500, "max_depth": 12, "min_samples_leaf": 2},
            ]
            specs.append(
                ModelSpec(
                    name="RandomForest",
                    key=name,
                    param_grid=grid,
                    build=lambda p, rs=random_state: RandomForestRegressor(
                        n_estimators=int(p["n_estimators"]),
                        max_depth=(None if p["max_depth"] is None else int(p["max_depth"])),
                        min_samples_leaf=int(p["min_samples_leaf"]),
                        random_state=rs,
                        n_jobs=-1,
                    ),
                )
            )
            continue

        if name == "extra_trees":
            grid = [
                {"n_estimators": 300, "max_depth": None, "min_samples_leaf": 2},
                {"n_estimators": 500, "max_depth": None, "min_samples_leaf": 1},
                {"n_estimators": 500, "max_depth": 12, "min_samples_leaf": 2},
            ]
            specs.append(
                ModelSpec(
                    name="ExtraTrees",
                    key=name,
                    param_grid=grid,
                    build=lambda p, rs=random_state: ExtraTreesRegressor(
                        n_estimators=int(p["n_estimators"]),
                        max_depth=(None if p["max_depth"] is None else int(p["max_depth"])),
                        min_samples_leaf=int(p["min_samples_leaf"]),
                        random_state=rs,
                        n_jobs=-1,
                    ),
                )
            )
            continue

        if name == "catboost":
            from catboost import CatBoostRegressor

            grid = [
                {"n_estimators": 300, "learning_rate": 0.05, "depth": 6, "l2_leaf_reg": 3.0},
                {"n_estimators": 500, "learning_rate": 0.03, "depth": 6, "l2_leaf_reg": 5.0},
                {"n_estimators": 500, "learning_rate": 0.05, "depth": 4, "l2_leaf_reg": 3.0},
            ]
            specs.append(
                ModelSpec(
                    name="CatBoost",
                    key=name,
                    param_grid=grid,
                    build=lambda p, rs=random_state: CatBoostRegressor(
                        n_estimators=int(p["n_estimators"]),
                        learning_rate=float(p["learning_rate"]),
                        depth=int(p["depth"]),
                        l2_leaf_reg=float(p["l2_leaf_reg"]),
                        loss_function="RMSE",
                        random_seed=rs,
                        verbose=0,
                    ),
                )
            )
            continue

        if name == "xgboost":
            from xgboost import XGBRegressor

            grid = [
                {"n_estimators": 300, "learning_rate": 0.05, "max_depth": 6, "subsample": 0.9, "colsample_bytree": 0.9},
                {"n_estimators": 500, "learning_rate": 0.03, "max_depth": 6, "subsample": 0.9, "colsample_bytree": 0.9},
                {"n_estimators": 500, "learning_rate": 0.05, "max_depth": 4, "subsample": 1.0, "colsample_bytree": 1.0},
            ]
            specs.append(
                ModelSpec(
                    name="XGBoost",
                    key=name,
                    param_grid=grid,
                    build=lambda p, rs=random_state: XGBRegressor(
                        n_estimators=int(p["n_estimators"]),
                        learning_rate=float(p["learning_rate"]),
                        max_depth=int(p["max_depth"]),
                        subsample=float(p["subsample"]),
                        colsample_bytree=float(p["colsample_bytree"]),
                        objective="reg:squarederror",
                        random_state=rs,
                        n_jobs=-1,
                        tree_method="hist",
                    ),
                )
            )
            continue

        if name == "lightgbm":
            from lightgbm import LGBMRegressor

            grid = [
                {"n_estimators": 300, "learning_rate": 0.05, "num_leaves": 31, "min_child_samples": 20},
                {"n_estimators": 500, "learning_rate": 0.03, "num_leaves": 31, "min_child_samples": 20},
                {"n_estimators": 500, "learning_rate": 0.05, "num_leaves": 63, "min_child_samples": 30},
            ]
            specs.append(
                ModelSpec(
                    name="LightGBM",
                    key=name,
                    param_grid=grid,
                    build=lambda p, rs=random_state: LGBMRegressor(
                        n_estimators=int(p["n_estimators"]),
                        learning_rate=float(p["learning_rate"]),
                        num_leaves=int(p["num_leaves"]),
                        min_child_samples=int(p["min_child_samples"]),
                        random_state=rs,
                        n_jobs=-1,
                        verbosity=-1,
                    ),
                )
            )
            continue

        raise ValueError(
            "Unsupported model. Use random_forest, extra_trees, catboost, xgboost, and/or lightgbm."
        )
    return specs


def build_trial_aware_split_indices(
    groups: pd.Series,
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    if test_size <= 0.0 or test_size >= 1.0:
        raise ValueError("test_size must be in (0, 1).")

    g = groups.astype(str).reset_index(drop=True)
    rng = np.random.default_rng(random_state)
    train_idx: list[int] = []
    test_idx: list[int] = []

    grouped = g.groupby(g, sort=True).indices
    for _, idx_list in grouped.items():
        arr = np.array(sorted(idx_list), dtype=int)
        rng.shuffle(arr)
        n = len(arr)
        if n < 2:
            train_idx.extend(arr.tolist())
            continue
        n_test = int(np.floor(test_size * n))
        n_test = min(max(n_test, 1), n - 1)
        test_idx.extend(arr[:n_test].tolist())
        train_idx.extend(arr[n_test:].tolist())

    tr = np.array(sorted(train_idx), dtype=int)
    te = np.array(sorted(test_idx), dtype=int)
    if len(te) == 0:
        raise ValueError("Trial-aware split produced an empty test set.")
    return tr, te


def impute_numeric_by_group(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    groups_train: pd.Series,
    groups_test: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Impute numeric missing values with train-only group medians, fallback to train global median."""
    xtr = x_train.copy()
    xte = x_test.copy()

    num_cols = xtr.select_dtypes(include=[np.number]).columns.tolist()
    if not num_cols:
        return xtr, xte

    gtr = groups_train.astype(str)
    gte = groups_test.astype(str)

    train_num = xtr[num_cols].apply(pd.to_numeric, errors="coerce")
    group_medians = train_num.groupby(gtr).median(numeric_only=True)
    global_medians = train_num.median(numeric_only=True)

    for col in num_cols:
        if col not in group_medians.columns:
            xtr[col] = train_num[col].fillna(global_medians.get(col, np.nan))
            xte[col] = pd.to_numeric(xte[col], errors="coerce").fillna(global_medians.get(col, np.nan))
            continue

        group_med_col = group_medians[col]
        train_group_fill = gtr.map(group_med_col)
        test_group_fill = gte.map(group_med_col)
        fallback = global_medians.get(col, np.nan)

        xtr[col] = train_num[col].fillna(train_group_fill).fillna(fallback)
        xte_num = pd.to_numeric(xte[col], errors="coerce")
        xte[col] = xte_num.fillna(test_group_fill).fillna(fallback)

    return xtr, xte


def evaluate_trial_aware(
    frame: pd.DataFrame,
    features: list[str],
    model_spec: ModelSpec,
    params: dict[str, object],
    *,
    test_size: float,
    random_state: int,
    trial_median_impute: bool = True,
) -> tuple[pd.DataFrame, dict[str, object]]:
    subset = frame[features + [cc.TARGET_COL, cc.GROUP_COL]].copy()
    subset[cc.TARGET_COL] = pd.to_numeric(subset[cc.TARGET_COL], errors="coerce")
    subset = subset[subset[cc.TARGET_COL].notna() & subset[cc.GROUP_COL].notna()].reset_index(drop=True)

    x = subset[features].copy()
    y = subset[cc.TARGET_COL].copy()
    groups = subset[cc.GROUP_COL].astype(str)

    tr_idx, te_idx = build_trial_aware_split_indices(groups=groups, test_size=test_size, random_state=random_state)
    xtr, xte = x.iloc[tr_idx].copy(), x.iloc[te_idx].copy()
    ytr, yte = y.iloc[tr_idx], y.iloc[te_idx]
    gtr, gte = groups.iloc[tr_idx].astype(str), groups.iloc[te_idx].astype(str)

    if trial_median_impute:
        xtr, xte = impute_numeric_by_group(
            x_train=xtr,
            x_test=xte,
            groups_train=gtr,
            groups_test=gte,
        )

    pipeline = Pipeline(
        [
            ("preprocess", make_preprocessor(xtr)),
            ("model", model_spec.build(params)),
        ]
    )

    t0 = time.perf_counter()
    pipeline.fit(xtr, ytr)
    fit_seconds = float(time.perf_counter() - t0)
    pred = pipeline.predict(xte)

    trial_rows: list[dict[str, object]] = []
    gte_reset = gte.reset_index(drop=True)
    yte_reset = yte.reset_index(drop=True)
    pred_s = pd.Series(pred)
    for trial in sorted(gte_reset.unique().tolist()):
        mask = gte_reset == trial
        y_true_t = yte_reset[mask]
        y_pred_t = pred_s[mask]
        trial_rows.append(
            {
                "trial_id": str(trial),
                "n_test": int(mask.sum()),
                "mae": float(mean_absolute_error(y_true_t, y_pred_t)),
                "rmse": float(np.sqrt(mean_squared_error(y_true_t, y_pred_t))),
                "r2": float(r2_score(y_true_t, y_pred_t)),
            }
        )

    summary = {
        "model": model_spec.name,
        "model_key": model_spec.key,
        "params_json": json.dumps(params, sort_keys=True),
        "n_features": int(len(features)),
        "n_trials_in_test": int(gte.nunique()),
        "n_train": int(len(tr_idx)),
        "n_test": int(len(te_idx)),
        "mae": float(mean_absolute_error(yte, pred)),
        "rmse": float(np.sqrt(mean_squared_error(yte, pred))),
        "r2": float(r2_score(yte, pred)),
        "fit_seconds": fit_seconds,
    }
    return pd.DataFrame(trial_rows), summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CAROB model comparison with trial-aware 80/20 split.")
    parser.add_argument("--scenario", type=str, default=DEFAULT_SCENARIO)
    parser.add_argument("--models", type=str, default=DEFAULT_MODELS)
    parser.add_argument("--test-size", type=float, default=DEFAULT_TEST_SIZE)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--trial-median-impute", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--candidates-path", type=str, default=str(CANDIDATES_PATH))
    parser.add_argument("--out-dir", type=str, default=str(OUT_DIR))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    candidates_path = Path(args.candidates_path)

    frame = load_frame_with_country_gate(candidates_path)
    candidates = load_hybrid_candidates(candidates_path)
    feature_sets = build_feature_sets(candidates)

    if args.scenario not in feature_sets:
        raise ValueError(f"Scenario '{args.scenario}' not found. Available: {sorted(feature_sets)}")

    features = filter_available_features(feature_sets[args.scenario], frame)
    scenario_feature_df = pd.DataFrame(
        [{"scenario": args.scenario, "rank_in_scenario": i, "feature": f} for i, f in enumerate(features, start=1)]
    )
    scenario_feature_df.to_csv(out_dir / "scenario_feature_sets.csv", index=False)

    model_specs = build_model_specs(parse_model_list(args.models), args.random_state)

    trial_all: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []
    for spec in model_specs:
        print(f"\nModel: {spec.name}")
        for i, params in enumerate(spec.param_grid, start=1):
            trial_df, summary = evaluate_trial_aware(
                frame=frame,
                features=features,
                model_spec=spec,
                params=params,
                test_size=float(args.test_size),
                random_state=int(args.random_state),
                trial_median_impute=bool(args.trial_median_impute),
            )
            trial_df.insert(0, "model", spec.name)
            trial_df.insert(1, "param_set", i)
            trial_df.insert(2, "scenario", args.scenario)
            summary["param_set"] = i
            summary["scenario"] = args.scenario
            trial_all.append(trial_df)
            summary_rows.append(summary)
            print(
                f"  set={i} | R2={summary['r2']:.4f} | RMSE={summary['rmse']:.2f} | MAE={summary['mae']:.2f} "
                f"| params={params}"
            )

    trial_metrics = pd.concat(trial_all, ignore_index=True)
    summary_df = pd.DataFrame(summary_rows).sort_values(["r2", "rmse"], ascending=[False, True]).reset_index(drop=True)

    trial_metrics.to_csv(out_dir / "trial_aware_trial_metrics.csv", index=False)
    summary_df.to_csv(out_dir / "model_comparison_summary.csv", index=False)

    best = summary_df.iloc[0]
    print("\nBest configuration:")
    print(best.to_string())
    print(f"\nSaved CAROB model-compare outputs to: {out_dir}")


if __name__ == "__main__":
    main()
