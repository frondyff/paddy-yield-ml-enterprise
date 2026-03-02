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
DEFAULT_VAL_SIZE = 0.2


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
    """Backward-compatible train/test split helper."""
    tr, _val, te = build_trial_aware_train_val_test_indices(
        groups=groups,
        test_size=test_size,
        val_size=DEFAULT_VAL_SIZE,
        random_state=random_state,
    )
    return tr, te


def build_trial_aware_train_val_test_indices(
    groups: pd.Series,
    *,
    test_size: float = DEFAULT_TEST_SIZE,
    val_size: float = DEFAULT_VAL_SIZE,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build train/validation/test indices with within-trial random splits."""
    if test_size <= 0.0 or test_size >= 1.0:
        raise ValueError("test_size must be in (0, 1).")
    if val_size <= 0.0 or val_size >= 1.0:
        raise ValueError("val_size must be in (0, 1).")
    if test_size + val_size >= 1.0:
        raise ValueError("test_size + val_size must be < 1.")

    g = groups.astype(str).reset_index(drop=True)
    rng = np.random.default_rng(random_state)

    train_idx: list[int] = []
    val_idx: list[int] = []
    test_idx: list[int] = []

    val_share_remaining = float(val_size / (1.0 - test_size))
    grouped = g.groupby(g, sort=True).indices

    for _, idx_list in grouped.items():
        arr = np.array(sorted(idx_list), dtype=int)
        rng.shuffle(arr)
        n = len(arr)

        if n < 2:
            train_idx.extend(arr.tolist())
            continue

        if n == 2:
            test_idx.append(int(arr[0]))
            train_idx.append(int(arr[1]))
            continue

        n_test = int(np.floor(test_size * n))
        n_test = min(max(n_test, 1), n - 2)
        remaining = n - n_test

        n_val = int(np.floor(val_share_remaining * remaining))
        n_val = min(max(n_val, 1), remaining - 1)

        test_idx.extend(arr[:n_test].tolist())
        val_idx.extend(arr[n_test : n_test + n_val].tolist())
        train_idx.extend(arr[n_test + n_val :].tolist())

    tr = np.array(sorted(train_idx), dtype=int)
    va = np.array(sorted(val_idx), dtype=int)
    te = np.array(sorted(test_idx), dtype=int)

    if len(tr) == 0:
        raise ValueError("Trial-aware split produced an empty train set.")
    if len(va) == 0:
        raise ValueError("Trial-aware split produced an empty validation set.")
    if len(te) == 0:
        raise ValueError("Trial-aware split produced an empty test set.")
    return tr, va, te


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


def prepare_modeling_subset(frame: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    subset = frame[features + [cc.TARGET_COL, cc.GROUP_COL]].copy()
    subset[cc.TARGET_COL] = pd.to_numeric(subset[cc.TARGET_COL], errors="coerce")
    subset = subset[subset[cc.TARGET_COL].notna() & subset[cc.GROUP_COL].notna()].reset_index(drop=True)
    x = subset[features].copy()
    y = subset[cc.TARGET_COL].copy()
    groups = subset[cc.GROUP_COL].astype(str)
    return x, y, groups


def fit_and_predict(
    *,
    model_spec: ModelSpec,
    params: dict[str, object],
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_eval: pd.DataFrame,
    groups_train: pd.Series,
    groups_eval: pd.Series,
    trial_median_impute: bool,
) -> tuple[np.ndarray, float]:
    xtr = x_train.copy()
    xev = x_eval.copy()
    if trial_median_impute:
        xtr, xev = impute_numeric_by_group(
            x_train=xtr,
            x_test=xev,
            groups_train=groups_train.astype(str),
            groups_test=groups_eval.astype(str),
        )

    pipeline = Pipeline(
        [
            ("preprocess", make_preprocessor(xtr)),
            ("model", model_spec.build(params)),
        ]
    )
    t0 = time.perf_counter()
    pipeline.fit(xtr, y_train)
    fit_seconds = float(time.perf_counter() - t0)
    pred = pipeline.predict(xev)
    return pred, fit_seconds


def regression_metrics(y_true: pd.Series, pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, pred))),
        "r2": float(r2_score(y_true, pred)),
    }


def trial_level_metrics(
    *,
    groups_eval: pd.Series,
    y_true: pd.Series,
    pred: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    g = groups_eval.reset_index(drop=True).astype(str)
    yt = y_true.reset_index(drop=True)
    yp = pd.Series(pred)
    for trial in sorted(g.unique().tolist()):
        mask = g == trial
        yt_t = yt[mask]
        yp_t = yp[mask]
        rows.append(
            {
                "trial_id": str(trial),
                "n_test": int(mask.sum()),
                "mae": float(mean_absolute_error(yt_t, yp_t)),
                "rmse": float(np.sqrt(mean_squared_error(yt_t, yp_t))),
                "r2": float(r2_score(yt_t, yp_t)),
            }
        )
    return pd.DataFrame(rows)


def split_audit_rows(
    *,
    groups: pd.Series,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
) -> pd.DataFrame:
    g = groups.astype(str).reset_index(drop=True)
    rows: list[dict[str, object]] = []
    idx_map = [
        ("train", train_idx),
        ("validation", val_idx),
        ("test", test_idx),
    ]
    for split_name, idx in idx_map:
        for i in idx.tolist():
            rows.append({"row_index": int(i), "trial_id": str(g.iloc[i]), "split": split_name})
    return pd.DataFrame(rows).sort_values(["split", "trial_id", "row_index"]).reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CAROB model comparison with trial-aware train/validation/test split."
    )
    parser.add_argument("--scenario", type=str, default=DEFAULT_SCENARIO)
    parser.add_argument("--models", type=str, default=DEFAULT_MODELS)
    parser.add_argument("--test-size", type=float, default=DEFAULT_TEST_SIZE)
    parser.add_argument("--val-size", type=float, default=DEFAULT_VAL_SIZE)
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

    x, y, groups = prepare_modeling_subset(frame, features)
    tr_idx, va_idx, te_idx = build_trial_aware_train_val_test_indices(
        groups=groups,
        test_size=float(args.test_size),
        val_size=float(args.val_size),
        random_state=int(args.random_state),
    )
    xtr, xva, xte = x.iloc[tr_idx].copy(), x.iloc[va_idx].copy(), x.iloc[te_idx].copy()
    ytr, yva, yte = y.iloc[tr_idx].copy(), y.iloc[va_idx].copy(), y.iloc[te_idx].copy()
    gtr, gva, gte = groups.iloc[tr_idx].copy(), groups.iloc[va_idx].copy(), groups.iloc[te_idx].copy()

    split_rows = split_audit_rows(groups=groups, train_idx=tr_idx, val_idx=va_idx, test_idx=te_idx)
    split_rows.to_csv(out_dir / "within_trial_split_definition.csv", index=False)
    split_summary = (
        split_rows.groupby("split", dropna=False)
        .agg(n_rows=("row_index", "size"), n_trials=("trial_id", "nunique"))
        .reset_index()
    )
    split_summary.to_csv(out_dir / "train_validate_test_split_summary.csv", index=False)
    split_trial_counts = (
        split_rows.groupby(["trial_id", "split"], dropna=False)
        .size()
        .unstack(fill_value=0)
        .reset_index()
        .sort_values("trial_id")
        .reset_index(drop=True)
    )
    split_trial_counts.to_csv(out_dir / "train_validate_test_trial_counts.csv", index=False)

    model_specs = build_model_specs(parse_model_list(args.models), args.random_state)

    validation_rows: list[dict[str, object]] = []
    test_trial_all: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []
    for spec in model_specs:
        print(f"\nModel: {spec.name}")
        model_val_rows: list[dict[str, object]] = []
        for i, params in enumerate(spec.param_grid, start=1):
            pred_val, fit_seconds_val = fit_and_predict(
                model_spec=spec,
                params=params,
                x_train=xtr,
                y_train=ytr,
                x_eval=xva,
                groups_train=gtr,
                groups_eval=gva,
                trial_median_impute=bool(args.trial_median_impute),
            )
            m_val = regression_metrics(yva, pred_val)
            row = {
                "model": spec.name,
                "model_key": spec.key,
                "param_set": int(i),
                "params_json": json.dumps(params, sort_keys=True),
                "scenario": args.scenario,
                "n_features": int(len(features)),
                "n_train": int(len(tr_idx)),
                "n_validation": int(len(va_idx)),
                "n_test": int(len(te_idx)),
                "val_mae": m_val["mae"],
                "val_rmse": m_val["rmse"],
                "val_r2": m_val["r2"],
                "fit_seconds_validation": fit_seconds_val,
            }
            model_val_rows.append(row)
            validation_rows.append(row)
            print(
                f"  set={i} | VAL_R2={m_val['r2']:.4f} | VAL_RMSE={m_val['rmse']:.2f} "
                f"| VAL_MAE={m_val['mae']:.2f} | params={params}"
            )

        model_val_df = pd.DataFrame(model_val_rows).sort_values(
            ["val_r2", "val_rmse"],
            ascending=[False, True],
        )
        best_val = model_val_df.iloc[0]
        best_param_set = int(best_val["param_set"])
        best_params = json.loads(str(best_val["params_json"]))

        x_train_final = pd.concat([xtr, xva], axis=0, ignore_index=True)
        y_train_final = pd.concat([ytr, yva], axis=0, ignore_index=True)
        g_train_final = pd.concat([gtr, gva], axis=0, ignore_index=True)

        pred_test, fit_seconds_test = fit_and_predict(
            model_spec=spec,
            params=best_params,
            x_train=x_train_final,
            y_train=y_train_final,
            x_eval=xte,
            groups_train=g_train_final,
            groups_eval=gte,
            trial_median_impute=bool(args.trial_median_impute),
        )
        m_test = regression_metrics(yte, pred_test)
        trial_df = trial_level_metrics(groups_eval=gte, y_true=yte, pred=pred_test)
        trial_df.insert(0, "model", spec.name)
        trial_df.insert(1, "model_key", spec.key)
        trial_df.insert(2, "scenario", args.scenario)
        trial_df.insert(3, "selected_param_set", best_param_set)
        test_trial_all.append(trial_df)

        summary_rows.append(
            {
                "model": spec.name,
                "model_key": spec.key,
                "params_json": json.dumps(best_params, sort_keys=True),
                "param_set": best_param_set,
                "n_features": int(len(features)),
                "n_trials_in_test": int(gte.nunique()),
                "n_train": int(len(tr_idx)),
                "n_validation": int(len(va_idx)),
                "n_train_plus_validation": int(len(tr_idx) + len(va_idx)),
                "n_test": int(len(te_idx)),
                "mae": m_test["mae"],
                "rmse": m_test["rmse"],
                "r2": m_test["r2"],
                "val_mae": float(best_val["val_mae"]),
                "val_rmse": float(best_val["val_rmse"]),
                "val_r2": float(best_val["val_r2"]),
                "fit_seconds_validation": float(best_val["fit_seconds_validation"]),
                "fit_seconds_test": fit_seconds_test,
                "selection_split": "validation",
                "scenario": args.scenario,
            }
        )
        print(
            f"  selected set={best_param_set} | TEST_R2={m_test['r2']:.4f} "
            f"| TEST_RMSE={m_test['rmse']:.2f} | TEST_MAE={m_test['mae']:.2f}"
        )

    validation_df = (
        pd.DataFrame(validation_rows)
        .sort_values(["model", "val_r2", "val_rmse"], ascending=[True, False, True])
        .reset_index(drop=True)
    )
    validation_df.to_csv(out_dir / "validation_grid_results.csv", index=False)

    test_trial_metrics = pd.concat(test_trial_all, ignore_index=True)
    test_trial_metrics.to_csv(out_dir / "trial_aware_trial_metrics.csv", index=False)

    summary_df = pd.DataFrame(summary_rows).sort_values(["r2", "rmse"], ascending=[False, True]).reset_index(drop=True)

    summary_df.to_csv(out_dir / "model_comparison_summary.csv", index=False)

    best = summary_df.iloc[0]
    print("\nBest model (selected on validation, reported on test):")
    print(best.to_string())
    print(f"\nSaved CAROB model-compare outputs to: {out_dir}")


if __name__ == "__main__":
    main()
