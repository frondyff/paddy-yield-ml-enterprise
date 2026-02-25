"""
Group-aware model comparison for Step 2 milestone.

Run:
  python src/paddy_yield_ml/pipelines/model_compare.py

Outputs (under ./outputs/model_compare):
  - feature_sets_used.csv
  - logo_fold_metrics.csv
  - model_comparison_summary.csv
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from paddy_yield_ml.pipelines import feature_prepare as fp

try:
    project_root = Path(__file__).resolve().parents[3]
except NameError:
    project_root = Path.cwd()

OUT_DIR = project_root / "outputs" / "model_compare"
CANDIDATES_PATH = project_root / "outputs" / "feature_prepare" / "hybrid_selection_candidates.csv"

RAW_TARGET_COL = fp.RAW_TARGET_COL
TARGET_COL = fp.TARGET_COL
GROUP_COL = fp.GROUP_COL
DATA_PATH = fp.DATA_PATH


@dataclass(frozen=True)
class ModelSpec:
    name: str
    build: Callable[[], object]


def dedupe_keep_order(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            out.append(item)
            seen.add(item)
    return out


def load_analysis_frame() -> pd.DataFrame:
    """Load, clean, and prepare the analysis dataset.
    
    Returns:
        Normalized and deduplicated analysis dataframe.
    """
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing dataset: {DATA_PATH}")

    raw_df = pd.read_csv(DATA_PATH)
    raw_df.columns = fp.clean_columns(raw_df.columns)

    if RAW_TARGET_COL not in raw_df.columns:
        raise ValueError(f"Target column not found: {RAW_TARGET_COL}")
    if GROUP_COL not in raw_df.columns:
        raise ValueError(f"Group column not found: {GROUP_COL}")

    dedup_df = raw_df.drop_duplicates().reset_index(drop=True)
    analysis_df, _ = fp.normalize_per_hectare(
        dedup_df,
        drop_original=False,
        create_input_scaled=False,
    )
    return analysis_df


def load_hybrid_candidates(auto_run_feature_prepare: bool) -> pd.DataFrame:
    if not CANDIDATES_PATH.exists():
        if not auto_run_feature_prepare:
            raise FileNotFoundError(
                f"Missing candidates file: {CANDIDATES_PATH}. "
                "Run feature_prepare first or pass --auto-run-feature-prepare."
            )
        print("Candidates file missing; running feature_prepare pipeline first...")
        fp.main()

    candidates_df = pd.read_csv(CANDIDATES_PATH)
    required = {"feature", "status"}
    missing = required - set(candidates_df.columns)
    if missing:
        raise ValueError(f"Candidates file missing required columns: {sorted(missing)}")
    return candidates_df


def build_feature_sets(candidates_df: pd.DataFrame) -> dict[str, list[str]]:
    """Construct feature sets from hybrid selection candidates.
    
    Returns three feature sets: modifiable_only, modifiable_plus_context, and hybrid_with_review.
    """
    cdf = candidates_df.copy()
    if "hybrid_priority_score" in cdf.columns:
        cdf = cdf.sort_values("hybrid_priority_score", ascending=False, na_position="last")

    modifiable = cdf.loc[cdf["status"] == "candidate_modifiable", "feature"].astype(str).tolist()
    redundant_review = (
        cdf.loc[
            cdf["status"] == "candidate_redundant_review",
            "feature",
        ]
        .astype(str)
        .tolist()
    )
    context = cdf.loc[cdf["status"] == "reserve_context", "feature"].astype(str).tolist()

    modifiable = dedupe_keep_order(modifiable)
    redundant_review = dedupe_keep_order(redundant_review)
    context = dedupe_keep_order(context)

    if not modifiable:
        raise ValueError("No candidate_modifiable features found in hybrid candidates output.")

    feature_sets = {
        "modifiable_only": modifiable,
        "modifiable_plus_context": dedupe_keep_order(modifiable + context),
        "hybrid_with_review": dedupe_keep_order(modifiable + context + redundant_review),
    }
    return feature_sets


def filter_available_features(features: Iterable[str], frame: pd.DataFrame) -> list[str]:
    reserved = {RAW_TARGET_COL, TARGET_COL, GROUP_COL}
    valid = [f for f in dedupe_keep_order(features) if f in frame.columns and f not in reserved]
    if not valid:
        raise ValueError("No valid features found after filtering against dataset columns.")
    return valid


def make_preprocessor(x_train: pd.DataFrame) -> ColumnTransformer:
    """Create a preprocessing pipeline for numeric and categorical features.
    
    Numeric features: imputed with median.
    Categorical features: imputed with mode, then one-hot encoded.
    """
    num_cols = x_train.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = x_train.select_dtypes(exclude=[np.number]).columns.tolist()

    transformers: list[tuple[str, object, list[str]]] = []
    if num_cols:
        transformers.append(
            (
                "num",
                Pipeline([("imputer", SimpleImputer(strategy="median"))]),
                num_cols,
            )
        )
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
        raise ValueError("No usable numeric/categorical columns found for preprocessing.")

    return ColumnTransformer(transformers=transformers, remainder="drop")


def parse_model_list(raw: str) -> list[str]:
    models = [m.strip().lower() for m in raw.split(",") if m.strip()]
    if not models:
        raise ValueError("At least one model must be provided.")
    return dedupe_keep_order(models)


def build_model_specs(model_names: list[str], n_estimators: int, random_state: int) -> list[ModelSpec]:
    """Build model specifications for comparison.
    
    Uses default arguments in lambdas to avoid closure issues.
    """
    supported = {"random_forest", "lightgbm", "xgboost", "catboost"}
    unknown = [m for m in model_names if m not in supported]
    if unknown:
        raise ValueError(f"Unsupported model names: {unknown}. Supported: {sorted(supported)}")

    specs: list[ModelSpec] = []
    for model_name in model_names:
        if model_name == "random_forest":
            specs.append(
                ModelSpec(
                    name="RandomForest",
                    build=lambda n_est=n_estimators, rs=random_state: RandomForestRegressor(
                        n_estimators=n_est,
                        min_samples_leaf=2,
                        random_state=rs,
                        n_jobs=-1,
                    ),
                )
            )
            continue

        if model_name == "lightgbm":
            from lightgbm import LGBMRegressor

            specs.append(
                ModelSpec(
                    name="LightGBM",
                    build=lambda n_est=n_estimators, rs=random_state: LGBMRegressor(
                        n_estimators=n_est,
                        learning_rate=0.05,
                        num_leaves=31,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        random_state=rs,
                    ),
                )
            )
            continue

        if model_name == "xgboost":
            from xgboost import XGBRegressor

            specs.append(
                ModelSpec(
                    name="XGBoost",
                    build=lambda n_est=n_estimators, rs=random_state: XGBRegressor(
                        n_estimators=n_est,
                        learning_rate=0.05,
                        max_depth=6,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        objective="reg:squarederror",
                        random_state=rs,
                        n_jobs=-1,
                    ),
                )
            )
            continue

        if model_name == "catboost":
            from catboost import CatBoostRegressor

            specs.append(
                ModelSpec(
                    name="CatBoost",
                    build=lambda n_est=n_estimators, rs=random_state: CatBoostRegressor(
                        iterations=n_est,
                        learning_rate=0.05,
                        depth=6,
                        loss_function="RMSE",
                        random_seed=rs,
                        verbose=False,
                        allow_writing_files=False,
                    ),
                )
            )
            continue

    return specs


def evaluate_logo(
    frame: pd.DataFrame,
    feature_set_name: str,
    features: list[str],
    model_specs: list[ModelSpec],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate models using Leave-One-Group-Out (LOGO) cross-validation.
    
    Args:
        frame: Input dataframe with features and target.
        feature_set_name: Name of the feature set being evaluated.
        features: List of feature column names.
        model_specs: List of ModelSpec objects to evaluate.
    
    Returns:
        Tuple of (fold_records_df, summary_df) with per-fold and aggregate metrics.
    """
    subset = frame[features + [TARGET_COL, GROUP_COL]].copy()
    subset[TARGET_COL] = pd.to_numeric(subset[TARGET_COL], errors="coerce")
    subset[GROUP_COL] = subset[GROUP_COL].astype("string")
    subset = subset[subset[TARGET_COL].notna() & subset[GROUP_COL].notna()].reset_index(drop=True)

    x = subset[features].copy()
    y = subset[TARGET_COL].copy()
    groups = subset[GROUP_COL].astype(str).copy()

    if groups.nunique() < 2:
        raise ValueError("Group-aware validation needs at least 2 unique groups.")

    logo = LeaveOneGroupOut()
    fold_records: list[dict] = []
    summary_records: list[dict] = []
    n_folds_total = logo.get_n_splits(x, y, groups)

    for spec in model_specs:
        maes: list[float] = []
        rmses: list[float] = []
        r2s: list[float] = []
        runtimes: list[float] = []
        print(f"  Evaluating {spec.name} on {feature_set_name} ({n_folds_total} folds)...")

        for fold_idx, (train_idx, test_idx) in enumerate(logo.split(x, y, groups), start=1):
            if fold_idx % max(1, n_folds_total // 5) == 0 or fold_idx == n_folds_total:
                print(f"    Fold {fold_idx}/{n_folds_total}")
            
            x_train = x.iloc[train_idx].copy()
            x_test = x.iloc[test_idx].copy()
            y_train = y.iloc[train_idx]
            y_test = y.iloc[test_idx]
            held_out_group = groups.iloc[test_idx].iloc[0]

            pipeline = Pipeline(
                [
                    ("preprocess", make_preprocessor(x_train)),
                    ("model", spec.build()),
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

            fold_records.append(
                {
                    "feature_set": feature_set_name,
                    "model": spec.name,
                    "fold": fold_idx,
                    "held_out_agriblock": held_out_group,
                    "n_train": int(len(train_idx)),
                    "n_test": int(len(test_idx)),
                    "mae": mae,
                    "rmse": rmse,
                    "r2": r2,
                    "fit_predict_seconds": runtime_s,
                }
            )

        summary_records.append(
            {
                "feature_set": feature_set_name,
                "model": spec.name,
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
        )

    return pd.DataFrame(fold_records), pd.DataFrame(summary_records)


def run(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Execute the model comparison pipeline.
    
    Args:
        args: Command-line arguments with models, feature_set, n_estimators, random_state.
    
    Returns:
        Tuple of (all_folds_df, all_summary_df) with complete evaluation results.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading and preparing data...")
    analysis_df = load_analysis_frame()
    candidates_df = load_hybrid_candidates(auto_run_feature_prepare=args.auto_run_feature_prepare)

    feature_sets = build_feature_sets(candidates_df)
    if args.feature_set != "all":
        if args.feature_set not in feature_sets:
            raise ValueError(f"Unknown --feature-set={args.feature_set}. Choices: all, {sorted(feature_sets.keys())}")
        feature_sets = {args.feature_set: feature_sets[args.feature_set]}

    model_names = parse_model_list(args.models)
    model_specs = build_model_specs(model_names, n_estimators=args.n_estimators, random_state=args.random_state)
    print(f"Comparing {len(model_specs)} model(s) on {len(feature_sets)} feature set(s)...\n")

    feature_rows: list[dict] = []
    fold_tables: list[pd.DataFrame] = []
    summary_tables: list[pd.DataFrame] = []

    for fs_name, fs_features in feature_sets.items():
        valid_features = filter_available_features(fs_features, analysis_df)
        for rank_idx, feature_name in enumerate(valid_features, start=1):
            feature_rows.append(
                {
                    "feature_set": fs_name,
                    "rank_in_set": rank_idx,
                    "feature": feature_name,
                }
            )

        fold_df, summary_df = evaluate_logo(
            frame=analysis_df,
            feature_set_name=fs_name,
            features=valid_features,
            model_specs=model_specs,
        )
        fold_tables.append(fold_df)
        summary_tables.append(summary_df)

    feature_sets_df = pd.DataFrame(feature_rows)
    all_folds_df = pd.concat(fold_tables, ignore_index=True)
    all_summary_df = pd.concat(summary_tables, ignore_index=True)
    all_summary_df = all_summary_df.sort_values(["r2_mean", "rmse_mean"], ascending=[False, True]).reset_index(
        drop=True
    )

    feature_sets_df.to_csv(OUT_DIR / "feature_sets_used.csv", index=False)
    all_folds_df.to_csv(OUT_DIR / "logo_fold_metrics.csv", index=False)
    all_summary_df.to_csv(OUT_DIR / "model_comparison_summary.csv", index=False)

    print("Saved model comparison artifacts to:", OUT_DIR)
    print("\nTop model configurations (by mean R^2):")
    print(
        all_summary_df[
            [
                "feature_set",
                "model",
                "n_features",
                "r2_mean",
                "r2_std",
                "rmse_mean",
                "mae_mean",
            ]
        ].head(10)
    )

    return all_folds_df, all_summary_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Agriblock-aware model comparison for Step 2 milestone.")
    parser.add_argument(
        "--models",
        type=str,
        default="random_forest,lightgbm,xgboost,catboost",
        help="Comma-separated list. Supported: random_forest, lightgbm, xgboost, catboost.",
    )
    parser.add_argument(
        "--feature-set",
        type=str,
        default="all",
        help="One of: all, modifiable_only, modifiable_plus_context, hybrid_with_review.",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=250,
        help="Tree iterations/estimators for compared models.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--auto-run-feature-prepare",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Auto-run feature_prepare if hybrid candidate artifact is missing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
