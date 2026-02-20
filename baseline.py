"""
Quick EDA + baseline model for paddydataset.csv.

Run:
  python baseline.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, LeaveOneGroupOut, train_test_split

from feature_prepare import FeaturePreparer, RAW_TARGET_COL, TARGET_COL, clean_columns

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None


def parse_args() -> argparse.Namespace:
    base_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Train/evaluate paddy yield baseline model")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=base_dir / "data" / "raw" / "paddydataset.csv",
        help="Path to source dataset CSV",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=base_dir / "results" / "baseline_outputs",
        help="Directory for output artifacts",
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="Holdout ratio")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    parser.add_argument("--n-estimators", type=int, default=200, help="Random forest trees")
    parser.add_argument(
        "--group-col",
        type=str,
        default="Agriblock",
        help="Column name for group-based evaluation",
    )
    return parser.parse_args()


def build_model(n_estimators: int, random_state: int) -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
    )


def print_metrics(header: str, y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))

    print(f"\n{header}")
    print(f"  MAE : {mae:,.2f}")
    print(f"  RMSE: {rmse:,.2f}")
    print(f"  R^2 : {r2:.4f}")
    return {"mae": mae, "rmse": rmse, "r2": r2}


def main() -> None:
    args = parse_args()

    if not args.data_path.exists():
        raise FileNotFoundError(f"Missing dataset: {args.data_path}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.data_path)
    df.columns = clean_columns(list(df.columns))

    if RAW_TARGET_COL not in df.columns:
        raise ValueError(f"Target column not found: {RAW_TARGET_COL}")

    print("Shape:", df.shape)
    print("\nColumns:", df.columns.tolist())
    print("\nDtypes:\n", df.dtypes)
    print("\nMissing values (top 10):\n", df.isna().sum().sort_values(ascending=False).head(10))
    print("\nTarget summary:\n", df[RAW_TARGET_COL].describe())

    dup_count = int(df.duplicated().sum())
    print(f"\nDuplicate rows (before dedup): {dup_count}")
    if dup_count > 0:
        print("Example duplicate rows (first 5):")
        print(df[df.duplicated(keep=False)].head(5))

    before_rows = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    after_rows = len(df)
    print(f"\nDeduplication: {before_rows} -> {after_rows} rows (removed {before_rows - after_rows})")

    X_train_raw, X_test_raw = train_test_split(
        df,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    preparer = FeaturePreparer()
    X_train, y_train = preparer.fit_transform(X_train_raw)
    X_test, y_test = preparer.transform(X_test_raw)

    if y_train.isna().any() or y_test.isna().any():
        valid_train = y_train.notna()
        valid_test = y_test.notna()
        X_train, y_train = X_train.loc[valid_train], y_train.loc[valid_train]
        X_test, y_test = X_test.loc[valid_test], y_test.loc[valid_test]

    model = build_model(args.n_estimators, args.random_state)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    metrics = {"random_split": print_metrics("Baseline model metrics (RandomForest):", y_test, preds)}

    high_corr_cols: list[str] = []
    train_df_corr = X_train.copy()
    train_df_corr[TARGET_COL] = y_train.values
    corr = train_df_corr.corr(numeric_only=True)[TARGET_COL].sort_values(ascending=False)
    print("\nTop correlations with target:\n", corr.head(8))
    print("\nBottom correlations with target:\n", corr.tail(8))

    high_corr_cols = [c for c, v in corr.items() if c != TARGET_COL and abs(v) >= 0.98]
    if high_corr_cols:
        print("\nLeakage check: dropping highly correlated features (|corr| >= 0.98)")
        print("Dropped:", high_corr_cols)

        X_train_l = X_train.drop(columns=[c for c in high_corr_cols if c in X_train.columns])
        X_test_l = X_test.drop(columns=[c for c in high_corr_cols if c in X_test.columns])

        model_l = build_model(args.n_estimators, args.random_state)
        model_l.fit(X_train_l, y_train)
        preds_l = model_l.predict(X_test_l)
        metrics["leakage_check"] = print_metrics("Leakage-check metrics (RandomForest):", y_test, preds_l)
    else:
        print("\nLeakage check skipped: no highly correlated features found.")

    if args.group_col in df.columns:
        gss = GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.random_state)
        groups = df[args.group_col].reset_index(drop=True)

        train_idx, test_idx = next(gss.split(df, groups=groups))
        train_g_raw = df.iloc[train_idx].reset_index(drop=True)
        test_g_raw = df.iloc[test_idx].reset_index(drop=True)

        preparer_g = FeaturePreparer()
        X_train_g, y_train_g = preparer_g.fit_transform(train_g_raw)
        X_test_g, y_test_g = preparer_g.transform(test_g_raw)

        valid_train = y_train_g.notna()
        valid_test = y_test_g.notna()
        X_train_g, y_train_g = X_train_g.loc[valid_train], y_train_g.loc[valid_train]
        X_test_g, y_test_g = X_test_g.loc[valid_test], y_test_g.loc[valid_test]

        if high_corr_cols:
            keep_train = [c for c in X_train_g.columns if c not in high_corr_cols]
            keep_test = [c for c in X_test_g.columns if c not in high_corr_cols]
            common = [c for c in keep_train if c in keep_test]
            X_train_g = X_train_g[common]
            X_test_g = X_test_g[common]

        model_g = build_model(args.n_estimators, args.random_state)
        model_g.fit(X_train_g, y_train_g)
        preds_g = model_g.predict(X_test_g)
        metrics["group_split"] = print_metrics("Group-based split metrics (Agriblock held-out):", y_test_g, preds_g)

        logo = LeaveOneGroupOut()
        maes: list[float] = []
        rmses: list[float] = []
        r2s: list[float] = []
        group_names: list[str] = []

        for tr_idx, te_idx in logo.split(df, groups=groups):
            train_lg_raw = df.iloc[tr_idx].reset_index(drop=True)
            test_lg_raw = df.iloc[te_idx].reset_index(drop=True)
            grp = str(groups.iloc[te_idx].iloc[0])

            preparer_lg = FeaturePreparer()
            X_train_lg, y_train_lg = preparer_lg.fit_transform(train_lg_raw)
            X_test_lg, y_test_lg = preparer_lg.transform(test_lg_raw)

            valid_train_lg = y_train_lg.notna()
            valid_test_lg = y_test_lg.notna()
            X_train_lg, y_train_lg = X_train_lg.loc[valid_train_lg], y_train_lg.loc[valid_train_lg]
            X_test_lg, y_test_lg = X_test_lg.loc[valid_test_lg], y_test_lg.loc[valid_test_lg]

            if len(y_train_lg) == 0 or len(y_test_lg) == 0:
                continue

            model_lg = build_model(args.n_estimators, args.random_state)
            model_lg.fit(X_train_lg, y_train_lg)
            preds_lg = model_lg.predict(X_test_lg)

            maes.append(float(mean_absolute_error(y_test_lg, preds_lg)))
            rmses.append(float(np.sqrt(mean_squared_error(y_test_lg, preds_lg))))
            r2s.append(float(r2_score(y_test_lg, preds_lg)))
            group_names.append(grp)

        if maes:
            print("\nLeave-one-Agriblock-out metrics:")
            for g, m, r, r2 in zip(group_names, maes, rmses, r2s):
                print(f"  {g}: MAE {m:,.2f} | RMSE {r:,.2f} | R^2 {r2:.4f}")
            print(f"  Mean: MAE {np.mean(maes):,.2f} | RMSE {np.mean(rmses):,.2f} | R^2 {np.mean(r2s):.4f}")
            print(f"  Std : MAE {np.std(maes):,.2f} | RMSE {np.std(rmses):,.2f} | R^2 {np.std(r2s):.4f}")

            metrics["logo_mean"] = {
                "mae": float(np.mean(maes)),
                "rmse": float(np.mean(rmses)),
                "r2": float(np.mean(r2s)),
            }

    if plt is not None:
        plt.figure(figsize=(8, 5))
        y_train.hist(bins=30)
        plt.title("Target Distribution (Train)")
        plt.xlabel(TARGET_COL)
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(args.out_dir / "target_distribution_train.png", dpi=150)
        plt.close()
    else:
        print("Skipped target distribution plot: matplotlib is not installed.")

    try:
        importances = model.feature_importances_
        feature_names = np.array(X_train.columns)
        top_idx = np.argsort(importances)[-15:][::-1]
        top_feats = pd.Series(importances[top_idx], index=feature_names[top_idx])
        top_feats.to_csv(args.out_dir / "top_feature_importances.csv")
    except Exception as exc:
        print("Skipped feature importances:", exc)

    pd.DataFrame([{"metric_set": k, **v} for k, v in metrics.items()]).to_csv(
        args.out_dir / "metrics_summary.csv", index=False
    )


if __name__ == "__main__":
    main()
