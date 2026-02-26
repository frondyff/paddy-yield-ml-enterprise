"""Interpretability stage artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from paddy_yield_ml.pipelines.baseline_train import (
    TARGET_COL,
    _metrics,
    add_per_hectare_features,
    build_preprocessor,
    load_dataset,
)
from paddy_yield_ml.pipelines.paths import project_root, resolve_data_path


def run_interpretability(data_path: Path, out_dir: Path, seed: int, test_size: float) -> dict[str, object]:
    df = add_per_hectare_features(load_dataset(data_path)).drop_duplicates().reset_index(drop=True)

    y = pd.to_numeric(df[TARGET_COL], errors="coerce")
    X = df.drop(columns=[TARGET_COL])
    valid = y.notna()
    X = X.loc[valid].reset_index(drop=True)
    y = y.loc[valid].reset_index(drop=True)

    full = pd.concat([X, y.rename(TARGET_COL)], axis=1).dropna().reset_index(drop=True)
    X = full.drop(columns=[TARGET_COL])
    y = full[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocess", build_preprocessor(X_train)),
            ("model", RandomForestRegressor(n_estimators=200, random_state=seed, n_jobs=-1)),
        ]
    )
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)

    out_dir.mkdir(parents=True, exist_ok=True)

    pred_frame = pd.DataFrame(
        {
            "y_true": y_test.reset_index(drop=True),
            "y_pred": pd.Series(preds),
        }
    )
    pred_frame["residual"] = pred_frame["y_true"] - pred_frame["y_pred"]
    pred_frame.to_csv(out_dir / "sample_predictions.csv", index=False)

    perm = permutation_importance(
        pipeline,
        X_test,
        y_test,
        n_repeats=8,
        random_state=seed,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )
    perm_frame = pd.DataFrame(
        {
            "feature": X_test.columns,
            "importance_mean": perm.importances_mean,
            "importance_std": perm.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)
    perm_frame.to_csv(out_dir / "permutation_importance.csv", index=False)

    metrics = _metrics(y_test, preds)
    with (out_dir / "interpretability_summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "metrics": metrics,
                "n_train_rows": int(X_train.shape[0]),
                "n_test_rows": int(X_test.shape[0]),
            },
            f,
            indent=2,
        )

    return {
        "metrics": metrics,
        "top_features": perm_frame.head(10)["feature"].tolist(),
    }


def parse_args() -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(description="Generate interpretability artifacts")
    parser.add_argument("--data-path", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=root / "outputs" / "interpretability")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = resolve_data_path(args.data_path)
    run_interpretability(data_path, args.out_dir, args.seed, args.test_size)


if __name__ == "__main__":
    main()
