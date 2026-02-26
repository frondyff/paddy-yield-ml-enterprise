"""Baseline training pipeline for paddy yield prediction."""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, LeaveOneGroupOut, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from paddy_yield_ml.pipelines.paths import project_root, resolve_data_path

RAW_TARGET_COL = "Paddy yield(in Kg)"
TARGET_COL = "Paddy yield_per_hectare(in Kg)"
SIZE_COL = "Hectares"
SIZE_SCALED_COLS = [
    "LP_nurseryarea(in Tonnes)",
    "Micronutrients_70Days",
    "Weed28D_thiobencarb",
    "Urea_40Days",
    "DAP_20days",
    "Nursery area (Cents)",
    "Pest_60Day(in ml)",
    "LP_Mainfield(in Tonnes)",
    "Seedrate(in Kg)",
    "Potassh_50Days",
]

LOGGER = logging.getLogger(__name__)


def clean_columns(cols: Sequence[object]) -> list[str]:
    return [" ".join(str(col).strip().split()) for col in cols]


def load_dataset(data_path: Path) -> pd.DataFrame:
    if not data_path.exists():
        raise FileNotFoundError(f"Missing dataset: {data_path}")

    df = pd.read_csv(data_path)
    df.columns = clean_columns(list(df.columns))
    return df


def add_per_hectare_features(
    df: pd.DataFrame,
    *,
    size_col: str = SIZE_COL,
    raw_target_col: str = RAW_TARGET_COL,
    target_col: str = TARGET_COL,
) -> pd.DataFrame:
    missing = [col for col in [size_col, raw_target_col] if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {missing}")

    out = df.copy()
    hectares = pd.to_numeric(out[size_col], errors="coerce")
    hectares = hectares.replace(0, np.nan)

    out[target_col] = pd.to_numeric(out[raw_target_col], errors="coerce") / hectares

    for col in SIZE_SCALED_COLS:
        if col in out.columns:
            out[f"{col}_per_hectare"] = pd.to_numeric(out[col], errors="coerce") / hectares
            out = out.drop(columns=[col])

    out = out.drop(columns=[raw_target_col])
    return out


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("num", "passthrough", numeric_cols),
        ],
        remainder="drop",
    )


def build_model(seed: int) -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=200,
        random_state=seed,
        n_jobs=-1,
    )


def _metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def _safe_group_r2(y_true: pd.Series, y_pred: np.ndarray) -> float:
    if len(y_true) < 2:
        return float("nan")
    return float(r2_score(y_true, y_pred))


def train_and_evaluate(
    df: pd.DataFrame,
    *,
    seed: int = 42,
    test_size: float = 0.2,
    group_col: str | None = "Agriblock",
    run_group_eval: bool = True,
    leakage_corr_threshold: float = 0.98,
) -> dict[str, Any]:
    required = [TARGET_COL]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {missing}")

    model_df = df.drop_duplicates().reset_index(drop=True)

    y = pd.to_numeric(model_df[TARGET_COL], errors="coerce")
    X = model_df.drop(columns=[TARGET_COL])

    valid_mask = y.notna()
    X = X.loc[valid_mask].reset_index(drop=True)
    y = y.loc[valid_mask].reset_index(drop=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
    )

    preprocessor = build_preprocessor(X_train)
    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", build_model(seed)),
        ]
    )
    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)
    metrics: dict[str, Any] = {"random_split": _metrics(y_test, preds)}

    numeric_train = X_train.select_dtypes(include=[np.number]).copy()
    if not numeric_train.empty:
        corr_frame = numeric_train.copy()
        corr_frame[TARGET_COL] = y_train.values
        corr = corr_frame.corr(numeric_only=True)[TARGET_COL].drop(labels=[TARGET_COL], errors="ignore")
        leakage_cols = corr.index[corr.abs() >= leakage_corr_threshold].tolist()
    else:
        leakage_cols = []

    if leakage_cols:
        X_train_l = X_train.drop(columns=[c for c in leakage_cols if c in X_train.columns])
        X_test_l = X_test.drop(columns=[c for c in leakage_cols if c in X_test.columns])

        pipeline_l = Pipeline(
            steps=[
                ("preprocess", build_preprocessor(X_train_l)),
                ("model", build_model(seed)),
            ]
        )
        pipeline_l.fit(X_train_l, y_train)
        preds_l = pipeline_l.predict(X_test_l)

        metrics["leakage_check"] = {
            "dropped_features": leakage_cols,
            "metrics": _metrics(y_test, preds_l),
        }
    else:
        metrics["leakage_check"] = {
            "dropped_features": [],
            "metrics": None,
        }

    group_enabled = run_group_eval and bool(group_col)
    if group_enabled and group_col in model_df.columns:
        groups = model_df.loc[valid_mask, group_col].reset_index(drop=True)
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        tr_idx, te_idx = next(gss.split(X, y, groups=groups))

        X_train_g, X_test_g = X.iloc[tr_idx], X.iloc[te_idx]
        y_train_g, y_test_g = y.iloc[tr_idx], y.iloc[te_idx]

        pipeline_g = Pipeline(
            steps=[
                ("preprocess", build_preprocessor(X_train_g)),
                ("model", build_model(seed)),
            ]
        )
        pipeline_g.fit(X_train_g, y_train_g)
        preds_g = pipeline_g.predict(X_test_g)

        metrics["group_split"] = _metrics(y_test_g, preds_g)

        logo = LeaveOneGroupOut()
        logo_rows: list[dict[str, Any]] = []
        maes: list[float] = []
        rmses: list[float] = []
        r2s: list[float] = []

        for lg_tr_idx, lg_te_idx in logo.split(X, y, groups=groups):
            X_train_lg, X_test_lg = X.iloc[lg_tr_idx], X.iloc[lg_te_idx]
            y_train_lg, y_test_lg = y.iloc[lg_tr_idx], y.iloc[lg_te_idx]
            group_value = str(groups.iloc[lg_te_idx].iloc[0])

            if len(y_train_lg) == 0 or len(y_test_lg) == 0:
                continue

            pipeline_lg = Pipeline(
                steps=[
                    ("preprocess", build_preprocessor(X_train_lg)),
                    ("model", build_model(seed)),
                ]
            )
            pipeline_lg.fit(X_train_lg, y_train_lg)
            preds_lg = pipeline_lg.predict(X_test_lg)

            row = {
                "group": group_value,
                "mae": float(mean_absolute_error(y_test_lg, preds_lg)),
                "rmse": float(np.sqrt(mean_squared_error(y_test_lg, preds_lg))),
                "r2": _safe_group_r2(y_test_lg, preds_lg),
                "n_rows": int(len(y_test_lg)),
            }
            logo_rows.append(row)
            maes.append(row["mae"])
            rmses.append(row["rmse"])
            if not np.isnan(row["r2"]):
                r2s.append(row["r2"])

        if logo_rows:
            metrics["logo"] = {
                "per_group": logo_rows,
                "summary": {
                    "mae_mean": float(np.mean(maes)),
                    "mae_std": float(np.std(maes)),
                    "rmse_mean": float(np.mean(rmses)),
                    "rmse_std": float(np.std(rmses)),
                    "r2_mean": float(np.mean(r2s)) if r2s else float("nan"),
                    "r2_std": float(np.std(r2s)) if r2s else float("nan"),
                },
            }

    result: dict[str, Any] = {
        "pipeline": pipeline,
        "metrics": metrics,
        "train_shape": [int(X_train.shape[0]), int(X_train.shape[1])],
        "test_shape": [int(X_test.shape[0]), int(X_test.shape[1])],
    }
    return result


def _extract_feature_importances(pipeline: Pipeline, fallback_columns: Sequence[str]) -> pd.DataFrame:
    model = pipeline.named_steps["model"]
    importances = getattr(model, "feature_importances_", None)

    if importances is None:
        return pd.DataFrame({"feature": list(fallback_columns), "importance": 0.0})

    preprocessor = pipeline.named_steps["preprocess"]
    try:
        feature_names = preprocessor.get_feature_names_out().tolist()
    except Exception:
        feature_names = list(fallback_columns)

    if len(feature_names) != len(importances):
        min_len = min(len(feature_names), len(importances))
        feature_names = feature_names[:min_len]
        importances = importances[:min_len]

    frame = pd.DataFrame({"feature": feature_names, "importance": importances})
    return frame.sort_values("importance", ascending=False).reset_index(drop=True)


def save_artifacts(
    *,
    out_dir: Path,
    train_result: dict[str, Any],
    config: dict[str, Any],
    source_df: pd.DataFrame,
    run_eda: bool = False,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "config": config,
        "metrics": train_result["metrics"],
        "train_shape": train_result["train_shape"],
        "test_shape": train_result["test_shape"],
    }
    with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, allow_nan=True)

    pipeline = train_result["pipeline"]
    joblib.dump(pipeline, out_dir / "model.joblib")

    fallback_columns = source_df.drop(columns=[TARGET_COL], errors="ignore").columns.tolist()
    importances = _extract_feature_importances(pipeline, fallback_columns)
    importances.to_csv(out_dir / "feature_importances.csv", index=False)

    if run_eda:
        target = pd.to_numeric(source_df[TARGET_COL], errors="coerce")
        valid_target = target.dropna()

        if not valid_target.empty:
            plt.figure(figsize=(8, 5))
            valid_target.hist(bins=30)
            plt.title("Target Distribution")
            plt.xlabel(TARGET_COL)
            plt.ylabel("Count")
            plt.tight_layout()
            plt.savefig(out_dir / "target_histogram.png", dpi=150)
            plt.close()

        numeric_df = source_df.select_dtypes(include=[np.number])
        if not numeric_df.empty:
            corr = numeric_df.corr(numeric_only=True)
            plt.figure(figsize=(10, 8))
            plt.imshow(corr, cmap="coolwarm", aspect="auto")
            plt.colorbar()
            labels = [str(col) for col in corr.columns]
            plt.xticks(range(len(labels)), labels, rotation=90, fontsize=6)
            plt.yticks(range(len(labels)), labels, fontsize=6)
            plt.title("Numeric Correlation Heatmap")
            plt.tight_layout()
            plt.savefig(out_dir / "numeric_corr_heatmap.png", dpi=150)
            plt.close()


def parse_args() -> argparse.Namespace:
    root = project_root()

    parser = argparse.ArgumentParser(description="Train/evaluate baseline paddy yield model")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=root / "outputs" / "baseline",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument(
        "--group-col",
        type=str,
        default="Agriblock",
        help="Set empty string to disable group evaluation.",
    )
    parser.add_argument("--run-eda", action="store_true")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    args = parse_args()
    data_path = resolve_data_path(args.data_path)
    group_col = args.group_col if args.group_col and args.group_col.strip() else None

    LOGGER.info("Loading dataset from %s", data_path)
    df = load_dataset(data_path)

    LOGGER.info("Engineering per-hectare features")
    featured = add_per_hectare_features(df)

    LOGGER.info("Training baseline model")
    result = train_and_evaluate(
        featured,
        seed=args.seed,
        test_size=args.test_size,
        group_col=group_col,
        run_group_eval=group_col is not None,
    )

    config = {
        "data_path": str(data_path),
        "out_dir": str(args.out_dir),
        "seed": args.seed,
        "test_size": args.test_size,
        "group_col": group_col,
        "run_eda": bool(args.run_eda),
        "target_col": TARGET_COL,
    }

    LOGGER.info("Saving artifacts to %s", args.out_dir)
    save_artifacts(
        out_dir=args.out_dir,
        train_result=result,
        config=config,
        source_df=featured,
        run_eda=args.run_eda,
    )

    random_metrics = result["metrics"]["random_split"]
    LOGGER.info(
        "Random split metrics | MAE=%.4f RMSE=%.4f R2=%.4f",
        random_metrics["mae"],
        random_metrics["rmse"],
        random_metrics["r2"],
    )


if __name__ == "__main__":
    main()
