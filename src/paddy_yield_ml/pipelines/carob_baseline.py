"""Quick CAROB EDA + baseline model for decision-support initialization."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from paddy_yield_ml.pipelines import carob_common as cc

try:
    project_root = Path(__file__).resolve().parents[3]
except NameError:
    project_root = Path.cwd()

OUT_DIR = project_root / "outputs" / "carob_baseline"


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

    return ColumnTransformer(transformers=transformers, remainder="drop")


def build_feature_list(df: pd.DataFrame, role_map: pd.DataFrame) -> list[str]:
    exclude = {cc.TARGET_COL, cc.GROUP_COL}
    exclude |= cc.IDENTIFIER_COLS
    exclude |= cc.POST_OUTCOME_COLS
    exclude.add("dataset_id")
    exclude.add("record_id")

    rec_map = role_map.set_index("column_name")["modeling_recommendation"].to_dict()
    role_map_flag = role_map.set_index("column_name")["final_role"].to_dict()

    features: list[str] = []
    for col in df.columns:
        if col in exclude:
            continue
        if int(df[col].notna().sum()) == 0:
            continue
        rec = str(rec_map.get(col, "")).strip().lower()
        role = str(role_map_flag.get(col, "context")).strip().lower()
        if rec.startswith("exclude"):
            continue
        if role == "proxy":
            continue
        features.append(col)
    return cc.dedupe_keep_order(features)


def logo_eval(
    x: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
) -> pd.DataFrame:
    logo = LeaveOneGroupOut()
    rows: list[dict[str, object]] = []

    for fold_idx, (tr, te) in enumerate(logo.split(x, y, groups), start=1):
        xtr, xte = x.iloc[tr].copy(), x.iloc[te].copy()
        ytr, yte = y.iloc[tr], y.iloc[te]
        gte = groups.iloc[te].astype(str)
        holdout_group = gte.mode().iat[0] if len(gte) > 0 else "unknown"

        pipe = Pipeline(
            [
                ("pre", make_preprocessor(xtr)),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=500,
                        min_samples_leaf=3,
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
        pipe.fit(xtr, ytr)
        pred = pipe.predict(xte)

        rows.append(
            {
                "fold": fold_idx,
                "holdout_trial": holdout_group,
                "n_test": int(len(yte)),
                "mae": float(mean_absolute_error(yte, pred)),
                "rmse": float(np.sqrt(mean_squared_error(yte, pred))),
                "r2": float(r2_score(yte, pred)),
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = cc.load_analysis_frame(require_treatment=True)
    role_map = cc.load_role_map(frame_cols=df.columns.tolist())

    print("CAROB shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print("Target summary:\n", pd.to_numeric(df[cc.TARGET_COL], errors="coerce").describe().round(3).to_string())
    print("\nTreatment split:\n", df[cc.TREATMENT_COL].value_counts(dropna=False).to_string())
    print("\nTrials:", int(df[cc.GROUP_COL].nunique()), "| Countries:", int(df["country"].nunique()))

    missing = df.isna().mean().mul(100).sort_values(ascending=False).reset_index()
    missing.columns = ["feature", "missing_pct"]
    missing.to_csv(OUT_DIR / "missingness_audit.csv", index=False)

    plt.figure(figsize=(8, 5))
    pd.to_numeric(df[cc.TARGET_COL], errors="coerce").hist(bins=35)
    plt.title("CAROB Target Distribution (yield)")
    plt.xlabel(cc.TARGET_COL)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "target_distribution.png", dpi=150)
    plt.close()

    plt.figure(figsize=(7, 5))
    tp = df[[cc.TREATMENT_COL, cc.TARGET_COL]].copy()
    tp.boxplot(by=cc.TREATMENT_COL, column=cc.TARGET_COL)
    plt.title("Yield by Treatment (+P vs -P)")
    plt.suptitle("")
    plt.xlabel(cc.TREATMENT_COL)
    plt.ylabel(cc.TARGET_COL)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "yield_by_treatment_boxplot.png", dpi=150)
    plt.close()

    trial_profile = (
        df.groupby(cc.GROUP_COL)
        .agg(
            n_rows=(cc.TARGET_COL, "size"),
            yield_mean=(cc.TARGET_COL, "mean"),
            yield_std=(cc.TARGET_COL, "std"),
            plus_share=(cc.TREATMENT_COL, lambda s: float((s == cc.TREATMENT_POS).mean())),
        )
        .reset_index()
    )
    trial_profile.to_csv(OUT_DIR / "trial_profile.csv", index=False)

    features = build_feature_list(df, role_map)
    x = df[features].copy()
    y = pd.to_numeric(df[cc.TARGET_COL], errors="coerce")
    groups = df[cc.GROUP_COL].astype(str)

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tr, te = next(gss.split(x, y, groups))
    xtr, xte = x.iloc[tr].copy(), x.iloc[te].copy()
    ytr, yte = y.iloc[tr], y.iloc[te]

    pipe = Pipeline(
        [
            ("pre", make_preprocessor(xtr)),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=500,
                    min_samples_leaf=3,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    pipe.fit(xtr, ytr)
    pred = pipe.predict(xte)

    holdout_metrics = pd.DataFrame(
        [
            {
                "split": "group_shuffle",
                "n_train": int(len(tr)),
                "n_test": int(len(te)),
                "mae": float(mean_absolute_error(yte, pred)),
                "rmse": float(np.sqrt(mean_squared_error(yte, pred))),
                "r2": float(r2_score(yte, pred)),
            }
        ]
    )
    holdout_metrics.to_csv(OUT_DIR / "group_holdout_metrics.csv", index=False)

    logo_df = logo_eval(x=x, y=y, groups=groups)
    logo_df.to_csv(OUT_DIR / "logo_fold_metrics.csv", index=False)

    logo_summary = pd.DataFrame(
        [
            {
                "metric": "mae",
                "mean": float(logo_df["mae"].mean()),
                "std": float(logo_df["mae"].std(ddof=0)),
            },
            {
                "metric": "rmse",
                "mean": float(logo_df["rmse"].mean()),
                "std": float(logo_df["rmse"].std(ddof=0)),
            },
            {
                "metric": "r2",
                "mean": float(logo_df["r2"].mean()),
                "std": float(logo_df["r2"].std(ddof=0)),
            },
        ]
    )
    logo_summary.to_csv(OUT_DIR / "logo_summary_metrics.csv", index=False)

    print("\nBaseline group holdout metrics:")
    print(holdout_metrics.to_string(index=False))
    print("\nLOGO summary:")
    print(logo_summary.to_string(index=False))
    print(f"\nSaved CAROB baseline outputs to: {OUT_DIR}")


if __name__ == "__main__":
    main()
