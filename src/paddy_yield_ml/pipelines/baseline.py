"""
Quick EDA + baseline model for paddydataset.csv

Run:
  python src/paddy_yield_ml/pipelines/baseline.py

Outputs:
  - prints dataset overview and basic stats
  - saves plots to ./outputs/baseline/
  - trains a simple baseline model and reports metrics
"""

from __future__ import annotations

from pathlib import Path

# Optional plotting (script still runs without display)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    silhouette_score,
)
from sklearn.model_selection import (
    GroupShuffleSplit,
    LeaveOneGroupOut,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    # Resolve project root from src/paddy_yield_ml/pipelines/baseline.py
    project_root = Path(__file__).resolve().parents[3]
except NameError:
    # Fallback for interactive runs.
    project_root = Path.cwd()

DATA_PATH = project_root / "data" / "input" / "paddydataset.csv"
OUT_DIR = project_root / "outputs" / "baseline"
RAW_TARGET_COL = "Paddy yield(in Kg)"
TARGET_COL = "Paddy yield_per_hectare(in Kg)"


def clean_columns(cols: list[str]) -> list[str]:
    # Normalize headers so downstream column lookups are stable.
    # Reason: real-world CSV headers often contain accidental spacing differences.
    return [" ".join(c.strip().split()) for c in cols]


def main() -> None:
    # Fail fast with a clear error if the expected dataset is not present.
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing dataset: {DATA_PATH}")

    # Ensure output folder exists before generating any plots/tables.
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data and normalize headers to avoid fragile exact-string mismatches.
    df = pd.read_csv(DATA_PATH)
    df.columns = clean_columns(list(df.columns))

    # Validate the target early so the script stops before expensive processing.
    if RAW_TARGET_COL not in df.columns:
        raise ValueError(f"Target column not found: {RAW_TARGET_COL}")

    # Print a compact data quality snapshot for quick sanity checks.
    print("Shape:", df.shape)
    print("\nColumns:", df.columns.tolist())
    print("\nDtypes:\n", df.dtypes)
    print(
        "\nMissing values (top 10):\n",
        df.isna().sum().sort_values(ascending=False).head(10),
    )

    # Basic target summary helps spot scale issues/outliers before transformations.
    print("\nTarget summary:\n", df[RAW_TARGET_COL].describe())

    # Check duplicate rows first so we can report data hygiene on the raw file.
    dup_count = df.duplicated().sum()
    print(f"\nDuplicate rows (before dedup): {dup_count}")
    if dup_count > 0:
        print("Example duplicate rows (first 5):")
        print(df[df.duplicated(keep=False)].head(5))

    # Remove exact duplicates to avoid overweighting repeated observations in modeling.
    before_rows = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    after_rows = len(df)
    print(
        f"\nDeduplication: {before_rows} -> {after_rows} rows (removed {before_rows - after_rows})"
    )

    # Convert totals to per-hectare values so farms of different size are comparable.
    # Reason: absolute totals (kg, ml, tonnes, etc.) naturally grow with area and can
    # hide true intensity/efficiency patterns.
    size_col = "Hectares"
    perfect_size_cols = [
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
    if size_col in df.columns:
        # Use numeric conversion with coercion to safely handle malformed entries.
        # Invalid strings become NaN rather than crashing the pipeline.
        hectares = pd.to_numeric(df[size_col], errors="coerce")
        # Build modeling target as yield intensity (kg per hectare) instead of raw yield.
        df[TARGET_COL] = pd.to_numeric(df[RAW_TARGET_COL], errors="coerce") / hectares
        for col in perfect_size_cols:
            if col in df.columns:
                per_col = f"{col}_per_hectare"
                # Create per-hectare version of each area-dependent input feature.
                df[per_col] = pd.to_numeric(df[col], errors="coerce") / hectares
                # Drop original totals to prevent redundant representations of same signal.
                df = df.drop(columns=[col])
        print(
            "\nNormalized size-scaled inputs and target by Hectares (per-hectare):",
            [
                f"{c}_per_hectare"
                for c in perfect_size_cols
                if c in df.columns or f"{c}_per_hectare" in df.columns
            ],
        )
    else:
        raise ValueError(f"Size column not found for normalization: {size_col}")

    # Unsupervised clustering gives a quick view of latent farm/activity segments.
    # This is exploratory only (not used directly in model training).
    try:
        # Sample for runtime control; enough rows for pattern discovery while staying fast.
        sample_df = df.sample(n=min(1000, len(df)), random_state=42)
        Xc = sample_df.drop(columns=[TARGET_COL])
        cat_cols_c = Xc.select_dtypes(exclude=[np.number]).columns.tolist()
        num_cols_c = Xc.select_dtypes(include=[np.number]).columns.tolist()

        # Encode categoricals + standardize numerics so KMeans distance is meaningful.
        pre_c = ColumnTransformer(
            transformers=[
                ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols_c),
                ("num", StandardScaler(), num_cols_c),
            ]
        )

        Xc_enc = pre_c.fit_transform(Xc)

        print("\nCluster analysis (sample up to 1000 rows):")
        best_k = None
        best_sil = -1.0
        best_labels = None
        # Try multiple k values and pick the one with best silhouette separation.
        for k in range(2, 7):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(Xc_enc)
            sil = silhouette_score(Xc_enc, labels)
            print(f"  k={k}: silhouette={sil:.4f}")
            if sil > best_sil:
                best_sil = sil
                best_k = k
                best_labels = labels

        # Always render a k=3 PCA plot as a consistent visual artifact for reports.
        # PCA compresses high-dimensional encoded features into 2D for plotting.
        Xc_dense = Xc_enc.toarray() if hasattr(Xc_enc, "toarray") else Xc_enc
        pca = PCA(n_components=2, random_state=42)
        X2 = pca.fit_transform(Xc_dense)
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        labels = kmeans.fit_predict(Xc_enc)

        plt.figure(figsize=(7, 5))
        plt.scatter(X2[:, 0], X2[:, 1], c=labels, s=12, cmap="tab10")
        plt.title("Cluster visualization (PCA 2D, k=3)")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.tight_layout()
        plt.savefig(OUT_DIR / "cluster_pca_k3.png", dpi=150)
        plt.close()

        if best_labels is not None:
            # Cluster sizes indicate whether segments are balanced or dominated by one group.
            vals, counts = np.unique(best_labels, return_counts=True)
            print(f"\nBest k by silhouette: {best_k} (score {best_sil:.4f})")
            for v, c in zip(vals, counts, strict=False):
                print(f"  cluster {v}: {c} rows")

            # Profile clusters using high-variance numeric features because they best
            # differentiate groups in practice.
            sample_df = sample_df.copy()
            sample_df["cluster"] = best_labels
            num_profile = sample_df.select_dtypes(include=[np.number])
            variances = num_profile.var().sort_values(ascending=False)
            top_features = variances.head(10).index.tolist()

            profile = sample_df.groupby("cluster")[top_features].mean().round(2)

            print("\nCluster profiling (mean of top-variance numeric features):")
            print(profile)
            profile.to_csv(OUT_DIR / "cluster_profile_top_variance.csv")
    except Exception as exc:
        # Keep the script robust: skip exploratory clustering if encoding/data edge cases occur.
        print("Skipped cluster analysis:", exc)

    # Correlation scan highlights strong linear associations with the target.
    # Also used to detect suspiciously high-correlation leakage candidates.
    num_df = df.select_dtypes(include=[np.number])
    high_corr_cols = []
    if not num_df.empty:
        corr = num_df.corr(numeric_only=True)[TARGET_COL].sort_values(ascending=False)
        print("\nTop correlations with target:\n", corr.head(8))
        print("\nBottom correlations with target:\n", corr.tail(8))
        # Very high absolute correlation is treated as potential leakage/proxy of target.
        high_corr_cols = [
            c for c, v in corr.items() if c != TARGET_COL and abs(v) >= 0.98
        ]

        # Save full numeric correlation heatmap for offline inspection/reporting.
        plt.figure(figsize=(10, 8))
        plt.imshow(num_df.corr(numeric_only=True), cmap="coolwarm", aspect="auto")
        plt.colorbar()
        plt.title("Numeric Feature Correlation Heatmap")
        plt.tight_layout()
        plt.savefig(OUT_DIR / "correlation_heatmap.png", dpi=150)
        plt.close()

    # Distribution plot reveals skewness and heavy tails that affect model errors.
    plt.figure(figsize=(8, 5))
    df[TARGET_COL].hist(bins=30)
    plt.title("Target Distribution")
    plt.xlabel(TARGET_COL)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "target_distribution.png", dpi=150)
    plt.close()

    # Build supervised learning matrices.
    # Remove both per-hectare target and raw-yield target from features to avoid leakage.
    X = df.drop(columns=[TARGET_COL])
    if RAW_TARGET_COL in X.columns:
        X = X.drop(columns=[RAW_TARGET_COL])
    y = df[TARGET_COL]

    # Domain-based leakage guard: remove feature suspected to encode outcome information.
    if "Trash(in bundles)" in X.columns:
        X = X.drop(columns=["Trash(in bundles)"])

    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    # Preprocess mixed data types:
    # - one-hot for categoricals (safe for unseen categories)
    # - passthrough numerics (tree models do not need scaling).
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", "passthrough", num_cols),
        ]
    )

    # Baseline model: Random Forest is a strong non-linear default with minimal tuning.
    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
    )

    # Pipeline ensures identical preprocessing is applied during train and predict.
    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )

    # Random holdout gives a first-pass estimate under i.i.d.-style assumptions.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Fit/evaluate baseline model.
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    print("\nBaseline model metrics (RandomForest):")
    print(f"  MAE : {mae:,.2f}")
    print(f"  RMSE: {rmse:,.2f}")
    print(f"  R^2 : {r2:.4f}")

    # Leakage stress test:
    # Re-train after dropping near-perfectly correlated features and compare metrics.
    # Large metric drops would suggest the baseline relied on leakage/proxy signals.
    if high_corr_cols:
        print("\nLeakage check: dropping highly correlated features (|corr| >= 0.98)")
        print("Dropped:", high_corr_cols)
        X_leak = X.drop(columns=[c for c in high_corr_cols if c in X.columns])

        cat_cols_l = X_leak.select_dtypes(exclude=[np.number]).columns.tolist()
        num_cols_l = X_leak.select_dtypes(include=[np.number]).columns.tolist()

        preprocessor_l = ColumnTransformer(
            transformers=[
                ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols_l),
                ("num", "passthrough", num_cols_l),
            ]
        )

        # Reuse same model class/config to isolate only the effect of feature removal.
        pipeline_l = Pipeline(
            steps=[
                ("preprocess", preprocessor_l),
                ("model", model),
            ]
        )

        X_train_l, X_test_l, y_train_l, y_test_l = train_test_split(
            X_leak, y, test_size=0.2, random_state=42
        )

        pipeline_l.fit(X_train_l, y_train_l)
        preds_l = pipeline_l.predict(X_test_l)

        mae_l = mean_absolute_error(y_test_l, preds_l)
        rmse_l = np.sqrt(mean_squared_error(y_test_l, preds_l))
        r2_l = r2_score(y_test_l, preds_l)

        print("\nLeakage-check metrics (RandomForest):")
        print(f"  MAE : {mae_l:,.2f}")
        print(f"  RMSE: {rmse_l:,.2f}")
        print(f"  R^2 : {r2_l:.4f}")
    else:
        print("\nLeakage check skipped: no highly correlated features found.")

    # Group-aware validation estimates generalization to unseen Agriblocks.
    # Reason: random splits can look optimistic when train/test share local patterns.
    group_col = "Agriblock"
    if group_col in df.columns:
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        groups = df[group_col]

        # Use the same feature set as leakage check (deduped + high-corr dropped)
        X_g = (
            X.drop(columns=[c for c in high_corr_cols if c in X.columns])
            if high_corr_cols
            else X
        )
        y_g = y

        # Group-evaluation pipeline mirrors baseline preprocessing/model for comparability.
        cat_cols_g = X_g.select_dtypes(exclude=[np.number]).columns.tolist()
        num_cols_g = X_g.select_dtypes(include=[np.number]).columns.tolist()

        preprocessor_g = ColumnTransformer(
            transformers=[
                ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols_g),
                ("num", "passthrough", num_cols_g),
            ]
        )

        pipeline_g = Pipeline(
            steps=[
                ("preprocess", preprocessor_g),
                ("model", model),
            ]
        )

        train_idx, test_idx = next(gss.split(X_g, y_g, groups=groups))
        X_train_g, X_test_g = X_g.iloc[train_idx], X_g.iloc[test_idx]
        y_train_g, y_test_g = y_g.iloc[train_idx], y_g.iloc[test_idx]

        # Train on some Agriblocks, test on different held-out Agriblocks.
        pipeline_g.fit(X_train_g, y_train_g)
        preds_g = pipeline_g.predict(X_test_g)

        mae_g = mean_absolute_error(y_test_g, preds_g)
        rmse_g = np.sqrt(mean_squared_error(y_test_g, preds_g))
        r2_g = r2_score(y_test_g, preds_g)

        print("\nGroup-based split metrics (Agriblock held-out):")
        print(f"  MAE : {mae_g:,.2f}")
        print(f"  RMSE: {rmse_g:,.2f}")
        print(f"  R^2 : {r2_g:.4f}")

        # Leave-one-group-out gives per-Agriblock robustness instead of one split.
        logo = LeaveOneGroupOut()
        maes, rmses, r2s, group_names = [], [], [], []
        for train_idx, test_idx in logo.split(X_g, y_g, groups=groups):
            X_train_lg, X_test_lg = X_g.iloc[train_idx], X_g.iloc[test_idx]
            y_train_lg, y_test_lg = y_g.iloc[train_idx], y_g.iloc[test_idx]
            grp = groups.iloc[test_idx].iloc[0]

            pipeline_g.fit(X_train_lg, y_train_lg)
            preds_lg = pipeline_g.predict(X_test_lg)

            maes.append(mean_absolute_error(y_test_lg, preds_lg))
            rmses.append(np.sqrt(mean_squared_error(y_test_lg, preds_lg)))
            r2s.append(r2_score(y_test_lg, preds_lg))
            group_names.append(grp)

        # Report each group's metrics plus mean/std to summarize variability.
        print("\nLeave-one-Agriblock-out metrics:")
        for g, m, r, r2 in zip(group_names, maes, rmses, r2s, strict=False):
            print(f"  {g}: MAE {m:,.2f} | RMSE {r:,.2f} | R^2 {r2:.4f}")
        print(
            f"  Mean: MAE {np.mean(maes):,.2f} | RMSE {np.mean(rmses):,.2f} | R^2 {np.mean(r2s):.4f}"
        )
        print(
            f"  Std : MAE {np.std(maes):,.2f} | RMSE {np.std(rmses):,.2f} | R^2 {np.std(r2s):.4f}"
        )

    # Export top feature importances for interpretability.
    # Note: tree importances are relative/approximate and can be biased for high-cardinality features.
    try:
        feature_names = pipeline.named_steps["preprocess"].get_feature_names_out()
        importances = pipeline.named_steps["model"].feature_importances_
        top_idx = np.argsort(importances)[-15:][::-1]
        top_feats = pd.Series(importances[top_idx], index=feature_names[top_idx])
        top_feats.to_csv(OUT_DIR / "top_feature_importances.csv")
    except Exception as exc:
        print("Skipped feature importances:", exc)


if __name__ == "__main__":
    main()
