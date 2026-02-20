from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd

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

MISSING_TOKEN = "__MISSING__"
UNKNOWN_TOKEN = "__UNKNOWN__"


def clean_columns(cols: Iterable[str]) -> list[str]:
    return [" ".join(str(c).strip().split()) for c in cols]


def normalize_per_hectare(
    df: pd.DataFrame,
    drop_original: bool = True,
    create_input_scaled: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    if SIZE_COL not in df.columns:
        raise ValueError(f"Size column not found: {SIZE_COL}")
    if RAW_TARGET_COL not in df.columns:
        raise ValueError(f"Target column not found: {RAW_TARGET_COL}")

    out = df.copy()
    hectares = pd.to_numeric(out[SIZE_COL], errors="coerce")
    out[TARGET_COL] = pd.to_numeric(out[RAW_TARGET_COL], errors="coerce") / hectares

    created_cols: list[str] = []
    if create_input_scaled:
        for col in SIZE_SCALED_COLS:
            if col in out.columns:
                new_col = f"{col}_per_hectare"
                out[new_col] = pd.to_numeric(out[col], errors="coerce") / hectares
                if drop_original:
                    out = out.drop(columns=[col])
                created_cols.append(new_col)
    return out, created_cols


@dataclass
class FeaturePreparer:
    drop_original_size_scaled_cols: bool = True
    create_size_scaled_inputs: bool = True
    drop_exact_duplicates: bool = True
    leakage_drop_cols: tuple[str, ...] = ("Trash(in bundles)",)
    fitted_: bool = False
    feature_columns_: list[str] = field(default_factory=list)
    numeric_cols_: list[str] = field(default_factory=list)
    categorical_cols_: list[str] = field(default_factory=list)
    numeric_fill_values_: dict[str, float] = field(default_factory=dict)
    categorical_levels_: dict[str, list[str]] = field(default_factory=dict)
    output_columns_: list[str] = field(default_factory=list)

    def _base_prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out.columns = clean_columns(out.columns)
        if self.drop_exact_duplicates:
            out = out.drop_duplicates().reset_index(drop=True)
        out, _ = normalize_per_hectare(
            out,
            drop_original=self.drop_original_size_scaled_cols,
            create_input_scaled=self.create_size_scaled_inputs,
        )
        return out

    def _extract_xy(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        if TARGET_COL not in df.columns:
            raise ValueError(f"Missing target column after preparation: {TARGET_COL}")

        X = df.drop(columns=[TARGET_COL]).copy()
        if RAW_TARGET_COL in X.columns:
            X = X.drop(columns=[RAW_TARGET_COL])

        for col in self.leakage_drop_cols:
            if col in X.columns:
                X = X.drop(columns=[col])

        y = pd.to_numeric(df[TARGET_COL], errors="coerce")
        return X, y

    def fit(self, df: pd.DataFrame) -> "FeaturePreparer":
        prepared = self._base_prepare(df)
        X, _ = self._extract_xy(prepared)

        self.feature_columns_ = list(X.columns)
        self.numeric_cols_ = X.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols_ = [c for c in self.feature_columns_ if c not in self.numeric_cols_]

        self.numeric_fill_values_ = {}
        for col in self.numeric_cols_:
            series = pd.to_numeric(X[col], errors="coerce")
            median = float(series.median()) if series.notna().any() else 0.0
            self.numeric_fill_values_[col] = median

        self.categorical_levels_ = {}
        for col in self.categorical_cols_:
            series = X[col].astype("string").fillna(MISSING_TOKEN)
            levels = sorted(series.unique().tolist())
            if UNKNOWN_TOKEN not in levels:
                levels.append(UNKNOWN_TOKEN)
            self.categorical_levels_[col] = levels

        output_columns: list[str] = []
        output_columns.extend(self.numeric_cols_)
        for col in self.categorical_cols_:
            output_columns.extend([f"{col}_{level}" for level in self.categorical_levels_[col]])
        self.output_columns_ = output_columns

        self.fitted_ = True
        return self

    def _transform_X(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted_:
            raise RuntimeError("FeaturePreparer must be fitted before transform")

        out = X.copy()
        for col in self.feature_columns_:
            if col not in out.columns:
                out[col] = np.nan
        out = out[self.feature_columns_]

        numeric_parts: list[pd.DataFrame] = []
        if self.numeric_cols_:
            num = out[self.numeric_cols_].apply(pd.to_numeric, errors="coerce")
            for col, fill_value in self.numeric_fill_values_.items():
                num[col] = num[col].fillna(fill_value)
            numeric_parts.append(num.astype(float))

        cat_parts: list[pd.DataFrame] = []
        for col in self.categorical_cols_:
            levels = self.categorical_levels_[col]
            series = out[col].astype("string").fillna(MISSING_TOKEN)
            known_levels = set(levels)
            series = series.where(series.isin(known_levels), UNKNOWN_TOKEN)

            dtype = pd.CategoricalDtype(categories=levels, ordered=True)
            series = series.astype(dtype)
            dummies = pd.get_dummies(series, prefix=col, dtype=float)

            expected_cols = [f"{col}_{level}" for level in levels]
            dummies = dummies.reindex(columns=expected_cols, fill_value=0.0)
            cat_parts.append(dummies)

        parts = numeric_parts + cat_parts
        Xt = pd.concat(parts, axis=1) if parts else pd.DataFrame(index=out.index)
        Xt = Xt.reindex(columns=self.output_columns_, fill_value=0.0)
        return Xt

    def transform(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        prepared = self._base_prepare(df)
        X, y = self._extract_xy(prepared)
        Xt = self._transform_X(X)
        return Xt, y

    def fit_transform(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        self.fit(df)
        return self.transform(df)

    def metadata(self) -> dict[str, object]:
        if not self.fitted_:
            raise RuntimeError("FeaturePreparer has not been fitted")
        return {
            "feature_columns": self.feature_columns_,
            "numeric_cols": self.numeric_cols_,
            "categorical_cols": self.categorical_cols_,
            "output_columns": self.output_columns_,
            "numeric_fill_values": self.numeric_fill_values_,
            "categorical_levels": self.categorical_levels_,
        }
