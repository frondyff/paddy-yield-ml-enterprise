"""Shared utilities and data contracts for CAROB AMAZXA pipelines."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pandas as pd

try:
    project_root = Path(__file__).resolve().parents[3]
except NameError:
    project_root = Path.cwd()

DATA_PATH = project_root / "data" / "input" / "carob_amazxa.csv"
DICT_PATH = project_root / "data" / "metadata" / "data_dictionary_carob_amazxa.csv"
META_PATH = project_root / "data" / "metadata" / "carob_amazxa_meta.csv"

TARGET_COL = "yield"
GROUP_COL = "trial_id"
TREATMENT_COL = "treatment"
TREATMENT_POS = "+P"
TREATMENT_NEG = "-P"

IDENTIFIER_COLS = {"dataset_id", "record_id"}
POST_OUTCOME_COLS = {"grain_P", "residue_P", "dmy_residue"}


def clean_columns(cols: Iterable[str]) -> list[str]:
    return [" ".join(str(c).strip().split()) for c in cols]


def dedupe_keep_order(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            out.append(item)
            seen.add(item)
    return out


def to_bool(series: pd.Series) -> pd.Series:
    mapping = {
        "true": True,
        "false": False,
        "yes": True,
        "no": False,
        "1": True,
        "0": False,
    }
    s = series.astype("string").str.strip().str.lower()
    out = s.map(mapping)
    return out


def safe_corr(x: pd.Series, y: pd.Series, method: str = "spearman") -> float:
    valid = x.notna() & y.notna()
    if int(valid.sum()) < 5:
        return float("nan")
    xv = pd.to_numeric(x[valid], errors="coerce")
    yv = pd.to_numeric(y[valid], errors="coerce")
    valid_xy = xv.notna() & yv.notna()
    if int(valid_xy.sum()) < 5:
        return float("nan")
    xv = xv[valid_xy]
    yv = yv[valid_xy]
    if int(xv.nunique(dropna=True)) < 2 or int(yv.nunique(dropna=True)) < 2:
        return float("nan")
    return float(xv.corr(yv, method=method))


def eta_squared(y: pd.Series, categories: pd.Series) -> float:
    valid = y.notna() & categories.notna()
    if int(valid.sum()) < 5:
        return float("nan")
    yv = pd.to_numeric(y[valid], errors="coerce")
    cv = categories[valid].astype(str)
    valid2 = yv.notna() & cv.notna()
    yv = yv[valid2]
    cv = cv[valid2]
    if len(yv) < 5 or int(cv.nunique(dropna=True)) < 2:
        return float("nan")

    mean_total = float(yv.mean())
    ss_total = float(((yv - mean_total) ** 2).sum())
    if ss_total <= 1e-12:
        return float("nan")

    ss_between = 0.0
    for _, grp in yv.groupby(cv):
        ss_between += len(grp) * float((grp.mean() - mean_total) ** 2)
    return float(ss_between / ss_total)


def load_metadata() -> pd.DataFrame:
    if not META_PATH.exists():
        raise FileNotFoundError(f"Missing CAROB meta file: {META_PATH}")
    return pd.read_csv(META_PATH)


def load_role_map(frame_cols: list[str] | None = None) -> pd.DataFrame:
    if not DICT_PATH.exists():
        raise FileNotFoundError(f"Missing CAROB dictionary file: {DICT_PATH}")
    dd = pd.read_csv(DICT_PATH)
    required = {
        "column_name",
        "feature_group",
        "data_type",
        "unit_or_levels",
        "meaning",
        "actionability",
        "leakage_risk",
        "modeling_recommendation",
        "final_role",
    }
    missing = required - set(dd.columns)
    if missing:
        raise ValueError(f"CAROB dictionary missing columns: {sorted(missing)}")

    dd = dd.copy()
    dd["column_name"] = dd["column_name"].astype(str)
    dd = dd.drop_duplicates(subset=["column_name"], keep="first")

    if frame_cols is None:
        return dd

    present = set(frame_cols)
    mapped = set(dd["column_name"])
    missing_rows = sorted(present - mapped)
    if missing_rows:
        add_rows = []
        for c in missing_rows:
            add_rows.append(
                {
                    "column_name": c,
                    "feature_group": "Unmapped",
                    "data_type": "numeric" if c in frame_cols else "unknown",
                    "unit_or_levels": "unknown",
                    "meaning": f"Auto-generated placeholder for {c}",
                    "evidence_basis": "auto",
                    "actionability": "depends_on_operational_context",
                    "leakage_risk": "none",
                    "modeling_recommendation": "review",
                    "final_role": "context",
                    "missing_pct": float("nan"),
                    "n_unique": float("nan"),
                }
            )
        dd = pd.concat([dd, pd.DataFrame(add_rows)], ignore_index=True)

    dd = dd[dd["column_name"].isin(present)].copy()
    dd = dd.set_index("column_name").loc[frame_cols].reset_index()
    return dd


def load_analysis_frame(require_treatment: bool = True) -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing CAROB data file: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    df.columns = clean_columns(df.columns)

    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column not found: {TARGET_COL}")
    if GROUP_COL not in df.columns:
        raise ValueError(f"Group column not found: {GROUP_COL}")

    for col in ["flooded", "on_farm", "is_survey", "irrigated", "geo_from_source"]:
        if col in df.columns:
            b = to_bool(df[col])
            if b.notna().any():
                df[col] = b

    for col in df.columns:
        if col in {"dataset_id", "treatment", "crop", "variety", "country", "location", "season", "yield_part"}:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            continue
        num = pd.to_numeric(df[col], errors="coerce")
        if float(num.notna().mean()) >= 0.9:
            df[col] = num

    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")
    df = df[df[TARGET_COL].notna()].copy()
    df[GROUP_COL] = df[GROUP_COL].astype(str)

    if require_treatment:
        if TREATMENT_COL not in df.columns:
            raise ValueError(f"Treatment column not found: {TREATMENT_COL}")
        df[TREATMENT_COL] = df[TREATMENT_COL].astype("string").str.strip()
        df = df[df[TREATMENT_COL].isin([TREATMENT_POS, TREATMENT_NEG])].copy()

    df = df.drop_duplicates().reset_index(drop=True)
    return df
