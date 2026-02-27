"""CAROB what-if decision support UI with optional scientific evidence mode."""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
from catboost import CatBoostRegressor
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

# -----------------------------------------------------------------------------
# Project root and imports
# -----------------------------------------------------------------------------


def _resolve_project_root() -> Path:
    candidates: list[Path] = []

    # 1) File location (preferred)
    try:
        here = Path(__file__).resolve()
        candidates.extend([here.parent] + list(here.parents))
    except NameError:
        pass

    # 2) Current working directory
    cwd = Path.cwd().resolve()
    candidates.extend([cwd] + list(cwd.parents))

    seen: set[Path] = set()
    for base in candidates:
        if base in seen:
            continue
        seen.add(base)
        if (base / "pyproject.toml").exists() and (base / "src").exists():
            return base

    raise RuntimeError(
        "Could not find project root. Expected a directory containing pyproject.toml and src/. "
        "Run Streamlit from within the project workspace."
    )


PROJECT_ROOT = _resolve_project_root()
SRC_DIR = PROJECT_ROOT / "src"
if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from paddy_yield_ml.pipelines import carob_common as cc  # noqa: E402
from paddy_yield_ml.pipelines import carob_model_compare as cm  # noqa: E402

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

DEFAULT_SCENARIO = "modifiable_plus_context"
DEFAULT_MODEL_COMPARE_DIR = "carob_model_compare"
DEFAULT_INTERPRETABILITY_RUN_TAG = "iter3_defensible_v5"
DEFAULT_CAUSAL_RUN_TAG = "rule_aipw_v2"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
CANDIDATES_PATH = OUTPUTS_DIR / "carob_feature_prepare" / "hybrid_selection_candidates.csv"
DICT_PATH = PROJECT_ROOT / "data" / "metadata" / "data_dictionary_carob_amazxa.csv"
RUN_MANIFEST_PATH = OUTPUTS_DIR / "run_all_manifest.json"

INTERPRETABILITY_BASE = OUTPUTS_DIR / "carob_interpretability"
CAUSAL_RULE_BASE = OUTPUTS_DIR / "carob_rule_causal_aipw"
CAUSAL_PILOT_BASE = OUTPUTS_DIR / "carob_causal_pilot"

COND_RE = re.compile(r"^(?P<feature>.+?)\s*(?P<op><=|>=|<|>)\s*(?P<threshold>-?\d+(?:\.\d+)?)$")


def _manifest_defaults() -> dict[str, str]:
    defaults = {
        "model_compare_dir": DEFAULT_MODEL_COMPARE_DIR,
        "interpretability_tag": DEFAULT_INTERPRETABILITY_RUN_TAG,
        "causal_tag": DEFAULT_CAUSAL_RUN_TAG,
    }
    if not RUN_MANIFEST_PATH.exists():
        return defaults

    try:
        manifest = json.loads(RUN_MANIFEST_PATH.read_text(encoding="utf-8"))
    except Exception:
        return defaults

    run_tags = manifest.get("run_tags", {})
    outputs = manifest.get("outputs", {})

    interp_tag = str(run_tags.get("interpretability", "")).strip()
    causal_tag = str(run_tags.get("causal", "")).strip()
    model_compare_output = str(outputs.get("model_compare", "")).strip()

    if interp_tag:
        defaults["interpretability_tag"] = interp_tag
    if causal_tag:
        defaults["causal_tag"] = causal_tag
    if model_compare_output:
        # Manifest may store a full relative path like outputs/carob_model_compare.
        defaults["model_compare_dir"] = Path(model_compare_output).name

    return defaults


MANIFEST_DEFAULTS = _manifest_defaults()


# -----------------------------------------------------------------------------
# Data classes
# -----------------------------------------------------------------------------


@dataclass
class InterpretabilityAssets:
    run_dir: Path | None
    shap_df: pd.DataFrame
    shap_path: Path | None
    shap_img_path: Path | None
    top_note: str
    top_note_path: Path | None
    rules_df: pd.DataFrame
    rules_path: Path | None
    rule_country_df: pd.DataFrame
    rule_country_path: Path | None
    top_features_df: pd.DataFrame
    top_features_path: Path | None
    local_examples_df: pd.DataFrame
    local_examples_path: Path | None
    confidence_df: pd.DataFrame
    confidence_path: Path | None
    runlog_path: Path | None
    diagnostics: list[str]


@dataclass
class CausalAssets:
    source: str
    run_dir: Path | None
    scorecard_df: pd.DataFrame
    scorecard_path: Path | None
    pair_summary_df: pd.DataFrame
    pair_summary_path: Path | None
    overall_df: pd.DataFrame
    overall_path: Path | None
    diagnostics: list[str]


# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------


def _fmt_timestamp(path: Path | None) -> str:
    if path is None or not path.exists():
        return "n/a"
    ts = datetime.fromtimestamp(path.stat().st_mtime)
    return ts.strftime("%Y-%m-%d %H:%M:%S")


def _safe_read_text(path: Path | None) -> str:
    if path is None or not path.exists():
        return ""
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""


def _safe_read_csv(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def _pick_run_dir(base_dir: Path, preferred_tag: str, label: str, diagnostics: list[str]) -> Path | None:
    if not base_dir.exists():
        diagnostics.append(f"{label}: base directory not found: {base_dir}")
        return None

    preferred = base_dir / preferred_tag
    if preferred.exists() and preferred.is_dir():
        return preferred

    latest = base_dir / "latest"
    if latest.exists() and latest.is_dir():
        diagnostics.append(f"{label}: requested tag '{preferred_tag}' not found; using 'latest'.")
        return latest

    run_dirs = [d for d in base_dir.iterdir() if d.is_dir()]
    if not run_dirs:
        diagnostics.append(f"{label}: no run directories found under {base_dir}")
        return None

    run_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    chosen = run_dirs[0]
    diagnostics.append(f"{label}: requested tag '{preferred_tag}' not found; using newest run '{chosen.name}'.")
    return chosen


def _read_csv_candidates(
    candidates: list[Path],
    *,
    required_cols: set[str] | None,
    label: str,
    diagnostics: list[str],
) -> tuple[pd.DataFrame, Path | None]:
    for path in candidates:
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path)
        except Exception as exc:
            diagnostics.append(f"{label}: failed to read {path.name}: {type(exc).__name__}: {exc}")
            continue

        if required_cols:
            missing = required_cols - set(df.columns)
            if missing:
                diagnostics.append(
                    f"{label}: {path.name} missing required columns {sorted(missing)}; skipping this file."
                )
                continue

        return df, path

    diagnostics.append(f"{label}: no compatible file found among {[p.name for p in candidates]}")
    return pd.DataFrame(), None


# -----------------------------------------------------------------------------
# Dictionary / labels
# -----------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def load_feature_dictionary(path_str: str) -> pd.DataFrame:
    path = Path(path_str)
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
    except Exception:
        return pd.DataFrame()
    if "column_name" not in df.columns:
        return pd.DataFrame()
    return df.drop_duplicates(subset=["column_name"], keep="first").copy()


def _fallback_feature_title(feature: str) -> str:
    return feature.replace("_", " ").replace("-", " ").strip().title()


def feature_display_name(feature: str, dd: pd.DataFrame) -> str:
    if dd.empty or feature not in set(dd["column_name"].astype(str)):
        return _fallback_feature_title(feature)

    row = dd.loc[dd["column_name"].astype(str) == feature].iloc[0]
    meaning = str(row.get("meaning", "")).strip()
    if meaning and meaning.lower() not in {"nan", "none"}:
        return meaning
    return _fallback_feature_title(feature)


def feature_help_text(feature: str, dd: pd.DataFrame) -> str:
    if dd.empty or feature not in set(dd["column_name"].astype(str)):
        return f"Variable: {feature}"
    row = dd.loc[dd["column_name"].astype(str) == feature].iloc[0]
    meaning = str(row.get("meaning", "")).strip()
    unit = str(row.get("unit_or_levels", "")).strip()
    role = str(row.get("final_role", "")).strip()
    parts = [f"Variable: {feature}"]
    if meaning and meaning.lower() not in {"nan", "none"}:
        parts.append(f"Meaning: {meaning}")
    if unit and unit.lower() not in {"nan", "none", "unknown"}:
        parts.append(f"Unit/Levels: {unit}")
    if role and role.lower() not in {"nan", "none"}:
        parts.append(f"Role: {role}")
    return " | ".join(parts)


# -----------------------------------------------------------------------------
# Model loading and training
# -----------------------------------------------------------------------------


def _model_summary_candidates(model_compare_dir: str) -> list[Path]:
    primary = OUTPUTS_DIR / model_compare_dir / "model_comparison_summary.csv"
    fallback = OUTPUTS_DIR / "model_compare" / "model_comparison_summary.csv"
    if model_compare_dir == "model_compare":
        return [primary, OUTPUTS_DIR / "carob_model_compare" / "model_comparison_summary.csv"]
    return [primary, fallback]


def _load_model_summary(model_compare_dir: str) -> tuple[pd.DataFrame, Path | None, list[str]]:
    diagnostics: list[str] = []
    candidates = _model_summary_candidates(model_compare_dir)

    df, path = _read_csv_candidates(
        candidates,
        required_cols={"model", "model_key", "params_json", "mae", "rmse", "r2"},
        label="Model comparison",
        diagnostics=diagnostics,
    )

    if not df.empty:
        df = df.copy()
        df["model_key"] = df["model_key"].astype(str).str.lower()
        if "scenario" not in df.columns:
            df["scenario"] = DEFAULT_SCENARIO

    return df, path, diagnostics


def _select_catboost_params(summary_df: pd.DataFrame, scenario: str) -> dict[str, Any]:
    if summary_df.empty:
        return {"n_estimators": 300, "learning_rate": 0.05, "depth": 6, "l2_leaf_reg": 3.0}

    rows = summary_df.copy()
    rows["scenario"] = rows["scenario"].astype(str)
    rows = rows[rows["model_key"].astype(str).str.lower() == "catboost"].copy()
    if rows.empty:
        return {"n_estimators": 300, "learning_rate": 0.05, "depth": 6, "l2_leaf_reg": 3.0}

    rows_scenario = rows[rows["scenario"] == scenario].copy()
    if not rows_scenario.empty:
        rows = rows_scenario

    rows = rows.sort_values(["r2", "rmse"], ascending=[False, True]).reset_index(drop=True)
    raw = str(rows.loc[0, "params_json"])
    try:
        params = json.loads(raw)
    except json.JSONDecodeError:
        params = {}

    if not isinstance(params, dict):
        params = {}
    if "iterations" in params and "n_estimators" not in params:
        params["n_estimators"] = int(params["iterations"])

    return {
        "n_estimators": int(params.get("n_estimators", 300)),
        "learning_rate": float(params.get("learning_rate", 0.05)),
        "depth": int(params.get("depth", 6)),
        "l2_leaf_reg": float(params.get("l2_leaf_reg", 3.0)),
    }


def _build_display_profiles(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy().reset_index(drop=True)
    out["_profile_id"] = out.index
    trial_vals = out[cc.GROUP_COL].astype(str) if cc.GROUP_COL in out.columns else "n/a"
    country_vals = out["country"].astype(str) if "country" in out.columns else "n/a"
    location_vals = out["location"].astype(str) if "location" in out.columns else "n/a"
    out["_profile_label"] = (
        "Profile "
        + out["_profile_id"].astype(str)
        + " | Trial "
        + trial_vals
        + " | "
        + country_vals
        + " / "
        + location_vals
    )
    return out


def _parse_onehot_feature(raw_feature: str, cat_cols: list[str]) -> tuple[str | None, str | None]:
    for col in sorted(cat_cols, key=len, reverse=True):
        prefix = f"{col}_"
        if raw_feature.startswith(prefix):
            return col, raw_feature[len(prefix) :]
    return None, None


def _format_split_condition(
    feature_name: str,
    threshold: float,
    *,
    is_right_branch: bool,
    cat_cols: list[str],
    bool_cols: list[str],
) -> str:
    if "__" in feature_name:
        _, raw = feature_name.split("__", 1)
    else:
        raw = feature_name

    cat_col, cat_value = _parse_onehot_feature(raw, cat_cols)
    if cat_col is not None and cat_value is not None:
        if abs(float(threshold) - 0.5) <= 0.51:
            return f"{cat_col} == {cat_value}" if is_right_branch else f"{cat_col} != {cat_value}"
        op = ">" if is_right_branch else "<="
        return f"{cat_col}:{cat_value} {op} {float(threshold):.3f}"

    if raw in bool_cols and abs(float(threshold) - 0.5) <= 0.51:
        return f"{raw} = True" if is_right_branch else f"{raw} = False"

    op = ">" if is_right_branch else "<="
    return f"{raw} {op} {float(threshold):.3f}"


def _extract_leaf_conditions(
    tree: DecisionTreeRegressor,
    feature_names: list[str],
    *,
    cat_cols: list[str],
    bool_cols: list[str],
) -> dict[int, list[str]]:
    children_left = tree.tree_.children_left
    children_right = tree.tree_.children_right
    feature_idx = tree.tree_.feature
    thresholds = tree.tree_.threshold
    leaves: dict[int, list[str]] = {}

    def walk(node_id: int, conds: list[str]) -> None:
        left = int(children_left[node_id])
        right = int(children_right[node_id])
        if left == right:
            leaves[int(node_id)] = conds
            return

        idx = int(feature_idx[node_id])
        fname = feature_names[idx]
        thresh = float(thresholds[node_id])

        left_cond = _format_split_condition(
            fname,
            thresh,
            is_right_branch=False,
            cat_cols=cat_cols,
            bool_cols=bool_cols,
        )
        right_cond = _format_split_condition(
            fname,
            thresh,
            is_right_branch=True,
            cat_cols=cat_cols,
            bool_cols=bool_cols,
        )
        walk(left, conds + [left_cond])
        walk(right, conds + [right_cond])

    walk(0, [])
    return leaves


def _build_surrogate_rules(
    modeling: pd.DataFrame,
    *,
    features: list[str],
    modifiable: list[str],
    pipeline: Pipeline,
) -> dict[str, Any]:
    x_mod = modeling[modifiable].copy()
    y_pred = pd.Series(pipeline.predict(modeling[features]))
    y_actual = pd.to_numeric(modeling[cc.TARGET_COL], errors="coerce")

    pre_mod = cm.make_preprocessor(x_mod)
    x_mod_trans = pre_mod.fit_transform(x_mod)

    min_samples_leaf = max(25, int(round(0.05 * len(x_mod))))
    tree = DecisionTreeRegressor(max_depth=3, min_samples_leaf=min_samples_leaf, random_state=42)
    tree.fit(x_mod_trans, y_pred)

    feature_names = pre_mod.get_feature_names_out().tolist()
    cat_cols = x_mod.select_dtypes(exclude=["number", "boolean"]).columns.astype(str).tolist()
    bool_cols = [c for c in x_mod.columns if pd.api.types.is_bool_dtype(x_mod[c])]
    leaf_conds = _extract_leaf_conditions(tree, feature_names, cat_cols=cat_cols, bool_cols=bool_cols)

    leaf_ids = pd.Series(tree.apply(x_mod_trans)).astype(int)
    pred_global = float(y_pred.mean())
    actual_global = float(y_actual.mean())

    rows: list[dict[str, Any]] = []
    for leaf_id in sorted(leaf_ids.unique().tolist()):
        mask = leaf_ids == leaf_id
        conds = leaf_conds.get(int(leaf_id), [])
        pred_mean = float(y_pred[mask].mean())
        actual_mean = float(y_actual[mask].mean())
        rows.append(
            {
                "rule_id": f"S{len(rows) + 1}",
                "leaf_id": int(leaf_id),
                "rule_conditions": " AND ".join(conds) if conds else "ALL",
                "support_count": int(mask.sum()),
                "support_pct": float(mask.mean()),
                "predicted_lift_vs_global": float(pred_mean - pred_global),
                "actual_lift_vs_global": float(actual_mean - actual_global),
                "confidence_level": "Low",
                "caveat_labels": "surrogate_rule,associative_only",
            }
        )

    rules = pd.DataFrame(rows).sort_values(
        ["predicted_lift_vs_global", "support_count"],
        ascending=[False, False],
    )
    rules = rules.reset_index(drop=True)

    return {
        "preprocessor": pre_mod,
        "tree": tree,
        "rules": rules,
    }


@st.cache_resource(show_spinner="Loading governed CAROB model...")
def load_serving_bundle(project_root_str: str, scenario: str, model_compare_dir: str) -> dict[str, Any]:
    root = Path(project_root_str)
    diagnostics: list[str] = []

    candidates_path = root / "outputs" / "carob_feature_prepare" / "hybrid_selection_candidates.csv"
    if not candidates_path.exists():
        raise FileNotFoundError(f"Missing feature candidates file: {candidates_path}")

    summary_df, summary_path, summary_diag = _load_model_summary(model_compare_dir)
    diagnostics.extend(summary_diag)

    frame = cm.load_frame_with_country_gate(candidates_path)
    frame = frame.loc[:, ~frame.columns.duplicated()].copy()

    candidates = cm.load_hybrid_candidates(candidates_path)
    feature_sets = cm.build_feature_sets(candidates)
    if scenario not in feature_sets:
        raise ValueError(f"Scenario '{scenario}' not available. Found: {sorted(feature_sets)}")

    features = cm.filter_available_features(feature_sets[scenario], frame)

    modeling_cols = cm.dedupe_keep_order(features + [cc.TARGET_COL, cc.GROUP_COL, "country", "location", "variety"])
    modeling_cols = [c for c in modeling_cols if c in frame.columns]
    modeling = frame[modeling_cols].copy()
    modeling = modeling.loc[:, ~modeling.columns.duplicated()].copy()
    modeling = modeling[modeling[cc.TARGET_COL].notna() & modeling[cc.GROUP_COL].notna()].reset_index(drop=True)

    modifiable = (
        candidates.loc[candidates["status"].astype(str) == "candidate_modifiable", "feature"].astype(str).tolist()
    )
    modifiable = [f for f in cm.dedupe_keep_order(modifiable) if f in features]
    context = [f for f in features if f not in modifiable]

    params = _select_catboost_params(summary_df=summary_df, scenario=scenario)
    model = CatBoostRegressor(
        n_estimators=int(params.get("n_estimators", 300)),
        learning_rate=float(params.get("learning_rate", 0.05)),
        depth=int(params.get("depth", 6)),
        l2_leaf_reg=float(params.get("l2_leaf_reg", 3.0)),
        loss_function="RMSE",
        random_seed=42,
        verbose=0,
        allow_writing_files=False,
    )

    preprocessor = cm.make_preprocessor(modeling[features])
    pipeline = Pipeline([("preprocess", preprocessor), ("model", model)])
    pipeline.fit(modeling[features], modeling[cc.TARGET_COL])

    surrogate_rules = _build_surrogate_rules(
        modeling=modeling,
        features=features,
        modifiable=modifiable,
        pipeline=pipeline,
    )

    return {
        "pipeline": pipeline,
        "modeling": modeling,
        "features": features,
        "modifiable": modifiable,
        "context": context,
        "params": params,
        "summary_df": summary_df,
        "summary_path": summary_path,
        "surrogate_rules": surrogate_rules,
        "diagnostics": diagnostics,
    }


# -----------------------------------------------------------------------------
# Interpretability and causal loaders
# -----------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def load_interpretability_assets(project_root_str: str, run_tag: str) -> InterpretabilityAssets:
    _ = project_root_str
    diagnostics: list[str] = []
    run_dir = _pick_run_dir(INTERPRETABILITY_BASE, run_tag, "Interpretability", diagnostics)

    if run_dir is None:
        return InterpretabilityAssets(
            run_dir=None,
            shap_df=pd.DataFrame(),
            shap_path=None,
            shap_img_path=None,
            top_note="",
            top_note_path=None,
            rules_df=pd.DataFrame(),
            rules_path=None,
            rule_country_df=pd.DataFrame(),
            rule_country_path=None,
            top_features_df=pd.DataFrame(),
            top_features_path=None,
            local_examples_df=pd.DataFrame(),
            local_examples_path=None,
            confidence_df=pd.DataFrame(),
            confidence_path=None,
            runlog_path=None,
            diagnostics=diagnostics,
        )

    shap_df, shap_path = _read_csv_candidates(
        [
            run_dir / "iteration1_global_shap_importance.csv",
            run_dir / "global_shap_importance.csv",
        ],
        required_cols={"feature", "mean_abs_shap"},
        label="Interpretability SHAP table",
        diagnostics=diagnostics,
    )
    if not shap_df.empty:
        shap_df = shap_df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    rules_df, rules_path = _read_csv_candidates(
        [
            run_dir / "ui_rules.csv",
            run_dir / "iteration3_rules_final.csv",
        ],
        required_cols={"rule_id", "rule_conditions"},
        label="Interpretability rules",
        diagnostics=diagnostics,
    )

    country_df, country_path = _read_csv_candidates(
        [run_dir / "iteration3_rule_country_generalization.csv"],
        required_cols={"rule_id", "country", "generalization_status"},
        label="Rule country generalization",
        diagnostics=diagnostics,
    )

    top_features_df, top_features_path = _read_csv_candidates(
        [run_dir / "ui_top_features.csv"],
        required_cols={"feature"},
        label="UI top features",
        diagnostics=diagnostics,
    )

    local_df, local_path = _read_csv_candidates(
        [
            run_dir / "ui_local_examples.csv",
            run_dir / "local_examples_overview.csv",
            run_dir / "iteration1_local_cases_overview.csv",
        ],
        required_cols=None,
        label="Local examples",
        diagnostics=diagnostics,
    )

    conf_df, conf_path = _read_csv_candidates(
        [run_dir / "ui_confidence_labels.csv"],
        required_cols={"rule_id", "confidence_level", "caveat_labels"},
        label="Rule confidence labels",
        diagnostics=diagnostics,
    )

    top_note_path = _first_existing(
        [
            run_dir / "iteration3_rules_final_english.md",
            run_dir / "top_feature_explanation.md",
        ]
    )
    top_note = _safe_read_text(top_note_path)

    shap_img_path = _first_existing(
        [
            run_dir / "iteration1_global_shap_top.png",
            run_dir / "global_shap_top_features.png",
        ]
    )

    runlog_path = _first_existing(
        [
            run_dir / "interpretability_runlog.txt",
            run_dir / "carob_interpretability_runlog.txt",
        ]
    )

    return InterpretabilityAssets(
        run_dir=run_dir,
        shap_df=shap_df,
        shap_path=shap_path,
        shap_img_path=shap_img_path,
        top_note=top_note,
        top_note_path=top_note_path,
        rules_df=rules_df,
        rules_path=rules_path,
        rule_country_df=country_df,
        rule_country_path=country_path,
        top_features_df=top_features_df,
        top_features_path=top_features_path,
        local_examples_df=local_df,
        local_examples_path=local_path,
        confidence_df=conf_df,
        confidence_path=conf_path,
        runlog_path=runlog_path,
        diagnostics=diagnostics,
    )


@st.cache_data(show_spinner=False)
def load_causal_assets(project_root_str: str, run_tag: str) -> CausalAssets:
    _ = project_root_str
    diagnostics: list[str] = []

    # Preferred: rule-as-treatment AIPW pipeline
    rule_run_dir = _pick_run_dir(CAUSAL_RULE_BASE, run_tag, "Causal AIPW", diagnostics)
    if rule_run_dir is not None:
        scorecard_df, scorecard_path = _read_csv_candidates(
            [rule_run_dir / "causal_rule_scorecard.csv"],
            required_cols={"rule_id", "country", "recommendation"},
            label="Causal scorecard",
            diagnostics=diagnostics,
        )
        pair_df, pair_path = _read_csv_candidates(
            [rule_run_dir / "pair_aipw_summary.csv"],
            required_cols={"rule_id", "country"},
            label="Pair AIPW summary",
            diagnostics=diagnostics,
        )

        if not scorecard_df.empty:
            return CausalAssets(
                source="rule_aipw",
                run_dir=rule_run_dir,
                scorecard_df=scorecard_df,
                scorecard_path=scorecard_path,
                pair_summary_df=pair_df,
                pair_summary_path=pair_path,
                overall_df=pd.DataFrame(),
                overall_path=None,
                diagnostics=diagnostics,
            )

    # Fallback: causal pilot summary
    pilot_run_dir = _pick_run_dir(CAUSAL_PILOT_BASE, "latest", "Causal pilot", diagnostics)
    if pilot_run_dir is not None:
        overall_df, overall_path = _read_csv_candidates(
            [pilot_run_dir / "causal_overall_summary.csv"],
            required_cols={"confidence_level"},
            label="Causal pilot overall summary",
            diagnostics=diagnostics,
        )
        if not overall_df.empty:
            return CausalAssets(
                source="pilot",
                run_dir=pilot_run_dir,
                scorecard_df=pd.DataFrame(),
                scorecard_path=None,
                pair_summary_df=pd.DataFrame(),
                pair_summary_path=None,
                overall_df=overall_df,
                overall_path=overall_path,
                diagnostics=diagnostics,
            )

    diagnostics.append("No causal artifacts are currently available.")
    return CausalAssets(
        source="none",
        run_dir=None,
        scorecard_df=pd.DataFrame(),
        scorecard_path=None,
        pair_summary_df=pd.DataFrame(),
        pair_summary_path=None,
        overall_df=pd.DataFrame(),
        overall_path=None,
        diagnostics=diagnostics,
    )


# -----------------------------------------------------------------------------
# Input widgets and profile matching
# -----------------------------------------------------------------------------


def _is_numeric_like(series: pd.Series) -> bool:
    numeric = pd.to_numeric(series, errors="coerce")
    return float(numeric.notna().mean()) >= 0.9


def _feature_kind(series: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(series):
        return "bool"
    if _is_numeric_like(series):
        return "numeric"
    return "categorical"


def _numeric_bounds(series: pd.Series) -> tuple[float, float]:
    vals = pd.to_numeric(series, errors="coerce").dropna()
    if vals.empty:
        return 0.0, 1.0
    lo = float(vals.quantile(0.01))
    hi = float(vals.quantile(0.99))
    if lo > hi:
        lo, hi = hi, lo
    if lo == hi:
        lo = float(vals.min())
        hi = float(vals.max())
    if lo == hi:
        return lo, hi
    return lo, hi


def _numeric_unique_levels(series: pd.Series, *, max_levels: int = 20) -> list[float]:
    vals = pd.to_numeric(series, errors="coerce").dropna()
    if vals.empty:
        return []
    uniq = sorted({float(v) for v in vals.to_numpy().tolist()})
    if len(uniq) > int(max_levels):
        return []
    return uniq


def _auto_coord_from_location(
    frame: pd.DataFrame,
    *,
    location_value: Any,
    coord_col: str,
) -> float | None:
    if "location" not in frame.columns or coord_col not in frame.columns:
        return None

    loc_rows = frame[frame["location"].astype(str) == str(location_value)].copy()
    if loc_rows.empty:
        return None

    coord_vals = pd.to_numeric(loc_rows[coord_col], errors="coerce").dropna()
    if coord_vals.empty:
        return None

    levels = sorted({float(v) for v in coord_vals.to_numpy().tolist()})
    if len(levels) == 1:
        return levels[0]
    return None


def _safe_default_numeric(current_value: Any, reference: pd.Series, fallback: float) -> float:
    cur = pd.to_numeric(pd.Series([current_value]), errors="coerce").iloc[0]
    if pd.notna(cur):
        return float(cur)
    vals = pd.to_numeric(reference, errors="coerce").dropna()
    if vals.empty:
        return fallback
    return float(vals.median())


def render_feature_input(
    *,
    feature: str,
    kind: str,
    label: str,
    help_text: str,
    current_value: Any,
    reference_all: pd.Series,
    reference_country: pd.Series,
    key: str,
    context_mode: bool,
) -> Any:
    if kind == "bool":
        if pd.isna(current_value):
            mode = reference_country.dropna().mode()
            if mode.empty:
                mode = reference_all.dropna().mode()
            default = bool(mode.iloc[0]) if not mode.empty else False
        else:
            default = bool(current_value)
        return st.toggle(label, value=default, help=help_text, key=key)

    if kind == "numeric":
        lo, hi = _numeric_bounds(reference_all)
        country_num = pd.to_numeric(reference_country, errors="coerce").dropna()
        all_num = pd.to_numeric(reference_all, errors="coerce").dropna()
        all_levels = _numeric_unique_levels(reference_all, max_levels=10)

        if context_mode:
            context_ref = country_num if not country_num.empty else all_num
            default = _safe_default_numeric(None, context_ref, fallback=(lo + hi) / 2.0)
        else:
            default = _safe_default_numeric(current_value, all_num, fallback=(lo + hi) / 2.0)

        # Sidebar context controls: if a numeric feature is effectively binary/discrete
        # (2 global levels), use a dropdown instead of a continuous slider.
        if context_mode and len(all_levels) == 2:
            low, high = all_levels[0], all_levels[1]
            default_level = low if abs(default - low) <= abs(default - high) else high
            if not country_num.empty:
                country_levels = _numeric_unique_levels(country_num, max_levels=10)
                if country_levels:
                    options = [v for v in all_levels if v in set(country_levels)]
                    if options:
                        all_levels = options
                        if default_level not in all_levels:
                            default_level = all_levels[0]
            return st.selectbox(
                label,
                options=all_levels,
                index=all_levels.index(default_level),
                format_func=lambda x: f"{x:g}",
                help=help_text,
                key=key,
            )

        slider_lo, slider_hi = lo, hi
        if context_mode and not country_num.empty:
            c_lo = float(country_num.quantile(0.05))
            c_hi = float(country_num.quantile(0.95))
            if c_lo > c_hi:
                c_lo, c_hi = c_hi, c_lo
            slider_lo = max(lo, c_lo)
            slider_hi = min(hi, c_hi)
            if slider_lo > slider_hi:
                slider_lo, slider_hi = lo, hi
            st.caption(f"Typical range in selected country: {slider_lo:.2f} to {slider_hi:.2f}")

        default = min(max(default, slider_lo), slider_hi)

        if slider_lo == slider_hi:
            st.caption(f"{label}: fixed at {slider_lo:.3f}")
            return slider_lo

        step = max((slider_hi - slider_lo) / 150.0, 0.01)
        return st.slider(
            label,
            min_value=float(slider_lo),
            max_value=float(slider_hi),
            value=float(default),
            step=float(step),
            help=help_text,
            key=key,
        )

    options_country = sorted(reference_country.dropna().astype(str).unique().tolist())
    options_all = sorted(reference_all.dropna().astype(str).unique().tolist())
    options = options_country if context_mode and options_country else options_all
    if not options:
        options = [""]

    default = str(current_value) if pd.notna(current_value) else options[0]
    if context_mode:
        default = options[0]
    if default not in options:
        options = [default] + options

    return st.selectbox(label, options=options, index=options.index(default), help=help_text, key=key)


def _relative_error(series: pd.Series, target: float) -> pd.Series:
    vals = pd.to_numeric(series, errors="coerce")
    span = float(vals.max() - vals.min()) if vals.notna().any() else 0.0
    denom = max(abs(float(target)), 0.1 * span, 1e-6)
    out = (vals - float(target)).abs() / denom
    return out.fillna(float("inf"))


def match_profile_from_context(
    candidates: pd.DataFrame,
    user_context: dict[str, Any],
    *,
    context_features: list[str],
    tolerance: float = 0.10,
) -> tuple[pd.Series, bool, dict[str, float], int]:
    if candidates.empty:
        raise ValueError("No profile candidates available for matching.")

    scores = pd.DataFrame(index=candidates.index)
    pass_mask = pd.Series(True, index=candidates.index)
    numeric_feats: list[str] = []
    categorical_feats: list[str] = []

    for feature in context_features:
        if feature == "country" or feature not in user_context:
            continue

        col = candidates[feature]
        target = user_context[feature]

        if _is_numeric_like(col):
            rel = _relative_error(col, float(target))
            scores[feature] = rel
            pass_mask &= rel <= tolerance
            numeric_feats.append(feature)
        else:
            mismatch = (col.astype(str) != str(target)).astype(float)
            scores[feature] = mismatch
            pass_mask &= mismatch == 0
            categorical_feats.append(feature)

    if pass_mask.any():
        pool = candidates.loc[pass_mask].copy()
        pool_scores = scores.loc[pass_mask].copy()
        within_tolerance = True
    else:
        pool = candidates.copy()
        pool_scores = scores.copy()
        within_tolerance = False

    if numeric_feats:
        num_score = pool_scores[numeric_feats].replace([float("inf")], 1e6).mean(axis=1)
    else:
        num_score = pd.Series(0.0, index=pool.index)

    if categorical_feats:
        cat_score = pool_scores[categorical_feats].mean(axis=1)
    else:
        cat_score = pd.Series(0.0, index=pool.index)

    total_score = num_score + cat_score
    best_idx = total_score.sort_values(ascending=True).index[0]
    best_row = pool.loc[best_idx]

    numeric_errors_pct: dict[str, float] = {}
    for feat in numeric_feats:
        numeric_errors_pct[feat] = float(scores.loc[best_idx, feat] * 100.0)

    cat_mismatch_count = int(scores.loc[best_idx, categorical_feats].sum()) if categorical_feats else 0
    return best_row, within_tolerance, numeric_errors_pct, cat_mismatch_count


# -----------------------------------------------------------------------------
# Rule parsing and evidence mapping
# -----------------------------------------------------------------------------


def parse_rule_conditions(rule_conditions: str) -> list[tuple[str, str, float]]:
    out: list[tuple[str, str, float]] = []
    parts = [p.strip() for p in str(rule_conditions).split("AND") if p.strip()]
    for part in parts:
        m = COND_RE.match(part)
        if not m:
            continue
        feat = m.group("feature").strip()
        op = m.group("op").strip()
        thr = float(m.group("threshold"))
        out.append((feat, op, thr))
    return out


def row_matches_rule(row: pd.Series, rule_conditions: str) -> bool:
    parsed = parse_rule_conditions(rule_conditions)
    if not parsed:
        return False

    for feat, op, thr in parsed:
        if feat not in row.index:
            return False
        val = pd.to_numeric(pd.Series([row[feat]]), errors="coerce").iloc[0]
        if pd.isna(val):
            return False
        if op == "<=":
            ok = float(val) <= thr
        elif op == ">":
            ok = float(val) > thr
        elif op == "<":
            ok = float(val) < thr
        elif op == ">=":
            ok = float(val) >= thr
        else:
            ok = False
        if not ok:
            return False
    return True


def format_rule_text(rule_conditions: str, dd: pd.DataFrame) -> str:
    parsed = parse_rule_conditions(rule_conditions)
    if not parsed:
        return str(rule_conditions)
    parts: list[str] = []
    for feat, op, thr in parsed:
        label = feature_display_name(feat, dd)
        parts.append(f"{label} {op} {thr:.3f}")
    return " and ".join(parts)


def sort_rules_for_display(rules_df: pd.DataFrame) -> pd.DataFrame:
    if rules_df.empty:
        return rules_df

    out = rules_df.copy()
    for col in [
        "support_pct",
        "predicted_lift_vs_global",
        "actual_lift_vs_global",
        "lift_score",
        "support_score",
    ]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    sort_cols: list[str] = []
    ascending: list[bool] = []

    if "rule_score" in out.columns:
        sort_cols.append("rule_score")
        ascending.append(False)
    if "support_score" in out.columns:
        sort_cols.append("support_score")
        ascending.append(False)
    elif "support_pct" in out.columns:
        sort_cols.append("support_pct")
        ascending.append(False)

    if "lift_score" in out.columns:
        sort_cols.append("lift_score")
        ascending.append(False)
    elif "actual_lift_vs_global" in out.columns:
        sort_cols.append("actual_lift_vs_global")
        ascending.append(False)

    if sort_cols:
        out = out.sort_values(sort_cols, ascending=ascending)

    return out.reset_index(drop=True)


def find_matching_rule(rules_df: pd.DataFrame, row: pd.Series) -> pd.Series | None:
    if rules_df.empty or "rule_conditions" not in rules_df.columns:
        return None

    ordered = sort_rules_for_display(rules_df)
    for _, r in ordered.iterrows():
        rule_text = str(r.get("rule_conditions", ""))
        if row_matches_rule(row, rule_text):
            return r
    return None


def friendly_country_status(status: str) -> str:
    mapping = {
        "works_here": "Historically works in this country",
        "conflicts_here": "Historically conflicts in this country",
        "insufficient_evidence": "Not enough country evidence",
        "unstable_or_small_effect": "Evidence is unstable or too small",
    }
    return mapping.get(status, status)


def friendly_decision_reason(reason: str) -> str:
    mapping = {
        "passes_effect_significance_stability_overlap_balance": (
            "Strong causal diagnostics across effect, overlap, and balance."
        ),
        "promising_signal_but_low_trial_diversity_in_one_arm": (
            "Promising effect but too little trial diversity in one treatment arm."
        ),
        "works_here_but_some_causal_diagnostics_not_fully_met": (
            "Predictive signal transfers, but some causal diagnostics are not fully met."
        ),
        "predictive_rule_direction_conflicts_in_country": (
            "Direction conflicts with country-specific predictive evidence."
        ),
        "causal_signal_present_but_predictive_transfer_was_unstable": (
            "Causal estimate exists, but predictive transfer was unstable."
        ),
        "unstable_or_small_effect_and_causal_diagnostics_weak": (
            "Effect appears unstable/small and diagnostics are weak."
        ),
        "unsupported_status": "Status not supported for recommendation.",
    }
    return mapping.get(reason, reason.replace("_", " "))


def values_different(left: Any, right: Any) -> bool:
    if pd.isna(left) and pd.isna(right):
        return False
    try:
        left_num = float(left)
        right_num = float(right)
        return abs(left_num - right_num) > 1e-9
    except (TypeError, ValueError):
        return str(left) != str(right)


# -----------------------------------------------------------------------------
# App rendering
# -----------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(page_title="CAROB What-If Yield Advisor", page_icon=":seedling:", layout="wide")

    st.title("CAROB What-If Yield Advisor")
    st.caption(
        "Interactive decision support for exploring how management changes can shift expected yield, "
        "with transparent evidence and caveats."
    )

    with st.sidebar:
        st.header("View Options")
        analysis_mode = st.radio(
            "Analysis mode",
            options=["Operational", "Scientific"],
            index=0,
            help="Operational: prediction + guidance only. Scientific: includes assumption-based causal evidence.",
        )

    model_compare_dir = MANIFEST_DEFAULTS["model_compare_dir"]
    interpretability_tag = MANIFEST_DEFAULTS["interpretability_tag"]
    causal_tag = MANIFEST_DEFAULTS["causal_tag"]

    try:
        bundle = load_serving_bundle(str(PROJECT_ROOT), DEFAULT_SCENARIO, model_compare_dir)
    except Exception as exc:
        st.error(f"Could not initialize app: {exc}")
        st.info(
            "Run CAROB pipelines first:\n"
            "1) `carob_feature_prepare.py`\n"
            "2) `carob_model_compare.py`\n"
            "Then reopen this app."
        )
        return

    interpretability = load_interpretability_assets(str(PROJECT_ROOT), interpretability_tag)
    causal_assets = load_causal_assets(str(PROJECT_ROOT), causal_tag)
    dd = load_feature_dictionary(str(DICT_PATH))

    pipeline: Pipeline = bundle["pipeline"]
    modeling: pd.DataFrame = bundle["modeling"]
    features: list[str] = bundle["features"]
    modifiable_features: list[str] = bundle["modifiable"]
    context_features: list[str] = bundle["context"]
    model_summary_df: pd.DataFrame = bundle["summary_df"]

    # Type map is global so control type never flips by country.
    feature_kind_map: dict[str, str] = {f: _feature_kind(modeling[f]) for f in features if f in modeling.columns}

    profiles = _build_display_profiles(modeling)
    if "country" not in profiles.columns:
        st.error("`country` column is missing in modeling frame; context matching cannot run.")
        return

    # ------------------------------------------------------------------
    # Sidebar context setup
    # ------------------------------------------------------------------
    with st.sidebar:
        st.header("1) Set Field Context")
        countries = sorted(profiles["country"].dropna().astype(str).unique().tolist())
        if not countries:
            st.error("No countries found in modeling data.")
            return

        selected_country = st.selectbox("Country", options=countries)
        country_slice = profiles[profiles["country"].astype(str) == selected_country].copy()
        if country_slice.empty:
            st.error("No governed rows found for this country.")
            return

        user_context: dict[str, Any] = {"country": selected_country}
        st.markdown("Set context signals (used to find the closest governed profile)")

        match_context_features = [f for f in context_features if f != "country" and f in profiles.columns]

        # Order context inputs for better UX: site first, then others, then coordinates.
        ordered_context_features: list[str] = []
        if "location" in match_context_features:
            ordered_context_features.append("location")
        ordered_context_features.extend(
            [f for f in match_context_features if f not in {"location", "latitude", "longitude"}]
        )
        for coord in ["latitude", "longitude"]:
            if coord in match_context_features:
                ordered_context_features.append(coord)

        selected_location = None
        for feature in ordered_context_features:
            label = feature_display_name(feature, dd)
            help_text = feature_help_text(feature, dd)

            if feature == "location":
                value = render_feature_input(
                    feature=feature,
                    kind=feature_kind_map.get(feature, _feature_kind(profiles[feature])),
                    label=label,
                    help_text=help_text,
                    current_value=None,
                    reference_all=profiles[feature],
                    reference_country=country_slice[feature],
                    key=f"context_{selected_country}_{feature}",
                    context_mode=True,
                )
                user_context[feature] = value
                selected_location = value
                continue

            if feature in {"latitude", "longitude"} and selected_location is not None:
                auto_coord = _auto_coord_from_location(
                    country_slice,
                    location_value=selected_location,
                    coord_col=feature,
                )
                if auto_coord is not None:
                    user_context[feature] = auto_coord
                    st.selectbox(
                        label,
                        options=[auto_coord],
                        index=0,
                        format_func=lambda x: f"{float(x):.5f}",
                        key=f"context_auto_{selected_country}_{feature}",
                        disabled=True,
                        help=f"{help_text} | Auto-set from selected site name.",
                    )
                    st.caption("Auto-set from selected site name.")
                    continue

            user_context[feature] = render_feature_input(
                feature=feature,
                kind=feature_kind_map.get(feature, _feature_kind(profiles[feature])),
                label=label,
                help_text=help_text,
                current_value=None,
                reference_all=profiles[feature],
                reference_country=country_slice[feature],
                key=f"context_{selected_country}_{feature}",
                context_mode=True,
            )

        baseline_row, within_tolerance, numeric_err_pct, cat_mismatch_count = match_profile_from_context(
            candidates=country_slice,
            user_context=user_context,
            context_features=context_features,
            tolerance=0.10,
        )

        if within_tolerance:
            st.success("Matched profile found within 10% numeric tolerance.")
        else:
            st.warning("No exact 10% match found; using nearest governed profile.")

        if numeric_err_pct:
            st.caption(f"Max numeric context error: {max(numeric_err_pct.values()):.2f}%")
        if cat_mismatch_count > 0:
            st.caption(f"Categorical mismatches vs nearest profile: {cat_mismatch_count}")

        st.caption(f"Matched profile: {baseline_row['_profile_label']}")

    baseline_input = {feature: baseline_row[feature] for feature in features}
    scenario_input = dict(baseline_input)

    # ------------------------------------------------------------------
    # Main: controls and predictions
    # ------------------------------------------------------------------
    left_col, right_col = st.columns([1.2, 1.0], gap="large")

    with left_col:
        st.subheader("2) Adjust Management Choices")
        st.write("Change only actionable levers to test practical what-if scenarios.")

        for feature in modifiable_features:
            scenario_input[feature] = render_feature_input(
                feature=feature,
                kind=feature_kind_map.get(feature, _feature_kind(modeling[feature])),
                label=feature_display_name(feature, dd),
                help_text=feature_help_text(feature, dd),
                current_value=baseline_row[feature],
                reference_all=modeling[feature],
                reference_country=country_slice[feature] if feature in country_slice.columns else modeling[feature],
                key=f"modifiable_{int(baseline_row['_profile_id'])}_{feature}",
                context_mode=False,
            )

        context_table = pd.DataFrame(
            [
                {
                    "Context signal": feature_display_name(f, dd),
                    "Selected value": user_context.get(f),
                    "Matched profile value": baseline_row.get(f),
                }
                for f in match_context_features
            ]
        )
        with st.expander("Context matching details", expanded=False):
            st.dataframe(context_table, use_container_width=True, hide_index=True)

    baseline_df = pd.DataFrame([baseline_input], columns=features)
    scenario_df = pd.DataFrame([scenario_input], columns=features)

    baseline_pred = float(pipeline.predict(baseline_df)[0])
    scenario_pred = float(pipeline.predict(scenario_df)[0])
    delta = scenario_pred - baseline_pred
    pct = (delta / baseline_pred * 100.0) if baseline_pred != 0 else 0.0

    changed_rows: list[dict[str, Any]] = []
    for feature in modifiable_features:
        old = baseline_input[feature]
        new = scenario_input[feature]
        if values_different(old, new):
            changed_rows.append(
                {
                    "Management lever": feature_display_name(feature, dd),
                    "Before": old,
                    "After": new,
                    "variable": feature,
                }
            )

    with right_col:
        st.subheader("3) Predicted Yield Impact")
        m1, m2, m3 = st.columns(3)
        m1.metric("Expected yield (baseline)", f"{baseline_pred:,.2f}")
        m2.metric("Expected yield (your scenario)", f"{scenario_pred:,.2f}", delta=f"{delta:+,.2f}")
        m3.metric("Relative change", f"{pct:+.2f}%")

        if changed_rows:
            changed_df = pd.DataFrame(changed_rows).drop(columns=["variable"])
            st.markdown("**Changes you made**")
            st.dataframe(changed_df, use_container_width=True, hide_index=True)
        else:
            st.info("No practice changes yet. Adjust controls to run a what-if scenario.")

        if changed_rows:
            impact_rows = []
            for row in changed_rows:
                feat = row["variable"]
                tmp = dict(baseline_input)
                tmp[feat] = scenario_input[feat]
                tmp_pred = float(pipeline.predict(pd.DataFrame([tmp], columns=features))[0])
                impact_rows.append(
                    {
                        "Management lever": feature_display_name(feat, dd),
                        "Isolated effect on prediction": float(tmp_pred - baseline_pred),
                    }
                )
            impact_df = pd.DataFrame(impact_rows).sort_values("Isolated effect on prediction", ascending=False)
            st.markdown("**Estimated one-change-at-a-time contribution**")
            st.dataframe(impact_df.round(3), use_container_width=True, hide_index=True)

    # Model comparison snapshot
    st.divider()
    st.subheader("Current Model Benchmark Snapshot")
    if model_summary_df.empty:
        st.info("Model comparison summary not available.")
    else:
        rows = model_summary_df.copy()
        if "scenario" in rows.columns:
            scoped = rows[rows["scenario"].astype(str) == DEFAULT_SCENARIO].copy()
            if not scoped.empty:
                rows = scoped
        rows = rows.sort_values(["r2", "rmse"], ascending=[False, True]).reset_index(drop=True)
        summary_cols = ["model", "r2", "rmse", "mae", "n_train", "n_test", "n_trials_in_test"]
        show_cols = [c for c in summary_cols if c in rows.columns]
        show_df = rows[show_cols].head(6).copy()
        rename_map = {
            "model": "Model",
            "r2": "R-squared",
            "rmse": "RMSE",
            "mae": "MAE",
            "n_train": "Train rows",
            "n_test": "Test rows",
            "n_trials_in_test": "Trials in test",
        }
        show_df = show_df.rename(columns=rename_map)
        st.dataframe(show_df.round(4), use_container_width=True, hide_index=True)

    # ------------------------------------------------------------------
    # Explainability
    # ------------------------------------------------------------------
    st.divider()
    st.subheader("4) Why the Model Predicts This")
    st.caption("Global importance comes from CatBoost SHAP outputs from the interpretability pipeline.")

    exp_left, exp_right = st.columns([1.1, 1.0], gap="large")

    with exp_left:
        if interpretability.shap_img_path is not None and interpretability.shap_img_path.exists():
            st.image(str(interpretability.shap_img_path), caption="Global driver importance")
        else:
            st.info("Global SHAP chart is unavailable for the selected run tag.")

        if interpretability.top_note:
            with st.expander("Interpretability note", expanded=False):
                st.markdown(interpretability.top_note)

    with exp_right:
        shap_df = interpretability.shap_df.copy()
        if shap_df.empty:
            st.info("No SHAP table available for the selected interpretability run.")
        else:
            shap_show = shap_df.head(10).copy()
            shap_show["feature_label"] = shap_show["feature"].map(lambda f: feature_display_name(str(f), dd))
            cols = ["feature_label", "mean_abs_shap"] + (["final_role"] if "final_role" in shap_show.columns else [])
            shap_show = shap_show[cols].rename(
                columns={
                    "feature_label": "Driver",
                    "mean_abs_shap": "Mean |SHAP|",
                    "final_role": "Role",
                }
            )
            st.dataframe(shap_show.round(3), use_container_width=True, hide_index=True)

            if changed_rows:
                changed_features = [r["variable"] for r in changed_rows]
                changed_shap = shap_df[shap_df["feature"].astype(str).isin(changed_features)].copy()
                st.markdown("**Your changed levers in SHAP ranking**")
                if changed_shap.empty:
                    st.caption("None of the changed levers appear in the available SHAP table.")
                else:
                    changed_shap["Driver"] = changed_shap["feature"].map(lambda f: feature_display_name(str(f), dd))
                    out = changed_shap[["Driver", "mean_abs_shap"]].rename(columns={"mean_abs_shap": "Mean |SHAP|"})
                    st.dataframe(out.round(3), use_container_width=True, hide_index=True)

    # ------------------------------------------------------------------
    # Decision guidance (rules)
    # ------------------------------------------------------------------
    st.divider()
    st.subheader("5) Decision Guidance and Safeguards")

    rules_df = interpretability.rules_df.copy()
    using_surrogate = False
    if rules_df.empty:
        using_surrogate = True
        rules_df = bundle["surrogate_rules"]["rules"].copy()

    rules_df = sort_rules_for_display(rules_df)

    if using_surrogate:
        st.warning(
            "Final interpretability rules were not found for this run tag. "
            "Showing fallback surrogate rules (use with caution)."
        )

    baseline_rule = find_matching_rule(rules_df, pd.Series(baseline_input)) if not rules_df.empty else None
    scenario_rule = find_matching_rule(rules_df, pd.Series(scenario_input)) if not rules_df.empty else None

    if rules_df.empty:
        st.info("No decision-rule table is available.")
    else:
        show = rules_df.head(10).copy()
        show["Rule"] = show["rule_conditions"].map(lambda t: format_rule_text(str(t), dd))

        # Keep table compact so it fits horizontally without scrolling.
        table_cols = ["rule_id", "Rule"]
        if "confidence_level" in show.columns:
            table_cols.append("confidence_level")

        table_df = show[table_cols].copy().rename(
            columns={
                "rule_id": "Rule ID",
                "confidence_level": "Confidence",
            }
        )
        st.table(table_df)

    st.markdown("**Matched rule segment**")
    if baseline_rule is None:
        st.info("No baseline rule match found.")
    else:
        st.write(f"Baseline: `{baseline_rule.get('rule_id', 'n/a')}`")
        st.caption(format_rule_text(str(baseline_rule.get("rule_conditions", "")), dd))

    if scenario_rule is None:
        st.info("No what-if rule match found.")
    else:
        st.write(f"What-if: `{scenario_rule.get('rule_id', 'n/a')}`")
        st.caption(format_rule_text(str(scenario_rule.get("rule_conditions", "")), dd))
        conf = str(scenario_rule.get("confidence_level", "n/a"))
        cav = str(scenario_rule.get("caveat_labels", "n/a"))
        st.write(f"Confidence: **{conf}**")
        st.write(f"Caveats: `{cav}`")

    if baseline_rule is not None and scenario_rule is not None:
        if str(baseline_rule.get("rule_id")) == str(scenario_rule.get("rule_id")):
            st.info("Scenario remains in the same guidance segment.")
        else:
            st.success("Scenario moved to a different guidance segment.")

    if scenario_rule is not None and not interpretability.rule_country_df.empty:
        country_df = interpretability.rule_country_df.copy()
        mask = (
            country_df["rule_id"].astype(str) == str(scenario_rule.get("rule_id"))
        ) & (country_df["country"].astype(str) == selected_country)
        row = country_df.loc[mask].head(1)
        if not row.empty:
            status = str(row.iloc[0].get("generalization_status", "unknown"))
            st.write(f"Country transfer status: **{friendly_country_status(status)}**")

    # ------------------------------------------------------------------
    # Scientific mode (causal)
    # ------------------------------------------------------------------
    if analysis_mode == "Scientific":
        st.divider()
        st.subheader("6) Scientific Evidence (Assumption-Based)")
        st.caption(
            "Shown only in Scientific mode. These estimates are observational and depend on overlap, "
            "balance, and unconfoundedness assumptions."
        )

        if causal_assets.source == "rule_aipw" and not causal_assets.scorecard_df.empty:
            score = causal_assets.scorecard_df.copy()

            target_row = pd.DataFrame()
            if scenario_rule is not None:
                rid = str(scenario_rule.get("rule_id"))
                target_row = score[
                    (score["rule_id"].astype(str) == rid) & (score["country"].astype(str) == selected_country)
                ].head(1)

            if target_row.empty:
                target_row = score[score["country"].astype(str) == selected_country].head(1)

            if target_row.empty:
                st.info("No causal pair estimate available for this country under the selected run tag.")
            else:
                row = target_row.iloc[0]
                rec = str(row.get("recommendation", "n/a"))
                reason = str(row.get("decision_reason", "n/a"))
                ate = float(row.get("ate_ref_seed", float("nan")))
                ci_low = float(row.get("ci_low_ref_seed", float("nan")))
                ci_high = float(row.get("ci_high_ref_seed", float("nan")))

                m1, m2, m3 = st.columns(3)
                m1.metric("Recommendation", rec)
                m2.metric("Estimated effect (kg/ha)", f"{ate:+.1f}")
                m3.metric("95% CI", f"[{ci_low:.1f}, {ci_high:.1f}]")

                st.write(f"Decision rationale: {friendly_decision_reason(reason)}")

                diag_cols = [
                    "sign_match_rate",
                    "overlap_fraction_common_support_ref_seed",
                    "max_abs_smd_after_ref_seed",
                    "n_seed_runs",
                    "n_total",
                    "n_treated",
                    "n_control",
                ]
                diag_view = pd.DataFrame([{c: row.get(c, None) for c in diag_cols}]).rename(
                    columns={
                        "sign_match_rate": "Sign stability",
                        "overlap_fraction_common_support_ref_seed": "Overlap fraction",
                        "max_abs_smd_after_ref_seed": "Max |SMD| after weighting",
                        "n_seed_runs": "Seed runs",
                        "n_total": "Rows",
                        "n_treated": "Treated rows",
                        "n_control": "Control rows",
                    }
                )
                st.dataframe(diag_view.round(4), use_container_width=True, hide_index=True)

        elif causal_assets.source == "pilot" and not causal_assets.overall_df.empty:
            row = causal_assets.overall_df.iloc[0]
            st.write("Using fallback causal pilot summary (not rule-level AIPW).")
            metrics = st.columns(3)
            metrics[0].metric("Confidence", str(row.get("confidence_level", "n/a")))
            metrics[1].metric("Fixed effect (kg/ha)", f"{float(row.get('fixed_effect', float('nan'))):+.2f}")
            metrics[2].metric("Random effect (kg/ha)", f"{float(row.get('random_effect', float('nan'))):+.2f}")
            st.caption(f"Caveats: {row.get('caveats', 'n/a')}")

        else:
            st.info(
                "No causal artifacts found. Run `carob_rule_causal_aipw.py` for rule-level Scientific mode evidence."
            )
    else:
        st.info("Scientific evidence is hidden in Operational mode.")

    # ------------------------------------------------------------------
    # Download summary
    # ------------------------------------------------------------------
    st.divider()
    st.subheader("Export Decision Summary")

    scenario_rule_id = str(scenario_rule.get("rule_id", "n/a")) if scenario_rule is not None else "n/a"
    changed_text = "\n".join(
        [f"- {row['Management lever']}: {row['Before']} -> {row['After']}" for row in changed_rows]
    )
    if not changed_text:
        changed_text = "- No lever changes made"

    summary_lines = [
        "# CAROB What-If Decision Summary",
        "",
        f"- Analysis mode: {analysis_mode}",
        f"- Country: {selected_country}",
        f"- Matched profile: {baseline_row['_profile_label']}",
        f"- Baseline expected yield: {baseline_pred:.2f}",
        f"- Scenario expected yield: {scenario_pred:.2f}",
        f"- Delta: {delta:+.2f} ({pct:+.2f}%)",
        f"- Matched scenario rule: {scenario_rule_id}",
        "",
        "## Lever Changes",
        changed_text,
    ]

    if analysis_mode == "Scientific" and causal_assets.source == "rule_aipw" and not causal_assets.scorecard_df.empty:
        score = causal_assets.scorecard_df.copy()
        causal_match = pd.DataFrame()
        if scenario_rule is not None:
            causal_match = score[
                (score["rule_id"].astype(str) == str(scenario_rule.get("rule_id")))
                & (score["country"].astype(str) == selected_country)
            ].head(1)
        if not causal_match.empty:
            r = causal_match.iloc[0]
            summary_lines += [
                "",
                "## Scientific Evidence",
                f"- Recommendation: {r.get('recommendation', 'n/a')}",
                f"- ATE (ref seed): {float(r.get('ate_ref_seed', float('nan'))):+.2f}",
                (
                    "- 95% CI: "
                    f"[{float(r.get('ci_low_ref_seed', float('nan'))):.2f}, "
                    f"{float(r.get('ci_high_ref_seed', float('nan'))):.2f}]"
                ),
                f"- Decision reason: {friendly_decision_reason(str(r.get('decision_reason', 'n/a')))}",
            ]

    summary_text = "\n".join(summary_lines)
    st.download_button(
        label="Download summary (.md)",
        data=summary_text,
        file_name="carob_what_if_summary.md",
        mime="text/markdown",
        use_container_width=False,
    )

    # ------------------------------------------------------------------
    # Diagnostics and provenance
    # ------------------------------------------------------------------
    diagnostics = []
    diagnostics.extend(bundle.get("diagnostics", []))
    diagnostics.extend(interpretability.diagnostics)
    diagnostics.extend(causal_assets.diagnostics)

    provenance_rows = [
        {
            "Artifact": "Model comparison summary",
            "Path": str(bundle.get("summary_path")) if bundle.get("summary_path") else "missing",
            "Last updated": _fmt_timestamp(bundle.get("summary_path")),
        },
        {
            "Artifact": "Interpretability run dir",
            "Path": str(interpretability.run_dir) if interpretability.run_dir else "missing",
            "Last updated": _fmt_timestamp(interpretability.run_dir),
        },
        {
            "Artifact": "Interpretability SHAP table",
            "Path": str(interpretability.shap_path) if interpretability.shap_path else "missing",
            "Last updated": _fmt_timestamp(interpretability.shap_path),
        },
        {
            "Artifact": "Interpretability rules",
            "Path": str(interpretability.rules_path) if interpretability.rules_path else "missing",
            "Last updated": _fmt_timestamp(interpretability.rules_path),
        },
        {
            "Artifact": "Causal source",
            "Path": str(causal_assets.run_dir) if causal_assets.run_dir else "missing",
            "Last updated": _fmt_timestamp(causal_assets.run_dir),
        },
        {
            "Artifact": "Causal scorecard",
            "Path": str(causal_assets.scorecard_path) if causal_assets.scorecard_path else "missing",
            "Last updated": _fmt_timestamp(causal_assets.scorecard_path),
        },
    ]

    with st.sidebar:
        with st.expander("Run Provenance", expanded=False):
            st.dataframe(pd.DataFrame(provenance_rows), use_container_width=True, hide_index=True)

        if diagnostics:
            with st.expander("Diagnostics", expanded=False):
                for msg in diagnostics:
                    st.write(f"- {msg}")

        with st.expander("Advanced variable names", expanded=False):
            adv_rows = [
                {
                    "Display label": feature_display_name(f, dd),
                    "Raw variable": f,
                    "Role": str(dd.loc[dd["column_name"].astype(str) == f, "final_role"].iloc[0])
                    if not dd.empty and f in set(dd["column_name"].astype(str))
                    else "unknown",
                }
                for f in features
            ]
            st.dataframe(pd.DataFrame(adv_rows), use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
