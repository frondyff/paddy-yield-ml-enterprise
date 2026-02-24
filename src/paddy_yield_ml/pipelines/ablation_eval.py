"""
Ablation matrix for strict vs operational evaluation of context-heavy features.

Run:
  python src/paddy_yield_ml/pipelines/ablation_eval.py --run-tag latest

Outputs (under ./outputs/ablation_eval/<run-tag>/):
  - scenario_feature_sets.csv
  - ablation_logo_fold_metrics.csv
  - ablation_logo_summary.csv
  - ablation_secondary_fold_metrics.csv
  - ablation_secondary_summary.csv
  - ablation_combined_summary.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from paddy_yield_ml.pipelines import model_compare as mc
from paddy_yield_ml.pipelines import model_select_tune as mst

try:
    project_root = Path(__file__).resolve().parents[3]
except NameError:
    project_root = Path.cwd()

OUT_ROOT = project_root / "outputs" / "ablation_eval"
DEFAULT_DICT_PATH = project_root / "data" / "metadata" / "data_dictionary_paddy.csv"
DEFAULT_SCENARIO_PATH = project_root / "outputs" / "model_select_tune" / "dual_eval" / "scenario_feature_sets.csv"
DEFAULT_PARAMS_PATH = project_root / "outputs" / "model_select_tune" / "dual_eval" / "logo_summary.csv"
BLOCKED_FEATURES = {mc.RAW_TARGET_COL, mc.TARGET_COL, "Trash(in bundles)"}
LOCATION_FEATURE_COL = "Agriblock_feature"


def load_dictionary(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing feature dictionary: {path}")
    out = pd.read_csv(path)
    required = {"column_name", "feature_group"}
    missing = required - set(out.columns)
    if missing:
        raise ValueError(f"Dictionary is missing required columns: {sorted(missing)}")
    return out


def load_base_features(path: Path, scenario_name: str) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Missing scenario file: {path}")
    out = pd.read_csv(path)
    required = {"scenario", "feature"}
    missing = required - set(out.columns)
    if missing:
        raise ValueError(f"Scenario file missing columns: {sorted(missing)}")
    selected = out.loc[out["scenario"].astype(str) == scenario_name, "feature"].astype(str).dropna().tolist()
    if not selected:
        available = sorted(out["scenario"].astype(str).unique().tolist())
        raise ValueError(f"Scenario '{scenario_name}' not found. Available: {available}")
    return mc.dedupe_keep_order(selected)


def extract_weather_water_features(dictionary_df: pd.DataFrame) -> list[str]:
    group_raw = dictionary_df["feature_group"].astype(str).str.lower()
    mask = group_raw.str.contains("weather") | group_raw.str.contains("water")
    features = dictionary_df.loc[mask, "column_name"].astype(str).tolist()
    return mc.dedupe_keep_order(features)


def keep_usable_non_leaky_features(frame: pd.DataFrame, features: list[str]) -> list[str]:
    return [f for f in mc.dedupe_keep_order(features) if f in frame.columns and f not in BLOCKED_FEATURES]


def build_ablation_scenarios(
    frame: pd.DataFrame,
    base_features: list[str],
    weather_water_features: list[str],
    include_location_scenario: bool,
) -> dict[str, list[str]]:
    conservative = keep_usable_non_leaky_features(frame, base_features)
    if not conservative:
        raise ValueError("Conservative scenario is empty after filtering.")

    plus_weather = keep_usable_non_leaky_features(frame, conservative + weather_water_features)
    scenarios: dict[str, list[str]] = {
        "conservative_base": conservative,
        "plus_weather_water": plus_weather,
    }

    if include_location_scenario:
        if LOCATION_FEATURE_COL not in frame.columns:
            raise ValueError(f"Expected location feature column not found: {LOCATION_FEATURE_COL}")
        plus_location = keep_usable_non_leaky_features(frame, plus_weather + [LOCATION_FEATURE_COL])
        scenarios["plus_weather_water_and_location"] = plus_location

    return scenarios


def load_model_params(path: Path, scenario_name: str, model_key: str) -> dict[str, Any]:
    if not path.exists():
        return {}
    out = pd.read_csv(path)
    required = {"feature_set", "model", "params_json"}
    if not required.issubset(out.columns):
        return {}
    exact = out[(out["feature_set"].astype(str) == scenario_name) & (out["model"].astype(str) == model_key)]
    if exact.empty:
        exact = out[out["model"].astype(str) == model_key]
    if exact.empty:
        return {}
    row = exact.sort_values("r2_mean", ascending=False).iloc[0]
    try:
        parsed = json.loads(str(row.get("params_json", "{}")))
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def run(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    run_dir = OUT_ROOT / args.run_tag
    run_dir.mkdir(parents=True, exist_ok=True)

    frame = mc.load_analysis_frame()
    frame[LOCATION_FEATURE_COL] = frame[mc.GROUP_COL].astype(str)

    dictionary_df = load_dictionary(Path(args.dictionary_path))
    base_features = load_base_features(Path(args.base_scenario_path), args.base_scenario_name)
    weather_features = extract_weather_water_features(dictionary_df)

    scenarios = build_ablation_scenarios(
        frame=frame,
        base_features=base_features,
        weather_water_features=weather_features,
        include_location_scenario=args.include_location_scenario,
    )
    scenario_rows: list[dict[str, Any]] = []
    for scenario_name, feature_list in scenarios.items():
        for rank, feature in enumerate(feature_list, start=1):
            scenario_rows.append(
                {
                    "scenario": scenario_name,
                    "rank_in_scenario": rank,
                    "feature": feature,
                }
            )
    pd.DataFrame(scenario_rows).to_csv(run_dir / "scenario_feature_sets.csv", index=False)

    model_params = load_model_params(Path(args.params_source_path), args.base_scenario_name, args.model)
    if model_params:
        print(f"Using model params loaded from {args.params_source_path}")
    else:
        print("No source params found; using model defaults.")

    logo_fold_tables: list[pd.DataFrame] = []
    logo_summary_rows: list[dict[str, Any]] = []
    secondary_fold_tables: list[pd.DataFrame] = []
    secondary_summary_tables: list[pd.DataFrame] = []

    print("\nAblation evaluation")
    for scenario_name, feature_list in scenarios.items():
        logo_folds_df, logo_summary = mst.evaluate_logo(
            frame=frame,
            feature_set_name=scenario_name,
            features=feature_list,
            model_key=args.model,
            params=model_params,
            random_state=args.random_state,
        )
        logo_folds_df["scenario_family"] = "ablation"
        logo_fold_tables.append(logo_folds_df)
        logo_summary_rows.append(logo_summary)

        secondary_folds_df, secondary_summary_df = mst.evaluate_secondary_splits(
            frame=frame,
            feature_set_name=scenario_name,
            features=feature_list,
            model_key=args.model,
            params=model_params,
            random_state=args.random_state,
            group_shuffle_splits=args.group_shuffle_splits,
            random_shuffle_splits=args.random_shuffle_splits,
            test_size=args.secondary_test_size,
        )
        secondary_folds_df["scenario_family"] = "ablation"
        secondary_fold_tables.append(secondary_folds_df)
        secondary_summary_tables.append(secondary_summary_df)

        print(
            f"  {scenario_name}: n_features={len(feature_list)} "
            f"| LOGO r2={logo_summary['r2_mean']:.4f} rmse={logo_summary['rmse_mean']:.2f}"
        )

    logo_folds_out = pd.concat(logo_fold_tables, ignore_index=True)
    logo_summary_out = pd.DataFrame(logo_summary_rows).sort_values(["r2_mean", "rmse_mean"], ascending=[False, True])
    secondary_folds_out = pd.concat(secondary_fold_tables, ignore_index=True)
    secondary_summary_out = pd.concat(secondary_summary_tables, ignore_index=True).sort_values(
        ["evaluation_scheme", "r2_mean", "rmse_mean"], ascending=[True, False, True]
    )

    logo_folds_out.to_csv(run_dir / "ablation_logo_fold_metrics.csv", index=False)
    logo_summary_out.to_csv(run_dir / "ablation_logo_summary.csv", index=False)
    secondary_folds_out.to_csv(run_dir / "ablation_secondary_fold_metrics.csv", index=False)
    secondary_summary_out.to_csv(run_dir / "ablation_secondary_summary.csv", index=False)

    logo_view = logo_summary_out.copy()
    logo_view["evaluation_scheme"] = "logo"
    combined = pd.concat(
        [
            logo_view[
                [
                    "evaluation_scheme",
                    "feature_set",
                    "model",
                    "n_features",
                    "n_folds",
                    "mae_mean",
                    "mae_std",
                    "rmse_mean",
                    "rmse_std",
                    "r2_mean",
                    "r2_std",
                    "fit_predict_seconds_mean",
                ]
            ],
            secondary_summary_out[
                [
                    "evaluation_scheme",
                    "feature_set",
                    "model",
                    "n_features",
                    "n_folds",
                    "mae_mean",
                    "mae_std",
                    "rmse_mean",
                    "rmse_std",
                    "r2_mean",
                    "r2_std",
                    "fit_predict_seconds_mean",
                ]
            ],
        ],
        ignore_index=True,
    ).sort_values(["evaluation_scheme", "r2_mean", "rmse_mean"], ascending=[True, False, True])

    combined.to_csv(run_dir / "ablation_combined_summary.csv", index=False)
    print(f"Saved artifacts to: {run_dir}")
    return logo_summary_out, secondary_summary_out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ablation matrix for conservative vs context-heavy feature sets.")
    parser.add_argument("--run-tag", type=str, default="latest")
    parser.add_argument("--dictionary-path", type=str, default=str(DEFAULT_DICT_PATH))
    parser.add_argument("--base-scenario-path", type=str, default=str(DEFAULT_SCENARIO_PATH))
    parser.add_argument("--base-scenario-name", type=str, default="full_review")
    parser.add_argument("--params-source-path", type=str, default=str(DEFAULT_PARAMS_PATH))
    parser.add_argument("--model", type=str, default="catboost")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--group-shuffle-splits", type=int, default=8)
    parser.add_argument("--random-shuffle-splits", type=int, default=8)
    parser.add_argument("--secondary-test-size", type=float, default=0.2)
    parser.add_argument("--include-location-scenario", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
