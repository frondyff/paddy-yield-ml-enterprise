"""Trial-aware model comparison on CAROB excluding Benin."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from paddy_yield_ml.pipelines import carob_model_compare as base

try:
    project_root = Path(__file__).resolve().parents[3]
except NameError:
    project_root = Path.cwd()

OUT_DIR = project_root / "outputs" / "carob_model_compare_excl_benin"
CANDIDATES_PATH = project_root / "outputs" / "carob_feature_prepare" / "hybrid_selection_candidates.csv"
DEFAULT_SCENARIO = "modifiable_plus_context"
DEFAULT_MODELS = "random_forest,extra_trees,catboost,xgboost,lightgbm"
DEFAULT_TEST_SIZE = 0.2
DEFAULT_EXCLUDE_COUNTRIES = "Benin"


def parse_excluded_countries(raw: str) -> list[str]:
    countries = [c.strip() for c in str(raw).split(",") if c.strip()]
    return base.dedupe_keep_order(countries)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CAROB model comparison with Benin excluded.")
    parser.add_argument("--scenario", type=str, default=DEFAULT_SCENARIO)
    parser.add_argument("--models", type=str, default=DEFAULT_MODELS)
    parser.add_argument("--test-size", type=float, default=DEFAULT_TEST_SIZE)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--trial-median-impute", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--exclude-countries", type=str, default=DEFAULT_EXCLUDE_COUNTRIES)
    parser.add_argument("--candidates-path", type=str, default=str(CANDIDATES_PATH))
    parser.add_argument("--out-dir", type=str, default=str(OUT_DIR))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    candidates_path = Path(args.candidates_path)

    frame = base.load_frame_with_country_gate(candidates_path)
    excluded_countries = parse_excluded_countries(args.exclude_countries)
    before_rows = len(frame)
    if excluded_countries and "country" in frame.columns:
        frame = frame[~frame["country"].astype(str).isin(excluded_countries)].reset_index(drop=True)
    after_rows = len(frame)

    candidates = base.load_hybrid_candidates(candidates_path)
    feature_sets = base.build_feature_sets(candidates)

    if args.scenario not in feature_sets:
        raise ValueError(f"Scenario '{args.scenario}' not found. Available: {sorted(feature_sets)}")

    features = base.filter_available_features(feature_sets[args.scenario], frame)
    scenario_feature_df = pd.DataFrame(
        [{"scenario": args.scenario, "rank_in_scenario": i, "feature": f} for i, f in enumerate(features, start=1)]
    )
    scenario_feature_df.to_csv(out_dir / "scenario_feature_sets.csv", index=False)

    exclusion_df = pd.DataFrame(
        [
            {
                "excluded_countries": ",".join(excluded_countries),
                "rows_before_exclusion": int(before_rows),
                "rows_after_exclusion": int(after_rows),
                "rows_removed": int(before_rows - after_rows),
            }
        ]
    )
    exclusion_df.to_csv(out_dir / "exclusion_summary.csv", index=False)

    model_specs = base.build_model_specs(base.parse_model_list(args.models), args.random_state)

    trial_all: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []
    for spec in model_specs:
        print(f"\nModel: {spec.name}")
        for i, params in enumerate(spec.param_grid, start=1):
            trial_df, summary = base.evaluate_trial_aware(
                frame=frame,
                features=features,
                model_spec=spec,
                params=params,
                test_size=float(args.test_size),
                random_state=int(args.random_state),
                trial_median_impute=bool(args.trial_median_impute),
            )
            trial_df.insert(0, "model", spec.name)
            trial_df.insert(1, "param_set", i)
            trial_df.insert(2, "scenario", args.scenario)
            summary["param_set"] = i
            summary["scenario"] = args.scenario
            summary["excluded_countries"] = ",".join(excluded_countries)
            trial_all.append(trial_df)
            summary_rows.append(summary)
            print(
                f"  set={i} | R2={summary['r2']:.4f} | RMSE={summary['rmse']:.2f} | MAE={summary['mae']:.2f} "
                f"| params={params}"
            )

    trial_metrics = pd.concat(trial_all, ignore_index=True)
    summary_df = pd.DataFrame(summary_rows).sort_values(["r2", "rmse"], ascending=[False, True]).reset_index(drop=True)

    trial_metrics.to_csv(out_dir / "trial_aware_trial_metrics.csv", index=False)
    summary_df.to_csv(out_dir / "model_comparison_summary.csv", index=False)

    best = summary_df.iloc[0]
    print("\nBest configuration:")
    print(best.to_string())
    print(
        f"\nSaved CAROB model-compare (excluding {','.join(excluded_countries)}) outputs to: {out_dir}"
    )

    best_line = (
        f"best_model={best['model']} | param_set={best['param_set']} | "
        f"R2={best['r2']:.4f} | RMSE={best['rmse']:.2f} | MAE={best['mae']:.2f}"
    )
    notes = [
        "# CAROB Model Compare (Exclude Countries)",
        "",
        f"- Excluded countries: `{','.join(excluded_countries)}`",
        f"- Rows before exclusion: `{before_rows}`",
        f"- Rows after exclusion: `{after_rows}`",
        f"- Scenario: `{args.scenario}`",
        f"- Models: `{args.models}`",
        "",
        "## Best Result",
        f"- {best_line}",
        "",
        "See `model_comparison_summary.csv` for full leaderboard.",
    ]
    (out_dir / "run_summary.md").write_text("\n".join(notes), encoding="utf-8")


if __name__ == "__main__":
    main()
