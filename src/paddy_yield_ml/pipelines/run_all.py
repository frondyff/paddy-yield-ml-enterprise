"""Run all project pipeline stages and write aligned output artifacts."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

from paddy_yield_ml.pipelines.baseline_train import (
    TARGET_COL,
    add_per_hectare_features,
    load_dataset,
    save_artifacts,
    train_and_evaluate,
)
from paddy_yield_ml.pipelines.interpretability import run_interpretability
from paddy_yield_ml.pipelines.paths import project_root, resolve_data_path


def parse_args() -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(description="Run all paddy yield pipeline stages")
    parser.add_argument("--data-path", type=Path, default=None)
    parser.add_argument("--outputs-root", type=Path, default=root / "outputs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    return parser.parse_args()


def _run_baseline(data_path: Path, out_dir: Path, seed: int, test_size: float) -> None:
    df = load_dataset(data_path)
    featured = add_per_hectare_features(df)
    result = train_and_evaluate(
        featured,
        seed=seed,
        test_size=test_size,
        group_col="Agriblock",
        run_group_eval=True,
    )
    save_artifacts(
        out_dir=out_dir,
        train_result=result,
        config={
            "data_path": str(data_path),
            "out_dir": str(out_dir),
            "seed": seed,
            "test_size": test_size,
            "group_col": "Agriblock",
            "run_eda": False,
            "target_col": TARGET_COL,
        },
        source_df=featured,
        run_eda=False,
    )


def _run_legacy_main(module_name: str, argv: list[str]) -> None:
    module = importlib.import_module(module_name)
    main = getattr(module, "main", None)
    if not callable(main):
        raise AttributeError(f"Module {module_name} has no callable main().")

    old_argv = sys.argv[:]
    try:
        sys.argv = [module_name.rsplit(".", maxsplit=1)[-1], *argv]
        main()
    finally:
        sys.argv = old_argv


def main() -> None:
    args = parse_args()
    data_path = resolve_data_path(args.data_path)
    outputs_root = args.outputs_root

    targets = {
        "baseline": outputs_root / "baseline",
        "feature_prepare": outputs_root / "feature_prepare",
        "model_compare": outputs_root / "model_compare",
        "model_select_tune": outputs_root / "model_select_tune",
        "ablation_eval": outputs_root / "ablation_eval",
        "interpretability": outputs_root / "interpretability",
    }

    _run_baseline(data_path, targets["baseline"], args.seed, args.test_size)
    # Legacy orchestration fallback: invoke pipeline entrypoints directly.
    _run_legacy_main("paddy_yield_ml.pipelines.feature_prepare", [])
    _run_legacy_main("paddy_yield_ml.pipelines.model_compare", [])
    _run_legacy_main("paddy_yield_ml.pipelines.model_select_tune", [])
    _run_legacy_main("paddy_yield_ml.pipelines.ablation_eval", [])
    run_interpretability(data_path, targets["interpretability"], args.seed, args.test_size)

    with (outputs_root / "run_all_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "data_path": str(data_path),
                "outputs": {name: str(path) for name, path in targets.items()},
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
