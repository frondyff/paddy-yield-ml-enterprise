"""Run CAROB top-2 contender tuning pipeline (ExtraTrees + CatBoost)."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def main() -> None:
    from paddy_yield_ml.pipelines.carob_model_tune_top2 import main as run

    run()


if __name__ == "__main__":
    main()
