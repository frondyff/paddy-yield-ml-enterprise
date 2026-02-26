"""Shared path helpers for pipeline entrypoints."""

from __future__ import annotations

from pathlib import Path

DEFAULT_DATA_CANDIDATES = (
    Path("data/input/paddydataset.csv"),
    Path("data/raw/paddydataset.csv"),
)


def project_root() -> Path:
    try:
        return Path(__file__).resolve().parents[3]
    except Exception:
        return Path.cwd()


def resolve_data_path(explicit_path: Path | None) -> Path:
    if explicit_path is not None:
        return explicit_path

    root = project_root()
    for rel_path in DEFAULT_DATA_CANDIDATES:
        candidate = root / rel_path
        if candidate.exists():
            return candidate

    return root / DEFAULT_DATA_CANDIDATES[0]
