"""Path constants. Honour RAFT_DATA_ROOT env var if set."""
from __future__ import annotations
import os
from pathlib import Path

DATA_ROOT = Path(os.environ.get("RAFT_DATA_ROOT", Path(__file__).resolve().parents[2] / "data"))
RAW_ROOT = DATA_ROOT / "raw"
DERIVED_ROOT = DATA_ROOT / "derived"
DERIVED_ROOT.mkdir(parents=True, exist_ok=True)


def raw_source(source: str) -> Path:
    return RAW_ROOT / f"source={source}"


def derived(name: str) -> Path:
    return DERIVED_ROOT / name
