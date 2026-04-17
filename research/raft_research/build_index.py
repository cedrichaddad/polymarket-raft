"""Build the retrieval index over historical state vectors (§17).

v1 uses `hnswlib` as a local stand-in for the Vibrato index: same interface
(basic top-k, L2 distance, filter-by-tag at retrieval time). The Rust service
can either:
    a) Query Vibrato directly over its native client, or
    b) Load this hnswlib index via the Python bridge for offline experiments.

Outputs:
    data/derived/retrieval.bin      — hnswlib index
    data/derived/retrieval_meta.parquet — tag/label columns aligned to index ids

Usage:
    python -m raft_research.build_index --m 16 --ef-construction 200
"""
from __future__ import annotations
import argparse
import logging

import hnswlib
import numpy as np
import pandas as pd

from .features import FEATURE_COLS, add_features
from .paths import derived

log = logging.getLogger(__name__)


def build(m: int = 16, ef_construction: int = 200, ef_query: int = 128) -> int:
    path = derived("market_state_1s_labeled.parquet")
    if not path.exists():
        raise SystemExit(f"missing {path} — run build_labels first")
    df = pd.read_parquet(path)
    df = df[df["resolved_up"].notna()].copy()
    df = add_features(df)

    X = df[FEATURE_COLS].to_numpy(dtype=np.float32)
    ids = np.arange(len(df), dtype=np.int64)

    index = hnswlib.Index(space="l2", dim=X.shape[1])
    index.init_index(max_elements=len(X), ef_construction=ef_construction, M=m)
    index.add_items(X, ids)
    index.set_ef(ef_query)

    index_path = derived("retrieval.bin")
    index.save_index(str(index_path))

    meta = df[["market_id", "state_ts_ms", "window_type", "tick_regime", "resolved_up"]].copy()
    meta["idx"] = ids
    meta_path = derived("retrieval_meta.parquet")
    meta.to_parquet(meta_path, index=False)

    log.info("built index with %d vectors; %s %s", len(X), index_path, meta_path)
    return len(X)


def _main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=16)
    ap.add_argument("--ef-construction", type=int, default=200)
    ap.add_argument("--ef-query", type=int, default=128)
    args = ap.parse_args()
    build(args.m, args.ef_construction, args.ef_query)


if __name__ == "__main__":
    _main()
