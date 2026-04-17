"""RAFT offline research stack (Python).

Modules mirror §15.2 of the design doc:

    collect_markets.py      -> raft_research.collect_markets
    collect_ws.py           -> raft_research.collect_ws (Rust service recommended)
    build_state_table.py    -> raft_research.build_state_table
    build_labels.py         -> raft_research.build_labels
    calibrate_prob.py       -> raft_research.calibrate
    build_vibrato_index.py  -> raft_research.build_index
    backtest_maker.py       -> raft_research.backtest_maker
    backtest_taker.py       -> raft_research.backtest_taker
    analyze_markouts.py     -> raft_research.markouts
    compare_models.py       -> raft_research.compare

These are deliberately thin: they read canonical Parquet under
``data/raw/source=.../date=.../*.parquet`` and write derived tables into
``data/derived/``.
"""
from .paths import DATA_ROOT, RAW_ROOT, DERIVED_ROOT  # noqa: F401
