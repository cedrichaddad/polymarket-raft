#!/usr/bin/env bash
# Sweep taker-gate variants: mode x entry window x persistence bars x fees.
# Writes data/derived/sweeps/taker_gate_sweep.csv with per-bucket breakdown.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/python3}"

OUT_DIR="${OUT_DIR:-$ROOT/data/derived/sweeps}"
mkdir -p "$OUT_DIR"

MODES=(${MODES:-baseline rule_gate model_gate})
WINDOWS=(${WINDOWS:-30 15 10 5})
PERSIST_BARS=(${PERSIST_BARS:-1 2 3})
FEES=(${FEES:-0.0 0.006 0.012})
SIGMAS=(${SIGMAS:-5e-4})

CALIBRATOR="$ROOT/data/derived/calibrator.json"
GATE_MODEL="$ROOT/data/derived/taker_gate_model.json"
TAKER_SUMMARY="$ROOT/data/derived/backtest_taker/summary.json"

CSV="$OUT_DIR/taker_gate_sweep.csv"

echo "mode,window_secs,persist_bars,taker_fee_prob,sigma_per_sec,n_trades,n_unique_markets,net_pnl_mean,net_pnl_std,t_stat,hit_rate,total_net_pnl,bucket_15_30_n,bucket_10_15_n,bucket_5_10_n,bucket_2_5_n,bucket_0_2_n,bucket_best_label,bucket_best_mean" > "$CSV"

run_bt() {
  (
    cd "$ROOT/research"
    "$PYTHON_BIN" -m raft_research.backtest_taker "$@" >/dev/null
  )
}

extract_row() {
  "$PYTHON_BIN" - "$TAKER_SUMMARY" <<'PY'
import json, sys, pathlib
p = pathlib.Path(sys.argv[1])
if not p.exists():
    print(",,,,,,,,,,,,,")
    sys.exit(0)
d = json.loads(p.read_text())
def g(k): v = d.get(k); return "" if v is None else v
buckets = {b["tte_bucket"]: b for b in d.get("by_tte_bucket", [])} if d.get("by_tte_bucket") else {}
def bn(label):
    b = buckets.get(label)
    return "" if not b else b.get("n", "")
best_label, best_mean = "", ""
if buckets:
    ranked = sorted(buckets.values(), key=lambda b: b.get("net_pnl_mean", -1e9), reverse=True)
    if ranked:
        best_label = ranked[0]["tte_bucket"]
        best_mean = ranked[0].get("net_pnl_mean", "")
fields = [
    g("n_trades"), g("n_unique_markets"), g("net_pnl_mean"), g("net_pnl_std"),
    g("t_stat"), g("hit_rate"), g("total_net_pnl"),
    bn("15_30"), bn("10_15"), bn("5_10"), bn("2_5"), bn("0_2"),
    best_label, best_mean,
]
print(",".join(str(x) for x in fields))
PY
}

for mode in "${MODES[@]}"; do
  for window in "${WINDOWS[@]}"; do
    for pb in "${PERSIST_BARS[@]}"; do
      # baseline is insensitive to persist_bars; only run once per (mode,window,fee).
      if [[ "$mode" == "baseline" && "$pb" != "${PERSIST_BARS[0]}" ]]; then continue; fi
      for fee in "${FEES[@]}"; do
        for sigma in "${SIGMAS[@]}"; do
          echo "==> mode=$mode window=$window persist=$pb fee=$fee sigma=$sigma"
          args=(--mode "$mode" --taker-fee-prob "$fee" --sigma-per-sec "$sigma")
          [[ -f "$CALIBRATOR" ]] && args+=(--calibrator "$CALIBRATOR")
          if [[ "$mode" == "baseline" ]]; then
            args+=(--taker-window-secs "$window")
          else
            args+=(--max-entry-window-secs "$window" --persist-bars "$pb" --require-persistence)
          fi
          if [[ "$mode" == "model_gate" ]]; then
            if [[ ! -f "$GATE_MODEL" ]]; then
              echo "  (missing $GATE_MODEL — skipping model_gate rows)"
              continue
            fi
            args+=(--gate-model "$GATE_MODEL")
          fi

          if ! run_bt "${args[@]}" 2>/dev/null; then
            echo "  (run failed — skipping)"
            continue
          fi
          row="$(extract_row)"
          echo "$mode,$window,$pb,$fee,$sigma,$row" >> "$CSV"
        done
      done
    done
  done
done

echo
echo "==> wrote $CSV"
echo
echo "==> top rows by total_net_pnl, t_stat, n_unique_markets"
"$PYTHON_BIN" - "$CSV" <<'PY'
import pandas as pd, sys, pathlib
p = pathlib.Path(sys.argv[1])
if not p.exists(): sys.exit(0)
df = pd.read_csv(p)
if df.empty: sys.exit(0)
for c in ("total_net_pnl","t_stat","n_unique_markets"):
    df[c] = pd.to_numeric(df[c], errors="coerce")
print(df.sort_values(["total_net_pnl","t_stat","n_unique_markets"], ascending=[False,False,False]).head(15).to_string(index=False))
PY
