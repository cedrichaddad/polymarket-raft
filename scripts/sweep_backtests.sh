#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

OUT_DIR="${OUT_DIR:-$ROOT/data/derived/sweeps}"
mkdir -p "$OUT_DIR"

TAKER_WINDOWS=(${TAKER_WINDOWS:-30 15})
TAKER_FEES=(${TAKER_FEES:-0.0 0.006 0.012})
TAKER_ALL_INS=(${TAKER_ALL_INS:-0.008 0.012 0.02})
SIGMAS=(${SIGMAS:-5e-4})

CENTER_LOWS=(${CENTER_LOWS:-0.40})
CENTER_HIGHS=(${CENTER_HIGHS:-0.60})
DWELLS=(${DWELLS:-2 3})
MIN_OPP_VOLUMES=(${MIN_OPP_VOLUMES:-25 50 100})
REBATE_MODES=(${REBATE_MODES:-none conservative optimistic})

CALIBRATOR="$ROOT/data/derived/calibrator.json"
TAKER_SUMMARY="$ROOT/data/derived/backtest_taker/summary.json"
MAKER_SUMMARY="$ROOT/data/derived/backtest_maker/summary.json"

run_python() {
  (
    cd "$ROOT/research"
    "$PYTHON_BIN" -m "$@"
  )
}

json_field() {
  local file="$1"
  local field="$2"
  "$PYTHON_BIN" - <<PY
import json, pathlib
p = pathlib.Path("$file")
if not p.exists():
    print("")
else:
    data = json.loads(p.read_text())
    v = data.get("$field", "")
    print(v if v is not None else "")
PY
}

TAKER_CSV="$OUT_DIR/taker_sweep.csv"
MAKER_CSV="$OUT_DIR/maker_sweep.csv"

echo "taker_window_secs,taker_fee_prob,all_in,sigma_per_sec,n_trades,net_pnl_mean,net_pnl_std,t_stat,hit_rate,total_net_pnl" > "$TAKER_CSV"
echo "center_low,center_high,min_dwell_secs,min_opp_volume,rebate_mode,n_fills,n_buy,n_sell,net_pnl_mean,net_pnl_std,total_net_pnl,markout_250ms_mean,markout_500ms_mean,markout_1000ms_mean,markout_2000ms_mean,markout_5000ms_mean" > "$MAKER_CSV"

echo "==> taker sweep"
for window in "${TAKER_WINDOWS[@]}"; do
  for fee in "${TAKER_FEES[@]}"; do
    for all_in in "${TAKER_ALL_INS[@]}"; do
      for sigma in "${SIGMAS[@]}"; do
        echo "taker: window=$window fee=$fee all_in=$all_in sigma=$sigma"
        if [[ -f "$CALIBRATOR" ]]; then
          run_python raft_research.backtest_taker \
            --taker-window-secs "$window" \
            --taker-fee-prob "$fee" \
            --all-in "$all_in" \
            --sigma-per-sec "$sigma" \
            --calibrator "$CALIBRATOR" >/dev/null
        else
          run_python raft_research.backtest_taker \
            --taker-window-secs "$window" \
            --taker-fee-prob "$fee" \
            --all-in "$all_in" \
            --sigma-per-sec "$sigma" >/dev/null
        fi

        n_trades="$(json_field "$TAKER_SUMMARY" n_trades)"
        net_pnl_mean="$(json_field "$TAKER_SUMMARY" net_pnl_mean)"
        net_pnl_std="$(json_field "$TAKER_SUMMARY" net_pnl_std)"
        t_stat="$(json_field "$TAKER_SUMMARY" t_stat)"
        hit_rate="$(json_field "$TAKER_SUMMARY" hit_rate)"
        total_net_pnl="$(json_field "$TAKER_SUMMARY" total_net_pnl)"

        echo "$window,$fee,$all_in,$sigma,$n_trades,$net_pnl_mean,$net_pnl_std,$t_stat,$hit_rate,$total_net_pnl" >> "$TAKER_CSV"
      done
    done
  done
done

echo "==> maker sweep"
for cl in "${CENTER_LOWS[@]}"; do
  for ch in "${CENTER_HIGHS[@]}"; do
    for dwell in "${DWELLS[@]}"; do
      for opp in "${MIN_OPP_VOLUMES[@]}"; do
        for rebate in "${REBATE_MODES[@]}"; do
          echo "maker: center=[$cl,$ch] dwell=$dwell opp=$opp rebate=$rebate"
          run_python raft_research.backtest_maker \
            --center-low "$cl" \
            --center-high "$ch" \
            --min-dwell-secs "$dwell" \
            --min-opp-volume "$opp" \
            --rebate-mode "$rebate" >/dev/null

          n_fills="$(json_field "$MAKER_SUMMARY" n_fills)"
          n_buy="$(json_field "$MAKER_SUMMARY" n_buy)"
          n_sell="$(json_field "$MAKER_SUMMARY" n_sell)"
          net_pnl_mean="$(json_field "$MAKER_SUMMARY" net_pnl_mean)"
          net_pnl_std="$(json_field "$MAKER_SUMMARY" net_pnl_std)"
          total_net_pnl="$(json_field "$MAKER_SUMMARY" total_net_pnl)"
          m250="$(json_field "$MAKER_SUMMARY" markout_250ms_mean)"
          m500="$(json_field "$MAKER_SUMMARY" markout_500ms_mean)"
          m1000="$(json_field "$MAKER_SUMMARY" markout_1000ms_mean)"
          m2000="$(json_field "$MAKER_SUMMARY" markout_2000ms_mean)"
          m5000="$(json_field "$MAKER_SUMMARY" markout_5000ms_mean)"

          echo "$cl,$ch,$dwell,$opp,$rebate,$n_fills,$n_buy,$n_sell,$net_pnl_mean,$net_pnl_std,$total_net_pnl,$m250,$m500,$m1000,$m2000,$m5000" >> "$MAKER_CSV"
        done
      done
    done
  done
done

echo
echo "==> wrote:"
echo "  $TAKER_CSV"
echo "  $MAKER_CSV"

echo
echo "==> top taker rows by total_net_pnl"
"$PYTHON_BIN" - <<PY
import pandas as pd
from pathlib import Path
p = Path("$TAKER_CSV")
if p.exists():
    df = pd.read_csv(p)
    if not df.empty:
        print(df.sort_values(["total_net_pnl","t_stat"], ascending=[False,False]).head(10).to_string(index=False))
PY

echo
echo "==> top maker rows by total_net_pnl"
"$PYTHON_BIN" - <<PY
import pandas as pd
from pathlib import Path
p = Path("$MAKER_CSV")
if p.exists():
    df = pd.read_csv(p)
    if not df.empty:
        print(df.sort_values(["total_net_pnl","net_pnl_mean"], ascending=[False,False]).head(10).to_string(index=False))
PY