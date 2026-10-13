#!/bin/bash
# Session-count sweep, Python and Rust interleaved per point so both see similar host noise.
# Usage: E2E_ROOT=... bench/sweep.sh TAG
# Mocks must already run (python e2e.py mocks start --upstreams 3).
# POINTS: gateway:sessions list; ATTEMPTS/QUIET_WAIT: reruns on a noisy gateway core.
set -u
TAG=${1:-final}
HERE=$(cd "$(dirname "$0")" && pwd)
PY="$E2E_ROOT/pyvenv/bin/python"
POINTS=${POINTS:-"py:150 rs:150 py:300 rs:300 py:450 rs:450 py:600 rs:600 rs:900 rs:1200 rs:1500 rs:1800"}
for point in $POINTS; do
  gw=${point%%:*}; n=${point##*:}
  shards=$(( n >= 600 ? 4 : 2 ))
  echo "== $(date +%T) $gw n=$n load=$(cut -d' ' -f1-3 /proc/loadavg)"
  "$PY" "$HERE/e2e.py" run --gateway "$gw" --sessions "$n" --shards "$shards" \
    --warmup "${WARMUP:-60}" --duration "${DURATION:-120}" --ramp "${RAMP:-30}" \
    --attempts "${ATTEMPTS:-1}" --quiet-wait "${QUIET_WAIT:-120}" --label "${TAG}_${gw}_n${n}"
done
echo "== $(date +%T) done"
