#!/bin/bash
#
# LoCoMo evaluation: Claude Code Prompted (vanilla auto-memory) without
# per-conv isolation. All 10 LoCoMo samples share one cwd / HOME so CC
# accumulates a single cross-sample MEMORY.md (mirrors `run_sdk_noiso.sh`
# but with no OpenViking in the loop).
#
# Same 4-phase ingest + judge as run_prompted.sh, plus a snapshot/restore
# step between QA convs so QA-time writes don't leak across samples.
#
# Required env:
#   ANTHROPIC_AUTH_TOKEN / ANTHROPIC_API_KEY
#   ANTHROPIC_BASE_URL
#   ANTHROPIC_MODEL
#
# Optional env:
#   LOCOMO_INPUT        - dataset path (default .tmp/locomo10.json)
#   WORK_ROOT           - cwd + HOME parent (default /tmp/locomo-prompted-noiso)
#                         MUST be outside the git working tree on purpose:
#                         CC's SDK injects `git status` / cwd / gitBranch into
#                         every claude -p call's system prompt. If WORK_ROOT
#                         sits inside this repo, code-fine-tuned models
#                         (e.g. doubao-seed-2-0-code-preview) read that
#                         injection as "you are in a coding agent" and run
#                         git operations against the real repo.
#   PROMPT_PREFIX       - the auto-memory trigger phrase (same default as
#                         run_prompted.sh; required for non-zero accuracy)
#   QA_PARALLEL         - within-conv QA parallelism (default 5)

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Strip any OpenViking env the parent shell may have exported (e.g. via the
# `claude` shell wrapper). Prompted baseline must be pure vanilla CC.
unset OPENVIKING_URL OPENVIKING_BASE_URL OPENVIKING_API_KEY OPENVIKING_BEARER_TOKEN \
      OPENVIKING_AGENT_ID OPENVIKING_USER OPENVIKING_CONFIG_FILE OPENVIKING_CLI_CONFIG_FILE \
      OPENVIKING_MEMORY_ENABLED OPENVIKING_DEBUG OPENVIKING_WRITE_PATH_ASYNC

INPUT="${LOCOMO_INPUT:-$SCRIPT_DIR/.tmp/locomo10.json}"
WORK_ROOT="${WORK_ROOT:-/tmp/locomo-prompted-noiso}"
PROJECT_ROOT="$WORK_ROOT/shared"
HOME_DIR="$WORK_ROOT/HOME"
RESULT_DIR="$SCRIPT_DIR/.tmp/result-prompted-noiso"
SNAP_DIR="$RESULT_DIR/HOME.postingest"
mkdir -p "$RESULT_DIR" "$PROJECT_ROOT"

PROMPT_PREFIX="${PROMPT_PREFIX:-Please remember the following conversation using auto-memory.}"
QA_PARALLEL="${QA_PARALLEL:-5}"

SAMPLES=(conv-26 conv-30 conv-41 conv-42 conv-43 conv-44 conv-47 conv-48 conv-49 conv-50)
if [ $# -ge 1 ]; then SAMPLES=("$1"); fi

# ---- Phase 1: ingest (shared cwd, sessions serial) ------------------------
echo "[1/5] ingest into shared cwd..."
for SID in "${SAMPLES[@]}"; do
  echo "  -- $SID --"
  uv run python "$SCRIPT_DIR/ingest.py" \
    --input "$INPUT" \
    --sample "$SID" \
    --shared-project-dir \
    --project-root "$PROJECT_ROOT" \
    --home "$HOME_DIR" \
    --record "$RESULT_DIR/.ingest_record.json" \
    --success-csv "$RESULT_DIR/ingest_success.csv" \
    --error-log "$RESULT_DIR/ingest_errors.log" \
    --prompt-prefix "$PROMPT_PREFIX"
done

# ---- Phase 2: snapshot HOME ----------------------------------------------
echo "[2/5] snapshot HOME -> $SNAP_DIR"
rm -rf "$SNAP_DIR"
cp -r "$HOME_DIR" "$SNAP_DIR"

# ---- Phase 3: per-conv QA with HOME restore -------------------------------
echo "[3/5] per-conv QA (parallel=$QA_PARALLEL within conv)"
for SID in "${SAMPLES[@]}"; do
  OUT="$RESULT_DIR/qa_results_${SID}.csv"
  [ -f "$OUT" ] && { echo "  [$SID] CSV exists; skipping"; continue; }

  echo "  -- $SID --"
  rm -rf "$HOME_DIR"
  cp -r "$SNAP_DIR" "$HOME_DIR"

  uv run python "$SCRIPT_DIR/eval.py" \
    --input "$INPUT" \
    --output "$OUT" \
    --sample "$SID" \
    --shared-cwd \
    --project-root "$PROJECT_ROOT" \
    --home "$HOME_DIR" \
    --parallel "$QA_PARALLEL"
done

# ---- Phase 4: merge per-conv CSVs ----------------------------------------
echo "[4/5] merge per-conv CSVs"
MERGED="$RESULT_DIR/qa_results.csv"
shopt -s nullglob
PER_SAMPLE=( "$RESULT_DIR"/qa_results_conv-*.csv )
head -1 "${PER_SAMPLE[0]}" > "$MERGED"
for f in "${PER_SAMPLE[@]}"; do tail -n +2 "$f" >> "$MERGED"; done

# ---- Phase 5: judge + stats -----------------------------------------------
echo "[5/5] judge + stats"
ARK_API_KEY="${ANTHROPIC_AUTH_TOKEN:-${ANTHROPIC_API_KEY:-}}" \
  uv run python "$SCRIPT_DIR/judge.py" --input "$MERGED" --parallel 40

uv run python "$SCRIPT_DIR/stat_judge_result.py" --input "$MERGED" \
  | tee "$RESULT_DIR/summary.txt"
