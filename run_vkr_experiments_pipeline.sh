#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

PYTHON_BIN="${PYTHON_BIN:-$PROJECT_DIR/.venv/bin/python}"

DATASET_ROOT="${DATASET_ROOT:-$PROJECT_DIR/data/preprocessed_data/preprocessed_DFDC02_16}"
DATASET_NAME="${DATASET_NAME:-dfdc02}"

CROSS_DATASET_ROOT="${CROSS_DATASET_ROOT:-$PROJECT_DIR/data/preprocessed_data/preprocessed_DFD01_16}"
CROSS_DATASET_NAME="${CROSS_DATASET_NAME:-dfd01}"

OUTPUT_DIR="${OUTPUT_DIR:-./experiments}"
RUN_LEVEL="${RUN_LEVEL:-mandatory}"   # mandatory | full
RUN_CROSS_DATASET="${RUN_CROSS_DATASET:-no}"

DEVICE="${DEVICE:-auto}"
NUM_WORKERS="${NUM_WORKERS:-0}"
BATCH_SIZE="${BATCH_SIZE:-8}"
MAX_EPOCHS="${MAX_EPOCHS:-30}"
SEED="${SEED:-42}"
DISABLE_AMP="${DISABLE_AMP:-no}"

timestamp() {
  date +"%Y-%m-%d %H:%M:%S"
}

log() {
  echo "[$(timestamp)] $*"
}

run_cmd() {
  log "RUN: $*"
  "$@"
}

require_file() {
  local path="$1"
  [[ -f "$path" ]] || { log "ERROR: file not found -> $path"; exit 1; }
}

require_dir() {
  local path="$1"
  [[ -d "$path" ]] || { log "ERROR: directory not found -> $path"; exit 1; }
}

require_file "$PROJECT_DIR/run_experiments.py"
require_file "$PROJECT_DIR/evaluate.py"
require_file "$PROJECT_DIR/train.py"
require_dir "$DATASET_ROOT"

if [[ ! -x "$PYTHON_BIN" ]]; then
  log "ERROR: python executable not found -> $PYTHON_BIN"
  exit 1
fi

mkdir -p "$OUTPUT_DIR"
LOG_DIR="$OUTPUT_DIR/overnight_logs"
mkdir -p "$LOG_DIR"

RUN_LOG="$LOG_DIR/vkr_pipeline_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$RUN_LOG") 2>&1

log "PROJECT_DIR=$PROJECT_DIR"
log "PYTHON_BIN=$PYTHON_BIN"
log "DATASET_ROOT=$DATASET_ROOT"
log "DATASET_NAME=$DATASET_NAME"
log "CROSS_DATASET_ROOT=$CROSS_DATASET_ROOT"
log "CROSS_DATASET_NAME=$CROSS_DATASET_NAME"
log "OUTPUT_DIR=$OUTPUT_DIR"
log "RUN_LEVEL=$RUN_LEVEL"
log "RUN_CROSS_DATASET=$RUN_CROSS_DATASET"
log "DEVICE=$DEVICE"
log "NUM_WORKERS=$NUM_WORKERS"
log "BATCH_SIZE=$BATCH_SIZE"
log "MAX_EPOCHS=$MAX_EPOCHS"
log "SEED=$SEED"
log "DISABLE_AMP=$DISABLE_AMP"

COMMON_ARGS=(
  --dataset_root "$DATASET_ROOT"
  --dataset_name "$DATASET_NAME"
  --output_dir "$OUTPUT_DIR"
  --level "$RUN_LEVEL"
  --device "$DEVICE"
  --num_workers "$NUM_WORKERS"
  --batch_size "$BATCH_SIZE"
  --max_epochs "$MAX_EPOCHS"
  --seed "$SEED"
)

if [[ "$DISABLE_AMP" == "yes" ]]; then
  COMMON_ARGS+=(--no_amp)
fi

if [[ "$RUN_CROSS_DATASET" == "yes" ]]; then
  require_dir "$CROSS_DATASET_ROOT"
  COMMON_ARGS+=(
    --cross_dataset_root "$CROSS_DATASET_ROOT"
    --cross_dataset_name "$CROSS_DATASET_NAME"
  )
fi

log "STEP 1: run experiment series"
run_cmd "$PYTHON_BIN" run_experiments.py "${COMMON_ARGS[@]}"

log "STEP 2: build final summary"

ALL_RESULTS_JSON="$OUTPUT_DIR/all_results.json"
RESULTS_TABLE_TXT="$OUTPUT_DIR/results_table.txt"
SUMMARY_MD="$OUTPUT_DIR/final_summary.md"

require_file "$ALL_RESULTS_JSON"
require_file "$RESULTS_TABLE_TXT"

OUTPUT_DIR_ABS="$(cd "$OUTPUT_DIR" && pwd)"

OUTPUT_DIR="$OUTPUT_DIR_ABS" "$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path

output_dir = Path(os.environ["OUTPUT_DIR"])
all_results_path = output_dir / "all_results.json"
results_table_path = output_dir / "results_table.txt"
summary_md_path = output_dir / "final_summary.md"

with open(all_results_path, "r", encoding="utf-8") as f:
    all_results = json.load(f)

success = [r for r in all_results if r.get("status") == "success"]
failed = [r for r in all_results if r.get("status") != "success"]

best = None
for r in success:
    auc = r.get("test", {}).get("auc")
    if auc is None:
        continue
    if best is None or auc > best.get("test", {}).get("auc", -1):
        best = r

lines = []
lines.append("# Final VKR Experiment Summary")
lines.append("")
lines.append(f"- Total experiments: {len(all_results)}")
lines.append(f"- Successful: {len(success)}")
lines.append(f"- Failed: {len(failed)}")
lines.append("")

if best is not None:
    lines.append("## Best experiment")
    lines.append("")
    lines.append(f"- Name: {best.get('experiment_name')}")
    lines.append(f"- Model: {best.get('model_type')}")
    lines.append(f"- Fusion: {best.get('fusion_type')}")
    lines.append(f"- Test AUC: {best.get('test', {}).get('auc')}")
    lines.append(f"- Test Accuracy: {best.get('test', {}).get('accuracy')}")
    lines.append(f"- Test F1: {best.get('test', {}).get('f1')}")
    if "cross_dataset_auc" in best:
        lines.append(f"- Cross-dataset AUC: {best.get('cross_dataset_auc')}")
    lines.append("")

if failed:
    lines.append("## Failed experiments")
    lines.append("")
    for r in failed:
        lines.append(f"- {r.get('experiment_name')}: {r.get('error', 'unknown error')}")
    lines.append("")

lines.append("## Results table")
lines.append("")
with open(results_table_path, "r", encoding="utf-8") as f:
    table_text = f.read().rstrip()

lines.append("```")
lines.append(table_text)
lines.append("```")
lines.append("")

with open(summary_md_path, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

print(f"Summary written to: {summary_md_path}")
PY

log "DONE"
log "LOG FILE: $RUN_LOG"
log "RESULTS JSON: $ALL_RESULTS_JSON"
log "RESULTS TABLE: $RESULTS_TABLE_TXT"
log "SUMMARY MD: $SUMMARY_MD"
