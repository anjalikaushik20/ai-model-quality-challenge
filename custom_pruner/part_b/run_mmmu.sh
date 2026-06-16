#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVALS_DIR="/data/akaush39/ai-model-quality-challenge/Evals/MMMU"
OUTPUT_DIR="$SCRIPT_DIR/output/mmmu"
LOG_DIR="$SCRIPT_DIR/logs"
LOG_FILE="$LOG_DIR/mmmu.log"
TARGET_SIZE="${1:-42}"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

echo "Running MMMU pruner  (target=$TARGET_SIZE)" | tee "$LOG_FILE"
echo "Output → $OUTPUT_DIR"
echo "Log    → $LOG_FILE"
echo "────────────────────────────────────────────────────"

python "$SCRIPT_DIR/run_pruner.py" \
    "$EVALS_DIR" \
    "$OUTPUT_DIR" \
    "$TARGET_SIZE" \
    2>&1 | tee -a "$LOG_FILE"

echo "Done. Exit code: $?"
