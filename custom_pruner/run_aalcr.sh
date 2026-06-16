#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVALS_DIR="/data/akaush39/ai-model-quality-challenge/Evals/Part 1"
OUTPUT_DIR="$SCRIPT_DIR/output/aalcr"
LOG_DIR="$SCRIPT_DIR/logs"
LOG_FILE="$LOG_DIR/aalcr.log"
TARGET_SIZE="${1:-0}"
TARGET_LABEL="${TARGET_SIZE:-auto}"
[ "$TARGET_SIZE" -eq 0 ] 2>/dev/null && TARGET_LABEL="auto"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

echo "Running AA-LCR pruner  (target=$TARGET_LABEL)" | tee "$LOG_FILE"
echo "Output → $OUTPUT_DIR"
echo "Log    → $LOG_FILE"
echo "────────────────────────────────────────────────────"

python "$SCRIPT_DIR/run_pruner.py" \
    aa_lcr \
    "$EVALS_DIR" \
    "$OUTPUT_DIR" \
    "$TARGET_SIZE" \
    2>&1 | tee -a "$LOG_FILE"

echo "Done. Exit code: $?"
