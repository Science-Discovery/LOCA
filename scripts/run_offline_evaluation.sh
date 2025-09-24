#!/bin/bash

# Offline evaluation Script - Edit parameters below before running
SCHEME="BENCHMARK_NAME"  # "phybench", "ABench"

# ===== EDIT THESE PARAMETERS =====
INPUT="PATH_TO_TEST.jsonl"
# CLEANED_INPUT="PATH_TO_CLEANED_TEST.jsonl"  # Optional, can be omitted
RESULTS="PATH_TO_RESULTS.jsonl"
OUTPUT="PATH_TO_OUTPUT_DIR"
# ==================================

# Build command
CMD="uv run offline_evaluate.py --input \"$INPUT\" --cleaned_input \"$CLEANED_INPUT\" --results \"$RESULTS\" --output_dir \"$OUTPUT\" --scheme \"$SCHEME\""

echo "Executing: $CMD"
eval $CMD
