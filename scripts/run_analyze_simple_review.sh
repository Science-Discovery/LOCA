#!/bin/bash


# ===== EDIT THESE PARAMETERS =====
INPUT="PATH_TO_RESULTS.jsonl"
TEST_REF="TEST_FILE.jsonl"
OUTPUT="ANALYSIS_OUTPUT.csv"
# ==================================

# Build command
CMD="python -m scripts.analyze_simple_review --input \"$INPUT\" --test_ref \"$TEST_REF\" --output \"$OUTPUT\""

echo "Executing: $CMD"
eval $CMD
