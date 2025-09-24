#!/bin/bash


# ===== EDIT THESE PARAMETERS =====
INPUT="PATH_TO_RESULTS.jsonl"
REF="CORRECTNESS_REFERENCE.jsonl"
TEST_REF="TEST_FILE.jsonl"
OUTPUT="ANALYSIS_OUTPUT.csv"
MODEL="gemini-2.5-flash"  # gpt-5
# ==================================

# Build command
# NOTE: scripts.analyze_improved_solutions_numeric for ABench or scripts.analyze_improved_solutions for others !!!
CMD="python -m scripts.analyze_improved_solutions --input \"$INPUT\" --ref \"$REF\" --test_ref \"$TEST_REF\" --output \"$OUTPUT\" --model \"$MODEL\""

echo "Executing: $CMD"
eval $CMD
