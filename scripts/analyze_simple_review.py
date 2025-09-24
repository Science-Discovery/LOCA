#!/usr/bin/env python3

import argparse
import asyncio
import csv
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple


async def load_jsonl_file(file_path: str) -> List[Optional[Dict]]:
    """Load JSONL file and return list of entries (None for null lines)."""
    entries = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line == 'null' or line == '':
                entries.append(None)
            else:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Warning: Failed to parse line: {line[:100]}...")
                    entries.append(None)
    return entries


async def main():
    parser = argparse.ArgumentParser(description="Analyze simple review results and generate CSV report")
    parser.add_argument("--input", required=True, help="Input JSONL file with simple review results")
    parser.add_argument("--test_ref", required=True, help="Original test JSONL file")
    parser.add_argument("--output", required=True, help="Output CSV file path")
    
    args = parser.parse_args()
    
    print(f"Loading input file: {args.input}")
    input_entries = await load_jsonl_file(args.input)
    
    # Load test reference file if specified
    test_ref_entries = None
    print(f"Loading test reference file: {args.test_ref}")
    test_ref_entries = await load_jsonl_file(args.test_ref)
    
    # Verify file lengths match
    if len(test_ref_entries) != len(input_entries):
        raise ValueError(f"Input file and test reference file must have the same number of entries.")
    
    # Collect tasks for concurrent processing
    results = []
    
    for i, (input_entry, ref_entry) in enumerate(zip(input_entries, test_ref_entries)):
        problem_id = ref_entry.get("id")
        if not problem_id:
            raise ValueError(f"Warning: No ID found in reference entry at line {i + 1}")
        
        # Verify ID consistency (if input entry is not null)
        if input_entry is not None:
            input_id = input_entry.get("id")
            if input_id != problem_id:
                raise ValueError(f"Warning: ID mismatch at line {i + 1}: input={input_id}, ref={problem_id}")
                continue
            if input_entry.get("vanilla_review_result", False):
                results.append((problem_id, "✅"))
            else:
                results.append((problem_id, "❌"))

    # Write CSV output
    with open(args.output, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write header
        writer.writerow([
            "Problem ID",
            "Passed Review",
        ])
        
        # Write data rows
        for i, result in enumerate(results):
            writer.writerow(result)
    
    print(f"Analysis complete. Results written to: {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
