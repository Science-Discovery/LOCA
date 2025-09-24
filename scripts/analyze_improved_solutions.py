#!/usr/bin/env python3

import argparse
import asyncio
import csv
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from chat import direct_chat


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


def extract_final_answer(improved_solution: str) -> Optional[str]:
    pattern_final = r'(?s)Final Answer[:\s]*(.*)$'
    matches_final = re.findall(pattern_final, improved_solution, re.IGNORECASE | re.MULTILINE)
    
    if not matches_final:
        return None
        # return "\n".join(improved_solution.strip().splitlines()[-5:])
    
    content_after_final = matches_final[-1]
    
    pattern_bracket = r'(?s)\[(.*?)\]'
    matches_bracket = re.findall(pattern_bracket, content_after_final)
    
    if matches_bracket:
        return matches_bracket[-1].strip()
    else:
        return content_after_final.strip()


async def compare_answers_with_llm(
    original_answer: str, 
    improved_answer: str, 
    model: str,
    problem_id: str
) -> bool:
    """Use LLM to compare if two answers are equivalent."""
    
    prompt = f"""You are a physics expert tasked with comparing two mathematical / physical answers to determine if they are equivalent.

Original Answer: {original_answer}
Improved Answer: {improved_answer}

Please carefully analyze whether the "Final Answer" provided in the end of `Improved Answer` are equivalent to the answer provided in `Original Answer`.
You MUST NOT pay much attention to the intermediate steps, but focus on the final result.
For the final result, you should especially pay attention to the sign, the coefficients, and each symbol used.

Your response must end with exactly one of these symbols:
- ✅ if the answers are equivalent
- ❌ if the answers are not equivalent

Example response format:
The original answer shows... while the improved answer presents... After analysis, these represent the same result.
✅

Now analyze the given answers:"""
    
    # print(f"---\n Problem id: {problem_id}\n{prompt}\n---\n")

    try:
        response = await direct_chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        
        # Extract the last ✅ or ❌ from the response
        if '✅' in response:
            last_check = response.rfind('✅')
            last_cross = response.rfind('❌')
            if last_check > last_cross:
                return True
        
        if '❌' in response:
            last_check = response.rfind('✅')
            last_cross = response.rfind('❌')
            if last_cross > last_check:
                return False
        
        # If no clear symbol found, default to False
        print(f"Warning: Unclear LLM response for {problem_id}: {response[-100:]}")
        return False
        
    except Exception as e:
        print(f"Error comparing answers for {problem_id}: {e}")
        return False


async def process_single_problem(
    problem_id: str,
    input_entry: Optional[Dict],
    ref_entry: Dict,
    model: str,
    test_ref_entry: Optional[Dict] = None
) -> Tuple[str, str, str, str]:
    """Process a single problem and return CSV row data."""
    
    # Column 1: Problem ID
    col1 = problem_id
    
    # Column 2: Passed role-based review
    col2 = "✅" if input_entry is not None else "❌"
    
    # Column 3: Final answer matches original
    col3 = "❌"  # Default
    if input_entry is not None:
        try:
            # Use final_answers from test_ref if available, otherwise from input_entry
            if test_ref_entry is not None:
                original_answer = test_ref_entry.get("final_answers", "")
            else:
                original_answer = input_entry.get("final_answers", "")
            # If the original answer is a list, convert to a string of list elements
            if isinstance(original_answer, list):
                if len(original_answer) > 1:
                    raise ValueError(f"Original answer for {problem_id} is a list with multiple elements: {original_answer}")
                original_answer = original_answer[0] if original_answer else ""
            improved_solution = input_entry.get("improved_solutions", "")
            
            if original_answer and improved_solution:
                improved_final_answer = extract_final_answer(improved_solution)
                # print(f"problem id: {problem_id} | Final Answer: {improved_final_answer}")
                # improved_final_answer = improved_solution
                if improved_final_answer:
                    is_equivalent = await compare_answers_with_llm(
                        original_answer, improved_final_answer, model, problem_id
                    )
                    col3 = "✅" if is_equivalent else "❌"
                else:
                    print(f"Warning: Could not extract final answer from improved solution for {problem_id}")
        except Exception as e:
            print(f"Error processing answer comparison for {problem_id}: {e}")
    
    # Column 4: Original answer correctness
    col4 = ref_entry.get("original_answer_correctness", "❌")
    
    return col1, col2, col3, col4


async def main():
    parser = argparse.ArgumentParser(description="Analyze improved solutions and generate CSV report")
    parser.add_argument("--input", required=True, help="Input JSONL file with improved solutions")
    parser.add_argument("--ref", required=True, help="Reference JSONL file with original correctness")
    parser.add_argument("--test_ref", help="Original test JSONL file")
    parser.add_argument("--output", required=True, help="Output CSV file path")
    parser.add_argument("--model", default="gemini-2.5-flash", help="Model for answer comparison")
    
    args = parser.parse_args()
    
    print(f"Loading input file: {args.input}")
    input_entries = await load_jsonl_file(args.input)
    
    print(f"Loading reference file: {args.ref}")
    ref_entries = await load_jsonl_file(args.ref)
    
    # Load test reference file if specified
    test_ref_entries = None
    if args.test_ref:
        print(f"Loading test reference file: {args.test_ref}")
        test_ref_entries = await load_jsonl_file(args.test_ref)
    
    # Verify file lengths match
    if len(input_entries) != len(ref_entries):
        print(f"Warning: Input file has {len(input_entries)} entries, ref file has {len(ref_entries)} entries")
    if test_ref_entries and len(test_ref_entries) != len(ref_entries):
        print(f"Warning: Test ref file has {len(test_ref_entries)} entries, ref file has {len(ref_entries)} entries")
    
    # Create problem ID mapping for reference entries
    ref_by_id = {}
    for ref_entry in ref_entries:
        if ref_entry is not None and "id" in ref_entry:
            ref_by_id[ref_entry["id"]] = ref_entry
    
    # Collect tasks for concurrent processing
    tasks = []
    problem_ids = []
    
    for i, (input_entry, ref_entry) in enumerate(zip(input_entries, ref_entries)):
        if ref_entry is None:
            continue
            
        problem_id = ref_entry.get("id")
        if not problem_id:
            print(f"Warning: No ID found in reference entry at line {i + 1}")
            continue
        
        # Verify ID consistency (if input entry is not null)
        if input_entry is not None:
            input_id = input_entry.get("id")
            if input_id != problem_id:
                print(f"Warning: ID mismatch at line {i + 1}: input={input_id}, ref={problem_id}")
                continue
        
        problem_ids.append(problem_id)
        # Get corresponding test_ref entry if available
        test_ref_entry = None
        if test_ref_entries and i < len(test_ref_entries):
            test_ref_entry = test_ref_entries[i]
        task = process_single_problem(problem_id, input_entry, ref_entry, args.model, test_ref_entry)
        tasks.append(task)
    
    print(f"Processing {len(tasks)} problems concurrently...")
    
    # Process all problems concurrently
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Write CSV output
    with open(args.output, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write header
        writer.writerow([
            "Problem ID",
            "Passed Role-based Review",
            "Final Answer Matches Original",
            "Original Answer Correctness"
        ])
        
        # Write data rows
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"Error processing {problem_ids[i]}: {result}")
                writer.writerow([problem_ids[i], "❌", "❌", "❌"])
            else:
                writer.writerow(result)
    
    print(f"Analysis complete. Results written to: {args.output}")
    
    # Print summary statistics
    successful_results = [r for r in results if not isinstance(r, Exception)]
    if successful_results:
        passed_review = sum(1 for r in successful_results if r[1] == "✅")
        matching_answers = sum(1 for r in successful_results if r[2] == "✅")
        correct_originals = sum(1 for r in successful_results if r[3] == "✅")
        
        print(f"\nSummary:")
        print(f"Total problems processed: {len(successful_results)}")
        print(f"Passed role-based review: {passed_review} ({passed_review / len(successful_results) * 100:.1f}%)")
        print(f"Final answers match original: {matching_answers} ({matching_answers / len(successful_results) * 100:.1f}%)")
        print(f"Original answers correct: {correct_originals} ({correct_originals / len(successful_results) * 100:.1f}%)")
        
        # Print unsuccessful problem IDs (those that encountered exceptions during processing)
        unsuccessful_ids = [problem_ids[i] for i, r in enumerate(results) if isinstance(r, Exception)]
        if unsuccessful_ids:
            print(f"\nUnsuccessful problems (processing exceptions): {len(unsuccessful_ids)}")
            print("Problem IDs:", ", ".join(unsuccessful_ids))


if __name__ == "__main__":
    asyncio.run(main())
