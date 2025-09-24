#!/usr/bin/env python3

import asyncio
import json
import os
import sys
import time
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.secretary import structured_secretary
from src.utils import load_jsonl


async def process_entry(entry, semaphore):
    """Process a single entry by calling structured_secretary on its improved_solutions."""
    async with semaphore:  # Limit concurrent API calls
        try:
            task_id = entry.get("id", "")
            improved_solutions = entry.get("improved_solutions", "")
            original_problem = entry.get("questions", "")
            
            if not improved_solutions:
                print(f"⚠️  Warning: No improved_solutions found for entry {task_id}")
                return entry
            
            # Call structured_secretary with is_final=True
            processed_solution = await structured_secretary(
                task_id=task_id,
                detailed_solution=improved_solutions,
                original_problem_statement=original_problem,
                model='gemini-2.5-pro',
                is_final=True,
            )
            
            # Create a new entry with the processed improved_solutions
            processed_entry = entry.copy()
            processed_entry["improved_solutions"] = processed_solution
            
            print(f"✅ Processed entry: {task_id}")
            return processed_entry
            
        except Exception as e:
            print(f"❌ Error processing entry {entry.get('id', 'unknown')}: {str(e)}")
            return entry


async def process_batch(entries, max_concurrent=8):
    """Process entries concurrently with controlled concurrency."""
    # Create semaphore to limit concurrent API calls
    semaphore = asyncio.Semaphore(max_concurrent)
    
    print(f"🚀 Processing {len(entries)} entries with max {max_concurrent} concurrent tasks...")
    start_time = time.time()
    
    # Create tasks for all entries
    tasks = [process_entry(entry, semaphore) for entry in entries]
    
    # Process all tasks concurrently
    processed_entries = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Handle any exceptions
    final_entries = []
    success_count = 0
    error_count = 0
    
    for i, result in enumerate(processed_entries):
        if isinstance(result, Exception):
            print(f"❌ Exception for entry {i}: {result}")
            final_entries.append(entries[i])  # Use original entry if processing failed
            error_count += 1
        else:
            final_entries.append(result)
            success_count += 1
    
    elapsed_time = time.time() - start_time
    print(f"🎉 Completed processing in {elapsed_time:.2f} seconds")
    print(f"📊 Success: {success_count}, Errors: {error_count}")
    
    return final_entries


async def main():
    """Main function to process all entries in the results file."""
    input_file = "test_results/PHYBench/gen/Gemini-2.5-pro/results.jsonl"
    output_file = "test_results/PHYBench/gen/Gemini-2.5-pro/results_processed.jsonl"
    
    # Configuration for concurrent processing
    MAX_CONCURRENT = 100  # Adjust based on API rate limits
    
    if not os.path.exists(input_file):
        print(f"❌ Error: Input file {input_file} not found!")
        return
    
    print(f"📂 Loading data from {input_file}...")
    
    try:
        entries = load_jsonl(input_file)
        print(f"📋 Loaded {len(entries)} entries")
    except Exception as e:
        print(f"❌ Error loading input file: {str(e)}")
        return
    
    # Process entries concurrently
    processed_entries = await process_batch(entries, MAX_CONCURRENT)
    
    print(f"💾 Writing processed results to {output_file}...")
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            for entry in processed_entries:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        
        print(f"✅ Successfully processed {len(processed_entries)} entries")
        print(f"📁 Results saved to {output_file}")
        
    except Exception as e:
        print(f"❌ Error writing output file: {str(e)}")


if __name__ == "__main__":
    asyncio.run(main())
