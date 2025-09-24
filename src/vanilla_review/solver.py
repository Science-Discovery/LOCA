import os
import json
import asyncio
from typing import Optional

from src.vanilla_review.reviewer import simple_review_final_answer


async def vanilla_review_solver(
    task_id: str,
    question_statement: str,
    solution: str,
    output_aux_path: Optional[str] = None,
    review_model: str = "gemini-2.5-pro",
    max_error_times: int = 5,
    target_times: int = 3,
    temperature: float = 0.0
) -> tuple[bool, str]:
    """
    Simple review of final answer with consistency checking using error_count and consistent_count.
    
    Args:
        task_id: The unique identifier for the current task
        question_statement: The physics question
        solution: The complete solution text
        output_aux_path: path of output aux file
        review_model: The model to use for reviewing
        max_error_times: Maximum number of allowed errors before failing
        target_times: Number of consecutive consistent results needed for success
        temperature: Temperature setting for the models
        
    Returns:
        tuple[bool, str]: (success, result_text)
            - If successful: (True, "Final answer is correct")
            - If failed: (False, "Final answer is incorrect")
    """

    error_count = 0
    consistent_count = 0
    turn = 0

    # Check for existing review files to resume from breakpoint
    if output_aux_path and os.path.exists(output_aux_path):
        # Scan for existing simple_review_*.json files
        existing_files = []
        for filename in os.listdir(output_aux_path):
            if filename.startswith("simple_review_") and filename.endswith(".json"):
                try:
                    turn_num = int(filename.replace("simple_review_", "").replace(".json", ""))
                    existing_files.append(turn_num)
                except ValueError:
                    continue
        
        if existing_files:
            existing_files.sort()
            print(f"Found existing review files for task {task_id}, resuming from turn {existing_files[-1] + 1}", flush=True)
            
            # Reconstruct state from existing files
            consecutive_correct = 0
            total_errors = 0
            
            for file_turn in existing_files:
                file_path = os.path.join(output_aux_path, f"simple_review_{file_turn}.json")
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        review_data = json.load(f)
                    
                    is_correct = review_data.get("judge", "Wrong") == "Correct"
                    if is_correct:
                        consecutive_correct += 1
                    else:
                        consecutive_correct = 0  # Reset consecutive count on any wrong answer
                        total_errors += 1
                
                except (json.JSONDecodeError, FileNotFoundError):
                    print(f"Warning: Could not read review file {file_path}, treating as corrupted")
                    continue
            
            # Check if we already reached target_times
            if consecutive_correct >= target_times:
                print(f"Task {task_id} already completed successfully in previous run", flush=True)
                return True, "Final answer is correct"
            
            # Check if we already exceeded max_error_times
            if total_errors > max_error_times:
                print(f"Task {task_id} already failed in previous run", flush=True)
                return False, "Final answer is incorrect"
            
            # Set state for resuming
            consistent_count = consecutive_correct
            error_count = total_errors
            turn = existing_files[-1]
            print(f"Resuming task {task_id}: turn={turn}, error_count={error_count}, consistent_count={consistent_count}",
                  flush=True)
    
    while True:
        turn += 1

        review_result = await simple_review_final_answer(
            task_id=task_id,
            question_statement=question_statement,
            solution=solution,
            model=review_model,
            temperature=temperature
        )

        if output_aux_path is not None:
            # Save simple review result
            record_file_at_aux_path = os.path.join(output_aux_path, f"simple_review_{turn}.json")
            with open(record_file_at_aux_path, "w", encoding="utf-8") as f:
                json.dump(review_result, f, ensure_ascii=False, indent=2)

        is_correct = review_result.get("judge", "Wrong") == "Correct"

        if not is_correct:
            consistent_count = 0
            error_count += 1

            if error_count > max_error_times:
                return False, "Final answer is incorrect"

        else:
            consistent_count += 1

            if consistent_count == target_times:
                return True, "Final answer is correct"
