import json
import os
import asyncio
from typing import Optional
from src import ROOT_DIR, RESULT_ROOT_DIR, load_prompt
from src.chat import direct_chat
from src.utils import Instruction, bold_stdout

PROMPT_AUG_ANSWER = load_prompt(os.path.join(ROOT_DIR, "src/loca/utils/prompt/augment_answer.json"))
if not PROMPT_AUG_ANSWER:
    raise ValueError("Augmentation prompt file not found or empty. Please check the path and content of the prompt file.")
print(bold_stdout("Augmentation prompt for answer loaded successfully"))

PROMPT_SIMPLE_REFINE_ANSWER = load_prompt(os.path.join(ROOT_DIR, "src/loca/utils/prompt/simple_refine_prompt.json"))
if not PROMPT_SIMPLE_REFINE_ANSWER:
    raise ValueError("Refine prompt file not found or empty. Please check the path and content of the prompt file.")
print(bold_stdout("Refine prompt for answer loaded successfully"))


async def augment_origin_answer(
    task_id: str,
    question_statement: str,
    solution: str,
    bugs_report: str,
    aug_model: str,
    temperature: Optional[float] = None,
    turns: Optional[int] = None,
) -> str:
    """
    Augment a solution for the given physics question using a language model.

    Args:
        task_id (str): The unique identifier for the current task.
        question_statement (str): The description of the question.
        solution (str): The solution to be augmented.
        bugs_report (str): The report of bugs has been found in the solution from last review.
        aug_model (str): The model to use for augmentation.
        temperature (Optional[float]): The temperature to use for the model.
        turns (Optional[int]): The number of augmentation turns.

    Returns:
        str: The augmented solution as a string.
    """
    if turns == 1:
        return solution  # An identical augmentation turn, return original solution

    aug_messages = PROMPT_AUG_ANSWER.replace({
        "question_statement": question_statement,
        "solution": solution,
        "bugs_report": bugs_report
    })

    aug_record_path = os.path.join(
        os.path.join(RESULT_ROOT_DIR, ".api.origin"),
        "_".join([task_id, "aug", aug_model]) + ".json"
    )
    
    augmented_solution = await direct_chat(
        model=aug_model,
        messages=aug_messages,
        record_full_api_path=aug_record_path,
        temperature=temperature
    )
    return augmented_solution


async def simple_refiner(
    task_id: str,
    question_statement: str,
    solution: str,
    feedback: str,
    model: str,
    temperature: float = 0.0,
    turns: Optional[int] = None
) -> str:
    """
    Simple refiner for ablation study.
    
    Args:
        task_id: The unique identifier for the current task
        question_statement: The physics question statement
        solution: The current solution to refine
        feedback: Feedback from the reviewer about errors
        model: The model to use for refinement
        temperature: Temperature setting for the model
        turns: The number of refinement turns
        
    Returns:
        str: The refined solution
    """
    if turns == 1:
        return solution  # An identical refinement turn, return original solution
    
    messages = PROMPT_SIMPLE_REFINE_ANSWER.replace({
        "question_statement": question_statement,
        "solution": solution,
        "feedback": feedback
    })
    
    try:
        refined_solution = await direct_chat(
            model=model,
            messages=messages,
            temperature=temperature
        )
        
        return refined_solution.strip()
        
    except Exception as e:
        print(f"Error in simple_refiner for task {task_id}: {e}")
        # In case of error, return original solution
        return solution
