import json
import os
import asyncio
from typing import Optional, Any, Literal
from src import ROOT_DIR, RESULT_ROOT_DIR, load_prompt
from src.chat import direct_chat, safe_chat
from src.utils import Instruction, bold_stdout
from src.instructions import JudgeInstruction

PROMPT_SIMPLE_REVIEWER = load_prompt(os.path.join(ROOT_DIR, "src/vanilla_review/prompt/simple_reviewer_prompt.json"))
if not PROMPT_SIMPLE_REVIEWER:
    raise ValueError("Simple reviewer prompt file not found or empty. Please check the path and content of the prompt file.")
print(bold_stdout("Simple reviewer prompt loaded successfully"))


async def simple_review_final_answer(
    task_id: str,
    question_statement: str,
    solution: str,
    model: str = None,
    temperature: float = 0.0,
) -> dict[str, Any]:
    """
    Simple review of final answer correctness.
    
    Args:
        task_id: The unique identifier for the current task
        question_statement: The original physics problem statement
        solution: The complete solution to be reviewed
        model: The model to use for reviewing
        temperature: Temperature setting for the model
        
    Returns:
        dict: Contains 'answer' (str) and 'judge' (str) - same format as other reviewers
    """
    # Replace placeholders in the prompt template
    messages = PROMPT_SIMPLE_REVIEWER.replace({
        "question_statement": question_statement,
        "solution": solution,
    })

    # Set up API call logging path
    record_full_api_path = os.path.join(
        os.path.join(RESULT_ROOT_DIR, ".api.origin"),
        "_".join([task_id, "simple_review", model]) + ".json"
    )

    # Make the API call with validation using JudgeInstruction
    result = await safe_chat(
        model=model,
        messages=messages,
        record_full_api_path=record_full_api_path,
        temperature=temperature,
        validator=JudgeInstruction.validator,
        formatter=JudgeInstruction.formatter,
        debug=True,
    )

    return result
