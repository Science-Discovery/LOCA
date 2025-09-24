import json
import os
import asyncio
from typing import Optional, Any, Literal
from src import ROOT_DIR, RESULT_ROOT_DIR, load_prompt
from src.chat import direct_chat, safe_chat
from src.utils import Instruction, bold_stdout
from src.instructions import JudgeInstruction
# Import from the new vanilla_review module to maintain compatibility
from src.vanilla_review.reviewer import simple_review_final_answer

PROMPT_ADVANCED_ANSWER = load_prompt(os.path.join(ROOT_DIR, "src/loca/utils/prompt/reviewer_answer.json"))
if not PROMPT_ADVANCED_ANSWER:
    raise ValueError("Answer Reviewer prompt file not found or empty. Please check the path and content of the prompt file.")
print(bold_stdout("Reviewer prompt for Answer loaded successfully"))

PROMPT_SIMPLE_REVIEWER = load_prompt(os.path.join(ROOT_DIR, "src/loca/utils/prompt/simple_reviewer_prompt.json"))
if not PROMPT_SIMPLE_REVIEWER:
    raise ValueError("Simple reviewer prompt file not found or empty. Please check the path and content of the prompt file.")
print(bold_stdout("Simple reviewer prompt loaded successfully"))


async def advanced_reviewer_answer(
    task_id: str,
    question_statement: str,
    solution: str,
    instructions: dict[str, Instruction],
    model: Optional[str] = None, 
    temperature: Optional[float] = None,
) -> dict[str, Any]:
    if not model:
        raise ValueError("Model must be specified for advanced reviewer question.")

    prompt_template = PROMPT_ADVANCED_ANSWER

    messages_dict = {
        key: prompt_template.replace({
            "question_statement": question_statement,
            "solution": solution,
            "instruction": instruction.text
        }) for key, instruction in instructions.items()
    }

    record_full_api_path_dict = {
        key: os.path.join(
            os.path.join(RESULT_ROOT_DIR, ".api.origin"),
            "_".join([task_id, "advanced_review", key, model]) + ".json"
        ) for key in messages_dict.keys()
    }

    result = dict(zip(
        messages_dict.keys(),
        await asyncio.gather(*[
            safe_chat(
                model=model,
                messages=messages_dict[key],
                record_full_api_path=record_full_api_path_dict[key],
                temperature=temperature,
                validator=instructions[key].validator,
                formatter=instructions[key].formatter,
                debug=True,
            ) for key in messages_dict.keys()
        ])
    ))
    for key, value in result.items():
        if instructions[key].post_processor:
            result[key] = await instructions[key].post_processor(value)
    return result
