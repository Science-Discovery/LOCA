
import os
import json
from src import ROOT_DIR, RESULT_ROOT_DIR, load_prompt
from src.chat import direct_chat
from src.utils import bold_stdout


PROMPT_SECRETARY_FOR_BUGS = load_prompt(os.path.join(ROOT_DIR, "src/loca/utils/prompt/secretary.json"))
if not PROMPT_SECRETARY_FOR_BUGS:
    raise ValueError("Secretary for bugs prompt file not found or empty. Please check the path and content of the prompt file.")
print(bold_stdout("Secretary for bugs prompt loaded successfully"))

async def secretary_for_bugs(
    task_id: str,
    detailed_review: str,
    model: str = None,
    temperature: float = 0,
) -> str:
    if model is None:
        raise ValueError("Model must be specified for secretary_for_bugs")
    template = PROMPT_SECRETARY_FOR_BUGS
    messages = template.replace({
        "review": detailed_review
    })
    return await direct_chat(
        model=model,
        messages=messages,
        temperature=temperature
    )
