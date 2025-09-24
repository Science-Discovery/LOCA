import os
from typing import Literal, Optional
import asyncio
import re
from src import ROOT_DIR, RESULT_ROOT_DIR, load_prompt
from src.chat import direct_chat
from src.arbiter import arbiter_for_voting

SINGLE_LLM_PROMPT_DIR = os.path.join(ROOT_DIR, "src/prompt/single_llm")
PROMPT = load_prompt(os.path.join(SINGLE_LLM_PROMPT_DIR, "prompt.json"))
PROMPT_ZERO_SHOT_COT = load_prompt(os.path.join(SINGLE_LLM_PROMPT_DIR, "prompt_zero_shot_cot.json"))
PROMPT_COT = load_prompt(os.path.join(SINGLE_LLM_PROMPT_DIR, "cot.json"))
PROMPT_FIX_HISTORY = load_prompt(
    os.path.join(SINGLE_LLM_PROMPT_DIR, "prompt_fix_history_answer.json")
)

MODEL_DEFAULT = "deepseek-r1"

async def single_llm_solver(
    task_id: str,
    problem_statement: str,
    model: str = None,
    temperature: float = None,
    history_answer: str = None,
    history_review: str = None,
    options: dict = None,
) -> str:
    if not model:
        model = MODEL_DEFAULT
    
    if options is None:
        options = {}

    voting_n = options.get("voting_n", 1)
    
    # CoT-SC: When voting_n > 1, use CoT prompt for multiple sampling and vote for the best answer
    if voting_n > 1:
        return await _cot_sc_solver(
            task_id=task_id,
            problem_statement=problem_statement,
            model=model,
            temperature=temperature,
            history_answer=history_answer,
            history_review=history_review,
            voting_n=voting_n,
        )
    
    zero_shot_cot = options.get("zero_shot_cot", False)
    cot = options.get("cot", False)

    if history_answer and history_review:
        messages = PROMPT_FIX_HISTORY.replace({
            "problem_statement": problem_statement,
            "history_answer": history_answer, "history_review": history_review,
        })
    else:
        if zero_shot_cot:
            messages = PROMPT_ZERO_SHOT_COT.replace({
                "problem_statement": problem_statement,
            })
        elif cot:
            messages = PROMPT_COT.replace({
                "problem_statement": problem_statement,
            })
        else:
            messages = PROMPT.replace({
                "problem_statement": problem_statement,
            })
    
    api_record_path = os.path.join(
        RESULT_ROOT_DIR, ".api.origin",
        f"{task_id}_single_llm_{model}.json"
    )
    
    result = await direct_chat(
        model=model,
        messages=messages,
        record_full_api_path=api_record_path,
        temperature=temperature
    )
    
    os.makedirs(os.path.join(RESULT_ROOT_DIR, task_id), exist_ok=True)
    with open(os.path.join(RESULT_ROOT_DIR, task_id, "single_llm_solution.md"), "w") as f:
        f.write(result)
    
    return result


def extract_final_answer(improved_solution: str) -> Optional[str]:

    pattern_final = r'(?s)Final Answer[:\s]*(.*)$'
    matches_final = re.findall(pattern_final, improved_solution, re.IGNORECASE | re.MULTILINE)
    
    if not matches_final:
        return None

    content_after_final = matches_final[-1]

    pattern_bracket = r'(?s)\[(.*?)\]'
    matches_bracket = re.findall(pattern_bracket, content_after_final)
    
    if matches_bracket:
        return matches_bracket[-1].strip()
    else:
        return content_after_final.strip()


async def _cot_sc_solver(
    task_id: str,
    problem_statement: str,
    model: str,
    temperature: float = None,
    history_answer: str = None,
    history_review: str = None,
    voting_n: int = 5,
) -> str:
    if history_answer and history_review:
        messages = PROMPT_FIX_HISTORY.replace({
            "problem_statement": problem_statement,
            "history_answer": history_answer, 
            "history_review": history_review,
        })
    else:
        messages = PROMPT_COT.replace({
            "problem_statement": problem_statement,
        })
    
    model_lower = model.lower()
    needs_sequential = "gemini" in model_lower or "r1" in model_lower
    
    if needs_sequential:
        candidate_results = []
        for i in range(voting_n):
            result = await direct_chat(
                model=model,
                messages=messages,
                temperature=temperature
            )
            candidate_results.append(result)
    else:
        tasks = []
        for i in range(voting_n):
            task = direct_chat(
                model=model,
                messages=messages,
                temperature=temperature
            )
            tasks.append(task)
        
        candidate_results = await asyncio.gather(*tasks)
    
    candidate_solutions = []
    for result in candidate_results:
        candidate_solutions.append({
            "solution": result,
            "final_answer": extract_final_answer(result),
        })
    
    final_result = await arbiter_for_voting(
        task_id=task_id,
        problem_statement=problem_statement,
        candidate_solutions=candidate_solutions,
        model=model,
        temperature=0.0,
    )
    
    return final_result
