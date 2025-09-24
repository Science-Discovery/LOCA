import os
import json
from typing import List, Dict, Any
from src import ROOT_DIR, load_prompt
from src.chat import direct_chat

PROMPT_ARBITER = load_prompt(os.path.join(ROOT_DIR, "src/prompt/arbiter_prompt.json"))

DEFAULT_MODEL = "deepseek-r1"

async def arbiter_for_voting(
    task_id: str,
    problem_statement: str,
    candidate_solutions: List[Dict[str, str]],
    model: str = None,
    temperature: float = 0.0,
) -> str:
    """
    Arbiter function for voting among multiple candidate solutions.
    
    Args:
        task_id: Task identifier
        problem_statement: Original problem statement
        candidate_solutions: List of dictionaries containing 'solution' and 'final_answer' keys
        model: Model to use for arbitration
        temperature: Temperature for model generation
        
    Returns:
        The final answer selected by the arbiter
    """
    if model is None:
        model = DEFAULT_MODEL
    
    # Format candidate solutions for the prompt
    formatted_candidates = ""
    for i, candidate in enumerate(candidate_solutions, 1):
        formatted_candidates += f"## Candidate Solution {i}\n"
        formatted_candidates += f"**Solution:**\n{candidate['solution']}\n\n"
        formatted_candidates += f"**Final Answer:**\n{candidate['final_answer']}\n\n"
        formatted_candidates += "---\n\n"
    
    messages = PROMPT_ARBITER.replace({
        "problem_statement": problem_statement,
        "candidate_solutions": formatted_candidates.strip(),
        "num_candidates": str(len(candidate_solutions))
    })
    
    return await direct_chat(
        model=model,
        messages=messages,
        temperature=temperature
    )
