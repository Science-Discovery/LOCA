import os
import json
import hashlib
from pathlib import Path
from dataclasses import dataclass
from typing import Literal, TypeAlias, TypedDict, Any, Callable, Optional, Awaitable, Union

Problem: TypeAlias = dict[Literal["id", "questions"], str]
ProblemSet: TypeAlias = list[Problem]
class ProblemWithAnswer(TypedDict):
    id: str
    questions: str
    solutions: str
    final_answers: list[str]
ProblemSetWithAnswer: TypeAlias = list[ProblemWithAnswer]
def extract_problem(pwa: ProblemWithAnswer) -> Problem:
    return {
        "id": pwa["id"],
        "questions": pwa["questions"],
        "solutions": pwa["solutions"],
        "final_answers": pwa["final_answers"],
        # "improved_solutions": pwa["improved_solutions"],
    }

class Message(TypedDict):
    role: Literal["user", "assistant", "system"]
    content: str
class MessageList(list[Message]):
    def replace(self, replacements: dict[str, Any]) -> "MessageList":
        """
        Replace placeholders in the prompt with actual values.
        """
        new_prompt = MessageList()
        for i in range(len(self)):
            new_prompt.append(self[i].copy())
            for key, value in replacements.items():
                new_prompt[i]["content"] = new_prompt[i]["content"].replace(f"{{{key}}}", str(value))
        return new_prompt


@dataclass
class Instruction:
    """LLM instruction package with validation and formatting"""
    text: str
    validator: Callable[[str], bool]
    formatter: Callable[[str], Any]
    post_processor: Optional[Union[Callable[[Any], Any], Callable[[Any], Awaitable[Any]]]] = None


def load_prompt(file_path: str) -> MessageList:
    """
    Load a prompt file and return a list of dictionaries.
    """
    if not file_path.endswith('.json'):
        raise ValueError("File path must end with .json")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} does not exist")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = MessageList(json.load(f))
    return data


def load_jsonl(file_path: str) -> ProblemSetWithAnswer:
    """
    Load a JSONL file and return a list of dictionaries.
    """
    if not file_path.endswith('.jsonl'):
        raise ValueError("File path must end with .jsonl")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} does not exist")
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def sha256_hash(text: str | None) -> str:
    """
    Generate a SHA-256 hash of the given text.
    """
    return hashlib.sha256((text or "None").encode('utf-8')).hexdigest()

def bold_stdout(text: str) -> str:
    """
    Bold the given text for standard output.
    """
    return f"\033[1m{text}\033[0m"


def load_ref_sol(path: str, task_id: str) -> str:
    try:
        with open(path, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if data.get("id") == task_id or data.get("id") == task_id.replace("_", "/"):
                        solution = data.get("llm_solution_v1", "")
                        if solution:
                            return solution
                except json.JSONDecodeError:
                    continue
        raise ValueError(f"No matching entry found for task_id: {task_id}")
    except Exception as e:
        raise ValueError(f"Error loading fixed plan: {str(e)}")

def validate_plan_steps(plans: list[dict[str, Any]]):
    if not plans:
        return
    
    steps = sorted(plan["step_id"] for plan in plans)
    expected_steps = list(range(1, len(steps) + 1))
    
    if steps != expected_steps:
        error_msg = (
            f"Invalid step sequence: expected {expected_steps}, got {steps}\n"
            f"Full plan: {json.dumps(plans, indent=2, ensure_ascii=False)}"
        )
        raise ValueError(error_msg)
