import os
import json
import asyncio
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from src import ROOT_DIR, RESULT_ROOT_DIR, load_prompt
from src.chat import direct_chat

# Load ToT-specific prompts
TOT_PROMPT_DIR = os.path.join(ROOT_DIR, "src/prompt/tot")
GENERATOR_PROMPT = load_prompt(os.path.join(TOT_PROMPT_DIR, "generator_prompt.json"))
EVALUATOR_PROMPT = load_prompt(os.path.join(TOT_PROMPT_DIR, "evaluator_prompt.json"))
FINAL_ANSWER_PROMPT = load_prompt(os.path.join(TOT_PROMPT_DIR, "final_answer_prompt.json"))

# Default models
MODEL_GENERATOR_DEFAULT = "deepseek-v3-250324"
MODEL_EVALUATOR_DEFAULT = "deepseek-v3-250324"

@dataclass
class ThoughtNode:
    """Represents a node in the Tree-of-Thought"""
    content: str  # The thought/reasoning step content
    score: float = 0.0  # Evaluation score (0-10)
    depth: int = 0  # Depth in the tree
    parent: Optional['ThoughtNode'] = None
    children: List['ThoughtNode'] = field(default_factory=list)
    is_complete: bool = False  # Whether this represents a complete solution
    
    def get_path_to_root(self) -> List[str]:
        """Get the reasoning path from root to this node"""
        path = []
        current = self
        while current is not None:
            path.append(current.content)
            current = current.parent
        return list(reversed(path))
    
    def get_full_reasoning(self) -> str:
        """Get the complete reasoning chain as a single string"""
        path = self.get_path_to_root()
        return "\n\n".join(f"Step {i + 1}: {step}" for i, step in enumerate(path))


async def generate_thoughts(
    problem_statement: str,
    current_reasoning: str,
    num_thoughts: int = 3,
    model: str = MODEL_GENERATOR_DEFAULT,
    temperature: float = 0.7,
    task_id: str = "unknown"
) -> List[str]:
    """
    Generate multiple diverse thoughts/reasoning steps from current state
    
    Args:
        problem_statement: The original physics problem
        current_reasoning: Current reasoning chain so far
        num_thoughts: Number of different thoughts to generate
        model: Model to use for generation
        temperature: Temperature for generation
        task_id: Task identifier for logging
        
    Returns:
        List of generated thought strings
    """
    messages = GENERATOR_PROMPT.replace({
        "problem_statement": problem_statement,
        "current_reasoning": current_reasoning,
        "num_thoughts": str(num_thoughts)
    })
    
    api_record_path = os.path.join(
        RESULT_ROOT_DIR, ".api.origin",
        f"{task_id}_tot_generator_{model}.json"
    )
    
    result = await direct_chat(
        model=model,
        messages=messages,
        record_full_api_path=api_record_path,
        temperature=temperature
    )
    
    # Parse the generated thoughts
    # Expected format: "THOUGHT 1: ...\nTHOUGHT 2: ...\nTHOUGHT 3: ..."
    thoughts = []
    lines = result.strip().split('\n')
    current_thought = ""
    
    for line in lines:
        if line.startswith("THOUGHT ") and ":" in line:
            if current_thought.strip():
                thoughts.append(current_thought.strip())
            current_thought = line.split(":", 1)[1].strip()
        else:
            if current_thought:
                current_thought += "\n" + line
    
    if current_thought.strip():
        thoughts.append(current_thought.strip())
    
    # Ensure we have the requested number of thoughts
    if len(thoughts) < num_thoughts:
        # If parsing failed, split by common delimiters
        fallback_thoughts = result.split('\n\n')
        thoughts = [t.strip() for t in fallback_thoughts if t.strip()][:num_thoughts]
    
    return thoughts[:num_thoughts]


async def evaluate_thought(
    problem_statement: str,
    full_reasoning: str,
    thought: str,
    model: str = MODEL_EVALUATOR_DEFAULT,
    temperature: float = 0.0,
    task_id: str = "unknown"
) -> Tuple[float, bool]:
    """
    Evaluate a single thought for quality and completeness
    
    Args:
        problem_statement: The original physics problem
        full_reasoning: Complete reasoning chain including this thought
        thought: The specific thought to evaluate
        model: Model to use for evaluation
        temperature: Temperature for evaluation
        task_id: Task identifier for logging
        
    Returns:
        Tuple of (score, is_complete) where score is 0-10 and is_complete indicates if this is a final solution
    """
    messages = EVALUATOR_PROMPT.replace({
        "problem_statement": problem_statement,
        "full_reasoning": full_reasoning,
        "current_thought": thought
    })
    
    api_record_path = os.path.join(
        RESULT_ROOT_DIR, ".api.origin",
        f"{task_id}_tot_evaluator_{model}.json"
    )
    
    result = await direct_chat(
        model=model,
        messages=messages,
        record_full_api_path=api_record_path,
        temperature=temperature
    )
    
    # Parse the evaluation result
    # Expected format: "SCORE: X\nCOMPLETE: Yes/No\nREASONING: ..."
    score = 5.0  # Default score
    is_complete = False
    
    lines = result.strip().split('\n')
    for line in lines:
        if line.startswith("SCORE:"):
            try:
                score_str = line.split(":", 1)[1].strip()
                score = float(score_str)
                score = max(0.0, min(10.0, score))  # Clamp to 0-10
            except (ValueError, IndexError):
                pass
        elif line.startswith("COMPLETE:"):
            complete_str = line.split(":", 1)[1].strip().lower()
            is_complete = complete_str in ["yes", "true", "1"]
    
    return score, is_complete


async def generate_final_answer(
    problem_statement: str,
    reasoning_path: List[str],
    model: str = MODEL_GENERATOR_DEFAULT,
    temperature: float = 0.0,
    task_id: str = "unknown"
) -> str:
    """
    Generate the final formatted answer from the best reasoning path
    
    Args:
        problem_statement: The original physics problem
        reasoning_path: List of reasoning steps from root to leaf
        model: Model to use for final answer generation
        temperature: Temperature for generation
        task_id: Task identifier for logging
        
    Returns:
        Final formatted answer string
    """
    reasoning_chain = "\n\n".join(f"Step {i + 1}: {step}" for i, step in enumerate(reasoning_path))
    
    messages = FINAL_ANSWER_PROMPT.replace({
        "problem_statement": problem_statement,
        "reasoning_chain": reasoning_chain
    })
    
    api_record_path = os.path.join(
        RESULT_ROOT_DIR, ".api.origin",
        f"{task_id}_tot_final_{model}.json"
    )
    
    result = await direct_chat(
        model=model,
        messages=messages,
        record_full_api_path=api_record_path,
        temperature=temperature
    )
    
    return result


async def tot_solver(
    task_id: str,
    problem_statement: str,
    generator_model: str = MODEL_GENERATOR_DEFAULT,
    evaluator_model: str = MODEL_EVALUATOR_DEFAULT,
    generator_temperature: float = 0.0,
    evaluator_temperature: float = 0.0,
    max_depth: int = 5,
    max_nodes: int = 20,
    num_thoughts_per_node: int = 3,
    prune_threshold: float = 3.0,
    output_aux_path: Optional[str] = None
) -> str:
    """
    Tree-of-Thought solver for physics problems
    
    Args:
        task_id: Task identifier
        problem_statement: The physics problem to solve
        generator_model: Model for generating thoughts
        evaluator_model: Model for evaluating thoughts
        generator_temperature: Temperature for thought generation
        evaluator_temperature: Temperature for evaluation
        max_depth: Maximum depth of the search tree
        max_nodes: Maximum total nodes to explore
        num_thoughts_per_node: Number of thoughts to generate per node
        prune_threshold: Score threshold below which nodes are pruned
        output_aux_path: Path to save auxiliary files
        
    Returns:
        Final solution string
    """
    # Initialize the root node
    root = ThoughtNode(
        content="Let me start by analyzing this physics problem step by step.",
        depth=0
    )
    
    # Keep track of all nodes and the best complete solution
    all_nodes = [root]
    best_complete_node = None
    nodes_explored = 0
    
    # BFS queue for exploration
    exploration_queue = [root]
    
    # Create output directory for auxiliary files
    if output_aux_path:
        os.makedirs(output_aux_path, exist_ok=True)
    
    while exploration_queue and nodes_explored < max_nodes:
        current_node = exploration_queue.pop(0)
        
        # Skip if we've reached max depth
        if current_node.depth >= max_depth:
            continue
            
        # Generate thoughts from current node
        current_reasoning = current_node.get_full_reasoning()
        
        try:
            thoughts = await generate_thoughts(
                problem_statement=problem_statement,
                current_reasoning=current_reasoning,
                num_thoughts=num_thoughts_per_node,
                model=generator_model,
                temperature=generator_temperature,
                task_id=task_id
            )
        except Exception as e:
            print(f"Error generating thoughts for {task_id}: {e}")
            continue
        
        # Evaluate each thought
        child_nodes = []
        for thought in thoughts:
            if not thought.strip():
                continue
                
            # Create child node
            child_node = ThoughtNode(
                content=thought,
                depth=current_node.depth + 1,
                parent=current_node
            )
            
            # Get full reasoning for evaluation
            full_reasoning = child_node.get_full_reasoning()
            
            try:
                score, is_complete = await evaluate_thought(
                    problem_statement=problem_statement,
                    full_reasoning=full_reasoning,
                    thought=thought,
                    model=evaluator_model,
                    temperature=evaluator_temperature,
                    task_id=task_id
                )
                
                child_node.score = score
                child_node.is_complete = is_complete
                
                # Track the best complete solution
                if is_complete and (best_complete_node is None or score > best_complete_node.score):
                    best_complete_node = child_node
                
                # Only keep nodes above the pruning threshold
                if score >= prune_threshold:
                    child_nodes.append(child_node)
                    all_nodes.append(child_node)
                    
            except Exception as e:
                print(f"Error evaluating thought for {task_id}: {e}")
                continue
        
        # Add child nodes to parent and exploration queue
        current_node.children = child_nodes
        
        # Sort children by score (best first) and add to exploration queue
        child_nodes.sort(key=lambda x: x.score, reverse=True)
        exploration_queue.extend(child_nodes)
        
        # Sort exploration queue by score to prioritize best nodes
        exploration_queue.sort(key=lambda x: x.score, reverse=True)
        
        nodes_explored += len(child_nodes)
        
        # Save intermediate results
        if output_aux_path:
            tree_state = {
                "nodes_explored": nodes_explored,
                "current_depth": current_node.depth,
                "queue_size": len(exploration_queue),
                "best_score": best_complete_node.score if best_complete_node else 0,
                "tree_structure": _serialize_tree(root)
            }
            
            with open(os.path.join(output_aux_path, f"{task_id}_tree_state.json"), "w") as f:
                json.dump(tree_state, f, indent=2, ensure_ascii=False)
    
    # Generate final answer
    if best_complete_node:
        # Use the best complete solution
        final_node = best_complete_node
        # print(f"Found complete solution with score {final_node.score}")
    else:
        # Use the highest-scoring node overall
        all_nodes.sort(key=lambda x: x.score, reverse=True)
        final_node = all_nodes[0]
        # print(f"No complete solution found, using best partial solution with score {final_node.score}")
    
    # Generate the final formatted answer
    reasoning_path = final_node.get_path_to_root()
    
    try:
        final_answer = await generate_final_answer(
            problem_statement=problem_statement,
            reasoning_path=reasoning_path,
            model=generator_model,
            temperature=evaluator_temperature,  # Use lower temperature for final answer
            task_id=task_id
        )
    except Exception as e:
        print(f"Error generating final answer for {task_id}: {e}")
        # Fallback to direct reasoning chain
        final_answer = final_node.get_full_reasoning()
    
    # Save final results
    if output_aux_path:
        # Save the reasoning tree
        with open(os.path.join(output_aux_path, f"{task_id}_reasoning_tree.json"), "w") as f:
            json.dump(_serialize_tree(root), f, indent=2, ensure_ascii=False)
        
        # Save the final reasoning path
        with open(os.path.join(output_aux_path, f"{task_id}_final_path.md"), "w") as f:
            f.write("# Tree-of-Thought Reasoning Path\n\n")
            for i, step in enumerate(reasoning_path):
                f.write(f"## Step {i + 1}\n{step}\n\n")
        
        # Save the final answer
        with open(os.path.join(output_aux_path, f"{task_id}_final_answer.md"), "w") as f:
            f.write(final_answer)
    
    # Also save to standard result directory
    os.makedirs(os.path.join(RESULT_ROOT_DIR, task_id), exist_ok=True)
    with open(os.path.join(RESULT_ROOT_DIR, task_id, "tot_solution.md"), "w") as f:
        f.write(final_answer)
    
    return final_answer


def _serialize_tree(node: ThoughtNode, max_depth: int = 10) -> Dict:
    """Serialize a tree node for JSON output"""
    if max_depth <= 0:
        return {"content": "...", "truncated": True}
    
    return {
        "content": node.content[:200] + "..." if len(node.content) > 200 else node.content,
        "score": node.score,
        "depth": node.depth,
        "is_complete": node.is_complete,
        "children": [_serialize_tree(child, max_depth - 1) for child in node.children]
    }
