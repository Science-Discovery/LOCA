import os
import json
import asyncio
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict, deque
import heapq
from src import ROOT_DIR, RESULT_ROOT_DIR, load_prompt
from src.chat import direct_chat

# Load GoT-specific prompts
GOT_PROMPT_DIR = os.path.join(ROOT_DIR, "src/prompt/got")
GENERATOR_PROMPT = load_prompt(os.path.join(GOT_PROMPT_DIR, "generator_prompt.json"))
EVALUATOR_PROMPT = load_prompt(os.path.join(GOT_PROMPT_DIR, "evaluator_prompt.json"))
AGGREGATOR_PROMPT = load_prompt(os.path.join(GOT_PROMPT_DIR, "aggregator_prompt.json"))
FINAL_ANSWER_PROMPT = load_prompt(os.path.join(GOT_PROMPT_DIR, "final_answer_prompt.json"))

# Default models
MODEL_GENERATOR_DEFAULT = "deepseek-v3-250324"
MODEL_EVALUATOR_DEFAULT = "deepseek-v3-250324"
MODEL_AGGREGATOR_DEFAULT = "deepseek-v3-250324"

@dataclass
class ThoughtNode:
    """Represents a node in the Graph-of-Thought"""
    content: str  # The thought/reasoning step content
    score: float = 0.0  # Evaluation score (0-10)
    depth: int = 0  # Depth in the graph
    parents: List['ThoughtNode'] = field(default_factory=list)  # Multiple parents for graph structure
    children: List['ThoughtNode'] = field(default_factory=list)
    is_complete: bool = False  # Whether this represents a complete solution
    node_id: str = ""  # Unique identifier for this node
    is_aggregated: bool = False  # Whether this node was created by aggregation
    
    def __post_init__(self):
        if not self.node_id:
            self.node_id = f"node_{id(self)}"
    
    def __hash__(self):
        """Make ThoughtNode hashable using node_id"""
        return hash(self.node_id)
    
    def __eq__(self, other):
        """Compare ThoughtNodes by node_id"""
        if not isinstance(other, ThoughtNode):
            return False
        return self.node_id == other.node_id
    
    def get_best_path_to_root(self) -> List[str]:
        """Get the best reasoning path from root to this node using BFS"""
        if not self.parents:
            return [self.content]
        
        # Find the highest-scoring parent path
        best_parent = max(self.parents, key=lambda p: p.score)
        parent_path = best_parent.get_best_path_to_root()
        return parent_path + [self.content]
    
    def get_full_reasoning(self) -> str:
        """Get the complete reasoning chain as a single string"""
        path = self.get_best_path_to_root()
        return "\n\n".join(f"Step {i + 1}: {step}" for i, step in enumerate(path))
    
    def get_all_ancestors(self) -> Set['ThoughtNode']:
        """Get all ancestor nodes (for cycle detection)"""
        ancestors = set()
        queue = deque(self.parents)
        while queue:
            node = queue.popleft()
            if node not in ancestors:
                ancestors.add(node)
                queue.extend(node.parents)
        return ancestors


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
        current_reasoning: Current reasoning context (may include multiple paths)
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
        f"{task_id}_got_generator_{model}.json"
    )
    
    result = await direct_chat(
        model=model,
        messages=messages,
        record_full_api_path=api_record_path,
        temperature=temperature
    )
    
    # Parse the generated thoughts
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
        full_reasoning: Complete reasoning context including this thought
        thought: The specific thought to evaluate
        model: Model to use for evaluation
        temperature: Temperature for evaluation
        task_id: Task identifier for logging
        
    Returns:
        Tuple of (score, is_complete)
    """
    messages = EVALUATOR_PROMPT.replace({
        "problem_statement": problem_statement,
        "full_reasoning": full_reasoning,
        "current_thought": thought
    })
    
    api_record_path = os.path.join(
        RESULT_ROOT_DIR, ".api.origin",
        f"{task_id}_got_evaluator_{model}.json"
    )
    
    result = await direct_chat(
        model=model,
        messages=messages,
        record_full_api_path=api_record_path,
        temperature=temperature
    )
    
    # Parse the evaluation result
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


async def aggregate_thoughts(
    problem_statement: str,
    reasoning_paths: List[str],
    model: str = MODEL_AGGREGATOR_DEFAULT,
    temperature: float = 0.3,
    task_id: str = "unknown"
) -> str:
    """
    Aggregate multiple reasoning paths into a single synthesized thought
    
    Args:
        problem_statement: The original physics problem
        reasoning_paths: List of reasoning paths to aggregate
        model: Model to use for aggregation
        temperature: Temperature for aggregation
        task_id: Task identifier for logging
        
    Returns:
        Aggregated thought string
    """
    if len(reasoning_paths) <= 1:
        return reasoning_paths[0] if reasoning_paths else ""
    
    # Format reasoning paths for the prompt
    formatted_paths = "\n\n".join(
        f"**Path {i + 1}:**\n{path}" for i, path in enumerate(reasoning_paths)
    )
    
    messages = AGGREGATOR_PROMPT.replace({
        "problem_statement": problem_statement,
        "reasoning_paths": formatted_paths
    })
    
    api_record_path = os.path.join(
        RESULT_ROOT_DIR, ".api.origin",
        f"{task_id}_got_aggregator_{model}.json"
    )
    
    result = await direct_chat(
        model=model,
        messages=messages,
        record_full_api_path=api_record_path,
        temperature=temperature
    )
    
    # Extract the aggregated thought
    if "AGGREGATED THOUGHT:" in result:
        return result.split("AGGREGATED THOUGHT:", 1)[1].strip()
    else:
        # Fallback: return the full result
        return result.strip()


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
        f"{task_id}_got_final_{model}.json"
    )
    
    result = await direct_chat(
        model=model,
        messages=messages,
        record_full_api_path=api_record_path,
        temperature=temperature
    )
    
    return result


def _save_graph_state(
    all_nodes: List[ThoughtNode],
    exploration_queue: List[ThoughtNode],
    best_complete_node: Optional[ThoughtNode],
    nodes_explored: int,
    output_aux_path: str,
    task_id: str
):
    """Save the current state of the graph for resumption"""
    state = {
        "nodes_explored": nodes_explored,
        "best_complete_score": best_complete_node.score if best_complete_node else 0,
        "queue_size": len(exploration_queue),
        "nodes": []
    }
    
    # Create node mapping for serialization
    node_to_id = {node: i for i, node in enumerate(all_nodes)}
    
    for i, node in enumerate(all_nodes):
        node_data = {
            "id": i,
            "content": node.content,
            "score": node.score,
            "depth": node.depth,
            "is_complete": node.is_complete,
            "is_aggregated": node.is_aggregated,
            "parent_ids": [node_to_id[parent] for parent in node.parents if parent in node_to_id],
            "child_ids": [node_to_id[child] for child in node.children if child in node_to_id]
        }
        state["nodes"].append(node_data)
    
    # Save queue state
    state["queue_node_ids"] = [node_to_id[node] for node in exploration_queue if node in node_to_id]
    state["best_complete_node_id"] = node_to_id[best_complete_node] if best_complete_node and best_complete_node in node_to_id else None
    
    with open(os.path.join(output_aux_path, f"{task_id}_graph_state.json"), "w") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)


def _load_graph_state(output_aux_path: str, task_id: str) -> Optional[Tuple[List[ThoughtNode], List[ThoughtNode], Optional[ThoughtNode], int]]:
    """Load the graph state for resumption"""
    state_file = os.path.join(output_aux_path, f"{task_id}_graph_state.json")
    if not os.path.exists(state_file):
        return None
    
    try:
        with open(state_file, "r") as f:
            state = json.load(f)
        
        # Reconstruct nodes
        all_nodes = []
        for node_data in state["nodes"]:
            node = ThoughtNode(
                content=node_data["content"],
                score=node_data["score"],
                depth=node_data["depth"],
                is_complete=node_data["is_complete"],
                is_aggregated=node_data.get("is_aggregated", False)
            )
            all_nodes.append(node)
        
        # Reconstruct relationships
        for i, node_data in enumerate(state["nodes"]):
            node = all_nodes[i]
            for parent_id in node_data["parent_ids"]:
                if 0 <= parent_id < len(all_nodes):
                    parent = all_nodes[parent_id]
                    node.parents.append(parent)
                    if node not in parent.children:
                        parent.children.append(node)
        
        # Reconstruct exploration queue
        exploration_queue = []
        for node_id in state.get("queue_node_ids", []):
            if 0 <= node_id < len(all_nodes):
                exploration_queue.append(all_nodes[node_id])
        
        # Reconstruct best complete node
        best_complete_node = None
        best_id = state.get("best_complete_node_id")
        if best_id is not None and 0 <= best_id < len(all_nodes):
            best_complete_node = all_nodes[best_id]
        
        nodes_explored = state.get("nodes_explored", 0)
        
        return all_nodes, exploration_queue, best_complete_node, nodes_explored
        
    except Exception as e:
        print(f"Error loading graph state for {task_id}: {e}")
        return None


async def got_solver(
    task_id: str,
    problem_statement: str,
    generator_model: str = MODEL_GENERATOR_DEFAULT,
    evaluator_model: str = MODEL_EVALUATOR_DEFAULT,
    aggregator_model: str = MODEL_AGGREGATOR_DEFAULT,
    generator_temperature: float = 0.0,
    evaluator_temperature: float = 0.0,
    aggregator_temperature: float = 0.0,
    max_depth: int = 4,
    max_nodes: int = 15,
    num_thoughts_per_node: int = 2,
    prune_threshold: float = 3.0,
    aggregation_threshold: float = 7.0,
    aggregation_interval: int = 3,
    output_aux_path: Optional[str] = None
) -> str:
    """
    Graph-of-Thought solver for physics problems
    
    Args:
        task_id: Task identifier
        problem_statement: The physics problem to solve
        generator_model: Model for generating thoughts
        evaluator_model: Model for evaluating thoughts
        aggregator_model: Model for aggregating thoughts
        generator_temperature: Temperature for thought generation
        evaluator_temperature: Temperature for evaluation
        aggregator_temperature: Temperature for aggregation
        max_depth: Maximum depth of the search graph
        max_nodes: Maximum total nodes to explore
        num_thoughts_per_node: Number of thoughts to generate per node
        prune_threshold: Score threshold below which nodes are pruned
        aggregation_threshold: Score threshold above which nodes are considered for aggregation
        aggregation_interval: How often to perform aggregation (every N nodes)
        output_aux_path: Path to save auxiliary files
        
    Returns:
        Final solution string
    """
    # Create output directory for auxiliary files
    if output_aux_path:
        os.makedirs(output_aux_path, exist_ok=True)
        
        # Check for completion flag
        completion_flag = os.path.join(output_aux_path, f"{task_id}_completed.flag")
        if os.path.exists(completion_flag):
            # Load and return previous result
            final_answer_file = os.path.join(output_aux_path, f"{task_id}_final_answer.md")
            if os.path.exists(final_answer_file):
                with open(final_answer_file, "r") as f:
                    return f.read()
        
        # Try to load previous state
        loaded_state = _load_graph_state(output_aux_path, task_id)
        if loaded_state:
            all_nodes, exploration_queue, best_complete_node, nodes_explored = loaded_state
            print(f"Resuming from previous state: {nodes_explored} nodes explored")
        else:
            # Initialize fresh state
            root = ThoughtNode(
                content="Let me start by analyzing this physics problem step by step.",
                depth=0
            )
            all_nodes = [root]
            exploration_queue = [root]
            best_complete_node = None
            nodes_explored = 0
    else:
        # Initialize fresh state
        root = ThoughtNode(
            content="Let me start by analyzing this physics problem step by step.",
            depth=0
        )
        all_nodes = [root]
        exploration_queue = [root]
        best_complete_node = None
        nodes_explored = 0
    
    # Priority queue for exploration (max-heap using negative scores)
    priority_queue = [(-node.score, id(node), node) for node in exploration_queue]
    heapq.heapify(priority_queue)
    
    while priority_queue and nodes_explored < max_nodes:
        # Get the highest-scoring node
        _, _, current_node = heapq.heappop(priority_queue)
        
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
        
        # Evaluate each thought and create child nodes
        child_nodes = []
        for thought in thoughts:
            if not thought.strip():
                continue
            
            # Create child node
            child_node = ThoughtNode(
                content=thought,
                depth=current_node.depth + 1,
                parents=[current_node]
            )
            
            # Check for cycles
            if child_node in current_node.get_all_ancestors():
                continue
            
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
        
        # Add child nodes to parent and priority queue
        current_node.children.extend(child_nodes)
        
        # Add children to priority queue
        for child in child_nodes:
            heapq.heappush(priority_queue, (-child.score, id(child), child))
        
        nodes_explored += len(child_nodes)
        
        # Perform aggregation periodically
        if nodes_explored % aggregation_interval == 0 and len(all_nodes) > 3:
            await _perform_aggregation(
                problem_statement, all_nodes, priority_queue, aggregation_threshold,
                aggregator_model, aggregator_temperature, task_id
            )
        
        # Save state periodically
        if output_aux_path and nodes_explored % 5 == 0:
            current_queue = [node for _, _, node in priority_queue]
            _save_graph_state(all_nodes, current_queue, best_complete_node, nodes_explored, output_aux_path, task_id)
    
    # Generate final answer
    if best_complete_node:
        final_node = best_complete_node
    else:
        # Use the highest-scoring node overall
        all_nodes.sort(key=lambda x: x.score, reverse=True)
        final_node = all_nodes[0]
    
    # Generate the final formatted answer
    reasoning_path = final_node.get_best_path_to_root()
    
    try:
        final_answer = await generate_final_answer(
            problem_statement=problem_statement,
            reasoning_path=reasoning_path,
            model=generator_model,
            temperature=evaluator_temperature,
            task_id=task_id
        )
    except Exception as e:
        print(f"Error generating final answer for {task_id}: {e}")
        final_answer = final_node.get_full_reasoning()
    
    # Save final results
    if output_aux_path:
        # Save the reasoning graph
        with open(os.path.join(output_aux_path, f"{task_id}_reasoning_graph.json"), "w") as f:
            json.dump(_serialize_graph(all_nodes), f, indent=2, ensure_ascii=False)
        
        # Save the final reasoning path
        with open(os.path.join(output_aux_path, f"{task_id}_final_path.md"), "w") as f:
            f.write("# Graph-of-Thought Reasoning Path\n\n")
            for i, step in enumerate(reasoning_path):
                f.write(f"## Step {i + 1}\n{step}\n\n")
        
        # Save the final answer
        with open(os.path.join(output_aux_path, f"{task_id}_final_answer.md"), "w") as f:
            f.write(final_answer)
        
        # Create completion flag
        with open(os.path.join(output_aux_path, f"{task_id}_completed.flag"), "w") as f:
            f.write("")
    
    # Also save to standard result directory
    os.makedirs(os.path.join(RESULT_ROOT_DIR, task_id), exist_ok=True)
    with open(os.path.join(RESULT_ROOT_DIR, task_id, "got_solution.md"), "w") as f:
        f.write(final_answer)
    
    return final_answer


async def _perform_aggregation(
    problem_statement: str,
    all_nodes: List[ThoughtNode],
    priority_queue: List,
    aggregation_threshold: float,
    aggregator_model: str,
    aggregator_temperature: float,
    task_id: str
):
    """Perform thought aggregation on high-scoring nodes"""
    # Find high-scoring leaf nodes for aggregation
    leaf_nodes = [node for node in all_nodes 
                  if not node.children and node.score >= aggregation_threshold and not node.is_aggregated]
    
    if len(leaf_nodes) >= 2:
        # Group nodes by depth for aggregation
        depth_groups = defaultdict(list)
        for node in leaf_nodes:
            depth_groups[node.depth].append(node)
        
        # Aggregate nodes at each depth level
        for depth, nodes in depth_groups.items():
            if len(nodes) >= 2:
                # Take top scoring nodes for aggregation
                nodes.sort(key=lambda x: x.score, reverse=True)
                top_nodes = nodes[:min(3, len(nodes))]  # Aggregate top 3 nodes
                
                # Get reasoning paths for aggregation
                reasoning_paths = [node.get_full_reasoning() for node in top_nodes]
                
                try:
                    aggregated_content = await aggregate_thoughts(
                        problem_statement=problem_statement,
                        reasoning_paths=reasoning_paths,
                        model=aggregator_model,
                        temperature=aggregator_temperature,
                        task_id=task_id
                    )
                    
                    # Create aggregated node
                    aggregated_node = ThoughtNode(
                        content=aggregated_content,
                        depth=depth + 1,
                        parents=top_nodes,
                        is_aggregated=True
                    )
                    
                    # Add to parents' children
                    for parent in top_nodes:
                        parent.children.append(aggregated_node)
                    
                    # Evaluate the aggregated node
                    full_reasoning = aggregated_node.get_full_reasoning()
                    score, is_complete = await evaluate_thought(
                        problem_statement=problem_statement,
                        full_reasoning=full_reasoning,
                        thought=aggregated_content,
                        model=aggregator_model,
                        temperature=0.0,
                        task_id=task_id
                    )
                    
                    aggregated_node.score = score
                    aggregated_node.is_complete = is_complete
                    
                    # Add to graph
                    all_nodes.append(aggregated_node)
                    heapq.heappush(priority_queue, (-score, id(aggregated_node), aggregated_node))
                    
                except Exception as e:
                    print(f"Error in aggregation for {task_id}: {e}")


def _serialize_graph(nodes: List[ThoughtNode], max_depth: int = 10) -> Dict:
    """Serialize the graph for JSON output"""
    if max_depth <= 0:
        return {"nodes": [], "truncated": True}
    
    node_to_id = {node: i for i, node in enumerate(nodes)}
    
    serialized_nodes = []
    for i, node in enumerate(nodes):
        node_data = {
            "id": i,
            "content": node.content[:200] + "..." if len(node.content) > 200 else node.content,
            "score": node.score,
            "depth": node.depth,
            "is_complete": node.is_complete,
            "is_aggregated": node.is_aggregated,
            "parent_ids": [node_to_id[parent] for parent in node.parents if parent in node_to_id],
            "child_ids": [node_to_id[child] for child in node.children if child in node_to_id]
        }
        serialized_nodes.append(node_data)
    
    return {
        "nodes": serialized_nodes,
        "total_nodes": len(nodes),
        "max_depth": max(node.depth for node in nodes) if nodes else 0
    }
