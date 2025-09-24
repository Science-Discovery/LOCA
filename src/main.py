import os
import json
import asyncio
from src.utils import load_jsonl, Problem, extract_problem
from src.single_llm import single_llm_solver
from src.tot_solver import tot_solver
from src.got_solver import got_solver
from src.loca.solver import loca_solver
from src.mad.physics_debate import mad_solver
from src.vanilla_review.solver import vanilla_review_solver


def load_initial_answers(result0_path: str) -> dict[str, str]:
    """
    Load initial answers from result0.jsonl file
    
    Args:
        result0_path: Path to the result0.jsonl file
        
    Returns:
        Dictionary mapping problem IDs to their initial answers
    """
    if not os.path.exists(result0_path):
        raise FileNotFoundError(f"Initial answers file not found: {result0_path}")
    
    initial_answers = {}
    dataset = load_jsonl(result0_path)
    
    for item in dataset:
        problem_id = item.get("id")
        # Try to get the answer from different possible fields
        answer = item.get("llm_solution") or item.get("llm_solution_v1") or item.get("answer")
        
        if problem_id and answer:
            initial_answers[problem_id] = answer
    
    print(f"Loaded {len(initial_answers)} initial answers from {result0_path}")
    return initial_answers


SCHEMES_NEEDING_AUX_DIR = [
    'single-llm',
    'tot',
    'got',
    'loca',
    'mad',
    'vanilla_review',
]

async def solve_problem(problem: Problem, scheme: str,
                        tools: list[str] = None,
                        aux_dir: str = None,
                        extra_params: dict = None) -> dict:
    problem_id = problem["id"].replace("/", "_")
    problem_statement = problem["questions"]
    generator_model = extra_params.get('generator_model') if extra_params else None
    planner_model = extra_params.get('planner_model') if extra_params else None
    reviewer_model = extra_params.get('reviewer_model') if extra_params else None
    
    # Parse temperature parameters with default value 0.0
    generator_temperature = float(extra_params.get('generator_temperature', 0.0)) if extra_params else 0.0
    planner_temperature = float(extra_params.get('planner_temperature', 0.0)) if extra_params else 0.0
    reviewer_temperature = float(extra_params.get('reviewer_temperature', 0.0)) if extra_params else 0.0
    
    # Parse reviewer options from extra_params
    refine_model = None
    voting_n = 1  # Default voting_n to 1
    
    # Parse single_llm options from extra_params
    zero_shot_cot = False  # Default to False
    cot = False  # Default to False
    
    # Parse ToT options from extra_params
    evaluator_model = None
    evaluator_temperature = 0.0
    max_depth = 5
    max_nodes = 20
    num_thoughts_per_node = 3
    prune_threshold = 3.0
    
    # Parse GoT options from extra_params
    aggregator_model = None
    aggregator_temperature = 0.0
    aggregation_threshold = 7.0
    aggregation_interval = 5
    
    # Parse LOCA options from extra_params
    max_error_times = 5  # Default value
    target_times = 3     # Default value
    ablation = False     # Default value
    keep_aug = False  # Default value
    keep_specific_verification = False  # Default value
    
    # Parse MAD options from extra_params
    debate_model = None  # Default to use generator_model
    max_rounds = 3       # Default value
    config_file = "src/mad/config.json"  # Default config file
    
    if extra_params:
        
        # Handle string to boolean conversion for reviewer options
        if 'refine_model' in extra_params:
            refine_model = extra_params['refine_model']
        if 'voting_n' in extra_params:
            voting_n = int(extra_params['voting_n'])
            if scheme not in ["single-llm"]:
                raise ValueError("voting_n is only applicable for and single-llm schemes")
        
        # Handle single_llm options
        if 'zero_shot_cot' in extra_params:
            zero_shot_cot = str(extra_params['zero_shot_cot']).lower() in ('true', '1', 'yes')
        if 'cot' in extra_params:
            cot = str(extra_params['cot']).lower() in ('true', '1', 'yes')
        
        # Handle ToT options
        if 'evaluator_model' in extra_params:
            evaluator_model = extra_params['evaluator_model']
        if 'evaluator_temperature' in extra_params:
            evaluator_temperature = float(extra_params['evaluator_temperature'])
        if 'max_depth' in extra_params:
            max_depth = int(extra_params['max_depth'])
            if scheme not in ["tot", "got"]:
                raise ValueError("max_depth is only applicable for tot and got schemes")
        if 'max_nodes' in extra_params:
            max_nodes = int(extra_params['max_nodes'])
            if scheme not in ["tot", "got"]:
                raise ValueError("max_nodes is only applicable for tot and got schemes")
        if 'num_thoughts_per_node' in extra_params:
            num_thoughts_per_node = int(extra_params['num_thoughts_per_node'])
            if scheme not in ["tot", "got"]:
                raise ValueError("num_thoughts_per_node is only applicable for tot and got schemes")
        if 'prune_threshold' in extra_params:
            prune_threshold = float(extra_params['prune_threshold'])
            if scheme not in ["tot", "got"]:
                raise ValueError("prune_threshold is only applicable for tot and got schemes")
        
        # Handle GoT options
        if 'aggregator_model' in extra_params:
            aggregator_model = extra_params['aggregator_model']
            if scheme != "got":
                raise ValueError("aggregator_model is only applicable for got scheme")
        if 'aggregator_temperature' in extra_params:
            aggregator_temperature = float(extra_params['aggregator_temperature'])
            if scheme != "got":
                raise ValueError("aggregator_temperature is only applicable for got scheme")
        if 'aggregation_threshold' in extra_params:
            aggregation_threshold = float(extra_params['aggregation_threshold'])
            if scheme != "got":
                raise ValueError("aggregation_threshold is only applicable for got scheme")
        if 'aggregation_interval' in extra_params:
            aggregation_interval = int(extra_params['aggregation_interval'])
            if scheme != "got":
                raise ValueError("aggregation_interval is only applicable for got scheme")
        
        # Handle LOCA options
        if 'max_error_times' in extra_params:
            max_error_times = int(extra_params['max_error_times'])
            if scheme not in ["loca", "vanilla_review"]:
                raise ValueError("max_error_times is only applicable for loca and vanilla_review schemes")
        if 'target_times' in extra_params:
            target_times = int(extra_params['target_times'])
            if scheme not in ["loca", "vanilla_review"]:
                raise ValueError("target_times is only applicable for loca and vanilla_review schemes")
        if 'ablation' in extra_params:
            ablation = str(extra_params['ablation']).lower() in ('true', '1', 'yes')
            if scheme != "loca":
                raise ValueError("ablation is only applicable for loca scheme")
        if 'keep_aug' in extra_params:
            keep_aug = str(extra_params['keep_aug']).lower() in ('true', '1', 'yes')
            if scheme != "loca":
                raise ValueError("keep_aug is only applicable for loca scheme")
        if 'keep_specific_verification' in extra_params:
            keep_specific_verification = str(extra_params['keep_specific_verification']).lower() in ('true', '1', 'yes')
            if scheme != "loca":
                raise ValueError("keep_specific_verification is only applicable for loca scheme")
        
        # Handle MAD options
        if 'debate_model' in extra_params:
            debate_model = extra_params['debate_model']
            if scheme != "mad":
                raise ValueError("debate_model is only applicable for mad scheme")
        if 'max_rounds' in extra_params:
            max_rounds = int(extra_params['max_rounds'])
            if scheme != "mad":
                raise ValueError("max_rounds is only applicable for mad scheme")
        if 'config_file' in extra_params:
            config_file = extra_params['config_file']
            if scheme != "mad":
                raise ValueError("config_file is only applicable for mad scheme")

    if scheme == "single-llm":
        output_aux_path = os.path.join(aux_dir, problem_id + ".cache.txt") if aux_dir else None
        if output_aux_path is not None and os.path.exists(output_aux_path):
            with open(output_aux_path, "r") as f:
                solution = f.read()
            print(f"Warning: cache exists for problem {problem['id']}, skip it", flush=True)
        else:
            print(f"Solving problem {problem_id} with single-llm scheme", flush=True)
            single_llm_options = {
                "zero_shot_cot": zero_shot_cot,
                "cot": cot,
                "voting_n": voting_n,
            }
            
            solution = await single_llm_solver(problem_id, problem_statement,
                                               model=generator_model,
                                               temperature=generator_temperature,
                                               options=single_llm_options)
            if output_aux_path:
                # Check if the directory exists, if not, create it
                if not os.path.exists(os.path.dirname(output_aux_path)):
                    os.makedirs(os.path.dirname(output_aux_path), exist_ok=True)
                # Write the solution to the cache file
                with open(output_aux_path, "w") as f:
                    f.write(solution)
        return problem | {"improved_solutions": solution}
    elif scheme == "tot":
        # Tree-of-Thought solver
        output_aux_path = os.path.join(aux_dir, problem_id) if aux_dir else None
        if output_aux_path and not os.path.exists(output_aux_path):
            os.makedirs(output_aux_path, exist_ok=True)
        if output_aux_path is not None and os.path.exists(os.path.join(output_aux_path, "cache.txt")):
            with open(os.path.join(output_aux_path, "cache.txt"), "r") as f:
                solution = f.read()
            print(f"Warning: cache exists for problem {problem['id']}, skip it")
        else:
            # Use parsed ToT parameters
            final_evaluator_model = evaluator_model if evaluator_model else generator_model
            
            solution = await tot_solver(
                task_id=problem_id,
                problem_statement=problem_statement,
                generator_model=generator_model,
                evaluator_model=final_evaluator_model,
                generator_temperature=generator_temperature,
                evaluator_temperature=evaluator_temperature,
                max_depth=max_depth,
                max_nodes=max_nodes,
                num_thoughts_per_node=num_thoughts_per_node,
                prune_threshold=prune_threshold,
                output_aux_path=output_aux_path
            )
            if output_aux_path:
                with open(os.path.join(output_aux_path, "cache.txt"), "w") as f:
                    f.write(solution)
        return problem | {"improved_solutions": solution}
    elif scheme == "got":
        # Graph-of-Thought solver
        output_aux_path = os.path.join(aux_dir, problem_id) if aux_dir else None
        if output_aux_path and not os.path.exists(output_aux_path):
            os.makedirs(output_aux_path, exist_ok=True)
        
        # Check for completion flag first
        if output_aux_path is not None and os.path.exists(os.path.join(output_aux_path, f"{problem_id}_completed.flag")):
            # Load cached result
            final_answer_file = os.path.join(output_aux_path, f"{problem_id}_final_answer.md")
            if os.path.exists(final_answer_file):
                with open(final_answer_file, "r") as f:
                    solution = f.read()
                print(f"✅ Completed result exists for problem {problem['id']}, loading from cache", flush=True)
            else:
                # Fallback to cache.txt if final_answer.md doesn't exist
                cache_file = os.path.join(output_aux_path, "cache.txt")
                if os.path.exists(cache_file):
                    with open(cache_file, "r") as f:
                        solution = f.read()
                else:
                    # No cache found, need to solve
                    solution = None
        else:
            solution = None
        
        if solution is None:
            # Use parsed GoT parameters
            final_evaluator_model = evaluator_model if evaluator_model else generator_model
            final_aggregator_model = aggregator_model if aggregator_model else generator_model
            
            solution = await got_solver(
                task_id=problem_id,
                problem_statement=problem_statement,
                generator_model=generator_model,
                evaluator_model=final_evaluator_model,
                aggregator_model=final_aggregator_model,
                generator_temperature=generator_temperature,
                evaluator_temperature=evaluator_temperature,
                aggregator_temperature=aggregator_temperature,
                max_depth=max_depth,
                max_nodes=max_nodes,
                num_thoughts_per_node=num_thoughts_per_node,
                prune_threshold=prune_threshold,
                aggregation_threshold=aggregation_threshold,
                aggregation_interval=aggregation_interval,
                output_aux_path=output_aux_path
            )
            
            # Save to cache.txt for compatibility
            if output_aux_path:
                with open(os.path.join(output_aux_path, "cache.txt"), "w") as f:
                    f.write(solution)
        
        return problem | {"improved_solutions": solution}
    elif scheme == "loca":
        # LOCA solver - iterative review and refinement
        output_aux_path = os.path.join(aux_dir, problem_id) if aux_dir else None
        if output_aux_path and not os.path.exists(output_aux_path):
            os.makedirs(output_aux_path, exist_ok=True)
        
        # Get initial solution from problem data (assuming it's in the solutions field)
        initial_solution = problem.get("solutions", "")
        # initial_solution = problem.get("improved_solutions", "")
        if not initial_solution:
            raise ValueError(f"LOCA scheme requires an initial solution in the 'solutions' field for problem {problem['id']}")
        
        success, improved_solution = await loca_solver(
            task_id=problem_id,
            question_statement=problem_statement,
            solution=initial_solution,
            output_aux_path=output_aux_path,
            augmentation_model=refine_model or generator_model,
            review_model=reviewer_model or generator_model,
            max_error_times=max_error_times,
            target_times=target_times,
            temperature=generator_temperature,
            ablation=ablation,
            keep_aug=keep_aug,
            keep_specific_verification=keep_specific_verification,
        )
        
        if success:
            # problem["improved_solutions"] = improved_solution
            return problem | {"improved_solutions": improved_solution}
        else:
            return None
        # return problem
    elif scheme == "mad":
        # MAD solver - Multi-Agent Debate
        output_aux_path = os.path.join(aux_dir, problem_id) if aux_dir else None
        if output_aux_path and not os.path.exists(output_aux_path):
            os.makedirs(output_aux_path, exist_ok=True)
        
        # Get initial solution from problem data (assuming it's in the solutions field)
        initial_solution = problem.get("solutions", "")
        if not initial_solution:
            raise ValueError(f"MAD scheme requires an initial solution in the 'solutions' field for problem {problem['id']}")
        
        success, improved_solution = await mad_solver(
            task_id=problem_id,
            question_statement=problem_statement,
            solution=initial_solution,
            output_aux_path=output_aux_path,
            debate_model=debate_model,
            temperature=generator_temperature,
            max_rounds=max_rounds,
            config_file=config_file
        )
        
        if success:
            return problem | {"improved_solutions": improved_solution}
        else:
            return None
    elif scheme == "vanilla_review":
        # Vanilla Review solver - simple review with consistency checking
        output_aux_path = os.path.join(aux_dir, problem_id) if aux_dir else None
        if output_aux_path and not os.path.exists(output_aux_path):
            os.makedirs(output_aux_path, exist_ok=True)
        
        # Get initial solution from problem data (assuming it's in the solutions field)
        initial_solution = problem.get("solutions", "")
        if not initial_solution:
            raise ValueError(f"Vanilla Review scheme requires an initial solution in the 'solutions' field for problem {problem['id']}")
        
        success, result = await vanilla_review_solver(
            task_id=problem_id,
            question_statement=problem_statement,
            solution=initial_solution,
            output_aux_path=output_aux_path,
            review_model=reviewer_model or generator_model,
            max_error_times=max_error_times,
            target_times=target_times,
            temperature=reviewer_temperature
        )
        
        if success:
            return problem | {"vanilla_review_result": success}
        else:
            return problem | {"vanilla_review_result": success}
    else:
        raise ValueError(f"Unsupported scheme: {scheme}")

async def main(input_path: str, output_path: str, scheme: str = 'single-llm',
               tools: list[str] = [],
               extra_args: list[str] = None) -> None:
    print("\nArgs parsed successfully, starting processing...", flush=True)
    # Validate input file exists
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    # Create output directory if needed
    output_dir = os.path.dirname(output_path)
    if output_dir:  # Only create if path contains directory
        os.makedirs(output_dir, exist_ok=True)

    # Create auxiliary directory if needed
    aux_dir = os.path.join(os.path.dirname(output_path), ".aux") if scheme in SCHEMES_NEEDING_AUX_DIR else None
    if aux_dir:
        # Ensure the auxiliary directory exists for multi-round scheme
        if not os.path.exists(aux_dir):
            print(f"Creating auxiliary directory: {aux_dir}")
            os.makedirs(aux_dir, exist_ok=True)

    extra_params = {}
    if extra_args:
        i = 0
        while i < len(extra_args):
            if extra_args[i].startswith('--'):
                key = extra_args[i][2:]
                if i + 1 < len(extra_args) and not extra_args[i + 1].startswith('--'):
                    extra_params[key] = extra_args[i + 1]
                    i += 2
                    continue
                else:
                    extra_params[key] = True
            i += 1

    print(f"Processing input: {input_path}", flush=True)
    print(f"Using scheme: {scheme}", flush=True)
    print(f"Tools selected :{tools}", flush=True)
    print(f"Extra arguments:", flush=True)
    for key, value in extra_params.items():
        print(f"  {key}: {value}", flush=True)
    print(f"Output will be directed to {output_dir}, where results will be deposited in {output_path}\n", flush=True)
    dataset = load_jsonl(input_path)
    problem_set = [extract_problem(pwa) for pwa in dataset]
    task_list = [solve_problem(problem, scheme, tools, aux_dir, extra_params) for problem in problem_set]
    results = await asyncio.gather(*task_list)
    
    # Save results to specified output path
    with open(output_path, "w") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
    print(f"\nResults saved to {output_path}\n")
