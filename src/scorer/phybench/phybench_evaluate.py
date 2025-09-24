import json
import re
import os
import sys
import csv
import numpy as np
import matplotlib.pyplot as plt
import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from eed_tools.EED import EED

def normalize_final_answer(answer_str):
    boxed_match = re.search(r'\\boxed\{(.*?)\}', answer_str, re.DOTALL)
    if boxed_match:
        return boxed_match.group(1).strip()
    
    if answer_str.startswith('$$') and answer_str.endswith('$$'):
        return answer_str[2:-2].strip()
    
    if answer_str.startswith('\\[') and answer_str.endswith('\\]'):
        return answer_str[2:-2].strip()
    
    cleaned = re.sub(r'\\[\[\]]', '', answer_str)
    
    cleaned = cleaned.replace('\\dfrac', '\\frac')
    cleaned = cleaned.replace('\\tfrac', '\\frac')
    
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    
    return cleaned

def extract_boxed_answer(text):
    start_idx = text.find("\\boxed{")
    if start_idx == -1:
        return None
    
    start_idx += len("\\boxed{")
    brace_count = 1
    current_idx = start_idx
    n = len(text)
    
    while current_idx < n and brace_count > 0:
        if text[current_idx] == "\\" and current_idx + 1 < n:
            current_idx += 2
            continue
            
        if text[current_idx] == "{":
            brace_count += 1
        elif text[current_idx] == "}":
            brace_count -= 1
            
        current_idx += 1
    
    if brace_count == 0:
        return text[start_idx:current_idx - 1].strip()
    else:
        return None

def load_jsonl(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def calculate_scores(test_data, cleaned_input_data, results_data):
    valid_ids = {item['id'] for item in cleaned_input_data}

    scores = {"llm_solution": [], "llm_solution_v1": []}
    detailed_scores = []
    
    for test_item, result_item in tqdm.tqdm(zip(test_data, results_data), total=len(test_data), desc="Calculating Scores"):
        ref_answer = normalize_final_answer(test_item["final_answers"])
        item_id = test_item["id"]
        if item_id not in valid_ids:
            continue
        row = {"id": item_id, "reference_answer": ref_answer}
        
        llm_answer = extract_boxed_answer(result_item.get("improved_solutions", ""))
        if llm_answer:
            score = EED(ref_answer, llm_answer)[0]
            scores["llm_solution"].append(score)
            row["llm_solution_answer"] = llm_answer
            row["llm_solution_score"] = score
        else:
            row["llm_solution_answer"] = ""
            row["llm_solution_score"] = 0
        
        llmv1_answer = extract_boxed_answer(result_item.get("improved_solutions", ""))
        if llmv1_answer:
            score = EED(ref_answer, llmv1_answer)[0]
            scores["llm_solution_v1"].append(score)
            row["llm_solution_v1_answer"] = llmv1_answer
            row["llm_solution_v1_score"] = score
        else:
            row["llm_solution_v1_answer"] = ""
            row["llm_solution_v1_score"] = 0
        
        detailed_scores.append(row)
    
    return scores, detailed_scores

def generate_stats(scores):
    stats = {}
    for key, values in scores.items():
        if not values: 
            continue
            
        stats[key] = {
            "mean": np.mean(values),
            "std": np.std(values),
            "min": min(values),
            "max": max(values),
            "pass_rate": sum(1 for x in values if x > 60) / len(values) * 100
        }
    return stats

def plot_comparison(detailed_scores, output_dir):
    ids = [item['id'].split('/')[-1] for item in detailed_scores]

    solution_scores = [item['llm_solution_score'] for item in detailed_scores]
    solution_v1_scores = [item['llm_solution_v1_score'] for item in detailed_scores]
    
    x = np.arange(len(ids))

    plt.figure(figsize=(max(15, len(ids) * 0.5), 8))
    
    plt.plot(x, solution_scores, 'o', label='llm_solution_score', markersize=5)
    plt.plot(x, solution_v1_scores, 'o', label='llm_solution_v1_score', markersize=5)

    plt.xlabel('Problem ID')
    plt.ylabel('EED Score')
    plt.title('Score Comparison per Problem (Sorted by score)')
    plt.xticks(x, ids, rotation=90, fontsize=8)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.ylim(-5, 105)
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, 'score_comparison.png'))
    plt.close()

def run_evaluation(test_path, cleaned_input, results_path, output_dir):
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test file not found: {test_path}")
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"Results file not found: {results_path}")

    os.makedirs(output_dir, exist_ok=True)
    
    test_data = load_jsonl(test_path)
    if cleaned_input:
        cleaned_input_data = load_jsonl(cleaned_input)
    else:
        cleaned_input_data = test_data
    results_data = load_jsonl(results_path)
    
    scores, detailed_scores = calculate_scores(test_data, cleaned_input_data, results_data)
    
    detailed_scores.sort(key=lambda x: (x.get('llm_solution_v1_score', 0), x.get('llm_solution_score', 0)), reverse=True)
    
    stats = generate_stats(scores)
    
    with open(os.path.join(output_dir, 'stats.json'), 'w') as f:
        json.dump(stats, f, indent=2)
    
    detailed_csv = os.path.join(output_dir, 'detailed_scores.csv')
    with open(detailed_csv, 'w', newline='') as f:
        fieldnames = ['id', 'llm_solution_score','llm_solution_v1_score',
                      'reference_answer', 'llm_solution_answer',
                      'llm_solution_v1_answer', ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(detailed_scores)
    
    plot_comparison(detailed_scores, output_dir)
    
    print(f"Evaluation finished! Results saved at: {output_dir}")
    print(f"- Detailed scores: {detailed_csv}")

def main():
    run_evaluation(
        test_path="ui/test.jsonl",
        results_path="ui/results.jsonl",
        output_dir="src/scorer/phybench/eval_results"
    )

if __name__ == "__main__":
    main()
