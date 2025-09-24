import os
import json
import csv
import pandas as pd
import sys
sys.path.append('libs/ABench/Physics/src')
from .evaluate import eval_Benchmark_A, eval_Benchmark_B
from typing import Literal

def load_jsonl(file_path) -> list[dict]:
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def run_evaluation(test_path, results_path, output_dir, type: Literal['Phy_A_fixed_400', 'Phy_B_dynamic_100'] = None):
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test file not found: {test_path}")
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"Results file not found: {results_path}")

    os.makedirs(output_dir, exist_ok=True)

    test_data = load_jsonl(test_path)
    results_data = load_jsonl(results_path)

    if type is None:
        type = test_data[0]['tag']
    form_value = []
    for i in range(len(test_data)):
        assert test_data[i]['id'] == results_data[i]['id']
        id = test_data[i]['id']
        if type == 'Phy_A_fixed_400':
            form_value.append([
                int(id),
                test_data[i]['questions'],
                test_data[i]['final_answers'][0],
                results_data[i]['llm_solution'],
            ])
        elif type == 'Phy_B_dynamic_100':
            id, sub_id = id.split('_')
            form_value.append([
                int(id),
                int(sub_id),
                test_data[i]['questions'],
                test_data[i]['final_answers'][0],
                results_data[i]['llm_solution'],
            ])
        else:
            raise ValueError(f"Unsupported type: {type}")
    if type == 'Phy_A_fixed_400':
        df = pd.DataFrame(form_value, columns=['mid', 'standard_question', 'standard_answer' , 'llm_solution'])
        df.to_csv(os.path.join(output_dir, 'results.csv'), index=False, encoding='utf-8-sig', quoting=csv.QUOTE_ALL)
        eval_Benchmark_A(os.path.join(output_dir, 'results.csv'), 'llm_solution', output_dir)
    elif type == 'Phy_B_dynamic_100':
        df = pd.DataFrame(form_value, columns=['mid', 'subid', 'standard_question', 'standard_answer' , 'llm_solution'])
        df.to_csv(os.path.join(output_dir, 'results.csv'), index=False, encoding='utf-8-sig', quoting=csv.QUOTE_ALL)
        eval_Benchmark_B(os.path.join(output_dir, 'results.csv'), 'llm_solution', output_dir)
    else:
        raise ValueError(f"Unsupported type: {type}")
