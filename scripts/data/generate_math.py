import os
import pandas as pd
import random
from typing import Dict, Any, Optional, List

import sys

# Add parent directory to sys.path
parent_dir = os.path.abspath("../..") 
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from utils import last_boxed_only_string, remove_boxed

def extract_solution(solution_str: str) -> str:
    """Extract the final boxed solution from a solution string."""
    return remove_boxed(last_boxed_only_string(solution_str))

def make_map_fn(split: str, num_tokens: int = -1):
    def process_fn(example: Dict[str, Any], idx: int) -> Optional[Dict[str, Any]]:
        question = example['problem']
        answer = example['solution'] if 'solution' in example else example['answer']
        random_number = num_tokens if num_tokens != -1 else random.randint(100, 4000)
        instruction = "Let's think step by step and output the final answer within \\boxed{}."
        if num_tokens != -1:
            if num_tokens < 0:
                instruction += f" Think for maximum {abs(num_tokens)} tokens."
            else:
                instruction += f" Think for {num_tokens} tokens."
        else:
            instruction += f" Think for {random_number} tokens."
        question = f"{question} {instruction}"
        data = {
            "data_source": "MATH",
            "prompt": [{
                "role": "user",
                "content": question
            }],
            "ability": "math",
            "reward_model": {
                "style": "rule",
                "ground_truth": answer,
                "num_tokens": random_number
            },
            "extra_info": {
                'split': split,
                'index': idx,
                'level': example.get('level', 'unknown'),
                'type': example.get('type', 'unknown'),
            }
        }
        return data
    return process_fn

def process_math_csv(input_csv: str, split: str, output_parquet: str, num_tokens: int = -1):
    df = pd.read_csv(input_csv)
    process_fn = make_map_fn(split, num_tokens)
    processed_data = []
    for idx, row in df.iterrows():
        processed = process_fn(row, idx)
        if processed is not None:
            processed_data.append(processed)
    pd.DataFrame(processed_data).to_parquet(output_parquet)
    print(f"Saved {len(processed_data)} examples to {output_parquet}")

if __name__ == '__main__':
    # Paths
    math_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "MATH"))
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "MATH_processed"))
    os.makedirs(output_dir, exist_ok=True)

    # File mapping: (csv_name, split, output_name)
    files = [
        ("math_train.csv", "train", "train.parquet"),
        ("math_val.csv", "val", "val.parquet"),
        ("math500_test.csv", "test", "test.parquet"),
    ]

    # You can set this to -1 for random, or a specific number for all
    NUM_TOKENS = -3600

    for csv_name, split, out_name in files:
        input_csv = os.path.join(math_dir, csv_name)
        output_parquet = os.path.join(output_dir, out_name)
        process_math_csv(input_csv, split, output_parquet, num_tokens=NUM_TOKENS)