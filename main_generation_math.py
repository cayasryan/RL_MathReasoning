import os
import csv
import numpy as np
import pandas as pd
from tqdm import tqdm
from tabulate import tabulate

from transformers import AutoTokenizer, AutoModelForCausalLM

from math_reward import math_reward_fn

def generate_responses(model, tokenizer, prompts, n_samples=1, max_new_tokens=256, batch_size=4):
    import torch
    from tqdm import tqdm

    responses = []
    token_lengths = []

    for i in tqdm(range(0, len(prompts), batch_size), desc="Generating"):
        batch_prompts = prompts[i:i+batch_size]
        batch_responses = [[] for _ in range(len(batch_prompts))]
        batch_token_lengths = [[] for _ in range(len(batch_prompts))]

        for _ in range(n_samples):
            # Tokenize the full batch
            inputs = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                return_attention_mask=True
            )

            # Move to GPU
            inputs = {k: v.cuda() for k, v in inputs.items()}

            # Generate responses for the batch
            outputs = model.module.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=max_new_tokens,
                do_sample=False,  # ← Disable sampling
                temperature=None,
                top_p=None,
                top_k=None,
                num_beams=1,
                pad_token_id=tokenizer.eos_token_id
            )

            # Decode and store responses
            for j, output in enumerate(outputs):
                gen_tokens = output[inputs["input_ids"].shape[1]:]  # Remove prompt tokens
                decoded = tokenizer.decode(gen_tokens, skip_special_tokens=True)
                batch_responses[j].append(decoded)
                batch_token_lengths[j].append(len(tokenizer.encode(decoded)))

        responses.extend(batch_responses)
        token_lengths.extend(batch_token_lengths)

    return responses, token_lengths

def evaluate_responses(responses, ground_truths):
    total_scores = []
    passes = 0
    for resp_list, gt in tqdm(zip(responses, ground_truths), total=len(responses), desc="Evaluating"):
        score_list = [math_reward_fn(r, gt) for r in resp_list]
        total_scores.append(score_list)
        if np.max(score_list) == 1:
            passes += 1
    pass_at_1 = np.mean([s[0] for s in total_scores])
    pass_at_n = passes / len(responses)
    return pass_at_1, pass_at_n, total_scores

def main():
    import os
    import torch
    import numpy as np

    # Set-up CUDA device
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
    # use a specific GPU
    os.environ["CUDA_VISIBLE_DEVICES"]="6"

    # Use GPU for inference
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # Settings
    model_path = "RL_MathReasoning/models/L1-Qwen-1.5B-Max"
    data_path = "RL_MathReasoning/scripts/data/MATH_processed/val.parquet"  # Change as needed
    output_dir = "RL_MathReasoning/scripts/data/MATH_outputs"
    os.makedirs(output_dir, exist_ok=True)
    n_samples = 1
    max_new_tokens = 8000
    batch_size = 16

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    model = torch.nn.DataParallel(model)  # This will use all available GPUs
    model = model.cuda()

    model.eval()

    # Load data
    df = pd.read_parquet(data_path)

    prompts = [p[0]["content"] if isinstance(p, np.ndarray) and len(p) > 0 else "" for p in df["prompt"]]
    ground_truths = [rm["ground_truth"] for rm in df["reward_model"]]

    # Generate responses
    responses, token_lengths = generate_responses(model, tokenizer, prompts, n_samples=n_samples, max_new_tokens=max_new_tokens, batch_size=batch_size)

    # Save responses
    df["responses"] = responses
    df["token_lengths"] = token_lengths
    output_parquet = os.path.join(output_dir, os.path.basename(data_path).replace(".parquet", "_with_responses.parquet"))
    df.to_parquet(output_parquet)
    print(f"Saved responses to {output_parquet}")

    # Evaluate
    pass_at_1, pass_at_n, total_scores = evaluate_responses(responses, ground_truths)
    print(f"pass@1: {pass_at_1:.4f}, pass@{n_samples}: {pass_at_n:.4f}")

    # Save metrics
    csv_path = os.path.join(output_dir, "pass.csv")
    row_data = {
        "model_path": model_path,
        "dataset": os.path.basename(data_path),
        "pass@1": pass_at_1,
        f"pass@{n_samples}": pass_at_n
    }
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=row_data.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_data)

    # Save total_scores
    total_scores_df = pd.DataFrame(total_scores)
    total_scores_df.to_parquet(os.path.join(output_dir, f"total_scores_{os.path.basename(data_path)}"))

    # Print summary
    table_data = [[k, v] for k, v in row_data.items()]
    print(tabulate(table_data, headers=['Metric', 'Value'], tablefmt='grid'))

if __name__ == "__main__":
    main()