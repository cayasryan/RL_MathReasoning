import os
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from trl import PPOTrainer, PPOConfig
from math_reward import math_reward_fn
import numpy as np
from datasets import load_dataset

def load_data(parquet_path, n_samples=None):
    df = pd.read_parquet(parquet_path)
    if n_samples:
        df = df.head(n_samples)
    prompts = [p[0]["content"] if isinstance(p, np.ndarray) and len(p) > 0 else "" for p in df["prompt"]]
    ground_truths = [rm["ground_truth"] for rm in df["reward_model"]]
    return prompts, ground_truths

def reward_fn(samples, ground_truths):
    # samples: list of generated responses
    # ground_truths: list of ground truth answers
    rewards = []
    for sample, gt in zip(samples, ground_truths):
        reward = math_reward_fn(sample, gt)
        rewards.append(float(reward))
    return rewards

def main():
    import os
    import torch
    import numpy as np

    # Set-up CUDA device
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
    # use a specific GPU
    os.environ["CUDA_VISIBLE_DEVICES"]="7"

    # Use GPU for inference
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Settings
    model_path = "models/L1-Qwen-1.5B-Max"
    data_path = "scripts/data/MATH_processed/train.parquet"
    output_dir = "scripts/data/MATH_finetuned"
    os.makedirs(output_dir, exist_ok=True)
    n_samples = 8  # for quick test, set to None for full dataset
    batch_size = 2
    ppo_epochs = 1  # increase for real training

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(model_path)
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    prompts, ground_truths = load_data(data_path, n_samples=n_samples)

    train_dataset = load_dataset("parquet", data_files={"train": data_path}, split="train")

    # PPO config
    config = PPOConfig(
        batch_size=batch_size,
        local_rollout_forward_batch_size=batch_size,
        mini_batch_size=batch_size,
        num_ppo_epochs=ppo_epochs,
        learning_rate=1.41e-5,
        report_to="wandb",
    )

    # PPO Trainer
    ppo_trainer = PPOTrainer(
        model=model,
        ref_model=None,  # You can provide a reference model for KL penalty, or leave as None
        reward_model=None,
        train_dataset=train_dataset,
        processing_class=None,
        value_model=None,  # Optional: you can provide a value model for advantage estimation 
        args=config,
    )

    # PPO Loop
    for epoch in range(ppo_epochs):
        print(f"Epoch {epoch+1}/{ppo_epochs}")
        for i in tqdm(range(0, len(prompts), batch_size)):
            batch_prompts = prompts[i:i+batch_size]
            batch_ground_truths = ground_truths[i:i+batch_size]

            # Tokenize and move to device
            batch_inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True)
            batch_inputs = {k: v.to(model.device) for k, v in batch_inputs.items()}
            input_len = batch_inputs["input_ids"].shape[1]

            # Generate responses
            generate_fn = model.module.generate if isinstance(model, torch.nn.DataParallel) else model.generate
            response_ids = generate_fn(**batch_inputs, max_new_tokens=256, pad_token_id=tokenizer.eos_token_id)

            responses = [
                tokenizer.decode(r[input_len:], skip_special_tokens=True)
                for r in response_ids
            ]

            # Compute rewards
            rewards = reward_fn(responses, batch_ground_truths)

            # PPO step
            stats = ppo_trainer.step(batch_prompts, responses, rewards)
            print(f"Batch {i//batch_size+1}: reward mean={sum(rewards)/len(rewards):.3f}")

    # Save the fine-tuned model
    model.save_pretrained(os.path.join(output_dir, "ppo_finetuned_model"))
    tokenizer.save_pretrained(os.path.join(output_dir, "ppo_finetuned_model"))
    print("Model saved.")

if __name__ == "__main__":
    main()