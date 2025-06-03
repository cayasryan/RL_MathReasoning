from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer
from transformers import AutoTokenizer, AutoModelForCausalLM

from torch.optim import AdamW  # to access `foreach=False` arg


from math_reward import math_reward_fn


import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import numpy as np

# Set-up CUDA device
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
# use a specific GPU
os.environ["CUDA_VISIBLE_DEVICES"]="6"


# Settings
model_path = "models/L1-Qwen-1.5B-Max"
data_path = "scripts/data/MATH_processed/train.parquet"
output_dir = "scripts/data/MATH_finetuned"
run_name = "GRPO_MATH_L1-Qwen-1.5B-Max"


train_dataset = load_dataset("parquet", data_files={"train": data_path}, split="train")
# small_train_dataset = train_dataset.select(range(2))

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
model = AutoModelForCausalLM.from_pretrained(model_path)
model = model.to("cuda" if torch.cuda.is_available() else "cpu")



class MyGRPOTrainer(GRPOTrainer):
    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer

        # Standard way to separate weight decay parameters
        decay_parameters = []
        no_decay_parameters = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if any(nd in name for nd in ["bias", "LayerNorm.weight"]):
                no_decay_parameters.append(param)
            else:
                decay_parameters.append(param)

        optimizer_grouped_parameters = [
            {"params": decay_parameters, "weight_decay": self.args.weight_decay},
            {"params": no_decay_parameters, "weight_decay": 0.0},
        ]

        self.optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.args.learning_rate,
            betas=(self.args.adam_beta1, self.args.adam_beta2),
            eps=getattr(self.args, "adam_epsilon", 1e-8),  # fallback default
            foreach=False,  # prevents OOM in foreach kernel
        )

        return self.optimizer


# Math-specific reward function
def math_reward_func(prompts, completions, completion_ids, data_source, ability, reward_model, extra_info):
    rewards = []
    for prompt, completion, c_ids, rm in zip(prompts, completions, completion_ids, reward_model):
        # Calculate math-specific reward
        ground_truth = rm["ground_truth"]
        num_tokens = rm["num_tokens"]

        solution = completion[0]["content"].strip()

        # Count the number of tokens in the solution
        if isinstance(solution, str):
            solution_tokens = tokenizer(solution, return_tensors="pt").input_ids[0]
            num_solution_tokens = len(solution_tokens)
        else:
            num_solution_tokens = 0

        reward = math_reward_fn(completion, ground_truth, num_tokens=num_tokens, valid_response_length = num_solution_tokens)
        rewards.append(reward)
    return rewards





training_args = GRPOConfig(
                            output_dir=output_dir,
                            run_name=run_name,
                            learning_rate=5e-6,
                            adam_beta1 = 0.9,
                            adam_beta2 = 0.99,
                            weight_decay = 0.1,
                            warmup_ratio = 0.1,
                            lr_scheduler_type='cosine',
                            logging_steps=10,
                            bf16=True,
                            per_device_train_batch_size=2,
                            gradient_accumulation_steps=1,
                            num_generations=2,
                            max_prompt_length=1536,
                            max_completion_length=4000,
                            num_train_epochs=1,
                            save_steps=100,
                            max_grad_norm=0.1,
                            report_to="wandb",
                            log_on_each_node=False,
                        )

trainer = MyGRPOTrainer(
    model=model,
    reward_funcs=math_reward_func,
    processing_class=tokenizer,
    args=training_args,
    train_dataset=train_dataset,
)

trainer.train()