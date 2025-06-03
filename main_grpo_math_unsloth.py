from unsloth import FastLanguageModel, FastModel


from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer

import wandb


from math_reward import math_reward_fn


# Settings
model_path = "models/L1-Qwen-1.5B-Max"
data_path = "scripts/data/MATH_processed_w_level_type/train.parquet"
output_dir = "scripts/data/MATH_finetuned"
run_name = "GRPO_MATH_UNSLOTH_L1-Qwen-1.5B-Fixed-Max"


train_dataset = load_dataset("parquet", data_files={"train": data_path}, split="train")
# small_train_dataset = train_dataset.select(range(10))

# Load model and tokenizer
# tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
# model = AutoModelForCausalLM.from_pretrained(model_path)


max_seq_length = 4000 # Can increase for longer reasoning traces
lora_rank = 64 # Larger rank = smarter, but slower

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_path,
    max_seq_length = max_seq_length,
    load_in_4bit = True, # False for LoRA 16bit
    fast_inference = True, # Enable vLLM fast inference
    max_lora_rank = lora_rank,
    gpu_memory_utilization = 0.5, # Reduce if out of memory
)

model = FastLanguageModel.get_peft_model(
    model,
    r = lora_rank, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ], # Remove QKVO if out of memory
    lora_alpha = lora_rank,
    use_gradient_checkpointing = "unsloth", # Enable long context finetuning
    random_state = 3407,
)




# Math-specific reward function
def math_reward_func(prompts, completions, completion_ids, data_source, ability, reward_model, extra_info):
    rewards = []
    for i, (prompt, completion, c_ids, rm) in enumerate(zip(prompts, completions, completion_ids, reward_model)):
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

        reward = math_reward_fn(solution, ground_truth, num_tokens=num_tokens, valid_response_length = num_solution_tokens)
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
                            gradient_accumulation_steps=4,
                            num_generations=2,
                            max_prompt_length=1536,
                            max_completion_length=4000,
                            num_train_epochs=2,
                            save_steps=200,
                            max_grad_norm=0.1,
                            report_to="wandb",
                            log_on_each_node=False,
                        )

trainer = GRPOTrainer(
    model=model,
    reward_funcs=math_reward_func,
    processing_class=tokenizer,
    args=training_args,
    train_dataset=train_dataset,
)


trainer.train()