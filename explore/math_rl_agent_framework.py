# math_rl_agent_framework.py
"""
Adaptive Thinking‑Time Control & Problem‑Aware Reasoning for MATH Benchmark
-------------------------------------------------------------------------
This code contains:

1. `ProblemClassifier` – supervised classifier for (type, difficulty)
2. `ReasoningEngine`   – lightweight wrapper around a frozen language model
3. `MathEnv`           – OpenAI‑Gym compliant RL environment
4. Utility functions   – probing, training (PPO), evaluation

Tested with:
    • Python 3.10
    • torch >= 2.1
    • transformers >= 4.41
    • datasets >= 2.19
    • gymnasium >= 0.29
    • stable‑baselines3 >= 3.3
"""

from __future__ import annotations

import os

import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import gymnasium as gym
import numpy as np
import torch
from datasets import Dataset
from gymnasium import spaces
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedTokenizerFast,
    LlamaForCausalLM
)
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback



# Set-up CUDA device
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
# use a specific GPU
os.environ["CUDA_VISIBLE_DEVICES"]="7"

# Use GPU for inference
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Print the device being used
print(f"Using device: {device}")

# Check the GPU name
if device.type == 'cuda':
    gpu_name = torch.cuda.get_device_name(0)  # 0 because CUDA_VISIBLE_DEVICES=4 means GPU 4 is now 0
    print("Using GPU:", gpu_name)


# ---------------------------------------------------------------------------
# 1. Supervised Problem Type & Difficulty Classifier
# ---------------------------------------------------------------------------
def build_label_map(dataset):
    """
    Returns:
        label_map: id -> (type_str, level_int)
        encode:  (type_str, level_int) -> id  (for training, if you want it)
    """
    # unique problem categories (7 in the dataset)
    types = sorted(set(dataset["type"]))
    
    # 'Level 3' → 3
    def lvl_int(lvl_str): 
        try:
            return int(lvl_str.strip().split()[1])
        except (IndexError, ValueError):
            return 6 
    levels = sorted({lvl_int(lvl) for lvl in dataset["level"]})  # [1-5]
    
    label_map, encode = {}, {}
    for t in types:
        for l in levels:
            idx = len(label_map)
            label_map[idx] = (t, l)
            encode[(t, l)] = idx
    return label_map, encode

class ProblemClassifier:
    """Lightweight text classifier.  Fine‑tune or plug‑in a checkpoint."""

    def __init__(self, model_name: str, label_map: Dict[int, Tuple[str, int]]):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.label_map = label_map  # id -> (type, level)
        self.model.eval()
        if torch.cuda.is_available():
            self.model.to("cuda")

    @torch.no_grad()
    def predict(self, text: str) -> Tuple[str, int]:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True)
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        logits = self.model(**inputs).logits[0]
        label_id = int(logits.argmax())
        return self.label_map[label_id]


# ---------------------------------------------------------------------------
# 2. Reasoning Engine (frozen LM) with step‑wise generation
# ---------------------------------------------------------------------------
from prompt_template import PROMPT_PREFIX

class ReasoningEngine:
    """Generates step‑by‑step chain‑of‑thought until told to stop."""

    def __init__(self, model_dir: str, max_tokens_per_step: int = 32):
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(model_dir, padding_side="left")
        self.model = LlamaForCausalLM.from_pretrained(model_dir)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.eval().requires_grad_(False)
        if torch.cuda.is_available():
            self.model.to("cuda")
        self.max_tokens_per_step = max_tokens_per_step
        self.prefix = PROMPT_PREFIX

    def reset_prompt(self, problem_text):
        # Called at env.reset()
        self.prompt = (
            f"{self.prefix}\n"
            f"Problem: {problem_text}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        )
        return self.prompt

    @torch.no_grad()
    def think_step(self, prompt: str) -> str:
        """Generate a *single* reasoning chunk."""
        inputs = self.tokenizer(prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        out = self.model.generate(
            **inputs,
            max_new_tokens=self.max_tokens_per_step,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        generated = self.tokenizer.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        return generated.strip()

    @staticmethod
    def extract_answer(text: str) -> str:
        from math_utils import last_boxed_only_string, normalize_final_answer, remove_boxed

        last_boxed_text = last_boxed_only_string(text)
        answer = normalize_final_answer(remove_boxed(last_boxed_text)) if last_boxed_text else None
        return answer


# ---------------------------------------------------------------------------
# 3. RL Environment
# ---------------------------------------------------------------------------
def load_local_math_dataset(dataset_split: str, root_dir="MATH"):
    data = []
    root = Path(root_dir)

    for split in ["train", "test"]:
        split_dir = root / split
        for subject_dir in split_dir.iterdir():
            if not subject_dir.is_dir():
                continue
            subject = subject_dir.name
            for file_path in subject_dir.glob("*.json"):
                with open(file_path, "r") as f:
                    item = json.load(f)
                    data.append({
                        "problem": item["problem"],
                        "solution": item["solution"],
                        "type": item["type"],
                        "level": item['level'],
                        "split": split,
                        "path": str(file_path)
                    })
    return Dataset.from_list(data)

class MathEnv(gym.Env):
    """Gym‑compatible environment where the agent decides to *think* or *stop*."""

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        dataset_split: str = "train[:2048]",  # small default slice
        model_dir: str = "../../../llm/llama/Llama-3.2-1B-Instruct",
        classifier_name: str = "distilbert-base-uncased",
        penalty_lambda: float = 0.003,
        max_steps: int = 50,
        seed: int = 42,
    ):
        super().__init__()
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Dataset ------------------------------------------------------------
        self.data = load_local_math_dataset(dataset_split, root_dir="MATH")

        # Build label map
        self.label_map, self.encode = build_label_map(self.data)

        # Components ---------------------------------------------------------
        self.classifier = ProblemClassifier(classifier_name, self.label_map)
        self.engine = ReasoningEngine(model_dir)

        # Hyper‑parameters ---------------------------------------------------
        self.penalty_lambda = penalty_lambda
        self.max_steps = max_steps

        # Spaces -------------------------------------------------------------
        # Observation: flattened categorical ids – replace w/ embeddings if needed
        self.observation_space = spaces.Box(low=0, high=1, shape=(10,), dtype=np.float32)
        # Action: 0 = CONTINUE, 1 = STOP
        self.action_space = spaces.Discrete(2)

        # Internal state -----------------------------------------------------
        self.idx = -1
        self.problem: Dict = {}
        self.prompt = ""
        self.reasoning_trace: List[str] = []
        self.step_count = 0

    # ---------------------------------------------------------------------
    # Gym API
    # ---------------------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: Dict | None = None):
        if seed is not None:
            super().reset(seed=seed)
        self.idx = (self.idx + 1) % len(self.data)
        self.problem = self.data[self.idx]
        self.prompt = self.engine.reset_prompt(self.problem["problem"])  # chain‑of‑thought prefix
        self.reasoning_trace = []
        self.step_count = 0


        obs = self._build_obs()
        info = {"answer": self.problem["solution"]}
        return obs, info

    def step(self, action: int):
        done = False
        reward = 0.0
        info = {}

        if action == 0:  # CONTINUE
            step_text = self.engine.think_step(self.prompt)
            self.reasoning_trace.append(step_text)
            self.prompt += step_text + "\n"
            self.step_count += 1
            done = self.step_count >= self.max_steps  # forced stop
            # living penalty per think step
            reward -= self.penalty_lambda
        else:  # STOP & answer
            from math_utils import is_equiv
            done = True
            generated_answer = self.engine.extract_answer(self.prompt)
            gt_answer = self.engine.extract_answer(self.problem["solution"]) 

            if generated_answer is not None or generated_answer != 24:
                try:
                    exact_match = int(generated_answer.strip() == gt_answer.strip())
                    # or is_equiv(generated_answer, gt_answer)
                except AttributeError:
                    exact_match = int(generated_answer.strip() == gt_answer)
            elif generated_answer is None or generated_answer == 24:
                exact_match = 0
                reward += -2
            else:
                exact_match = 0
            
            correct = exact_match > 0
            if correct:
                reward += 1.0
            else:
                reward -= 1.0
            info["correct"] = correct
            info["generated_answer"] = generated_answer

            print(f"\n🧠 Final prompt:\n{self.prompt}")
            print(f"✅ GT: {gt_answer}")
            print(f"📝 Model Response: {generated_answer}")
            print(f"🎯 Correct: {correct}, Reward: {reward}") 


        obs = self._build_obs() if not done else np.zeros_like(self.observation_space.sample())
            
        return obs, reward, done, False, info  # Gymnasium API (v0.29)

    # ------------------------------------------------------------------
    def _build_obs(self):
        """Cheap hand‑crafted obs: [type_id, level, step_count, ...padding]"""
        p_type, level = self.classifier.predict(self.problem["problem"])
        type_id = hash(p_type) % 7  # up to 7 base types – update as needed
        vec = np.zeros(self.observation_space.shape, dtype=np.float32)
        vec[0] = type_id / 7.0
        vec[1] = level / 5.0  # assume 1‑5 scale
        vec[2] = self.step_count / float(self.max_steps)
        return vec

    # ------------------------------------------------------------------
    def render(self):
        print("\nProblem:", self.problem["problem"])
        print("\nReasoning Trace:")
        for t in self.reasoning_trace:
            print("  >", t)


# ---------------------------------------------------------------------------
# 4. Probing, Training & Evaluation Helpers
# ---------------------------------------------------------------------------

class ProgressCallback(BaseCallback):
    def __init__(self, verbose=1):
        super().__init__(verbose)
        self.episode = 0

    def _on_step(self) -> bool:
        # Check if a new episode has started
        if self.locals.get("dones") and any(self.locals["dones"]):
            self.episode += 1
            if self.verbose:
                print(f"Episode: {self.episode} | Timesteps: {self.num_timesteps}")
        return True

def run_probing(env: MathEnv, n_eps: int = 100):
    """Estimate accuracy vs. steps without RL – baseline."""
    stats: Dict[int, List[bool]] = {}
    for _ in range(n_eps):
        obs, info = env.reset()
        for step in range(env.max_steps):
            obs, reward, done, _, info = env.step(0)  # always continue
            if done:
                stats.setdefault(step + 1, []).append(info.get("correct", False))
                break
    for k, vals in stats.items():
        acc = np.mean(vals)
        print(f"steps={k:2d}  accuracy={acc:.3f}  n={len(vals)}")
    return stats


def train_policy(total_timesteps: int = 5_000):
    """Train PPO agent on MathEnv."""
    env_fn = lambda: MathEnv()
    vec_env = DummyVecEnv([env_fn])
    model = PPO("MlpPolicy", vec_env, verbose=1, batch_size=128, device="cpu")
    callback = ProgressCallback()
    model.learn(total_timesteps=total_timesteps, callback=callback)
    model.save("ppo_math_agent")
    return model


def eval_policy(model_path: str = "ppo_math_agent", episodes: int = 50):
    env = MathEnv(dataset_split="test[:256]")
    model = PPO.load(model_path)
    correct, steps_used = 0, []
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, info = env.step(int(action))
        if info.get("correct"):
            correct += 1
        steps_used.append(env.step_count)
    print(f"Accuracy: {correct/episodes:.3f} | Avg steps: {np.mean(steps_used):.2f}")


# ---------------------------------------------------------------------------
# 5. CLI Entry‑points
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Adaptive Thinking‑Time Agent for MATH")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub_probe = sub.add_parser("probe")
    sub_probe.add_argument("--episodes", type=int, default=100)

    sub_train = sub.add_parser("train")
    sub_train.add_argument("--steps", type=int, default=10_000)

    sub_eval = sub.add_parser("eval")
    sub_eval.add_argument("--episodes", type=int, default=100)

    args = parser.parse_args()

    if args.cmd == "probe":
        env = MathEnv()
        run_probing(env, n_eps=args.episodes)

    elif args.cmd == "train":
        train_policy(total_timesteps=args.steps)

    elif args.cmd == "eval":
        eval_policy(episodes=args.episodes)
