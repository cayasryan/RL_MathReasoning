import pandas as pd
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
import torch

import wandb

wandb.init(project="MATH-finetuning", name="bert_level_classification")

# === Load data ===
train_df = pd.read_json("scripts/data/MATH/math_train.json")
val_df = pd.read_csv("scripts/data/MATH/math_val.csv")

# === Label encoding for level ===
level_encoder = LabelEncoder()
train_df["level_id"] = level_encoder.fit_transform(train_df["difficulty"])
val_df["level_id"] = level_encoder.transform(val_df["level"])  # use same encoder!

# Rename problem column to question for consistency
train_df.rename(columns={"problem": "question"}, inplace=True)
val_df.rename(columns={"problem": "question"}, inplace=True)

# === Tokenizer ===
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_and_format(df, label_col):
    ds = Dataset.from_pandas(df[["question", label_col]])
    ds = ds.map(lambda e: tokenizer(e["question"], truncation=True, padding="max_length", max_length=512), batched=True)
    ds = ds.rename_column(label_col, "labels")
    ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return ds

train_dataset = tokenize_and_format(train_df, "level_id")
val_dataset = tokenize_and_format(val_df, "level_id")

# === Model & Trainer ===
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=len(level_encoder.classes_))

training_args = TrainingArguments(
    output_dir="./bert_level",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=20,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=2e-5,
    weight_decay=0.01,
    load_best_model_at_end=True,
    logging_dir="./logs",
    report_to=["wandb"],        # Enable W&B reporting
    run_name="bert_level_classification", 
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],  # Stop if no improvement for 5 evals
    tokenizer=tokenizer,
)

trainer.train()
