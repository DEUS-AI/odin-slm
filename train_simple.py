#!/usr/bin/env python3
"""Simplified training script for debugging"""

import os
os.environ["ACCELERATE_USE_FSDP"] = "false"
os.environ["ACCELERATE_USE_DEEPSPEED"] = "false"
os.environ["ACCELERATE_MIXED_PRECISION"] = "bf16"  # Don't use FP8
os.environ["ACCELERATE_USE_FP8"] = "false"

import warnings
warnings.filterwarnings("ignore", category=SyntaxWarning)

# CRITICAL: Patch FP8BackendType BEFORE any other imports
from enum import Enum
import sys

# Pre-import accelerate and patch it
import accelerate.utils
if not hasattr(accelerate.utils, 'FP8BackendType'):
    class FP8BackendType(Enum):
        MSAMP = "MS_AMP"
        TE = "TE"
    accelerate.utils.FP8BackendType = FP8BackendType

print("=" * 80)
print("SIMPLE MEDICAL NER TRAINING")
print("=" * 80)
print(f"✓ FP8BackendType: {hasattr(accelerate.utils, 'FP8BackendType')}")

# Import unsloth first (AFTER FP8 patch)
print("\n1. Importing Unsloth...")
from unsloth import FastLanguageModel

print("2. Importing other libraries...")
import torch
from transformers import TrainingArguments
from trl import SFTTrainer
from datasets import Dataset
import json

print("3. Loading model...")
# Use Mistral 7B - well supported by unsloth 2024.10.7
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/mistral-7b-v0.3-bnb-4bit",
    max_seq_length=512,
    dtype=None,
    load_in_4bit=True,
)
print("✓ Model loaded")

print("4. Adding LoRA...")
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
)
print("✓ LoRA added")

print("5. Loading dataset...")
with open("data/datasets/formatted/train.json") as f:
    train_data = json.load(f)
with open("data/datasets/formatted/val.json") as f:
    val_data = json.load(f)

train_dataset = Dataset.from_list([{"text": f"{d['instruction']}\\n\\n{d['input']}\\n\\n{d['output']}"} for d in train_data[:100]])  # Use only 100 for testing
val_dataset = Dataset.from_list([{"text": f"{d['instruction']}\\n\\n{d['input']}\\n\\n{d['output']}"} for d in val_data[:20]])

print(f"✓ Train: {len(train_dataset)}, Val: {len(val_dataset)}")

print("6. Setting up trainer...")

# Fix for TRL compatibility issue
from dataclasses import dataclass, field
import sys

# Monkey patch TrainingArguments if needed
if not hasattr(TrainingArguments, 'model_init_kwargs'):
    original_post_init = TrainingArguments.__post_init__
    def patched_post_init(self):
        if not hasattr(self, 'model_init_kwargs'):
            self.model_init_kwargs = None
        original_post_init(self)
    TrainingArguments.__post_init__ = patched_post_init

training_args = TrainingArguments(
    output_dir="./experiments/medical_ner_test",
    num_train_epochs=1,  # Just 1 epoch for testing
    per_device_train_batch_size=2,
    gradient_accumulation_steps=2,
    learning_rate=2e-4,
    fp16=False,  # Model uses bfloat16
    bf16=True,   # Use bfloat16 instead
    logging_steps=5,
    save_steps=50,
    eval_steps=50,
    eval_strategy="steps",  # Updated from evaluation_strategy
    save_strategy="steps",
    optim="adamw_8bit",
    warmup_steps=10,
    weight_decay=0.01,
    lr_scheduler_type="linear",
    seed=42,
    report_to="none",
    dataloader_pin_memory=False,
    # Disable distributed training features
    local_rank=-1,
    ddp_backend=None,
    fsdp="",
    deepspeed=None,
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    dataset_text_field="text",
    max_seq_length=512,
    args=training_args,
    packing=False,
)
print("✓ Trainer created")

print("\n🚀 Starting training...")
print("=" * 80)

trainer.train()

print("\n✓ TRAINING COMPLETE!")
