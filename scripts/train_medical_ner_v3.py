#!/usr/bin/env python3
"""Train medical NER/RE model with v3 staged training

v3 Staged Training Strategy:
- Stage 1: Train entities only (3 epochs, LoRA rank 16)
- Stage 2: Continue with full NER+RE (5 epochs, lower LR)

Usage:
    # Stage 1
    python scripts/train_medical_ner_v3.py --config configs/medical_ner_re_config_v3_stage1.yaml --stage 1

    # Stage 2
    python scripts/train_medical_ner_v3.py --config configs/medical_ner_re_config_v3_stage2.yaml --stage 2
"""

import warnings
import argparse
import sys
from pathlib import Path
import json

# Suppress warnings
warnings.filterwarnings("ignore", category=SyntaxWarning, module="unsloth_zoo")
warnings.filterwarnings("ignore", category=SyntaxWarning, module="unsloth")

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

print("DEBUG: Importing yaml...")
import yaml

print("DEBUG: Importing unsloth...")
# IMPORTANT: Import unsloth FIRST, before torch/transformers
from unsloth import FastLanguageModel

print("DEBUG: Importing torch...")
import torch

print("DEBUG: Importing transformers...")
from transformers import TrainingArguments

print("DEBUG: Importing trl...")
from trl import SFTTrainer

print("DEBUG: Importing datasets...")
from datasets import Dataset

print("DEBUG: All imports complete!")


def load_config(config_path):
    """Load training configuration"""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_instruction_dataset(file_path):
    """Load instruction-formatted dataset"""
    print(f"Loading dataset from {file_path}...")

    with open(file_path, "r") as f:
        data = json.load(f)

    # Format for SFTTrainer
    formatted_data = []
    for item in data:
        # Combine instruction, input, and output into a single text
        text = f"""### Instruction:
{item['instruction']}

### Input:
{item['input']}

### Output:
{item['output']}"""
        formatted_data.append({"text": text})

    dataset = Dataset.from_list(formatted_data)
    print(f"✓ Loaded {len(dataset)} examples")
    return dataset


def train_medical_ner_v3(config_path, stage=1):
    """Main training function for v3 staged training

    Args:
        config_path: Path to config file
        stage: Training stage (1 or 2)
    """

    print("=" * 80)
    print(f"MEDICAL NER/RE MODEL TRAINING - V3 STAGE {stage}")
    print("=" * 80)
    if stage == 1:
        print("STAGE 1: Entity-only training (3 epochs)")
    else:
        print("STAGE 2: Full NER+RE training (5 epochs, from Stage 1 checkpoint)")
    print("=" * 80)

    # Load configuration
    print(f"\nLoading configuration from {config_path}...")
    config = load_config(config_path)
    print("✓ Configuration loaded")

    # Load base model for both stages
    print(f"\nLoading base model: {config['model']['name']}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config['model']['name'],
        max_seq_length=config['model']['max_seq_length'],
        dtype=None,
        load_in_4bit=config['model']['load_in_4bit'],
    )
    print("✓ Model loaded")

    # Add LoRA adapters
    print("\nAdding LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=config['lora']['r'],
        target_modules=config['lora']['target_modules'],
        lora_alpha=config['lora']['lora_alpha'],
        lora_dropout=config['lora']['lora_dropout'],
        bias=config['lora']['bias'],
        use_gradient_checkpointing="unsloth",
        random_state=config['dataset']['seed'],
    )
    print("✓ LoRA adapters added")

    # Stage 2: Verify Stage 1 checkpoint exists
    resume_checkpoint = None
    if stage == 2:
        # Use the last training checkpoint from Stage 1
        stage1_checkpoint = Path("./experiments/medical_ner_re_v3/stage1/checkpoint-750")
        if not stage1_checkpoint.exists():
            print(f"❌ ERROR: Stage 1 checkpoint not found at {stage1_checkpoint}")
            print("Please run Stage 1 training first:")
            print("  python scripts/train_medical_ner_v3.py --config configs/medical_ner_re_config_v3_stage1.yaml --stage 1")
            sys.exit(1)
        resume_checkpoint = str(stage1_checkpoint)
        print(f"✓ Will resume from Stage 1 checkpoint: {resume_checkpoint}")

    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTrainable parameters: {trainable_params:,} / {total_params:,} ({trainable_params/total_params*100:.2f}%)")

    # Load datasets
    print("\nLoading datasets...")
    train_dataset = load_instruction_dataset(config['dataset']['train_file'])
    val_dataset = load_instruction_dataset(config['dataset']['val_file'])

    # Training arguments
    print("\nSetting up training arguments...")
    training_args = TrainingArguments(
        output_dir=config['training']['output_dir'],
        num_train_epochs=config['training']['num_train_epochs'],
        per_device_train_batch_size=config['training']['per_device_train_batch_size'],
        per_device_eval_batch_size=config['training']['per_device_train_batch_size'],
        gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
        warmup_steps=config['training']['warmup_steps'],
        learning_rate=config['training']['learning_rate'],
        fp16=config['training']['fp16'],
        logging_steps=config['training']['logging_steps'],
        save_steps=config['training']['save_steps'],
        eval_steps=config['training']['eval_steps'],
        evaluation_strategy=config['training']['evaluation_strategy'],
        save_strategy=config['training']['save_strategy'],
        load_best_model_at_end=config['training']['load_best_model_at_end'],
        metric_for_best_model=config['training']['metric_for_best_model'],
        optim=config['training']['optim'],
        weight_decay=config['training']['weight_decay'],
        max_grad_norm=config['training']['max_grad_norm'],
        lr_scheduler_type="linear",
        seed=config['dataset']['seed'],
        save_total_limit=config['training']['save_total_limit'],
        report_to="wandb" if config['wandb']['enabled'] else "none",
    )

    # Create trainer
    print("\nCreating trainer...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        dataset_text_field="text",
        max_seq_length=config['model']['max_seq_length'],
        args=training_args,
        packing=False,  # Don't pack sequences for instruction tuning
    )
    print("✓ Trainer created")

    # Check GPU info
    if torch.cuda.is_available():
        print(f"\n🖥️  GPU: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # Training info
    total_steps = len(train_dataset) // (
        config['training']['per_device_train_batch_size'] *
        config['training']['gradient_accumulation_steps']
    ) * config['training']['num_train_epochs']

    print("\n" + "=" * 80)
    print(f"TRAINING CONFIGURATION - STAGE {stage}")
    print("=" * 80)
    print(f"Training samples:     {len(train_dataset)}")
    print(f"Validation samples:   {len(val_dataset)}")
    print(f"Epochs:              {config['training']['num_train_epochs']}")
    print(f"Batch size:          {config['training']['per_device_train_batch_size']}")
    print(f"Gradient accum:      {config['training']['gradient_accumulation_steps']}")
    print(f"Effective batch:     {config['training']['per_device_train_batch_size'] * config['training']['gradient_accumulation_steps']}")
    print(f"Total steps:         ~{total_steps}")
    print(f"Learning rate:       {config['training']['learning_rate']}")
    print(f"LoRA rank:           {config['lora']['r']}")
    print(f"Max seq length:      {config['model']['max_seq_length']}")
    if stage == 2 and resume_checkpoint:
        print(f"Resuming from:       {resume_checkpoint}")
    print("=" * 80)

    # Start training
    print(f"\n🚀 Starting Stage {stage} training...\n")

    # Stage 2: Resume from Stage 1 checkpoint
    if stage == 2 and resume_checkpoint:
        trainer.train(resume_from_checkpoint=resume_checkpoint)
    else:
        trainer.train()

    print("\n" + "=" * 80)
    print(f"✓ STAGE {stage} TRAINING COMPLETE")
    print("=" * 80)

    # Save final model
    output_dir = Path(config['training']['output_dir'])
    final_model_path = output_dir / "final_model"

    print(f"\nSaving final model to {final_model_path}...")
    model.save_pretrained(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    print("✓ Model saved")

    print("\n" + "=" * 80)
    print(f"STAGE {stage} TRAINING SESSION COMPLETE")
    print("=" * 80)
    print(f"\nModel saved to: {final_model_path}")
    print(f"Checkpoints in: {output_dir}")

    if stage == 1:
        print("\nNext steps:")
        print("1. Run Stage 2 training:")
        print("   python scripts/train_medical_ner_v3.py --config configs/medical_ner_re_config_v3_stage2.yaml --stage 2")
    else:
        print("\nNext steps:")
        print("1. Evaluate on test set:")
        print(f"   python scripts/evaluate_medical_ner.py --model {final_model_path} --test_data data/datasets/formatted/test.json")
        print("2. Compare results to v1/v2 baselines")

    return trainer, model, tokenizer


def main():
    parser = argparse.ArgumentParser(description="Train medical NER/RE model with v3 staged training")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to training configuration file",
    )
    parser.add_argument(
        "--stage",
        type=int,
        choices=[1, 2],
        required=True,
        help="Training stage (1=entities, 2=relations)",
    )

    args = parser.parse_args()

    # Train model
    trainer, model, tokenizer = train_medical_ner_v3(args.config, args.stage)

    return trainer, model, tokenizer


if __name__ == "__main__":
    main()
