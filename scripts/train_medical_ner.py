#!/usr/bin/env python3
"""Train medical NER/RE model with Unsloth

Usage:
    python scripts/train_medical_ner.py --config configs/medical_ner_re_config.yaml
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
sys.path.insert(0, str(project_root))

# Import MLflow configuration
from mlflow_config import setup_mlflow, get_git_commit_hash, log_system_info

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
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

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


def formatting_prompts_func(examples):
    """Format examples for training"""
    texts = []
    for text in examples["text"]:
        texts.append(text)
    return {"text": texts}


def train_medical_ner(config_path):
    """Main training function"""

    print("=" * 80)
    print("MEDICAL NER/RE MODEL TRAINING")
    print("=" * 80)

    # Load configuration
    print(f"\nLoading configuration from {config_path}...")
    config = load_config(config_path)
    print("✓ Configuration loaded")

    # Initialize MLflow tracking
    mlflow = setup_mlflow()
    run_name = f"train-{Path(config['training']['output_dir']).name}"
    mlflow.start_run(run_name=run_name)

    print(f"\n📊 MLflow run started: {run_name}")

    # Log git commit hash for reproducibility
    mlflow.set_tag("git_commit", get_git_commit_hash())
    mlflow.set_tag("stage", "training")

    # Log system information
    log_system_info(mlflow)

    # Log all hyperparameters
    mlflow.log_params({
        "model_name": config['model']['name'],
        "max_seq_length": config['model']['max_seq_length'],
        "load_in_4bit": config['model']['load_in_4bit'],
        "num_train_epochs": config['training']['num_train_epochs'],
        "per_device_train_batch_size": config['training']['per_device_train_batch_size'],
        "gradient_accumulation_steps": config['training']['gradient_accumulation_steps'],
        "learning_rate": config['training']['learning_rate'],
        "warmup_steps": config['training']['warmup_steps'],
        "weight_decay": config['training']['weight_decay'],
        "max_grad_norm": config['training']['max_grad_norm'],
        "lora_r": config['lora']['r'],
        "lora_alpha": config['lora']['lora_alpha'],
        "lora_dropout": config['lora']['lora_dropout'],
        "optim": config['training']['optim'],
        "seed": config['dataset']['seed'],
    })

    # Log config file as artifact
    mlflow.log_artifact(config_path, "configs")

    # Load model
    print(f"\nLoading model: {config['model']['name']}...")
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

    # Create data collator for completion-only training
    # This ensures we only compute loss on the output portion, not the instruction/input
    print("\nSetting up completion-only data collator...")
    response_template = "### Output:\n"
    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template,
        tokenizer=tokenizer,
        mlm=False,
    )
    print("✓ Data collator created (trains only on output after '### Output:')")

    # Create trainer
    print("\nCreating trainer...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        dataset_text_field="text",
        max_seq_length=config['model']['max_seq_length'],
        data_collator=collator,  # CRITICAL: Use completion-only collator
        args=training_args,
        packing=False,  # Don't pack sequences for instruction tuning
    )
    print("✓ Trainer created with completion-only training")

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
    print("TRAINING CONFIGURATION")
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
    print("=" * 80)

    # Start training
    print("\n🚀 Starting training...\n")

    trainer.train()

    print("\n" + "=" * 80)
    print("✓ TRAINING COMPLETE")
    print("=" * 80)

    # Log training metrics to MLflow
    train_history = trainer.state.log_history
    if train_history:
        # Get final metrics
        final_train_loss = None
        final_eval_loss = None

        for entry in reversed(train_history):
            if final_train_loss is None and 'loss' in entry:
                final_train_loss = entry['loss']
            if final_eval_loss is None and 'eval_loss' in entry:
                final_eval_loss = entry['eval_loss']
            if final_train_loss is not None and final_eval_loss is not None:
                break

        if final_train_loss is not None:
            mlflow.log_metric("final_train_loss", final_train_loss)
        if final_eval_loss is not None:
            mlflow.log_metric("final_eval_loss", final_eval_loss)

    # Save final model
    output_dir = Path(config['training']['output_dir'])
    final_model_path = output_dir / "final_model"

    print(f"\nSaving final model to {final_model_path}...")
    model.save_pretrained(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    print("✓ Model saved")

    # Log model artifacts to MLflow
    mlflow.log_artifact(str(final_model_path / "adapter_config.json"), "model")
    if (output_dir / "trainer_state.json").exists():
        mlflow.log_artifact(str(output_dir / "trainer_state.json"), "training")

    # End MLflow run
    mlflow.end_run()
    print("✓ MLflow run completed")

    print("\n" + "=" * 80)
    print("TRAINING SESSION COMPLETE")
    print("=" * 80)
    print(f"\nModel saved to: {final_model_path}")
    print(f"Checkpoints in: {output_dir}")
    print("\nNext steps:")
    print("1. Evaluate on test set")
    print("2. Test inference on sample clinical notes")
    print("3. Calculate F1 scores for entities and relations")

    return trainer, model, tokenizer


def main():
    parser = argparse.ArgumentParser(description="Train medical NER/RE model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/medical_ner_re_config.yaml",
        help="Path to training configuration file",
    )

    args = parser.parse_args()

    # Train model
    trainer, model, tokenizer = train_medical_ner(args.config)

    return trainer, model, tokenizer


if __name__ == "__main__":
    main()
