"""Training script for SLM with Unsloth"""

import warnings

# Suppress SyntaxWarnings from unsloth_zoo library (Python 3.12 compatibility)
# These are harmless - unsloth uses "\s" instead of r"\s" in regex patterns
warnings.filterwarnings("ignore", category=SyntaxWarning, module="unsloth_zoo")
warnings.filterwarnings("ignore", category=SyntaxWarning, module="unsloth")

import torch
from unsloth import FastLanguageModel
from transformers import TrainingArguments
from trl import SFTTrainer
from datasets import load_dataset
import yaml
from pathlib import Path


class SLMTrainer:
    """Trainer for Small Language Models using Unsloth"""

    def __init__(self, config_path: str = "configs/training_config.yaml"):
        """Initialize trainer with configuration"""
        self.config = self._load_config(config_path)
        self.model = None
        self.tokenizer = None

    def _load_config(self, config_path: str) -> dict:
        """Load training configuration from YAML file"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def load_model(self):
        """Load and prepare model with Unsloth optimizations"""
        model_config = self.config['model']

        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_config['name'],
            max_seq_length=model_config['max_seq_length'],
            dtype=None,  # Auto-detect
            load_in_4bit=model_config['load_in_4bit'],
        )

        # Add LoRA adapters
        lora_config = self.config['lora']
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=lora_config['r'],
            target_modules=lora_config['target_modules'],
            lora_alpha=lora_config['lora_alpha'],
            lora_dropout=lora_config['lora_dropout'],
            bias=lora_config['bias'],
            use_gradient_checkpointing="unsloth",  # Unsloth's optimized checkpointing
            random_state=self.config['dataset']['seed'],
        )

        return self.model, self.tokenizer

    def prepare_dataset(self, dataset_name: str):
        """Load and prepare dataset for training"""
        # This is a template - adapt to your specific dataset
        dataset = load_dataset(dataset_name, split="train")

        # Split into train/val
        split_config = self.config['dataset']
        dataset = dataset.train_test_split(
            test_size=1 - split_config['train_split'],
            seed=split_config['seed']
        )

        return dataset['train'], dataset['test']

    def create_trainer(self, train_dataset, eval_dataset=None):
        """Create SFTTrainer with Unsloth optimizations"""
        train_config = self.config['training']

        training_args = TrainingArguments(
            output_dir=train_config['output_dir'],
            num_train_epochs=train_config['num_train_epochs'],
            per_device_train_batch_size=train_config['per_device_train_batch_size'],
            gradient_accumulation_steps=train_config['gradient_accumulation_steps'],
            warmup_steps=train_config['warmup_steps'],
            learning_rate=train_config['learning_rate'],
            fp16=train_config['fp16'],
            logging_steps=train_config['logging_steps'],
            save_steps=train_config['save_steps'],
            eval_steps=train_config.get('eval_steps', 500),
            optim=train_config['optim'],
            weight_decay=train_config['weight_decay'],
            lr_scheduler_type="linear",
            seed=self.config['dataset']['seed'],
            report_to="wandb" if self.config['wandb']['enabled'] else "none",
        )

        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            dataset_text_field="text",  # Adjust based on your dataset
            max_seq_length=self.config['model']['max_seq_length'],
            args=training_args,
        )

        return trainer

    def train(self, dataset_name: str):
        """Main training loop"""
        print("Loading model...")
        self.load_model()

        print("Preparing dataset...")
        train_dataset, eval_dataset = self.prepare_dataset(dataset_name)

        print("Creating trainer...")
        trainer = self.create_trainer(train_dataset, eval_dataset)

        print("Starting training...")
        trainer.train()

        return trainer


if __name__ == "__main__":
    # Example usage
    trainer = SLMTrainer()
    # trainer.train("your-dataset-name")
