#!/usr/bin/env python3
"""
Standalone training script for Odin SLM

Usage:
    python train.py                    # Use default config
    python train.py --config custom.yaml  # Use custom config
"""

import warnings
import sys
import argparse

# Suppress SyntaxWarnings from unsloth_zoo (Python 3.12 compatibility)
warnings.filterwarnings("ignore", category=SyntaxWarning, module="unsloth_zoo")
warnings.filterwarnings("ignore", category=SyntaxWarning, module="unsloth")

from src.odin_slm.training import SLMTrainer


def main():
    parser = argparse.ArgumentParser(description="Train SLM with Unsloth")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/training_config.yaml",
        help="Path to training configuration file",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="Dataset name or path (required for training)",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("Odin SLM Training")
    print("=" * 70)
    print()

    # Initialize trainer
    print(f"Loading configuration from: {args.config}")
    trainer = SLMTrainer(config_path=args.config)
    print("✓ Configuration loaded")
    print()

    if args.dataset:
        print(f"Training on dataset: {args.dataset}")
        print()
        trainer.train(args.dataset)
    else:
        print("No dataset specified. To train a model, provide --dataset argument:")
        print(f"  python {sys.argv[0]} --dataset your-dataset-name")
        print()
        print("Or use the trainer programmatically:")
        print("  from odin_slm.training import SLMTrainer")
        print("  trainer = SLMTrainer()")
        print("  trainer.train('your-dataset-name')")
        print()


if __name__ == "__main__":
    main()
