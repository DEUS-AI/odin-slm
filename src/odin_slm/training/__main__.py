"""Entry point for running the trainer as a module with: python -m odin_slm.training"""

import warnings
import sys

# Suppress SyntaxWarnings from unsloth_zoo library (Python 3.12 compatibility)
# These warnings are harmless - unsloth uses "\s" instead of r"\s" in regex patterns
warnings.filterwarnings("ignore", category=SyntaxWarning, module="unsloth_zoo")
warnings.filterwarnings("ignore", category=SyntaxWarning, module="unsloth")

from .trainer import SLMTrainer


def main():
    """Main entry point for training"""
    print("=" * 60)
    print("Odin SLM Training")
    print("=" * 60)
    print()

    # Example usage - customize as needed
    print("Initializing trainer...")
    trainer = SLMTrainer()

    print()
    print("Configuration loaded from: configs/training_config.yaml")
    print()
    print("To train a model, prepare your dataset and call:")
    print("  trainer.train('your-dataset-name')")
    print()
    print("Or use the trainer programmatically:")
    print("  from odin_slm.training import SLMTrainer")
    print("  trainer = SLMTrainer()")
    print("  trainer.train('dataset-name')")
    print()


if __name__ == "__main__":
    main()
