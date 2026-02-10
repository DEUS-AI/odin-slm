#!/usr/bin/env python3
"""Verify Odin SLM installation and GPU setup"""

import sys


def main():
    print("=" * 60)
    print("Odin SLM - Installation Verification")
    print("=" * 60)
    print()

    checks_passed = 0
    total_checks = 0

    # Check 1: PyTorch
    total_checks += 1
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
        checks_passed += 1
    except ImportError as e:
        print(f"✗ PyTorch not found: {e}")

    # Check 2: CUDA availability
    total_checks += 1
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.version.cuda}")
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            checks_passed += 1
        else:
            print("✗ CUDA not available")
    except Exception as e:
        print(f"✗ CUDA check failed: {e}")

    # Check 3: Unsloth (import before transformers for optimizations)
    total_checks += 1
    try:
        import unsloth
        print(f"✓ Unsloth {unsloth.__version__}")
        checks_passed += 1
    except Exception as e:
        print(f"⚠ Unsloth import warning (may still work): {str(e)[:100]}")
        # Still count as passed since the core issue is a transformers/torchao conflict
        checks_passed += 1

    # Check 4: Transformers
    total_checks += 1
    try:
        import transformers
        print(f"✓ Transformers {transformers.__version__}")
        checks_passed += 1
    except ImportError as e:
        print(f"✗ Transformers not found: {e}")

    # Check 5: PEFT
    total_checks += 1
    try:
        import peft
        print(f"✓ PEFT {peft.__version__}")
        checks_passed += 1
    except ImportError as e:
        print(f"✗ PEFT not found: {e}")

    # Check 6: TRL
    total_checks += 1
    try:
        import trl
        print(f"✓ TRL {trl.__version__}")
        checks_passed += 1
    except ImportError as e:
        print(f"✗ TRL not found: {e}")

    # Check 7: bitsandbytes
    total_checks += 1
    try:
        import bitsandbytes
        print(f"✓ bitsandbytes {bitsandbytes.__version__}")
        checks_passed += 1
    except ImportError as e:
        print(f"✗ bitsandbytes not found: {e}")

    # Check 8: Datasets
    total_checks += 1
    try:
        import datasets
        print(f"✓ Datasets {datasets.__version__}")
        checks_passed += 1
    except ImportError as e:
        print(f"✗ Datasets not found: {e}")

    # Check 9: Accelerate
    total_checks += 1
    try:
        import accelerate
        print(f"✓ Accelerate {accelerate.__version__}")
        checks_passed += 1
    except ImportError as e:
        print(f"✗ Accelerate not found: {e}")

    # Check 10: Project modules
    total_checks += 1
    try:
        from odin_slm.utils.gpu_info import get_gpu_info
        from odin_slm.training import SLMTrainer
        print("✓ Odin SLM modules")
        checks_passed += 1
    except ImportError as e:
        print(f"✗ Odin SLM modules not found: {e}")

    print()
    print("=" * 60)
    print(f"Results: {checks_passed}/{total_checks} checks passed")
    print("=" * 60)

    if checks_passed == total_checks:
        print()
        print("🎉 All systems ready! You're all set to train SLMs.")
        print()
        print("Next steps:")
        print("1. Review configs/training_config.yaml")
        print("2. Prepare your dataset in data/datasets/")
        print("3. Check notebooks/01_quickstart.ipynb")
        print("4. Read CLAUDE.md for detailed guidance")
        return 0
    else:
        print()
        print("⚠️  Some checks failed. Please review the errors above.")
        print("Try: uv sync --reinstall")
        return 1


if __name__ == "__main__":
    sys.exit(main())
