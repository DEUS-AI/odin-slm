# Odin SLM - Setup Summary

**Date**: 2026-02-08
**Status**: ✅ Complete

## System Configuration

### Hardware Analysis
- **GPU**: NVIDIA GeForce RTX 4090 Laptop GPU
- **VRAM**: 16,376 MiB (~16 GB)
- **CUDA**: Version 12.0 (V12.0.140)
- **Driver**: 550.163.01
- **OS**: Ubuntu Linux 6.14.0-33-generic
- **Python**: 3.12.8

### Optimization Profile
Your RTX 4090 Laptop GPU is excellent for SLM training:
- ✅ **1B-3B parameter models**: Full fine-tuning with 4-bit quantization
- ✅ **Up to 7B parameters**: LoRA/QLoRA fine-tuning
- ✅ **Batch size**: 2-8 (depending on model size and sequence length)
- ✅ **Sequence length**: Up to 4096 tokens with gradient checkpointing
- ✅ **Mixed precision**: FP16 training fully supported

## Project Structure Created

```
odin-slm/
├── CLAUDE.md                       # 📖 Comprehensive project documentation
├── README.md                       # 📄 Quick start guide
├── SETUP_SUMMARY.md               # 📋 This file
├── pyproject.toml                 # 📦 Python project configuration
├── .gitignore                     # 🚫 Git ignore rules
├── .python-version                # 🐍 Python version (3.12)
│
├── configs/
│   └── training_config.yaml       # ⚙️ Training configuration (RTX 4090 optimized)
│
├── data/
│   ├── raw/                       # 📥 Raw datasets
│   ├── processed/                 # 🔄 Preprocessed data
│   └── datasets/                  # 📊 Formatted training datasets
│
├── docs/
│   ├── Hypergraph_Integration_Analysis.docx
│   └── SLM_Training_Research_Report.docx
│
├── experiments/                   # 🧪 Training runs and checkpoints
│
├── notebooks/
│   └── 01_quickstart.ipynb       # 📓 Interactive tutorial
│
├── scripts/
│   └── setup.sh                  # 🔧 Automated setup script
│
├── src/odin_slm/
│   ├── __init__.py
│   ├── models/                   # 🤖 Model architectures
│   ├── data/                     # 📊 Data loading
│   ├── training/
│   │   ├── __init__.py
│   │   └── trainer.py           # 🎯 Main training loop (SLMTrainer)
│   ├── evaluation/              # 📈 Evaluation metrics
│   └── utils/
│       ├── __init__.py
│       └── gpu_info.py         # 🖥️ GPU utilities
│
└── tests/                       # 🧪 Unit tests
```

## Dependencies Installed

### Core ML Stack
- ✅ **Unsloth** (2024.12+): Optimized fine-tuning (2x faster, 50% less memory)
- ✅ **PyTorch** (2.5.0+): With CUDA 12.1 support
- ✅ **Transformers** (4.46.0+): Hugging Face models
- ✅ **PEFT** (0.13.0+): LoRA/QLoRA implementations
- ✅ **TRL** (0.12.0+): SFTTrainer for instruction tuning
- ✅ **bitsandbytes** (0.45.0+): 4-bit/8-bit quantization

### Data & Utilities
- ✅ **datasets**: Hugging Face datasets
- ✅ **accelerate**: Distributed training
- ✅ **wandb**: Experiment tracking
- ✅ **numpy**, **pandas**: Data manipulation
- ✅ **scikit-learn**: Metrics
- ✅ **matplotlib**, **seaborn**: Visualization

### Development Tools
- ✅ **jupyter**, **ipykernel**: Interactive notebooks
- ✅ **pytest**: Testing framework
- ✅ **black**, **ruff**: Code formatting
- ✅ **mypy**: Type checking

## Key Configuration Files

### [configs/training_config.yaml](configs/training_config.yaml)
Optimized for your RTX 4090:
- Model: `unsloth/llama-3.2-1b-instruct-bnb-4bit` (1B params, 4-bit)
- Batch size: 4 (with 4x gradient accumulation = effective batch 16)
- Sequence length: 2048 tokens
- LoRA rank: 16 (trainable params: ~16M vs 1B total)
- Mixed precision: FP16 enabled
- Optimizer: AdamW-8bit (memory efficient)
- Estimated VRAM usage: ~10-12 GB

## Quick Start Commands

### 1. Activate Environment
```bash
source .venv/bin/activate
```

### 2. Verify GPU Setup
```bash
# Using project utility
uv run python src/odin_slm/utils/gpu_info.py

# Or manually
nvidia-smi
```

### 3. Test Installation
```bash
# Open the quickstart notebook
uv run jupyter notebook notebooks/01_quickstart.ipynb

# Or test directly
uv run python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

### 4. Prepare Your Dataset
```python
from datasets import Dataset

# Your data format
data = {
    "text": [
        "Your training example 1...",
        "Your training example 2...",
        # ... more examples
    ]
}

dataset = Dataset.from_dict(data)
dataset.save_to_disk("data/datasets/my_dataset")
```

### 5. Train a Model
```bash
# Using the trainer (after preparing dataset)
uv run python -m odin_slm.training.trainer
```

## Memory Optimization Guide

Your 16GB VRAM can handle:

| Model Size | Quantization | Batch Size | Seq Len | Est. VRAM | Training Time* |
|------------|-------------|------------|---------|-----------|----------------|
| 1B params  | 4-bit       | 4-8        | 2048    | ~10 GB    | ~2-3 hrs/epoch |
| 3B params  | 4-bit       | 2-4        | 2048    | ~14 GB    | ~4-6 hrs/epoch |
| 7B params  | 4-bit       | 1-2        | 2048    | ~15 GB    | ~8-12 hrs/epoch|

*Approximate times for 10K examples with Unsloth optimizations

### If You Hit OOM (Out of Memory):
1. Reduce `per_device_train_batch_size` to 2 or 1
2. Increase `gradient_accumulation_steps` to maintain effective batch size
3. Reduce `max_seq_length` (2048 → 1024)
4. Ensure gradient checkpointing is enabled (it is by default)
5. Close other GPU applications

## Next Steps

### Immediate Tasks
1. ✅ Environment set up (DONE)
2. ✅ Project structure created (DONE)
3. ⏭️ Prepare your training dataset
4. ⏭️ Review and customize `configs/training_config.yaml`
5. ⏭️ Run first training experiment

### Dataset Preparation
For hypergraph integration tasks, consider:
- Knowledge graph triples
- Hypergraph relationship descriptions
- Multi-relational reasoning examples
- Graph traversal instructions
- Structured reasoning tasks

### Recommended Workflow
1. Start with a small dataset (1K examples) to test the pipeline
2. Use the quickstart notebook to verify model loading and inference
3. Run a short training (100 steps) to validate configuration
4. Scale up to full dataset and full training
5. Track experiments with WandB (optional but recommended)

### Model Selection
Current config uses **Llama 3.2 1B** (good starting point). Consider:
- **Phi-3.5 Mini (3.8B)**: Better reasoning, fits in 16GB with 4-bit
- **Mistral 7B**: Stronger performance, 4-bit required, batch size 1-2
- **Qwen 2.5 (1.5B/3B)**: Good multilingual support if needed

## Documentation

- **[CLAUDE.md](CLAUDE.md)**: Comprehensive guide (read this!)
  - Detailed system specs
  - Training strategies
  - Troubleshooting
  - Best practices

- **[README.md](README.md)**: Quick reference

- **[Training Config](configs/training_config.yaml)**: Adjust hyperparameters

- **[Quickstart Notebook](notebooks/01_quickstart.ipynb)**: Interactive tutorial

## Useful Resources

### Unsloth
- [GitHub](https://github.com/unslothai/unsloth)
- [Documentation](https://docs.unsloth.ai/)
- [Example Notebooks](https://github.com/unslothai/unsloth/tree/main/notebooks)

### Hugging Face
- [Transformers](https://huggingface.co/docs/transformers)
- [PEFT/LoRA](https://huggingface.co/docs/peft)
- [TRL](https://huggingface.co/docs/trl)
- [Datasets](https://huggingface.co/docs/datasets)

### Papers
- [QLoRA](https://arxiv.org/abs/2305.14314): Efficient 4-bit fine-tuning
- [LoRA](https://arxiv.org/abs/2106.09685): Low-rank adaptation
- [Unsloth](https://github.com/unslothai/unsloth#performance): Performance benchmarks

## Support

For issues or questions:
1. Check [CLAUDE.md](CLAUDE.md) troubleshooting section
2. Review Unsloth [GitHub issues](https://github.com/unslothai/unsloth/issues)
3. Consult Hugging Face [forums](https://discuss.huggingface.co/)

## Notes

- **CUDA 12.0** is installed and compatible with PyTorch 2.5.0+
- **Unsloth optimizations** provide ~2x speedup over standard fine-tuning
- **4-bit quantization** reduces memory by ~75% with minimal quality loss
- **LoRA** trains only 1-2% of parameters, enabling fast iteration
- **Gradient checkpointing** trades compute for memory (essential for 16GB)

---

**Setup completed successfully! 🚀**

Ready to train your first SLM. See [CLAUDE.md](CLAUDE.md) for detailed guidance.
