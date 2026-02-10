# Odin SLM - Small Language Model Training Project

## Overview

Odin SLM is a project for training Small Language Models (SLMs) using Unsloth, optimized for hypergraph integration and advanced knowledge representation. This project leverages state-of-the-art efficient fine-tuning techniques to train compact, specialized language models on consumer hardware.

## System Specifications

### Hardware Configuration
- **GPU**: NVIDIA GeForce RTX 4090 Laptop GPU
- **VRAM**: 16,376 MiB (≈16 GB)
- **CUDA Version**: 12.0 (V12.0.140)
- **Driver Version**: 550.163.01
- **OS**: Ubuntu Linux (Kernel 6.14.0-33-generic)
- **Python**: 3.12.8

### Optimization Profile
The RTX 4090 Laptop GPU with 16GB VRAM is well-suited for:
- Training models up to 3B parameters with 4-bit quantization
- Fine-tuning models up to 7B parameters with LoRA/QLoRA
- Batch sizes of 2-8 depending on sequence length and model size
- Sequence lengths up to 4096 tokens with gradient checkpointing

## Project Structure

```
odin-slm/
├── CLAUDE.md                    # This file - project documentation
├── pyproject.toml              # Python project configuration
├── configs/                    # Training configurations
│   └── training_config.yaml   # Main training config (RTX 4090 optimized)
├── data/                       # Dataset storage
│   ├── raw/                   # Raw, unprocessed data
│   ├── processed/             # Preprocessed datasets
│   └── datasets/              # Formatted training datasets
├── docs/                       # Project documentation
│   ├── Hypergraph_Integration_Analysis.docx
│   └── SLM_Training_Research_Report.docx
├── experiments/                # Training runs and checkpoints
├── notebooks/                  # Jupyter notebooks for experimentation
├── src/odin_slm/              # Main source code
│   ├── __init__.py
│   ├── models/                # Model architectures and configs
│   ├── data/                  # Data loading and preprocessing
│   ├── training/              # Training scripts
│   │   ├── __init__.py
│   │   └── trainer.py         # Main training loop
│   ├── evaluation/            # Evaluation metrics and scripts
│   └── utils/                 # Utility functions
│       ├── __init__.py
│       └── gpu_info.py        # GPU information utilities
└── tests/                     # Unit tests
```

## Technology Stack

### Core Dependencies
- **Unsloth**: Fast and memory-efficient LLM fine-tuning (2x faster, 50% less memory)
- **PyTorch**: 2.5.0+ with CUDA 12.1 support
- **Transformers**: Hugging Face transformers library
- **PEFT**: Parameter-Efficient Fine-Tuning (LoRA, QLoRA)
- **TRL**: Transformer Reinforcement Learning (SFTTrainer)
- **bitsandbytes**: 8-bit and 4-bit quantization

### Supporting Libraries
- **datasets**: Hugging Face datasets library
- **accelerate**: Distributed training and mixed precision
- **wandb**: Experiment tracking and visualization
- **numpy**, **pandas**: Data manipulation
- **scikit-learn**: Metrics and preprocessing

## Getting Started

### 1. Installation

Install dependencies using UV (ultra-fast Python package manager):

```bash
# Install all dependencies
uv sync

# Or install with development dependencies
uv sync --extra dev

# Activate the virtual environment
source .venv/bin/activate
```

### 2. Verify GPU Setup

Check your GPU configuration:

```bash
# Using the provided utility
uv run python src/odin_slm/utils/gpu_info.py

# Or manually
nvidia-smi
```

Expected output should show:
- RTX 4090 Laptop GPU
- ~16GB total memory
- CUDA 12.0

### 3. Configure Training

Edit [configs/training_config.yaml](configs/training_config.yaml) to adjust:
- Model selection and size
- Batch size and gradient accumulation
- Learning rate and optimization settings
- LoRA hyperparameters
- Dataset splits

**Current Recommended Settings for RTX 4090 (16GB):**
- Model: `unsloth/llama-3.2-1b-instruct-bnb-4bit` (1B parameters)
- Batch size: 4 with gradient accumulation steps: 4
- Sequence length: 2048 tokens
- 4-bit quantization enabled
- LoRA rank: 16

### 4. Prepare Your Dataset

Datasets should be in Hugging Face format. Example structure:

```python
from datasets import Dataset

# Your data should have a 'text' field (or customize in trainer.py)
data = {
    "text": [
        "Example training text 1",
        "Example training text 2",
        # ... more examples
    ]
}

dataset = Dataset.from_dict(data)
dataset.save_to_disk("data/datasets/my_dataset")
```

### 5. Train a Model

```bash
# Method 1: Using the standalone script (recommended)
uv run python train.py --dataset your-dataset-name

# Method 2: Using module execution
uv run python -m odin_slm.training

# Method 3: Programmatically
python
>>> from odin_slm.training import SLMTrainer
>>> trainer = SLMTrainer(config_path="configs/training_config.yaml")
>>> trainer.train("your-dataset-name")
```

**Note**: All syntax warnings from the unsloth library are automatically suppressed. See [docs/WARNINGS_EXPLAINED.md](docs/WARNINGS_EXPLAINED.md) for details.

## Training Configuration Details

### Memory Optimization Strategies

The project uses several techniques to maximize training efficiency on 16GB VRAM:

1. **4-bit Quantization (QLoRA)**: Reduces model memory footprint by 75%
2. **Gradient Checkpointing**: Trades compute for memory, enabling larger batches
3. **LoRA Adapters**: Only trains 1-2% of parameters (16M vs 1B)
4. **Mixed Precision (FP16)**: Reduces memory and speeds up computation
5. **8-bit Optimizers**: AdamW-8bit reduces optimizer state memory by 75%

### Scaling Guidelines

| Model Size | Quantization | Max Batch Size | Seq Length | Est. VRAM |
|------------|-------------|----------------|------------|-----------|
| 1B params  | 4-bit       | 4-8            | 2048       | ~10 GB    |
| 3B params  | 4-bit       | 2-4            | 2048       | ~14 GB    |
| 7B params  | 4-bit       | 1-2            | 2048       | ~15 GB    |
| 1B params  | 4-bit       | 2-4            | 4096       | ~14 GB    |

*Note: Actual memory usage varies based on gradient accumulation and other factors*

## Unsloth Advantages

Unsloth provides significant performance improvements over standard fine-tuning:

- **2x Faster Training**: Optimized CUDA kernels
- **50% Less Memory**: Custom implementations of attention and MLP layers
- **No Accuracy Loss**: Numerically equivalent to standard training
- **Easy Integration**: Drop-in replacement for Hugging Face Trainer

## Experiment Tracking

### Weights & Biases (WandB)

To enable experiment tracking:

1. Sign up at [wandb.ai](https://wandb.ai)
2. Login: `wandb login`
3. Update [configs/training_config.yaml](configs/training_config.yaml):
   ```yaml
   wandb:
     project: "odin-slm"
     entity: "your-username"
     enabled: true
   ```

WandB will track:
- Training/validation loss
- Learning rate schedule
- GPU utilization
- Model hyperparameters
- System metrics

## Development Workflow

### Running Tests

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=src/odin_slm
```

### Code Formatting

```bash
# Format code with Black
uv run black src/

# Lint with Ruff
uv run ruff check src/

# Type checking with mypy
uv run mypy src/
```

### Jupyter Notebooks

```bash
# Start Jupyter server
uv run jupyter notebook

# Or use JupyterLab
uv run jupyter lab
```

## Project Goals

### Primary Objectives

1. **Hypergraph Integration**: Train SLMs to understand and reason over hypergraph structures
2. **Knowledge Representation**: Develop models specialized in knowledge graph tasks
3. **Efficient Training**: Demonstrate effective SLM training on consumer hardware
4. **Reproducibility**: Create a reproducible pipeline for SLM fine-tuning

### Research Questions

- How effectively can SLMs (1B-3B params) learn hypergraph relationships?
- What is the optimal model size vs. performance trade-off for hypergraph tasks?
- Can specialized SLMs outperform general LLMs on domain-specific tasks?

## Next Steps

1. **Data Preparation**
   - [ ] Collect and preprocess hypergraph dataset
   - [ ] Create training/validation splits
   - [ ] Format data for instruction tuning

2. **Model Selection**
   - [ ] Benchmark different base models (Llama, Phi, Mistral)
   - [ ] Determine optimal model size for task complexity
   - [ ] Test different quantization strategies

3. **Training Pipeline**
   - [ ] Implement custom evaluation metrics for hypergraph tasks
   - [ ] Set up automated hyperparameter tuning
   - [ ] Create checkpoint management system

4. **Evaluation**
   - [ ] Define task-specific benchmarks
   - [ ] Compare against baseline models
   - [ ] Analyze failure modes and limitations

## Troubleshooting

### CUDA Out of Memory

If you encounter OOM errors:
1. Reduce `per_device_train_batch_size` in config
2. Increase `gradient_accumulation_steps` to maintain effective batch size
3. Reduce `max_seq_length` (2048 → 1024)
4. Enable gradient checkpointing (should be on by default)
5. Close other GPU-intensive applications

### Slow Training

To improve training speed:
1. Ensure CUDA 12.0 is properly installed
2. Verify Unsloth is using optimized kernels (check logs)
3. Consider reducing sequence length if not needed
4. Use flash attention if available for your model
5. Monitor GPU utilization with `nvidia-smi -l 1`

### Import Errors

If modules aren't found:
```bash
# Ensure virtual environment is activated
source .venv/bin/activate

# Reinstall dependencies
uv sync --reinstall

# Install package in editable mode
uv pip install -e .
```

## Resources

### Documentation
- [Unsloth Documentation](https://github.com/unslothai/unsloth)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [PEFT Documentation](https://huggingface.co/docs/peft)
- [TRL Documentation](https://huggingface.co/docs/trl)

### Tutorials
- [Unsloth Notebooks](https://github.com/unslothai/unsloth/tree/main/notebooks)
- [QLoRA Paper](https://arxiv.org/abs/2305.14314)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)

## Contributing

This is a research project. Contributions and suggestions are welcome:
- Document experiments in `experiments/` with descriptive names
- Add reusable utilities to `src/odin_slm/utils/`
- Update this CLAUDE.md with new findings and best practices

## License

[Specify license here]

## Contact

[Specify contact information or team members]

---

**Last Updated**: 2026-02-08
**CUDA Version**: 12.0
**Python Version**: 3.12.8
**Primary GPU**: NVIDIA RTX 4090 Laptop (16GB)
