# Odin SLM

> Small Language Model Training with Unsloth - Optimized for Hypergraph Integration

## Quick Start

```bash
# Install dependencies
uv sync

# Activate virtual environment
source .venv/bin/activate

# Check GPU configuration
uv run python src/odin_slm/utils/gpu_info.py

# Start training (after preparing your dataset)
uv run python train.py --dataset your-dataset-name
```

## System Requirements

- NVIDIA GPU with 16GB+ VRAM (Optimized for RTX 4090)
- CUDA 12.0+
- Python 3.12+
- Ubuntu Linux (or compatible)

## Key Features

- **Fast Training**: 2x faster with Unsloth optimizations
- **Memory Efficient**: 4-bit quantization + LoRA for 16GB GPUs
- **Production Ready**: Structured project with best practices
- **Hypergraph Focus**: Specialized for knowledge representation tasks

## Documentation

See [CLAUDE.md](CLAUDE.md) for comprehensive documentation including:
- Detailed system specifications
- Training configuration guide
- Memory optimization strategies
- Troubleshooting tips
- Development workflow

## Project Structure

```
odin-slm/
├── configs/           # Training configurations
├── data/             # Datasets
├── experiments/      # Training runs
├── src/odin_slm/    # Source code
└── notebooks/        # Jupyter notebooks
```

## Tech Stack

- **Unsloth**: Efficient fine-tuning
- **PyTorch**: Deep learning framework
- **Transformers**: Model architectures
- **PEFT/LoRA**: Parameter-efficient training
- **UV**: Fast Python package management

---

For detailed documentation, see [CLAUDE.md](CLAUDE.md)
