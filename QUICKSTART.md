# Odin SLM - Quick Start Guide

## Setup Complete! ✅

Your SLM training environment is ready with:
- **Unsloth 2025.3.3**: Fast fine-tuning framework
- **PyTorch 2.5.0**: With CUDA 12.4 support
- **Transformers 4.47.1**: Hugging Face models
- **RTX 4090 Laptop GPU**: 16GB VRAM optimized configuration

## Immediate Next Steps

### 1. Activate Environment

```bash
source .venv/bin/activate
```

### 2. Verify Installation

```bash
# Run verification script
python scripts/verify_setup.py

# Or check GPU manually
python src/odin_slm/utils/gpu_info.py
```

### 3. Try the Quickstart Notebook

```bash
jupyter notebook notebooks/01_quickstart.ipynb
```

This notebook shows you how to:
- Load a model with Unsloth
- Add LoRA adapters
- Test inference
- Prepare datasets

## Your First Training Run

### Step 1: Prepare Your Dataset

Create a dataset file in `data/datasets/`:

```python
from datasets import Dataset

# Example dataset
data = {
    "text": [
        "Hypergraphs extend traditional graphs by allowing edges to connect multiple vertices.",
        "Knowledge representation in AI often uses graph structures.",
        # Add your training examples...
    ]
}

dataset = Dataset.from_dict(data)
dataset.save_to_disk("data/datasets/my_first_dataset")
```

### Step 2: Review Configuration

Edit [configs/training_config.yaml](configs/training_config.yaml):

```yaml
model:
  name: "unsloth/llama-3.2-1b-instruct-bnb-4bit"  # 1B model (recommended start)

training:
  per_device_train_batch_size: 4  # Adjust based on memory
  num_train_epochs: 3
```

### Step 3: Train

```bash
# Recommended: Use the standalone script
uv run python train.py --dataset path/to/your/dataset

# Or programmatically:
python
>>> from odin_slm.training import SLMTrainer
>>> trainer = SLMTrainer()
>>> trainer.train("path/to/your/dataset")
```

## Important Notes

### IDE Python Interpreter

Your IDE may show warnings about packages not being installed. This is because it's using a different Python interpreter. To fix:

**VSCode:**
1. Press `Cmd/Ctrl + Shift + P`
2. Type "Python: Select Interpreter"
3. Choose `.venv/bin/python` from this project

**PyCharm:**
1. Go to Settings → Project → Python Interpreter
2. Click gear icon → Add
3. Select "Existing Environment" → Choose `.venv/bin/python`

### GPU Memory Management

Your RTX 4090 has 16GB VRAM. Recommended configurations:

| Model Size | Batch Size | Seq Length | Est. VRAM |
|------------|------------|------------|-----------|
| 1B params  | 4          | 2048       | ~10 GB    |
| 3B params  | 2          | 2048       | ~14 GB    |
| 7B params  | 1          | 2048       | ~15 GB    |

If you get OOM (Out of Memory) errors:
1. Reduce `per_device_train_batch_size` to 2 or 1
2. Increase `gradient_accumulation_steps` to maintain effective batch size
3. Reduce `max_seq_length` from 2048 to 1024

## Project Files

- **[CLAUDE.md](CLAUDE.md)**: Comprehensive documentation (READ THIS!)
- **[configs/training_config.yaml](configs/training_config.yaml)**: Training hyperparameters
- **[notebooks/01_quickstart.ipynb](notebooks/01_quickstart.ipynb)**: Interactive tutorial
- **[src/odin_slm/](src/odin_slm/)**: Source code modules

## Useful Commands

```bash
# Activate environment
source .venv/bin/activate

# Check GPU status
nvidia-smi

# Run GPU info utility
python src/odin_slm/utils/gpu_info.py

# Start Jupyter
jupyter notebook

# Install new package
uv add package-name

# Update all packages
uv sync --upgrade

# Run tests (when you add them)
pytest
```

## Learning Resources

### Unsloth
- [GitHub Repository](https://github.com/unslothai/unsloth)
- [Example Notebooks](https://github.com/unslothai/unsloth/tree/main/notebooks)
- [Documentation](https://docs.unsloth.ai/)

### Fine-tuning Guides
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [QLoRA Paper](https://arxiv.org/abs/2305.14314)
- [Hugging Face PEFT Docs](https://huggingface.co/docs/peft)

## Common Issues

### "CUDA out of memory"
→ Reduce batch size or sequence length in config

### "Module not found" errors
→ Ensure you've activated the virtual environment: `source .venv/bin/activate`

### Training is slow
→ Verify CUDA is being used: `python -c "import torch; print(torch.cuda.is_available())"`

### IDE shows package warnings
→ Select the correct Python interpreter (`.venv/bin/python`)

## Next Steps

1. ✅ Environment is set up
2. ⏭️ Read [CLAUDE.md](CLAUDE.md) for detailed documentation
3. ⏭️ Try the quickstart notebook
4. ⏭️ Prepare your hypergraph dataset
5. ⏭️ Run your first training experiment
6. ⏭️ Evaluate and iterate

## Get Help

- Check [CLAUDE.md](CLAUDE.md) troubleshooting section
- Review [Unsloth GitHub Issues](https://github.com/unslothai/unsloth/issues)
- Consult [Hugging Face Forums](https://discuss.huggingface.co/)

---

**Ready to train! 🚀**

For detailed information, see [CLAUDE.md](CLAUDE.md)
