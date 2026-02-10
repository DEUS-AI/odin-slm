# Training Setup Complete - Ready to Train!

**Status**: ✅ All setup complete | ⚠️ CUDA context needs reset
**Date**: 2026-02-08

---

## What's Ready

### ✅ Dataset (Complete)
- **Training set**: 4,000 formatted documents
- **Validation set**: 500 formatted documents
- **Test set**: 500 formatted documents
- **Location**: `data/datasets/formatted/`
- **Format**: Instruction-tuning format

### ✅ Training Configuration (Complete)
- **File**: [configs/medical_ner_re_config.yaml](configs/medical_ner_re_config.yaml)
- **Model**: Llama 3.2 1B (4-bit quantized)
- **Batch size**: 8 (effective: 16 with gradient accumulation)
- **Epochs**: 3
- **Max sequence**: 512 tokens (optimized for medical notes)
- **LoRA rank**: 16

### ✅ Training Script (Complete)
- **File**: [scripts/train_medical_ner.py](scripts/train_medical_ner.py)
- **Features**:
  - Automatic dataset loading
  - LoRA fine-tuning with Unsloth
  - Validation during training
  - Best model checkpoint saving
  - Full training metrics

---

## How to Start Training

### Option 1: After System Restart (Recommended)

1. **Restart your system** (to reset CUDA):
   ```bash
   # After restart
   cd /home/pablo/code/odin-slm
   source .venv/bin/activate
   ```

2. **Verify GPU is available**:
   ```bash
   nvidia-smi
   python -c "import torch; print('CUDA:', torch.cuda.is_available())"
   ```

3. **Launch training**:
   ```bash
   uv run python scripts/train_medical_ner.py --config configs/medical_ner_re_config.yaml
   ```

### Option 2: Reset NVIDIA Driver (No Restart)

```bash
# Requires sudo
sudo rmmod nvidia_uvm
sudo rmmod nvidia_drm
sudo rmmod nvidia_modeset
sudo rmmod nvidia
sudo modprobe nvidia
sudo modprobe nvidia_modeset
sudo modprobe nvidia_drm
sudo modprobe nvidia_uvm

# Then verify and train
nvidia-smi
uv run python scripts/train_medical_ner.py --config configs/medical_ner_re_config.yaml
```

### Option 3: Run in Background

```bash
# After CUDA is available again
nohup uv run python scripts/train_medical_ner.py \
    --config configs/medical_ner_re_config.yaml \
    > training_output.log 2>&1 &

# Monitor progress
tail -f training_output.log
```

---

## What to Expect During Training

### Training Overview
```
Dataset:
  Training samples:     4,000
  Validation samples:   500
  Test samples:         500

Configuration:
  Epochs:              3
  Batch size:          8
  Gradient accum:      2
  Effective batch:     16
  Total steps:         ~750
  Learning rate:       2.0e-4
  LoRA rank:           16
  Max seq length:      512

GPU Usage:
  Model (4-bit):       ~3-4 GB
  Training overhead:   ~4-5 GB
  Total estimate:      ~7-9 GB (well within 16GB limit)
```

### Training Time Estimate
- **Per epoch**: ~15-20 minutes (RTX 4090)
- **Total (3 epochs)**: ~45-60 minutes
- **With Unsloth**: 2x faster than standard fine-tuning!

### Expected Output
```
================================================================================
MEDICAL NER/RE MODEL TRAINING
================================================================================

Loading configuration from configs/medical_ner_re_config.yaml...
✓ Configuration loaded

Loading model: unsloth/llama-3.2-1b-instruct-bnb-4bit...
✓ Model loaded

Adding LoRA adapters...
✓ LoRA adapters added

Trainable parameters: 16,777,216 / 1,235,894,272 (1.36%)

Loading datasets...
Loading dataset from data/datasets/formatted/train.json...
✓ Loaded 4000 examples
Loading dataset from data/datasets/formatted/val.json...
✓ Loaded 500 examples

Setting up training arguments...

Creating trainer...
✓ Trainer created

🖥️  GPU: NVIDIA GeForce RTX 4090 Laptop GPU
   VRAM: 16.85 GB

================================================================================
TRAINING CONFIGURATION
================================================================================
Training samples:     4000
Validation samples:   500
Epochs:              3
Batch size:          8
Gradient accum:      2
Effective batch:     16
Total steps:         ~750
Learning rate:       2.0e-4
LoRA rank:           16
Max seq length:      512
================================================================================

🚀 Starting training...

[Progress bars and loss metrics will appear here]

✓ TRAINING COMPLETE
```

---

## Monitoring Training

### Check Progress
```bash
# If running in terminal
# Watch the progress bars and loss values

# If running in background
tail -f training_output.log

# Check GPU usage
watch -n 1 nvidia-smi
```

### Key Metrics to Watch
1. **Training Loss**: Should decrease (target: < 0.5)
2. **Validation Loss**: Should decrease and stay close to training loss
3. **GPU Memory**: Should stay under 10GB
4. **Training Speed**: ~2-3 steps/second with Unsloth

---

## After Training Completes

### 1. Find Your Model
```bash
# Best checkpoint (based on validation loss)
experiments/medical_ner_re/checkpoint-best/

# Final model
experiments/medical_ner_re/final_model/
```

### 2. Test Inference
```python
from unsloth import FastLanguageModel

# Load trained model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="experiments/medical_ner_re/final_model",
    max_seq_length=512,
    dtype=None,
    load_in_4bit=True,
)

FastLanguageModel.for_inference(model)

# Test on clinical note
test_text = """### Instruction:
Extract all medical entities and their relations from the following clinical text.

### Input:
Patient presents with chest pain and shortness of breath. Blood pressure 140/90.
Prescribed aspirin and scheduled for cardiac catheterization.

### Output:
"""

inputs = tokenizer(test_text, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=200)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### 3. Evaluate Performance
```bash
# Create evaluation script (to be done)
python scripts/evaluate_medical_ner.py \
    --model experiments/medical_ner_re/final_model \
    --test_data data/datasets/formatted/test.json
```

### 4. Calculate F1 Scores
```python
from odin_slm.data.evaluator import NERREvaluator

# Run predictions on test set
# predictions = model.predict(test_data)

# Evaluate entities
evaluator = NERREvaluator(matching_mode="exact")
results = evaluator.evaluate_entities_per_type(predictions, gold)

# Print results
NERREvaluator.print_results(results, "Test Set Performance")

# Target scores:
# Entity F1: ≥ 85%
# Relation F1: ≥ 75%
```

---

## Troubleshooting

### CUDA Out of Memory
If you get OOM errors:
1. Reduce batch size: `per_device_train_batch_size: 4`
2. Increase gradient accumulation: `gradient_accumulation_steps: 4`
3. Reduce sequence length: `max_seq_length: 256`

### Training Too Slow
- Verify Unsloth is loaded (should see "🦥 Unsloth" messages)
- Check GPU utilization: `nvidia-smi` (should be 80-100%)
- Ensure FP16 is enabled in config

### Validation Loss Not Decreasing
- Increase learning rate: `learning_rate: 3.0e-4`
- More epochs: `num_train_epochs: 5`
- Check for data issues: Review formatted dataset

### Model Not Learning
- Verify dataset format is correct
- Check if examples are properly formatted
- Ensure instruction, input, output structure is clear

---

## What We've Accomplished

### Complete Pipeline ✅
1. ✅ Research: 30+ sources, SOTA benchmarks
2. ✅ Synthetic Generation: 5,000 medical documents
3. ✅ Data Analysis: Quality score 76.5/100
4. ✅ Data Formatting: Instruction-tuning format
5. ✅ Train/Val/Test Splits: 80/10/10
6. ✅ Training Configuration: Optimized for RTX 4090
7. ✅ Training Script: Full pipeline with Unsloth
8. ⏭️ Training: Ready to launch (after CUDA reset)

### Files Created
```
odin-slm/
├── data/datasets/
│   ├── medical_ner_re_train.json (5,000 docs)
│   ├── medical_ner_re_test.json (100 docs)
│   └── formatted/
│       ├── train.json (4,000 docs)
│       ├── val.json (500 docs)
│       └── test.json (500 docs)
├── configs/
│   └── medical_ner_re_config.yaml
├── scripts/
│   ├── generate_medical_dataset.py
│   ├── inspect_dataset.py
│   ├── analyze_dataset_quality.py
│   ├── format_for_training.py
│   └── train_medical_ner.py
└── docs/
    ├── Medical_NER_RE_Research.md
    ├── Medical_NER_RE_QuickStart.md
    └── DATASET_ANALYSIS_SUMMARY.md
```

---

## Summary

**Status**: Everything is ready for training! Just need to reset CUDA context.

**Quick Start After Reboot**:
```bash
cd /home/pablo/code/odin-slm
source .venv/bin/activate
uv run python scripts/train_medical_ner.py --config configs/medical_ner_re_config.yaml
```

**Training Time**: ~45-60 minutes for 3 epochs
**Expected F1**: 60-70% (baseline), target 85% entities, 75% relations

**Next Steps**:
1. Reset CUDA (restart or reload driver)
2. Launch training
3. Monitor progress
4. Evaluate on test set
5. Iterate and improve

---

**You're all set! Just need a quick system restart or CUDA reset to begin training.** 🚀
