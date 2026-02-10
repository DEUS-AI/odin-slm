# Experiment Tracking Implementation Plan

## Overview
Implement comprehensive experiment tracking to compare training runs and manage model versions.

## Phase 1: Weights & Biases Integration (Recommended First)

### Step 1: Setup W&B
```bash
# Install
uv pip install wandb

# Login (one-time)
wandb login
```

### Step 2: Enable in Existing Configs
Update all config files to enable W&B:
```yaml
wandb:
  project: "odin-slm-medical-ner"
  name: "mistral-7b-medical-ner-v4"  # Unique per run
  entity: null  # Or your W&B username/team
  enabled: true  # Change from false to true
```

### Step 3: Track Custom Metrics
Add to evaluation script (`scripts/evaluate_medical_ner.py`):
```python
import wandb

# After evaluation
if wandb.run is not None:
    wandb.log({
        "test/entity_f1": entity_f1,
        "test/relation_f1": relation_f1,
        "test/entity_precision": entity_precision,
        "test/entity_recall": entity_recall,
        "test/relation_precision": relation_precision,
        "test/relation_recall": relation_recall,
    })
```

### Step 4: Log Predictions (Optional)
```python
# Create W&B table with predictions
table = wandb.Table(columns=["text", "true_entities", "pred_entities", "true_relations", "pred_relations"])
for example in test_samples[:100]:  # First 100 examples
    table.add_data(
        example["text"],
        example["true_entities"],
        example["pred_entities"],
        example["true_relations"],
        example["pred_relations"]
    )
wandb.log({"predictions": table})
```

## Phase 2: MLflow (Optional, for Self-Hosted)

### When to Use MLflow
- Need offline tracking
- Want self-hosted solution
- Building production ML platform
- Need model registry with stage transitions (Staging → Production)

### Setup
```bash
# Install
uv pip install mlflow

# Start server (in separate terminal)
mlflow server --host 0.0.0.0 --port 5000
```

### Integration
```python
import mlflow

# In training script
with mlflow.start_run(run_name="mistral-7b-v4"):
    # Log hyperparameters
    mlflow.log_params({
        "model": config['model']['name'],
        "epochs": config['training']['num_train_epochs'],
        "lr": config['training']['learning_rate'],
        "lora_r": config['lora']['r'],
    })

    # Train model
    trainer.train()

    # Log metrics
    mlflow.log_metrics({
        "train_loss": trainer.state.log_history[-1]['loss'],
        "entity_f1": entity_f1,
        "relation_f1": relation_f1,
    })

    # Log model
    mlflow.transformers.log_model(model, "model")
```

## Phase 3: Hugging Face Hub (For Model Sharing)

### Use Cases
- Share best model publicly
- Enable easy inference
- Create model card for documentation
- Deploy with Inference API

### Push Model to Hub
```python
from huggingface_hub import HfApi

# After training best model
model.push_to_hub("your-username/mistral-7b-medical-ner-v4")
tokenizer.push_to_hub("your-username/mistral-7b-medical-ner-v4")

# Or via CLI
huggingface-cli login
python scripts/push_to_hub.py --model experiments/medical_ner_re_v4/final_model
```

### Create Model Card
```markdown
---
language: en
license: apache-2.0
tags:
- medical
- ner
- relation-extraction
- mistral
- unsloth
datasets:
- custom-medical-ner-re
metrics:
- f1
---

# Mistral 7B Medical NER/RE v4

Fine-tuned Mistral 7B for medical Named Entity Recognition and Relation Extraction.

## Performance

- **Entity F1**: 90.5%
- **Relation F1**: 52.3%

## Training

- Base model: Mistral 7B
- Fine-tuning: LoRA (rank 16)
- Epochs: 5
- Dataset: Synthetic medical clinical notes

## Usage

\`\`\`python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("your-username/mistral-7b-medical-ner-v4")
model = AutoModelForCausalLM.from_pretrained("your-username/mistral-7b-medical-ner-v4")

# Your inference code
\`\`\`
```

## Comparison: What to Use When

| Tool | Use Case | When to Use |
|------|----------|-------------|
| **W&B** | Experiment tracking, hyperparameter tuning | During active development, comparing runs |
| **MLflow** | Self-hosted tracking, model registry | Production pipelines, need offline tracking |
| **HF Hub** | Model sharing, deployment | Final model ready to share/deploy |

## Recommended Setup for Your Project

### Immediate (Phase 1)
1. ✅ Enable W&B in configs
2. ✅ Run `wandb login`
3. ✅ Re-train v4 with W&B enabled (or log retroactively)
4. ✅ Add evaluation metrics to W&B

### Short Term (Phase 2 - Optional)
- Add MLflow if you want self-hosted tracking
- Useful if building internal ML platform

### When Ready to Share (Phase 3)
- Push best model to Hugging Face Hub
- Create comprehensive model card
- Enable inference API for demos

## Benefits

**With W&B:**
- See all v1, v2, v3, v4 runs in one dashboard
- Compare hyperparameters side-by-side
- Track which config produced best results
- Visualize training curves (loss, learning rate)
- Monitor GPU/memory usage
- Share results with team/stakeholders

**With HF Hub:**
- Easy model distribution
- One-line inference for users
- Professional model documentation
- Community visibility

## Implementation Checklist

- [ ] Install wandb: `uv pip install wandb`
- [ ] Login: `wandb login`
- [ ] Update all configs: set `wandb.enabled: true`
- [ ] Add evaluation logging to `scripts/evaluate_medical_ner.py`
- [ ] Re-run v4 training with W&B enabled
- [ ] Create W&B report comparing v1, v2, v3, v4
- [ ] (Optional) Set up MLflow server
- [ ] (When ready) Push best model to HF Hub
- [ ] (When ready) Create model card

## Cost

- **W&B**: Free for personal use (unlimited runs)
- **MLflow**: Free (self-hosted)
- **HF Hub**: Free for public models

## Next Steps

1. Run `wandb login` and create account
2. Enable in v4 config
3. Re-run training or log existing results
4. Compare all versions in W&B dashboard
