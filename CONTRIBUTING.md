# Contributing to Odin SLM

## Git Workflow

This project uses git version control to track all code changes and MLflow for experiment tracking. **Always commit before running experiments** to maintain reproducibility.

### Commit Points

#### 1. Before Data Preparation
**When:** After acquiring new raw data, before formatting
**What to commit:** Raw dataset files, data acquisition metadata
**Example:**
```bash
git add data/raw/new_medical_dataset.json
git commit -m "[data] Add new medical NER dataset (N samples)"
```

#### 2. After Data Formatting
**When:** After running `scripts/format_for_training.py`
**What to commit:** Formatted train/val/test splits
**Example:**
```bash
git add data/datasets/formatted/
git commit -m "[data] Format medical NER/RE dataset (train:4000, val:500, test:500)"
```

####3. Before Training
**When:** Before starting model training, after finalizing config
**What to commit:** Training configuration files
**Example:**
```bash
git add configs/medical_ner_re_config_v5.yaml
git commit -m "[config] Add v5 training config (3 epochs, lr=2e-4, rank=16)"
```

#### 4. After Training
**When:** After training completes successfully
**What to commit:** Training metadata (not model weights - those are gitignored)
**Example:**
```bash
git add experiments/medical_ner_re_v5/trainer_state.json
git commit -m "[train] Complete v5 training (final loss=0.17)

- Epochs: 3
- Entity focus training
- MLflow run ID: abc123"
```

#### 5. After Evaluation
**When:** After evaluating model on test set
**What to commit:** Evaluation results JSON
**Example:**
```bash
git add experiments/medical_ner_re_v5/evaluation_results.json
git commit -m "[eval] Add v5 evaluation results

Entity F1 (Micro): 88.5%
Relation F1 (Micro): 47.2%"
```

#### 6. Code Changes
**When:** After modifying scripts, source code, or fixing bugs
**What to commit:** Modified source files
**Example:**
```bash
git add scripts/evaluate_medical_ner.py
git commit -m "[fix] Fix case sensitivity bug in entity matching

- Lowercase entity types before comparison
- Fixes false mismatches between [Disease] and [disease]"
```

### Commit Message Format

Use the format: `[stage] action (key metric/detail)`

**Stages:**
- `[data]` - Data preparation, formatting, acquisition
- `[config]` - Configuration file changes
- `[train]` - Model training
- `[eval]` - Model evaluation
- `[fix]` - Bug fixes
- `[feat]` - New features
- `[docs]` - Documentation updates
- `[refactor]` - Code refactoring

**Examples:**
```bash
git commit -m "[data] Add synthetic medical dataset (5000 samples)"
git commit -m "[config] Update learning rate to 1e-4 for fine-tuning"
git commit -m "[train] Complete Stage 2 training (Entity F1=95%, Relation F1=58%)"
git commit -m "[eval] Re-evaluate v1 with fixed validation (Entity F1=87.2%)"
git commit -m "[fix] Fix silent relation parsing failures"
git commit -m "[feat] Add MLflow experiment tracking"
```

### What NOT to Commit

- Model weights (`*.bin`, `*.safetensors`, `*.pth`) - gitignored
- Large datasets - gitignored (only metadata/summaries)
- MLflow tracking directories (`mlruns/`, `mlartifacts/`) - gitignored
- Logs and temporary files - gitignored
- Virtual environments - gitignored

### MLflow Integration

Every training and evaluation run is automatically tracked in MLflow. To view experiments:

```bash
# Start MLflow UI
mlflow ui --port 5000

# Open in browser
http://localhost:5000
```

**What MLflow tracks:**
- All hyperparameters (learning rate, epochs, LoRA rank, etc.)
- Training metrics (loss, eval_loss, learning_rate per step)
- Evaluation metrics (Entity F1, Relation F1, precision, recall)
- Git commit hash (for reproducibility)
- Artifacts (configs, results, model metadata)

### Reproducibility

Every experiment must be reproducible. This requires:

1. **Git commit:** Code version tracked
2. **MLflow run:** Hyperparameters and metrics logged
3. **Deterministic generation:** Seeds set for evaluation
4. **Config files:** All settings documented in YAML

### Workflow Example

Complete workflow for a new training run:

```bash
# 1. Update configuration
vim configs/medical_ner_re_config_v5.yaml
git add configs/medical_ner_re_config_v5.yaml
git commit -m "[config] Add v5 training config"

# 2. Train model (MLflow tracks automatically)
uv run python scripts/train_medical_ner.py \
    --config configs/medical_ner_re_config_v5.yaml

# 3. Commit training completion
git add experiments/medical_ner_re_v5/trainer_state.json
git commit -m "[train] Complete v5 training (loss=0.17)"

# 4. Evaluate model (MLflow tracks automatically)
uv run python scripts/evaluate_medical_ner.py \
    --model experiments/medical_ner_re_v5/final_model \
    --test_data data/datasets/formatted/test.json

# 5. Commit evaluation results
git add experiments/medical_ner_re_v5/evaluation_results.json
git commit -m "[eval] Add v5 evaluation (Entity F1=88.5%, Relation F1=47.2%)"

# 6. View in MLflow UI
mlflow ui
```

### Branch Strategy (Future)

Currently working on `master` branch. When collaborating:

- `master` - Stable, working code
- `feature/*` - New features or experiments
- `fix/*` - Bug fixes

### Questions?

If you're unsure whether to commit, **commit anyway**. It's better to have too many commits than to lose track of changes.

**Remember:** Commit BEFORE every training/evaluation attempt, not after.
