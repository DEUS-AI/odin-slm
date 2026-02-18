# Odin SLM: End-to-End Guide

**From raw data to a published model on Hugging Face Hub.**

This guide covers every stage of the Odin SLM pipeline for training and publishing Small Language Models fine-tuned for medical Named Entity Recognition (NER) and Relation Extraction (RE). It assumes familiarity with Python and machine learning fundamentals, but explains all domain-specific concepts (LoRA, QLoRA, quantization, etc.) from scratch.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Fine-Tuning Concepts](#2-fine-tuning-concepts)
3. [Data Pipeline](#3-data-pipeline)
4. [Training](#4-training)
5. [Evaluation](#5-evaluation)
6. [Experiment Tracking with MLflow](#6-experiment-tracking-with-mlflow)
7. [Publishing to Hugging Face Hub](#7-publishing-to-hugging-face-hub)
8. [Creating a New Version](#8-creating-a-new-version)
9. [Version History and Lessons Learned](#9-version-history-and-lessons-learned)
10. [Troubleshooting](#10-troubleshooting)
11. [Glossary](#11-glossary)

---

## 1. Overview

### What This Project Does

Odin SLM fine-tunes a large language model to extract **medical entities** and their **relationships** from clinical text. Given a passage like:

> "The patient developed acute renal failure after treatment with enalapril."

The model outputs structured extractions:

```
### Entities:
1. [Drug] enalapril
2. [Symptom] acute renal failure

### Relations:
1. enalapril --[causes]--> acute renal failure
```

### Task Definition

| Aspect | Details |
|--------|---------|
| **Entity types** | Disease, Drug, Symptom |
| **Relation types** | associated_with, causes, interacts_with, treats |
| **Base model** | Meta Llama 3.1 8B (4-bit quantized) |
| **Fine-tuning method** | QLoRA via Unsloth |
| **Hardware** | NVIDIA RTX 4090 Laptop GPU (16 GB VRAM) |

### Why an SLM Instead of a General-Purpose LLM?

- **Latency**: A fine-tuned 8B model runs locally in milliseconds, no API round-trip.
- **Cost**: No per-token API fees. Inference runs on a single consumer GPU.
- **Privacy**: Clinical data never leaves the local machine.
- **Specialization**: A task-specific model outperforms general LLMs on structured extraction. Our best model achieves Entity F1=0.918 on medical NER — competitive with much larger models.

### Pipeline at a Glance

```
 DATA PREP              TRAIN              EVALUATE
 ┌──────────┐         ┌──────────┐       ┌──────────┐
 │ convert_ │──┐      │ train_   │──────▶│evaluate_ │
 │ ade      │  │      │ medical_ │       │ medical_ │
 │ bc5cdr   │  ├────▶ │ ner.py   │       │ ner.py   │
 │ biored   │  │      └──────────┘       └────┬─────┘
 └──────────┘  │           ▲                   │
      │        │           │                   ▼
      ▼        │     ┌──────────┐        ┌──────────┐
 ┌──────────┐  │     │ config   │        │ eval     │
 │combine_  │──┘     │ v15.yaml │        │ results  │
 │datasets  │        └──────────┘        └────┬─────┘
 └──────────┘                                  │
                                               │
 TRACK                   PUBLISH               │
 ┌──────────┐          ┌──────────┐            │
 │ MLflow   │          │push_model│◀───────────┘
 │ backfill │          │_hf.py   │
 │ register │          ├──────────┤
 └──────────┘          │push_data│
                       │set_hf.py│
                       └────┬─────┘
                            │
                            ▼
                     Hugging Face Hub
```

---

## 2. Fine-Tuning Concepts

This section explains the key techniques used in the project. If you already know LoRA and QLoRA, skip to [Section 3](#3-data-pipeline).

### 2.1 Why Fine-Tune Instead of Training from Scratch?

Training a language model from scratch requires:
- Trillions of tokens of training data
- Thousands of GPU-hours
- Millions of dollars in compute

**Fine-tuning** starts from a pre-trained model that already understands language and teaches it a specific task. It's like hiring someone who already speaks English and training them in medical terminology — much faster than teaching a baby to speak.

### 2.2 Full Fine-Tuning vs Parameter-Efficient Fine-Tuning (PEFT)

**Full fine-tuning** updates every parameter in the model:

```
Llama 3.1 8B = 8,000,000,000 parameters
× 4 bytes (FP32) = 32 GB just for weights
+ gradients + optimizer states = ~100+ GB total

Result: Doesn't fit on any single consumer GPU.
```

**PEFT** methods train only a small fraction of parameters while keeping the rest frozen. The most popular PEFT method is **LoRA**.

### 2.3 LoRA (Low-Rank Adaptation)

LoRA is the core technique behind this project. Here's how it works.

#### The Intuition

A pre-trained model's weight matrices are large (e.g., 4096 x 4096 for Llama 3.1 8B). LoRA's insight is that the *change* needed to adapt a model to a new task is **low-rank** — it can be represented by two much smaller matrices multiplied together.

Instead of updating a full weight matrix `W`, LoRA freezes `W` and adds a low-rank update `B × A`:

```
STANDARD FINE-TUNING                    LoRA FINE-TUNING
═══════════════════                     ════════════════

Input ──▶ [ W ] ──▶ Output             Input ──▶ [ W₀ (frozen) ] ──┐
           ▲                                                        ├──▶ Output
           │ update ALL params                   [ B ]──[ A ]  ────┘
           │ (8B weights)                         ▲       ▲
                                                  │ only train these
                                                  │ (~67M params)
```

Mathematically:

```
W_new = W₀ + (α / r) × B · A

Where:
  W₀  = original frozen weights       (d × d, e.g., 4096 × 4096)
  B   = down-projection matrix         (d × r, e.g., 4096 × 32)
  A   = up-projection matrix           (r × d, e.g., 32 × 4096)
  r   = rank (bottleneck dimension)    (32 in our config)
  α   = scaling factor                 (32 in our config)
```

#### Rank (r) — The Capacity Knob

The rank `r` controls how much capacity the adapter has to learn new behavior. Think of it as the "bandwidth" of the adaptation:

| Rank (r) | Trainable Params | Capacity | Use Case |
|----------|-----------------|----------|----------|
| 4 | ~8M | Minimal | Simple style transfer |
| 8 | ~17M | Low | Basic classification |
| 16 | ~34M | Medium | Our early experiments (V1–V8) |
| **32** | **~67M** | **High** | **Our current config (V9+)** — complex extraction |
| 64 | ~134M | Very high | Diminishing returns for our task |
| 128 | ~268M | Maximum | Rarely needed |

We increased from r=16 to r=32 at V9 when switching to the 8B base model — the larger model benefits from more adapter capacity.

**Rule of thumb**: Start with r=16. Increase to 32 if the model underfits (low training loss but poor eval metrics). Going beyond 64 rarely helps and increases memory usage.

#### Alpha (α) — The Scaling Factor

Alpha controls how strongly the LoRA weights influence the output. The effective scaling is `α / r`:

```
When α = r  (both 32):  scaling = 32/32 = 1.0  (full strength)
When α = 16, r = 32:    scaling = 16/32 = 0.5  (half strength)
When α = 64, r = 32:    scaling = 64/32 = 2.0  (double strength)
```

**Our config**: `α = r = 32`, which gives a scaling factor of 1.0. This is the standard starting point. If training is unstable (loss spikes), reduce alpha. If the model isn't learning enough, increase it.

#### Target Modules — Where LoRA is Applied

A transformer layer has several weight matrices. Our config applies LoRA to all of them:

```
TRANSFORMER LAYER (1 of 32 layers in Llama 3.1 8B)
═══════════════════════════════════════════════════

  ┌──────────────── Self-Attention ─────────────────┐
  │                                                 │
  │  q_proj  ◀── LoRA   Query: "What am I looking  │
  │                              for?"              │
  │  k_proj  ◀── LoRA   Key: "What information do  │
  │                            I contain?"          │
  │  v_proj  ◀── LoRA   Value: "Here's my actual   │
  │                              content"           │
  │  o_proj  ◀── LoRA   Output projection           │
  │                                                 │
  └─────────────────────────────────────────────────┘
                        │
                        ▼
  ┌──────────────── Feed-Forward (MLP) ─────────────┐
  │                                                 │
  │  gate_proj  ◀── LoRA   Controls information     │
  │                         flow (gating mechanism)  │
  │  up_proj    ◀── LoRA   Expands to higher dim    │
  │  down_proj  ◀── LoRA   Projects back down       │
  │                                                 │
  └─────────────────────────────────────────────────┘

  7 target modules × 32 layers = 224 LoRA adapter pairs
```

Applying LoRA to all 7 modules gives maximum expressiveness. Some practitioners only target the attention projections (q_proj, v_proj) to save memory — we found full coverage works better for our structured extraction task.

### 2.4 QLoRA — Quantized LoRA

QLoRA combines LoRA with **4-bit quantization** of the base model. This is what makes it possible to fine-tune an 8B parameter model on a 16 GB GPU.

#### What is Quantization?

Quantization reduces the precision of model weights to use less memory:

```
Format      Bits   Bytes/param   8B model size   Quality
────────    ────   ───────────   ─────────────   ───────
FP32        32     4.0           32 GB           Perfect
FP16/BF16   16     2.0           16 GB           Near-perfect
INT8        8      1.0           8 GB            Slight degradation
NF4         4      0.5           4 GB            Minimal degradation
```

**NF4 (Normal Float 4-bit)** is the format used by QLoRA. It's specifically designed for neural network weights, which tend to follow a normal distribution. It provides information-theoretically optimal 4-bit quantization for normally-distributed values.

#### Why QLoRA Works

```
MEMORY BUDGET: 16 GB VRAM (RTX 4090 Laptop)
════════════════════════════════════════════

Full precision (FP32):
  Base model:     32.0 GB  ✗ doesn't fit
  + LoRA:          0.5 GB
  + Gradients:     0.5 GB
  + Optimizer:     1.0 GB
  Total:          34.0 GB

QLoRA (4-bit base + FP16 adapters):
  Base model:      4.0 GB  ✓ fits!
  + LoRA (FP16):   0.5 GB
  + Gradients:     0.5 GB
  + Optimizer:     1.0 GB
  + Activations:   ~4 GB   (depends on batch size / seq length)
  Total:          ~10 GB   ✓ room to spare
```

The key insight: **base weights are frozen**, so quantization error doesn't accumulate during training. Only the small LoRA adapters are trained in full precision (FP16/BF16).

### 2.5 Unsloth

[Unsloth](https://github.com/unslothai/unsloth) is a library that provides optimized CUDA kernels for LoRA fine-tuning. It's a drop-in replacement for the standard Hugging Face training loop that delivers:

- **2x faster training**: Custom fused kernels for attention and MLP layers
- **50% less memory**: Optimized memory layout and gradient checkpointing
- **No accuracy loss**: Numerically equivalent to standard training

We use Unsloth's `FastLanguageModel` class for both training and inference. It handles model loading, LoRA injection, and 4-bit quantization transparently.

### 2.6 The Instruction Format (Alpaca)

The model is trained using the **Alpaca instruction format**, a simple template that separates the task description from the input and expected output:

```
### Instruction:
Extract all medical entities and their relations from the following clinical
text. Identify diseases, symptoms, drugs, procedures, and lab tests. For each
entity found, specify its type. Then identify relations between entities.

### Input:
The patient developed acute renal failure after treatment with enalapril.

### Output:
### Entities:
1. [Drug] enalapril
2. [Symptom] acute renal failure

### Relations:
1. enalapril --[causes]--> acute renal failure
```

This format is used throughout the pipeline:
- **Data preparation**: All datasets are converted to this format
- **Training**: The model learns to generate the Output section given Instruction + Input
- **Inference**: We provide Instruction + Input and let the model generate the Output
- **Evaluation**: We compare generated Output against gold-standard Output

---

## 3. Data Pipeline

### 3.1 Data Sources

The training dataset combines three publicly available biomedical corpora:

| Source | Domain | Entities | Relations | Train Examples |
|--------|--------|----------|-----------|---------------|
| **ADE Corpus V2** | Adverse drug events from case reports | Drug, Symptom | causes | ~3,400 |
| **BC5CDR** | Chemical-disease relations from PubMed | Drug, Disease | causes | ~480 |
| **BioRED** | Biomedical relations from PubMed | Drug, Disease | causes, treats, associated_with, interacts_with | ~170 |

**ADE Corpus V2** is the largest source and provides the strongest signal. Our best model (V13) was trained on ADE data alone. BC5CDR and BioRED add coverage for Disease entities and multi-relation types (treats, interacts_with, associated_with).

There is also a **synthetic data generator** (`scripts/generate_medical_dataset.py`) that produces rule-based clinical notes. We found that real data significantly outperforms synthetic data — synthetic is useful for initial prototyping only.

### 3.2 Converting Data Sources

Each source has its own format and requires a conversion script. All scripts output the same standardized JSON format.

**Convert ADE Corpus V2** (downloads automatically from Hugging Face):
```bash
python scripts/convert_ade_corpus.py \
  --output_dir data/datasets/ade_formatted \
  --max_tokens 480
```

**Convert BC5CDR** (downloads automatically from Hugging Face):
```bash
python scripts/convert_bc5cdr.py \
  --output_dir data/datasets/bc5cdr_formatted \
  --max_tokens 480
```

**Convert BioRED** (requires manual download of BioC JSON from NCBI):
```bash
python scripts/convert_biored.py \
  --data_dir data/raw/biored/BioRED \
  --output_dir data/datasets/biored_formatted \
  --max_tokens 480
```

The `--max_tokens` parameter controls the maximum input length. Longer texts contain more context but use more GPU memory during training. We use 480 for standard configs and 960 for the V15 config (to recover longer PubMed abstracts).

Each script creates `train.json`, `val.json`, and `test.json` in the output directory. The JSON format is:

```json
[
  {
    "instruction": "Extract all medical entities and their relations...",
    "input": "The patient developed acute renal failure after treatment with enalapril.",
    "output": "### Entities:\n1. [Drug] enalapril\n2. [Symptom] acute renal failure\n\n### Relations:\n1. enalapril --[causes]--> acute renal failure"
  }
]
```

### 3.3 Combining Datasets

To merge all sources into a single training dataset:

```bash
python scripts/combine_datasets.py \
  --output_dir data/datasets/combined_v16 \
  --ade_dir data/datasets/ade_formatted \
  --bc5cdr_dir data/datasets/bc5cdr_formatted \
  --biored_dir data/datasets/biored_formatted \
  --synthetic_ratio 0.0 \
  --seed 42
```

| Parameter | Description |
|-----------|-------------|
| `--output_dir` | Where to write the combined train/val/test.json files |
| `--ade_dir` | Path to converted ADE dataset |
| `--bc5cdr_dir` | Path to converted BC5CDR dataset |
| `--biored_dir` | Path to converted BioRED dataset |
| `--synthetic_ratio` | Fraction of synthetic data to add (0.0 = none, 0.25 = 25% of real data size) |
| `--seed` | Random seed for reproducible shuffling |

**Recommendation**: Use `--synthetic_ratio 0.0` (real data only). Our experiments consistently showed that real data alone produces better results than mixed real+synthetic.

### 3.4 Output Format

The structured output format the model learns to generate:

```
### Entities:
1. [Drug] aspirin
2. [Disease] myocardial infarction
3. [Symptom] chest pain

### Relations:
1. aspirin --[treats]--> myocardial infarction
2. myocardial infarction --[causes]--> chest pain
```

Entity types and their meanings:

| Type | Description | Example |
|------|-------------|---------|
| **Drug** | Pharmaceutical compounds, chemicals | aspirin, enalapril, metformin |
| **Disease** | Medical conditions, disorders | diabetes, hypertension, cancer |
| **Symptom** | Signs, symptoms, adverse effects | nausea, acute renal failure, headache |

Relation types and their meanings:

| Relation | Description | Example |
|----------|-------------|---------|
| **causes** | Drug/disease causes a symptom/condition | aspirin --[causes]--> bleeding |
| **treats** | Drug is used to treat a disease | metformin --[treats]--> diabetes |
| **associated_with** | General association between entities | obesity --[associated_with]--> diabetes |
| **interacts_with** | Drug-drug interaction | warfarin --[interacts_with]--> aspirin |

---

## 4. Training

### 4.1 Configuration Anatomy

Training is controlled by a YAML config file. Here is the latest config (`configs/medical_ner_re_config_v15.yaml`) with every field explained:

```yaml
# ── Model Configuration ──────────────────────────────────────
model:
  name: "unsloth/Meta-Llama-3.1-8B-bnb-4bit"  # HF model ID (pre-quantized by Unsloth)
  max_seq_length: 1024   # Max tokens per example (longer = more memory)
  load_in_4bit: true     # Enable QLoRA 4-bit quantization
  dtype: "bfloat16"      # Computation dtype (BF16 preferred on Ampere+ GPUs)

# ── Training Hyperparameters ─────────────────────────────────
training:
  output_dir: "./experiments/medical_ner_re_v15"  # Where checkpoints are saved
  num_train_epochs: 3                # Number of passes over the training data
  per_device_train_batch_size: 2     # Examples per GPU per step
  gradient_accumulation_steps: 8     # Accumulate gradients over 8 steps
                                     # Effective batch size = 2 × 8 = 16
  gradient_checkpointing: true       # Trade compute for memory (essential for 16GB)
  warmup_steps: 50                   # Linear LR warmup from 0 to learning_rate
  learning_rate: 2.0e-4              # Peak learning rate (standard for LoRA)
  weight_decay: 0.01                 # L2 regularization
  logging_steps: 10                  # Log metrics every N steps
  save_steps: 127                    # Save checkpoint every N steps
  eval_steps: 127                    # Run validation every N steps
  evaluation_strategy: "steps"       # Evaluate by steps (not epochs)
  save_strategy: "steps"             # Save by steps (not epochs)
  load_best_model_at_end: true       # Keep the best checkpoint by eval_loss
  metric_for_best_model: "eval_loss" # Metric to determine "best"
  fp16: false                        # Don't use FP16 (we use BF16)
  bf16: true                         # Use BFloat16 mixed precision
  optim: "adamw_8bit"               # 8-bit AdamW (75% less optimizer memory)
  max_grad_norm: 1.0                 # Gradient clipping threshold
  save_total_limit: 3                # Keep only 3 most recent checkpoints

# ── LoRA Configuration ───────────────────────────────────────
lora:
  r: 32                              # Rank — adapter capacity (see Section 2.3)
  lora_alpha: 32                     # Scaling factor (α/r = 1.0)
  lora_dropout: 0                    # No dropout (Unsloth recommendation)
  bias: "none"                       # Don't train bias terms
  task_type: "CAUSAL_LM"            # Causal language modeling task
  target_modules:                    # Which weight matrices get LoRA (all 7)
    - "q_proj"
    - "k_proj"
    - "v_proj"
    - "o_proj"
    - "gate_proj"
    - "up_proj"
    - "down_proj"

# ── Dataset Paths ────────────────────────────────────────────
dataset:
  train_file: "data/datasets/combined_v15/train.json"
  val_file: "data/datasets/combined_v15/val.json"
  test_file: "data/datasets/combined_v15/test.json"
  text_field: "input"
  format: "instruction"
  seed: 42

# ── Weights & Biases (optional) ──────────────────────────────
wandb:
  project: "odin-slm-medical-ner"
  name: "llama31-8b-base-medical-ner-re-v15"
  entity: null
  enabled: false
```

### 4.2 Key Hyperparameter Relationships

```
Effective batch size = per_device_train_batch_size × gradient_accumulation_steps
                     = 2 × 8 = 16

Steps per epoch = num_train_examples / effective_batch_size
                = 4062 / 16 ≈ 254 steps

Total training steps = steps_per_epoch × num_train_epochs
                     = 254 × 3 = 762 steps
```

The `save_steps` and `eval_steps` are set to 127 (roughly half an epoch), giving ~6 checkpoints across 3 epochs and ~6 evaluation points to track convergence.

### 4.3 Running Training

**Prerequisites**:
- NVIDIA GPU with CUDA 12.0+
- Python 3.12+ with the project virtual environment activated
- Dependencies installed (`uv sync`)

```bash
# Activate virtual environment
source .venv/bin/activate

# Run training
python scripts/train_medical_ner.py --config configs/medical_ner_re_config_v15.yaml
```

**What happens during training**:
1. Loads the base model in 4-bit quantization (~4 GB VRAM)
2. Injects LoRA adapters into all target modules
3. Loads the training and validation datasets
4. Trains for the configured number of epochs
5. Saves checkpoints at regular intervals
6. Logs metrics to MLflow automatically
7. Saves the final model to `<output_dir>/final_model/`

**What to monitor**:
- `train_loss`: Should decrease steadily. If it plateaus early, consider increasing the learning rate or rank.
- `eval_loss`: Should decrease alongside train_loss. If eval_loss increases while train_loss decreases, the model is overfitting — reduce epochs or increase dropout.
- GPU utilization: Run `nvidia-smi -l 1` in another terminal. Should be >90%.
- VRAM usage: If you hit OOM, reduce `per_device_train_batch_size` or `max_seq_length`.

### 4.4 Training Output

After training completes, the experiment directory contains:

```
experiments/medical_ner_re_v15/
├── final_model/                    # Best model (by eval_loss)
│   ├── adapter_model.safetensors   # LoRA weights (~300 MB)
│   ├── adapter_config.json         # LoRA configuration
│   ├── tokenizer.json              # Tokenizer vocabulary
│   ├── tokenizer.model             # SentencePiece model
│   ├── tokenizer_config.json       # Tokenizer settings
│   └── special_tokens_map.json     # Special token mappings
├── checkpoint-127/                 # Intermediate checkpoint
├── checkpoint-254/                 # Intermediate checkpoint
├── checkpoint-381/                 # Intermediate checkpoint
└── trainer_state.json              # Full training history
```

The `final_model/` directory contains only the LoRA adapter — **not** the full 8B base model. To run inference, you need both the base model (downloaded automatically from Hugging Face) and the adapter.

---

## 5. Evaluation

### 5.1 Metrics

The evaluation pipeline measures two things:

**Entity Extraction**: Did the model find the correct entities with the correct types?
- Uses **exact match**: the entity text and type must both match exactly (case-insensitive)
- Reports precision, recall, and F1 in both micro and macro averages

**Relation Extraction**: Did the model identify the correct relationships?
- Uses **exact match** on the triple: (head entity, relation type, tail entity)
- All three components must match (case-insensitive)

#### Understanding Precision, Recall, and F1

```
                         Predicted
                    ┌──────┬──────┐
                    │  Yes │  No  │
              ┌─────┼──────┼──────┤
  Actual  Yes │     │  TP  │  FN  │   Recall = TP / (TP + FN)
              ├─────┼──────┼──────┤   "Of all real entities, how many did we find?"
          No  │     │  FP  │  TN  │
              └─────┼──────┼──────┤   Precision = TP / (TP + FP)
                    └──────┴──────┘   "Of all predicted entities, how many were correct?"

  F1 = 2 × (Precision × Recall) / (Precision + Recall)
  "Harmonic mean — balances precision and recall"
```

**Micro vs Macro averaging**:
- **Micro**: Aggregates TP/FP/FN across all examples, then computes metrics. Weighted by example difficulty (examples with more entities count more).
- **Macro**: Computes metrics per example, then averages. Treats all examples equally.

We primarily report **micro F1** because it better reflects real-world performance on datasets with varying numbers of entities per example.

### 5.2 Running Evaluation

```bash
python scripts/evaluate_medical_ner.py \
  --model experiments/medical_ner_re_v14/final_model \
  --test_data data/datasets/combined_v15/test.json \
  --num_samples 100  # Optional: evaluate on a subset for quick checks
```

| Parameter | Description |
|-----------|-------------|
| `--model` | Path to the trained adapter (the `final_model/` directory) |
| `--test_data` | Path to the test split JSON file |
| `--num_samples` | Optional: limit to first N examples (useful for quick sanity checks) |
| `--output_file` | Optional: custom path for results JSON (default: `<experiment_dir>/evaluation_results.json`) |
| `--run_name` | Optional: override the MLflow run name |

The script:
1. Loads the trained LoRA adapter with Unsloth
2. Runs greedy decoding on each test example
3. Parses the generated output into structured entities and relations
4. Compares against gold-standard annotations
5. Computes micro and macro F1 for both entities and relations
6. Saves detailed results to `evaluation_results.json`
7. Logs metrics to MLflow

### 5.3 Interpreting Results

**Current benchmarks** (our best model, V13):

| Metric | Score | Target |
|--------|-------|--------|
| Entity F1 (micro) | 0.918 | >= 0.85 |
| Relation F1 (micro) | 0.842 | >= 0.75 |

**Guidelines for "is it good enough?"**:
- Entity F1 > 0.85 and Relation F1 > 0.75: **Production-ready** for most use cases
- Entity F1 > 0.90 and Relation F1 > 0.80: **Strong performance** — this is where V13 sits
- Entity F1 < 0.80: Model needs more training data or hyperparameter tuning

**Common failure modes to check in `evaluation_results.json`**:
- **Hallucinated entities**: Model extracts entities not present in the input (high FP → low precision)
- **Missed entities**: Model fails to find entities that are there (high FN → low recall)
- **Wrong relation direction**: e.g., predicts `disease --[causes]--> drug` instead of `drug --[causes]--> disease`
- **Extra relations**: Model invents relationships between entities that aren't related

---

## 6. Experiment Tracking with MLflow

### 6.1 Overview

MLflow provides a web UI for comparing experiments, tracking metrics, and managing model versions. All training and evaluation runs are automatically logged.

```
┌──────────────────────────────────────────────────┐
│                 MLflow Architecture               │
├──────────────────────────────────────────────────┤
│                                                  │
│  Training Script ───▶ MLflow Tracking Server     │
│  Evaluation Script ──▶  (Docker container)       │
│                          │                       │
│                          ├── SQLite backend      │
│                          │   (run metadata)      │
│                          │                       │
│                          └── Local artifacts     │
│                              (configs, results)  │
│                                                  │
│  Browser ──▶ http://localhost:5000               │
│              (Experiment UI + Model Registry)    │
│                                                  │
└──────────────────────────────────────────────────┘
```

### 6.2 Starting the MLflow Server

The MLflow server runs in Docker:

```bash
# Start the server (runs in background)
docker-compose up -d

# Check it's running
docker ps  # Should show "odin-mlflow" container

# Access the UI
open http://localhost:5000
```

The `docker-compose.yml` configuration:
- **Image**: `ghcr.io/mlflow/mlflow:v3.9.0`
- **Port**: 5000
- **Backend**: SQLite at `./mlflow-data/db/mlflow.db`
- **Artifacts**: Local storage at `./mlflow-data/mlartifacts/`
- **Persistence**: Data survives container restarts (volume-mounted)

### 6.3 Automatic Logging

Both the training and evaluation scripts log to MLflow automatically via `mlflow_config.py`:

**Training runs log**:
- All hyperparameters (model name, learning rate, LoRA rank, etc.)
- System info (GPU, CUDA version, PyTorch version)
- Git commit hash (for reproducibility)
- Config file as an artifact
- Tagged with `stage: training`

**Evaluation runs log**:
- Entity and relation F1/precision/recall (micro and macro)
- TP/FP/FN counts
- The full `evaluation_results.json` as an artifact
- Tagged with `stage: evaluation` and `model_version`

### 6.4 Backfilling Historical Experiments

If you have experiments on disk that weren't logged to MLflow (e.g., from before MLflow was set up):

```bash
# Backfill all historical experiments (V1–V15)
python scripts/backfill_mlflow.py
```

This reads configs, trainer states, and evaluation results from disk and creates MLflow runs retroactively. It's idempotent — safe to run multiple times.

### 6.5 Model Registry

The Model Registry provides a centralized catalog of all trained models with versioning and aliases:

```bash
# Register all models in the MLflow Model Registry
python scripts/register_models_mlflow.py
```

This creates:
- A registered model named `odin-medical-ner-re`
- One version per experiment (v1–v15)
- The **champion** alias pointing to the best model (currently V13)

**Viewing the registry**: Navigate to `http://localhost:5000/#/models/odin-medical-ner-re`

### 6.6 Comparing Experiments in the UI

The MLflow UI is most useful for **comparing experiments side-by-side**:

1. Go to `http://localhost:5000`
2. Select the `odin-slm-medical-ner` experiment
3. Check the runs you want to compare
4. Click "Compare" to see metrics side-by-side
5. Use the chart view to plot `entity_f1_micro` and `relation_f1_micro` across versions

This helps answer questions like:
- "Did increasing the rank from 16 to 32 help?"
- "Is V15 (combined data, longer sequences) better than V13 (ADE-only)?"
- "What's the impact of more training epochs?"

---

## 7. Publishing to Hugging Face Hub

### 7.1 Authentication

Before pushing anything to Hugging Face, authenticate:

```bash
# Option 1: Interactive login (recommended, one-time)
huggingface-cli login

# Option 2: Environment variable (for CI/scripts)
export HF_TOKEN=hf_your_token_here
```

The token is stored at `~/.cache/huggingface/token` and reused by all scripts.

### 7.2 Publishing a Model

```bash
python scripts/push_model_hf.py \
  --experiment-dir experiments/medical_ner_re_v14 \
  --repo-id pabloformoso/odin-llama3.1-medical-ner-v14 \
  --private  # Optional: make the repo private
```

| Parameter | Description |
|-----------|-------------|
| `--experiment-dir` | Path to the experiment directory |
| `--repo-id` | HF repo identifier (`username/repo-name`) |
| `--checkpoint` | Optional: upload a specific checkpoint instead of `final_model/` |
| `--private` | Optional: create as a private repo (default: public) |

**Naming convention**: Derivative models based on Llama **must include "Llama" in the name** per Meta's license. Use a pattern like: `username/odin-llama3.1-medical-ner-v14`.

The script automatically generates a **model card** (README.md) containing:
- Base model and LoRA configuration
- Evaluation metrics (if `evaluation_results.json` exists)
- Entity and relation types
- A code example showing how to load and use the model

### 7.3 Publishing a Dataset

```bash
python scripts/push_dataset_hf.py \
  --dataset-dir data/datasets/combined_v15 \
  --repo-id pabloformoso/medical-ner-re-dataset \
  --private  # Optional
```

| Parameter | Description |
|-----------|-------------|
| `--dataset-dir` | Path to the dataset directory (must contain `train.json`) |
| `--repo-id` | HF dataset repo identifier |
| `--private` | Optional: create as a private repo |

The script:
1. Loads `train.json`, `val.json`, `test.json` into a HF `DatasetDict`
2. Generates a **dataset card** with statistics, sources, and usage example
3. Pushes to HF Hub in Parquet format (the standard for HF datasets)

### 7.4 Loading Published Artifacts

Once published, anyone with access can load the model and dataset:

```python
# Load the dataset
from datasets import load_dataset
dataset = load_dataset("pabloformoso/medical-ner-re-dataset")
print(f"Train: {len(dataset['train'])} examples")

# Load the model
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_model = "unsloth/Meta-Llama-3.1-8B-bnb-4bit"
adapter_id = "pabloformoso/odin-llama3.1-medical-ner-v14"

tokenizer = AutoTokenizer.from_pretrained(adapter_id)
model = AutoModelForCausalLM.from_pretrained(base_model, device_map="auto")
model = PeftModel.from_pretrained(model, adapter_id)

# Run inference
prompt = """### Instruction:
Extract all medical entities and their relations from the following clinical text.

### Input:
The patient developed acute renal failure after treatment with enalapril.

### Output:"""

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=256)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

---

## 8. Creating a New Version

### 8.1 When to Create a New Version

Create a new version when you're changing any of:
- The base model
- The training data (new sources, different mix, different preprocessing)
- LoRA hyperparameters (rank, alpha, target modules)
- Training hyperparameters (epochs, learning rate, batch size)
- Sequence length

**Don't** create a new version for code-only changes (refactoring scripts, fixing logging, etc.).

### 8.2 Step-by-Step Recipe

```bash
# ──────────────────────────────────────────────────────────
# STEP 1: Create a new config
# ──────────────────────────────────────────────────────────

# Copy the latest config as a starting point
cp configs/medical_ner_re_config_v15.yaml configs/medical_ner_re_config_v16.yaml

# Edit the new config:
#   - Update output_dir: ./experiments/medical_ner_re_v16
#   - Update wandb name: llama31-8b-base-medical-ner-re-v16-<description>
#   - Update dataset paths if using new data
#   - Change the hyperparameters you're experimenting with
#   - Add a comment block at the top documenting what changed vs previous version

# ──────────────────────────────────────────────────────────
# STEP 2: Prepare data (if using new data sources)
# ──────────────────────────────────────────────────────────

# Convert any new data sources
python scripts/convert_new_source.py --output_dir data/datasets/new_source_formatted

# Combine into a new dataset version
python scripts/combine_datasets.py \
  --output_dir data/datasets/combined_v16 \
  --ade_dir data/datasets/ade_formatted \
  --bc5cdr_dir data/datasets/bc5cdr_formatted \
  --biored_dir data/datasets/biored_formatted \
  --synthetic_ratio 0.0

# ──────────────────────────────────────────────────────────
# STEP 3: Train
# ──────────────────────────────────────────────────────────

python scripts/train_medical_ner.py --config configs/medical_ner_re_config_v16.yaml

# ──────────────────────────────────────────────────────────
# STEP 4: Evaluate
# ──────────────────────────────────────────────────────────

python scripts/evaluate_medical_ner.py \
  --model experiments/medical_ner_re_v16/final_model \
  --test_data data/datasets/combined_v16/test.json

# ──────────────────────────────────────────────────────────
# STEP 5: Compare to champion in MLflow
# ──────────────────────────────────────────────────────────

# Open http://localhost:5000 and compare V16 metrics against the champion (V13):
#   Champion: Entity F1=0.918, Relation F1=0.842
#
# If V16 is better:
#   - Update the champion alias in MLflow Model Registry
#   - Publish to HF Hub (see Section 7)

# ──────────────────────────────────────────────────────────
# STEP 6: Publish (if the new version is an improvement)
# ──────────────────────────────────────────────────────────

python scripts/push_model_hf.py \
  --experiment-dir experiments/medical_ner_re_v16 \
  --repo-id pabloformoso/odin-llama3.1-medical-ner-v16

python scripts/push_dataset_hf.py \
  --dataset-dir data/datasets/combined_v16 \
  --repo-id pabloformoso/medical-ner-re-dataset-v16
```

### 8.3 Config Conventions

Each config file should have a **comment block at the top** documenting what changed. Follow the existing pattern:

```yaml
# Medical NER/RE Training Configuration - V16
# Key changes vs V15:
#   - <describe what you changed and why>
#   - <e.g., "Added new data source X with Y examples">
#   - <e.g., "Increased epochs from 3 to 5">
# V15 result: Entity F1=X.XXX, Relation F1=X.XXX
# Optimized for RTX 4090 Laptop GPU (16GB VRAM)
```

### 8.4 Trying a Different Base Model

To experiment with a different base model (e.g., Mistral, Phi, Gemma):

1. Find the Unsloth-optimized model on Hugging Face (e.g., `unsloth/Phi-3.5-mini-instruct-bnb-4bit`)
2. Update `model.name` in the config
3. You may need to adjust `max_seq_length` — different models have different context windows
4. LoRA `target_modules` may differ between architectures — check the model's layer names
5. Start with a lower learning rate (1e-4) when switching architectures

### 8.5 Adding a New Data Source

To incorporate a new biomedical corpus:

1. Write a conversion script following the pattern in `scripts/convert_ade_corpus.py`:
   - Download or load the raw data
   - Map entity types to our schema (Disease, Drug, Symptom)
   - Map relation types to our schema (causes, treats, associated_with, interacts_with)
   - Output `train.json`, `val.json`, `test.json` in the instruction format
2. Update `scripts/combine_datasets.py` to include the new source
3. Create a new combined dataset version
4. Train and evaluate as usual

---

## 9. Version History and Lessons Learned

### 9.1 Version Timeline

| Version | Base Model | Data | Key Change | Entity F1 | Relation F1 |
|---------|-----------|------|------------|-----------|-------------|
| V1 | Llama 3.2 1B Instruct | Synthetic | Initial baseline | ~0.60 | ~0.40 |
| V2 | Llama 3.2 1B Instruct | Synthetic v2 | More entity/relation types | ~0.65 | ~0.45 |
| V3 | Llama 3.2 1B Instruct | Synthetic | Two-stage training | — | — |
| V4–V8 | Llama 3.2 1B Instruct | Synthetic | Various tuning experiments | ~0.70 | ~0.50 |
| **V9** | **Llama 3.1 8B Base** | Synthetic | **Switched to 8B base model** | ~0.75 | ~0.55 |
| **V10** | Llama 3.1 8B Base | Synthetic | **Fixed eval bug (removed repetition_penalty)** | ~0.80 | ~0.65 |
| V11 | Llama 3.1 8B Base | Synthetic | 5 epochs (no improvement over 3) | ~0.80 | ~0.65 |
| V12 | Llama 3.1 8B Base | Synthetic | Data augmentation, vocab expansion | ~0.82 | ~0.68 |
| V12b | Llama 3.1 8B Base | **ADE real** | **First real data experiment** | ~0.88 | ~0.78 |
| **V13** | **Llama 3.1 8B Base** | **ADE real** | **5 epochs, ADE-only** | **0.918** | **0.842** |
| V14 | Llama 3.1 8B Base | Combined | Added BC5CDR + BioRED | 0.911 | 0.832 |
| V15 | Llama 3.1 8B Base | Combined | Longer sequences (1024), 3 epochs | — | — |

### 9.2 Key Lessons

**1. Real data crushes synthetic data.**
V12b (first real data) immediately outperformed all synthetic experiments. V13 (ADE-only, real) set the all-time best scores. If you have real annotated data, use it.

**2. Bigger base model = better, but with diminishing returns.**
Switching from Llama 3.2 1B to Llama 3.1 8B at V9 was the single largest improvement. The 8B model has more capacity to learn the structured extraction task. Going beyond 8B would require more VRAM than our hardware allows.

**3. More data isn't always better.**
V14 (combined ADE + BC5CDR + BioRED) performed slightly *worse* than V13 (ADE-only). The likely reason: BC5CDR and BioRED have different annotation styles and entity type distributions, adding noise. Quality > quantity for task-specific fine-tuning.

**4. Don't use repetition_penalty for NER.**
V10 fixed a critical bug: `repetition_penalty` and `no_repeat_ngram_size` in the generation config were preventing the model from copying entity text verbatim from the input — which is exactly what NER requires. Removing them gave an immediate F1 boost.

**5. 3 epochs is the sweet spot.**
V11 tried 5 epochs and showed no improvement over 3. More epochs just overfit on this dataset size (~3,400 examples).

**6. LoRA rank 32 > 16 for this task.**
The increase from r=16 (V1–V8) to r=32 (V9+) gave the adapter enough capacity for the complex extraction task. The 8B model benefits from the additional adapter parameters.

---

## 10. Troubleshooting

### CUDA Out of Memory

If training crashes with OOM:

1. **Reduce batch size**: Lower `per_device_train_batch_size` (e.g., 2 → 1)
2. **Increase gradient accumulation**: Keep effective batch size the same (e.g., if batch 2→1, set accumulation 8→16)
3. **Reduce sequence length**: Lower `max_seq_length` (e.g., 1024 → 512)
4. **Check for other GPU processes**: Run `nvidia-smi` to see if something else is using VRAM
5. **Ensure gradient checkpointing is on**: `gradient_checkpointing: true` in config

### Evaluation Metrics Are Suspiciously Low

- **Check the generation config**: Make sure `repetition_penalty` and `no_repeat_ngram_size` are NOT set. These prevent the model from outputting entity names that appear in the input.
- **Check the output parsing**: Run with `--num_samples 5` and inspect the raw predictions. Is the model generating the expected format?
- **Check the test data**: Is the test set from the same distribution as the training data?

### Evaluation Metrics Are Suspiciously High

- **Check for data leakage**: Are any test examples also in the training set? Run deduplication.
- **Check the sample size**: Evaluating on very few examples can give misleadingly high scores.

### MLflow Server Won't Start

```bash
# Check if the container exists but is stopped
docker ps -a | grep odin-mlflow

# Check logs
docker logs odin-mlflow

# Restart
docker-compose down && docker-compose up -d

# Check if port 5000 is in use by something else
lsof -i :5000
```

### HF Push Fails

- **Authentication**: Run `huggingface-cli whoami` to verify you're logged in
- **Repo name format**: Must be `username/repo-name` or `org/repo-name`
- **File size**: Adapter files are typically 100–300 MB and upload fine. If you're trying to push the full base model (>4 GB), something is wrong — you should only push the adapter.

---

## 11. Glossary

| Term | Definition |
|------|-----------|
| **Adapter** | The small set of trained LoRA weights that modify the base model's behavior. Saved as `adapter_model.safetensors`. |
| **Alpha (α)** | LoRA scaling factor. Controls how strongly the adapter influences the output. Effective scaling = α/r. |
| **Base model** | The pre-trained language model that is frozen during fine-tuning (e.g., Llama 3.1 8B). |
| **BF16 (BFloat16)** | A 16-bit floating-point format optimized for deep learning. Same range as FP32 but half the precision. |
| **Checkpoint** | A snapshot of model weights saved during training. Used to resume training or pick the best model. |
| **F1 Score** | Harmonic mean of precision and recall. Ranges from 0 (worst) to 1 (best). |
| **Fine-tuning** | Adapting a pre-trained model to a specific task by training on task-specific data. |
| **Gradient accumulation** | Simulates larger batch sizes by accumulating gradients over multiple forward passes before updating weights. |
| **Gradient checkpointing** | Trades compute for memory by recomputing activations during the backward pass instead of storing them. |
| **LoRA** | Low-Rank Adaptation. A PEFT method that adds small trainable matrices to frozen model weights. |
| **Macro average** | Computes a metric per example, then averages. Treats all examples equally regardless of size. |
| **Micro average** | Aggregates counts (TP/FP/FN) across all examples, then computes the metric. Weighted by example difficulty. |
| **NER** | Named Entity Recognition. The task of identifying and classifying entities in text (e.g., "enalapril" → Drug). |
| **NF4** | Normal Float 4-bit. A quantization format optimized for normally-distributed neural network weights. |
| **PEFT** | Parameter-Efficient Fine-Tuning. A family of methods (LoRA, Prefix Tuning, etc.) that train only a small subset of parameters. |
| **QLoRA** | Quantized LoRA. Combines 4-bit quantization of the base model with LoRA fine-tuning. |
| **Rank (r)** | The bottleneck dimension of the LoRA matrices. Controls adapter capacity. Higher rank = more parameters = more capacity. |
| **RE** | Relation Extraction. The task of identifying relationships between entities (e.g., enalapril --[causes]--> renal failure). |
| **SFT** | Supervised Fine-Tuning. Training a model on labeled input-output pairs. |
| **Target modules** | The specific weight matrices inside the transformer that receive LoRA adapters (e.g., q_proj, v_proj). |
| **Unsloth** | A library providing optimized CUDA kernels for LoRA fine-tuning. 2x faster, 50% less memory vs standard training. |
