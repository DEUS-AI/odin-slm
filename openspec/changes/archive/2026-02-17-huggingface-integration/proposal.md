## Why

The trained medical NER/RE model and combined dataset (ADE Corpus V2 + BC5CDR + BioRED) currently live only on local disk. Publishing them to Hugging Face Hub makes the work reproducible, shareable, and discoverable — and enables easy loading via `transformers` and `datasets` libraries without manual file management.

## What Changes

- Add a standalone script to push the trained LoRA adapter model to Hugging Face Hub with a proper model card (README, tags, metrics, usage examples)
- Add a standalone script to push the combined medical NER/RE dataset to Hugging Face Hub in the `datasets` library format with a dataset card
- Add HF Hub authentication configuration (token handling via environment variable or `huggingface-cli login`)
- Add model card and dataset card templates populated with training metadata (base model, entity types, relation types, evaluation metrics, data sources)

## Capabilities

### New Capabilities
- `hf-model-upload`: Script and model card generation for pushing LoRA adapter checkpoints to Hugging Face Hub
- `hf-dataset-upload`: Script and dataset card generation for pushing the combined medical NER/RE dataset to Hugging Face Hub

### Modified Capabilities
_(none — no existing specs are affected)_

## Impact

- **New files**: `scripts/push_model_hf.py`, `scripts/push_dataset_hf.py`
- **Dependencies**: `huggingface-hub` already in `pyproject.toml` (>=0.26.0) — no new dependencies needed
- **Authentication**: Requires a HF token (via `HF_TOKEN` env var or prior `huggingface-cli login`)
- **Data formats**: Dataset will be converted from the current JSON instruction format to HF `datasets` arrow format with proper splits (train/val/test)
- **Existing code**: No changes to training or evaluation scripts
