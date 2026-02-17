## Context

The project has trained multiple versions of a medical NER/RE model (v1–v14+) as LoRA adapters on top of `unsloth/Meta-Llama-3.1-8B-bnb-4bit`. The best result so far is v14 (Entity F1=0.911, Relation F1=0.832). The combined dataset (v15) has 4,062 train / 950 val / 958 test examples sourced from ADE Corpus V2, BC5CDR, and BioRED, stored as JSON files with `instruction`, `input`, `output` keys.

Both the model and dataset currently live only on local disk under `experiments/` and `data/datasets/`. The `huggingface-hub` library (>=0.26.0) is already a dependency.

## Goals / Non-Goals

**Goals:**
- Push any LoRA adapter checkpoint to HF Hub with a single command
- Push the combined dataset to HF Hub as a proper `datasets`-compatible repo with train/val/test splits
- Auto-generate model cards and dataset cards with metadata (base model, metrics, entity types, relation types, data sources, usage examples)
- Support both private and public repos
- Work with existing HF auth (`HF_TOKEN` env var or `huggingface-cli login`)

**Non-Goals:**
- Merging LoRA weights into the base model before upload (users can do this separately)
- Auto-pushing after training (standalone scripts only, per user preference)
- Publishing to model registries beyond HF Hub (e.g., MLflow Model Registry)
- Hosting inference endpoints on HF

## Decisions

### 1. Use `huggingface_hub` Python API (not `git` LFS)

**Choice**: Use `HfApi.upload_folder()` and `HfApi.create_repo()` from `huggingface_hub`.

**Why over git-based push**: The Python API handles LFS, repo creation, and README updates in one call without requiring `git-lfs` to be installed. It's also what PEFT's `model.push_to_hub()` uses internally.

**Why not use PEFT's built-in `push_to_hub`**: That would require loading the model into GPU memory just to push files. Since the adapter is already saved to disk, uploading the folder directly is simpler and doesn't need a GPU.

### 2. Dataset format: keep instruction-style JSON, add HF metadata

**Choice**: Convert the existing JSON files to a HF `datasets` repo using `Dataset.from_json()` + `push_to_hub()` with train/val/test splits.

**Why not convert to a different schema**: The instruction format (`instruction`, `input`, `output`) is already a widely-understood structure on HF. Changing it would break compatibility with the training script. The JSON files load cleanly into HF `datasets`.

### 3. Model card generated from adapter_config.json + evaluation_results.json

**Choice**: The upload script reads `adapter_config.json` for model metadata and `evaluation_results.json` (if present) for metrics, then generates a README.md model card.

**Why**: This keeps the card accurate and up-to-date without manual editing. The card includes: base model, LoRA config, evaluation metrics, entity/relation types, and a usage example.

### 4. Repo naming convention

**Choice**: User provides the HF repo name via CLI argument (e.g., `--repo-id username/odin-medical-ner-re-v14`). No automatic naming.

**Why**: HF namespace rules vary (org vs personal), and version naming is a user preference. The script validates the format but doesn't impose a convention.

### 5. Script location and structure

**Choice**: Two standalone scripts in `scripts/`:
- `scripts/push_model_hf.py` — pushes a LoRA adapter experiment directory
- `scripts/push_dataset_hf.py` — pushes a combined dataset directory

**Why two scripts**: Model and dataset are independent concerns with different metadata. Separate scripts keep each focused and allow pushing one without the other.

## Risks / Trade-offs

- **Large file uploads may fail on slow connections** → `huggingface_hub` supports chunked uploads and retries by default. The adapter files are small (~100-200MB for safetensors).
- **Token leakage** → Scripts read `HF_TOKEN` from env var only, never accept it as a CLI argument, never log it. Rely on `huggingface-cli login` as the primary auth method.
- **Stale model cards** → If metrics change after re-evaluation, the card on HF won't auto-update. Mitigation: re-run the push script to overwrite the card.
- **Dataset licensing ambiguity** → ADE Corpus V2, BC5CDR, and BioRED each have their own licenses. The dataset card should clearly state source licenses and not claim a single permissive license for the combined dataset.
