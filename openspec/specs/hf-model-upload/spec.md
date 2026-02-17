## ADDED Requirements

### Requirement: Upload LoRA adapter to HF Hub
The script `scripts/push_model_hf.py` SHALL upload a LoRA adapter experiment directory to Hugging Face Hub. It SHALL accept `--experiment-dir` (path to the experiment, e.g., `experiments/medical_ner_re_v14`) and `--repo-id` (HF repo identifier, e.g., `username/odin-medical-ner-v14`) as required CLI arguments.

#### Scenario: Successful model upload
- **WHEN** the user runs `python scripts/push_model_hf.py --experiment-dir experiments/medical_ner_re_v14 --repo-id user/my-model`
- **THEN** the script creates the HF repo (if it does not exist), uploads all adapter files (`adapter_model.safetensors`, `adapter_config.json`, `tokenizer*`, `special_tokens_map.json`), and generates a `README.md` model card in the repo

#### Scenario: Experiment directory does not exist
- **WHEN** the user provides an `--experiment-dir` that does not exist or does not contain `adapter_config.json`
- **THEN** the script exits with a non-zero exit code and prints an error message identifying the missing path

### Requirement: Upload from final_model or specific checkpoint
The script SHALL look for adapter files in `<experiment-dir>/final_model/` by default. It SHALL accept an optional `--checkpoint` argument to upload a specific checkpoint subdirectory instead (e.g., `--checkpoint checkpoint-762`).

#### Scenario: Upload final model (default)
- **WHEN** the user runs the script without `--checkpoint` and `final_model/` exists in the experiment directory
- **THEN** the script uploads files from `<experiment-dir>/final_model/`

#### Scenario: Upload a specific checkpoint
- **WHEN** the user provides `--checkpoint checkpoint-762`
- **THEN** the script uploads files from `<experiment-dir>/checkpoint-762/`

#### Scenario: Neither final_model nor checkpoint found
- **WHEN** the user omits `--checkpoint` and `final_model/` does not exist in the experiment directory
- **THEN** the script exits with a non-zero exit code and suggests available subdirectories

### Requirement: Generate model card with metadata
The script SHALL generate a `README.md` model card containing: base model name (from `adapter_config.json`), LoRA configuration (rank, alpha, target modules), evaluation metrics (from `evaluation_results.json` if present), supported entity types and relation types, and a usage code example showing how to load the adapter with PEFT.

#### Scenario: Model card with evaluation metrics
- **WHEN** `evaluation_results.json` exists in the experiment directory
- **THEN** the model card includes an "Evaluation Results" section with entity F1 and relation F1 scores

#### Scenario: Model card without evaluation metrics
- **WHEN** `evaluation_results.json` does not exist in the experiment directory
- **THEN** the model card is generated without the evaluation section and does not error

### Requirement: Support private and public repos
The script SHALL accept an optional `--private` flag. When set, the HF repo SHALL be created as private. The default SHALL be public.

#### Scenario: Create a private repo
- **WHEN** the user provides `--private`
- **THEN** the HF repo is created with `private=True`

#### Scenario: Create a public repo (default)
- **WHEN** the user omits `--private`
- **THEN** the HF repo is created with `private=False`

### Requirement: HF authentication
The script SHALL authenticate using the token from `huggingface-cli login` (stored in `~/.cache/huggingface/token`) or the `HF_TOKEN` environment variable. It SHALL NOT accept a token as a CLI argument.

#### Scenario: Authenticated via huggingface-cli login
- **WHEN** the user has previously run `huggingface-cli login` and no `HF_TOKEN` is set
- **THEN** the script authenticates using the cached token

#### Scenario: Authenticated via HF_TOKEN env var
- **WHEN** `HF_TOKEN` is set in the environment
- **THEN** the script authenticates using that token

#### Scenario: No authentication available
- **WHEN** no cached token exists and `HF_TOKEN` is not set
- **THEN** the script exits with a non-zero exit code and prints instructions to authenticate
