## ADDED Requirements

### Requirement: Upload dataset to HF Hub
The script `scripts/push_dataset_hf.py` SHALL upload a combined dataset directory to Hugging Face Hub. It SHALL accept `--dataset-dir` (path to the dataset, e.g., `data/datasets/combined_v15`) and `--repo-id` (HF repo identifier, e.g., `username/medical-ner-re-dataset`) as required CLI arguments.

#### Scenario: Successful dataset upload
- **WHEN** the user runs `python scripts/push_dataset_hf.py --dataset-dir data/datasets/combined_v15 --repo-id user/my-dataset`
- **THEN** the script creates the HF dataset repo (if it does not exist), converts the JSON files to a HF `datasets`-compatible format with train/val/test splits, and uploads them along with a generated `README.md` dataset card

#### Scenario: Dataset directory does not exist
- **WHEN** the user provides a `--dataset-dir` that does not exist
- **THEN** the script exits with a non-zero exit code and prints an error message identifying the missing path

#### Scenario: Missing split files
- **WHEN** the dataset directory does not contain `train.json`
- **THEN** the script exits with a non-zero exit code and prints an error listing which expected files (`train.json`, `val.json`, `test.json`) are missing

### Requirement: Convert JSON to HF datasets format
The script SHALL read `train.json`, `val.json`, and `test.json` from the dataset directory. Each JSON file contains an array of objects with `instruction`, `input`, and `output` fields. The script SHALL load them using `Dataset.from_json()` and push as a `DatasetDict` with named splits (`train`, `validation`, `test`).

#### Scenario: All three splits present
- **WHEN** `train.json`, `val.json`, and `test.json` all exist
- **THEN** the uploaded dataset has `train`, `validation`, and `test` splits

#### Scenario: Only train split present
- **WHEN** only `train.json` exists (no `val.json` or `test.json`)
- **THEN** the uploaded dataset has only the `train` split and the script prints a warning about missing splits

### Requirement: Generate dataset card with metadata
The script SHALL generate a `README.md` dataset card containing: task description (medical NER and relation extraction), data sources (ADE Corpus V2, BC5CDR, BioRED), split statistics (number of examples per split), field descriptions (`instruction`, `input`, `output`), entity types (Disease, Drug, Symptom), relation types (associated_with, causes, interacts_with, treats), and a usage code example showing `load_dataset()`.

#### Scenario: Dataset card includes split statistics
- **WHEN** the script processes the dataset directory
- **THEN** the dataset card includes a table with the number of examples in each split

### Requirement: Support private and public repos
The script SHALL accept an optional `--private` flag. When set, the HF dataset repo SHALL be created as private. The default SHALL be public.

#### Scenario: Create a private dataset repo
- **WHEN** the user provides `--private`
- **THEN** the HF dataset repo is created with `private=True`

#### Scenario: Create a public dataset repo (default)
- **WHEN** the user omits `--private`
- **THEN** the HF dataset repo is created with `private=False`

### Requirement: HF authentication
The script SHALL authenticate using the token from `huggingface-cli login` (stored in `~/.cache/huggingface/token`) or the `HF_TOKEN` environment variable. It SHALL NOT accept a token as a CLI argument.

#### Scenario: No authentication available
- **WHEN** no cached token exists and `HF_TOKEN` is not set
- **THEN** the script exits with a non-zero exit code and prints instructions to authenticate
