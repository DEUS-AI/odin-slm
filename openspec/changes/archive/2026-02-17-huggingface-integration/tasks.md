## 1. Model Upload Script

- [x] 1.1 Create `scripts/push_model_hf.py` with CLI argument parsing (`--experiment-dir`, `--repo-id`, `--checkpoint`, `--private`)
- [x] 1.2 Implement HF authentication check (cached token or `HF_TOKEN` env var) with clear error message on failure
- [x] 1.3 Implement experiment directory validation: resolve adapter source path (`final_model/` default or `--checkpoint`), verify `adapter_config.json` exists, suggest available subdirectories on failure
- [x] 1.4 Implement model card generation: read `adapter_config.json` for LoRA config and base model, read `evaluation_results.json` for metrics (if present), render README.md with metadata, entity/relation types, and PEFT usage example
- [x] 1.5 Implement upload via `HfApi.create_repo()` + `HfApi.upload_folder()` with `--private` flag support
- [x] 1.6 Test the script end-to-end against `experiments/medical_ner_re_v14` with a private repo

## 2. Dataset Upload Script

- [x] 2.1 Create `scripts/push_dataset_hf.py` with CLI argument parsing (`--dataset-dir`, `--repo-id`, `--private`)
- [x] 2.2 Implement dataset directory validation: check for `train.json` (required), warn on missing `val.json` or `test.json`
- [x] 2.3 Implement JSON-to-HF conversion: load each split with `Dataset.from_json()`, build a `DatasetDict` with `train`/`validation`/`test` keys
- [x] 2.4 Implement dataset card generation: render README.md with task description, data sources, split statistics table, field descriptions, entity/relation types, and `load_dataset()` usage example
- [x] 2.5 Implement upload via `DatasetDict.push_to_hub()` with `--private` flag support
- [x] 2.6 Test the script end-to-end against `data/datasets/combined_v15` with a private repo
