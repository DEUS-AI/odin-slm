#!/usr/bin/env python3
"""Push a combined medical NER/RE dataset to Hugging Face Hub.

Usage:
    python scripts/push_dataset_hf.py --dataset-dir data/datasets/combined_v15 --repo-id username/medical-ner-re-dataset
    python scripts/push_dataset_hf.py --dataset-dir data/datasets/combined_v15 --repo-id username/medical-ner-re-dataset --private
"""

import argparse
import json
import sys
from pathlib import Path

from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi, HfFolder


def check_auth():
    """Check for HF authentication. Returns token or exits."""
    token = HfFolder.get_token()
    if token is None:
        print("Error: No Hugging Face authentication found.")
        print()
        print("Authenticate using one of:")
        print("  1. Run: huggingface-cli login")
        print("  2. Set HF_TOKEN environment variable")
        sys.exit(1)
    return token


def validate_dataset_dir(dataset_dir: Path) -> dict[str, Path]:
    """Validate dataset directory and return available split paths."""
    if not dataset_dir.exists():
        print(f"Error: Dataset directory does not exist: {dataset_dir}")
        sys.exit(1)

    split_files = {
        "train": dataset_dir / "train.json",
        "validation": dataset_dir / "val.json",
        "test": dataset_dir / "test.json",
    }

    # train.json is required
    if not split_files["train"].exists():
        missing = [name for name, path in split_files.items() if not path.exists()]
        print(f"Error: Required file 'train.json' not found in {dataset_dir}")
        print(f"Missing files: {', '.join(missing)}")
        sys.exit(1)

    # Warn about missing optional splits
    available = {}
    for split_name, path in split_files.items():
        if path.exists():
            available[split_name] = path
        else:
            print(f"Warning: '{path.name}' not found — '{split_name}' split will be omitted")

    return available


def load_splits(split_paths: dict[str, Path]) -> DatasetDict:
    """Load JSON split files into a DatasetDict."""
    splits = {}
    for split_name, path in split_paths.items():
        ds = Dataset.from_json(str(path))
        splits[split_name] = ds
        print(f"  Loaded {split_name}: {len(ds)} examples")
    return DatasetDict(splits)


def generate_dataset_card(dataset_dict: DatasetDict, repo_id: str) -> str:
    """Generate a README.md dataset card."""
    # Build split statistics
    split_rows = []
    for split_name, ds in dataset_dict.items():
        split_rows.append(f"| {split_name} | {len(ds):,} |")

    total = sum(len(ds) for ds in dataset_dict.values())

    lines = [
        "---",
        "task_categories:",
        "- token-classification",
        "- text-generation",
        "tags:",
        "- medical-ner",
        "- relation-extraction",
        "- biomedical",
        "- clinical-text",
        "language:",
        "- en",
        "size_categories:",
        f"- {'1K<n<10K' if total < 10000 else '10K<n<100K'}",
        "---",
        "",
        f"# {repo_id.split('/')[-1]}",
        "",
        "A combined dataset for **medical Named Entity Recognition (NER) and Relation Extraction (RE)**,",
        "formatted as instruction-following examples for fine-tuning language models.",
        "",
        "## Task",
        "",
        "Given a clinical text passage, extract:",
        "- **Entities**: Disease, Drug, Symptom",
        "- **Relations**: associated_with, causes, interacts_with, treats",
        "",
        "## Data Sources",
        "",
        "| Source | Description |",
        "|--------|------------|",
        "| ADE Corpus V2 | Drug–adverse drug event relations from case reports |",
        "| BC5CDR | Chemical–disease relations from PubMed abstracts |",
        "| BioRED | Biomedical relation extraction from PubMed |",
        "",
        "## Split Statistics",
        "",
        "| Split | Examples |",
        "|-------|----------|",
        *split_rows,
        f"| **Total** | **{total:,}** |",
        "",
        "## Fields",
        "",
        "Each example has three fields:",
        "",
        "| Field | Description |",
        "|-------|-------------|",
        "| `instruction` | Task description (extract entities and relations) |",
        "| `input` | Clinical text passage |",
        "| `output` | Structured extraction with entities and relations |",
        "",
        "## Usage",
        "",
        "```python",
        "from datasets import load_dataset",
        "",
        f'dataset = load_dataset("{repo_id}")',
        "",
        "# Access splits",
        'train = dataset["train"]',
        'print(f"Train examples: {len(train)}")',
        "print(train[0])",
        "```",
        "",
        "## Output Format",
        "",
        "The `output` field contains structured extractions:",
        "",
        "```",
        "### Entities:",
        "1. [Drug] aspirin",
        "2. [Symptom] gastrointestinal bleeding",
        "",
        "### Relations:",
        "1. aspirin --[causes]--> gastrointestinal bleeding",
        "```",
        "",
        "## Licensing",
        "",
        "This dataset combines data from multiple sources, each with their own licenses.",
        "Please refer to the original datasets for licensing terms:",
        "- ADE Corpus V2",
        "- BC5CDR (BioCreative V CDR)",
        "- BioRED",
        "",
    ]

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Push a combined medical NER/RE dataset to Hugging Face Hub"
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        required=True,
        help="Path to the dataset directory (e.g., data/datasets/combined_v15)",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="HF repo identifier (e.g., username/medical-ner-re-dataset)",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create the repo as private (default: public)",
    )
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)

    # Step 1: Check authentication
    print("Checking Hugging Face authentication...")
    token = check_auth()
    print("  Authenticated.")

    # Step 2: Validate dataset directory
    print(f"Validating dataset directory: {dataset_dir}")
    split_paths = validate_dataset_dir(dataset_dir)

    # Step 3: Load splits
    print("Loading dataset splits...")
    dataset_dict = load_splits(split_paths)

    # Step 4: Generate dataset card
    print("Generating dataset card...")
    card = generate_dataset_card(dataset_dict, args.repo_id)
    readme_path = dataset_dir / "README.md"
    readme_path.write_text(card)
    print(f"  Written to {readme_path}")

    # Step 5: Create repo and upload
    api = HfApi()
    print(f"Creating dataset repo '{args.repo_id}' (private={args.private})...")
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="dataset",
        private=args.private,
        exist_ok=True,
    )

    print(f"Uploading dataset...")
    dataset_dict.push_to_hub(
        args.repo_id,
        private=args.private,
        token=token,
    )

    # Upload the README separately (push_to_hub generates its own)
    api.upload_file(
        path_or_fileobj=str(readme_path),
        path_in_repo="README.md",
        repo_id=args.repo_id,
        repo_type="dataset",
    )

    visibility = "private" if args.private else "public"
    print(f"\nDone! Dataset uploaded to https://huggingface.co/datasets/{args.repo_id} ({visibility})")


if __name__ == "__main__":
    main()
