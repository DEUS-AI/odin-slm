#!/usr/bin/env python3
"""Push a trained LoRA adapter to Hugging Face Hub.

Usage:
    python scripts/push_model_hf.py --experiment-dir experiments/medical_ner_re_v14 --repo-id username/odin-medical-ner-v14
    python scripts/push_model_hf.py --experiment-dir experiments/medical_ner_re_v14 --repo-id username/odin-medical-ner-v14 --checkpoint checkpoint-762 --private
"""

import argparse
import json
import sys
from pathlib import Path

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


def resolve_adapter_path(experiment_dir: Path, checkpoint: str | None) -> Path:
    """Resolve the adapter source directory."""
    if checkpoint:
        adapter_path = experiment_dir / checkpoint
    else:
        adapter_path = experiment_dir / "final_model"

    if not adapter_path.exists():
        if checkpoint:
            print(f"Error: Checkpoint '{checkpoint}' not found in {experiment_dir}")
        else:
            print(f"Error: 'final_model/' not found in {experiment_dir}")

        # Suggest available subdirectories
        subdirs = sorted(
            [d.name for d in experiment_dir.iterdir() if d.is_dir()],
        )
        if subdirs:
            print(f"\nAvailable subdirectories:")
            for d in subdirs:
                print(f"  - {d}")
            print(f"\nUse --checkpoint <name> to select one.")
        sys.exit(1)

    if not (adapter_path / "adapter_config.json").exists():
        print(f"Error: No adapter_config.json found in {adapter_path}")
        print("This does not appear to be a valid LoRA adapter directory.")
        sys.exit(1)

    return adapter_path


def generate_model_card(adapter_path: Path, experiment_dir: Path, repo_id: str) -> str:
    """Generate a README.md model card from adapter metadata."""
    # Read adapter config
    with open(adapter_path / "adapter_config.json") as f:
        adapter_config = json.load(f)

    base_model = adapter_config.get("base_model_name_or_path", "unknown")
    lora_r = adapter_config.get("r", "?")
    lora_alpha = adapter_config.get("lora_alpha", "?")
    target_modules = adapter_config.get("target_modules", [])

    # Read evaluation results if available
    eval_results = None
    eval_path = experiment_dir / "evaluation_results.json"
    if eval_path.exists():
        with open(eval_path) as f:
            eval_results = json.load(f)

    # Build model card
    lines = [
        "---",
        f"base_model: {base_model}",
        "library_name: peft",
        "tags:",
        "- medical-ner",
        "- relation-extraction",
        "- lora",
        "- biomedical",
        "- unsloth",
        "language:",
        "- en",
        "pipeline_tag: text-generation",
        "---",
        "",
        f"# {repo_id.split('/')[-1]}",
        "",
        "A LoRA adapter fine-tuned for **medical Named Entity Recognition (NER) and Relation Extraction (RE)**.",
        "",
        "## Task",
        "",
        "Extracts medical entities and their relationships from clinical text.",
        "",
        "- **Entity types**: Disease, Drug, Symptom",
        "- **Relation types**: associated_with, causes, interacts_with, treats",
        "",
        "## Model Details",
        "",
        f"- **Base model**: [{base_model}](https://huggingface.co/{base_model})",
        f"- **LoRA rank (r)**: {lora_r}",
        f"- **LoRA alpha**: {lora_alpha}",
        f"- **Target modules**: {', '.join(target_modules)}",
        f"- **PEFT type**: {adapter_config.get('peft_type', 'LORA')}",
        "",
    ]

    if eval_results:
        entity = eval_results.get("entity_metrics", {}).get("micro", {})
        relation = eval_results.get("relation_metrics", {}).get("micro", {})

        lines.extend([
            "## Evaluation Results",
            "",
            "| Metric | Precision | Recall | F1 |",
            "|--------|-----------|--------|----|",
        ])
        if entity:
            lines.append(
                f"| Entity (micro) | {entity.get('precision', 0):.3f} | "
                f"{entity.get('recall', 0):.3f} | {entity.get('f1', 0):.3f} |"
            )
        if relation:
            lines.append(
                f"| Relation (micro) | {relation.get('precision', 0):.3f} | "
                f"{relation.get('recall', 0):.3f} | {relation.get('f1', 0):.3f} |"
            )
        lines.append("")

    lines.extend([
        "## Usage",
        "",
        "```python",
        "from peft import PeftModel",
        "from transformers import AutoModelForCausalLM, AutoTokenizer",
        "",
        f'base_model = "{base_model}"',
        f'adapter_id = "{repo_id}"',
        "",
        "tokenizer = AutoTokenizer.from_pretrained(adapter_id)",
        "model = AutoModelForCausalLM.from_pretrained(base_model, device_map=\"auto\")",
        "model = PeftModel.from_pretrained(model, adapter_id)",
        "",
        'prompt = """### Instruction:',
        "Extract all medical entities and their relations from the following clinical text.",
        "",
        "### Input:",
        "The patient developed acute renal failure after treatment with enalapril.",
        "",
        '### Output:"""',
        "",
        "inputs = tokenizer(prompt, return_tensors=\"pt\").to(model.device)",
        "outputs = model.generate(**inputs, max_new_tokens=256)",
        "print(tokenizer.decode(outputs[0], skip_special_tokens=True))",
        "```",
        "",
        "## Training Data",
        "",
        "Combined dataset from:",
        "- **ADE Corpus V2**: Drug–adverse effect relations",
        "- **BC5CDR**: Chemical–disease relations",
        "- **BioRED**: Biomedical relation extraction",
        "",
    ])

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Push a trained LoRA adapter to Hugging Face Hub"
    )
    parser.add_argument(
        "--experiment-dir",
        type=str,
        required=True,
        help="Path to the experiment directory (e.g., experiments/medical_ner_re_v14)",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="HF repo identifier (e.g., username/odin-medical-ner-v14)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Specific checkpoint subdirectory to upload (default: final_model)",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create the repo as private (default: public)",
    )
    args = parser.parse_args()

    experiment_dir = Path(args.experiment_dir)
    if not experiment_dir.exists():
        print(f"Error: Experiment directory does not exist: {experiment_dir}")
        sys.exit(1)

    # Step 1: Check authentication
    print("Checking Hugging Face authentication...")
    token = check_auth()
    print("  Authenticated.")

    # Step 2: Resolve adapter path
    adapter_path = resolve_adapter_path(experiment_dir, args.checkpoint)
    print(f"  Adapter source: {adapter_path}")

    # Step 3: Generate model card
    print("Generating model card...")
    model_card = generate_model_card(adapter_path, experiment_dir, args.repo_id)
    readme_path = adapter_path / "README.md"
    readme_path.write_text(model_card)
    print(f"  Written to {readme_path}")

    # Step 4: Create repo and upload
    api = HfApi()
    print(f"Creating repo '{args.repo_id}' (private={args.private})...")
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="model",
        private=args.private,
        exist_ok=True,
    )

    print(f"Uploading adapter files from {adapter_path}...")
    api.upload_folder(
        folder_path=str(adapter_path),
        repo_id=args.repo_id,
        repo_type="model",
    )

    visibility = "private" if args.private else "public"
    print(f"\nDone! Model uploaded to https://huggingface.co/{args.repo_id} ({visibility})")


if __name__ == "__main__":
    main()
