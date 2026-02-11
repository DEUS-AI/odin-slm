#!/usr/bin/env python3
"""Format medical NER/RE dataset for instruction tuning

Usage:
    python scripts/format_for_training.py --input data/datasets/medical_ner_re_train.json
"""

import argparse
import json
import sys
from pathlib import Path
import random

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from odin_slm.data.synthetic_generator import MedicalTextGenerator


def format_for_instruction_tuning(doc, entities_only=False):
    """Convert MedicalDocument to instruction-tuning format

    Args:
        doc: MedicalDocument with entities and relations
        entities_only: If True, only include entities in output (for Stage 1 training)

    Format:
    {
        "instruction": "Extract medical entities and relations from the clinical text.",
        "input": "<clinical text>",
        "output": "<structured entities and relations>"
    }
    """
    if entities_only:
        instruction = "Extract all medical entities from the following clinical text. Identify diseases, symptoms, drugs, procedures, and lab tests."
    else:
        instruction = "Extract all medical entities and their relations from the following clinical text. Identify diseases, symptoms, drugs, procedures, and lab tests, along with their relationships."

    input_text = doc.text

    # Format entities
    if doc.entities:
        entities_lines = ["### Entities:"]
        for i, e in enumerate(doc.entities, 1):
            entities_lines.append(f"{i}. [{e.type}] {e.text}")
    else:
        entities_lines = ["### Entities:", "None found."]

    # Format relations (skip if entities_only=True)
    if entities_only:
        output = "\n".join(entities_lines)
    else:
        if doc.relations:
            # Deduplicate relations (same head text, type, tail text)
            seen_rels = set()
            unique_relations = []
            for r in doc.relations:
                key = (r.head.text.lower(), r.type.lower(), r.tail.text.lower())
                if key not in seen_rels:
                    seen_rels.add(key)
                    unique_relations.append(r)

            relations_lines = ["", "### Relations:"]
            for i, r in enumerate(unique_relations, 1):
                relations_lines.append(f"{i}. {r.head.text} --[{r.type}]--> {r.tail.text}")
        else:
            relations_lines = ["", "### Relations:", "None found."]

        output = "\n".join(entities_lines + relations_lines)

    return {
        "instruction": instruction,
        "input": input_text,
        "output": output
    }


def format_for_chat(doc):
    """Convert MedicalDocument to chat format (for chat-tuned models)

    Format:
    {
        "messages": [
            {"role": "user", "content": "..."},
            {"role": "assistant", "content": "..."}
        ]
    }
    """
    user_message = f"""Extract medical entities and relations from this clinical text:

{doc.text}

Please identify:
1. Entities: diseases, symptoms, drugs, procedures, lab tests
2. Relations: treats, causes, indicates, interacts_with"""

    # Format assistant response
    entities_text = "\n".join([f"- [{e.type}] {e.text}" for e in doc.entities])

    if doc.relations:
        relations_text = "\n".join([
            f"- {r.head.text} --[{r.type}]--> {r.tail.text}"
            for r in doc.relations
        ])
    else:
        relations_text = "No relations found."

    assistant_message = f"""**Entities:**
{entities_text}

**Relations:**
{relations_text}"""

    return {
        "messages": [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": assistant_message}
        ]
    }


def create_splits(data, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    """Split data into train/val/test sets"""
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    random.seed(seed)
    random.shuffle(data)

    n = len(data)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    return {
        "train": data[:train_end],
        "val": data[train_end:val_end],
        "test": data[val_end:]
    }


def main():
    parser = argparse.ArgumentParser(description="Format dataset for instruction tuning")
    parser.add_argument(
        "--input",
        type=str,
        default="data/datasets/medical_ner_re_train.json",
        help="Input dataset JSON file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/datasets/formatted",
        help="Output directory for formatted datasets",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["instruction", "chat"],
        default="instruction",
        help="Output format (instruction or chat)",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="Training set ratio",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.1,
        help="Validation set ratio",
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.1,
        help="Test set ratio",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splits")
    parser.add_argument(
        "--entities_only",
        action="store_true",
        help="Format for entity extraction only (Stage 1 training)"
    )

    args = parser.parse_args()

    print("=" * 80)
    print("FORMATTING DATASET FOR TRAINING")
    print("=" * 80)
    print(f"\nInput: {args.input}")
    print(f"Output: {args.output_dir}")
    print(f"Format: {args.format}")
    print(f"Splits: {args.train_ratio:.1%} train / {args.val_ratio:.1%} val / {args.test_ratio:.1%} test")

    # Load dataset
    print("\nLoading dataset...")
    generator = MedicalTextGenerator()
    docs = generator.load_dataset(args.input)
    print(f"✓ Loaded {len(docs)} documents")

    # Format documents
    if args.entities_only:
        print(f"\nFormatting documents as '{args.format}' (entities only)...")
    else:
        print(f"\nFormatting documents as '{args.format}'...")

    if args.format == "instruction":
        formatted_data = [format_for_instruction_tuning(doc, entities_only=args.entities_only) for doc in docs]
    else:
        formatted_data = [format_for_chat(doc) for doc in docs]

    if args.entities_only:
        print(f"✓ Formatted {len(formatted_data)} documents (entities only)")
    else:
        print(f"✓ Formatted {len(formatted_data)} documents")

    # Create splits
    print("\nCreating train/val/test splits...")
    splits = create_splits(
        formatted_data,
        args.train_ratio,
        args.val_ratio,
        args.test_ratio,
        args.seed
    )

    print(f"  Train: {len(splits['train'])} documents")
    print(f"  Val:   {len(splits['val'])} documents")
    print(f"  Test:  {len(splits['test'])} documents")

    # Save splits
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for split_name, split_data in splits.items():
        output_path = output_dir / f"{split_name}.json"
        with open(output_path, "w") as f:
            json.dump(split_data, f, indent=2)
        print(f"✓ Saved {split_name} split to {output_path}")

    # Show example
    print("\n" + "=" * 80)
    print("EXAMPLE FORMATTED DOCUMENT")
    print("=" * 80)
    example = splits['train'][0]

    if args.format == "instruction":
        print(f"\nInstruction:\n{example['instruction']}")
        print(f"\nInput:\n{example['input'][:200]}...")
        print(f"\nOutput:\n{example['output'][:300]}...")
    else:
        print(f"\nUser:\n{example['messages'][0]['content'][:200]}...")
        print(f"\nAssistant:\n{example['messages'][1]['content'][:300]}...")

    print("\n" + "=" * 80)
    print("✓ FORMATTING COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
