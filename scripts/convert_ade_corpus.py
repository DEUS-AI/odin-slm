#!/usr/bin/env python3
"""Convert ADE Corpus V2 from Hugging Face to our instruction-tuning format.

Downloads the ADE (Adverse Drug Events) corpus and converts it to the same
instruction/input/output format used by our training pipeline.

Entity mapping:
  - drug → Drug
  - effect → Symptom (adverse effects are symptoms)

Relation mapping:
  - drug-effect → causes (drug causes adverse effect)

Usage:
    python scripts/convert_ade_corpus.py --output_dir data/datasets/ade_formatted
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

ALPACA_TEMPLATE = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}"""


def load_ade_corpus():
    """Load ADE Corpus V2 drug-ADE relation subset from Hugging Face."""
    from datasets import load_dataset

    print("Downloading ADE Corpus V2 from Hugging Face...")
    ds = load_dataset(
        "ade-benchmark-corpus/ade_corpus_v2", "Ade_corpus_v2_drug_ade_relation"
    )
    print(f"  Loaded {len(ds['train'])} drug-effect relation rows")
    return ds["train"]


def group_by_sentence(dataset):
    """Group all drug-effect pairs by their source sentence."""
    sentence_data = defaultdict(lambda: {"drugs": set(), "effects": set(), "relations": set()})

    for ex in dataset:
        text = ex["text"]
        drug = ex["drug"].strip()
        effect = ex["effect"].strip()

        sentence_data[text]["drugs"].add(drug)
        sentence_data[text]["effects"].add(effect)
        sentence_data[text]["relations"].add((drug, "causes", effect))

    return sentence_data


def format_example(text, data):
    """Convert a sentence with its entities/relations to instruction format."""
    instruction = (
        "Extract all medical entities and their relations from the following "
        "clinical text. Identify diseases, symptoms, drugs, procedures, and "
        "lab tests, along with their relationships."
    )

    # Build entity list (drugs first, then effects/symptoms)
    entities = []
    for drug in sorted(data["drugs"]):
        entities.append({"type": "Drug", "text": drug})
    for effect in sorted(data["effects"]):
        entities.append({"type": "Symptom", "text": effect})

    # Build entity lines
    entities_lines = ["### Entities:"]
    for i, e in enumerate(entities, 1):
        entities_lines.append(f"{i}. [{e['type']}] {e['text']}")

    # Build relation lines (deduplicated)
    relations = sorted(data["relations"])
    relations_lines = ["", "### Relations:"]
    if relations:
        for i, (head, rel_type, tail) in enumerate(relations, 1):
            relations_lines.append(f"{i}. {head} --[{rel_type}]--> {tail}")
    else:
        relations_lines.append("None found.")

    output = "\n".join(entities_lines + relations_lines)

    return {
        "instruction": instruction,
        "input": text,
        "output": output,
    }


def create_splits(data, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    """Split data into train/val/test sets."""
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    random.seed(seed)
    shuffled = list(data)
    random.shuffle(shuffled)

    n = len(shuffled)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    return {
        "train": shuffled[:train_end],
        "val": shuffled[train_end:val_end],
        "test": shuffled[val_end:],
    }


def main():
    parser = argparse.ArgumentParser(description="Convert ADE Corpus V2 to training format")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/datasets/ade_formatted",
        help="Output directory for formatted datasets",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splits")
    parser.add_argument(
        "--min_relations",
        type=int,
        default=1,
        help="Minimum relations per sentence to include",
    )
    parser.add_argument(
        "--train_ratio", type=float, default=0.8, help="Training set ratio"
    )
    parser.add_argument(
        "--val_ratio", type=float, default=0.1, help="Validation set ratio"
    )
    parser.add_argument(
        "--test_ratio", type=float, default=0.1, help="Test set ratio"
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=480,
        help="Maximum token length (skip longer examples)",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("CONVERTING ADE CORPUS V2 TO TRAINING FORMAT")
    print("=" * 80)

    # Load dataset
    dataset = load_ade_corpus()

    # Group by sentence
    print("\nGrouping by sentence...")
    sentence_data = group_by_sentence(dataset)
    print(f"  Unique sentences: {len(sentence_data)}")

    # Filter by minimum relations
    filtered = {
        text: data
        for text, data in sentence_data.items()
        if len(data["relations"]) >= args.min_relations
    }
    print(f"  After filtering (min {args.min_relations} relations): {len(filtered)}")

    # Format examples and filter by token length
    print("\nFormatting examples...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("unsloth/Meta-Llama-3.1-8B-bnb-4bit")
    max_tokens = args.max_tokens

    formatted = []
    skipped = 0
    for text, data in filtered.items():
        ex = format_example(text, data)
        full_text = ALPACA_TEMPLATE.format(**ex)
        n_tokens = len(tokenizer.encode(full_text))
        if n_tokens > max_tokens:
            skipped += 1
            continue
        formatted.append(ex)
    print(f"  Skipped {skipped} examples exceeding {max_tokens} tokens")

    # Stats
    total_entities = sum(
        len(sentence_data[text]["drugs"]) + len(sentence_data[text]["effects"])
        for text in filtered
    )
    total_relations = sum(len(sentence_data[text]["relations"]) for text in filtered)
    print(f"  Total examples: {len(formatted)}")
    print(f"  Total entities: {total_entities}")
    print(f"  Total relations: {total_relations}")
    print(f"  Avg entities/example: {total_entities / len(formatted):.1f}")
    print(f"  Avg relations/example: {total_relations / len(formatted):.1f}")

    # Text length stats
    input_lens = [len(ex["input"]) for ex in formatted]
    print(f"\n  Input length (chars): min={min(input_lens)}, max={max(input_lens)}, "
          f"mean={sum(input_lens)/len(input_lens):.0f}")

    # Create splits
    print(f"\nCreating splits ({args.train_ratio:.0%}/{args.val_ratio:.0%}/{args.test_ratio:.0%})...")
    splits = create_splits(
        formatted, args.train_ratio, args.val_ratio, args.test_ratio, args.seed
    )
    print(f"  Train: {len(splits['train'])}")
    print(f"  Val:   {len(splits['val'])}")
    print(f"  Test:  {len(splits['test'])}")

    # Save
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for split_name, split_data in splits.items():
        output_path = output_dir / f"{split_name}.json"
        with open(output_path, "w") as f:
            json.dump(split_data, f, indent=2)
        print(f"  Saved {output_path}")

    # Show example
    print("\n" + "=" * 80)
    print("EXAMPLE")
    print("=" * 80)
    example = splits["train"][0]
    print(f"\nInstruction:\n{example['instruction']}")
    print(f"\nInput:\n{example['input']}")
    print(f"\nOutput:\n{example['output']}")

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()
