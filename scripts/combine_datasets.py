#!/usr/bin/env python3
"""Combine multiple formatted datasets into a single training dataset.

Merges real annotated data (ADE, BC5CDR, BioRED) with an optional subsample
of synthetic data to create a combined dataset with broad entity/relation coverage.

Usage:
    python scripts/combine_datasets.py --output_dir data/datasets/combined_v14
    python scripts/combine_datasets.py --output_dir data/datasets/combined_v14 --synthetic_ratio 0.25
"""

import argparse
import json
import random
from pathlib import Path


def load_split(dataset_dir, split_name):
    """Load a split from a dataset directory."""
    path = Path(dataset_dir) / f"{split_name}.json"
    if not path.exists():
        return []
    with open(path) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Combine datasets for training")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/datasets/combined_v14",
        help="Output directory",
    )
    parser.add_argument(
        "--synthetic_ratio",
        type=float,
        default=0.25,
        help="Ratio of synthetic data relative to total real data (0.25 = 25%%)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--ade_dir", type=str, default="data/datasets/ade_formatted", help="ADE dataset dir")
    parser.add_argument("--bc5cdr_dir", type=str, default="data/datasets/bc5cdr_formatted", help="BC5CDR dataset dir")
    parser.add_argument("--biored_dir", type=str, default="data/datasets/biored_formatted", help="BioRED dataset dir")
    parser.add_argument("--synthetic_dir", type=str, default="data/datasets/formatted_v2", help="Synthetic dataset dir")
    args = parser.parse_args()

    random.seed(args.seed)

    print("=" * 80)
    print("COMBINING DATASETS")
    print("=" * 80)

    # Define source datasets
    real_datasets = {
        "ADE Corpus V2": args.ade_dir,
        "BC5CDR": args.bc5cdr_dir,
        "BioRED": args.biored_dir,
    }
    synthetic_dataset = args.synthetic_dir

    for split in ["train", "val", "test"]:
        print(f"\n--- {split.upper()} split ---")
        combined = []

        # Add all real data
        total_real = 0
        for name, path in real_datasets.items():
            data = load_split(path, split)
            combined.extend(data)
            total_real += len(data)
            print(f"  {name}: {len(data)}")

        # Add synthetic subsample
        synthetic = load_split(synthetic_dataset, split)
        n_synthetic = int(total_real * args.synthetic_ratio)
        if n_synthetic > 0 and synthetic:
            n_synthetic = min(n_synthetic, len(synthetic))
            sampled = random.sample(synthetic, n_synthetic)
            combined.extend(sampled)
            print(f"  Synthetic (sampled): {n_synthetic} / {len(synthetic)}")
        else:
            print(f"  Synthetic: 0")

        # Shuffle
        random.shuffle(combined)
        print(f"  TOTAL: {len(combined)}")

        # Save
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{split}.json"
        with open(output_path, "w") as f:
            json.dump(combined, f, indent=2)

    # Summary stats
    print(f"\n{'='*80}")
    print("DATASET COMPOSITION")
    print(f"{'='*80}")
    for split in ["train", "val", "test"]:
        with open(Path(args.output_dir) / f"{split}.json") as f:
            data = json.load(f)

        # Count entity types and relation types
        entity_types = {}
        relation_types = {}
        import re
        for ex in data:
            for line in ex["output"].split("\n"):
                m = re.search(r"\[(\w+(?:_\w+)*)\]", line)
                if m and not line.strip().startswith("###"):
                    if "--[" in line:
                        t = m.group(1)
                        relation_types[t] = relation_types.get(t, 0) + 1
                    else:
                        t = m.group(1)
                        entity_types[t] = entity_types.get(t, 0) + 1

        print(f"\n{split}: {len(data)} examples")
        print(f"  Entity types: {dict(sorted(entity_types.items()))}")
        print(f"  Relation types: {dict(sorted(relation_types.items()))}")

    print(f"\nSaved to: {args.output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
