#!/usr/bin/env python3
"""Generate synthetic medical NER/RE dataset

Usage:
    python scripts/generate_medical_dataset.py --num_samples 1000 --output data/datasets/medical_ner_re.json
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from odin_slm.data.synthetic_generator import MedicalTextGenerator
from odin_slm.data.evaluator import NERREvaluator, EntityPrediction


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic medical NER/RE dataset")
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1000,
        help="Number of synthetic documents to generate",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/datasets/medical_ner_re.json",
        help="Output path for dataset JSON file",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--min_entities", type=int, default=3, help="Minimum entities per document"
    )
    parser.add_argument(
        "--max_entities", type=int, default=8, help="Maximum entities per document"
    )

    args = parser.parse_args()

    print("=" * 70)
    print("Medical NER/RE Synthetic Dataset Generation")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Samples: {args.num_samples}")
    print(f"  Output: {args.output}")
    print(f"  Seed: {args.seed}")
    print(f"  Entities per doc: {args.min_entities}-{args.max_entities}")
    print()

    # Initialize generator
    generator = MedicalTextGenerator(seed=args.seed)

    # Generate dataset
    print("Generating dataset...")
    documents = []

    import random

    random.seed(args.seed)

    for i in range(args.num_samples):
        num_entities = random.randint(args.min_entities, args.max_entities)
        num_relations = random.randint(2, min(5, num_entities - 1))

        doc = generator.generate_clinical_note(num_entities, num_relations)
        documents.append(doc)

        if (i + 1) % 100 == 0:
            print(f"  Generated {i + 1}/{args.num_samples} documents")

    print(f"\n✓ Generated {len(documents)} documents")

    # Statistics
    total_entities = sum(len(doc.entities) for doc in documents)
    total_relations = sum(len(doc.relations) for doc in documents)

    print(f"\nDataset Statistics:")
    print(f"  Total entities: {total_entities}")
    print(f"  Total relations: {total_relations}")
    print(f"  Avg entities per doc: {total_entities / len(documents):.2f}")
    print(f"  Avg relations per doc: {total_relations / len(documents):.2f}")

    # Entity type distribution
    entity_types = {}
    for doc in documents:
        for entity in doc.entities:
            entity_types[entity.type] = entity_types.get(entity.type, 0) + 1

    print(f"\nEntity Type Distribution:")
    for entity_type, count in sorted(entity_types.items(), key=lambda x: -x[1]):
        print(f"  {entity_type:15s}: {count:5d} ({count/total_entities*100:5.1f}%)")

    # Relation type distribution
    relation_types = {}
    for doc in documents:
        for relation in doc.relations:
            relation_types[relation.type] = relation_types.get(relation.type, 0) + 1

    print(f"\nRelation Type Distribution:")
    for rel_type, count in sorted(relation_types.items(), key=lambda x: -x[1]):
        print(f"  {rel_type:15s}: {count:5d} ({count/total_relations*100:5.1f}%)")

    # Save dataset
    print(f"\nSaving dataset to {args.output}...")
    generator.save_dataset(documents, Path(args.output))

    print("\n" + "=" * 70)
    print("✓ Dataset generation complete!")
    print("=" * 70)

    # Show example
    print("\nExample Document:")
    print("-" * 70)
    example = documents[0]
    print(f"Text: {example.text[:200]}...")
    print(f"\nEntities ({len(example.entities)}):")
    for entity in example.entities[:5]:
        print(f"  [{entity.type}] {entity.text}")
    print(f"\nRelations ({len(example.relations)}):")
    for rel in example.relations[:3]:
        print(f"  {rel.head.text} --[{rel.type}]--> {rel.tail.text}")


if __name__ == "__main__":
    main()
