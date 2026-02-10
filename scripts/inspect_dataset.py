#!/usr/bin/env python3
"""Inspect medical NER/RE dataset and show detailed examples

Usage:
    python scripts/inspect_dataset.py --input data/datasets/medical_ner_re_train.json
"""

import argparse
import sys
from pathlib import Path
from collections import Counter
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from odin_slm.data.synthetic_generator import MedicalTextGenerator


def print_document_examples(docs, num_examples=5):
    """Print detailed examples of documents"""
    print("\n" + "=" * 80)
    print(f"DOCUMENT EXAMPLES (showing {num_examples} of {len(docs)})")
    print("=" * 80)

    for i, doc in enumerate(docs[:num_examples], 1):
        print(f"\n{'─' * 80}")
        print(f"Example {i}")
        print('─' * 80)

        # Text
        print(f"\n📄 TEXT:")
        print(f'   "{doc.text}"')

        # Entities
        print(f"\n🏷️  ENTITIES ({len(doc.entities)}):")
        for j, entity in enumerate(doc.entities, 1):
            print(
                f"   {j}. [{entity.type:12s}] '{entity.text}' "
                f"(position: {entity.start}-{entity.end})"
            )

        # Relations
        print(f"\n🔗 RELATIONS ({len(doc.relations)}):")
        if doc.relations:
            for j, rel in enumerate(doc.relations, 1):
                print(
                    f"   {j}. {rel.head.text:25s} --[{rel.type:15s}]--> "
                    f"{rel.tail.text:25s} (conf: {rel.confidence:.2f})"
                )
        else:
            print("   (none)")


def analyze_entity_patterns(docs):
    """Analyze entity patterns and co-occurrences"""
    print("\n" + "=" * 80)
    print("ENTITY PATTERN ANALYSIS")
    print("=" * 80)

    # Entity type combinations
    type_combinations = Counter()
    for doc in docs:
        types = tuple(sorted(set(e.type for e in doc.entities)))
        type_combinations[types] += 1

    print("\n📊 Most Common Entity Type Combinations:")
    for combo, count in type_combinations.most_common(10):
        print(f"   {', '.join(combo):60s} : {count:4d} docs")

    # Entity text frequencies
    entity_texts = Counter()
    for doc in docs:
        for entity in doc.entities:
            entity_texts[f"[{entity.type}] {entity.text}"] += 1

    print("\n🔝 Most Frequent Entities:")
    for entity, count in entity_texts.most_common(20):
        print(f"   {entity:50s} : {count:4d} occurrences")


def analyze_relation_patterns(docs):
    """Analyze relation patterns"""
    print("\n" + "=" * 80)
    print("RELATION PATTERN ANALYSIS")
    print("=" * 80)

    # Relation type patterns (head_type -> tail_type)
    relation_patterns = Counter()
    for doc in docs:
        for rel in doc.relations:
            pattern = f"{rel.head.type} --[{rel.type}]--> {rel.tail.type}"
            relation_patterns[pattern] += 1

    print("\n🔗 Relation Type Patterns:")
    for pattern, count in relation_patterns.most_common(15):
        print(f"   {pattern:60s} : {count:4d} occurrences")

    # Specific relation instances
    specific_relations = Counter()
    for doc in docs:
        for rel in doc.relations:
            instance = f"{rel.head.text} --[{rel.type}]--> {rel.tail.text}"
            specific_relations[instance] += 1

    print("\n💊 Most Common Specific Relations:")
    for relation, count in specific_relations.most_common(20):
        print(f"   {relation:70s} : {count:3d}x")


def analyze_text_characteristics(docs):
    """Analyze text characteristics"""
    print("\n" + "=" * 80)
    print("TEXT CHARACTERISTICS")
    print("=" * 80)

    text_lengths = [len(doc.text) for doc in docs]
    word_counts = [len(doc.text.split()) for doc in docs]

    print(f"\n📏 Text Length Statistics:")
    print(f"   Min length:     {min(text_lengths):5d} characters")
    print(f"   Max length:     {max(text_lengths):5d} characters")
    print(f"   Mean length:    {sum(text_lengths)/len(text_lengths):5.1f} characters")
    print(f"   Median length:  {sorted(text_lengths)[len(text_lengths)//2]:5d} characters")

    print(f"\n📝 Word Count Statistics:")
    print(f"   Min words:      {min(word_counts):5d} words")
    print(f"   Max words:      {max(word_counts):5d} words")
    print(f"   Mean words:     {sum(word_counts)/len(word_counts):5.1f} words")
    print(f"   Median words:   {sorted(word_counts)[len(word_counts)//2]:5d} words")

    # Entity density
    entity_densities = [len(doc.entities) / max(len(doc.text.split()), 1) for doc in docs]
    print(f"\n🏷️  Entity Density (entities per word):")
    print(f"   Mean:           {sum(entity_densities)/len(entity_densities):.3f}")

    # Relation density
    relation_densities = [len(doc.relations) / max(len(doc.entities), 1) for doc in docs]
    print(f"\n🔗 Relation Density (relations per entity):")
    print(f"   Mean:           {sum(relation_densities)/len(relation_densities):.3f}")


def show_diverse_examples(docs):
    """Show examples with different characteristics"""
    print("\n" + "=" * 80)
    print("DIVERSE EXAMPLES")
    print("=" * 80)

    # Find examples with different characteristics
    max_entities = max(docs, key=lambda d: len(d.entities))
    max_relations = max(docs, key=lambda d: len(d.relations))
    longest_text = max(docs, key=lambda d: len(d.text))

    examples = [
        ("Most Entities", max_entities),
        ("Most Relations", max_relations),
        ("Longest Text", longest_text),
    ]

    for title, doc in examples:
        print(f"\n{'─' * 80}")
        print(f"📌 {title}")
        print('─' * 80)
        print(f"\nText: {doc.text}")
        print(f"\nEntities: {len(doc.entities)}")
        for entity in doc.entities:
            print(f"  - [{entity.type}] {entity.text}")
        print(f"\nRelations: {len(doc.relations)}")
        for rel in doc.relations:
            print(f"  - {rel.head.text} --[{rel.type}]--> {rel.tail.text}")


def export_sample_for_manual_review(docs, output_path, n=50):
    """Export a sample for manual review"""
    import random

    sample = random.sample(docs, min(n, len(docs)))

    # Convert to simple format
    review_data = []
    for doc in sample:
        review_data.append(
            {
                "text": doc.text,
                "entities": [
                    {"text": e.text, "type": e.type, "start": e.start, "end": e.end}
                    for e in doc.entities
                ],
                "relations": [
                    {
                        "type": r.type,
                        "head": r.head.text,
                        "tail": r.tail.text,
                    }
                    for r in doc.relations
                ],
                "review_notes": "",  # For manual annotation
            }
        )

    with open(output_path, "w") as f:
        json.dump(review_data, f, indent=2)

    print(f"\n✓ Exported {len(review_data)} samples to {output_path} for manual review")


def main():
    parser = argparse.ArgumentParser(description="Inspect medical NER/RE dataset")
    parser.add_argument(
        "--input",
        type=str,
        default="data/datasets/medical_ner_re_train.json",
        help="Input dataset JSON file",
    )
    parser.add_argument(
        "--num_examples", type=int, default=5, help="Number of examples to show"
    )
    parser.add_argument(
        "--export_sample",
        type=str,
        help="Export sample for manual review (path)",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("MEDICAL NER/RE DATASET INSPECTION")
    print("=" * 80)
    print(f"\nDataset: {args.input}")

    # Load dataset
    print("\nLoading dataset...")
    generator = MedicalTextGenerator()
    docs = generator.load_dataset(args.input)
    print(f"✓ Loaded {len(docs)} documents")

    # Show examples
    print_document_examples(docs, args.num_examples)

    # Analyze patterns
    analyze_entity_patterns(docs)
    analyze_relation_patterns(docs)
    analyze_text_characteristics(docs)

    # Show diverse examples
    show_diverse_examples(docs)

    # Export sample if requested
    if args.export_sample:
        export_sample_for_manual_review(docs, args.export_sample)

    print("\n" + "=" * 80)
    print("✓ INSPECTION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
