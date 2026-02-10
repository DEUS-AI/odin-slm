#!/usr/bin/env python3
"""Analyze dataset quality and identify potential issues

Usage:
    python scripts/analyze_dataset_quality.py --input data/datasets/medical_ner_re_train.json
"""

import argparse
import sys
from pathlib import Path
from collections import Counter, defaultdict

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from odin_slm.data.synthetic_generator import MedicalTextGenerator


def check_entity_quality(docs):
    """Check for entity annotation issues"""
    print("\n" + "=" * 80)
    print("ENTITY QUALITY CHECKS")
    print("=" * 80)

    issues = {
        "overlapping_entities": [],
        "duplicate_entities": [],
        "empty_text": [],
        "misaligned_positions": [],
    }

    for doc_id, doc in enumerate(docs):
        # Check for overlapping entities
        entities_sorted = sorted(doc.entities, key=lambda e: e.start)
        for i in range(len(entities_sorted) - 1):
            if entities_sorted[i].end > entities_sorted[i + 1].start:
                issues["overlapping_entities"].append(
                    (doc_id, entities_sorted[i].text, entities_sorted[i + 1].text)
                )

        # Check for duplicate entities
        entity_spans = [(e.start, e.end, e.type) for e in doc.entities]
        if len(entity_spans) != len(set(entity_spans)):
            issues["duplicate_entities"].append(doc_id)

        # Check for empty text
        for entity in doc.entities:
            if not entity.text.strip():
                issues["empty_text"].append((doc_id, entity))

        # Check position alignment
        for entity in doc.entities:
            actual_text = doc.text[entity.start : entity.end]
            if actual_text != entity.text:
                issues["misaligned_positions"].append(
                    (doc_id, entity.text, actual_text, entity.start, entity.end)
                )

    # Report issues
    print(f"\n✓ Overlapping entities: {len(issues['overlapping_entities'])}")
    if issues["overlapping_entities"][:3]:
        for doc_id, e1, e2 in issues["overlapping_entities"][:3]:
            print(f"   - Doc {doc_id}: '{e1}' overlaps with '{e2}'")

    print(f"\n✓ Duplicate entities: {len(issues['duplicate_entities'])}")
    if issues["duplicate_entities"][:3]:
        print(f"   - In documents: {issues['duplicate_entities'][:3]}")

    print(f"\n✓ Empty entity text: {len(issues['empty_text'])}")

    print(f"\n✓ Position misalignments: {len(issues['misaligned_positions'])}")
    if issues["misaligned_positions"][:3]:
        for doc_id, expected, actual, start, end in issues["misaligned_positions"][:3]:
            print(f"   - Doc {doc_id}: Expected '{expected}', got '{actual}' at {start}:{end}")

    return issues


def check_relation_quality(docs):
    """Check for relation annotation issues"""
    print("\n" + "=" * 80)
    print("RELATION QUALITY CHECKS")
    print("=" * 80)

    issues = {
        "invalid_entity_refs": [],
        "self_loops": [],
        "duplicate_relations": [],
    }

    for doc_id, doc in enumerate(docs):
        entity_ids = {id(e) for e in doc.entities}

        for rel in doc.relations:
            # Check if relation entities exist in document
            if id(rel.head) not in entity_ids or id(rel.tail) not in entity_ids:
                issues["invalid_entity_refs"].append((doc_id, rel.type))

            # Check for self-loops
            if rel.head == rel.tail:
                issues["self_loops"].append((doc_id, rel.head.text, rel.type))

        # Check for duplicate relations
        relation_tuples = [
            (rel.type, rel.head.text, rel.tail.text) for rel in doc.relations
        ]
        if len(relation_tuples) != len(set(relation_tuples)):
            issues["duplicate_relations"].append(doc_id)

    # Report issues
    print(f"\n✓ Invalid entity references: {len(issues['invalid_entity_refs'])}")
    print(f"\n✓ Self-loops (entity relates to itself): {len(issues['self_loops'])}")
    if issues["self_loops"][:5]:
        for doc_id, entity, rel_type in issues["self_loops"][:5]:
            print(f"   - Doc {doc_id}: {entity} --[{rel_type}]--> {entity}")

    print(f"\n✓ Duplicate relations: {len(issues['duplicate_relations'])}")
    if issues["duplicate_relations"][:3]:
        print(f"   - In documents: {issues['duplicate_relations'][:3]}")

    return issues


def check_diversity(docs):
    """Check dataset diversity"""
    print("\n" + "=" * 80)
    print("DIVERSITY ANALYSIS")
    print("=" * 80)

    # Unique texts
    unique_texts = len(set(doc.text for doc in docs))
    print(f"\n📝 Text Diversity:")
    print(f"   Total documents:    {len(docs)}")
    print(f"   Unique texts:       {unique_texts}")
    print(f"   Duplicates:         {len(docs) - unique_texts}")
    print(f"   Uniqueness ratio:   {unique_texts / len(docs) * 100:.1f}%")

    # Entity vocabulary
    entity_vocab = set()
    for doc in docs:
        for entity in doc.entities:
            entity_vocab.add((entity.type, entity.text))

    print(f"\n🏷️  Entity Vocabulary:")
    print(f"   Unique entity instances: {len(entity_vocab)}")

    # Relation patterns
    relation_patterns = set()
    for doc in docs:
        for rel in doc.relations:
            relation_patterns.add((rel.type, rel.head.type, rel.tail.type))

    print(f"\n🔗 Relation Patterns:")
    print(f"   Unique patterns:    {len(relation_patterns)}")
    print("\n   Patterns found:")
    for pattern in sorted(relation_patterns):
        print(f"      {pattern[1]:12s} --[{pattern[0]:15s}]--> {pattern[2]:12s}")


def check_medical_validity(docs):
    """Check for potential medical validity issues"""
    print("\n" + "=" * 80)
    print("MEDICAL VALIDITY CHECKS (HEURISTIC)")
    print("=" * 80)

    # Check for unlikely relations
    unlikely_relations = []

    # Rules for unlikely relations
    for doc_id, doc in enumerate(docs):
        for rel in doc.relations:
            # Drug treating symptom (should treat disease)
            if rel.type == "treats" and rel.head.type == "Drug" and rel.tail.type == "Symptom":
                unlikely_relations.append(
                    (doc_id, "Drug treats Symptom", f"{rel.head.text} treats {rel.tail.text}")
                )

            # Lab test causes something (tests don't cause things)
            if rel.type == "causes" and rel.head.type == "Lab_Test":
                unlikely_relations.append(
                    (doc_id, "Lab_Test causes", f"{rel.head.text} causes {rel.tail.text}")
                )

            # Symptom treats something
            if rel.type == "treats" and rel.head.type == "Symptom":
                unlikely_relations.append(
                    (
                        doc_id,
                        "Symptom treats",
                        f"{rel.head.text} treats {rel.tail.text}",
                    )
                )

    print(f"\n⚠️  Potentially Unlikely Relations: {len(unlikely_relations)}")
    if unlikely_relations[:10]:
        print("\n   Examples:")
        for doc_id, rule, example in unlikely_relations[:10]:
            print(f"   - Doc {doc_id} ({rule}): {example}")

    if not unlikely_relations:
        print("   ✓ No obvious medical validity issues found!")


def generate_quality_report(docs, output_path=None):
    """Generate comprehensive quality report"""
    print("\n" + "=" * 80)
    print("QUALITY SCORE SUMMARY")
    print("=" * 80)

    # Calculate scores
    entity_issues = check_entity_quality(docs)
    relation_issues = check_relation_quality(docs)

    # Count total issues
    total_entity_issues = sum(len(v) for v in entity_issues.values())
    total_relation_issues = sum(len(v) for v in relation_issues.values())

    # Calculate quality score
    max_possible_issues = len(docs) * 5  # Arbitrary scaling
    entity_score = max(0, 100 - (total_entity_issues / max_possible_issues * 100))
    relation_score = max(0, 100 - (total_relation_issues / max_possible_issues * 100))
    overall_score = (entity_score + relation_score) / 2

    print(f"\n📊 Quality Scores:")
    print(f"   Entity Quality:     {entity_score:.1f}/100")
    print(f"   Relation Quality:   {relation_score:.1f}/100")
    print(f"   Overall Quality:    {overall_score:.1f}/100")

    # Interpretation
    print(f"\n💡 Interpretation:")
    if overall_score >= 90:
        print("   ✓ Excellent quality - Ready for training")
    elif overall_score >= 75:
        print("   ✓ Good quality - Minor issues may need attention")
    elif overall_score >= 60:
        print("   ⚠️  Fair quality - Review and fix issues before training")
    else:
        print("   ❌ Poor quality - Significant issues need fixing")

    if output_path:
        import json

        report = {
            "overall_score": overall_score,
            "entity_score": entity_score,
            "relation_score": relation_score,
            "entity_issues": {k: len(v) for k, v in entity_issues.items()},
            "relation_issues": {k: len(v) for k, v in relation_issues.items()},
        }

        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)

        print(f"\n✓ Quality report saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze medical NER/RE dataset quality")
    parser.add_argument(
        "--input",
        type=str,
        default="data/datasets/medical_ner_re_train.json",
        help="Input dataset JSON file",
    )
    parser.add_argument(
        "--output_report",
        type=str,
        help="Output path for quality report JSON",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("MEDICAL NER/RE DATASET QUALITY ANALYSIS")
    print("=" * 80)
    print(f"\nDataset: {args.input}")

    # Load dataset
    print("\nLoading dataset...")
    generator = MedicalTextGenerator()
    docs = generator.load_dataset(args.input)
    print(f"✓ Loaded {len(docs)} documents")

    # Run quality checks
    check_entity_quality(docs)
    check_relation_quality(docs)
    check_diversity(docs)
    check_medical_validity(docs)

    # Generate overall quality score
    generate_quality_report(docs, args.output_report)

    print("\n" + "=" * 80)
    print("✓ QUALITY ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
