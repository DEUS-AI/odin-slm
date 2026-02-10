#!/usr/bin/env python3
"""Post-processing script to deduplicate relation predictions"""

def deduplicate_relations(relations):
    """Remove duplicate relations from predictions"""
    seen = set()
    unique_relations = []

    for rel in relations:
        rel_tuple = (rel['head'].lower(), rel['relation'], rel['tail'].lower())
        if rel_tuple not in seen:
            seen.add(rel_tuple)
            unique_relations.append(rel)

    return unique_relations


def filter_by_confidence(relations, entities, min_confidence=0.5):
    """Filter relations where both entities exist in the text"""
    entity_texts = {e['text'].lower() for e in entities}

    valid_relations = []
    for rel in relations:
        if rel['head'].lower() in entity_texts and rel['tail'].lower() in entity_texts:
            valid_relations.append(rel)

    return valid_relations


if __name__ == "__main__":
    # Example usage
    test_relations = [
        {'head': 'metformin', 'relation': 'treats', 'tail': 'diabetes'},
        {'head': 'metformin', 'relation': 'treats', 'tail': 'diabetes'},  # Duplicate
        {'head': 'Metformin', 'relation': 'treats', 'tail': 'Diabetes'},  # Case duplicate
        {'head': 'aspirin', 'relation': 'treats', 'tail': 'pain'},
    ]

    test_entities = [
        {'text': 'metformin', 'type': 'Drug'},
        {'text': 'diabetes', 'type': 'Disease'},
        {'text': 'aspirin', 'type': 'Drug'},
        {'text': 'pain', 'type': 'Symptom'},
    ]

    print("Original relations:", len(test_relations))
    deduped = deduplicate_relations(test_relations)
    print("After deduplication:", len(deduped))

    for rel in deduped:
        print(f"  - {rel['head']} --[{rel['relation']}]--> {rel['tail']}")
