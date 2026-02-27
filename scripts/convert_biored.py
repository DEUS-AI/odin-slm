#!/usr/bin/env python3
"""Convert BioRED corpus to our instruction-tuning format.

Parses BioRED BioC JSON files (already downloaded) and converts entity/relation
annotations to our instruction-tuning format.

Entity mapping:
  - ChemicalEntity → Drug
  - DiseaseOrPhenotypicFeature → Disease
  - GeneOrGeneProduct → skipped (not in our schema)
  - OrganismTaxon → skipped
  - SequenceVariant → skipped
  - CellLine → skipped

Relation mapping:
  - Positive_Correlation → causes
  - Negative_Correlation → treats
  - Association → associated_with
  - Bind → interacts_with
  - Drug_Interaction → interacts_with
  - Cotreatment → associated_with
  - Comparison → skipped
  - Conversion → skipped

Usage:
    python scripts/convert_biored.py --output_dir data/datasets/biored_formatted
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

ALPACA_TEMPLATE = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}"""

INSTRUCTION = (
    "Extract all medical entities and their relations from the following "
    "clinical text. Identify diseases, symptoms, drugs, procedures, and "
    "lab tests, along with their relationships."
)

ENTITY_TYPE_MAP = {
    "ChemicalEntity": "Drug",
    "DiseaseOrPhenotypicFeature": "Disease",
}

RELATION_TYPE_MAP = {
    "Positive_Correlation": "causes",
    "Negative_Correlation": "treats",
    "Association": "associated_with",
    "Bind": "interacts_with",
    "Drug_Interaction": "interacts_with",
    "Cotreatment": "associated_with",
}


def parse_bioc_json(file_path):
    """Parse BioC JSON file and extract documents with entities and relations."""
    with open(file_path) as f:
        data = json.load(f)

    documents = []
    for doc in data["documents"]:
        pmid = doc["id"]

        # Combine title + abstract text
        passages = []
        for passage in doc["passages"]:
            passages.append(passage["text"])
        full_text = " ".join(passages)

        # Extract entities, mapping normalized IDs to text spans
        id_to_texts = defaultdict(set)
        id_to_type = {}
        entities = []
        seen_entities = set()

        for passage in doc["passages"]:
            for ann in passage.get("annotations", []):
                ent_type = ann["infons"]["type"]
                ent_id = ann["infons"].get("identifier", "")
                ent_text = ann["text"]

                if ent_type not in ENTITY_TYPE_MAP:
                    continue

                mapped_type = ENTITY_TYPE_MAP[ent_type]

                # Track ID → text mapping for relation resolution
                if ent_id:
                    id_to_texts[ent_id].add(ent_text)
                    id_to_type[ent_id] = mapped_type

                key = (ent_text.lower(), mapped_type)
                if key not in seen_entities:
                    seen_entities.add(key)
                    entities.append({"text": ent_text, "type": mapped_type})

        # Extract and resolve relations
        resolved_relations = set()
        for rel in doc.get("relations", []):
            rel_type = rel["infons"]["type"]
            if rel_type not in RELATION_TYPE_MAP:
                continue

            mapped_rel = RELATION_TYPE_MAP[rel_type]
            ent1_id = rel["infons"]["entity1"]
            ent2_id = rel["infons"]["entity2"]

            # Both entities must be in our type map
            if ent1_id not in id_to_type or ent2_id not in id_to_type:
                continue

            # Resolve IDs to text
            for text1 in id_to_texts.get(ent1_id, set()):
                for text2 in id_to_texts.get(ent2_id, set()):
                    resolved_relations.add((text1, mapped_rel, text2))

        if entities and resolved_relations:
            documents.append({
                "pmid": pmid,
                "text": full_text,
                "entities": entities,
                "relations": resolved_relations,
            })

    return documents


def format_document(doc):
    """Convert a BioRED document to instruction format."""
    # Group entities by type
    drugs = sorted(set(e["text"] for e in doc["entities"] if e["type"] == "Drug"))
    diseases = sorted(set(e["text"] for e in doc["entities"] if e["type"] == "Disease"))

    entities_lines = ["### Entities:"]
    idx = 1
    for drug in drugs:
        entities_lines.append(f"{idx}. [Drug] {drug}")
        idx += 1
    for disease in diseases:
        entities_lines.append(f"{idx}. [Disease] {disease}")
        idx += 1

    # Build relation lines
    relations = sorted(doc["relations"])
    relations_lines = ["", "### Relations:"]
    for i, (head, rel_type, tail) in enumerate(relations, 1):
        relations_lines.append(f"{i}. {head} --[{rel_type}]--> {tail}")

    output = "\n".join(entities_lines + relations_lines)

    return {
        "instruction": INSTRUCTION,
        "input": doc["text"],
        "output": output,
    }


def main():
    parser = argparse.ArgumentParser(description="Convert BioRED to training format")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/raw/biored/BioRED",
        help="Directory containing BioRED BioC JSON files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/datasets/biored_formatted",
        help="Output directory for formatted datasets",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=480,
        help="Maximum token length (skip longer examples)",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("CONVERTING BioRED TO TRAINING FORMAT")
    print("=" * 80)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("unsloth/Meta-Llama-3.1-8B-bnb-4bit")

    data_dir = Path(args.data_dir)
    split_files = {
        "train": data_dir / "Train.BioC.JSON",
        "val": data_dir / "Dev.BioC.JSON",
        "test": data_dir / "Test.BioC.JSON",
    }

    all_splits = {}
    total_docs = 0
    total_skipped_tokens = 0

    for split_name, file_path in split_files.items():
        print(f"\nProcessing {split_name} split...")
        documents = parse_bioc_json(file_path)
        print(f"  Documents with Drug/Disease entities and relations: {len(documents)}")
        total_docs += len(documents)

        formatted = []
        skipped_tokens = 0

        for doc in documents:
            example = format_document(doc)

            # Token length filter
            full_text = ALPACA_TEMPLATE.format(**example)
            n_tokens = len(tokenizer.encode(full_text))
            if n_tokens > args.max_tokens:
                skipped_tokens += 1
                continue

            formatted.append(example)

        all_splits[split_name] = formatted
        total_skipped_tokens += skipped_tokens
        print(f"  Kept: {len(formatted)}, Skipped (too long): {skipped_tokens}")

    # Save
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    total_examples = 0
    for split_name, split_data in all_splits.items():
        output_path = output_dir / f"{split_name}.json"
        with open(output_path, "w") as f:
            json.dump(split_data, f, indent=2)
        print(f"  Saved {output_path} ({len(split_data)} examples)")
        total_examples += len(split_data)

    print(f"\n  Total documents with valid entities+relations: {total_docs}")
    print(f"  Total examples after token filter: {total_examples}")
    print(f"  Skipped (too long): {total_skipped_tokens}")

    # Show example
    if all_splits["train"]:
        print("\n" + "=" * 80)
        print("EXAMPLE")
        print("=" * 80)
        example = all_splits["train"][0]
        print(f"\nInput:\n{example['input'][:300]}...")
        print(f"\nOutput:\n{example['output']}")

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()
