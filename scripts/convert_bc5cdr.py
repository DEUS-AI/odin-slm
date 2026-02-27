#!/usr/bin/env python3
"""Convert BC5CDR corpus to our instruction-tuning format.

Downloads BC5CDR (BioCreative V CDR) PubTator files from HuggingFace Hub
and converts chemical-induced-disease (CID) relations to our format.

Entity mapping:
  - Chemical → Drug
  - Disease → Disease

Relation mapping:
  - CID → causes (chemical causes/induces disease)

Usage:
    python scripts/convert_bc5cdr.py --output_dir data/datasets/bc5cdr_formatted
"""

import argparse
import json
import zipfile
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

TYPE_MAP = {
    "Chemical": "Drug",
    "Disease": "Disease",
}


def download_bc5cdr():
    """Download BC5CDR PubTator files from HuggingFace Hub."""
    from huggingface_hub import hf_hub_download

    print("Downloading BC5CDR from HuggingFace Hub...")
    zip_path = hf_hub_download(
        repo_id="bigbio/bc5cdr",
        filename="CDR_Data.zip",
        repo_type="dataset",
    )
    return zip_path


def parse_pubtator(text):
    """Parse PubTator format into documents with entities and relations.

    Returns list of dicts with keys: pmid, title, abstract, entities, relations.
    """
    documents = []
    current_doc = None

    for line in text.strip().split("\n"):
        line = line.strip()
        if not line:
            if current_doc:
                documents.append(current_doc)
                current_doc = None
            continue

        if "|t|" in line:
            pmid, _, title = line.partition("|t|")
            current_doc = {
                "pmid": pmid,
                "title": title,
                "abstract": "",
                "entities": [],
                "mesh_to_texts": defaultdict(set),
                "relations": [],
            }
        elif "|a|" in line and current_doc:
            _, _, abstract = line.partition("|a|")
            current_doc["abstract"] = abstract
        elif "\t" in line and current_doc:
            parts = line.split("\t")
            if len(parts) >= 6 and parts[1] != "CID":
                # Entity annotation: PMID \t start \t end \t text \t type \t mesh_id
                _, start, end, ent_text, ent_type, mesh_id = parts[:6]
                if ent_type in TYPE_MAP:
                    current_doc["entities"].append({
                        "text": ent_text,
                        "type": ent_type,
                        "mesh_id": mesh_id,
                    })
                    current_doc["mesh_to_texts"][mesh_id].add(ent_text)
            elif len(parts) >= 4 and parts[1] == "CID":
                # Relation: PMID \t CID \t Chemical_MeSH \t Disease_MeSH
                _, _, chem_mesh, disease_mesh = parts[:4]
                current_doc["relations"].append({
                    "chemical_mesh": chem_mesh,
                    "disease_mesh": disease_mesh,
                })

    if current_doc:
        documents.append(current_doc)

    return documents


def format_document(doc):
    """Convert a BC5CDR document to instruction format.

    Returns None if no valid relations can be resolved.
    """
    full_text = doc["title"] + " " + doc["abstract"]

    # Deduplicate entities by (text, type)
    seen_entities = set()
    unique_entities = []
    for ent in doc["entities"]:
        key = (ent["text"].lower(), ent["type"])
        if key not in seen_entities:
            seen_entities.add(key)
            unique_entities.append(ent)

    # Resolve CID relations to text spans
    resolved_relations = set()
    for rel in doc["relations"]:
        chem_texts = doc["mesh_to_texts"].get(rel["chemical_mesh"], set())
        disease_texts = doc["mesh_to_texts"].get(rel["disease_mesh"], set())
        for chem in chem_texts:
            for disease in disease_texts:
                resolved_relations.add((chem, "causes", disease))

    if not resolved_relations:
        return None

    # Build entity list (drugs first, then diseases)
    drugs = sorted(set(
        ent["text"] for ent in unique_entities if ent["type"] == "Chemical"
    ))
    diseases = sorted(set(
        ent["text"] for ent in unique_entities if ent["type"] == "Disease"
    ))

    entities_lines = ["### Entities:"]
    idx = 1
    for drug in drugs:
        entities_lines.append(f"{idx}. [Drug] {drug}")
        idx += 1
    for disease in diseases:
        entities_lines.append(f"{idx}. [Disease] {disease}")
        idx += 1

    # Build relation lines
    relations = sorted(resolved_relations)
    relations_lines = ["", "### Relations:"]
    for i, (head, rel_type, tail) in enumerate(relations, 1):
        relations_lines.append(f"{i}. {head} --[{rel_type}]--> {tail}")

    output = "\n".join(entities_lines + relations_lines)

    return {
        "instruction": INSTRUCTION,
        "input": full_text,
        "output": output,
    }


def main():
    parser = argparse.ArgumentParser(description="Convert BC5CDR to training format")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/datasets/bc5cdr_formatted",
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
    print("CONVERTING BC5CDR TO TRAINING FORMAT")
    print("=" * 80)

    # Download
    zip_path = download_bc5cdr()

    # Parse all splits
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("unsloth/Meta-Llama-3.1-8B-bnb-4bit")

    split_files = {
        "train": "CDR_Data/CDR.Corpus.v010516/CDR_TrainingSet.PubTator.txt",
        "val": "CDR_Data/CDR.Corpus.v010516/CDR_DevelopmentSet.PubTator.txt",
        "test": "CDR_Data/CDR.Corpus.v010516/CDR_TestSet.PubTator.txt",
    }

    all_splits = {}
    total_docs = 0
    total_skipped_no_rel = 0
    total_skipped_tokens = 0

    with zipfile.ZipFile(zip_path, "r") as zf:
        for split_name, file_path in split_files.items():
            print(f"\nProcessing {split_name} split...")
            content = zf.read(file_path).decode("utf-8")
            documents = parse_pubtator(content)
            print(f"  Parsed {len(documents)} documents")
            total_docs += len(documents)

            formatted = []
            skipped_no_rel = 0
            skipped_tokens = 0

            for doc in documents:
                example = format_document(doc)
                if example is None:
                    skipped_no_rel += 1
                    continue

                # Token length filter
                full_text = ALPACA_TEMPLATE.format(**example)
                n_tokens = len(tokenizer.encode(full_text))
                if n_tokens > args.max_tokens:
                    skipped_tokens += 1
                    continue

                formatted.append(example)

            all_splits[split_name] = formatted
            total_skipped_no_rel += skipped_no_rel
            total_skipped_tokens += skipped_tokens
            print(f"  Kept: {len(formatted)}, Skipped (no relations): {skipped_no_rel}, "
                  f"Skipped (too long): {skipped_tokens}")

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

    print(f"\n  Total documents: {total_docs}")
    print(f"  Total examples: {total_examples}")
    print(f"  Skipped (no relations): {total_skipped_no_rel}")
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
