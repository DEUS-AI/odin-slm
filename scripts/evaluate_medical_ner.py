#!/usr/bin/env python3
"""Evaluate trained medical NER/RE model on test set

Usage:
    python scripts/evaluate_medical_ner.py --model experiments/medical_ner_re/final_model --test_data data/datasets/formatted/test.json
"""

import warnings
warnings.filterwarnings("ignore", category=SyntaxWarning)

import argparse
import json
import re
from pathlib import Path
from typing import List, Dict, Tuple
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from unsloth import FastLanguageModel
import torch
from tqdm import tqdm


def parse_output_format(text: str) -> Tuple[List[Dict], List[Dict]]:
    """Parse model output to extract entities and relations

    Expected format:
    ### Entities:
    1. [Type] entity_text
    2. [Type] entity_text

    ### Relations:
    1. entity1 --[relation]--> entity2
    """
    entities = []
    relations = []

    lines = text.strip().split('\n')
    current_section = None

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Detect sections
        if 'entities' in line.lower() and line.startswith('#'):
            current_section = 'entities'
            continue
        elif 'relations' in line.lower() and line.startswith('#'):
            current_section = 'relations'
            continue

        # Parse entities: 1. [Type] text
        if current_section == 'entities' and re.match(r'^\d+\.', line):
            match = re.match(r'^\d+\.\s*\[(\w+)\]\s*(.+)', line)
            if match:
                entity_type, text = match.groups()
                entities.append({
                    'text': text.strip(),
                    'type': entity_type.strip()
                })

        # Parse relations: 1. entity1 --[relation]--> entity2
        elif current_section == 'relations' and re.match(r'^\d+\.', line):
            # Handle both --[rel]--> and --> formats
            match = re.match(r'^\d+\.\s*(.+?)\s*--\[(.+?)\]-->\s*(.+)', line)
            if not match:
                # Try alternate format without brackets
                match = re.match(r'^\d+\.\s*(.+?)\s*-->\s*(.+)', line)
                if match:
                    head, tail = match.groups()
                    # Fix: Don't skip! Add with 'unknown' relation type instead of silently losing it
                    relations.append({
                        'head': head.strip(),
                        'relation': 'unknown',
                        'tail': tail.strip()
                    })
                    continue

            if match and len(match.groups()) == 3:
                head, rel_type, tail = match.groups()
                relations.append({
                    'head': head.strip(),
                    'relation': rel_type.strip(),
                    'tail': tail.strip()
                })

    return entities, relations


def extract_ground_truth(example: Dict) -> Tuple[List[Dict], List[Dict]]:
    """Extract ground truth entities and relations from test example"""
    output_text = example['output']
    return parse_output_format(output_text)


def generate_prediction(model, tokenizer, example: Dict, max_new_tokens: int = 512) -> str:
    """Generate prediction for a single example"""

    # Format prompt
    prompt = f"""### Instruction:
{example['instruction']}

### Input:
{example['input']}

### Output:
"""

    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to("cuda")

    # Fix: Set random seeds for reproducible, deterministic generation
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)

    # Generate with deterministic decoding (do_sample=False for reproducibility)
    # NOTE: repetition_penalty and no_repeat_ngram_size were REMOVED in V10 -
    # they sabotage entity extraction by preventing the model from copying
    # entity text verbatim from the input (which is exactly what NER requires)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Greedy decoding for determinism
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract only the generated output (after "### Output:")
    if "### Output:" in full_output:
        prediction = full_output.split("### Output:")[-1].strip()
    else:
        prediction = full_output

    return prediction


def deduplicate_relations(relations: List[Dict]) -> List[Dict]:
    """Remove duplicate relations from predictions"""
    seen = set()
    unique_relations = []

    for rel in relations:
        # Fix: Lowercase relation type for consistency with matching logic
        rel_tuple = (rel['head'].lower(), rel['relation'].lower(), rel['tail'].lower())
        if rel_tuple not in seen:
            seen.add(rel_tuple)
            unique_relations.append(rel)

    return unique_relations


def calculate_entity_metrics(pred_entities: List[Dict], gold_entities: List[Dict]) -> Dict:
    """Calculate precision, recall, F1 for entities (exact match)"""

    # Convert to sets of (text, type) tuples for exact matching
    # Fix: Lowercase entity TYPE to avoid case sensitivity issues
    pred_set = {(e['text'].lower(), e['type'].lower()) for e in pred_entities}
    gold_set = {(e['text'].lower(), e['type'].lower()) for e in gold_entities}

    # Calculate metrics
    true_positives = len(pred_set & gold_set)
    false_positives = len(pred_set - gold_set)
    false_negatives = len(gold_set - pred_set)

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': true_positives,
        'fp': false_positives,
        'fn': false_negatives
    }


def calculate_relation_metrics(pred_relations: List[Dict], gold_relations: List[Dict]) -> Dict:
    """Calculate precision, recall, F1 for relations"""

    # Convert to sets of (head, relation, tail) tuples
    # Fix: Lowercase relation TYPE to avoid case sensitivity issues
    pred_set = {(r['head'].lower(), r['relation'].lower(), r['tail'].lower()) for r in pred_relations}
    gold_set = {(r['head'].lower(), r['relation'].lower(), r['tail'].lower()) for r in gold_relations}

    # Calculate metrics
    true_positives = len(pred_set & gold_set)
    false_positives = len(pred_set - gold_set)
    false_negatives = len(gold_set - pred_set)

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': true_positives,
        'fp': false_positives,
        'fn': false_negatives
    }


def evaluate_model(model_path: str, test_data_path: str, num_samples: int = None):
    """Main evaluation function"""

    print("=" * 80)
    print("MEDICAL NER/RE MODEL EVALUATION")
    print("=" * 80)

    # Load model
    print(f"\nLoading model from {model_path}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=512,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)  # Enable inference mode
    print("✓ Model loaded")

    # Load test data
    print(f"\nLoading test data from {test_data_path}...")
    with open(test_data_path, 'r') as f:
        test_data = json.load(f)

    if num_samples:
        test_data = test_data[:num_samples]

    print(f"✓ Loaded {len(test_data)} test examples")

    # Run predictions
    print("\n🔮 Generating predictions...")
    predictions = []
    entity_metrics_list = []
    relation_metrics_list = []

    for i, example in enumerate(tqdm(test_data, desc="Evaluating")):
        # Generate prediction
        pred_text = generate_prediction(model, tokenizer, example)

        # Parse prediction
        pred_entities, pred_relations = parse_output_format(pred_text)

        # Fix: Add validation logging for debugging parsing issues
        if len(pred_entities) == 0 and "### Entities:" in pred_text:
            print(f"⚠️  WARNING: No entities parsed from output (example {i})")
        if len(pred_relations) == 0 and "### Relations:" in pred_text and "None found" not in pred_text:
            print(f"⚠️  WARNING: No relations parsed from output (example {i})")

        # Deduplicate relations (critical for accurate metrics)
        pred_relations = deduplicate_relations(pred_relations)

        # Extract ground truth
        gold_entities, gold_relations = extract_ground_truth(example)

        # Calculate metrics for this example
        entity_metrics = calculate_entity_metrics(pred_entities, gold_entities)
        relation_metrics = calculate_relation_metrics(pred_relations, gold_relations)

        entity_metrics_list.append(entity_metrics)
        relation_metrics_list.append(relation_metrics)

        predictions.append({
            'example_id': i,
            'input': example['input'],
            'prediction': pred_text,
            'gold_output': example['output'],
            'pred_entities': pred_entities,
            'gold_entities': gold_entities,
            'pred_relations': pred_relations,
            'gold_relations': gold_relations,
            'entity_metrics': entity_metrics,
            'relation_metrics': relation_metrics
        })

    # Aggregate metrics
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)

    # Entity metrics (macro average)
    entity_precision = sum(m['precision'] for m in entity_metrics_list) / len(entity_metrics_list)
    entity_recall = sum(m['recall'] for m in entity_metrics_list) / len(entity_metrics_list)
    entity_f1 = sum(m['f1'] for m in entity_metrics_list) / len(entity_metrics_list)

    # Entity metrics (micro average)
    entity_tp = sum(m['tp'] for m in entity_metrics_list)
    entity_fp = sum(m['fp'] for m in entity_metrics_list)
    entity_fn = sum(m['fn'] for m in entity_metrics_list)
    entity_precision_micro = entity_tp / (entity_tp + entity_fp) if (entity_tp + entity_fp) > 0 else 0
    entity_recall_micro = entity_tp / (entity_tp + entity_fn) if (entity_tp + entity_fn) > 0 else 0
    entity_f1_micro = 2 * entity_precision_micro * entity_recall_micro / (entity_precision_micro + entity_recall_micro) if (entity_precision_micro + entity_recall_micro) > 0 else 0

    print("\n📊 ENTITY EXTRACTION METRICS")
    print("-" * 80)
    print(f"Macro Average:")
    print(f"  Precision: {entity_precision:.3f}")
    print(f"  Recall:    {entity_recall:.3f}")
    print(f"  F1 Score:  {entity_f1:.3f}")
    print(f"\nMicro Average:")
    print(f"  Precision: {entity_precision_micro:.3f}")
    print(f"  Recall:    {entity_recall_micro:.3f}")
    print(f"  F1 Score:  {entity_f1_micro:.3f}")
    print(f"\nCounts:")
    print(f"  True Positives:  {entity_tp}")
    print(f"  False Positives: {entity_fp}")
    print(f"  False Negatives: {entity_fn}")

    # Relation metrics (macro average)
    relation_precision = sum(m['precision'] for m in relation_metrics_list) / len(relation_metrics_list)
    relation_recall = sum(m['recall'] for m in relation_metrics_list) / len(relation_metrics_list)
    relation_f1 = sum(m['f1'] for m in relation_metrics_list) / len(relation_metrics_list)

    # Relation metrics (micro average)
    relation_tp = sum(m['tp'] for m in relation_metrics_list)
    relation_fp = sum(m['fp'] for m in relation_metrics_list)
    relation_fn = sum(m['fn'] for m in relation_metrics_list)
    relation_precision_micro = relation_tp / (relation_tp + relation_fp) if (relation_tp + relation_fp) > 0 else 0
    relation_recall_micro = relation_tp / (relation_tp + relation_fn) if (relation_tp + relation_fn) > 0 else 0
    relation_f1_micro = 2 * relation_precision_micro * relation_recall_micro / (relation_precision_micro + relation_recall_micro) if (relation_precision_micro + relation_recall_micro) > 0 else 0

    print("\n📊 RELATION EXTRACTION METRICS")
    print("-" * 80)
    print(f"Macro Average:")
    print(f"  Precision: {relation_precision:.3f}")
    print(f"  Recall:    {relation_recall:.3f}")
    print(f"  F1 Score:  {relation_f1:.3f}")
    print(f"\nMicro Average:")
    print(f"  Precision: {relation_precision_micro:.3f}")
    print(f"  Recall:    {relation_recall_micro:.3f}")
    print(f"  F1 Score:  {relation_f1_micro:.3f}")
    print(f"\nCounts:")
    print(f"  True Positives:  {relation_tp}")
    print(f"  False Positives: {relation_fp}")
    print(f"  False Negatives: {relation_fn}")

    # Overall summary
    print("\n" + "=" * 80)
    print("📈 OVERALL SUMMARY")
    print("=" * 80)
    print(f"Test Examples: {len(test_data)}")
    print(f"\nEntity F1 (Micro):   {entity_f1_micro:.3f} {'✓ Target: 0.85' if entity_f1_micro >= 0.85 else '✗ Target: 0.85'}")
    print(f"Relation F1 (Micro): {relation_f1_micro:.3f} {'✓ Target: 0.75' if relation_f1_micro >= 0.75 else '✗ Target: 0.75'}")

    # Save detailed results
    results_path = Path(model_path).parent / "evaluation_results.json"
    with open(results_path, 'w') as f:
        json.dump({
            'entity_metrics': {
                'macro': {
                    'precision': entity_precision,
                    'recall': entity_recall,
                    'f1': entity_f1
                },
                'micro': {
                    'precision': entity_precision_micro,
                    'recall': entity_recall_micro,
                    'f1': entity_f1_micro,
                    'tp': entity_tp,
                    'fp': entity_fp,
                    'fn': entity_fn
                }
            },
            'relation_metrics': {
                'macro': {
                    'precision': relation_precision,
                    'recall': relation_recall,
                    'f1': relation_f1
                },
                'micro': {
                    'precision': relation_precision_micro,
                    'recall': relation_recall_micro,
                    'f1': relation_f1_micro,
                    'tp': relation_tp,
                    'fp': relation_fp,
                    'fn': relation_fn
                }
            },
            'predictions': predictions[:10]  # Save first 10 for inspection
        }, f, indent=2)

    print(f"\n✓ Detailed results saved to: {results_path}")
    print("\n" + "=" * 80)

    return {
        'entity_f1_micro': entity_f1_micro,
        'relation_f1_micro': relation_f1_micro,
        'predictions': predictions
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate medical NER/RE model")
    parser.add_argument(
        "--model",
        type=str,
        default="experiments/medical_ner_re/final_model",
        help="Path to trained model",
    )
    parser.add_argument(
        "--test_data",
        type=str,
        default="data/datasets/formatted/test.json",
        help="Path to test data",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Number of test samples to evaluate (default: all)",
    )

    args = parser.parse_args()

    # Run evaluation
    results = evaluate_model(args.model, args.test_data, args.num_samples)

    return results


if __name__ == "__main__":
    main()
