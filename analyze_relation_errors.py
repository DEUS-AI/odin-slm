#!/usr/bin/env python3
"""Analyze V9 relation extraction errors to understand failure patterns"""
import warnings
warnings.filterwarnings("ignore")

import sys
from pathlib import Path
import json
import torch
from collections import Counter, defaultdict

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

from unsloth import FastLanguageModel

print("=" * 80)
print("V9 RELATION EXTRACTION ERROR ANALYSIS")
print("=" * 80)

# Load model
print("\nLoading V9 model...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="experiments/medical_ner_re_v9/final_model",
    max_seq_length=512,
    dtype=None,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model)
print("Model loaded")

# Load test data
with open("data/datasets/formatted/test.json", "r") as f:
    test_data = json.load(f)
print(f"Loaded {len(test_data)} test examples")

# Import the parsing function from evaluate script
import re

def parse_output_format(text):
    entities = []
    relations = []
    lines = text.strip().split('\n')
    current_section = None
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if 'entities' in line.lower() and line.startswith('#'):
            current_section = 'entities'
            continue
        elif 'relations' in line.lower() and line.startswith('#'):
            current_section = 'relations'
            continue
        if current_section == 'entities' and re.match(r'^\d+\.', line):
            match = re.match(r'^\d+\.\s*\[(\w+)\]\s*(.+)', line)
            if match:
                entity_type, text_val = match.groups()
                entities.append({'text': text_val.strip(), 'type': entity_type.strip()})
        elif current_section == 'relations' and re.match(r'^\d+\.', line):
            match = re.match(r'^\d+\.\s*(.+?)\s*--\[(.+?)\]-->\s*(.+)', line)
            if not match:
                match = re.match(r'^\d+\.\s*(.+?)\s*-->\s*(.+)', line)
                if match:
                    head, tail = match.groups()
                    relations.append({'head': head.strip(), 'relation': 'unknown', 'tail': tail.strip()})
                    continue
            if match and len(match.groups()) == 3:
                head, rel_type, tail = match.groups()
                relations.append({'head': head.strip(), 'relation': rel_type.strip(), 'tail': tail.strip()})
    return entities, relations

# Categorize errors
error_categories = Counter()
fp_relation_types = Counter()  # False positive relation types
fn_relation_types = Counter()  # False negative relation types
tp_relation_types = Counter()  # True positive relation types

# Detailed error examples
fp_examples = []  # False positives (hallucinated relations)
fn_examples = []  # False negatives (missed relations)
wrong_type_examples = []  # Right entities, wrong relation type
partial_match_examples = []  # Head or tail matches but not both

# Track per-example stats
examples_with_no_gold_rels = 0
examples_with_no_pred_rels = 0
examples_with_gold_rels = 0
examples_perfect_relations = 0
examples_with_duplicate_preds = 0

from tqdm import tqdm

for i, example in enumerate(tqdm(test_data, desc="Analyzing")):
    prompt = f"""### Instruction:
{example['instruction']}

### Input:
{example['input']}

### Output:
"""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to("cuda")
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    output_part = generated.split("### Output:")[-1].strip()
    expected = example['output']

    pred_entities, pred_relations = parse_output_format(output_part)
    gold_entities, gold_relations = parse_output_format(expected)

    # Check for duplicates in predictions
    pred_rel_tuples = [(r['head'].lower(), r['relation'].lower(), r['tail'].lower()) for r in pred_relations]
    if len(pred_rel_tuples) != len(set(pred_rel_tuples)):
        examples_with_duplicate_preds += 1

    # Deduplicate
    seen = set()
    unique_pred_relations = []
    for r in pred_relations:
        t = (r['head'].lower(), r['relation'].lower(), r['tail'].lower())
        if t not in seen:
            seen.add(t)
            unique_pred_relations.append(r)
    pred_relations = unique_pred_relations

    # Create sets for comparison
    pred_set = {(r['head'].lower(), r['relation'].lower(), r['tail'].lower()) for r in pred_relations}
    gold_set = {(r['head'].lower(), r['relation'].lower(), r['tail'].lower()) for r in gold_relations}

    if not gold_relations:
        examples_with_no_gold_rels += 1
    else:
        examples_with_gold_rels += 1

    if not pred_relations:
        examples_with_no_pred_rels += 1

    if pred_set == gold_set:
        examples_perfect_relations += 1

    # True positives
    tp = pred_set & gold_set
    for t in tp:
        tp_relation_types[t[1]] += 1

    # False positives (predicted but not in gold)
    fp = pred_set - gold_set
    for t in fp:
        fp_relation_types[t[1]] += 1

        # Check if it's a wrong-type error (entities match, relation type differs)
        for g in gold_set:
            if t[0] == g[0] and t[2] == g[2]:
                wrong_type_examples.append({
                    'example_id': i,
                    'predicted': f"{t[0]} --[{t[1]}]--> {t[2]}",
                    'expected': f"{g[0]} --[{g[1]}]--> {g[2]}",
                })
                error_categories['wrong_relation_type'] += 1
                break
            elif t[0] == g[0] or t[2] == g[2]:
                partial_match_examples.append({
                    'example_id': i,
                    'predicted': f"{t[0]} --[{t[1]}]--> {t[2]}",
                    'expected': f"{g[0]} --[{g[1]}]--> {g[2]}",
                })
                error_categories['partial_entity_match'] += 1
                break
        else:
            # No matching gold relation at all
            if not gold_relations:
                error_categories['hallucinated_on_no_gold'] += 1
            else:
                error_categories['hallucinated_relation'] += 1

            if len(fp_examples) < 30:
                fp_examples.append({
                    'example_id': i,
                    'input': example['input'][:150],
                    'predicted': f"{t[0]} --[{t[1]}]--> {t[2]}",
                    'gold_relations': [f"{g[0]} --[{g[1]}]--> {g[2]}" for g in gold_set],
                })

    # False negatives (in gold but not predicted)
    fn = gold_set - pred_set
    for t in fn:
        fn_relation_types[t[1]] += 1

        # Check if it's partially predicted
        for p in pred_set:
            if t[0] == p[0] and t[2] == p[2]:
                # Already counted as wrong_type above
                break
        else:
            error_categories['completely_missed'] += 1

            if len(fn_examples) < 30:
                fn_examples.append({
                    'example_id': i,
                    'input': example['input'][:150],
                    'missed': f"{t[0]} --[{t[1]}]--> {t[2]}",
                    'pred_relations': [f"{p[0]} --[{p[1]}]--> {p[2]}" for p in pred_set],
                })


# Print results
print("\n" + "=" * 80)
print("OVERVIEW")
print("=" * 80)
print(f"Total test examples:               {len(test_data)}")
print(f"Examples with gold relations:       {examples_with_gold_rels}")
print(f"Examples with NO gold relations:    {examples_with_no_gold_rels}")
print(f"Examples with NO pred relations:    {examples_with_no_pred_rels}")
print(f"Examples with perfect relations:    {examples_perfect_relations}")
print(f"Examples with duplicate predictions:{examples_with_duplicate_preds}")

print("\n" + "=" * 80)
print("ERROR CATEGORIES")
print("=" * 80)
for cat, count in error_categories.most_common():
    print(f"  {cat}: {count}")

print("\n" + "=" * 80)
print("TRUE POSITIVE RELATION TYPES")
print("=" * 80)
for rt, count in tp_relation_types.most_common():
    print(f"  {rt}: {count}")

print("\n" + "=" * 80)
print("FALSE POSITIVE RELATION TYPES (hallucinated/wrong)")
print("=" * 80)
for rt, count in fp_relation_types.most_common():
    print(f"  {rt}: {count}")

print("\n" + "=" * 80)
print("FALSE NEGATIVE RELATION TYPES (missed)")
print("=" * 80)
for rt, count in fn_relation_types.most_common():
    print(f"  {rt}: {count}")

print("\n" + "=" * 80)
print("WRONG RELATION TYPE EXAMPLES (right entities, wrong relation)")
print("=" * 80)
for ex in wrong_type_examples[:15]:
    print(f"  Example {ex['example_id']}:")
    print(f"    Predicted: {ex['predicted']}")
    print(f"    Expected:  {ex['expected']}")
    print()

print("\n" + "=" * 80)
print("FALSE POSITIVE EXAMPLES (hallucinated relations)")
print("=" * 80)
for ex in fp_examples[:15]:
    print(f"  Example {ex['example_id']}:")
    print(f"    Input: {ex['input']}...")
    print(f"    Hallucinated: {ex['predicted']}")
    print(f"    Gold: {ex['gold_relations']}")
    print()

print("\n" + "=" * 80)
print("FALSE NEGATIVE EXAMPLES (completely missed)")
print("=" * 80)
for ex in fn_examples[:15]:
    print(f"  Example {ex['example_id']}:")
    print(f"    Input: {ex['input']}...")
    print(f"    Missed: {ex['missed']}")
    print(f"    Predicted: {ex['pred_relations']}")
    print()
