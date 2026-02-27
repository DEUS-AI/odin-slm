#!/usr/bin/env python3
"""Analyze direction reversal in relation extraction errors.

Key hypothesis: Many FP/FN pairs are semantically equivalent but direction-swapped:
  - "A --[causes]--> B" ≈ "B --[indicates]--> A"  (inverse pair)
  - "A --[interacts_with]--> B" = "B --[interacts_with]--> A"  (symmetric)
  - "A --[treats]--> B" ≈ "B --[treated_by]--> A"  (but model only uses treats)

This script quantifies how many errors fall into each category and computes
"relaxed" metrics that account for these equivalences.
"""
import warnings
warnings.filterwarnings("ignore")

import sys
from pathlib import Path
import json
import re
import torch
from collections import Counter, defaultdict

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

from unsloth import FastLanguageModel
from tqdm import tqdm

print("=" * 80)
print("DIRECTION REVERSAL ANALYSIS")
print("=" * 80)

# Define semantic equivalences
# (type1, type2) means: "A --[type1]--> B" is equivalent to "B --[type2]--> A"
INVERSE_PAIRS = {
    ("causes", "indicates"),    # A causes B ↔ B indicates A
    ("indicates", "causes"),    # B indicates A ↔ A causes B
}

SYMMETRIC_TYPES = {"interacts_with"}  # A interacts_with B = B interacts_with A


def are_semantically_equivalent(pred_tuple, gold_tuple):
    """Check if pred and gold relations are semantically equivalent.

    Each tuple is (head, relation_type, tail) - all lowercase.

    Returns: (is_equivalent, equivalence_type)
    """
    ph, pr, pt = pred_tuple
    gh, gr, gt = gold_tuple

    # Exact match
    if pred_tuple == gold_tuple:
        return True, "exact"

    # Symmetric: same entities and type, just swapped direction
    if pr == gr and pr in SYMMETRIC_TYPES and ph == gt and pt == gh:
        return True, "symmetric_swap"

    # Inverse pair: causes/indicates with swapped direction
    if ph == gt and pt == gh and (pr, gr) in INVERSE_PAIRS:
        return True, "inverse_pair"

    # Same entities, same direction, wrong type
    if ph == gh and pt == gt and pr != gr:
        return True, "wrong_type_only"

    # Same entities swapped, same type (non-symmetric)
    if ph == gt and pt == gh and pr == gr and pr not in SYMMETRIC_TYPES:
        return True, "direction_swap_same_type"

    return False, None


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

# Counters
strict_tp = 0
strict_fp = 0
strict_fn = 0

relaxed_tp = 0
relaxed_fp = 0
relaxed_fn = 0

equivalence_types = Counter()
detailed_examples = defaultdict(list)

# Per-example tracking
examples_strict_perfect = 0
examples_relaxed_perfect = 0

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

    _, pred_relations = parse_output_format(output_part)
    _, gold_relations = parse_output_format(example['output'])

    # Deduplicate predictions
    seen = set()
    unique_pred = []
    for r in pred_relations:
        t = (r['head'].lower(), r['relation'].lower(), r['tail'].lower())
        if t not in seen:
            seen.add(t)
            unique_pred.append(r)
    pred_relations = unique_pred

    pred_set = {(r['head'].lower(), r['relation'].lower(), r['tail'].lower()) for r in pred_relations}
    gold_set = {(r['head'].lower(), r['relation'].lower(), r['tail'].lower()) for r in gold_relations}

    # --- STRICT metrics ---
    s_tp = len(pred_set & gold_set)
    s_fp = len(pred_set - gold_set)
    s_fn = len(gold_set - pred_set)
    strict_tp += s_tp
    strict_fp += s_fp
    strict_fn += s_fn

    if s_fp == 0 and s_fn == 0:
        examples_strict_perfect += 1

    # --- RELAXED metrics ---
    # For each FP, check if it matches any FN via semantic equivalence
    fp_tuples = list(pred_set - gold_set)
    fn_tuples = list(gold_set - pred_set)

    matched_fp = set()
    matched_fn = set()

    for fp_idx, fp in enumerate(fp_tuples):
        for fn_idx, fn in enumerate(fn_tuples):
            if fn_idx in matched_fn:
                continue
            is_equiv, equiv_type = are_semantically_equivalent(fp, fn)
            if is_equiv:
                matched_fp.add(fp_idx)
                matched_fn.add(fn_idx)
                equivalence_types[equiv_type] += 1

                if len(detailed_examples[equiv_type]) < 8:
                    detailed_examples[equiv_type].append({
                        'example_id': i,
                        'predicted': f"{fp[0]} --[{fp[1]}]--> {fp[2]}",
                        'gold': f"{fn[0]} --[{fn[1]}]--> {fn[2]}",
                    })
                break

    r_tp = s_tp + len(matched_fp)  # strict TPs + newly matched pairs
    r_fp = s_fp - len(matched_fp)  # remaining unmatched FPs
    r_fn = s_fn - len(matched_fn)  # remaining unmatched FNs

    relaxed_tp += r_tp
    relaxed_fp += r_fp
    relaxed_fn += r_fn

    if r_fp == 0 and r_fn == 0:
        examples_relaxed_perfect += 1


# Compute F1 scores
def compute_prf(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) > 0 else 0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
    return p, r, f1


strict_p, strict_r, strict_f1 = compute_prf(strict_tp, strict_fp, strict_fn)
relaxed_p, relaxed_r, relaxed_f1 = compute_prf(relaxed_tp, relaxed_fp, relaxed_fn)

print("\n" + "=" * 80)
print("STRICT vs RELAXED METRICS")
print("=" * 80)

print(f"\n{'Metric':<25} {'Strict':>10} {'Relaxed':>10} {'Delta':>10}")
print("-" * 55)
print(f"{'Precision':<25} {strict_p:>10.3f} {relaxed_p:>10.3f} {relaxed_p-strict_p:>+10.3f}")
print(f"{'Recall':<25} {strict_r:>10.3f} {relaxed_r:>10.3f} {relaxed_r-strict_r:>+10.3f}")
print(f"{'F1':<25} {strict_f1:>10.3f} {relaxed_f1:>10.3f} {relaxed_f1-strict_f1:>+10.3f}")
print(f"{'True Positives':<25} {strict_tp:>10} {relaxed_tp:>10} {relaxed_tp-strict_tp:>+10}")
print(f"{'False Positives':<25} {strict_fp:>10} {relaxed_fp:>10} {relaxed_fp-strict_fp:>+10}")
print(f"{'False Negatives':<25} {strict_fn:>10} {relaxed_fn:>10} {relaxed_fn-strict_fn:>+10}")
print(f"{'Perfect examples':<25} {examples_strict_perfect:>10} {examples_relaxed_perfect:>10} {examples_relaxed_perfect-examples_strict_perfect:>+10}")

print("\n" + "=" * 80)
print("EQUIVALENCE TYPES FOUND (FP-FN pairs resolved)")
print("=" * 80)
total_resolved = sum(equivalence_types.values())
for et, count in equivalence_types.most_common():
    print(f"  {et}: {count} ({count/total_resolved*100:.1f}%)")
print(f"  TOTAL resolved: {total_resolved}")

print("\n" + "=" * 80)
print("REMAINING UNRESOLVED ERRORS")
print("=" * 80)
print(f"  Unresolved FP (truly hallucinated): {relaxed_fp}")
print(f"  Unresolved FN (truly missed):       {relaxed_fn}")

for equiv_type, examples in detailed_examples.items():
    print(f"\n{'=' * 80}")
    print(f"EXAMPLES: {equiv_type}")
    print(f"{'=' * 80}")
    for ex in examples:
        print(f"  Example {ex['example_id']}:")
        print(f"    Predicted: {ex['predicted']}")
        print(f"    Gold:      {ex['gold']}")
        print()
