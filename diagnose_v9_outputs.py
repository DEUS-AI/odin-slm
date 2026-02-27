#!/usr/bin/env python3
"""Diagnose V9 model outputs - compare with expected to understand errors"""
import warnings
warnings.filterwarnings("ignore")

import sys
from pathlib import Path
import json
import torch

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

from unsloth import FastLanguageModel

print("=" * 80)
print("V9 MODEL OUTPUT DIAGNOSIS (Llama 3.1 8B Base)")
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
print("✓ V9 model loaded")

# Load test data
with open("data/datasets/formatted/test.json", "r") as f:
    test_data = json.load(f)
print(f"✓ Loaded {len(test_data)} test examples")

# Categorize results
correct_entities = 0
misspelled_entities = 0
wrong_type_entities = 0
hallucinated_entities = 0
missed_entities = 0
format_errors = 0

# Detailed examples
examples_good = []
examples_bad_spelling = []
examples_bad_format = []
examples_with_relations = []

for i in range(min(50, len(test_data))):
    example = test_data[i]

    prompt = f"""### Instruction:
{example['instruction']}

### Input:
{example['input']}

### Output:
"""

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=300,
            do_sample=False,
            repetition_penalty=1.5,
            no_repeat_ngram_size=3,
        )

    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    output_part = generated.split("### Output:")[-1].strip()
    expected = example['output']

    # Check for key patterns
    has_correct_format = "### Entities:" in output_part or "Entities:" in output_part
    has_relations_section = "### Relations:" in output_part or "Relations:" in output_part
    has_relation_arrows = "--[" in output_part or "-->" in output_part

    # Extract entity names from expected
    expected_entities = []
    for line in expected.split('\n'):
        if ']' in line and '[' in line:
            parts = line.split(']')
            if len(parts) >= 2:
                entity_name = parts[-1].strip()
                entity_type = line.split('[')[1].split(']')[0].strip()
                expected_entities.append((entity_type, entity_name))

    # Extract entity names from predicted
    pred_entities = []
    for line in output_part.split('\n'):
        if ']' in line and '[' in line:
            parts = line.split(']')
            if len(parts) >= 2:
                entity_name = parts[-1].strip()
                entity_type = line.split('[')[1].split(']')[0].strip()
                pred_entities.append((entity_type, entity_name))

    # Check spelling
    has_misspelling = False
    for ptype, pname in pred_entities:
        pname_lower = pname.lower().strip()
        found = False
        for etype, ename in expected_entities:
            if pname_lower == ename.lower().strip():
                found = True
                break
        if not found and pname_lower:
            # Check if it's a misspelling (present in input but different)
            if any(pname_lower[:4] in ename.lower() for _, ename in expected_entities if len(pname_lower) >= 4):
                has_misspelling = True

    # Categorize
    if i < 10 or has_misspelling or has_relation_arrows or not has_correct_format:
        print(f"\n{'='*80}")
        print(f"EXAMPLE {i+1}")
        print(f"{'='*80}")
        print(f"INPUT: {example['input'][:150]}...")
        print(f"\nEXPECTED:")
        for line in expected.split('\n')[:8]:
            print(f"  {line}")
        print(f"\nV9 OUTPUT:")
        for line in output_part.split('\n')[:12]:
            print(f"  {line}")

        # Analysis flags
        flags = []
        if not has_correct_format:
            flags.append("BAD_FORMAT")
            format_errors += 1
        if has_misspelling:
            flags.append("MISSPELLING")
            misspelled_entities += 1
        if has_relation_arrows:
            flags.append("HAS_RELATIONS")
        if not flags:
            flags.append("OK")

        print(f"\nFLAGS: {', '.join(flags)}")
        print(f"Expected entities: {len(expected_entities)}, Predicted: {len(pred_entities)}")

# Summary stats
print(f"\n{'='*80}")
print("SUMMARY (first 50 examples)")
print(f"{'='*80}")
print(f"Format errors: {format_errors}")
print(f"Examples with misspellings: {misspelled_entities}")
