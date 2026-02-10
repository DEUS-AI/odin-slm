#!/usr/bin/env python3
"""Diagnose v6 model outputs to understand why performance is worse than v1"""
import warnings
warnings.filterwarnings("ignore")

import sys
from pathlib import Path
import json
import torch

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

from unsloth import FastLanguageModel

print("=" * 80)
print("V6 MODEL OUTPUT DIAGNOSIS")
print("=" * 80)

# Load v6 model
print("\nLoading v6 model...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="experiments/medical_ner_re_v6/final_model",
    max_seq_length=512,
    dtype=None,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model)
print("✓ V6 model loaded")

# Load test data
print("\nLoading test data...")
with open("data/datasets/formatted/test.json", "r") as f:
    test_data = json.load(f)
print(f"✓ Loaded {len(test_data)} test examples")

# Test on first 5 examples
print("\n" + "=" * 80)
print("SAMPLE OUTPUTS (First 5 Examples)")
print("=" * 80)

for i in range(min(5, len(test_data))):
    example = test_data[i]

    # Create prompt
    prompt = f"""### Instruction:
{example['instruction']}

### Input:
{example['input']}

### Output:
"""

    print(f"\n{'=' * 80}")
    print(f"EXAMPLE {i+1}")
    print(f"{'=' * 80}")
    print(f"\nINPUT TEXT:")
    print(f"  {example['input']}")

    print(f"\nEXPECTED OUTPUT:")
    expected_lines = example['output'].split('\n')
    for line in expected_lines[:10]:  # First 10 lines
        print(f"  {line}")
    if len(expected_lines) > 10:
        print(f"  ... ({len(expected_lines) - 10} more lines)")

    # Generate
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=300,
            do_sample=False,
            repetition_penalty=1.5,
            no_repeat_ngram_size=3,
        )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    output_part = generated_text.split("### Output:")[-1].strip()

    print(f"\nV6 MODEL OUTPUT:")
    output_lines = output_part.split('\n')
    for line in output_lines[:10]:  # First 10 lines
        print(f"  {line}")
    if len(output_lines) > 10:
        print(f"  ... ({len(output_lines) - 10} more lines)")

    # Quick analysis
    has_entities = "### Entities:" in output_part or "[" in output_part
    has_relations = "### Relations:" in output_part or "-->" in output_part or "--[" in output_part

    print(f"\nQUICK ANALYSIS:")
    print(f"  Contains entity markers: {has_entities}")
    print(f"  Contains relation markers: {has_relations}")
    print(f"  Output length: {len(output_part)} chars")

print("\n" + "=" * 80)
print("DIAGNOSIS COMPLETE")
print("=" * 80)
