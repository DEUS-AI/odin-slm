#!/usr/bin/env python3
import warnings
warnings.filterwarnings("ignore")

from unsloth import FastLanguageModel
import torch
import json

# Load v2 model
print("Loading v2 model...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="experiments/medical_ner_re_v2/final_model",
    max_seq_length=512,
    dtype=None,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model)
print("✓ Model loaded\n")

# Load test data
with open("data/datasets/formatted/test.json") as f:
    test_data = json.load(f)

# Use first test example
example = test_data[0]

print("=" * 80)
print("V2 INFERENCE TEST - REAL TEST EXAMPLE #0")
print("=" * 80)

print("\nINPUT TEXT:")
print("-" * 80)
print(example["input"])
print("-" * 80)

print("\nGROUND TRUTH OUTPUT:")
print("-" * 80)
print(example["output"])
print("-" * 80)

# Generate prediction
prompt = f"""### Instruction:
{example['instruction']}

### Input:
{example['input']}

### Output:
"""

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=300,
        do_sample=False,  # Greedy for determinism
        repetition_penalty=1.5,
        no_repeat_ngram_size=3,
    )

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
output_part = generated_text.split("### Output:")[-1].strip()

print("\nMODEL PREDICTION:")
print("-" * 80)
print(output_part)
print("-" * 80)
