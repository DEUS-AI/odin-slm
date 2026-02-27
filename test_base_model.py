#!/usr/bin/env python3
"""Test the base Mistral model (no LoRA) to isolate corruption issues"""
import warnings
warnings.filterwarnings("ignore")

from unsloth import FastLanguageModel
import torch

print("Loading base model (no LoRA)...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/mistral-7b-v0.3-bnb-4bit",
    max_seq_length=512,
    dtype=None,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model)
print("✓ Base model loaded\n")

# Test with same example
test_instruction = "Extract all medical entities and their relationships from the following clinical text. Format the output as a list of entities with their types, followed by relationships."

test_input = "Clinical case: hemoglobin A1c with omeprazole administered with thyroid function tests. Follow-up scheduled."

prompt = f"""### Instruction:
{test_instruction}

### Input:
{test_input}

### Output:
"""

print("=" * 80)
print("BASE MODEL TEST (NO LORA)")
print("=" * 80)
print("\nINPUT TEXT:")
print("-" * 80)
print(test_input)
print("-" * 80)

# Generate with deterministic settings
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)

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

print("\nBASE MODEL OUTPUT:")
print("-" * 80)
print(output_part)
print("-" * 80)

# Check for Unicode corruption
import unicodedata
has_soft_hyphens = '\u00ad' in output_part
has_zwj = '\u200b' in output_part or '\u200d' in output_part

print("\n" + "=" * 80)
print("CORRUPTION CHECK:")
print("=" * 80)
print(f"Contains soft hyphens (\\u00ad): {has_soft_hyphens}")
print(f"Contains zero-width joiners: {has_zwj}")

if has_soft_hyphens or has_zwj:
    print("\n⚠️  WARNING: Base model is producing Unicode corruption!")
    print("This indicates a tokenizer or Unsloth patching issue.")
else:
    print("\n✓ No Unicode corruption detected in base model output.")
    print("Corruption likely introduced during LoRA training.")
