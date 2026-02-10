#!/usr/bin/env python3
import warnings
warnings.filterwarnings("ignore")

from unsloth import FastLanguageModel
import torch

# Load v1 model
print("Loading v1 model...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="experiments/medical_ner_re/final_model",
    max_seq_length=512,
    dtype=None,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model)
print("✓ Model loaded\n")

# Test prompt (same as before)
test_text = "Patient has diabetes and takes metformin daily."

prompt = f"""### Instruction:
Extract all medical entities and their relations from the following clinical text. Identify diseases, symptoms, drugs, procedures, and lab tests, along with their relationships.

### Input:
{test_text}

### Output:
"""

print("=" * 80)
print("V1 INFERENCE TEST - GREEDY DECODING (do_sample=False)")
print("=" * 80)
print(f"\nInput: {test_text}")
print("\nGenerating output with GREEDY decoding...")

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=False,  # GREEDY decoding
        repetition_penalty=1.5,
        no_repeat_ngram_size=3,
    )

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
output_part = generated_text.split("### Output:")[-1].strip()

print("\nModel Output:")
print("-" * 80)
print(output_part)
print("-" * 80)

print("\n" + "=" * 80)
print("V1 INFERENCE TEST - SAMPLING (do_sample=True, like evaluation script)")
print("=" * 80)
print("\nGenerating output with SAMPLING...")

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        temperature=0.3,
        do_sample=True,
        top_p=0.9,
        repetition_penalty=1.5,
        no_repeat_ngram_size=3,
        pad_token_id=tokenizer.eos_token_id,
    )

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
output_part = generated_text.split("### Output:")[-1].strip()

print("\nModel Output:")
print("-" * 80)
print(output_part)
print("-" * 80)
