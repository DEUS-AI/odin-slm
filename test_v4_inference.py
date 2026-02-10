#!/usr/bin/env python3
import warnings
warnings.filterwarnings("ignore")

from unsloth import FastLanguageModel
import torch

# Load v4 model
print("Loading v4 model...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="experiments/medical_ner_re_v4/final_model",
    max_seq_length=512,
    dtype=None,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model)  # Enable inference mode
print("✓ Model loaded\n")

# Test prompt
test_text = "Patient has diabetes and takes metformin daily."

prompt = f"""### Instruction:
Extract all medical entities and their relations from the following clinical text. Identify diseases, symptoms, drugs, procedures, and lab tests, along with their relationships.

### Input:
{test_text}

### Output:
"""

print("=" * 80)
print("MANUAL INFERENCE TEST - V4")
print("=" * 80)
print(f"\nInput: {test_text}")
print("\nGenerating output...")

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        temperature=0.3,
        repetition_penalty=1.5,
        no_repeat_ngram_size=3,
    )

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

# Extract just the output part
output_part = generated_text.split("### Output:")[-1].strip()

print("\nModel Output:")
print("-" * 80)
print(output_part)
print("-" * 80)
