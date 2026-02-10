# Training Failure Investigation Report

**Date:** 2026-02-10
**Models Investigated:** v3, v4
**Status:** 🔴 CRITICAL FAILURE IDENTIFIED

## Executive Summary

Both v3 and v4 models failed catastrophically with Entity F1 ~16% and Relation F1 0%. After systematic investigation, we discovered the **root cause**: the models are learning the output format but NOT the extraction task. They hallucinate random medical entities instead of extracting from input text.

## Investigation Steps Completed

### ✅ Step 1: Baseline Verification
- v1 results confirmed: Entity 88.4%, Relation 46.7% ✅
- v2 results confirmed: Entity 99.8%, Relation 42.3% ✅
- v1 re-evaluation in progress (to verify current environment)

### ✅ Step 2: Training Loss Analysis
**V4 Training Metrics:**
- Initial loss: ~0.20
- Final loss: ~0.17
- Eval loss: ~0.18
- **Conclusion:** Loss decreased normally, model appeared to be learning

### ✅ Step 3: Dataset Integrity Check
**Formatted Dataset Analysis:**
- 4000 training examples ✅
- Proper instruction format ✅
- Both entities and relations present ✅
- Random sample verification ✅

**Example:**
```
Input: "A 63-year-old patient presents to the clinic. urinalysis..."

Output:
### Entities:
1. [Lab_Test] urinalysis
2. [Lab_Test] blood glucose
...

### Relations:
1. migraine --[causes]--> joint pain
```

**Conclusion:** Dataset is properly formatted

### ✅ Step 4: Manual Inference Test
**CRITICAL FINDING - Root Cause Identified**

**Test Input:** "Patient has diabetes and takes metformin daily."

**V4 Model Output:**
```
### Entities:
1. [Disease] diabetes  ✅ CORRECT
2. [Medication] metforming treatment  ⚠️  PARTIAL (metformin -> metforming)
   The patient also has urinalysis.  ❌ HALLUCINATED
   Patient reports blood glucose.  ❌ HALLUCINATED
   Follow-up scheduled.  ❌ HALLUCINATED
3. [Lab_Test] urinary analysis  ❌ HALLUCINATED
4. [Blood Glucose] blood sugar level  ❌ HALLUCINATED
5. Metoprolol was prescribed.  ❌ HALLUCINATED
   Additionally, albuterol administered.  ❌ HALLUCINATED
6. [Procedure] endoscopy  ❌ HALLUCINATED
7. [Symptom] fatigue  ❌ HALLUCINATED
8. [Xray] MRI  ❌ HALLUCINATED
9. [Migraine] Headache  ❌ HALLUCINATED
```

**Analysis:**
- Model correctly identified 1 out of 2 entities from input
- Model hallucinated ~9 entities NOT present in input text
- Model generated proper format (### Entities:, numbered list, [Type] labels)
- Model knows medical terms and types
- **BUT: Model is NOT actually reading/extracting from input**

## Root Cause Analysis

### Problem: Format Learning vs Task Learning

The model successfully learned:
- ✅ Output structure (### Entities:, ### Relations:)
- ✅ Medical vocabulary (diseases, symptoms, drugs, procedures)
- ✅ Entity type labels ([Disease], [Symptom], etc.)
- ✅ Relation format (entity1 --[relation]--> entity2)

The model FAILED to learn:
- ❌ Extract entities FROM the input text
- ❌ Identify which entities are actually mentioned
- ❌ Ground predictions in the actual text
- ❌ Distinguish between what's present vs medical knowledge

### Why This Happened

**Hypothesis 1: Instruction Format Issue**
The instruction template may not emphasize extraction from input:
```
### Instruction:
Extract all medical entities and their relations...

### Input:
{text}

### Output:
{entities and relations}
```

The model may be treating this as "generate medical entities" rather than "extract entities FROM the input text".

**Hypothesis 2: Synthetic Data Pattern Memorization**
With only 5,000 synthetic examples, the model may have:
- Memorized common patterns in synthetic data
- Learned to generate typical medical scenarios
- Not learned the extraction mapping (input → output)

**Hypothesis 3: Training Epochs**
- v1 (3 epochs): 88.4% entity F1 ✅ WORKED
- v4 (5 epochs): 15.6% entity F1 ❌ FAILED

Extended training may have caused:
- Catastrophic forgetting of the extraction task
- Overfitting to surface patterns
- Collapse into a generation mode

**Hypothesis 4: Something Changed**
Since v1/v2 worked but v3/v4 failed using same scripts:
- Library versions updated?
- Dataset re-formatted incorrectly?
- Random seed effects?

## Comparison Table

| Metric | v1 (3 epochs) | v4 (5 epochs) | Change |
|--------|---------------|---------------|--------|
| **Entity F1** | 88.4% | 15.6% | -72.8 pp |
| **Relation F1** | 46.7% | 0.0% | -46.7 pp |
| **Training Loss** | ~0.20 → 0.17 | ~0.20 → 0.17 | Same ✅ |
| **Model Behavior** | Extraction | Hallucination | Complete failure |

## Evidence Summary

1. ✅ **Training metrics look normal** - Loss decreased appropriately
2. ✅ **Dataset is properly formatted** - Checked multiple samples
3. ❌ **Model outputs are wrong** - Hallucinating entities
4. ❌ **Extraction task not learned** - Generating instead of extracting

## Next Steps - Recommendations

### Immediate Actions (Required)

1. **Wait for v1 re-evaluation results**
   - Confirm v1 still works in current environment
   - Rule out environment/library changes
   - ETA: ~15 minutes

2. **Compare v1 vs v4 inference side-by-side**
   - Test same input on both models
   - Confirm v1 extracts correctly, v4 hallucinates
   - Verify the behavior difference

3. **Check if dataset changed**
   - Did we re-format the data between v1 and v4?
   - Are we using the SAME test set?
   - Could the formatted data have changed?

### Root Cause Investigation

**Option A: Test with Different Epochs**
- Train with v1's exact config (3 epochs)
- See if 5 epochs causes the failure
- Hypothesis: Extended training breaks extraction

**Option B: Test Instruction Format**
- Try more explicit instructions emphasizing extraction
- Example: "Extract ONLY the medical entities that appear in the following text:"
- Hypothesis: Current instructions ambiguous

**Option C: Check Dataset Splits**
- Verify train/val/test splits are same as v1
- Check if seed changed
- Hypothesis: Dataset changed between versions

**Option D: Library Version Investigation**
- Check if Unsloth/Transformers updated
- Rollback to v1 library versions
- Hypothesis: Library bug introduced

### Strategic Decisions

**If v1 re-evaluation succeeds (88%+ entity F1):**
→ Something specific to v3/v4 is wrong
→ Focus on: epochs, random effects, or dataset drift
→ **Action:** Retrain with 3 epochs (v1 config) to test epoch hypothesis

**If v1 re-evaluation FAILS (<50% entity F1):**
→ Environment changed (libraries, CUDA, etc.)
→ All models are now broken
→ **Action:** Investigate environment changes, rollback libraries

## Critical Questions

1. **Did the dataset format change between v1 and v4?**
   - Need to verify the training data is identical

2. **Why did 3 epochs work but 5 epochs fail?**
   - Catastrophic forgetting?
   - Overfitting to wrong objective?

3. **Can we reproduce v1's success?**
   - Use exact v1 config
   - Verify we can get 88%+ again

## Conclusion

We have identified a **fundamental training failure**: models are learning to generate medical entities in the correct format, but NOT learning to extract entities from input text. This explains the catastrophic evaluation scores.

The model behavior is internally consistent:
- Training loss decreases (generating valid outputs)
- Evaluation fails (outputs don't match input)
- Format is correct (learned structure)
- Content is wrong (hallucinated entities)

**Next critical step:** Wait for v1 re-evaluation to determine if this is an environmental issue or specific to v3/v4 training runs.

## Files Generated

- `test_v4_inference.py` - Manual inference test script
- `INVESTIGATION_REPORT.md` - This report

## Timeline

- 09:54 - v4 evaluation completed (15.6% entity F1)
- 10:30 - Investigation started
- 11:15 - Root cause identified (hallucination behavior)
- 11:30 - v1 re-evaluation in progress
- **Status:** Awaiting v1 results
