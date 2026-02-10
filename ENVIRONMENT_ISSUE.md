# Environment Compatibility Issue

**Status**: Blocking training
**Date**: 2026-02-08

## The Problem

There's a fundamental incompatibility between library versions in the current environment:

```
accelerate 1.12.0 (locked)
    ↓ (missing)
FP8BackendType

transformers 4.47.1 (required by unsloth>=2024.12)
    ↓ (expects)
FP8BackendType in accelerate

ERROR: NameError: name 'FP8BackendType' is not defined
```

## Root Cause

1. **Unsloth** (2025.3.3) requires `transformers>=4.46.1`
2. **Transformers** (4.47.1) has code that expects `FP8BackendType` from accelerate
3. **Accelerate** (1.12.0) doesn't have `FP8BackendType` (added in 1.3.0+)
4. Accelerate is locked at 1.12.0 due to other dependency constraints

## Why Monkey-Patching Failed

The error occurs in compiled/cached code (`<string>` in traceback), which our runtime patches can't reach:

```python
File "<string>", line 135, in prepare
NameError: name 'FP8BackendType' is not defined
```

## Attempted Solutions

1. ✗ Monkey-patch accelerate.utils.FP8BackendType → Failed (compiled code)
2. ✗ Update accelerate to >=1.3.0 → Locked at 1.12.0
3. ✗ Downgrade transformers <4.46.0 → Incompatible with unsloth
4. ✗ Environment variables to disable FP8 → Not effective
5. ✗ Clear caches → Code still compiled

## Possible Solutions

### Option 1: Use Older Unsloth (Easiest)
```bash
# Downgrade to unsloth version that works with older transformers
uv add "unsloth[cu121-torch250]<2024.12"
uv add "transformers>=4.40.0,<4.46.0"
uv sync
```

**Pros**: Should work immediately
**Cons**: Older unsloth version, may miss recent optimizations

### Option 2: Manual Training Loop (Custom)
Create a custom training loop that bypasses SFTTrainer entirely.

**Pros**: Full control, no trainer dependencies
**Cons**: More code to write, manual gradient accumulation

### Option 3: Wait for Fix
Wait for either:
- New unsloth version compatible with newer accelerate
- New accelerate version in the dependency tree

**Pros**: Get latest versions eventually
**Cons**: Unknown timeline

### Option 4: Use Different Environment
Create a fresh environment with known-compatible versions:
```
python==3.11
unsloth[cu121-torch250]==2024.11
transformers==4.45
accelerate==0.34
```

## Recommended Action

**For immediate training**: Try Option 1 (downgrade unsloth)

```bash
cd /home/pablo/code/odin-slm

# Edit pyproject.toml
# Change: "unsloth[cu121-torch250]>=2024.12"
# To: "unsloth[cu121-torch250]>=2024.11,<2024.12"

# Change: "transformers>=4.46.0,<4.48.0"
# To: "transformers>=4.40.0,<4.46.0"

uv sync
python train_simple.py
```

## Current Environment

```
Python: 3.12.8
PyTorch: 2.10.0 (upgraded from 2.5.0)
CUDA: 12.4
Unsloth: 2025.3.3
Transformers: 4.47.1
Accelerate: 1.12.0 (LOCKED - missing FP8BackendType)
TRL: 0.17.0
```

## Next Steps

1. Try downgrading unsloth and transformers
2. If that fails, implement custom training loop
3. Document working configuration for future reference

---

**This is a known compatibility issue in the Hugging Face ecosystem.**
See: https://github.com/huggingface/accelerate/pull/1287
