# Understanding Warnings in Odin SLM

## SyntaxWarnings from Unsloth (FIXED ✅)

### What Were They?

When running with Python 3.12, you might have seen warnings like:

```
SyntaxWarning: invalid escape sequence '\s'
  left = re.match("[\s\n]{4,}", leftover).span()[1]
```

### Why Did They Happen?

Python 3.12 introduced stricter checking for string escape sequences. The unsloth_zoo library uses strings like `"\s"` instead of raw strings `r"\s"` for regex patterns. While the code still works correctly, Python 3.12 warns about this.

### How We Fixed It

We added warning filters in our training scripts:

```python
import warnings
warnings.filterwarnings("ignore", category=SyntaxWarning, module="unsloth_zoo")
warnings.filterwarnings("ignore", category=SyntaxWarning, module="unsloth")
```

This suppresses the warnings without affecting functionality. The unsloth library works perfectly - these were purely cosmetic warnings about code style.

### Are They Harmful?

**No!** These warnings are completely harmless. The unsloth library functions correctly despite them. We've suppressed them to keep your output clean.

## Other Common Warnings

### UV Virtual Environment Warning

```
warning: `VIRTUAL_ENV=/home/pablo/.pyenv/versions/3.12.8` does not match...
```

**What it means**: UV detects that you have another Python environment activated, but it's using the project's `.venv` anyway.

**Is it a problem?** No, UV handles this correctly. You can ignore this warning.

**To silence it**: Deactivate other environments or use `uv run --active` if you want UV to use the currently active environment.

### Unsloth Import Order Warning

```
WARNING: Unsloth should be imported before [transformers]...
```

**What it means**: For maximum optimization, unsloth should be imported first.

**Is it a problem?** Not really. Our code already imports unsloth before transformers where it matters (in the actual model loading code).

**To fix**: If you write custom scripts, always `import unsloth` before `import transformers`.

## Training Script Usage

Now you have multiple ways to run training (all suppress warnings automatically):

### Method 1: Module Execution
```bash
uv run python -m odin_slm.training
```

### Method 2: Standalone Script
```bash
# Without dataset (shows help)
uv run python train.py

# With dataset
uv run python train.py --dataset your-dataset-name

# With custom config
uv run python train.py --config custom_config.yaml --dataset your-dataset
```

### Method 3: Programmatic
```python
from odin_slm.training import SLMTrainer

trainer = SLMTrainer(config_path="configs/training_config.yaml")
trainer.train("your-dataset-name")
```

## Summary

✅ **SyntaxWarnings**: Fixed - suppressed harmlessly
✅ **RuntimeWarning**: Fixed - improved module structure
⚠️ **UV VIRTUAL_ENV**: Harmless - can be ignored
ℹ️ **Unsloth Import**: Informational - already handled correctly

All warnings are either fixed or harmless. Your training environment is ready to use!

## Technical Details

For those interested in the technical details:

**Python 3.12 Changes**: Python 3.12 made SyntaxWarnings more visible as part of improved string handling. Invalid escape sequences like `"\s"` (which should be `r"\s"` or `"\\s"`) now trigger warnings.

**Unsloth Library**: The unsloth_zoo library (version 2025.3.1) has these escape sequences in its compiler.py and other files. The library works correctly because Python interprets `"\s"` as a literal backslash followed by 's', which is what the regex needs.

**Our Solution**: We use Python's warnings module to filter out these specific warnings from the unsloth modules while keeping other warnings visible. This is the recommended approach for third-party library warnings that you can't fix yourself.

**Future**: The unsloth team may update their library to use raw strings (r"...") in a future version, at which point these filters won't be necessary.
