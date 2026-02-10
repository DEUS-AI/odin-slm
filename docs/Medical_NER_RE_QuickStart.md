# Medical NER/RE Training - Quick Start Guide

**Task**: Named Entity Recognition (NER) + Relation Extraction (RE) for Medical Text
**Domain**: Clinical/Biomedical
**Status**: ✅ Research Complete | ✅ Synthetic Generator Ready

---

## Quick Start (5 Minutes)

### 1. Generate Synthetic Dataset

```bash
# Generate 1,000 synthetic medical documents
uv run python scripts/generate_medical_dataset.py \
    --num_samples 1000 \
    --output data/datasets/medical_ner_re_1k.json

# Generate larger dataset (10K)
uv run python scripts/generate_medical_dataset.py \
    --num_samples 10000 \
    --output data/datasets/medical_ner_re_10k.json
```

### 2. Inspect Dataset

```python
from odin_slm.data.synthetic_generator import MedicalTextGenerator

# Load dataset
generator = MedicalTextGenerator()
docs = generator.load_dataset("data/datasets/medical_ner_re_1k.json")

# Inspect first document
doc = docs[0]
print(f"Text: {doc.text}")
print(f"Entities: {len(doc.entities)}")
print(f"Relations: {len(doc.relations)}")
```

### 3. Evaluate Performance

```python
from odin_slm.data.evaluator import NERREvaluator, EntityPrediction

# Create evaluator
evaluator = NERREvaluator(matching_mode="exact")

# Evaluate (with your model predictions)
results = evaluator.evaluate_entities(predictions, gold_standard)
NERREvaluator.print_results(results)
```

---

## What's Included

### ✅ Research Document
- **Location**: [docs/Medical_NER_RE_Research.md](Medical_NER_RE_Research.md)
- **Content**:
  - 10+ baseline datasets (BioRED, ChemProt, BC5CDR, etc.)
  - Evaluation metrics and benchmarks
  - Synthetic data generation methods
  - SOTA performance targets (F1: 75-90%)

### ✅ Synthetic Data Generator
- **Location**: [src/odin_slm/data/synthetic_generator.py](../src/odin_slm/data/synthetic_generator.py)
- **Features**:
  - Generates medical clinical notes with entities and relations
  - 5 entity types: Disease, Symptom, Drug, Procedure, Lab_Test
  - 4 relation types: treats, causes, indicates, prevents
  - Configurable complexity (entities/relations per document)
  - JSON export format

### ✅ Evaluation Module
- **Location**: [src/odin_slm/data/evaluator.py](../src/odin_slm/data/evaluator.py)
- **Metrics**:
  - Precision, Recall, F1 (exact/partial/type matching)
  - Macro and Micro F1 for multi-class
  - Per-entity-type and per-relation-type metrics
  - Standard NER/RE evaluation protocols

### ✅ Dataset Generation Script
- **Location**: [scripts/generate_medical_dataset.py](../scripts/generate_medical_dataset.py)
- **Usage**: Command-line tool for bulk generation
- **Features**: Statistics, distribution analysis, example output

---

## Dataset Specifications

### Entity Types

| Type | Description | Examples |
|------|-------------|----------|
| **Disease** | Medical conditions | diabetes mellitus, hypertension, asthma |
| **Symptom** | Clinical signs | chest pain, fever, headache |
| **Drug** | Medications | metformin, aspirin, lisinopril |
| **Procedure** | Medical procedures | CT scan, MRI, biopsy |
| **Lab_Test** | Laboratory tests | blood glucose, lipid panel, CBC |

### Relation Types

| Relation | Head → Tail | Description |
|----------|-------------|-------------|
| **treats** | Drug → Disease | Drug treats/manages disease |
| **causes** | Disease → Symptom | Disease causes symptom |
| **indicates** | Test/Symptom → Disease | Test result or symptom indicates disease |
| **prevents** | Drug/Procedure → Disease | Intervention prevents disease |

### Example Document Structure

```json
{
  "text": "Patient presents with chest pain. Laboratory tests including troponin were ordered. Diagnosis: coronary artery disease. Treatment plan: aspirin prescribed.",
  "entities": [
    {"text": "chest pain", "type": "Symptom", "start": 21, "end": 31},
    {"text": "troponin", "type": "Lab_Test", "start": 63, "end": 71},
    {"text": "coronary artery disease", "type": "Disease", "start": 96, "end": 119},
    {"text": "aspirin", "type": "Drug", "start": 139, "end": 146}
  ],
  "relations": [
    {
      "type": "indicates",
      "head": {"text": "chest pain", "type": "Symptom", ...},
      "tail": {"text": "coronary artery disease", "type": "Disease", ...}
    },
    {
      "type": "treats",
      "head": {"text": "aspirin", "type": "Drug", ...},
      "tail": {"text": "coronary artery disease", "type": "Disease", ...}
    }
  ]
}
```

---

## Baseline Datasets (Optional)

### Recommended for Evaluation

1. **BioRED** (Primary Benchmark)
   - Download: https://ftp.ncbi.nlm.nih.gov/pub/lu/BioRED/
   - Format: PubTator
   - Use for: Cross-validation of synthetic data quality

2. **BC5CDR** (Chemical-Disease Relations)
   - Download: https://github.com/JHLiu7/BioNLP-OST-2019-BC5CDR
   - Use for: Specific domain evaluation

3. **ADE Corpus** (Adverse Drug Effects)
   - Download: https://www.kaggle.com/datasets/sid321axn/ade-corpus-v2
   - Use for: Drug safety relation extraction

---

## Training Pipeline

### Step 1: Data Preparation

```bash
# Generate training dataset
uv run python scripts/generate_medical_dataset.py \
    --num_samples 5000 \
    --output data/datasets/train.json \
    --seed 42

# Generate validation dataset (different seed)
uv run python scripts/generate_medical_dataset.py \
    --num_samples 1000 \
    --output data/datasets/val.json \
    --seed 123
```

### Step 2: Format for Training

Convert to instruction-tuning format:

```python
def format_for_training(doc):
    """Convert MedicalDocument to instruction format"""

    instruction = "Extract medical entities and relations from the following clinical text:"
    input_text = doc.text

    # Format entities
    entities_str = "\n".join([
        f"- [{e.type}] {e.text} (pos: {e.start}-{e.end})"
        for e in doc.entities
    ])

    # Format relations
    relations_str = "\n".join([
        f"- {r.head.text} --[{r.type}]--> {r.tail.text}"
        for r in doc.relations
    ])

    output = f"Entities:\n{entities_str}\n\nRelations:\n{relations_str}"

    return {
        "instruction": instruction,
        "input": input_text,
        "output": output
    }
```

### Step 3: Train with Unsloth

```python
from odin_slm.training import SLMTrainer

# Initialize trainer
trainer = SLMTrainer(config_path="configs/training_config.yaml")

# Train
trainer.train("data/datasets/medical_ner_re_formatted")
```

### Step 4: Evaluate

```python
from odin_slm.data.evaluator import NERREvaluator

# Load test data
test_docs = generator.load_dataset("data/datasets/test.json")

# Run model predictions
predictions = model.predict(test_docs)

# Evaluate
evaluator = NERREvaluator(matching_mode="exact")
results = evaluator.evaluate_entities_per_type(predictions, test_docs)

# Print results
NERREvaluator.print_results(results, "Test Set Results")
```

---

## Performance Targets

### Minimum Viable Performance (MVP)
- **Entity Recognition F1**: ≥ 70%
- **Relation Extraction F1**: ≥ 60%
- **Dataset Size**: 1,000+ examples

### Target Performance
- **Entity Recognition F1**: ≥ 85% (match PubMedBERT baseline)
- **Relation Extraction F1**: ≥ 75% (competitive with ChemProt)
- **Dataset Size**: 5,000+ examples

### Stretch Goals (SOTA)
- **Entity Recognition F1**: ≥ 90%
- **Relation Extraction F1**: ≥ 80%
- **Zero-shot**: Generalize to new medical domains

---

## Advanced: LLM-Based Synthetic Generation

### Using GPT-4 for Higher Quality

```python
import openai

def generate_with_gpt4(prompt_template, num_samples):
    """Generate synthetic medical text with GPT-4"""

    prompt = """Generate a clinical note with the following entities and relations:

Entities: [Disease, Symptom, Drug, Procedure, Lab_Test]
Relations: [treats, causes, indicates, prevents]

Format the output as JSON with 'text', 'entities', and 'relations' fields."""

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a medical documentation assistant."},
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content
```

### Data Augmentation Strategies

1. **Paraphrasing**: Rephrase clinical notes while preserving entities
2. **Entity Substitution**: Replace entities with synonyms
3. **Relation Rewording**: Express relations differently
4. **Context Expansion**: Add more clinical context

Expected improvement: **+24.7% F1** (from research)

---

## Evaluation Checklist

### Intrinsic Metrics ✓
- [ ] Entity Recognition F1 (strict match)
- [ ] Relation Extraction F1 (boundaries + type)
- [ ] Macro F1 (per entity/relation type)
- [ ] Micro F1 (overall aggregate)

### Extrinsic Validation ✓
- [ ] Test on BioRED benchmark
- [ ] Compare with baseline models
- [ ] Zero-shot performance on unseen domains
- [ ] Human expert review (sample subset)

### Quality Checks ✓
- [ ] Entity distribution analysis
- [ ] Relation distribution analysis
- [ ] Text diversity metrics
- [ ] Factual consistency validation

---

## Next Steps

1. **Generate Initial Dataset**
   ```bash
   uv run python scripts/generate_medical_dataset.py --num_samples 5000
   ```

2. **Review Research Document**
   - Read [Medical_NER_RE_Research.md](Medical_NER_RE_Research.md)
   - Study baseline datasets and benchmarks

3. **Optional: Download Baseline Data**
   - BioRED for validation
   - BC5CDR for comparison

4. **Format for Training**
   - Convert to instruction-tuning format
   - Create train/val/test splits

5. **Train SLM**
   - Use Llama 3.2 1B as starting point
   - Fine-tune with Unsloth
   - Monitor F1 scores

6. **Evaluate and Iterate**
   - Run evaluation metrics
   - Analyze errors
   - Generate more diverse data
   - Retrain and improve

---

## Troubleshooting

### Low F1 Scores (< 60%)
- Increase dataset size (5K → 10K)
- Add data augmentation
- Try larger model (1B → 3B)
- Increase training epochs

### Poor Relation Extraction
- Ensure entities are correct first (cascade effect)
- Add more relation examples
- Increase relation diversity
- Use relation-specific training

### Overfitting
- Increase dataset diversity
- Add regularization (dropout, weight decay)
- Use early stopping
- Generate more synthetic examples

---

## References

### Research Papers
- [BioRED Dataset](https://academic.oup.com/bib/article/23/5/bbac282/6645993)
- [MedSyn Framework](https://arxiv.org/html/2408.02056v1)
- [Synthetic Health Care Data](https://www.jmir.org/2025/1/e66279)
- [Benchmarking LLMs for Biomedical NLP](https://www.nature.com/articles/s41467-025-56989-2)

### Complete Research Document
- [Medical_NER_RE_Research.md](Medical_NER_RE_Research.md) - Full details, datasets, methods

### Code Documentation
- [synthetic_generator.py](../src/odin_slm/data/synthetic_generator.py) - Generator implementation
- [evaluator.py](../src/odin_slm/data/evaluator.py) - Evaluation metrics

---

**Ready to train!** Start by generating your first dataset:

```bash
uv run python scripts/generate_medical_dataset.py --num_samples 1000
```

**Questions?** See [Medical_NER_RE_Research.md](Medical_NER_RE_Research.md) for comprehensive details.
