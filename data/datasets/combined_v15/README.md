---
task_categories:
- token-classification
- text-generation
tags:
- medical-ner
- relation-extraction
- biomedical
- clinical-text
language:
- en
size_categories:
- 1K<n<10K
---

# medical-ner-re-dataset

A combined dataset for **medical Named Entity Recognition (NER) and Relation Extraction (RE)**,
formatted as instruction-following examples for fine-tuning language models.

## Task

Given a clinical text passage, extract:
- **Entities**: Disease, Drug, Symptom
- **Relations**: associated_with, causes, interacts_with, treats

## Data Sources

| Source | Description |
|--------|------------|
| ADE Corpus V2 | Drug–adverse drug event relations from case reports |
| BC5CDR | Chemical–disease relations from PubMed abstracts |
| BioRED | Biomedical relation extraction from PubMed |

## Split Statistics

| Split | Examples |
|-------|----------|
| train | 4,062 |
| validation | 950 |
| test | 958 |
| **Total** | **5,970** |

## Fields

Each example has three fields:

| Field | Description |
|-------|-------------|
| `instruction` | Task description (extract entities and relations) |
| `input` | Clinical text passage |
| `output` | Structured extraction with entities and relations |

## Usage

```python
from datasets import load_dataset

dataset = load_dataset("pabloformoso/medical-ner-re-dataset")

# Access splits
train = dataset["train"]
print(f"Train examples: {len(train)}")
print(train[0])
```

## Output Format

The `output` field contains structured extractions:

```
### Entities:
1. [Drug] aspirin
2. [Symptom] gastrointestinal bleeding

### Relations:
1. aspirin --[causes]--> gastrointestinal bleeding
```

## Licensing

This dataset combines data from multiple sources, each with their own licenses.
Please refer to the original datasets for licensing terms:
- ADE Corpus V2
- BC5CDR (BioCreative V CDR)
- BioRED
