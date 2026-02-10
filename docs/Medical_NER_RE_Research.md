# Medical Entity and Relation Extraction - Research Summary

**Date**: 2026-02-08
**Domain**: Biomedical/Clinical Text
**Task**: Named Entity Recognition (NER) + Relation Extraction (RE)

## Executive Summary

This document summarizes current state-of-the-art approaches for medical entity and relation extraction, including baseline datasets, evaluation metrics, and synthetic data generation methods using LLMs.

---

## 1. Baseline Datasets

### 1.1 Primary Benchmark Datasets

#### **BioRED** (Recommended Baseline)
- **Description**: Rich biomedical relation extraction dataset with multiple entity types
- **Entities**: 6 types (gene, disease, chemical, variant, species, cell line)
- **Relations**: 8 different types (e.g., positive correlation)
- **Size**: 600 abstracts (train), 1000 abstracts (validation)
- **Features**: Novelty annotations to distinguish known facts from novel findings
- **Benchmark F1**: 74.4% (baseline) → 79.6% (SOTA)
- **Source**: [BioRED Dataset](https://academic.oup.com/bib/article/23/5/bbac282/6645993)

#### **ChemProt**
- **Description**: Chemical-protein interaction relations
- **Use Case**: Drug-target interactions
- **Benchmark F1**: ~77.0% (BioBERT)

#### **BC5CDR**
- **Description**: Chemical-disease relations
- **Entities**: Chemicals, Diseases
- **Standard benchmark for chemical-disease NER/RE

#### **DrugProt**
- **Description**: Drug-protein relations
- **Recent benchmark from BioCreative VII

### 1.2 Additional Medical Datasets

#### **MedMentions**
- **Size**: 4,000+ abstracts, 350,000+ linked mentions
- **Coverage**: 3 million UMLS 2017 concepts
- **Use**: Large-scale medical entity linking

#### **ADE Corpus**
- **Focus**: Adverse drug effects
- **Size**: ~30,000 MEDLINE documents
- **Entities**: Drugs, Adverse effects
- **Relations**: Drug-ADE relations

#### **GAD Corpus**
- **Focus**: Gene-disease associations
- **Size**: 5,329 relations
- **Source**: PubMed sentences

#### **Clinical Datasets**
- **n2c2/i2b2**: Clinical notes with entity annotations
- **MIMIC-III**: Clinical records (requires approval)

### 1.3 Standard Evaluation Datasets

Eight commonly used biomedical NER datasets:
- AnatEM (anatomy)
- BC4CHEMD (chemicals)
- BioNLP13CG (cancer genetics)
- JNLPBA (genes/proteins)
- Linnaeus (species)
- NCBI-Disease
- S800 (species)

---

## 2. Entity and Relation Types

### 2.1 Common Medical Entity Types

1. **Clinical Entities**
   - Disease/Disorder
   - Symptom/Sign
   - Medication/Drug
   - Procedure/Treatment
   - Anatomy/Body Part
   - Lab Test/Measurement

2. **Biomedical Research Entities**
   - Gene/Protein
   - Chemical/Compound
   - Cell Line
   - Species/Organism
   - Variant/Mutation

3. **Context Entities**
   - Dosage
   - Frequency
   - Duration
   - Severity
   - Temporal expressions

### 2.2 Common Relation Types

1. **Clinical Relations**
   - Disease-Treatment
   - Drug-Adverse Effect
   - Symptom-Disease
   - Test-Disease
   - Drug-Dosage
   - Contraindication

2. **Biomedical Relations**
   - Gene-Disease association
   - Drug-Target interaction
   - Protein-Protein interaction
   - Chemical-Disease relation
   - Gene-Gene interaction

3. **Temporal/Causal Relations**
   - Causes
   - Prevents
   - Treats
   - Worsens
   - Before/After

---

## 3. Evaluation Metrics

### 3.1 Standard Metrics

#### **Entity Recognition (NER)**
```
Precision = TP / (TP + FP)
Recall = TP / (TP + FN)
F1 = 2 * (Precision * Recall) / (Precision + Recall)
```

**Matching Strategies**:
- **Exact Match**: Predicted span must exactly match gold standard
- **Partial Match**: Overlap allowed
- **Token-level**: Token-by-token evaluation

#### **Relation Extraction (RE)**
```
Precision_RE = Correct Relations / Predicted Relations
Recall_RE = Correct Relations / Gold Relations
F1_RE = 2 * (Precision_RE * Recall_RE) / (Precision_RE + Recall_RE)
```

**Evaluation Modes**:
- **Strict**: Both entities and relation type must be correct
- **Relaxed**: Allows partial entity match
- **Boundaries**: Entity boundaries + relation
- **Type**: Relation type only

### 3.2 Multi-Class Metrics

- **Macro F1**: Average F1 across all classes (treats all classes equally)
- **Micro F1**: Global F1 across all predictions (weighted by frequency)
- **Weighted F1**: Weighted by class frequency

### 3.3 Performance Benchmarks (2025-2026)

| Task | Dataset | Model | F1 Score |
|------|---------|-------|----------|
| NER | Medical (general) | PubMedBERT | 88.8% |
| NER | Clinical EHR | BioBERT | 99.71% |
| RE | BioRED | SOTA | 79.6% |
| RE | ChemProt | BioBERT | 77.0% |
| RE | Natural Products | BioGPT-Large | 59.0% |
| RE (synthetic) | Various | +Augmentation | +24.7% |

**Sources**:
- [Entity Relation Extraction Evaluation](https://pmc.ncbi.nlm.nih.gov/articles/PMC9622485/)
- [Benchmarking LLMs for Biomedical NLP](https://www.nature.com/articles/s41467-025-56989-2)

---

## 4. Synthetic Data Generation Approaches

### 4.1 Recent Frameworks (2025-2026)

#### **MedSyn Framework**
- **Approach**: LLM + Medical Knowledge Graph (MKG)
- **Models**: GPT-4, Fine-tuned LLaMA
- **Process**:
  1. Sample medical information from MKG
  2. Generate synthetic clinical notes
  3. Apply NER tagging or ICD coding
- **Performance**: Significant improvement in downstream tasks
- **Source**: [MedSyn Framework](https://arxiv.org/html/2408.02056v1)

#### **Three-Step Pipeline**
1. **Generate**: Use GPT-2/GPT-4 to generate synthetic medical text
2. **Annotate**: Use GPT-3.5-Turbo/GPT-4 for entity annotation
3. **Fine-tune**: Train NER models on synthetic data
- **Results**: F1 = 0.69 (drug extraction), F1 = 0.38 (procedure extraction)
- **Source**: [Synthetic Health Care Data Study](https://www.jmir.org/2025/1/e66279)

#### **SDG Framework (Portuguese Clinical NER)**
- **Architecture**: System/user prompt + few-shot examples
- **Model**: GPT-4o
- **Approach**: Feed real clinical annotated texts to generate synthetic data
- **Result**: Increases NER model performance significantly
- **Source**: [LLM-Based Framework](https://link.springer.com/chapter/10.1007/978-3-032-05176-9_26)

### 4.2 Generation Methods (2025-2026)

Distribution of generation methods in recent studies:
- **LLM Prompting**: 74.6%
- **Fine-tuning**: 20.3%
- **Specialized Models**: 5.1%

**Data Modalities**:
- Unstructured text: 78.0%
- Tabular data: 13.6%
- Multimodal: 8.4%

### 4.3 Best Practices

1. **Hybrid Approaches**
   - Combine retrieval-based grounding with domain-specific fine-tuning
   - Improves factual accuracy and reduces hallucinations

2. **Data Augmentation**
   - Average improvement: +24.7% F1 score
   - Techniques: Paraphrasing, entity substitution, relation rewording

3. **Quality Control**
   - Use multiple LLMs for cross-validation
   - Human expert review of sample subset
   - Automatic consistency checks

4. **Iterative Refinement**
   - WHERE and WHICH debate approach for biomedical data
   - Iterative improvement of synthetic examples

**Source**: [Synthetic Data Generation in Biomedical Research](https://link.springer.com/article/10.1007/s41666-026-00229-9)

---

## 5. Recommended Approach for This Project

### 5.1 Baseline Strategy

1. **Start with BioRED**
   - Use as baseline for evaluation
   - Well-established benchmark
   - Multiple entity and relation types

2. **Supplement with Domain-Specific Data**
   - BC5CDR for chemical-disease relations
   - ADE for adverse effects
   - Clinical datasets if available

### 5.2 Synthetic Data Generation Pipeline

```
1. Knowledge Source → 2. Prompt Engineering → 3. LLM Generation → 4. Annotation → 5. Validation → 6. Training
```

**Phase 1: Knowledge Base**
- Medical ontologies (UMLS, SNOMED-CT)
- Extract entity-relation templates
- Create diverse medical scenarios

**Phase 2: Generation**
- Use GPT-4 or fine-tuned LLaMA for text generation
- Generate clinical notes/abstracts with embedded entities
- Ensure factual consistency with medical knowledge

**Phase 3: Annotation**
- Automatic NER labeling using LLM
- Relation annotation with structured output
- Format: BIO/BIOES tagging + relation triples

**Phase 4: Quality Assurance**
- Cross-validation with multiple models
- Factual consistency checks
- Diversity metrics

### 5.3 Evaluation Strategy

1. **Intrinsic Evaluation**
   - Entity recognition F1 (strict match)
   - Relation extraction F1 (boundaries + type)
   - Macro and Micro F1 for multi-class

2. **Extrinsic Evaluation**
   - Test on BioRED test set
   - Compare with baseline models
   - Zero-shot performance on unseen entity/relation types

3. **Ablation Studies**
   - Synthetic only vs. Real only vs. Hybrid
   - Different generation strategies
   - Data augmentation impact

---

## 6. Implementation Roadmap

### Phase 1: Data Collection (Week 1)
- [ ] Download BioRED dataset
- [ ] Prepare BC5CDR and ADE corpora
- [ ] Set up data preprocessing pipeline
- [ ] Analyze entity/relation distributions

### Phase 2: Baseline Evaluation (Week 1-2)
- [ ] Implement evaluation metrics
- [ ] Test existing models on baseline
- [ ] Establish performance benchmarks

### Phase 3: Synthetic Generation (Week 2-3)
- [ ] Design prompt templates
- [ ] Implement MedSyn-inspired pipeline
- [ ] Generate synthetic medical texts
- [ ] Automatic annotation with LLM

### Phase 4: Training (Week 3-4)
- [ ] Fine-tune SLM on synthetic + real data
- [ ] Hyperparameter optimization
- [ ] Model evaluation and comparison

### Phase 5: Analysis (Week 4)
- [ ] Performance analysis
- [ ] Error analysis
- [ ] Documentation and reporting

---

## 7. Tools and Libraries

### Data Processing
- **datasets**: Hugging Face datasets
- **pandas**: Data manipulation
- **spacy**: Text preprocessing
- **brat**: Annotation format handling

### Synthetic Generation
- **OpenAI API**: GPT-4 for generation
- **Unsloth**: Fast fine-tuning for custom models
- **LangChain**: Prompt management
- **Guidance**: Structured LLM output

### Evaluation
- **seqeval**: NER evaluation metrics
- **scikit-learn**: Classification metrics
- **nervaluate**: Advanced NER evaluation

### Visualization
- **matplotlib/seaborn**: Performance plots
- **wandb**: Experiment tracking
- **displacy**: Entity visualization

---

## 8. Key References

### Datasets
- [BioRED: Rich Biomedical Relation Extraction Dataset](https://academic.oup.com/bib/article/23/5/bbac282/6645993)
- [Surveying Biomedical Relation Extraction](https://academic.oup.com/bib/article/25/3/bbae132/7644532)
- [Named Entity Recognition Survey](https://www.sciencedirect.com/science/article/abs/pii/S0925231224019428)

### Synthetic Data Generation
- [MedSyn: LLM-based Medical Text Generation](https://arxiv.org/html/2408.02056v1)
- [Synthetic Health Care Data for NER](https://www.jmir.org/2025/1/e66279)
- [Scoping Review of Synthetic Data in Biomedical Research](https://link.springer.com/article/10.1007/s41666-026-00229-9)

### Evaluation
- [Entity Relation Extraction Evaluation](https://pmc.ncbi.nlm.nih.gov/articles/PMC9622485/)
- [Benchmarking LLMs for Biomedical NLP](https://www.nature.com/articles/s41467-025-56989-2)
- [Relation Extraction in Underexplored Domains](https://direct.mit.edu/coli/article/50/3/953/121178/Relation-Extraction-in-Underexplored-Biomedical)

### Methods
- [Optimized Biomedical Entity Relation Extraction with GPT-4](https://pmc.ncbi.nlm.nih.gov/articles/PMC11463225/)
- [Data Augmentation in Medical Domain](https://atm.amegroups.org/article/view/102515/html)

---

## 9. Success Criteria

### Minimum Viable Product (MVP)
- F1 ≥ 70% on entity recognition (medical domain)
- F1 ≥ 60% on relation extraction (key relation types)
- 1,000+ synthetic examples with high quality

### Target Performance
- F1 ≥ 85% on entity recognition (match PubMedBERT baseline)
- F1 ≥ 75% on relation extraction (competitive with ChemProt)
- 5,000+ diverse synthetic examples

### Stretch Goals
- F1 ≥ 90% on entity recognition
- F1 ≥ 80% on relation extraction (approach SOTA)
- Zero-shot generalization to new medical domains

---

**Document Version**: 1.0
**Last Updated**: 2026-02-08
**Author**: Odin SLM Research Team
