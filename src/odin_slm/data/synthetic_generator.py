"""Synthetic medical text generation for entity and relation extraction

This module implements MedSyn-inspired synthetic data generation for medical NER/RE tasks.
V2: Expanded knowledge base (30+ terms/category), more relation patterns,
    diverse clinical note templates, fixed Lab_Test case bug.
"""

import json
import random
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class Entity:
    """Represents a medical entity"""

    text: str
    type: str
    start: int
    end: int


@dataclass
class Relation:
    """Represents a relation between two entities"""

    type: str
    head: Entity
    tail: Entity
    confidence: float = 1.0


@dataclass
class MedicalDocument:
    """Represents a medical document with entities and relations"""

    text: str
    entities: List[Entity]
    relations: List[Relation]
    metadata: Dict = None

    def to_dict(self):
        """Convert to dictionary for JSON serialization"""
        return {
            "text": self.text,
            "entities": [asdict(e) for e in self.entities],
            "relations": [
                {
                    "type": r.type,
                    "head": asdict(r.head),
                    "tail": asdict(r.tail),
                    "confidence": r.confidence,
                }
                for r in self.relations
            ],
            "metadata": self.metadata or {},
        }


class MedicalTextGenerator:
    """Generate synthetic medical texts with entity and relation annotations

    Based on MedSyn framework and recent LLM-based approaches.
    """

    # Entity types used in training
    ENTITY_TYPES = [
        "Disease",
        "Symptom",
        "Drug",
        "Procedure",
        "Lab_Test",
    ]

    # Relation types used in training
    RELATION_TYPES = [
        "treats",
        "causes",
        "indicates",
        "interacts_with",
    ]

    # Expanded medical knowledge base (~35 terms per category)
    KNOWLEDGE_BASE = {
        "diseases": [
            # Chronic diseases
            "diabetes mellitus", "hypertension", "coronary artery disease",
            "chronic kidney disease", "asthma", "rheumatoid arthritis",
            "depression", "migraine", "COPD", "heart failure",
            "osteoarthritis", "gout", "anemia", "osteoporosis",
            "hypothyroidism", "hyperthyroidism",
            # Acute conditions
            "pneumonia", "urinary tract infection", "sepsis",
            "pulmonary embolism", "deep vein thrombosis", "acute pancreatitis",
            "meningitis", "cellulitis", "appendicitis", "diverticulitis",
            # Neurological/autoimmune
            "multiple sclerosis", "Parkinson's disease", "epilepsy",
            "systemic lupus erythematosus", "Crohn's disease",
            # Cardiovascular
            "atrial fibrillation", "stroke", "peripheral artery disease",
            "aortic stenosis",
        ],
        "symptoms": [
            # Cardiopulmonary
            "chest pain", "shortness of breath", "palpitations", "wheezing",
            "cough", "hemoptysis", "orthopnea",
            # General/systemic
            "fever", "fatigue", "weight loss", "night sweats", "malaise",
            "chills", "loss of appetite",
            # Neurological
            "headache", "dizziness", "confusion", "numbness", "tingling",
            "blurred vision", "seizures", "tremor",
            # GI
            "nausea", "vomiting", "abdominal pain", "diarrhea",
            "constipation", "blood in stool", "dysphagia",
            # MSK
            "joint pain", "back pain", "muscle weakness", "swelling",
            "joint stiffness",
            # Other
            "rash", "insomnia", "frequent urination", "excessive thirst",
            "edema",
        ],
        "drugs": [
            # Diabetes
            "metformin", "insulin", "glipizide", "sitagliptin",
            # Cardiovascular
            "lisinopril", "atorvastatin", "aspirin", "metoprolol",
            "amlodipine", "losartan", "warfarin", "clopidogrel",
            "furosemide", "hydrochlorothiazide", "carvedilol",
            "spironolactone", "apixaban", "digoxin",
            # Respiratory
            "albuterol", "montelukast", "fluticasone", "tiotropium",
            # GI
            "omeprazole", "pantoprazole",
            # Pain/inflammation
            "ibuprofen", "acetaminophen", "prednisone", "naproxen",
            "colchicine",
            # Antibiotics
            "amoxicillin", "ciprofloxacin", "azithromycin", "doxycycline",
            "vancomycin", "piperacillin-tazobactam",
            # Psych/neuro
            "sertraline", "gabapentin", "duloxetine", "levetiracetam",
            # Other
            "levothyroxine", "methotrexate", "hydroxychloroquine",
            "allopurinol", "enoxaparin",
        ],
        "procedures": [
            # Imaging
            "CT scan", "MRI", "X-ray", "ultrasound", "PET scan",
            "mammography", "echocardiogram",
            # Interventional
            "coronary angiography", "cardiac catheterization", "angioplasty",
            "stent placement", "joint replacement", "appendectomy",
            "cholecystectomy",
            # Diagnostic
            "endoscopy", "colonoscopy", "bronchoscopy", "lumbar puncture",
            "bone marrow biopsy", "biopsy", "EEG", "EMG", "stress test",
            "spirometry",
            # Therapeutic
            "dialysis", "blood transfusion", "wound debridement",
            "physical therapy", "mechanical ventilation",
            # Monitoring
            "blood pressure monitoring", "glucose monitoring",
            "cardiac monitoring", "pulse oximetry",
        ],
        "lab_tests": [
            # Hematology
            "complete blood count", "coagulation panel", "prothrombin time",
            "INR", "D-dimer", "blood culture", "ferritin", "ESR",
            # Chemistry
            "basic metabolic panel", "electrolyte panel", "creatinine",
            "blood glucose", "blood urea nitrogen", "serum albumin",
            "bilirubin", "amylase", "lipase", "serum lactate",
            # Cardiac
            "troponin", "BNP", "lipid panel",
            # Endocrine/metabolic
            "hemoglobin A1c", "thyroid function tests", "vitamin D level",
            # Inflammatory/immune
            "C-reactive protein", "procalcitonin",
            # Liver/kidney
            "liver function tests",
            # Urinary
            "urinalysis", "urine culture",
            # Other
            "arterial blood gas", "PSA",
        ],
    }

    # Category name to entity type mapping (fixes the .rstrip("s").title() bug)
    _CATEGORY_TO_TYPE = {
        "diseases": "Disease",
        "symptoms": "Symptom",
        "drugs": "Drug",
        "procedures": "Procedure",
        "lab_tests": "Lab_Test",
    }

    # Diverse intro templates
    INTROS = [
        "A {age}-year-old {gender} presents to the clinic. ",
        "A {age}-year-old {gender} was admitted to the hospital. ",
        "Patient is a {age}-year-old {gender} presenting for evaluation. ",
        "Clinical note: {age}-year-old {gender} seen in follow-up. ",
        "Emergency department evaluation of a {age}-year-old {gender}. ",
        "Outpatient visit for a {age}-year-old {gender}. ",
        "Consultation note for a {age}-year-old {gender}. ",
        "Progress note: {age}-year-old {gender}, day {day} of admission. ",
        "A {age}-year-old {gender} presents with multiple complaints. ",
        "Discharge summary: {age}-year-old {gender}. ",
        "Medical record review: {age}-year-old {gender}. ",
        "Inpatient note for a {age}-year-old {gender}. ",
        "Preoperative assessment: {age}-year-old {gender}. ",
        "Urgent care visit: {age}-year-old {gender}. ",
    ]

    # Diverse connectors between entities
    CONNECTORS = [
        # Additive
        ". The patient also has ",
        ". Additionally, ",
        ". Furthermore, ",
        ". The patient reports ",
        ". History is also notable for ",
        ". Past medical history includes ",
        ". The workup revealed ",
        ". Also noted is ",
        ". Of note, ",
        # Causal/sequential
        ". This led to evaluation with ",
        ". Given these findings, ",
        ". As a result, ",
        ". Subsequently, ",
        ". This prompted ",
        # Examination-related
        ". Physical examination demonstrates ",
        ". On exam, ",
        ". Assessment reveals ",
        ". Review of systems positive for ",
        ". Findings include ",
        # Treatment-related
        ". The patient was started on ",
        ". Management included ",
        ". Treatment plan includes ",
        ". Currently taking ",
        ". Medications include ",
        # Lab/test-related
        ". Laboratory studies showed ",
        ". Diagnostic testing with ",
        ". Imaging with ",
        ". Results of ",
        ". Ordered ",
    ]

    # Type-specific suffixes added after entity mention
    ENTITY_SUFFIXES = {
        "Drug": [
            " was prescribed", " was administered", " therapy initiated",
            " was started", " was continued", " dosage adjusted",
            " twice daily", " as needed", " daily",
            "",  # sometimes no suffix
        ],
        "Disease": [
            " was diagnosed", " is well-controlled", " has been stable",
            " with recent exacerbation", " requiring management",
            " per history",
            "", "", "",  # often no suffix
        ],
        "Symptom": [
            " reported by patient", " worsening over past week",
            " with onset 3 days ago", " intermittent in nature",
            " described as moderate", " acute onset",
            "", "", "", "",  # usually no suffix
        ],
        "Procedure": [
            " was performed", " was ordered", " completed",
            " showed normal results", " is scheduled",
            " results reviewed",
        ],
        "Lab_Test": [
            " was ordered", " results pending", " showed abnormal values",
            " within normal limits", " elevated",
            " results reviewed", " drawn on admission",
        ],
    }

    # Conclusions
    CONCLUSIONS = [
        ". Follow-up scheduled in {weeks} weeks.",
        ". Patient to return for reassessment.",
        ". Continue current management and monitor.",
        ". Will reassess at next visit.",
        ". Plan discussed with patient.",
        ". Patient counseled on medication compliance.",
        ". Referral placed for specialist evaluation.",
        ". Discharge with above medications and follow-up.",
        ". Close monitoring recommended.",
        ". Patient advised on lifestyle modifications.",
    ]

    def __init__(self, seed: int = 42):
        """Initialize the generator

        Args:
            seed: Random seed for reproducibility
        """
        random.seed(seed)
        self.generated_count = 0

    def generate_clinical_note(
        self, num_entities: int = 5, num_relations: int = 3
    ) -> MedicalDocument:
        """Generate a synthetic clinical note with entities and relations

        Args:
            num_entities: Number of entities to include
            num_relations: Number of relations to include

        Returns:
            MedicalDocument with text, entities, and relations
        """
        # Select random entities
        entities_data = self._select_random_entities(num_entities)

        # Generate text with entities
        text, entities = self._generate_text_with_entities(entities_data)

        # Generate relations
        relations = self._generate_relations(entities, num_relations)

        self.generated_count += 1

        return MedicalDocument(
            text=text,
            entities=entities,
            relations=relations,
            metadata={"id": self.generated_count, "synthetic": True},
        )

    def _select_random_entities(self, n: int) -> List[Tuple[str, str]]:
        """Select random medical entities ensuring type diversity

        Args:
            n: Number of entities to select

        Returns:
            List of (entity_text, entity_type) tuples
        """
        entities = []
        categories = list(self.KNOWLEDGE_BASE.keys())
        selected_categories = []

        for i in range(n):
            # Ensure at least 2 different categories for first picks
            if i < 2:
                available = [c for c in categories if c not in selected_categories]
                if available:
                    category = random.choice(available)
                else:
                    category = random.choice(categories)
            else:
                category = random.choice(categories)

            selected_categories.append(category)
            entity_text = random.choice(self.KNOWLEDGE_BASE[category])
            entity_type = self._CATEGORY_TO_TYPE[category]
            entities.append((entity_text, entity_type))

        return entities

    def _generate_text_with_entities(
        self, entities_data: List[Tuple[str, str]]
    ) -> Tuple[str, List[Entity]]:
        """Generate clinical text incorporating entities

        Args:
            entities_data: List of (text, type) tuples

        Returns:
            Tuple of (text, list of Entity objects with positions)
        """
        text_parts = []
        entities = []
        current_pos = 0

        # Rich intro
        age = random.randint(18, 85)
        gender = random.choice(["male", "female"])
        day = random.randint(1, 14)

        intro = random.choice(self.INTROS).format(age=age, gender=gender, day=day)
        text_parts.append(intro)
        current_pos = len(intro)

        # Add entities with context
        for i, (entity_text, entity_type) in enumerate(entities_data):
            # Add connector before entity (except first)
            if i > 0:
                connector = random.choice(self.CONNECTORS)
                text_parts.append(connector)
                current_pos += len(connector)

            # Add entity
            start = current_pos
            text_parts.append(entity_text)
            end = current_pos + len(entity_text)

            entities.append(Entity(text=entity_text, type=entity_type, start=start, end=end))

            current_pos = end

            # Add type-appropriate suffix
            suffixes = self.ENTITY_SUFFIXES.get(entity_type, [""])
            suffix = random.choice(suffixes)
            if suffix:
                text_parts.append(suffix)
                current_pos += len(suffix)

        # Add conclusion
        conclusion = random.choice(self.CONCLUSIONS).format(weeks=random.randint(1, 12))
        text_parts.append(conclusion)

        text = "".join(text_parts)

        return text, entities

    def _generate_relations(
        self, entities: List[Entity], n: int
    ) -> List[Relation]:
        """Generate relations between entities

        Args:
            entities: List of Entity objects
            n: Number of relations to generate

        Returns:
            List of Relation objects
        """
        relations = []

        if len(entities) < 2:
            return relations

        # Valid relation patterns (head_type, tail_type) -> relation_type
        valid_patterns = {
            ("Drug", "Disease"): "treats",
            ("Drug", "Symptom"): "treats",         # drugs treat symptoms
            ("Procedure", "Disease"): "treats",     # procedures treat diseases
            ("Disease", "Symptom"): "causes",
            ("Drug", "Symptom"): "causes",          # drugs cause side effects
            ("Lab_Test", "Disease"): "indicates",   # Fixed: was "Lab_test" (case bug)
            ("Symptom", "Disease"): "indicates",
            ("Lab_Test", "Symptom"): "indicates",   # labs indicate symptoms
            ("Drug", "Drug"): "interacts_with",
        }

        # Drug→Symptom can be either treats or causes (randomly chosen per pair)
        drug_symptom_ambiguous = True

        seen_pairs = set()
        attempts = 0
        max_attempts = n * 15

        while len(relations) < n and attempts < max_attempts:
            head = random.choice(entities)
            tail = random.choice([e for e in entities if e != head])

            # Avoid duplicate entity pairs
            pair_key = (head.text, tail.text)
            if pair_key in seen_pairs:
                attempts += 1
                continue

            pattern = (head.type, tail.type)
            if pattern in valid_patterns:
                rel_type = valid_patterns[pattern]

                # Drug→Symptom: randomly choose treats or causes
                if pattern == ("Drug", "Symptom") and drug_symptom_ambiguous:
                    rel_type = random.choice(["treats", "causes"])

                seen_pairs.add(pair_key)
                relations.append(
                    Relation(
                        type=rel_type,
                        head=head,
                        tail=tail,
                        confidence=round(random.uniform(0.85, 1.0), 2),
                    )
                )

            attempts += 1

        return relations

    def generate_dataset(
        self, n: int, output_path: Optional[Path] = None
    ) -> List[MedicalDocument]:
        """Generate a dataset of synthetic medical documents

        Args:
            n: Number of documents to generate
            output_path: Optional path to save dataset as JSON

        Returns:
            List of MedicalDocument objects
        """
        print(f"Generating {n} synthetic medical documents...")

        documents = []
        for i in range(n):
            # Vary complexity
            num_entities = random.randint(3, 8)
            num_relations = random.randint(2, min(5, num_entities - 1))

            doc = self.generate_clinical_note(num_entities, num_relations)
            documents.append(doc)

            if (i + 1) % 100 == 0:
                print(f"  Generated {i + 1}/{n} documents")

        print(f"✓ Generated {len(documents)} documents")

        # Save if output path provided
        if output_path:
            self.save_dataset(documents, output_path)

        return documents

    def save_dataset(self, documents: List[MedicalDocument], output_path: Path):
        """Save dataset to JSON file

        Args:
            documents: List of MedicalDocument objects
            output_path: Path to save JSON file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = [doc.to_dict() for doc in documents]

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        print(f"✓ Saved dataset to {output_path}")

    def load_dataset(self, input_path: Path) -> List[MedicalDocument]:
        """Load dataset from JSON file

        Args:
            input_path: Path to JSON file

        Returns:
            List of MedicalDocument objects
        """
        with open(input_path) as f:
            data = json.load(f)

        documents = []
        for item in data:
            entities = [Entity(**e) for e in item["entities"]]
            relations = [
                Relation(
                    type=r["type"],
                    head=Entity(**r["head"]),
                    tail=Entity(**r["tail"]),
                    confidence=r.get("confidence", 1.0),
                )
                for r in item["relations"]
            ]

            doc = MedicalDocument(
                text=item["text"],
                entities=entities,
                relations=relations,
                metadata=item.get("metadata", {}),
            )
            documents.append(doc)

        return documents


if __name__ == "__main__":
    # Example usage
    generator = MedicalTextGenerator(seed=42)

    # Generate single document
    doc = generator.generate_clinical_note(num_entities=5, num_relations=3)

    print("=" * 70)
    print("Example Synthetic Medical Document")
    print("=" * 70)
    print(f"\nText:\n{doc.text}\n")

    print(f"Entities ({len(doc.entities)}):")
    for i, entity in enumerate(doc.entities, 1):
        print(f"  {i}. [{entity.type}] {entity.text} ({entity.start}:{entity.end})")

    print(f"\nRelations ({len(doc.relations)}):")
    for i, rel in enumerate(doc.relations, 1):
        print(
            f"  {i}. {rel.head.text} --[{rel.type}]--> {rel.tail.text} "
            f"(confidence: {rel.confidence})"
        )

    # Generate small dataset
    print("\n" + "=" * 70)
    print("Generating Dataset")
    print("=" * 70)

    dataset = generator.generate_dataset(10, output_path="data/datasets/medical_ner_re_sample.json")
    print(f"\n✓ Generated {len(dataset)} documents")
