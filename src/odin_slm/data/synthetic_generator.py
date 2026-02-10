"""Synthetic medical text generation for entity and relation extraction

This module implements MedSyn-inspired synthetic data generation for medical NER/RE tasks.
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

    # Medical entity types
    ENTITY_TYPES = [
        "Disease",
        "Symptom",
        "Drug",
        "Procedure",
        "Anatomy",
        "Lab_Test",
        "Gene",
        "Protein",
        "Chemical",
    ]

    # Relation types
    RELATION_TYPES = [
        "treats",
        "causes",
        "prevents",
        "indicates",
        "contraindicates",
        "interacts_with",
        "part_of",
        "associated_with",
    ]

    # Medical knowledge templates
    KNOWLEDGE_BASE = {
        "diseases": [
            "diabetes mellitus",
            "hypertension",
            "coronary artery disease",
            "chronic kidney disease",
            "asthma",
            "rheumatoid arthritis",
            "depression",
            "migraine",
            "pneumonia",
            "urinary tract infection",
        ],
        "symptoms": [
            "chest pain",
            "shortness of breath",
            "fever",
            "headache",
            "fatigue",
            "nausea",
            "dizziness",
            "cough",
            "abdominal pain",
            "joint pain",
        ],
        "drugs": [
            "metformin",
            "lisinopril",
            "atorvastatin",
            "aspirin",
            "levothyroxine",
            "albuterol",
            "metoprolol",
            "omeprazole",
            "amlodipine",
            "losartan",
        ],
        "procedures": [
            "coronary angiography",
            "CT scan",
            "MRI",
            "blood pressure monitoring",
            "glucose monitoring",
            "physical therapy",
            "endoscopy",
            "biopsy",
            "X-ray",
            "ultrasound",
        ],
        "lab_tests": [
            "complete blood count",
            "lipid panel",
            "hemoglobin A1c",
            "creatinine",
            "blood glucose",
            "liver function tests",
            "thyroid function tests",
            "urinalysis",
            "C-reactive protein",
            "troponin",
        ],
    }

    # Relation templates
    RELATION_TEMPLATES = {
        "treats": [
            "{drug} is used to treat {disease}",
            "{drug} effectively manages {disease}",
            "Treatment with {drug} improved {disease}",
            "The patient was prescribed {drug} for {disease}",
        ],
        "causes": [
            "{disease} caused {symptom}",
            "{symptom} was attributed to {disease}",
            "Patient presented with {symptom} due to {disease}",
            "{disease} resulted in {symptom}",
        ],
        "indicates": [
            "{test} showed elevated levels indicating {disease}",
            "{symptom} may indicate {disease}",
            "{test} results suggest {disease}",
            "The presence of {symptom} indicates possible {disease}",
        ],
        "prevents": [
            "{drug} helps prevent {disease}",
            "{procedure} can prevent {disease} progression",
            "Prophylactic {drug} prevents {disease}",
        ],
    }

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
        """Select random medical entities

        Args:
            n: Number of entities to select

        Returns:
            List of (entity_text, entity_type) tuples
        """
        entities = []
        categories = list(self.KNOWLEDGE_BASE.keys())

        for _ in range(n):
            category = random.choice(categories)
            entity_text = random.choice(self.KNOWLEDGE_BASE[category])
            entity_type = category.rstrip("s").title()  # disease -> Disease
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
        # Template for clinical notes
        templates = [
            "Patient presents with {symptom}. Physical examination reveals {finding}. "
            "Laboratory tests including {test} were ordered. "
            "Diagnosis: {disease}. Treatment plan: {drug} prescribed.",
            "Chief complaint: {symptom}. History: Patient has {disease} managed with {drug}. "
            "Assessment: {finding} on examination. Plan: {procedure} scheduled.",
            "A {age}-year-old patient with known {disease} presents with worsening {symptom}. "
            "{test} showed {finding}. Started on {drug}.",
        ]

        # Build text with entities
        text_parts = []
        entities = []
        current_pos = 0

        # Start with template
        intro = random.choice(
            [
                f"A {random.randint(25, 80)}-year-old patient presents to the clinic. ",
                "Medical record summary: ",
                "Clinical case: ",
            ]
        )
        text_parts.append(intro)
        current_pos = len(intro)

        # Add entities with context
        for i, (entity_text, entity_type) in enumerate(entities_data):
            # Add context before entity
            if i > 0:
                connector = random.choice(
                    [" The patient also has ", ". Additionally, ", ". Patient reports ", " with "]
                )
                text_parts.append(connector)
                current_pos += len(connector)

            # Add entity
            start = current_pos
            text_parts.append(entity_text)
            end = current_pos + len(entity_text)

            entities.append(Entity(text=entity_text, type=entity_type, start=start, end=end))

            current_pos = end

            # Add context after entity
            if entity_type == "Drug":
                suffix = random.choice([" was prescribed", " administered", " treatment"])
                text_parts.append(suffix)
                current_pos += len(suffix)

        # Add conclusion
        conclusion = ". Follow-up scheduled."
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

        # Define valid relation patterns
        valid_patterns = {
            ("Drug", "Disease"): "treats",
            ("Disease", "Symptom"): "causes",
            ("Lab_test", "Disease"): "indicates",
            ("Symptom", "Disease"): "indicates",
            ("Drug", "Drug"): "interacts_with",
        }

        # Try to create relations
        attempts = 0
        max_attempts = n * 10

        while len(relations) < n and attempts < max_attempts:
            # Select random pair
            head = random.choice(entities)
            tail = random.choice([e for e in entities if e != head])

            # Check if valid pattern
            pattern = (head.type, tail.type)
            if pattern in valid_patterns:
                rel_type = valid_patterns[pattern]
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
