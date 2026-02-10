"""Evaluation metrics for Named Entity Recognition and Relation Extraction

Implements standard evaluation metrics for medical NER/RE tasks.
"""

from typing import List, Dict, Set, Tuple
from dataclasses import dataclass
from collections import defaultdict
import numpy as np


@dataclass
class EntityPrediction:
    """Entity prediction with position"""

    text: str
    type: str
    start: int
    end: int


@dataclass
class RelationPrediction:
    """Relation prediction"""

    type: str
    head: EntityPrediction
    tail: EntityPrediction


class NERREvaluator:
    """Evaluator for Named Entity Recognition and Relation Extraction

    Implements standard metrics: Precision, Recall, F1 (Exact, Partial, Type)
    """

    def __init__(self, matching_mode: str = "exact"):
        """Initialize evaluator

        Args:
            matching_mode: One of 'exact', 'partial', 'type'
                - exact: Exact span match required
                - partial: Overlap allowed
                - type: Only entity type matters
        """
        self.matching_mode = matching_mode
        self.reset()

    def reset(self):
        """Reset evaluation counters"""
        self.tp = 0  # True positives
        self.fp = 0  # False positives
        self.fn = 0  # False negatives

        # Per-type metrics
        self.per_type_tp = defaultdict(int)
        self.per_type_fp = defaultdict(int)
        self.per_type_fn = defaultdict(int)

    def evaluate_entities(
        self, predictions: List[EntityPrediction], gold: List[EntityPrediction]
    ) -> Dict:
        """Evaluate entity predictions

        Args:
            predictions: List of predicted entities
            gold: List of gold standard entities

        Returns:
            Dictionary with precision, recall, F1 scores
        """
        self.reset()

        # Convert to sets for matching
        pred_set = set()
        gold_set = set()

        for entity in predictions:
            pred_set.add(self._entity_to_tuple(entity))

        for entity in gold:
            gold_set.add(self._entity_to_tuple(entity))

        # Calculate matches
        if self.matching_mode == "exact":
            matches = pred_set & gold_set
        elif self.matching_mode == "partial":
            matches = self._partial_match(predictions, gold)
        elif self.matching_mode == "type":
            matches = self._type_match(predictions, gold)
        else:
            raise ValueError(f"Unknown matching mode: {self.matching_mode}")

        self.tp = len(matches)
        self.fp = len(pred_set) - len(matches)
        self.fn = len(gold_set) - len(matches)

        # Calculate metrics
        precision = self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0.0
        recall = self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "support": len(gold),
        }

    def evaluate_entities_per_type(
        self, predictions: List[EntityPrediction], gold: List[EntityPrediction]
    ) -> Dict:
        """Evaluate entities per type (Macro/Micro F1)

        Args:
            predictions: List of predicted entities
            gold: List of gold standard entities

        Returns:
            Dictionary with per-type and aggregate metrics
        """
        # Group by type
        pred_by_type = defaultdict(list)
        gold_by_type = defaultdict(list)

        for entity in predictions:
            pred_by_type[entity.type].append(entity)

        for entity in gold:
            gold_by_type[entity.type].append(entity)

        # Get all types
        all_types = set(pred_by_type.keys()) | set(gold_by_type.keys())

        # Evaluate per type
        per_type_results = {}
        for entity_type in all_types:
            preds = pred_by_type[entity_type]
            golds = gold_by_type[entity_type]

            result = self.evaluate_entities(preds, golds)
            per_type_results[entity_type] = result

        # Calculate macro and micro averages
        macro_p = np.mean([r["precision"] for r in per_type_results.values()])
        macro_r = np.mean([r["recall"] for r in per_type_results.values()])
        macro_f1 = np.mean([r["f1"] for r in per_type_results.values()])

        total_tp = sum(r["tp"] for r in per_type_results.values())
        total_fp = sum(r["fp"] for r in per_type_results.values())
        total_fn = sum(r["fn"] for r in per_type_results.values())

        micro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        micro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        micro_f1 = (
            2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0.0
        )

        return {
            "per_type": per_type_results,
            "macro": {"precision": macro_p, "recall": macro_r, "f1": macro_f1},
            "micro": {"precision": micro_p, "recall": micro_r, "f1": micro_f1},
        }

    def evaluate_relations(
        self,
        predictions: List[RelationPrediction],
        gold: List[RelationPrediction],
        entity_matching: str = "exact",
    ) -> Dict:
        """Evaluate relation extraction

        Args:
            predictions: List of predicted relations
            gold: List of gold standard relations
            entity_matching: How to match entities ('exact', 'type')

        Returns:
            Dictionary with precision, recall, F1 scores
        """
        pred_set = set()
        gold_set = set()

        for rel in predictions:
            pred_set.add(self._relation_to_tuple(rel, entity_matching))

        for rel in gold:
            gold_set.add(self._relation_to_tuple(rel, entity_matching))

        # Calculate matches
        matches = pred_set & gold_set

        tp = len(matches)
        fp = len(pred_set) - len(matches)
        fn = len(gold_set) - len(matches)

        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "support": len(gold),
        }

    def evaluate_relations_per_type(
        self, predictions: List[RelationPrediction], gold: List[RelationPrediction]
    ) -> Dict:
        """Evaluate relations per type

        Args:
            predictions: List of predicted relations
            gold: List of gold standard relations

        Returns:
            Dictionary with per-type and aggregate metrics
        """
        # Group by type
        pred_by_type = defaultdict(list)
        gold_by_type = defaultdict(list)

        for rel in predictions:
            pred_by_type[rel.type].append(rel)

        for rel in gold:
            gold_by_type[rel.type].append(rel)

        # Get all types
        all_types = set(pred_by_type.keys()) | set(gold_by_type.keys())

        # Evaluate per type
        per_type_results = {}
        for rel_type in all_types:
            preds = pred_by_type[rel_type]
            golds = gold_by_type[rel_type]

            result = self.evaluate_relations(preds, golds)
            per_type_results[rel_type] = result

        # Calculate macro and micro averages
        macro_p = np.mean([r["precision"] for r in per_type_results.values()])
        macro_r = np.mean([r["recall"] for r in per_type_results.values()])
        macro_f1 = np.mean([r["f1"] for r in per_type_results.values()])

        total_tp = sum(r["tp"] for r in per_type_results.values())
        total_fp = sum(r["fp"] for r in per_type_results.values())
        total_fn = sum(r["fn"] for r in per_type_results.values())

        micro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        micro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        micro_f1 = (
            2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0.0
        )

        return {
            "per_type": per_type_results,
            "macro": {"precision": macro_p, "recall": macro_r, "f1": macro_f1},
            "micro": {"precision": micro_p, "recall": micro_r, "f1": micro_f1},
        }

    def _entity_to_tuple(self, entity: EntityPrediction) -> Tuple:
        """Convert entity to hashable tuple

        Args:
            entity: EntityPrediction object

        Returns:
            Tuple representation
        """
        if self.matching_mode == "exact":
            return (entity.text, entity.type, entity.start, entity.end)
        elif self.matching_mode == "type":
            return (entity.type,)
        else:
            return (entity.text, entity.type, entity.start, entity.end)

    def _partial_match(
        self, predictions: List[EntityPrediction], gold: List[EntityPrediction]
    ) -> Set:
        """Find partial matches between predictions and gold

        Args:
            predictions: Predicted entities
            gold: Gold standard entities

        Returns:
            Set of matched prediction tuples
        """
        matches = set()

        for pred in predictions:
            for gold_entity in gold:
                # Check overlap and type match
                if pred.type == gold_entity.type:
                    # Check span overlap
                    if not (pred.end <= gold_entity.start or pred.start >= gold_entity.end):
                        matches.add(self._entity_to_tuple(pred))
                        break

        return matches

    def _type_match(
        self, predictions: List[EntityPrediction], gold: List[EntityPrediction]
    ) -> Set:
        """Match entities by type only

        Args:
            predictions: Predicted entities
            gold: Gold standard entities

        Returns:
            Set of matched types
        """
        pred_types = [p.type for p in predictions]
        gold_types = [g.type for g in gold]

        matches = set()
        for pred_type in pred_types:
            if pred_type in gold_types:
                matches.add((pred_type,))

        return matches

    def _relation_to_tuple(
        self, relation: RelationPrediction, entity_matching: str
    ) -> Tuple:
        """Convert relation to hashable tuple

        Args:
            relation: RelationPrediction object
            entity_matching: How to represent entities

        Returns:
            Tuple representation
        """
        if entity_matching == "exact":
            return (
                relation.type,
                (
                    relation.head.text,
                    relation.head.type,
                    relation.head.start,
                    relation.head.end,
                ),
                (
                    relation.tail.text,
                    relation.tail.type,
                    relation.tail.start,
                    relation.tail.end,
                ),
            )
        elif entity_matching == "type":
            return (relation.type, relation.head.type, relation.tail.type)
        else:
            return (
                relation.type,
                (
                    relation.head.text,
                    relation.head.type,
                    relation.head.start,
                    relation.head.end,
                ),
                (
                    relation.tail.text,
                    relation.tail.type,
                    relation.tail.start,
                    relation.tail.end,
                ),
            )

    @staticmethod
    def print_results(results: Dict, title: str = "Evaluation Results"):
        """Pretty print evaluation results

        Args:
            results: Results dictionary
            title: Title for the results
        """
        print("=" * 70)
        print(title)
        print("=" * 70)

        if "per_type" in results:
            # Per-type results
            print("\nPer-Type Results:")
            print("-" * 70)
            for entity_type, metrics in results["per_type"].items():
                print(
                    f"  {entity_type:20s} | "
                    f"P: {metrics['precision']:.3f} | "
                    f"R: {metrics['recall']:.3f} | "
                    f"F1: {metrics['f1']:.3f} | "
                    f"Support: {metrics['support']}"
                )

            print("\nAggregate Metrics:")
            print("-" * 70)
            print(
                f"  {'Macro Average':20s} | "
                f"P: {results['macro']['precision']:.3f} | "
                f"R: {results['macro']['recall']:.3f} | "
                f"F1: {results['macro']['f1']:.3f}"
            )
            print(
                f"  {'Micro Average':20s} | "
                f"P: {results['micro']['precision']:.3f} | "
                f"R: {results['micro']['recall']:.3f} | "
                f"F1: {results['micro']['f1']:.3f}"
            )
        else:
            # Simple results
            print(
                f"\nPrecision: {results['precision']:.3f} | "
                f"Recall: {results['recall']:.3f} | "
                f"F1: {results['f1']:.3f}"
            )
            print(
                f"TP: {results['tp']} | FP: {results['fp']} | "
                f"FN: {results['fn']} | Support: {results['support']}"
            )

        print("=" * 70)


if __name__ == "__main__":
    # Example usage
    print("NER/RE Evaluator - Example Usage\n")

    # Example entity predictions
    gold_entities = [
        EntityPrediction("diabetes", "Disease", 0, 8),
        EntityPrediction("metformin", "Drug", 20, 29),
        EntityPrediction("fever", "Symptom", 40, 45),
    ]

    pred_entities = [
        EntityPrediction("diabetes", "Disease", 0, 8),  # Correct
        EntityPrediction("metformin", "Drug", 20, 29),  # Correct
        EntityPrediction("headache", "Symptom", 50, 58),  # False positive
    ]

    # Evaluate entities
    evaluator = NERREvaluator(matching_mode="exact")
    results = evaluator.evaluate_entities(pred_entities, gold_entities)

    NERREvaluator.print_results(results, "Entity Recognition Results")
