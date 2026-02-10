"""Data processing modules for Odin SLM"""

from .synthetic_generator import MedicalTextGenerator
from .evaluator import NERREvaluator

__all__ = ["MedicalTextGenerator", "NERREvaluator"]
