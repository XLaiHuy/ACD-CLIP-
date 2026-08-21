"""Shared evaluation contracts for canonical Phase2B/SABRA experiments."""

from .evaluator import evaluate_records, image_score
from .metrics import binary_average_precision, binary_auroc

__all__ = ["binary_average_precision", "binary_auroc", "evaluate_records", "image_score"]
