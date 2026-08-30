"""Shared evaluation contracts for canonical Phase2B/SABRA experiments."""

from .evaluator import evaluate_records, evaluate_spool, image_score
from .metrics import binary_average_precision, binary_auroc, binary_metrics

__all__ = ["binary_average_precision", "binary_auroc", "binary_metrics", "evaluate_records", "evaluate_spool", "image_score"]
