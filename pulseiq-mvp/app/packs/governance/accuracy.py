"""Precision/recall/F1 against ground_truth.csv.

Ground truth is never passed into agent prompts - it exists only so tests
and (for the seeded demo dataset) the dashboard can report detection
accuracy per category.
"""

from __future__ import annotations


def precision_recall_f1(predicted: set[str], actual: set[str]) -> dict[str, float | int]:
    """predicted/actual are sets of log_ids flagged by the agent / ground truth."""
    tp = len(predicted & actual)
    fp = len(predicted - actual)
    fn = len(actual - predicted)
    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }
