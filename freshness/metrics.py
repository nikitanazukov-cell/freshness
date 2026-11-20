"""Classification metrics without an extra scikit-learn dependency."""
from __future__ import annotations
import math
import numpy as np
from .common import CLASSES


def classification_metrics(truth: list[int], predicted: list[int]) -> dict:
    if not truth or len(truth) != len(predicted):
        raise ValueError("Для метрик нужны непустые списки одинаковой длины.")
    cm = np.zeros((len(CLASSES), len(CLASSES)), dtype=np.int64)
    for actual, guess in zip(truth, predicted):
        cm[actual, guess] += 1
    per_class = {}
    for i, label in enumerate(CLASSES):
        tp = int(cm[i, i])
        support = int(cm[i].sum())
        positive = int(cm[:, i].sum())
        precision = tp / positive if positive else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {"precision": precision, "recall": recall, "f1": f1,
                            "support": support}
    n, correct = len(truth), int(np.trace(cm))
    accuracy = correct / n
    # Wilson 95% interval describes sampling uncertainty, not labeling bias.
    z = 1.959963984540054
    center = (accuracy + z*z/(2*n)) / (1 + z*z/n)
    radius = z * math.sqrt(accuracy*(1-accuracy)/n + z*z/(4*n*n)) / (1 + z*z/n)
    return {
        "n": n, "correct": correct, "accuracy": accuracy,
        "accuracy_wilson_95": [max(0., center-radius), min(1., center+radius)],
        "macro_f1": sum(v["f1"] for v in per_class.values()) / len(CLASSES),
        "per_class": per_class, "class_order": list(CLASSES),
        "confusion_matrix": cm.tolist(),
    }


def timing_stats(values: list[float]) -> dict:
    if not values:
        raise ValueError("Нет измерений времени.")
    arr = np.asarray(values, dtype=float)
    return {"n": len(values), "mean_ms": float(arr.mean()),
            "median_ms": float(np.median(arr)), "p95_ms": float(np.percentile(arr, 95)),
            "min_ms": float(arr.min()), "max_ms": float(arr.max())}
