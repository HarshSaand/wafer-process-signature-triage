"""Classification, calibration, and retrieval metrics."""
from __future__ import annotations
import numpy as np


def expected_calibration_error(y_true, probabilities, n_bins: int = 15) -> float:
    y = np.asarray(y_true); p = np.asarray(probabilities, dtype=float)
    confidence, predicted = p.max(1), p.argmax(1)
    ece = 0.0
    edges = np.linspace(0, 1, n_bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (confidence > lo) & (confidence <= hi)
        if mask.any():
            ece += mask.mean() * abs((predicted[mask] == y[mask]).mean() - confidence[mask].mean())
    return float(ece)


def multiclass_brier_score(y_true, probabilities) -> float:
    y = np.asarray(y_true); p = np.asarray(probabilities, dtype=float)
    one_hot = np.eye(p.shape[1])[y]
    return float(np.mean(np.sum((p - one_hot) ** 2, axis=1)))


def classification_metrics(y_true, probabilities, class_names=None) -> dict:
    """Return JSON-safe aggregate/per-class metrics and confusion matrix."""
    from sklearn.metrics import (balanced_accuracy_score, confusion_matrix,
        f1_score, precision_recall_fscore_support, accuracy_score, log_loss)
    y = np.asarray(y_true); p = np.asarray(probabilities); pred = p.argmax(1)
    labels = np.arange(p.shape[1]); names = class_names or [str(i) for i in labels]
    precision, recall, f1, support = precision_recall_fscore_support(
        y, pred, labels=labels, zero_division=0)
    return {
        "n_samples": int(len(y)), "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "negative_log_likelihood": float(log_loss(y, p, labels=labels)),
        "ece_15": expected_calibration_error(y, p),
        "brier_score": multiclass_brier_score(y, p),
        "confusion_matrix": confusion_matrix(y, pred, labels=labels).tolist(),
        "per_class": {str(name): {"precision": float(precision[i]),
            "recall": float(recall[i]), "f1": float(f1[i]),
            "support": int(support[i])} for i, name in enumerate(names)}
    }


def nearest_neighbors(query_embedding, reference_embeddings, reference_labels,
                      k: int = 5) -> list[dict]:
    """Cosine-similarity case retrieval for engineer-facing precedent review."""
    q = np.asarray(query_embedding, float).reshape(-1)
    r = np.asarray(reference_embeddings, float)
    sims = (r @ q) / (np.linalg.norm(r, axis=1) * np.linalg.norm(q) + 1e-12)
    idx = np.argsort(-sims)[:k]
    return [{"index": int(i), "label": str(reference_labels[i]),
             "similarity": float(sims[i])} for i in idx]

