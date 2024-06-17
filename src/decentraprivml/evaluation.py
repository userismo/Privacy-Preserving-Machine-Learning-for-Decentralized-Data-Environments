"""Utility evaluation metrics."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score

from .model import predict_proba


def evaluate_binary(weights: np.ndarray, x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    probs = np.clip(predict_proba(weights, x), 1e-9, 1.0 - 1e-9)
    predictions = (probs >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y, predictions)),
        "f1": float(f1_score(y, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, probs)),
        "log_loss": float(log_loss(y, probs, labels=[0, 1])),
    }
