"""Simple privacy attacks used to compare leakage across methods."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

from .model import binary_losses


def loss_threshold_membership_auc(
    weights: np.ndarray,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    *,
    seed: int,
) -> float:
    """Measure membership inference using negative per-example loss as score.

    An AUC of 0.5 indicates chance-level separation; larger values indicate
    that training samples are easier to distinguish from held-out samples.
    """
    rng = np.random.default_rng(seed)
    sample_count = min(len(y_train), len(y_test))
    train_idx = rng.choice(len(y_train), size=sample_count, replace=False)
    test_idx = rng.choice(len(y_test), size=sample_count, replace=False)
    train_scores = -binary_losses(weights, x_train[train_idx], y_train[train_idx])
    test_scores = -binary_losses(weights, x_test[test_idx], y_test[test_idx])
    labels = np.r_[np.ones(sample_count), np.zeros(sample_count)]
    scores = np.r_[train_scores, test_scores]
    return float(roc_auc_score(labels, scores))
