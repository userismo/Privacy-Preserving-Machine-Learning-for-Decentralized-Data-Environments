"""Small, dependency-light logistic regression used by the benchmark."""

from __future__ import annotations

import numpy as np


def add_bias(x: np.ndarray) -> np.ndarray:
    """Append a constant feature so the bias is part of the weight vector."""
    return np.hstack([x, np.ones((x.shape[0], 1), dtype=x.dtype)])


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    z = np.clip(z, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-z))


def predict_proba(weights: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Return P(y=1) for each row."""
    return sigmoid(add_bias(x) @ weights)


def binary_losses(weights: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Per-example binary cross-entropy losses."""
    p = np.clip(predict_proba(weights, x), 1e-9, 1.0 - 1e-9)
    return -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))


def train_logistic(
    initial_weights: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    *,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    l2: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float]:
    """Train logistic regression with shuffled mini-batch gradient descent."""
    weights = initial_weights.astype(float, copy=True)
    xb = add_bias(x)
    n = len(y)
    if n == 0:
        raise ValueError("Cannot train on an empty client dataset")

    for _ in range(epochs):
        order = rng.permutation(n)
        for start in range(0, n, batch_size):
            idx = order[start : start + batch_size]
            x_batch = xb[idx]
            y_batch = y[idx]
            probs = sigmoid(x_batch @ weights)
            grad = (x_batch.T @ (probs - y_batch)) / len(idx)
            # Do not regularize the bias term.
            reg = np.r_[weights[:-1], 0.0]
            weights -= learning_rate * (grad + l2 * reg)

    return weights, float(binary_losses(weights, x, y).mean())
