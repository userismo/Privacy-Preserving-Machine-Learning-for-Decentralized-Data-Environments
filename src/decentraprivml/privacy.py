"""Privacy mechanisms used in the controlled benchmark."""

from __future__ import annotations

import numpy as np


def l2_clip(update: np.ndarray, clipping_norm: float) -> tuple[np.ndarray, float]:
    """Clip a vector to a maximum L2 norm and return its original norm."""
    if clipping_norm <= 0:
        raise ValueError("clipping_norm must be positive")
    norm = float(np.linalg.norm(update))
    if norm <= clipping_norm or norm == 0.0:
        return update.copy(), norm
    return update * (clipping_norm / norm), norm


def gaussian_noise(
    shape: tuple[int, ...], *, standard_deviation: float, rng: np.random.Generator
) -> np.ndarray:
    """Generate Gaussian noise, returning zeros when the scale is disabled."""
    if standard_deviation < 0:
        raise ValueError("standard_deviation cannot be negative")
    if standard_deviation == 0:
        return np.zeros(shape, dtype=float)
    return rng.normal(0.0, standard_deviation, size=shape)


def secure_weighted_average_simulation(
    updates: list[np.ndarray],
    weights: np.ndarray,
    *,
    rng: np.random.Generator,
) -> np.ndarray:
    """Simulate pairwise masks that cancel in a secure weighted sum.

    This validates the algebra of secure aggregation but is not a production
    cryptographic protocol. Every pair receives an opposite random mask; only
    the final sum is unmasked.
    """
    if len(updates) == 0:
        raise ValueError("At least one update is required")
    if len(updates) != len(weights):
        raise ValueError("updates and weights must have equal length")
    if not np.isclose(weights.sum(), 1.0):
        raise ValueError("weights must sum to one")

    masked = [weight * update.copy() for update, weight in zip(updates, weights)]
    for left in range(len(masked)):
        for right in range(left + 1, len(masked)):
            mask = rng.normal(0.0, 3.0, size=masked[left].shape)
            masked[left] += mask
            masked[right] -= mask
    return np.sum(masked, axis=0)
