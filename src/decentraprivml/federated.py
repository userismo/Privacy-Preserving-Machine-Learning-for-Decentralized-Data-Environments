"""Federated and peer-to-peer training algorithms."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from .data import ClientDataset
from .evaluation import evaluate_binary
from .model import train_logistic
from .privacy import gaussian_noise, l2_clip, secure_weighted_average_simulation


@dataclass
class TrainingResult:
    weights: np.ndarray
    history: list[dict[str, float | int | str]]
    wall_time_seconds: float
    model_communication_bytes: int
    mean_preclip_update_norm: float


def _rngs(seed: int, count: int) -> list[np.random.Generator]:
    return [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(count)]


def train_centralized(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    *,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    l2: float,
    seed: int,
) -> TrainingResult:
    rng = np.random.default_rng(seed)
    initial = np.zeros(x_train.shape[1] + 1, dtype=float)
    start = perf_counter()
    weights, train_loss = train_logistic(
        initial,
        x_train,
        y_train,
        epochs=epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        l2=l2,
        rng=rng,
    )
    elapsed = perf_counter() - start
    metrics = evaluate_binary(weights, x_test, y_test)
    return TrainingResult(
        weights=weights,
        history=[{"round": 1, "train_loss": train_loss, **metrics}],
        wall_time_seconds=elapsed,
        model_communication_bytes=0,
        mean_preclip_update_norm=0.0,
    )


def train_fedavg(
    clients: list[ClientDataset],
    x_test: np.ndarray,
    y_test: np.ndarray,
    *,
    rounds: int,
    local_epochs: int,
    learning_rate: float,
    batch_size: int,
    l2: float,
    client_fraction: float,
    clipping_norm: float | None,
    noise_multiplier: float,
    secure_aggregation: bool,
    seed: int,
    method_name: str,
) -> TrainingResult:
    if not 0 < client_fraction <= 1:
        raise ValueError("client_fraction must be in (0, 1]")
    if noise_multiplier > 0 and clipping_norm is None:
        raise ValueError("Noise requires clipping to bound sensitivity")

    selection_rng, local_rng, noise_rng, mask_rng = _rngs(seed, 4)
    weights = np.zeros(clients[0].x.shape[1] + 1, dtype=float)
    selected_count = max(1, int(np.ceil(client_fraction * len(clients))))
    parameter_bytes = weights.nbytes
    communication_bytes = 0
    history: list[dict[str, float | int | str]] = []
    observed_norms: list[float] = []
    start = perf_counter()

    for round_number in range(1, rounds + 1):
        selected_ids = selection_rng.choice(len(clients), size=selected_count, replace=False)
        updates: list[np.ndarray] = []
        counts: list[int] = []
        local_losses: list[float] = []

        for client_id in selected_ids:
            client = clients[int(client_id)]
            client_weights, local_loss = train_logistic(
                weights,
                client.x,
                client.y,
                epochs=local_epochs,
                learning_rate=learning_rate,
                batch_size=batch_size,
                l2=l2,
                rng=local_rng,
            )
            update = client_weights - weights
            if clipping_norm is not None:
                update, original_norm = l2_clip(update, clipping_norm)
            else:
                original_norm = float(np.linalg.norm(update))
            observed_norms.append(original_norm)
            updates.append(update)
            counts.append(len(client.y))
            local_losses.append(local_loss)

        aggregation_weights = np.asarray(counts, dtype=float)
        aggregation_weights /= aggregation_weights.sum()
        if secure_aggregation:
            aggregate_update = secure_weighted_average_simulation(
                updates, aggregation_weights, rng=mask_rng
            )
        else:
            aggregate_update = np.sum(
                [weight * update for weight, update in zip(aggregation_weights, updates)],
                axis=0,
            )

        if noise_multiplier > 0:
            # Central-DP-style Gaussian perturbation after clipping. This project
            # intentionally does not claim a formal epsilon without a full accountant.
            std = noise_multiplier * float(clipping_norm) / selected_count
            aggregate_update += gaussian_noise(
                aggregate_update.shape, standard_deviation=std, rng=noise_rng
            )

        weights += aggregate_update
        communication_bytes += selected_count * parameter_bytes * 2  # download + upload
        metrics = evaluate_binary(weights, x_test, y_test)
        history.append(
            {
                "method": method_name,
                "round": round_number,
                "train_loss": float(np.average(local_losses, weights=counts)),
                "selected_clients": selected_count,
                "mean_update_norm": float(np.mean(observed_norms[-selected_count:])),
                **metrics,
            }
        )

    return TrainingResult(
        weights=weights,
        history=history,
        wall_time_seconds=perf_counter() - start,
        model_communication_bytes=communication_bytes,
        mean_preclip_update_norm=float(np.mean(observed_norms)),
    )


def train_gossip(
    clients: list[ClientDataset],
    x_test: np.ndarray,
    y_test: np.ndarray,
    *,
    rounds: int,
    local_epochs: int,
    learning_rate: float,
    batch_size: int,
    l2: float,
    clipping_norm: float | None,
    local_noise_multiplier: float,
    seed: int,
    method_name: str,
) -> TrainingResult:
    """Train with synchronous ring gossip and no coordinating parameter server."""
    if local_noise_multiplier > 0 and clipping_norm is None:
        raise ValueError("Local noise requires clipping")
    local_rng, noise_rng = _rngs(seed, 2)
    num_clients = len(clients)
    models = [np.zeros(clients[0].x.shape[1] + 1, dtype=float) for _ in clients]
    parameter_bytes = models[0].nbytes
    history: list[dict[str, float | int | str]] = []
    observed_norms: list[float] = []
    communication_bytes = 0
    start = perf_counter()

    for round_number in range(1, rounds + 1):
        locally_trained: list[np.ndarray] = []
        local_losses: list[float] = []
        for client, current in zip(clients, models):
            candidate, local_loss = train_logistic(
                current,
                client.x,
                client.y,
                epochs=local_epochs,
                learning_rate=learning_rate,
                batch_size=batch_size,
                l2=l2,
                rng=local_rng,
            )
            update = candidate - current
            if clipping_norm is not None:
                update, original_norm = l2_clip(update, clipping_norm)
            else:
                original_norm = float(np.linalg.norm(update))
            observed_norms.append(original_norm)
            if local_noise_multiplier > 0:
                update += gaussian_noise(
                    update.shape,
                    standard_deviation=local_noise_multiplier * float(clipping_norm),
                    rng=noise_rng,
                )
            locally_trained.append(current + update)
            local_losses.append(local_loss)

        # Ring topology: mix each model with its left and right neighbours.
        models = [
            (locally_trained[(i - 1) % num_clients] + locally_trained[i] + locally_trained[(i + 1) % num_clients])
            / 3.0
            for i in range(num_clients)
        ]
        communication_bytes += num_clients * 2 * parameter_bytes
        readout_model = np.mean(models, axis=0)
        metrics = evaluate_binary(readout_model, x_test, y_test)
        history.append(
            {
                "method": method_name,
                "round": round_number,
                "train_loss": float(np.mean(local_losses)),
                "selected_clients": num_clients,
                "mean_update_norm": float(np.mean(observed_norms[-num_clients:])),
                **metrics,
            }
        )

    return TrainingResult(
        weights=np.mean(models, axis=0),
        history=history,
        wall_time_seconds=perf_counter() - start,
        model_communication_bytes=communication_bytes,
        mean_preclip_update_norm=float(np.mean(observed_norms)),
    )
