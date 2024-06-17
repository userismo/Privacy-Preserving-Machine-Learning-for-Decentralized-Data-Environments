"""Dataset loading and non-IID client partitioning."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class DataBundle:
    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]


@dataclass(frozen=True)
class ClientDataset:
    client_id: int
    x: np.ndarray
    y: np.ndarray


def load_benchmark_data(
    *,
    test_size: float,
    seed: int,
    privacy_stress_noise_features: int = 0,
    noise_feature_scale: float = 1.0,
) -> DataBundle:
    """Load the breast-cancer dataset and optionally add nuisance features.

    Independent Gaussian nuisance features form a controlled privacy stress test:
    they carry no label signal but give an over-parameterized model more capacity
    to memorize idiosyncrasies of the training set.
    """
    dataset = load_breast_cancer()
    x_train, x_test, y_train, y_test = train_test_split(
        dataset.data.astype(float),
        dataset.target.astype(int),
        test_size=test_size,
        random_state=seed,
        stratify=dataset.target,
    )
    scaler = StandardScaler().fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    feature_names = list(dataset.feature_names)
    if privacy_stress_noise_features < 0:
        raise ValueError("privacy_stress_noise_features cannot be negative")
    if privacy_stress_noise_features:
        noise_rng = np.random.default_rng(seed + 999)
        x_train = np.hstack(
            [
                x_train,
                noise_rng.normal(
                    0.0, noise_feature_scale, size=(len(y_train), privacy_stress_noise_features)
                ),
            ]
        )
        x_test = np.hstack(
            [
                x_test,
                noise_rng.normal(
                    0.0, noise_feature_scale, size=(len(y_test), privacy_stress_noise_features)
                ),
            ]
        )
        feature_names.extend(
            [f"privacy_stress_noise_{index}" for index in range(privacy_stress_noise_features)]
        )

    return DataBundle(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        feature_names=feature_names,
    )


def dirichlet_partition(
    x: np.ndarray,
    y: np.ndarray,
    *,
    num_clients: int,
    alpha: float,
    min_client_size: int,
    seed: int,
    max_attempts: int = 500,
) -> list[ClientDataset]:
    """Create label-skewed non-IID partitions using a Dirichlet distribution."""
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    if num_clients * min_client_size > len(y):
        raise ValueError("min_client_size is infeasible for this dataset")

    rng = np.random.default_rng(seed)
    classes = np.unique(y)

    for _ in range(max_attempts):
        client_indices: list[list[int]] = [[] for _ in range(num_clients)]
        for cls in classes:
            cls_indices = np.flatnonzero(y == cls)
            rng.shuffle(cls_indices)
            proportions = rng.dirichlet(np.full(num_clients, alpha))
            split_points = (np.cumsum(proportions)[:-1] * len(cls_indices)).astype(int)
            chunks = np.split(cls_indices, split_points)
            for client_id, chunk in enumerate(chunks):
                client_indices[client_id].extend(chunk.tolist())

        sizes = [len(indices) for indices in client_indices]
        if min(sizes) >= min_client_size:
            clients = []
            for client_id, indices in enumerate(client_indices):
                idx = np.asarray(indices, dtype=int)
                rng.shuffle(idx)
                clients.append(ClientDataset(client_id, x[idx], y[idx]))
            return clients

    raise RuntimeError(
        "Unable to create a valid Dirichlet partition; increase alpha or reduce min_client_size"
    )


def partition_statistics(clients: list[ClientDataset]) -> list[dict[str, float | int]]:
    """Return client sizes and positive-class ratios for reporting."""
    return [
        {
            "client_id": client.client_id,
            "samples": len(client.y),
            "positive_ratio": float(client.y.mean()),
        }
        for client in clients
    ]
