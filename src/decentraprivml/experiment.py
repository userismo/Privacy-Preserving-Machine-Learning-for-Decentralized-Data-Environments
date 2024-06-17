"""Experiment orchestration, result aggregation and plot generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from .attacks import loss_threshold_membership_auc
from .data import dirichlet_partition, load_benchmark_data, partition_statistics
from .evaluation import evaluate_binary
from .federated import train_centralized, train_fedavg, train_gossip
from .frameworks import framework_comparison


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _final_row(
    *,
    method: str,
    seed: int,
    result: Any,
    data: Any,
    raw_data_bytes: int,
    privacy_mechanism: str,
    topology: str,
    secure_aggregation: bool,
) -> dict[str, Any]:
    metrics = evaluate_binary(result.weights, data.x_test, data.y_test)
    attack_auc = loss_threshold_membership_auc(
        result.weights,
        data.x_train,
        data.y_train,
        data.x_test,
        data.y_test,
        seed=seed + 100_000,
    )
    return {
        "method": method,
        "seed": seed,
        **metrics,
        "membership_attack_auc": attack_auc,
        "membership_advantage": max(0.0, 2.0 * attack_auc - 1.0),
        "wall_time_seconds": result.wall_time_seconds,
        "raw_data_bytes_shared": raw_data_bytes,
        "model_communication_bytes": result.model_communication_bytes,
        "total_estimated_bytes": raw_data_bytes + result.model_communication_bytes,
        "mean_preclip_update_norm": result.mean_preclip_update_norm,
        "privacy_mechanism": privacy_mechanism,
        "topology": topology,
        "secure_aggregation": secure_aggregation,
    }


def run_experiment_suite(config: dict[str, Any], output_dir: str | Path) -> dict[str, Path]:
    output = Path(output_dir)
    plot_dir = output / "plots"
    output.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    training = config["training"]
    privacy = config["privacy"]
    data_cfg = config["data"]
    all_rows: list[dict[str, Any]] = []
    histories: list[dict[str, Any]] = []
    partition_rows: list[dict[str, Any]] = []

    for seed in config["seeds"]:
        data = load_benchmark_data(
            test_size=data_cfg["test_size"],
            seed=seed,
            privacy_stress_noise_features=data_cfg.get("privacy_stress_noise_features", 0),
            noise_feature_scale=data_cfg.get("noise_feature_scale", 1.0),
        )
        clients = dirichlet_partition(
            data.x_train,
            data.y_train,
            num_clients=data_cfg["num_clients"],
            alpha=data_cfg["dirichlet_alpha"],
            min_client_size=data_cfg["min_client_size"],
            seed=seed + 17,
        )
        for stat in partition_statistics(clients):
            partition_rows.append({"seed": seed, **stat})

        raw_bytes = int(data.x_train.nbytes + data.y_train.nbytes)
        central = train_centralized(
            data.x_train,
            data.y_train,
            data.x_test,
            data.y_test,
            epochs=training["rounds"] * training["local_epochs"],
            learning_rate=training["learning_rate"],
            batch_size=training["batch_size"],
            l2=training["l2"],
            seed=seed,
        )
        all_rows.append(
            _final_row(
                method="Centralized",
                seed=seed,
                result=central,
                data=data,
                raw_data_bytes=raw_bytes,
                privacy_mechanism="None; raw training data pooled",
                topology="Central server with centralized data",
                secure_aggregation=False,
            )
        )
        histories.extend({"seed": seed, **row} for row in central.history)

        method_specs = [
            ("FedAvg", False, 0.0, "Data minimization only"),
            ("Secure FedAvg", True, 0.0, "Pairwise-mask secure aggregation simulation"),
            (
                "DP-FedAvg (moderate)",
                True,
                privacy["central_noise_moderate"],
                "Clipping + central Gaussian noise + secure aggregation simulation",
            ),
            (
                "DP-FedAvg (strong)",
                True,
                privacy["central_noise_strong"],
                "Stronger central Gaussian noise + secure aggregation simulation",
            ),
        ]
        for method, secure, noise, mechanism in method_specs:
            fed_result = train_fedavg(
                clients,
                data.x_test,
                data.y_test,
                rounds=training["rounds"],
                local_epochs=training["local_epochs"],
                learning_rate=training["learning_rate"],
                batch_size=training["batch_size"],
                l2=training["l2"],
                client_fraction=training["client_fraction"],
                clipping_norm=privacy["clipping_norm"] if noise > 0 else None,
                noise_multiplier=noise,
                secure_aggregation=secure,
                seed=seed,
                method_name=method,
            )
            all_rows.append(
                _final_row(
                    method=method,
                    seed=seed,
                    result=fed_result,
                    data=data,
                    raw_data_bytes=0,
                    privacy_mechanism=mechanism,
                    topology="Coordinator-based federated learning",
                    secure_aggregation=secure,
                )
            )
            histories.extend({"seed": seed, **row} for row in fed_result.history)

        gossip_specs = [
            ("GossipAvg", 0.0, "Peer-to-peer data minimization"),
            (
                "Local-DP Gossip",
                privacy["local_noise"],
                "Client update clipping + local Gaussian noise",
            ),
        ]
        for method, noise, mechanism in gossip_specs:
            gossip_result = train_gossip(
                clients,
                data.x_test,
                data.y_test,
                rounds=training["rounds"],
                local_epochs=training["local_epochs"],
                learning_rate=training["learning_rate"],
                batch_size=training["batch_size"],
                l2=training["l2"],
                clipping_norm=privacy["clipping_norm"] if noise > 0 else None,
                local_noise_multiplier=noise,
                seed=seed,
                method_name=method,
            )
            all_rows.append(
                _final_row(
                    method=method,
                    seed=seed,
                    result=gossip_result,
                    data=data,
                    raw_data_bytes=0,
                    privacy_mechanism=mechanism,
                    topology="Serverless ring gossip",
                    secure_aggregation=False,
                )
            )
            histories.extend({"seed": seed, **row} for row in gossip_result.history)

    runs = pd.DataFrame(all_rows)
    history = pd.DataFrame(histories)
    partitions = pd.DataFrame(partition_rows)
    numeric_metrics = [
        "accuracy",
        "f1",
        "roc_auc",
        "log_loss",
        "membership_attack_auc",
        "membership_advantage",
        "wall_time_seconds",
        "raw_data_bytes_shared",
        "model_communication_bytes",
        "total_estimated_bytes",
        "mean_preclip_update_norm",
    ]
    summary = runs.groupby("method", sort=False)[numeric_metrics].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary = summary.reset_index()

    runs_path = output / "all_runs.csv"
    summary_path = output / "benchmark_summary.csv"
    history_path = output / "round_history.csv"
    partition_path = output / "client_partitions.csv"
    framework_path = output / "framework_comparison.csv"
    config_path = output / "config_used.yaml"
    metadata_path = output / "run_metadata.json"
    runs.to_csv(runs_path, index=False)
    summary.to_csv(summary_path, index=False)
    history.to_csv(history_path, index=False)
    partitions.to_csv(partition_path, index=False)
    framework_comparison().to_csv(framework_path, index=False)
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    metadata = {
        "dataset": "scikit-learn breast cancer diagnostic dataset with configured nuisance features",
        "num_seeds": len(config["seeds"]),
        "methods": list(runs["method"].unique()),
        "formal_dp_guarantee_claimed": False,
        "secure_aggregation_is_cryptographic": False,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    _generate_plots(runs, history, plot_dir)
    return {
        "runs": runs_path,
        "summary": summary_path,
        "history": history_path,
        "partitions": partition_path,
        "frameworks": framework_path,
        "config": config_path,
        "metadata": metadata_path,
    }


def _generate_plots(runs: pd.DataFrame, history: pd.DataFrame, plot_dir: Path) -> None:
    method_order = list(runs["method"].drop_duplicates())
    grouped = runs.groupby("method", sort=False)

    means = grouped["accuracy"].mean().reindex(method_order)
    errors = grouped["accuracy"].std().reindex(method_order).fillna(0)
    plt.figure(figsize=(11, 6))
    plt.bar(method_order, means, yerr=errors, capsize=4)
    plt.ylim(max(0.5, float(means.min() - 0.1)), 1.0)
    plt.ylabel("Test accuracy")
    plt.xticks(rotation=30, ha="right")
    plt.title("Utility across decentralized privacy configurations")
    plt.tight_layout()
    plt.savefig(plot_dir / "accuracy_comparison.png", dpi=180)
    plt.close()

    attack_means = grouped["membership_attack_auc"].mean().reindex(method_order)
    attack_errors = grouped["membership_attack_auc"].std().reindex(method_order).fillna(0)
    plt.figure(figsize=(11, 6))
    plt.bar(method_order, attack_means, yerr=attack_errors, capsize=4)
    plt.axhline(0.5, linestyle="--", linewidth=1, label="Chance level")
    plt.ylim(0.45, max(0.75, float(attack_means.max() + 0.08)))
    plt.ylabel("Membership-inference AUC")
    plt.xticks(rotation=30, ha="right")
    plt.title("Empirical privacy leakage (lower is better)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_dir / "membership_attack_auc.png", dpi=180)
    plt.close()

    comm = grouped["total_estimated_bytes"].mean().reindex(method_order)
    plt.figure(figsize=(9, 6))
    plt.scatter(comm / 1024.0, means, s=80)
    for method in method_order:
        plt.annotate(method, (comm[method] / 1024.0, means[method]), xytext=(5, 5), textcoords="offset points")
    plt.xlabel("Estimated communication (KiB)")
    plt.ylabel("Mean test accuracy")
    plt.title("Communication–utility trade-off")
    plt.tight_layout()
    plt.savefig(plot_dir / "communication_vs_accuracy.png", dpi=180)
    plt.close()

    selected = ["FedAvg", "DP-FedAvg (moderate)", "DP-FedAvg (strong)", "GossipAvg", "Local-DP Gossip"]
    curve = (
        history[history["method"].isin(selected)]
        .groupby(["method", "round"], as_index=False)["accuracy"]
        .mean()
    )
    plt.figure(figsize=(10, 6))
    for method in selected:
        part = curve[curve["method"] == method]
        plt.plot(part["round"], part["accuracy"], label=method)
    plt.xlabel("Communication round")
    plt.ylabel("Mean test accuracy")
    plt.title("Convergence under non-IID client partitions")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_dir / "convergence.png", dpi=180)
    plt.close()
