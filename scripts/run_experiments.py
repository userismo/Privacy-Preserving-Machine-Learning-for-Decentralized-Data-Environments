#!/usr/bin/env python3
"""Convenience wrapper for running the complete benchmark."""

from pathlib import Path

from decentraprivml.experiment import load_config, run_experiment_suite


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    outputs = run_experiment_suite(
        load_config(project_root / "configs" / "default.yaml"),
        project_root / "results",
    )
    for key, value in outputs.items():
        print(f"{key}: {value}")
