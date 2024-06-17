"""Command-line entry point."""

from __future__ import annotations

import argparse
from pathlib import Path

from .experiment import load_config, run_experiment_suite


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DecentraPrivML-Bench experiments")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run", help="Run all benchmark methods")
    run_parser.add_argument("--config", default="configs/default.yaml")
    run_parser.add_argument("--output", default="results")
    args = parser.parse_args()

    if args.command == "run":
        paths = run_experiment_suite(load_config(args.config), Path(args.output))
        print("Experiment complete")
        for name, path in paths.items():
            print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
