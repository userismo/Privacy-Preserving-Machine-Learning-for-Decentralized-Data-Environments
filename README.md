# DecentraPrivML-Bench

A complete, reproducible research project for **investigating privacy-preserving machine-learning frameworks in decentralized data environments**.

The repository combines two layers:

1. **A framework study** comparing Flower, TensorFlow Federated, OpenFL, and PySyft.
2. **An executable benchmark** comparing centralized learning, Federated Averaging, secure-aggregation simulation, differentially private FedAvg, serverless gossip learning, and locally perturbed gossip learning.

## Why this project matters

Keeping records on client devices or inside organizations reduces raw-data exposure, but federated learning alone is not a complete privacy guarantee. Model updates and final models can still leak information. This benchmark therefore measures both predictive utility and empirical membership-inference leakage.

## Included deliverables

- Fully runnable Python package and CLI
- Non-IID decentralized-data simulator
- Centralized, FedAvg, secure FedAvg, DP-FedAvg, GossipAvg, and Local-DP Gossip implementations
- Pairwise-mask secure-aggregation simulation
- Loss-threshold membership-inference attack
- Five-seed reproducible experiments
- CSV results, round histories, configuration, and metadata
- Publication-style plots
- Final research report and executive summary
- Framework comparison matrix
- Unit tests, Docker support, and GitHub Actions CI
- Jupyter analysis notebook


## Included benchmark results

The repository already contains a completed five-seed run. Lower membership-inference AUC is better; `0.5` is chance-level.

| Method | Accuracy | Membership attack AUC | Estimated communication (KiB) |
|---|---:|---:|---:|
| Centralized | 0.9399 ± 0.0189 | 0.5529 ± 0.0344 | 935.2 |
| FedAvg | 0.9427 ± 0.0229 | 0.5540 ± 0.0328 | 922.0 |
| Secure FedAvg | 0.9427 ± 0.0229 | 0.5540 ± 0.0328 | 922.0 |
| DP-FedAvg (moderate) | 0.9231 ± 0.0204 | 0.5519 ± 0.0238 | 922.0 |
| DP-FedAvg (strong) | 0.8364 ± 0.0355 | 0.5402 ± 0.0335 | 922.0 |
| GossipAvg | 0.9483 ± 0.0235 | 0.5480 ± 0.0335 | 1317.2 |
| Local-DP Gossip | 0.9371 ± 0.0164 | 0.5435 ± 0.0233 | 1317.2 |

The results are a controlled benchmark, not a formal proof that one mechanism is private. In particular, the stronger DP-style setting reduces the measured attack AUC but causes a substantial utility loss.

## Experimental design

The built-in breast-cancer diagnostic dataset is split into training and test data and augmented with 250 independent nuisance features. These features contain no label signal but create a controlled over-parameterized privacy stress test. The training split is distributed across 10 clients using a Dirichlet label distribution (`alpha = 0.45`) to create non-IID shards. Each configuration runs for 30 communication rounds and five random seeds.

| Method | Raw data leaves clients? | Coordinator? | Update confidentiality | Noise location |
|---|---:|---:|---|---|
| Centralized | Yes | Yes | Not applicable | None |
| FedAvg | No | Yes | No | None |
| Secure FedAvg | No | Yes | Simulated aggregate-only visibility | None |
| DP-FedAvg | No | Yes | Simulated secure aggregation | Aggregate |
| GossipAvg | No | No | Neighbours see shared models | None |
| Local-DP Gossip | No | No | Perturbed before sharing | Client |

## Metrics

- **Accuracy, F1, ROC-AUC, log loss:** predictive utility.
- **Membership-inference AUC:** ability to distinguish training members from held-out records. A value near `0.5` is chance-level.
- **Communication bytes:** approximate raw-data or model-vector traffic.
- **Wall time and convergence:** computational behavior.

## Important privacy and security boundaries

This is a research simulator, not a production privacy system.

- The secure-aggregation implementation demonstrates pairwise-mask cancellation in one process. It is not an authenticated cryptographic protocol.
- The DP experiments use clipping and Gaussian noise but do not report a formal epsilon. A valid privacy claim requires a neighboring relation, sampling assumptions, an accountant, a delta choice, and end-to-end enforcement.
- Real deployments should use audited framework mechanisms, authenticated transport, hardened identity, secure aggregation, privacy accounting, access control, and security review.

## Using the results

Start with:

- `reports/final_report.md` for the full study
- `reports/executive_summary.md` for the conclusion
- `results/benchmark_summary.csv` for aggregate metrics
- `notebooks/results_analysis.ipynb` for interactive analysis

## Framework choice in one paragraph

Flower is a strong general-purpose choice for framework-agnostic FL experiments and deployment. TensorFlow Federated is suited to TensorFlow-native algorithm research. OpenFL is oriented toward cross-silo collaborations with infrastructure security controls. PySyft focuses on governed remote data science, code approval, and controlled result release. The included reference simulator is useful before committing to one of these heavier platforms.
LICENSE`.
