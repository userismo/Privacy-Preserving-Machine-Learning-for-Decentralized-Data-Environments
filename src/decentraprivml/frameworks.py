"""Curated framework comparison used in the generated report."""

from __future__ import annotations

import pandas as pd


def framework_comparison() -> pd.DataFrame:
    rows = [
        {
            "framework": "Flower",
            "primary_focus": "Framework-agnostic federated learning research and deployment",
            "model_ecosystem": "PyTorch, TensorFlow, JAX, scikit-learn and others",
            "simulation": "Scalable simulation runtime",
            "differential_privacy": "Built-in central and local DP mechanisms (preview)",
            "secure_aggregation": "Built-in SecAgg and SecAgg+",
            "governance": "Application and federation configuration",
            "best_fit": "Fast prototyping that can grow into multi-node deployment",
        },
        {
            "framework": "TensorFlow Federated",
            "primary_focus": "Research-oriented federated computations in TensorFlow",
            "model_ecosystem": "TensorFlow / Keras",
            "simulation": "Federated simulation datasets and runtimes",
            "differential_privacy": "DP aggregators and TensorFlow Privacy integration",
            "secure_aggregation": "Secure-sum aggregators",
            "governance": "Algorithmic composition rather than data-access governance",
            "best_fit": "Custom FL algorithms and rigorous TensorFlow experiments",
        },
        {
            "framework": "OpenFL",
            "primary_focus": "Cross-silo federated learning for organizations",
            "model_ecosystem": "Backend-agnostic, including PyTorch and TensorFlow",
            "simulation": "Local and federated runtimes",
            "differential_privacy": "Opacus-based options and privacy reporting",
            "secure_aggregation": "TaskRunner and Workflow API support",
            "governance": "PKI, mTLS, collaborator authorization and plans",
            "best_fit": "Institutional collaborations with explicit infrastructure security",
        },
        {
            "framework": "PySyft",
            "primary_focus": "Governed remote data science on private data assets",
            "model_ecosystem": "Python code execution over governed assets",
            "simulation": "Datasite-oriented development with mock data",
            "differential_privacy": "Policy and release workflows; mechanism depends on study",
            "secure_aggregation": "Not its sole design center",
            "governance": "Strong request, approval, dataset and result-release workflows",
            "best_fit": "Data-access governance where researchers must not see raw data",
        },
        {
            "framework": "DecentraPrivML-Bench",
            "primary_focus": "Transparent educational benchmark and framework selection",
            "model_ecosystem": "NumPy / scikit-learn reference model",
            "simulation": "Single-machine non-IID client simulation",
            "differential_privacy": "Clipping plus Gaussian-noise experiments; no epsilon claim",
            "secure_aggregation": "Algebraic pairwise-mask simulation",
            "governance": "Documented threat model; not a deployment control plane",
            "best_fit": "Reproducible comparison, teaching and pre-framework experimentation",
        },
    ]
    return pd.DataFrame(rows)
