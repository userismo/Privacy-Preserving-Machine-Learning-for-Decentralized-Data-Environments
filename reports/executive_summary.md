# Executive Summary

This repository evaluates privacy-preserving machine learning when data remains distributed across 10 non-IID clients. The best mean utility was **GossipAvg** with **94.83%** accuracy, while **DP-FedAvg (strong)** produced the lowest observed membership-inference AUC (**0.5402**).

The main engineering conclusion is that no single framework dominates every use case. Flower is the most flexible general-purpose option; TensorFlow Federated is strongest for TensorFlow-native algorithm research; OpenFL emphasizes cross-silo security controls; and PySyft emphasizes governed remote data access. Secure aggregation hides individual updates from an aggregator but does not stop leakage from the final model. Differential privacy can address model-level leakage, but it requires a formal accountant and carefully chosen sampling assumptions before any privacy guarantee can be claimed.
