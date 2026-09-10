# Result Artifacts

- `all_runs.csv`: one row per method and seed.
- `benchmark_summary.csv`: mean and standard deviation across five seeds.
- `round_history.csv`: per-round convergence metrics.
- `client_partitions.csv`: client sample counts and class ratios.
- `framework_comparison.csv`: framework-selection matrix.
- `config_used.yaml`: exact experiment configuration.
- `run_metadata.json`: assumptions and non-claims.
- `plots/`: generated publication-style figures.

Regenerate everything with:

```bash
python -m decentraprivml.cli run --config configs/default.yaml --output results
python scripts/generate_report.py
```
