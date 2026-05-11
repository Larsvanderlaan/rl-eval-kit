---
title: FQE Benchmarks
description: Benchmark commands and artifacts for FQE workflows.
---

Run the smoke benchmark from the repository root after installing the package:

```bash
fqe-benchmark --stage smoke --no-plots
```

Common options:

```bash
fqe-benchmark \
  --stage smoke \
  --automl-tuning fast \
  --stationary-gamma-ratio 0.99 \
  --output-root outputs/fqe_benchmark \
  --no-plots
```

## Stages

| Stage | Purpose |
| --- | --- |
| `smoke` | Tiny correctness and integration checks |
| `core` | Controlled synthetic settings with known reference values |
| `full` | Adds larger optional workloads when required artifacts exist |

## Outputs

Benchmark outputs are written under `<output-root>/<stage>/`:

- `results.csv`
- `summary.csv`
- `diagnostics.json`
- `manifest.json`
- `tuning_results.csv`
- `value_error.png`, `q_mse.png`, and `runtime.png` when plotting is enabled

Benchmark reference values are for interpreting benchmark tables. User-facing
tuning reports should focus on the losses, validation data, and diagnostics
available in the run.
