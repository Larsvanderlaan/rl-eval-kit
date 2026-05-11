---
title: Discounted Occupancy Ratios Benchmarks
description: Benchmark commands, profiles, and artifacts for occupancy-ratio workflows.
---

Run the smoke profile:

```bash
occupancy-ratio-benchmark \
  --profile smoke \
  --estimators boosted_tree neural_network \
  --no-google-dualdice \
  --no-plots
```

Controlled benchmark configs may include additional reference estimators for
method development. Keep the first smoke command focused on estimators users can
run on their own logged data.

## Profiles

| Profile | Purpose |
| --- | --- |
| `smoke` | Fast correctness and integration checks |
| `medium` | Larger controlled screens |
| `full` | Broader benchmark coverage |
| `overnight` | Gymnasium continuous-control stress settings |
| `high_stakes` | Conservative estimator comparison and longer budgets |
| `dualdice-paper` | Google DualDICE comparison runs |

## Artifacts

Occupancy runs write resumable partial outputs:

- `results.partial.csv`
- `tuning_results.partial.csv`
- `defaults_report.md`
- `benchmark_readout.md`
- manifest files with git and config-hash metadata
- plots for OPE error, ratio quality, ESS, runtime, source-correction checks,
  and skip or timeout rates

## Google DualDICE audit

```bash
PYTHONPATH=packages/occupancy-ratio .venv/bin/python -m occupancy_ratio_benchmark.run \
  --config packages/occupancy-ratio/occupancy_ratio_benchmark/configs/dualdice_smoke.json \
  --external-repo-path /tmp/google-research \
  --output-root outputs/occupancy_ratio_dualdice_audit
```

Missing optional dependencies should produce structured skip or error rows.
