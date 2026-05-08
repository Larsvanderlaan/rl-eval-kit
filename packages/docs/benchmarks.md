# Benchmark Guide

## FQE

Run the FQE benchmark with:

```bash
fqe-benchmark --stage smoke --no-plots
```

Stages are `smoke`, `core`, and `full`. The benchmark writes `results.csv`,
`summary.csv`, `diagnostics.json`, and `manifest.json` under
`<output-root>/<stage>/`.

## Occupancy Ratio

Run the occupancy benchmark with:

```bash
occupancy-ratio-benchmark \
  --profile smoke \
  --estimators oracle boosted_tree neural_network \
  --no-google-dualdice \
  --no-plots
```

Profiles are `smoke`, `medium`, `full`, `overnight`, and `dualdice-paper`.
Occupancy runs write `results.partial.csv` and `tuning_results.partial.csv`
after each estimator and resume from those files by default. Use `--no-resume`
for a clean rerun. Per-estimator timeouts default to 120 seconds for smoke and
600 seconds for larger profiles.

The `overnight` profile adds Gymnasium continuous-control settings for
realistic OPE stress tests. Controlled settings keep oracle ratio diagnostics;
Gymnasium settings use Monte Carlo target-policy values and leave ratio-truth
columns blank. After a run, summarize default candidates with:

```bash
occupancy-ratio-defaults-report /tmp/occupancy_default_overnight/overnight/results.csv
```

Google DualDICE comparisons require a local Google Research checkout and its
TensorFlow dependencies. Missing optional dependencies produce structured skip
or error rows instead of stopping smoke runs.
