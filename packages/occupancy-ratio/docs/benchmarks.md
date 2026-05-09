# Benchmarks

Install benchmark dependencies:

```bash
python -m pip install -e "packages/occupancy-ratio[benchmark]"
```

Run a smoke benchmark:

```bash
occupancy-ratio-benchmark \
  --profile smoke \
  --estimators oracle boosted_tree neural_network \
  --no-google-dualdice \
  --no-plots
```

## AutoML Benchmarking

Benchmark tuning uses the same product harness as `tune_occupancy_ratio_auto`.

```bash
occupancy-ratio-benchmark \
  --profile smoke \
  --estimators oracle boosted_tree \
  --boosted-estimator-presets stable \
  --tune-cv \
  --automl-tuning balanced \
  --no-google-dualdice \
  --no-plots
```

`--automl-tuning fast` runs a smaller sweep. `--tune-cv` maps to balanced
AutoML unless an explicit mode is supplied.

## Profiles

The benchmark runner supports smoke, medium, full, overnight, high-stakes, and
DualDICE-oriented profiles. JSON configs in
`occupancy_ratio_benchmark/configs/` define reproducible production audits.

## Output

Result rows include OPE error, ratio-quality diagnostics when truth is
available, ESS/tail/clipping metrics, nuisance diagnostics, fixed-point history
summaries, selected tuning candidates, and runtime.

Treat oracle ratios and target-policy Monte Carlo values as reporting fields
only. They must not enter model selection.
