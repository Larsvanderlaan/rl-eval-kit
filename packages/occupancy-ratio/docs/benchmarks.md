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

## ICLR Review Experiments

The ICLR-facing configs separate deployable defaults, non-oracle tuning, and
oracle diagnostics:

```bash
occupancy-ratio-benchmark \
  --config occupancy_ratio_benchmark/configs/iclr_core.json

occupancy-ratio-benchmark \
  --config occupancy_ratio_benchmark/configs/iclr_dualdice_fairness.json \
  --external-repo-path /tmp/google-research

occupancy-ratio-benchmark \
  --config occupancy_ratio_benchmark/configs/iclr_gym.json \
  --external-repo-path /tmp/google-research
```

`neural_fori_default_stable` is the predeclared FORI neural baseline,
`neural_fori_cv_size` uses the built-in non-oracle Bellman-GMM tuning path, and
`neural_fori_oracle` is a diagnostic upper envelope that uses ratio truth only
when available. `google_dualdice_default` runs the official Google Research
DualDICE implementation with fixed defaults, while `dualdice_gmm_tuned` and
`dualdice_oracle` use the same DualDICE objective over a fixed candidate grid.
Oracle-tuned estimators are excluded from deployable winner tables.

## Output

Result rows include OPE error, ratio-quality diagnostics when truth is
available, ESS/tail/clipping metrics, nuisance diagnostics, fixed-point history
summaries, selected tuning candidates, and runtime.

Treat oracle ratios and target-policy Monte Carlo values as reporting fields
only. They must not enter model selection.
