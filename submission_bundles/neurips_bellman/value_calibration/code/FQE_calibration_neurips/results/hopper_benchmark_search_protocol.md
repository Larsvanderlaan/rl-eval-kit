# Hopper Benchmark Search Protocol

This benchmark is a practical external-validity check for Bellman calibration.
It should only enter the main paper if calibrated neural FQE improves both held-out
Bellman calibration error and scalar OPE error against official Deep OPE returns.

## Stages

Run from the repository root. The pilot is intentionally a screen: it uses three
representative Hopper target-policy snapshots and three seeds with a small
neural-FQE training budget. Promising rows must be expanded to all target-policy
snapshots with `--stage pilot_all_policies` and then rerun with `--stage final`.

```bash
./.venv/bin/python FQE_calibration_neurips/scripts/run_hopper_calibration_benchmark.py --stage smoke
./.venv/bin/python FQE_calibration_neurips/scripts/run_hopper_calibration_benchmark.py --stage pilot
```

Inspect:

```bash
sed -n '1,200p' FQE_calibration_neurips/results/hopper_calibration_pilot/hopper_calibration_readout.md
cat FQE_calibration_neurips/results/hopper_calibration_pilot/hopper_calibration_audit.csv
```

If the representative-policy pilot is promising, run all Hopper-medium target
policies:

```bash
./.venv/bin/python FQE_calibration_neurips/scripts/run_hopper_calibration_benchmark.py --stage pilot_all_policies
```

If Hopper-medium is mixed or negative, run the local/available-dataset expansion:

```bash
./.venv/bin/python FQE_calibration_neurips/scripts/run_hopper_calibration_benchmark.py --stage expansion
```

Only run final after a pilot or expansion row passes the audit gates:

```bash
./.venv/bin/python FQE_calibration_neurips/scripts/run_hopper_calibration_benchmark.py --stage final
```

## Promotion Gates

A calibrated row is promotable only if all of the following hold:

- raw neural FQE has positive policy-level Pearson and Spearman correlation with official returns;
- raw neural FQE has nonzero held-out Bellman calibration error;
- relative scalar OPE absolute error is below `0.90`;
- relative Bellman calibration error is below `0.85`;
- both OPE-error and Bellman-calibration win rates are at least `0.60`;
- no rows are diagnostic-only and all metrics are finite;
- at least three target policies and at least two seeds are present.

Rows that fail these gates can still be reported as appendix limitations, but
should not support a main-text benchmark claim.

## Fallback Search

If no Hopper row passes, prioritize benchmarks with official target-policy
snapshots and return labels:

1. Deep OPE / RL Unplugged policy-snapshot tasks.
2. D4RL or Minari MuJoCo HalfCheetah and Walker2d OPE tasks.
3. Newer validation-style benchmarks only if they provide target-policy return
   labels and feasible offline data access.

Useful benchmark sources:

- Deep OPE policies: https://github.com/google-research/deep_ope
- D4RL OPE tasks: https://github.com/Farama-Foundation/d4rl/wiki/Off-Policy-Evaluation
- D4RL Hopper TFDS variants: https://www.tensorflow.org/datasets/catalog/d4rl_mujoco_hopper
- RL Unplugged: https://papers.nips.cc/paper/2020/hash/51200d29d1fc15f5a71c1dab4bb54f7c-Abstract.html

The repository also includes an offline screen over the official Deep OPE FQE-L2
prediction tables:

```bash
./.venv/bin/python FQE_calibration_neurips/scripts/screen_deep_ope_value_calibration.py
```

This screen is useful for choosing a next real transition-level benchmark, but
it is not itself a Bellman-transition calibration experiment because it calibrates
policy-value predictions against held-out policy-return labels.
