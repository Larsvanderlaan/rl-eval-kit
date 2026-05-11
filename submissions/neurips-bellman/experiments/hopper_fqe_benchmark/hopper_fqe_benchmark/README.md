# Hopper Medium Benchmark

This folder contains an isolated Hopper benchmark study aimed at reproducing the
Deep OPE `hopper-medium` setting more closely than the earlier provisional
ONNX/v2 harness.

The study now uses:

- the official Berkeley `hopper_medium.hdf5` dataset (`hopper-medium-v0`);
- the 11 official Deep OPE Hopper policy snapshots (`hopper_online_*.pkl`);
- the official Deep OPE benchmark files in `benchmark.zip`:
  - `d4rl_gt.pkl`
  - `d4rl_policys.pkl`
  - `d4rl_fqel2.pkl`
  - `d4rl_dice.pkl`

## Methods

- `standard_fqe`: official TensorFlow `policy_eval` FQE critic run on the local Hopper HDF5 data
- `dual_dice`: official TensorFlow `policy_eval` DualDICE run on the local Hopper HDF5 data
- `weighted_dual_dice`: stationary-weighted FQE using DualDICE-derived weights with the same official FQE critic
- `weighted_linear`: stationary-weighted FQE with closed-form linear ratios

Optional ablations still available from the CLI:

- `weighted_neural_saddle`: stationary-weighted FQE using the neural saddle-point ratio estimator from `FQE_neurips`
- `weighted_rkhs`: stationary-weighted FQE with RKHS-based ratio estimation

The runner also writes the official `FQE-L2` and `DICE` benchmark predictions as
reference rows so local results can be compared side by side.

## Training Budget

The official `policy_eval` code we are reproducing uses:

- batch size `256`
- `num_updates = 1_000_000`

This appears both in the flag defaults in [`/tmp/google-research/policy_eval/train_eval.py`](/tmp/google-research/policy_eval/train_eval.py) and in the Deep OPE example script [`/tmp/deep_ope/run_fqe_l2_example.sh`](/tmp/deep_ope/run_fqe_l2_example.sh).

For that reason, this benchmark now supports budget presets:

- `--budget smoke`: `2,000` updates
- `--budget pilot`: `100,000` updates
- `--budget paper`: `1,000,000` updates

The earlier tiny runs were only smoke tests and are not paper-close.

## Layout

- `data.py`: loads and processes `hopper-medium-v0` in the same trajectory style as the official code
- `policies.py`: loads the official Deep OPE Hopper pickle policies
- `official_tf_baselines.py`: bridge that runs the official TensorFlow `QFitter` and `DualDICE` classes on the local Hopper data
- `fqe.py`: legacy local PyTorch FQE module kept for reference
- `dice.py`: legacy local PyTorch DualDICE module kept for reference
- `features.py`: feature map for ratio estimation
- `runner.py`: benchmark sweep, metrics, and plotting
- `study.py`: CLI entrypoint

## Run

From the repo root:

```bash
python -m hopper_fqe_benchmark.study \
  --budget paper \
  --seeds 0 1 2 \
  --methods standard_fqe weighted_dual_dice weighted_linear
```

For a quicker smoke test:

```bash
python -m hopper_fqe_benchmark.study \
  --budget smoke \
  --seeds 0 \
  --max-trajectories 128 \
  --methods standard_fqe weighted_dual_dice weighted_linear
```

## Outputs

The study writes into `hopper_fqe_benchmark/outputs`:

- `hopper_medium_results.csv`: per-policy predictions
- `hopper_medium_metrics.csv`: per-seed benchmark metrics
- `hopper_medium_summary.csv`: aggregate metric summary
- `hopper_medium_metrics.png`: comparison plot
- `hopper_medium_config.json`: exact run config

## Notes

- The benchmark is isolated from `FQE_neurips`; it imports weighting code from there but does not modify that folder.
- The baseline reproduction path now uses the official TensorFlow `policy_eval` classes from [`/tmp/google-research/policy_eval`](/tmp/google-research/policy_eval) rather than the earlier local PyTorch ports.
- The policy ids follow the benchmark naming convention (`hopper-medium_00` ... `hopper-medium_10`).
- The benchmark ground truth and official reference baselines come from the public Deep OPE `benchmark.zip` bundle.
