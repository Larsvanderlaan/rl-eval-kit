# RL Unplugged `cheetah_run` Bellman Calibration Benchmark

This benchmark implements a strict transition-level Bellman calibration check on
RL Unplugged Control Suite `cheetah_run` using official Deep OPE target-policy
snapshots and official policy-return labels.

## Protocol

- Data: `tensorflow_datasets` loader for `rlu_control_suite/cheetah_run`.
- Splits: episodes are split into disjoint train, calibration, and diagnostic
  sets. Calibration never uses diagnostic transitions or official return labels.
- Policies: official Deep OPE RL Unplugged SavedModel policies from
  `rlunplugged_policys.pkl`; OPE error is evaluated against
  `rlunplugged_gt.pkl`.
- Learners:
  - linear FQE with standardized linear/quadratic state-action features;
  - RF FQE with standardized random Fourier state-action features;
  - neural FQE with a two-layer MLP, AdamW, gradient clipping, and hard target
    updates at each fitted Bellman iteration;
  - RKHS minimax Bellman-error estimator using RF value features and an RBF
    critic/moment class.
- Calibrators: none, linear, isotonic.
- Calibration target: held-out one-step Bellman outcome
  `R + gamma * V_hat(S')` under the same target policy.
- Promotion gates: positive raw policy-rank correlation, relative Bellman
  calibration error below `0.85`, relative scalar OPE absolute error below
  `0.90`, win rates at least `0.60`, adequate coverage ESS, and no leakage.

## Reproduction Commands

Smoke:

```bash
./.venv/bin/python FQE_calibration_neurips/scripts/run_rlu_cheetah_benchmark.py \
  --stage smoke \
  --learners linear_fqe rf_fqe rkhs_minimax_fqe \
  --fqe_iters 10 \
  --rf_components 128 \
  --max_cache_episodes 20 \
  --cache_path FQE_calibration_neurips/results/rlu_cache/cheetah_run_20ep.npz \
  --tfds_data_dir FQE_calibration_neurips/results/tfds
```

Learning-curve screen used for the current audit:

```bash
./.venv/bin/python FQE_calibration_neurips/scripts/run_rlu_cheetah_benchmark.py \
  --stage screen \
  --learners linear_fqe rf_fqe neural_fqe rkhs_minimax_fqe \
  --fqe_iters 150 \
  --rf_components 96 \
  --neural_iters 50 \
  --neural_epochs 2 \
  --neural_target_tau 1.0 \
  --max_cache_episodes 50 \
  --max_transitions_per_split 2500 \
  --policy_indices 0 1 2 3 4 5 6 7 \
  --cache_path FQE_calibration_neurips/results/rlu_cache/cheetah_run_50ep.npz \
  --tfds_data_dir FQE_calibration_neurips/results/tfds \
  --action_samples 1 \
  --output_dir FQE_calibration_neurips/results/rlu_cheetah_learning_curve_screen
```

## Main-Text Benchmark Outcome

The clean main-text benchmark uses the official Deep OPE FQE-L2 value scores on
the first `cheetah_run` learning curve. These scores are informative but
mis-scaled: raw policy-rank Spearman is `0.881`, while raw OPE MAE is large.
Held-out policy calibration gives a stable positive result:

- alternating policy split: linear relative OPE MAE `0.140`, isotonic `0.172`;
- early-to-late split: linear `0.348`, isotonic `0.336`;
- early-five-to-late-three split: linear `0.324`, isotonic `0.301`.

All calibrated rows win on every held-out evaluation policy in these splits.
This is the recommended main-text standard benchmark result: a standard Deep OPE
learned value score is rank-informative but poorly scaled, and a small held-out
calibration set repairs the practically relevant OPE error.

The transition-level RLDS runner remains useful as an audit/development harness
for Bellman calibration diagnostics, neural FQE training, and RKHS minimax
experiments, but the current main-text benchmark evidence should emphasize the
official Deep OPE FQE-L2 score calibration result above.
