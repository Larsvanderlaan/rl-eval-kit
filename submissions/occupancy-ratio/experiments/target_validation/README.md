# Target-Validation Gym Benchmark

This supported experiment script evaluates target-validation-assisted tuning
when a true Gym/Gymnasium generator is available. It fits each candidate once
per setting/seed, then rescores the fitted candidates under finite target-policy
validation rollouts with different rollout counts and horizon caps.

Finite target rollouts are validation samples, not exact infinite-horizon
truth. Use the reported truncation-tail diagnostics to judge whether a horizon
is long enough for the chosen discount.

## Compact Gate

```bash
./.venv/bin/python submissions/occupancy-ratio/experiments/target_validation/gym_target_validation_benchmark.py \
  --output-dir outputs/target_validation_gym_compact \
  --settings gym_pendulum gym_mountain_car_continuous \
  --seeds 0 1 \
  --sample-size 300 \
  --validation-rollouts 4 16 64 \
  --horizons 25 100 200 \
  --truth-rollouts 128 \
  --fqe-candidates 4 \
  --occupancy-candidates 4 \
  --enforce-compact-gate \
  --rerun
```

The compact gate requires no target-validation selection failures, decreasing
tail mass as horizon grows, FQE `target_n_step_td_min` oracle rate at least
`0.80`, occupancy `scalar_finite_prefix_min` oracle rate at least `0.80`, and
occupancy `target_discounted_moments_min` oracle rate at least `0.50`.

## Higher-Validation Pendulum Check

```bash
./.venv/bin/python submissions/occupancy-ratio/experiments/target_validation/gym_target_validation_benchmark.py \
  --output-dir outputs/target_validation_gym_pendulum_high_validation \
  --settings gym_pendulum \
  --seeds 0 1 \
  --sample-size 300 \
  --validation-rollouts 64 256 \
  --horizons 100 200 \
  --truth-rollouts 512 \
  --fqe-candidates 4 \
  --occupancy-candidates 4 \
  --skip-proxy \
  --rerun
```

## Outputs

- `selected_rows.csv`: one row per selector, setting, seed, validation rollout
  count, and horizon. Includes selected candidate, oracle candidate, regret,
  truncation-tail mass, and direct target-return diagnostics.
- `candidate_rows.csv`: one row per fitted candidate and validation cell.
  Includes target-validation scores, scalar scores, true-error diagnostics, and
  occupancy weight diagnostics.
- `summary.csv`: aggregate oracle-selection rates, regret, selected true
  error, and tail mass grouped by selector and validation grid cell.
- `report.md`: compact human-readable summary.
- `gate_report.json`: pass/fail details for the compact gate.

Rows ending in `_min` use raw minimum validation score. Rows without `_min`
use the conservative one-standard-error selector. The package APIs default to
minimum score for target-validation tuning, while retaining `selection_rule="one_se"`
as an explicit opt-in.
