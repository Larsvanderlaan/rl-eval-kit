# AutoML Tuning

`tune_occupancy_ratio_auto(...)` is the recommended product tuning entrypoint.
It performs deterministic, capped cross-validation and final refit on all data.

The default is deliberately conservative:

- neural-only candidate family;
- row-wise 3-fold CV unless `groups` are supplied;
- proxy-only scoring;
- full evaluation of the stable baseline;
- final stable-baseline fallback guardrail.

```python
from occupancy_ratio import OccupancyTuningConfig, tune_occupancy_ratio_auto

tuned = tune_occupancy_ratio_auto(
    states=states,
    actions=actions,
    next_states=next_states,
    target_actions=target_actions,
    gamma=0.99,
    rewards=rewards,
    initial_states=initial_states,
    initial_actions=initial_actions,
    config=OccupancyTuningConfig(budget="balanced"),
)

model = tuned.model
candidate_rows = tuned.candidate_rows()
fold_rows = tuned.fold_rows()
```

## Budgets

| Budget | Candidates | Promotion | Use when |
| --- | --- | --- | --- |
| `fast` | up to 8 | up to 2 | CI smoke checks and interactive triage. |
| `balanced` | up to 16 | up to 4 | Default product tuning. |

## Families

```python
# Default: neural only.
config = OccupancyTuningConfig()

# Boosted only.
config = OccupancyTuningConfig(families=("boosted",))

# Compare boosted and neural candidates.
config = OccupancyTuningConfig(families=("boosted", "neural"))
```

Google DualDICE is not included by default. To add it, use a neural family,
provide joint initial state-action rows, and set `include_google_dualdice=True`.

## Selection Signals

The scorer uses held-out proxy signals:

- fixed-point moment balance;
- validation/convergence diagnostics;
- optional reward-weighted OPE stability;
- weight quality and tail behavior;
- runtime.

It never selects using oracle ratios, target-policy Monte Carlo values, or
benchmark truth. ESS is a diagnostic, not a success metric by itself: severe
tail collapse and near-uniform collapse under meaningful policy shift are both
penalized.

## Result Tables

`candidate_rows()` returns one row per candidate-stage combination with score,
runtime, selected/promoted flags, errors, and metric columns.

`fold_rows()` returns one row per candidate fold with fold-level proxy metrics.
These rows are intended for product debugging and benchmark reports.
