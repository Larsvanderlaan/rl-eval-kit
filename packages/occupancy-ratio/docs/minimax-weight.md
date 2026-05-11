# Minimax Weight Backends

`fit_minimax_weight(...)` provides a common UI for external DICE and minimax
weight estimators while preserving the usual occupancy-ratio prediction helpers.
The estimand is the same discounted state-action occupancy weight used by the
FORI fitters; use a large discount below one, such as `gamma=0.95` or `0.99`,
when the goal is a near-stationary approximation.

```python
from occupancy_ratio import MinimaxWeightConfig, fit_minimax_weight

model = fit_minimax_weight(
    states=states,
    actions=actions,
    next_states=next_states,
    target_actions=target_actions,
    target_next_actions=target_next_actions,
    gamma=0.95,
    initial_states=initial_states,
    initial_actions=initial_actions,
    method="auto",
    config=MinimaxWeightConfig(method="auto"),
)

weights = model.predict_state_action_ratio(states, actions)
diagnostics = model.diagnostics
```

Supported methods are:

- `auto`: currently resolves to `google_policy_eval_dualdice`.
- `google_policy_eval_dualdice`: official Google Research
  `policy_eval.dual_dice.DualDICE`.
- `google_dice_rl_dualdice_exact`: official DICE-RL NeuralDice with the
  DualDICE-recovery flags.
- `google_dice_rl_recommended`: official DICE-RL NeuralDice with the
  recommended regularized form.
- `scope_rl_minimax_state_action`: SCOPE-RL continuous state-action minimax
  weight learning.
- `scope_rl_minimax_state`: SCOPE-RL continuous state minimax weights; requires
  `behavior_action_pscore`.

Google methods require target-policy `initial_states`, `initial_actions`, and
`target_next_actions`. Pass `rewards` when using
`google_dice_rl_recommended` so the official regularized NeuralDice objective
matches the benchmark setting; exact DualDICE-recovery mode ignores rewards.
SCOPE-RL methods work best with ordered trajectories:
pass `step_per_trajectory`, or pass aligned `episode_ids` and `timesteps` so the
wrapper can keep complete trajectory blocks. Optional dependencies are loaded
only by preflight or fit calls.

```python
from occupancy_ratio import ScopeRLMinimaxWeightConfig, MinimaxWeightConfig

config = MinimaxWeightConfig(
    method="scope_rl_minimax_state_action",
    scope_rl=ScopeRLMinimaxWeightConfig(
        n_steps=5000,
        n_steps_per_epoch=5000,
        device="cpu",
    ),
)
```
