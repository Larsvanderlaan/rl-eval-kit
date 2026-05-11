# Target-Validation Assisted Tuning

Target-validation tuning is an opt-in path for settings where independent
target-policy validation rollouts or simulator labels are available. The
standard `tune_occupancy_ratio` and `tune_occupancy_ratio_auto` APIs remain
proxy-only and never select using target-policy Monte Carlo values or benchmark
truth.

```python
from occupancy_ratio import tune_occupancy_ratio_with_target_validation

tuned = tune_occupancy_ratio_with_target_validation(
    states=states,
    actions=actions,
    next_states=next_states,
    target_actions=target_actions_under_pi,
    target_next_actions=target_next_actions_under_pi,
    rewards=rewards,
    gamma=0.99,
    initial_states=initial_states,
    initial_actions=initial_actions_under_pi,
    validation_states=target_states,
    validation_actions=target_actions,
    validation_rewards=target_rewards,
    validation_episode_ids=target_episode_ids,
    validation_timestep=target_timesteps,
    validation_continuation=target_continuation,
)
```

The default `score_mode="discounted_moments"` compares candidate
reference-weighted moments, `E_ref[w f]`, against empirical target-policy
discounted-occupancy moments. Finite validation rollouts are validation
samples, not exact infinite-horizon truth; inspect
`truncation_tail_mass_mean` and `truncation_tail_mass_max` in
`validation_diagnostics`.

Selection defaults to `selection_rule="min_score"` after occupancy guardrails.
Pass `selection_rule="one_se"` for a conservative one-standard-error selector.
Diagnostics always include both `selected_min_score_candidate_id` and
`selected_one_se_candidate_id`.

Scalar-only mode is available as `score_mode="scalar_ope"`. In that mode, the
selector compares `mean_ref(w * reward)` to `(1 - gamma) * target_value`, using
the package's normalized discounted-occupancy convention. Scalar mode is
value-only and does not validate the full ratio function. Guardrails still run
before scalar comparison.
