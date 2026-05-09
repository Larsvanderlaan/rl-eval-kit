# Data Shapes

All fitters accept NumPy-like arrays and coerce one-dimensional inputs into
two-dimensional matrices. The safest convention is to pass explicit
two-dimensional arrays.

| Argument | Shape | Meaning |
| --- | --- | --- |
| `states` | `(n, state_dim)` | Current states from logged behavior transitions. |
| `actions` | `(n, action_dim)` | Observed behavior-policy actions. |
| `next_states` | `(n, state_dim)` | Next states for each logged transition. |
| `target_actions` | `(n, action_dim)` or sampled rows | Target-policy actions at current states. |
| `target_next_actions` | `(n, action_dim)` or sampled rows | Target-policy actions at next states for direct one-step ratios. |
| `initial_states` | `(m, state_dim)` | Evaluation initial-state samples. |
| `initial_actions` | `(m, action_dim)` | Target-policy actions at `initial_states`. |
| `initial_weights` | `(m,)` | Optional weights for initial samples, normalized within the numerator block. |
| `terminals` | `(n,)` | True environment terminal flags. |
| `timeouts` | `(n,)` | Truncation/time-limit flags. |

## Target Actions

`target_actions` are not behavior actions. They should be sampled from the
evaluation policy at the same current states used in `states`.

If you can also sample target-policy actions at `next_states`, pass
`target_next_actions`. That enables the direct one-step-ratio path and avoids
unnecessary transition-ratio Monte Carlo variance.

## Terminals And Timeouts

`terminals` stop Bellman continuation. `timeouts` depend on the data source:

- `handle_timeouts="nonterminal"` treats time limits as truncations, so
  continuation remains active.
- `handle_timeouts="terminal"` treats timeouts as episode-ending transitions.

Use `absorbing_state=True` only when the data encoding and downstream method
expect absorbing-state continuation semantics.

## Known Action Ratios

If you already know `pi(a | s) / pi0(a | s)` on logged rows, pass
`action_ratio_values`. If you have log probabilities instead, pass
`behavior_log_prob` and `target_log_prob`.

Known ratios can be clipped with `known_action_ratio_clip_max` and optionally
mean-normalized with `known_action_ratio_normalize`. When known ratios are used,
public action-ratio prediction is only available on exact fitted rows.
