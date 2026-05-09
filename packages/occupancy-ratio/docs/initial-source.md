# Initial-Source Correction

The discounted fixed point needs an initial-source term. In this package,
`initial_ratio_mode="auto"` chooses the safest available source object from the
data you provide.

## Auto Mode

| Inputs | Auto behavior |
| --- | --- |
| `initial_states` and `initial_actions` | Fit the joint initial state-action ratio. |
| `initial_states` only | Fit a state-only source ratio and multiply by the action ratio. |
| no `initial_states` | Use source ratio `1`, preserving backward compatibility. |

## Joint Path

When both `initial_states` and `initial_actions` are available, the fitted
source object is

```text
rho_initial(s) * pi(a | s) / [rho_ref(s) * pi0(a | s)]
```

The denominator rows are behavior/reference state-action rows `(states,
actions)`. The numerator rows are target initial state-action rows
`(initial_states, initial_actions)`.

This is the preferred algorithmic path because it directly estimates the
source term used by the occupancy fixed point.

## Factored Fallback

When initial actions are unavailable, the package estimates

```text
rho_initial(s) / rho_ref(s)
```

on states, then multiplies the query source-state ratio by the action ratio:

```text
source_state_ratio_query * action_ratio
```

This fallback is useful for datasets that expose initial states but cannot
sample target-policy actions at those states.

## Weights

`initial_weights` are normalized within the numerator block. They change the
relative contribution of initial samples without rescaling the density-ratio
objective.

## Diagnostics

Joint-source diagnostics use keys such as:

- `initial_joint_ratio_enabled`
- `initial_joint_ratio_mean`
- `initial_joint_ratio_ess_fraction`
- `initial_joint_ratio_density_ratio_loss`

Factored fallback diagnostics use keys such as:

- `source_state_ratio_enabled`
- `source_state_ratio_mean`
- `source_state_ratio_ess_fraction`
- `source_state_ratio_density_ratio_loss`
