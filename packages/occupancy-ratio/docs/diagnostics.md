# Diagnostics And Calibration

Occupancy-ratio estimation is useful only when the weights are both meaningful
and stable. Always inspect final weights, nuisance ratios, and fixed-point
history before relying on an OPE number.

## Weight Summary

```python
from occupancy_ratio import weight_summary

omega = model.predict_state_action_ratio(states, actions)
summary = weight_summary(omega, cap=model.occupancy_ratio_max)
```

Important fields:

- `ess_fraction`: effective sample size divided by number of finite weights.
- `cv`: coefficient of variation.
- `p95`, `p99`, `max`: tail diagnostics.
- `clipped_fraction`: fraction at the supplied cap.
- `nonfinite_fraction`: invalid predictions before postprocessing.

High ESS with near-zero CV is not automatically good. If the behavior and
target policies differ meaningfully, nearly uniform weights can indicate
collapse or over-regularization.

## Regularization Path

`regularization_path_report(...)` summarizes available intermediate and final
weight stages from a fitted model. Use it when debugging whether stabilization,
normalization, nuisance fits, or final projection is driving the result.

## Bellman-Moment Calibration

The calibration utilities provide a conservative post-hoc adjustment of fitted
weights using backward occupancy Bellman moments:

```python
from occupancy_ratio import calibrate_occupancy_bellman_binning

calibrated = calibrate_occupancy_bellman_binning(
    omega_hat=omega,
    h=h_features,
    h_next=h_next_features,
    init_moments=init_moments,
    gamma=0.99,
    w_max=model.occupancy_ratio_max,
)
omega_cal = calibrated["omega_cal"]
diagnostics = calibrated["diagnostics"]
```

Calibration can reduce supported Bellman residuals, but it cannot recover
unsupported target occupancy mass. Use `calibration_recommendation` and tail
diagnostics before replacing the original weights.
