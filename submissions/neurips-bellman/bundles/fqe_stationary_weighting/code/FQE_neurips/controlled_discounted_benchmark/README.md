# Controlled Discounted-Occupancy FQE Benchmark

This subpackage implements a controlled offline RL benchmark for comparing standard fitted Q evaluation (FQE) against target-discounted-occupancy-weighted FQE. The main scientific question is whether weighting the Bellman regression toward the target discounted occupancy distribution `d_{pi,gamma}` improves Q approximation where policy evaluation actually cares, and whether that improves target-policy value estimation.

The gamma-sweep extension separates two roles of gamma:

- `value_gamma`: the Bellman discount defining `Q^pi`, `V^pi`, and the policy-value estimand.
- `ratio_gamma`: the weighting target. `ratio_gamma < 1` targets discounted occupancy; `ratio_gamma = 1` targets the stationary state-action occupancy ratio.

## Benchmark

- State space: `S_t in R^2`
- Action space: `A_t in R`
- Dynamics: `S_{t+1} = B S_t + C A_t + eps_t`
- Reward: quadratic state-control cost
- Behavior policy: linear Gaussian with a shift parameter relative to the target policy
- Target policy: fixed linear Gaussian

The primary data distribution is the behavior discounted occupancy distribution generated from the same Gaussian initial-state distribution `nu0`. The benchmark computes high-accuracy ground truth analytically:

- `Q^pi`
- `V^pi`
- `psi_pi`
- discounted state-action occupancies for the target and behavior policies

## Frozen Configuration

The final main panel uses the misspecified-affine FQE feature class, `n=4000`, `20` seeds, `gamma=0.95`, `process_noise_sd=0.05`, `behavior_action_sd=0.10`, and shifts:

- low: `0.0`
- moderate: `1.1`
- severe: `1.35`

The moderate shift was selected after the initial design search exposed raw-oracle instability at `shift=1.0`; a targeted intermediate-shift refinement found that `shift=1.1` gives a stable estimated-weight value improvement while preserving a clear severe overlap-collapse regime.

## Estimators

The study reports these estimators:

1. `standard_fqe`
2. `oracle_weighted_fqe`
3. `oracle_weighted_fqe_clipped`
4. `estimated_weighted_fqe`
5. `estimated_weighted_fqe_clipped`
6. `estimated_weighted_fqe_clip95`
7. `estimated_weighted_fqe_clip99_ess40`

The oracle estimator uses exact discounted-occupancy density ratios from the analytic Gaussian-mixture benchmark. The estimated-weight variants use a local linear minimax ratio estimator with the correct discounted-occupancy moment equation and `nu0` right-hand side.

## Metrics

Primary metrics:

- target discounted-occupancy Q MSE
- behavior-occupancy Q MSE
- policy-value absolute error
- policy-value squared error
- policy-value bias, variance, and MSE across seeds

Secondary metrics:

- initial-state V MSE
- target-state V MSE
- behavior-state V MSE
- Bellman residual MSE under target and behavior occupancies
- weight diagnostics: mean, standard deviation, max, q90, q95, q99, ESS, ESS fraction, and fraction clipped
- estimated-vs-oracle ratio diagnostics: log-ratio RMSE, weight correlation, MAE, relative MSE
- weighted design condition number and unstable-run flags/reasons
- Monte Carlo SEs for the simulation-estimated Q/V MSE metrics

## Main Final Result

In the final `misspecified_affine, n=4000` panel:

- Low shift behaves as a sanity check: standard and oracle FQE coincide, with target-Q MSE about `3.19` and policy-value MSE about `0.17`.
- Moderate shift shows the mechanism: standard FQE has target-Q MSE about `500` and policy-value MSE about `152`; oracle-clipped FQE reduces them to about `28` and `6.5`; estimated `clip99/ESS40` reduces them to about `9.2` and `14.4`.
- Severe shift is the overlap-failure regime: raw oracle ESS is about `0.008` with max weight about `579`, and raw estimated weights have unstable-run fraction `0.65`; clipped/ESS variants are more stable but remain biased.

The raw oracle row is intentionally retained. At moderate and severe shifts it can be value-unstable despite excellent target-Q error, which the report attributes to finite-sample overlap stress and weighted least-squares sensitivity.

## Gamma-Sweep Extension

The gamma-sweep stages keep `value_gamma=0.95` and vary `ratio_gamma in {0.95, 0.99, 1.0}`. This makes weighting gamma a regularization/targeting parameter: lower values emphasize the initial transient discounted occupancy, while `ratio_gamma=1.0` is the stationary weighting limit.

Additional estimators include:

- `linear_neural_weighted_clipped_fqe`: linear FQE using neural/RKHS estimated ratio weights.
- `neural_standard_fqe`: continuous-action neural FQE without weighting.
- `neural_neural_weighted_clipped_fqe`: neural FQE using neural/RKHS estimated ratio weights.

The neural ratio estimator is a positive MLP weight model trained against a finite RKHS/RBF critic moment objective. The neural FQE is a continuous-action Q-network using Gauss-Hermite quadrature for `E_{A'~pi} Q(S',A')`.

## Output Layout

- Results CSVs: `FQE_neurips/results/`
- Figures: `FQE_neurips/results/figures/`
- Design-search report: `FQE_neurips/results/design_search_report.md`

## Commands

Quick smoke test:

```bash
python -m FQE_neurips.controlled_discounted_benchmark.run_experiment --stage smoke --output-root FQE_neurips/results
```

Exploratory design search:

```bash
python -m FQE_neurips.controlled_discounted_benchmark.run_experiment --stage design_search --output-root FQE_neurips/results
```

Frozen final rerun:

```bash
python -m FQE_neurips.controlled_discounted_benchmark.run_experiment --stage final --output-root FQE_neurips/results
```

Reproduce figures and the estimator-by-shift summary table:

```bash
python -m FQE_neurips.controlled_discounted_benchmark.plot_results --results-root FQE_neurips/results
```

Gamma-sweep smoke test:

```bash
python -m FQE_neurips.controlled_discounted_benchmark.run_experiment --stage gamma_smoke --output-root FQE_neurips/results
```

Gamma-sweep design run:

```bash
python -m FQE_neurips.controlled_discounted_benchmark.run_experiment --stage gamma_design --output-root FQE_neurips/results
```

Gamma-sweep final run:

```bash
python -m FQE_neurips.controlled_discounted_benchmark.run_experiment --stage gamma_final --output-root FQE_neurips/results
```

## Notes

- The main study is explicitly framed around target discounted occupancy norms rather than stationary norms.
- In the gamma sweep, `ratio_gamma=1.0` is stationary weighting only; it is not an undiscounted FQE value equation.
- MuJoCo-style benchmarks are deferred; the main benchmark is the controlled linear-Gaussian design.
- FQE and ratio-estimation hyperparameters are fixed across the design-search grid to avoid hidden estimator-specific tuning.
- Some supporting feature/sample-size slices remain unstable, especially well-specified small-sample weighted fits and diagonal-quadratic misspecification. These are kept as diagnostics rather than hidden.
