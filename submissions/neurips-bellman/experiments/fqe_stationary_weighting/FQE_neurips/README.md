# Stationary-Weighted FQE in `FQE_neurips`

This folder contains a self-contained experimental pipeline for stationary-weighted fitted Q evaluation (FQE), together with two benchmark families:

- a realistic latent Garnet benchmark for the correction-versus-stability tradeoff;
- a harder hub-spoke benchmark designed to expose off-policy projected-FQE failure modes.

The code supports both stationary ratios (`gamma_ratio=1`) and discounted occupancy ratios (`gamma_ratio<1`).

## Core modules

- `ratio_estimation.py`: closed-form linear ratio estimation, linear saddle solver, and neural saddle ratio estimation from the paper's moment equations.
- `neural_rkhs_weights.py`: RKHS-critic / neural-weight estimator with Nystr\"om-style kernel machinery and explicit stabilization.
- `saddle_optim.py`: reusable extragradient-style saddle optimizer.
- `fqe.py`: weighted neural FQE.
- `fqe_linear.py`: weighted linear FQE, including an iterative projected-FQE solver and a direct projected fixed-point solve.
- `sw_fqe.py`: user-facing end-to-end stationary-weighted FQE wrapper.
- `utils.py`: shared data containers, tabular exact-solution helpers, stabilization utilities, and diagnostics.

## Ratio estimation

The stationary ratio satisfies the moment equation

`E[d(S,A) {g(S,A) - g(S',A')}] = 0`.

To regularize the problem, the estimators expose `gamma_ratio` explicitly and use the discounted analogue

`E[d_gamma(S,A) {g(S,A) - gamma_ratio g(S',A')}] = (1-gamma_ratio) E[g(S,A)]`.

- `gamma_ratio = 1` corresponds to stationary weighting.
- `gamma_ratio < 1` corresponds to discounted occupancy weighting.

Implemented estimators:

1. Linear ratio estimation
   - `estimate_ratio_closed_form_linear`
   - `estimate_ratio_saddle_linear`
   - both take feature matrices explicitly, so basis choices are swappable.

2. Neural saddle ratio estimation
   - `estimate_ratio_saddle_neural`
   - neural actor and critic with explicit ridge, normalization, clipping, and early stopping.

3. Neural RKHS-critic ratio estimation
   - `estimate_ratio_neural_rkhs`
   - neural weight model with a kernel-ridge critic approximation.

Default stabilization is shared across estimators:

- positivity enforcement;
- quantile / hard upper clipping;
- normalization to mean one;
- adaptive shrinkage toward uniform to hit a target ESS fraction when requested.

## FQE

The FQE code is modular by design:

- `fit_weighted_fqe_nn(..., weights=...)` runs weighted neural FQE directly;
- `fit_weighted_linear_fqe(..., weights=...)` runs weighted linear FQE directly;
- `fit_stationary_weighted_fqe(...)` is the end-to-end wrapper that estimates weights and then runs FQE.

Important stability note:

- caller-supplied weights are no longer re-stabilized internally; FQE only enforces positivity / normalization on the final vector it receives.
- this avoids accidentally shrinking already calibrated stationary weights back toward uniform.

## Benchmarks

### 1. Realistic benchmark

- `latent_garnet_benchmark.py`
- nonlinear latent-state MDP with exact tabular truth still available.
- intended for the realistic correction-versus-stability story.

Data generation modes:

- `trajectory`
- `multi_trajectory`
- `stationary_iid`
- `mixed`

The recommended default is now:

- `data_mode="mixed"`
- `n_trajectories=200`
- `iid_fraction=0.5`

This gives a blend of short trajectory data and stationary-like i.i.d. transitions.

### 2. Hard benchmark

- `hub_spoke_latent_benchmark.py`
- hub/spoke/goal latent MDP with strong off-policy mismatch and exact truth.
- designed so linear FQE can be realizable but not Bellman complete.

This benchmark is the main mechanism test:

- zero-init linear FQE;
- stationary target-norm evaluation;
- oracle stationary weighting used during tuning;
- policy-ratio weighting included as a baseline.
- linear FQE keeps the oracle realizability feature, while neural FQE now uses richer non-oracle `neural_structured` inputs built from raw observations plus coarse geometry.

## Main experiment drivers

- `hard_benchmark_experiment.py`
  - searches and evaluates the tuned hard regime.
- `hard_benchmark_consistency_experiment.py`
  - supporting sample-size study for the hard benchmark.
- `hard_benchmark_rkhs_interpolation_experiment.py`
  - interpolates flexible neural FQE between unweighted and RKHS-weighted endpoints.
- `fqe_coverage_study.py`
  - realistic benchmark coverage sweep.
- `ratio_accuracy_experiment.py`
  - exact weight-recovery study.
- `compare_ratio_targets_experiment.py`
  - compares stationary versus discounted ratio targets.
- `hard_benchmark_longrun_plot.py`
  - 50-seed long-run undamped linear-FQE convergence figure on the selected hard setting.
- `paper_outputs.py`
  - final paper-output generator with:
    - realistic pilot selection of stationary methods,
    - compact frozen method set,
    - updated Q/V metric bundle,
    - hard linear convergence artifacts and secondary hard neural table.

## Recommended runs

From the repo root:

Hard benchmark screen:

```bash
python -m FQE_neurips.hard_benchmark_experiment
```

Hard benchmark consistency diagnostic:

```bash
python -m FQE_neurips.hard_benchmark_consistency_experiment
```

Hard benchmark RKHS interpolation:

```bash
python -m FQE_neurips.hard_benchmark_rkhs_interpolation_experiment
```

Realistic benchmark coverage sweep:

```bash
python -m FQE_neurips.fqe_coverage_study \
  --data-mode mixed \
  --n-trajectories 200 \
  --iid-fraction 0.5
```

Minimal end-to-end SW-FQE run:

```bash
python -m FQE_neurips.experiment
```

Final paper outputs:

```bash
python -m FQE_neurips.paper_outputs
```

## Current experiment framing

Recommended paper story:

1. Hard benchmark
   - main sharp result.
   - shows long-run undamped linear-FQE convergence where stationary weighting clearly beats unweighted.

2. Realistic benchmark
   - main neural benchmark.
   - uses mixed trajectory/IID data and the compact frozen method set after pilot selection.
   - the recommended paper setting uses a milder evaluation horizon and richer observations for neural FQE (`gamma_eval=0.95`, `observation_mode="rich"`).
   - reports stationary and behavior norms for `Q`, stationary and behavior norms for `V`, and initial-value errors.

3. Hard-benchmark neural comparison
   - secondary fairness check using non-oracle structured neural inputs.
   - not the primary neural claim.

4. Consistency experiment / RKHS interpolation
   - supporting diagnostics and ablations, not the main benchmark claim.

## Notes

- The hard benchmark is intentionally controlled, but not tuned through pathological initialization; the primary runs use standard zero initialization.
- The infinite-horizon setting is the default throughout; `gamma_eval < 1` is the discounted infinite-horizon case, not a finite-horizon truncation.
- Oracle information is only used inside the hard benchmark's linear FQE basis to make the class realizable; it is not exposed to ratio estimation or neural FQE.
