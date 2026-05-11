---
title: Discounted Occupancy Ratios
description: Discounted occupancy-ratio estimation with FORI, diagnostics, tuning, and benchmarks.
template: splash
hero:
  title: Discounted Occupancy Ratios
  tagline: Target-policy reweighting, source correction, and ratio diagnostics for logged data.
  image:
    html: |
      <div class="package-hero-visual package-hero-visual--ratio" aria-hidden="true">
        <div class="package-hero-visual__top">
          <span>FORI fit</span>
          <strong>source correction</strong>
        </div>
        <div class="ratio-rings">
          <span></span>
          <span></span>
          <span></span>
        </div>
        <div class="package-visual-metrics">
          <span>state-action ratio</span>
          <span>ESS</span>
          <span>tail checks</span>
        </div>
      </div>
  actions:
    - text: Quickstart
      link: ./quickstart/
    - text: Methods
      link: ./methods/
      variant: secondary
    - text: Diagnostics
      link: ./diagnostics/
      variant: minimal
---

<div class="package-site-shell package-site--ratio">
  <nav class="package-site-tabs" aria-label="Discounted occupancy-ratio site sections">
    <a aria-current="page" href="./">Overview</a>
    <a href="./quickstart/">Quickstart</a>
    <a href="./methods/">Methods</a>
    <a href="./diagnostics/">Diagnostics</a>
    <a href="./benchmarks/">Benchmarks</a>
  </nav>
  <section class="package-positioning">
    <div>
      <p class="eyebrow">Ratio workflow</p>
      <h2>Target-policy reweighting with source-aware diagnostics.</h2>
      <p>
        The <code>occupancy-ratio</code> package estimates discounted
        occupancy ratios from logged reference transitions and target-policy
        action samples. The importable package is <code>occupancy_ratio</code>.
      </p>
    </div>
    <dl class="package-metric-panel">
      <div>
        <dt>Primary output</dt>
        <dd>Discounted state-action weights</dd>
      </div>
      <div>
        <dt>Default path</dt>
        <dd>Stable FORI with LSIF nuisance ratios</dd>
      </div>
      <div>
        <dt>Audit trail</dt>
        <dd>ESS, tails, clipping, source correction</dd>
      </div>
    </dl>
  </section>
  <section class="package-pillars" aria-label="Occupancy-ratio package pillars">
    <article>
      <span>01</span>
      <h3>Reweight rows</h3>
      <p>Predict state-action weights that move reference rows toward the target policy's normalized discounted occupancy.</p>
    </article>
    <article>
      <span>02</span>
      <h3>Anchor sources</h3>
      <p>Use joint initial ratios when initial states and target-policy initial actions are available.</p>
    </article>
    <article>
      <span>03</span>
      <h3>Review weight quality</h3>
      <p>Summarize ESS, variation, tail concentration, clipping, and source-correction status.</p>
    </article>
  </section>
</div>

## What is estimated?

The package estimates a discounted state-action density ratio. Under support
assumptions, the ratio identifies expectations under the target policy's
normalized discounted occupancy by reweighting logged reference rows.

<div class="estimand-card estimand-card--ratio">
  <p class="estimand-label">Discounted state-action density ratio</p>
  <pre class="math-display"><code>w^\pi_\gamma(s,a)
= \frac{d^\pi_\gamma(s,a)}{d^{\mathrm{ref}}(s,a)}
= \frac{\rho^\pi_\gamma(s)\,\pi(a \mid s)}
       {\rho^{\mathrm{ref}}(s)\,\pi_0(a \mid s)}</code></pre>
</div>

The defining identity is a reweighting identity: averages under the target
policy's normalized discounted occupancy should match weighted averages over
the logged reference rows.

<div class="estimand-card estimand-card--ratio">
  <p class="estimand-label">How the weights are used</p>
  <pre class="math-display"><code>\mathbb{E}_{(S,A)\sim d^\pi_\gamma}[f(S,A)]
\approx
\mathbb{E}_{(S,A)\sim d^{\mathrm{ref}}}
  [w^\pi_\gamma(S,A)\,f(S,A)]</code></pre>
</div>

`model.predict_state_action_ratio(...)` returns this fitted weight for queried
state-action rows.

Notation:

| Symbol | Meaning |
| --- | --- |
| `rho^pi_gamma(s)` | Target policy's normalized discounted state occupancy |
| `rho^ref(s)` | Reference or behavior state distribution represented by the rows |
| `pi(a | s)` | Target policy action density or probability |
| `pi0(a | s)` | Behavior/reference policy action density or probability |
| `d^ref(s,a)` | Logged state-action reference distribution |

## Install

```bash
python -m pip install -e "packages/occupancy-ratio[neural,benchmark]"
```

Use docs extras when building the package-local MkDocs reference:

```bash
python -m pip install -e "packages/occupancy-ratio[docs]"
```

## Minimal example

```python
from occupancy_ratio import (
    ActionRatioConfig,
    OccupancyRegressionConfig,
    TransitionRatioConfig,
    fit_discounted_occupancy_ratio,
)

model = fit_discounted_occupancy_ratio(
    states=states,
    actions=actions,
    next_states=next_states,
    target_actions=target_actions_under_pi,
    gamma=0.99,
    occupancy=OccupancyRegressionConfig.stable_defaults(seed=123),
    action_ratio=ActionRatioConfig.stable_defaults(show_progress=False),
    transition_ratio=TransitionRatioConfig.stable_defaults(show_progress=False),
)

weights = model.predict_state_action_ratio(states, actions)
state_ratios = model.predict_state_ratio(states, actions)
```

## Data contract

| Field | Required | Shape intent |
| --- | --- | --- |
| `states`, `actions` | Yes | Behavior/reference rows |
| `next_states` | Yes | One next state per reference row |
| `target_actions` | Yes | Target-policy current actions for reference states |
| `target_next_actions` | For target validation and some workflows | Target-policy actions at next states |
| `initial_states` | For source correction | Initial target-state numerator rows |
| `initial_actions` | For joint source correction | Target-policy initial actions |
| `rewards` | For OPE-aware tuning | Reward proxy and validation diagnostics |
| `sample_weight` | Optional | User row weights |

## Source correction

| Available data | Source path | Operational meaning |
| --- | --- | --- |
| No `initial_states` | Source ratio is 1 | Backward-compatible fit against the reference rows |
| `initial_states` only | Factored state-source correction | Fit `rho_initial / rho_ref`, then multiply by the target-action ratio |
| `initial_states` and `initial_actions` | Joint initial state-action correction | Fit the initial state-action source directly using target-policy initial actions |

## When to use it

- You need discounted density ratios for OPE, weighted FQE, or diagnostics.
- You want a fitted-regression alternative to coupled minimax or DICE-style
  saddle optimization.
- You want automatic tuning and ratio-quality diagnostics.
- You want Google DualDICE as an optional external comparator or backend.

## Ratio quality checks

- ESS is a diagnostic, not a target. Under meaningful behavior-target mismatch,
  near-one ESS with near-zero weight variation can indicate a nearly constant
  ratio rather than a successful fit.
- Clipping and tail summaries should show whether a few rows dominate the
  reweighted estimate.
- Source-correction diagnostics should make clear whether the fit used no
  initial source, factored state-source correction, or joint initial
  state-action correction.
- Optional backends such as Google DualDICE should report clear adapter status
  when dependencies or external checkouts are unavailable.

## Methods

| Method | Entry point | Use case |
| --- | --- | --- |
| Boosted FORI | `fit_discounted_occupancy_ratio` | Stable single fit with LightGBM |
| Neural FORI | `fit_discounted_occupancy_ratio_neural` | Larger continuous-control workloads |
| Automatic tuning | `tune_occupancy_ratio_auto` | Deterministic candidate search and final refit |
| Target validation | `tune_occupancy_ratio_with_target_validation` | Independent target-policy moment or scalar validation |
| Google DualDICE | `fit_google_dualdice_occupancy_ratio` | Optional external comparator/backend |
| Benchmarks | `occupancy-ratio-benchmark` | Controlled and realistic OPE screens |

## Papers

- [Fitted Occupancy-Ratio Iteration for Offline Reinforcement Learning](../papers/)
- [Fitted Q Evaluation Without Bellman Completeness via Stationary Weighting](../papers/)

## API links

- [Package README](https://github.com/Larsvanderlaan/rl-eval-kit/blob/main/packages/occupancy-ratio/README.md)
- [Package docs source](https://github.com/Larsvanderlaan/rl-eval-kit/tree/main/packages/occupancy-ratio/docs)
- [Top-level exports](https://github.com/Larsvanderlaan/rl-eval-kit/blob/main/packages/occupancy-ratio/occupancy_ratio/__init__.py)
