---
title: FQE
description: Fitted Q evaluation, tuning, stationary weighting, calibration, and validation.
template: splash
hero:
  title: FQE
  tagline: Fitted Q evaluation for target-policy values, calibration, and model selection under logged-data shift.
  image:
    html: |
      <div class="package-hero-visual package-hero-visual--fqe" aria-hidden="true">
        <div class="package-hero-visual__top">
          <span>Bellman fit</span>
          <strong>stable defaults</strong>
        </div>
        <div class="mini-chart mini-chart--fqe">
          <span style="height: 38%"></span>
          <span style="height: 58%"></span>
          <span style="height: 44%"></span>
          <span style="height: 72%"></span>
          <span style="height: 64%"></span>
        </div>
        <div class="package-visual-metrics">
          <span>Q model</span>
          <span>policy value</span>
          <span>calibration</span>
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

<div class="package-site-shell package-site--fqe">
  <nav class="package-site-tabs" aria-label="FQE site sections">
    <a aria-current="page" href="./">Overview</a>
    <a href="./quickstart/">Quickstart</a>
    <a href="./methods/">Methods</a>
    <a href="./diagnostics/">Diagnostics</a>
    <a href="./benchmarks/">Benchmarks</a>
  </nav>
  <section class="package-positioning">
    <div>
      <p class="eyebrow">Policy-value workflow</p>
      <h2>Direct-method OPE with calibration hooks.</h2>
      <p>
        <code>fqe</code> estimates a fixed target policy's Q-function and
        initial-state policy value from logged transitions, target-policy next
        actions, rewards, and bootstrap masks.
      </p>
    </div>
    <dl class="package-metric-panel">
      <div>
        <dt>Primary output</dt>
        <dd>Q-functions and policy values</dd>
      </div>
      <div>
        <dt>Default path</dt>
        <dd>Boosted FQE, then neural when needed</dd>
      </div>
      <div>
        <dt>Audit trail</dt>
        <dd>Bellman losses, calibration, target validation</dd>
      </div>
    </dl>
  </section>
  <section class="package-pillars" aria-label="FQE package pillars">
    <article>
      <span>01</span>
      <h3>Estimate values</h3>
      <p>Fit reusable Q models and average initial target-policy actions for policy-value estimates.</p>
    </article>
    <article>
      <span>02</span>
      <h3>Select models</h3>
      <p>Compare boosted, neural, value-only, and validation-assisted candidates without oracle leakage.</p>
    </article>
    <article>
      <span>03</span>
      <h3>Review diagnostics</h3>
      <p>Summarize Bellman fit, calibration, target-validation coverage, and value stability.</p>
    </article>
  </section>
</div>

## What is estimated?

FQE estimates the action-value function for a fixed target policy `pi`. In
plain terms, `Q^pi(s,a)` is the discounted reward you expect after taking
action `a` in state `s`, then following `pi`.

<div class="estimand-card estimand-card--fqe">
  <p class="estimand-label">Target-policy Q-function</p>
  <pre class="math-display"><code>Q^\pi(s,a)
= \mathbb{E}_\pi\!\left[
    \sum_{t=0}^{\infty} \gamma^t R_t
    \mid S_0=s,\ A_0=a
  \right]</code></pre>
</div>

The fitted model is trained to satisfy the Bellman equation: today's reward
plus the discounted value of the next target-policy action.

<div class="estimand-card estimand-card--fqe">
  <p class="estimand-label">Bellman equation fitted by regression</p>
  <pre class="math-display"><code>Q^\pi(s,a)
= r(s,a)
  + \gamma\,
    \mathbb{E}\!\left[
      Q^\pi(S', A')
      \mid S=s,\ A=a,\ A' \sim \pi(\cdot \mid S')
    \right]</code></pre>
</div>

`model.predict_q(...)` returns fitted Q values for queried state-action rows.
After fitting `Q_hat`, the initial-state policy value is the average
target-policy Q value at the initial rows:

<div class="estimand-card estimand-card--fqe">
  <p class="estimand-label">Policy value reported by the estimator</p>
  <pre class="math-display"><code>V_0^\pi
= \mathbb{E}_{S_0 \sim \rho_0,\ A_0 \sim \pi(\cdot \mid S_0)}
    [Q^\pi(S_0,A_0)]</code></pre>
</div>

Notation: `gamma` is the discount, `R_t` is reward, `S'` is the next
state, `A'` is a target-policy next action, and `rho_0` is the initial-state
distribution represented by `initial_states`.

## Install

```bash
python -m pip install -e "packages/fqe[neural,benchmark]"
```

Use the narrower package if you only need boosted FQE:

```bash
python -m pip install -e "packages/fqe"
```

## Minimal example

```python
from fqe import BoostedFQEConfig, fit_fqe_lgbm

model = fit_fqe_lgbm(
    states=states,
    actions=actions,
    next_states=next_states,
    next_actions=next_actions_under_eval_policy,
    rewards=rewards,
    gamma=0.99,
    terminals=dones,
    sample_weight=row_weights,
    config=BoostedFQEConfig.stable_defaults(seed=123),
)

q_values = model.predict_q(states, actions)
policy_value = model.estimate_policy_value(initial_states, initial_actions)
```

## Data contract

| Field | Required | Shape intent |
| --- | --- | --- |
| `states`, `actions` | Yes | Logged transition rows |
| `next_states` | Yes | One next state per row |
| `next_actions` | Yes in Q-mode | One or many sampled evaluation-policy actions per next state |
| `rewards` | Yes | One reward per row |
| `gamma` | Yes | Value-estimation discount |
| `terminals` | Recommended | Terminal mask for Bellman targets |
| `sample_weight` | Optional | User row weights propagated through fitting and validation |
| `initial_states`, `initial_actions` | For value estimates and tuning | Evaluation-policy initial rows |

`next_actions` can be shape `(n, action_dim)` or `(n, n_action_samples,
action_dim)`. Multiple actions are averaged in the Bellman target.

## When to use it

- You have logged transitions and actions sampled from an evaluation policy.
- You need a direct-method OPE estimate or reusable Q-function.
- You want stable boosted defaults first, then neural FQE when the function
  class or workload calls for it.
- You need target-validation assisted selection, Bellman calibration, or
  post-hoc model selection among many Q candidates.

## What to review before relying on an estimate

- Evaluation-policy actions should be plausible in the logged state space.
- Held-out Bellman risk is most useful when read alongside target-policy
  coverage and value stability.
- Target-validation reports should show how much discounted rollout mass remains
  after the observed prefix.
- Tuning reports should make the selected model, score components, and final
  refit diagnostics easy to inspect.

## Methods

| Method | Entry point | Use case |
| --- | --- | --- |
| Boosted FQE | `fit_fqe_lgbm` | Stable default, tabular or structured arrays |
| Neural FQE | `fit_fqe_neural` | Larger continuous-control workloads |
| Value-only FVI | `fit_value_lgbm`, `fit_value_neural` | Bellman operator already expressed over states |
| Automatic tuning | `tune_fqe_auto` | Candidate search and final refit |
| Target validation | `tune_fqe_with_target_validation` | Independent target-policy rollouts or labels |
| Stationary weighting | `fit_stationary_weighted_fqe` | Reweighted Bellman regression under distribution shift |
| Bellman calibration | `fit_bellman_calibrator` | Post-hoc calibration diagnostics and correction |
| Low-rank SBV | `LowRankOperatorSBVValidator` | Efficient selection among many Q candidates |

## Papers

- [Fitted Q Evaluation Without Bellman Completeness via Stationary Weighting](../papers/)
- [Bellman Calibration for V-Learning in Offline Reinforcement Learning](../papers/)
- [Stationary Reweighting Yields Local Convergence of Soft Fitted Q-Iteration](../papers/)

## API links

- [Package README](https://github.com/Larsvanderlaan/rl-eval-kit/blob/main/packages/fqe/README.md)
- [Top-level exports](https://github.com/Larsvanderlaan/rl-eval-kit/blob/main/packages/fqe/fqe/__init__.py)
- [Low-rank SBV docs](https://github.com/Larsvanderlaan/rl-eval-kit/blob/main/packages/fqe/docs/low_rank_operator_sbv.md)
