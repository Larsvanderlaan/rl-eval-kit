---
title: genPQR
description: Modular generalized policy-to-Q-to-reward tools for inverse reinforcement learning.
template: splash
hero:
  title: genPQR
  tagline: A modular inverse-RL site for behavior-policy estimation, Q evaluation, and normalized rewards.
  image:
    html: |
      <div class="package-hero-visual package-hero-visual--genpqr" aria-hidden="true">
        <div class="package-hero-visual__top">
          <span>Reward fit</span>
          <strong>adapter ready</strong>
        </div>
        <div class="reward-map">
          <span></span>
          <span></span>
          <span></span>
          <span></span>
          <span></span>
          <span></span>
        </div>
        <div class="package-visual-metrics">
          <span>policy</span>
          <span>Q</span>
          <span>reward</span>
        </div>
      </div>
  actions:
    - text: Quickstart
      link: ./quickstart/
    - text: Workflows
      link: ./workflows/
      variant: secondary
    - text: Deployment
      link: ./deployment/
      variant: minimal
---

<div class="package-site-shell package-site--genpqr">
  <nav class="package-site-tabs" aria-label="genPQR site sections">
    <a aria-current="page" href="./">Overview</a>
    <a href="./quickstart/">Quickstart</a>
    <a href="./methods/">Methods</a>
    <a href="./workflows/">Workflows</a>
    <a href="./deployment/">Deployment</a>
  </nav>
  <section class="package-positioning">
    <div>
      <p class="eyebrow">Reward site</p>
      <h2>Normalized reward estimation with modular learners.</h2>
      <p>
        <code>genpqr</code> estimates a normalized reward representation from
        logged behavior by combining a fitted behavior policy, Q evaluation,
        and an explicit normalization policy or anchor.
      </p>
    </div>
    <dl class="package-metric-panel">
      <div>
        <dt>Primary object</dt>
        <dd>Normalized reward functions</dd>
      </div>
      <div>
        <dt>Default path</dt>
        <dd>Native behavior cloning plus boosted FQE</dd>
      </div>
      <div>
        <dt>Audit trail</dt>
        <dd>Adapter status, anchors, serialization</dd>
      </div>
    </dl>
  </section>
  <section class="package-pillars" aria-label="genPQR package pillars">
    <article>
      <span>01</span>
      <h3>Fit behavior</h3>
      <p>Estimate policy probabilities or densities through native learners or optional adapters.</p>
    </article>
    <article>
      <span>02</span>
      <h3>Evaluate Q</h3>
      <p>Use FQE-backed continuation values as the bridge from behavior to reward.</p>
    </article>
    <article>
      <span>03</span>
      <h3>Export rewards</h3>
      <p>Return reward functions, report checks, and portable fitted artifacts.</p>
    </article>
  </section>
</div>

## What is estimated?

GenPQR estimates a normalized reward representation under a maximum-entropy or
Gumbel-shock style behavioral model. The normalization policy `mu` and
anchor `g(s)` are part of the estimand: they say which reward scale and offset
the package should report.

<div class="estimand-card estimand-card--genpqr">
  <p class="estimand-label">Utility signal implied by the fitted behavior policy</p>
  <pre class="math-display"><code>u(s,a)
= \log \hat\pi_0(a \mid s) - g(s)</code></pre>
</div>

GenPQR fits a continuation value for this utility signal:

<div class="estimand-card estimand-card--genpqr">
  <p class="estimand-label">Q value used to construct the reward</p>
  <pre class="math-display"><code>Q_u(s,a)
= \mathbb{E}\!\left[
    \sum_{t=0}^{\infty} \gamma^t u(S_t,A_t)
    \mid S_0=s,\ A_0=a
  \right]</code></pre>
</div>

The reported reward subtracts the normalization-policy continuation value and
adds back the anchor:

<div class="estimand-card estimand-card--genpqr">
  <p class="estimand-label">Normalized reward estimand</p>
  <pre class="math-display"><code>r_{\mu,g}(s,a)
= Q_u(s,a)
  - \mathbb{E}_{A \sim \mu(\cdot \mid s)}[Q_u(s,A)]
  + g(s)</code></pre>
</div>

This is a normalized reward estimand. A unique environment reward requires
additional identifying restrictions beyond logged behavior alone.

`result.reward_function.predict_reward(...)` returns this normalized reward for
queried state-action rows.

## Install

Core import only requires NumPy:

```bash
python -m pip install -e "packages/genpqr"
```

Install optional learners only when needed:

```bash
python -m pip install -e "packages/genpqr[fqe,torch]"
python -m pip install -e "packages/genpqr[imitation]"
python -m pip install -e "packages/genpqr[d3rlpy,scope-rl]"
```

## Minimal example

```python
from genpqr import GenPQRConfig, fit_genpqr

result = fit_genpqr(
    dataset=dataset,
    gamma=0.95,
    config=GenPQRConfig(policy="behavior_cloning_native", q="fqe_boosted"),
)

rewards = result.reward_function.predict_reward(states, actions)
```

## Data contract

| Object | Role |
| --- | --- |
| `TransitionDataset` | Validated row-wise workflow |
| `EpisodeDataset` | Trajectory-preserving workflow |
| `ActionSpaceSpec` | Explicit discrete or continuous action contract |
| `normalization_policy` | Defines the reward normalization target |
| Policy estimator | Supplies action probabilities or log densities |
| Q estimator | Supplies continuation values for reward estimation |
| `GenPQRResult` | Reward function, typed diagnostics, serialization |

## When to use it

- You have demonstrations or logged behavior and want a reward explanation.
- You need to compare BC/FQE, AIRL/FQE, GAIL/FQE, SCOPE-RL diagnostics, or
  DeepPQR anchor workflows behind one public interface.
- You want custom policy and Q estimators through public protocols.

## Reward-identification checks

- Confirm that the normalization policy or anchor is meaningful for the domain.
- For anchor workflows, review anchor counts, weighted anchor counts, and support
  flags before interpreting reward values.
- For continuous actions, review Monte Carlo normalization standard errors when
  normalization expectations are sampled.
- Behavior-policy probabilities or densities should be credible for observed,
  queried, normalization, and anchor actions.
- Optional adapters should report clear adapter status. A configured fallback is
  a different method choice, not an equivalent result.

## Method surface

| Workflow | Entry point | Use case |
| --- | --- | --- |
| Lightweight BC + FQE | `fit_genpqr` with `behavior_cloning_native` and `fqe_boosted` | Dependency-light first pass |
| AIRL/GAIL + FQE | `fit_genpqr` with imitation adapters | Adversarial imitation workflows |
| DeepGenPQR | `fit_deep_genpqr` | Neural workflow with AIRL and action-aware neural FQE |
| DeepPQR anchor | `DeepPQRAnchorQEstimator` or `q_mode="anchor_deeppqr"` | Anchor-action reward identification |
| Cross-fitting | `fit_genpqr_crossfit` | Out-of-fold reward predictions |
| Serialization | `result.save`, `load_genpqr_result` | Portable fitted artifacts |

## Papers

- [Modular Inverse Reinforcement Learning via Policy Estimation and Q-Evaluation](../papers/)
- [Efficient Inference for Inverse Reinforcement Learning and Dynamic Discrete Choice Models](../papers/)

## API links

- [Package README](https://github.com/Larsvanderlaan/rl-eval-kit/blob/main/packages/genpqr/README.md)
- [Docs source](https://github.com/Larsvanderlaan/rl-eval-kit/tree/main/packages/genpqr/docs)
- [Example gallery](https://github.com/Larsvanderlaan/rl-eval-kit/tree/main/packages/genpqr/examples)
