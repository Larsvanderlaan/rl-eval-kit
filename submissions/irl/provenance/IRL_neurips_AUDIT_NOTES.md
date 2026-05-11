# Modular Audit Notes

This note records the debugging/checking plan, the intended role of each module,
the main verified properties, and the remaining risks.

## Debugging Plan

The audit is organized in the order that mathematical dependencies flow through
the pipeline:

1. data generation and experiment split
2. policy estimation
3. Q evaluation
4. reward recovery
5. DeepPQR baseline
6. experiment-story sanity checks

This ordering is important because downstream failures can easily be caused by
upstream bugs.

## 1. Data Generation

File:

- `data_generation.py`

Intended role:

- generate a synthetic infinite-horizon MDP in the DeepPQR anchor-action style
- enforce the normalization condition `sum_a mu(a|s) r(s,a) = g(s)`
- generate behavior trajectories from a soft energy-based policy

Main checks:

- `normalize_reward_matrix` correctly enforces the statewise normalization.
- with `mu = make_anchor_mu(0, n_actions)` and `g = make_zero_g()`, the anchor
  action reward is exactly zero in the noiseless reward matrix.
- train/test data must share the same underlying MDP parameters.

Confirmed bug fixed:

- train and test datasets were previously generated from different MDP
  parameters because `seed` controlled both the trajectory RNG and the sampled
  simulator parameters. This made evaluation invalid.
- fix: `generate_deeppqr_style_data(..., simulation_parameters=...)` now allows
  test data to reuse the training MDP parameters, and
  `experiment_runner.py` now uses that path.

Important remaining risk:

- the simulator planner is still an approximate soft-Q solver, not an exact
  one. Its Bellman residual is larger than ideal. This likely hurts DeepPQR
  more than GenPQR because DeepPQR relies more directly on the exact
  energy-policy/Q identity.

## 2. Policy Estimation

File:

- `policy_estimation.py`

Intended role:

- provide modular policy estimators with a shared interface
- expose `predict_proba`, `sample_actions`, and when relevant a Q/logit surrogate

Main checks:

- `EstimatedPolicy.predict_proba` now applies renormalized probability clipping
  when configured. This stabilizes both GenPQR and DeepPQR symmetrically.
- default clipping is now `0.01 / 0.99` for fitted policies.
- behavior cloning is a straightforward multiclass baseline.
- MaxEnt-IRL is implemented as a softmax policy induced by a learned state-action
  score/Q surrogate.

Confirmed bug fixed:

- AIRL minibatches were shuffled, but the code was still pairing each batch with
  a contiguous slice of `next_states` and `dones`. This meant AIRL was often
  training on mismatched transitions.
- fix: the torch AIRL path now stores `next_states`, `dones`, and original
  indices in the `TensorDataset`, and uses the actual minibatch indices.

Remaining risk:

- the AIRL implementation is still a simplified in-house version. It is much
  better after the transition-alignment fix, but it should still be treated as a
  practical approximation rather than a reproduction-grade AIRL benchmark.

## 3. Q Evaluation

File:

- `q_evaluation.py`

Intended role:

- estimate `Q^mu_{u-g}` for GenPQR with either neural FQE or boosted FQE

Main checks:

- the FQE target uses the pseudo-reward passed by the caller, not the logged
  environment reward, which is correct for the GenPQR theorem.
- the continuation term is taken under the supplied evaluation policy.
- in GenPQR experiments, the evaluation policy is the normalization policy `mu`,
  not the behavior-policy estimate.

Status:

- no new structural bug was found in this module during the audit.

## 4. Reward Recovery

File:

- `reward_recovery.py`

Intended role:

- recover reward and continuation from the GenPQR identity

Verified formula:

- `r(s,a) = Q(s,a) - sum_a' mu(a'|s) Q(s,a') + g(s)`
- `v(s,a) = (log pi(a|s) - g(s) - Q(s,a)) / gamma`

Confirmed bug fixed earlier:

- the code originally used a generic Bellman subtraction
  `r = Q - gamma E_pi[Q(next)]`, which is not the GenPQR formula.
- it now uses the theorem from the paper.

## 5. DeepPQR Baseline

File:

- `deeppqr_baseline.py`

Intended role:

- estimate policy
- estimate state-only anchor Q-function on the anchor-action subset
- recover full state-action Q-function from the log-policy ratio identity
- regress the reward-step auxiliary function

Main checks:

- the extra full-Q regression that had been inserted earlier was removed.
- full Q is now recovered directly from the DeepPQR formula.
- the reward-regression target now correctly zeroes terminal continuations.

Confirmed bug fixed:

- comparison runs previously let DeepPQR fit its own separate policy, while
  GenPQR used another policy estimate. That confounded the first experiment.
- fix: `fit_deeppqr_baseline` now accepts an external policy estimate, and
  `experiment_runner.py` passes the same shared policy into both methods in the
  matched comparison path.

Important diagnosis:

- DeepPQR is extremely sensitive to any violation of the exact softmax/Q
  relationship.
- a previous simulator change introduced a probability floor by clipping and
  renormalizing the behavior policy. That stabilized support, but it broke the
  exact policy-ratio identity required by DeepPQR.
- fix: the direct behavior-policy floor was removed. Probability clipping now
  happens on the estimated policy side only.

Remaining risk:

- even after the fixes above, DeepPQR performance is still sensitive to the
  approximate quality of the simulator planner and the policy estimator.
- this is likely a mixture of real statistical fragility and residual simulator
  approximation error.

## 6. Experiment-Level Sanity Checks

Files:

- `experiments_neurips/experiment_runner.py`
- `experiments_neurips/EXPERIMENT_STUDY.md`

Main checks:

- the first experiment should compare GenPQR and DeepPQR using the same fitted
  policy estimator.
- effective sample size is now reported:
  total transitions, anchor-action count, and anchor fraction.
- the second experiment should not use an oracle-assisted SPL-GD baseline in the
  main table.

Important design correction:

- the current SPL-GD-style baseline in code is still too favorable because it is
  closer to an oracle-assisted appendix variant than a realistic linear DDC
  baseline.
- for the paper, the main comparison should use a practical linear baseline
  without privileged access to true planner objects.

## Current Bottom Line

The audit found and fixed several real implementation bugs:

- invalid train/test split across different MDPs
- AIRL transition misalignment under shuffled minibatches
- unfair policy-estimation mismatch between GenPQR and DeepPQR
- a simulator policy-floor distortion that broke the DeepPQR identity
- inconsistent numerical stabilization across methods

The main remaining risk is not an obvious coding bug, but the fact that the
simulator still uses an approximate planner. That approximation appears to hurt
DeepPQR more than GenPQR. This should be kept in mind when finalizing
Experiment 1.
