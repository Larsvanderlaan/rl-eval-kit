# Low-Rank Operator SBV

Low-Rank Operator Supervised Bellman Validation (SBV) is a post-hoc selector
for many fitted FQE/Q candidates under one fixed target policy. It avoids
training one Bellman regression model per candidate by learning a shared
conditional operator for the candidate next-value matrix.

For each transition and candidate, SBV builds

```text
H[i, m] = (1 - done_i) * E_{a' ~ pi(. | next_obs_i)} Q_m(next_obs_i, a')
```

The terminal mask is applied only when creating the operator target. Scoring
does not multiply by observed `done` again:

```text
T_hat[i, m] = r_hat(obs_i, action_i) + gamma * H_hat[i, m]
score[m] = mean_i (Q_m(obs_i, action_i) - T_hat[i, m])^2
```

## Data Splits

Use trajectory-level splits only. A typical workflow is:

```python
from fqe import TransitionDataset, split_by_episode_ids

dataset = TransitionDataset(obs, actions, rewards, next_obs, done, episode_id, timestep)
splits = split_by_episode_ids(dataset, {"D_Q": 0.5, "D_B": 0.3, "D_score": 0.2}, seed=123)
b_splits = split_by_episode_ids(splits["D_B"], {"D_B_train": 0.8, "D_B_val": 0.2}, seed=124)
```

`D_B_train` fits the SVD basis and learned operator. `D_B_val` selects rank and
early stopping. `D_score` is used only after validator training to select the
final FQE candidate.

## Minimal Example

```python
from fqe import FQECandidate, LowRankOperatorSBVValidator

candidates = [FQECandidate("iter_010", q10, fqe_iteration=10),
              FQECandidate("iter_020", q20, fqe_iteration=20)]

validator = LowRankOperatorSBVValidator(gamma=0.99, ranks=[4, 8, 16], seed=123)
result = validator.fit_score(
    candidates,
    b_splits["D_B_train"],
    b_splits["D_B_val"],
    splits["D_score"],
    target_policy,
    action_space,
    initial_states=initial_states,
)

print(result.selected_candidate_id)
print(result.rows)
```

For discrete actions, pass either an integer action count or an array of encoded
action rows such as `np.eye(n_actions)`. If a Q model returns all action values
from `q(obs)`, SBV uses the exact policy expectation; otherwise it enumerates
actions and calls `predict_q(obs, action)` or `predict(obs, action)`.

## Generative Baseline

`GenerativeBellmanValidator` fits a conditional density model
`p(next_obs, reward, done | obs, action)` on the same `D_B_train/D_B_val`
splits. The default model is for vector observations: diagonal Gaussian dynamics
over `next_obs - obs`, Gaussian reward, and Bernoulli termination. It reports
Monte Carlo and deterministic mean-backup Bellman validation scores.

Image or categorical observation models require a custom adapter; the default
baseline raises a clear unsupported-observation error instead of fitting an
inappropriate pixel Gaussian.

## When To Prefer SBV

Low-rank operator SBV is usually preferable when there are many FQE checkpoints
or hyperparameter candidates and the expensive part is validating all of them.
It trains one reward/coefficient model per rank/config, not one Bellman model
per candidate. The generative baseline is useful as a model-based comparator
and diagnostic, but it can be harder to fit well than the supervised Bellman
operator when the transition density is complex.

For neural-network size selection on fast Gym-style benchmarks, held-out TD can
be a strong cheap selector. The shipped conservative rule is
`select_td_with_sbv_audit`: TD chooses the candidate, while SBV only reports
agreement, yellow disagreement, or red strong disagreement. The audit does not
automatically veto TD, because the Gym NN-width evidence showed SBV
disagreements were more often false alarms than improvements.

The `low_rank_sbv_cv` and `td_sbv_audit_cv` benchmark rows tune the SBV
operator regression itself by episode-level CV on `D_B` only. The final
candidate is still selected only on `D_score`.

```python
from fqe import LowRankOperatorSBVValidator, select_td_with_sbv_audit

lowrank_result = LowRankOperatorSBVValidator(gamma=0.99, ranks=[4, 8, 16]).fit_score(...)
audit_result = select_td_with_sbv_audit(lowrank_result.rows, candidates)

print(audit_result.selected_candidate_id)          # TD one-SE winner
print(audit_result.diagnostics["td_sbv_audit_status"])  # green/yellow/red
```

## CLI

```bash
fqe-compare-validators --config configs/fqe_validation.yaml
```

The config may be JSON or YAML. If omitted, the CLI runs a tiny built-in smoke
example and writes `validator_comparison.csv` plus diagnostics JSON.

For a ground-truth model-selection experiment on synthetic MDPs:

```bash
fqe-sbv-experiment \
  --seeds 0 1 2 3 4 5 6 7 \
  --n-episodes 900 \
  --include-generative \
  --output-dir outputs/fqe_sbv_experiment_many
```

This writes per-seed selector rows and a summary JSON comparing low-rank SBV,
naive one-sample TD, the generative Bellman validator, and direct multi-output
SBV where the candidate count is small enough.

For a fast Gym neural-width model-selection benchmark:

```bash
fqe-sbv-gym-nn-size \
  --seeds 0 1 2 \
  --sample-sizes 600 1200 2400 \
  --reward-noise-sds 0 3 6 \
  --include-generative \
  --output-dir outputs/fqe_sbv_gym_nn_size
```

This trains a small grid of neural FQE widths on `D_Q`, tunes SBV operators on
`D_B_train/D_B_val`, and evaluates selectors against independent target-policy
rollout values. It is intended as a realistic smoke benchmark, not an oracle
selection rule for production.
