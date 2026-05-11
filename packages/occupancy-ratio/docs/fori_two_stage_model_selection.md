# FORI Two-Stage Model Selection

This page describes the FORI two-stage cross-validation selector exposed by
`occupancy_ratio.fori_model_selection`.

The selector is for model selection only. It does not implement a minimax,
witness, or critic game estimator. Its primary score is held-out adjoint
Bellman error, abbreviated ABE in logs and result tables.

## Setting

FORI estimates the normalized infinite-horizon discounted occupancy ratio

```text
omega*(x) = d_pi,gamma(x) / dnu(x)
d_pi,gamma = (1 - gamma) sum_t gamma^t Law_pi(X_t)
```

for state-action rows `X = (S, A)` sampled from the logged/base distribution
`nu`. With `X_plus = (S_plus, A_plus)` and `A_plus ~ pi(. | S_plus)`, the
adjoint fixed point is

```text
omega = (1 - gamma) omega0 + gamma c_pi M_pi omega
M_pi omega(x) = E[omega(X) | X_plus = x].
```

The first-stage nuisances are:

- `omega0(x)`, the initial state-action density ratio;
- `c_pi(x)`, the one-step target-policy pushforward density ratio.

In this repository, `c_pi` is the existing direct one-step ratio path
(`one_step_ratio_mode="direct"`).

## Stage 1

`FirstStageDensityRatioCV` tunes `omega0` and `c_pi` using standard held-out
density-ratio risk. LSIF/uLSIF is the default. Logistic odds ratios are
supported through the existing nuisance infrastructure.

For the initial ratio:

- if `initial_states` and `initial_actions` are supplied, the joint initial
  state-action ratio is fitted;
- if only `initial_states` are supplied, the factored state-source fallback is
  fitted and multiplied by the target action ratio;
- if no initial states are supplied, the initial/source term is the
  backward-compatible constant one.

No rows from the final score split are used for nuisance tuning or fitting.

## Stage 2

`LowRankAdjointBellmanCV` scores candidate FORI ratios. For candidates
`omega_1, ..., omega_M`, it forms the pseudo-outcome matrix

```text
W[i, m] = omega_m(X_i)
```

on the backup training split. It centers `W`, computes a truncated SVD/PCA
basis, and trains one compressed adjoint backup regressor

```text
X_plus -> z_hat in R^rank.
```

This compression amortizes the candidate backup regressions. It is analogous to
low-rank primal Bellman CV/SBV, but applied to FORI's adjoint equation:

```text
primal: Q - T_hat Q
FORI:   omega - B_hat^*_pi omega
```

Rank and backup-regressor hyperparameters are selected using the backup
validation split only. The historical rank criterion is held-out adjoint
regression MSE. For ablations, `backup_rank_selection_metric="abe_val"` selects
the rank by a backup-validation adjoint Bellman residual, and
`"direct_agreement"` compares the low-rank backup to the direct multi-output
backup when the candidate count is small enough. None of these rank-selection
criteria may use the final score split.

Final scoring uses the score split only:

```text
ABE_m = mean_i [
  omega_m(X_i)
  - {
      (1 - gamma) omega0_hat(X_i)
      + gamma c_pi_hat(X_i) M_hat omega_m(X_i)
    }
]^2.
```

The learned adjoint backup is evaluated at current `X_i` for this residual,
not at `X_plus_i`.

## Splitting And Leakage

All splits are trajectory-level splits by `episode_ids`:

- nuisance CV/trainval;
- FORI candidate training;
- backup training;
- backup validation;
- final score.

The final score split is used only for ABE scoring and trajectory bootstrap
standard errors. The implementation hard-checks split disjointness and checks
candidate `trained_on_episode_ids` metadata against the score split.

## Terminal Handling

Two conventions are supported.

`absorbing_state` maps terminal successor rows to a configured absorbing
observation/action and keeps continuation mass equal to one. This is preferred
when the representation is available.

`live_only_submarkov` uses nonterminal successor mass only. In that mode,
continuation weights are used for `c_pi` numerator fitting and adjoint backup
training. The selector does not normalize away missing continuation mass unless
the caller chooses a different convention upstream.

`terminal_mode="auto"` uses absorbing-state handling when an adapter is
configured; otherwise it falls back to `live_only_submarkov` with a warning.

## Example

```python
from occupancy_ratio import FORICandidateSpec, FORITwoStageCV, FORITwoStageCVConfig

selector = FORITwoStageCV(
    FORITwoStageCVConfig(
        backup_regressor_backend="lightgbm",
        low_rank_ranks=(4, 8, 16, 32),
        n_bootstrap=200,
    )
)

result = selector.fit(
    states=states,
    actions=actions,
    next_states=next_states,
    target_actions=target_actions,
    target_next_actions=target_next_actions,
    gamma=0.99,
    episode_ids=episode_ids,
    initial_states=initial_states,
    initial_actions=initial_actions,
    candidates=[
        FORICandidateSpec(candidate_id="stable", model=stable_model),
        FORICandidateSpec(candidate_id="less_clipped", model=less_clipped_model),
    ],
)

rows = result.candidate_rows()
selected = result.selected_candidate_id  # one-standard-error recommendation by default
raw_min = result.selected_min_score_candidate_id
```

For large candidate grids, pass cached predictions and configure
`prediction_memmap_dir`, `candidate_block_size`, and `transition_batch_size`.

## CLI

```bash
python scripts/fori_two_stage_model_selection.py \
  --config configs/fori_two_stage_cv.json
```

The CLI reads JSON by default and YAML when PyYAML is installed. The dataset
loader expects an `.npz` file containing arrays such as `states`, `actions`,
`next_states`, `target_actions`, `target_next_actions`, and `episode_ids`.

## Diagnostics

Result rows include ABE, bootstrap standard errors, final score, selected flags,
rank, backup validation MSE, optional direct multi-output ABE, ratio moments,
ESS, quantiles, split sizes, runtime, and first-stage IDs. They also report
diagnostic-only near-uniform collapse flags and policy-action shift summaries;
these diagnostics do not veto or penalize candidates by default.

The one-standard-error rule picks the simplest candidate whose final score is
within one bootstrap SE of the minimum-score candidate. This is the default
recommendation rule: `result.selected_candidate_id` follows one-SE, while
`result.selected_min_score_candidate_id` keeps the raw minimum-ABE choice for
diagnostics. Set `selection_rule="min_score"` to make `selected_candidate_id`
use the raw minimizer instead.

Both one-SE variants are reported. `one_se_method="marginal"` is the historical
rule based on the raw minimum candidate's marginal bootstrap SE.
`one_se_method="paired"` instead uses trajectory-bootstrap SEs of paired score
differences versus the minimum-score candidate. The result includes
`selected_one_se_marginal_candidate_id` and `selected_one_se_paired_candidate_id`
so benchmark reports can compare the ablation without redefining the primary
squared ABE criterion.

Reward/OPE-focused Bellman-GMM tuning in the broader tuning suite is a
deployment comparator, not a replacement for low-rank ABE as a FORI
ratio-selection criterion. Reports should keep ratio-selection, value/OPE
selection, and deployment recommendation separate.

## Limitations

The v1 native candidate path fits each boosted/neural candidate on the FORI
training split. High-scale workflows should pass cached/model-backed candidates
and use memmap prediction matrices. The low-rank evaluator itself trains one
compressed backup model with rank-sized outputs, not one backup model per
candidate. Some matrix paths still materialize dense arrays after prediction;
avoid advertising truly out-of-core `1e7 x 5000` workflows until those scoring
and SVD paths are blockwise end to end.
