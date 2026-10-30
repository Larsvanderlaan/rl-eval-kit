# Complementary-Source Large-Action Experiment

## Decision

The confirmatory experiment supports the proposed data-fusion example. In this
20-action synthetic setting, estimated data-fusion GenPQR becomes more accurate
as the independent state-outcome source grows. With at least 250 auxiliary
trajectories, it improves on behavior-only anchor GenPQR in paired seed-level
comparisons. Every data-fusion cell improves substantially on DeepPQR.

These conclusions apply to the specified synthetic family. The experiment does
not establish performance under source-specific covariate shift or outside the
shared-policy data-fusion model.

## Design

- Actions: 20.
- State dimension: 5.
- Horizon: 10.
- Discount: 0.9.
- Behavioral source: 2,500 trajectories, or 25,000 transitions, containing
  `(S, A, S')` but no rewards.
- Auxiliary source sizes: 100, 250, 500, 1,000, 2,500, and 5,000 trajectories.
  Each trajectory contributes 10 `(S, Y)` observations; actions are withheld.
- Test source: 300 independent trajectories.
- Confirmatory seeds: 920000--920099.
- Replicates: 100.

All sources share the same structural MDP parameters within a seed but use
independent trajectories. The behavior and reference policies coincide. The
true reward satisfies `r(s, 0)=0`, while

```text
g(s) = sum_a pi(a | s) r(s, a) = E[Y | S=s].
```

The Gaussian initial-state sampler is truncated to the simulator's declared
state cube. This is a support correction to the old simulator: extremely rare
out-of-cube initial draws previously broke the intended exact anchor constraint.

## Confirmatory Results

Entries are means and 95% normal intervals over 100 independent seeds. Reward
MSE is weighted by the true behavior policy over all action-specific rewards.
The oracle-nuisance row uses true `(pi, g)` but still estimates the FQE stage; it
is an upper-bound diagnostic, not a deployable method.

| Method | Auxiliary trajectories | Behavior-weighted reward MSE | Uniform-action reward MSE | Uniform-action correlation | Oracle diagnostic: g RMSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| DeepPQR | 0 | 2.253 +/- 0.089 | 2.247 +/- 0.089 | 0.560 +/- 0.025 | -- |
| DeepPQR, anchor projected | 0 | 2.174 +/- 0.086 | 2.187 +/- 0.087 | 0.566 +/- 0.025 | -- |
| Anchor GenPQR | 0 | 0.443 +/- 0.064 | 0.454 +/- 0.064 | 0.790 +/- 0.011 | -- |
| Data-fusion GenPQR | 100 | 0.452 +/- 0.073 | 0.471 +/- 0.074 | 0.795 +/- 0.009 | 0.388 +/- 0.035 |
| Data-fusion GenPQR | 250 | 0.319 +/- 0.047 | 0.337 +/- 0.048 | 0.841 +/- 0.006 | 0.291 +/- 0.026 |
| Data-fusion GenPQR | 500 | 0.259 +/- 0.035 | 0.277 +/- 0.036 | 0.864 +/- 0.006 | 0.230 +/- 0.021 |
| Data-fusion GenPQR | 1,000 | 0.234 +/- 0.031 | 0.252 +/- 0.032 | 0.876 +/- 0.006 | 0.181 +/- 0.017 |
| Data-fusion GenPQR | 2,500 | 0.217 +/- 0.027 | 0.233 +/- 0.028 | 0.884 +/- 0.007 | 0.137 +/- 0.013 |
| Data-fusion GenPQR | 5,000 | 0.210 +/- 0.026 | 0.227 +/- 0.027 | 0.888 +/- 0.007 | 0.114 +/- 0.011 |
| GenPQR with oracle nuisances | -- | 0.097 +/- 0.017 | 0.105 +/- 0.018 | 0.963 +/- 0.003 | 0 |

At 100 auxiliary trajectories, data-fusion and anchor GenPQR are statistically
indistinguishable: their paired behavior-MSE difference is
`0.008 +/- 0.026`, with fusion winning 54% of seeds. At 250 trajectories, the
paired difference becomes `-0.125 +/- 0.024`, and fusion wins 96% of seeds.
At 5,000 trajectories, the difference is `-0.233 +/- 0.042`, and fusion wins
all 100 seeds.

Relative to the mean DeepPQR MSE, the 5,000-trajectory fusion estimate reduces
behavior-weighted MSE by 90.7%. Relative to anchor GenPQR, it reduces MSE by
52.6%. Enforcing the known anchor exactly on DeepPQR's final predictions lowers
its MSE by only 3.5%, so anchor violation does not explain the main gap.

The auxiliary regression behaves as expected: mean `g` RMSE falls monotonically
from 0.388 to 0.114 across the six source sizes. Within cells, its correlation
with reward MSE ranges from 0.92 to 0.96. The oracle-nuisance result shows that
normalization-nuisance error remains material even at the largest auxiliary
sample.

The mean reward-MSE curve is monotone, but the seed-level curve is not
universally monotone: 44% of seeds improve at every auxiliary-size step, and
68% improve from 2,500 to 5,000 trajectories. The results should therefore be
described as an average sample-size trend. The largest small-source outlier,
seed 920051, has behavior MSE 2.17 and `g` RMSE 0.95 at 100 trajectories, but
improves to MSE 0.65 at 5,000 trajectories. Median and trimmed-mean checks
preserve the reported method ordering.

## Runtime and Integrity

- Raw rows: 1,000.
- Complete seeds: 100.
- Failed or nonfinite rows: 0.
- Duplicate `(seed, method, auxiliary size)` cells: 0.
- Mean anchor count: 1,556 of 25,000 transitions (6.22%).
- Minimum anchor count: 1,342.
- Wall time with four workers: 28.4 minutes.
- Source fingerprint:
  `11f9fef809f34a687b0ad2b80597d18a39fc52f3f55384622c33964555022949`.

Mean fit-only times, excluding common world generation, were 5.2 seconds for
the largest data-fusion cell, 5.0 seconds for anchor GenPQR, and 37.0 seconds
for DeepPQR.

## Reproduction

From the repository root:

```bash
MPLCONFIGDIR=/tmp/irl_mpl \
XDG_CACHE_HOME=/tmp/irl_xdg \
.venv/bin/python \
submissions/irl/experiments/conference_genpqr/repro/experiments/data_fusion_large_action_runner.py \
  --stage confirmatory \
  --jobs 4 \
  --output-dir \
  submissions/irl/experiments/conference_genpqr/repro/experiments/outputs/data_fusion_confirmatory_20260725
```

Primary artifacts:

- `experiments/outputs/data_fusion_confirmatory_20260725/results.csv`
- `experiments/outputs/data_fusion_confirmatory_20260725/summary.csv`
- `experiments/outputs/data_fusion_confirmatory_20260725/paired_contrasts.csv`
- `experiments/outputs/data_fusion_confirmatory_20260725/reward_mse_vs_outcome_sample.png`
- `experiments/outputs/data_fusion_confirmatory_20260725/g_rmse_vs_outcome_sample.png`
- `experiments/outputs/data_fusion_confirmatory_20260725/manifest.json`

## Paper Caveats

- The paper should say that neither source alone identifies the data-driven
  normalization pair `(pi, g)`. The behavior source alone still identifies the
  reward under the known zero-anchor restriction used by DeepPQR.
- DeepPQR's anchor fixed-point stage uses only anchor-action transitions; its
  final continuation regression uses all actions. Avoid saying that the entire
  method discards non-anchor observations.
- The manuscript's current comparison paragraph states that DeepPQR directly
  estimates GenPQR's anchored value. The implemented DeepPQR instead estimates
  the entropy-regularized anchor Q-function, which can be transformed into the
  GenPQR quantity. That paragraph should be corrected separately.
