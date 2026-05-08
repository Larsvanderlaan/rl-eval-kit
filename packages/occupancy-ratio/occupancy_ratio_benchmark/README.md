# Occupancy Ratio Benchmark

Reproducible diagnostics for boosted-tree and neural discounted
occupancy-ratio estimators, with optional official Google Research DualDICE
baselines.

## Quick Run

```bash
python -m occupancy_ratio_benchmark.run \
  --profile smoke \
  --output-root outputs/occupancy_ratio_benchmark \
  --no-google-dualdice
```

Profiles are `smoke`, `medium`, `full`, `overnight`, and `dualdice-paper`. The
`boosted_tree` and `neural_network` estimator groups expand through
configurable presets such as `squared`, `huber`, `stable`, `transition_norm`,
`crossfit2`, `calibrated`, `bellman_moment_calibrated`,
`stable_logistic_nuisance`, `google_parity`, and `auto`. Result rows include relative MSE, log-ratio RMSE, L1/TV-style ratio
errors when ratio truth is available, OPE error, ESS, clipping/negative-weight
rates, first-stage diagnostics, validation loss, Huber delta, and runtime.

For a medium linear-Gaussian comparison of stabilized defaults:

```bash
python -m occupancy_ratio_benchmark.run \
  --profile medium \
  --settings linear_gaussian \
  --estimators oracle boosted_tree neural_network google_dualdice_neural \
  --boosted-estimator-presets squared huber stable transition_norm calibrated \
  --neural-estimator-presets squared huber stable transition_norm calibrated \
  --output-root outputs/occupancy_ratio_medium_linear_gaussian
```

Add `--tune-cv` to tune nuisance and iterative occupancy settings. Boosted CV
uses the package's LSIF nuisance scores and composite occupancy score; neural CV
uses the neural helper's fold validation history.

The `bellman_moment_calibrated` preset applies post-hoc backward occupancy
Bellman-moment calibration to the final occupancy weights. It learns
nonnegative score-bin multipliers with shrinkage and optional weight caps; this
is a conservative supported-moment adjustment and does not recover unsupported
target occupancy mass. A coverage sweep can be run by varying the existing
linear-Gaussian policy shift:

```bash
python -m occupancy_ratio_benchmark.run \
  --profile smoke \
  --settings linear_gaussian \
  --estimators oracle boosted_tree neural_network google_dualdice_neural \
  --boosted-estimator-presets stable bellman_moment_calibrated \
  --neural-estimator-presets stable bellman_moment_calibrated \
  --linear-gaussian-policy-shifts 0.25 1.0 2.0 \
  --seeds 0 \
  --sample-sizes 300 \
  --gammas 0.9 \
  --output-root outputs/occupancy_ratio_bellman_calibration_smoke \
  --no-plots
```

The calibration utilities are also exported for direct post-hoc use:

- `calibrate_occupancy_bellman_binning`: fit bin multipliers and return
  calibrated weights.
- `occupancy_bellman_calibration_diagnostics`: compare before/after Bellman
  residuals, ESS, tails, and score-bin contribution norms without oracle truth.
- `plot_occupancy_bellman_calibration_diagnostics`: write a compact diagnostic
  figure.
- `estimate_ope_bellman_control_variate`: report a Bellman-moment
  control-variate OPE sensitivity estimate beside the raw weighted estimate.

Diagnostics include `calibration_recommendation`, one of `apply`, `neutral`, or
`do_not_apply`. The recommendation is non-oracle and conservative: it suggests
applying calibration only when Bellman residual reduction is meaningful and the
ESS/tail costs stay within tolerance.

## Optional Tabular Benchmarks

Install the optional tabular loaders with:

```bash
python -m pip install -e "packages/occupancy-ratio[tabular-benchmark]"
```

The `medium`, `full`, and `overnight` profiles include optional tabular
settings. `openml_contextual_bandit` and `openml_finite_mdp` use OpenML task
features with synthetic behavior/target policies and ratio truth;
`obp_logged_bandit` uses logged OBP propensities with a synthetic target policy;
`minari_pointmaze` and `minari_minigrid` use lightweight Minari/D4RL datasets
without oracle ratio truth. Missing packages, unavailable local data, or failed
downloads produce skipped rows with `skip_reason` rather than import-time
failures.

Useful controls are:

```bash
python -m occupancy_ratio_benchmark.run \
  --profile medium \
  --settings openml_contextual_bandit openml_finite_mdp \
  --openml-task-ids 31 37 54 \
  --openml-max-tasks 2 \
  --tabular-state-cap 256 \
  --estimators oracle boosted_tree \
  --boosted-estimator-presets stable stable_logistic_nuisance auto \
  --no-google-dualdice \
  --no-plots
```

## Neural Estimator

The additive PyTorch estimator can be run beside the boosted-tree estimator:

```bash
python -m occupancy_ratio_benchmark.run \
  --profile smoke \
  --estimators oracle neural_network \
  --no-google-dualdice
```

`boosted_tree_stable` is the recommended boosted default. Logistic density-ratio
nuisances are available as an opt-in stability check:

```bash
python -m occupancy_ratio_benchmark.run \
  --profile smoke \
  --estimators oracle boosted_tree \
  --boosted-estimator-presets huber stable stable_logistic_nuisance \
  --no-google-dualdice \
  --no-plots
```

The `logistic_nuisance` and `stable_logistic_nuisance` presets use LightGBM's
binary logistic objective for action and transition density ratios, then apply
the usual nonnegative/capped/tempered post-processing. LSIF remains the default.
Use `--boosted-density-ratio-loss logistic` to make all boosted nuisance presets
use logistic ratios, and `--boosted-logistic-logit-clip` to change the odds cap.

`boosted_tree_auto` compares Huber and stable boosted settings on validation and
stability diagnostics and reports `selected_preset` and `selection_score`.
`neural_network_auto` compares stable, calibrated, and stable-logistic neural
settings without using oracle ratio or target-value diagnostics. The
`neural_network_google_parity` preset uses stable neural occupancy with a
`(256, 256)` ReLU MLP to match the official Google DualDICE critic scale.

Runs write `results.partial.csv` and `tuning_results.partial.csv` as they go and
resume from those files by default. Pass `--no-resume` for a clean rerun. Smoke
runs default to a 120 second per-estimator timeout; medium, full, and overnight
use 600 seconds. Override with `--estimator-timeout-sec`.

## Modern-Control Defaults Sweep

The `overnight` profile adds Gymnasium continuous-control settings
(`gym_pendulum`, `gym_mountain_car_continuous`, `gym_halfcheetah`, and
`gym_hopper`) to the controlled truth settings. These rows do not have oracle
ratio truth; they instead estimate target-policy value by Monte Carlo rollouts
and leave ratio-error columns blank.

```bash
python -m occupancy_ratio_benchmark.run \
  --profile overnight \
  --output-root /tmp/occupancy_default_overnight \
  --external-repo-path /tmp/google-research \
  --no-plots

python -m occupancy_ratio_benchmark.defaults_report \
  /tmp/occupancy_default_overnight/overnight/results.csv
```

## Neural Estimator

`neural_network_stable` uses projection, fixed-point damping, clipped
pseudo-outcomes, and scalar nuisance calibration. `neural_network_transition_norm`
also enables transition-cache normalization. The neural path uses gradient
updates for the action ratio, transition ratio, and occupancy fixed-point
regression, with only a few mini-batch updates per fixed-point refresh by
default in smoke runs.

## External DualDICE

The external comparator is the official Google Research implementation:

- repository: `https://github.com/google-research/google-research`
- neural module: `policy_eval.dual_dice.DualDICE`
- GridWalk module: `dual_dice.algos.dual_dice.TabularDualDice`
- default local path: `/tmp/google-research`

Smoke runs skip Google DualDICE if TensorFlow, TensorFlow Addons, or the repo is
missing. Full and overnight runs fail fast when Google DualDICE is requested
but unavailable.

## Outputs

Each run writes to `<output-root>/<stage>/`:

- `results.csv`: one row per setting, seed, estimator, sample size, and gamma
- `summary.csv`: grouped means and standard deviations
- `winner_table.csv`: best non-oracle estimator per benchmark cell
- `tuning_results.csv`: CV candidate scores when `--tune-cv` is enabled
- `diagnostics.json`: preflight and failure details
- `manifest.json`: exact benchmark config and dependency versions
- `defaults_report.md`: optional report from `occupancy-ratio-defaults-report`
- plots for relative MSE, log RMSE, OPE error, ESS, tail weights, and runtime
  when Matplotlib is installed

## Google DualDICE Paper GridWalk

The original DualDICE paper GridWalk benchmark remains available directly:

```bash
python -m occupancy_ratio_benchmark.dualdice_grid \
  --output-root outputs/occupancy_ratio_benchmark_google_paper \
  --seeds 0 1 2 \
  --alphas 0.0 0.5 \
  --gammas 0.9 \
  --num-trajectories 20 \
  --max-trajectory-length 50 \
  --boosted-losses huber squared \
  --include-neural \
  --include-bellman-moment-calibration
```

This imports Google Research's `dual_dice` GridWalk environment and tabular
DualDICE solver, collects the same trajectory data, and evaluates the boosted
occupancy estimator with the paper's average step-reward metric.

It can also be requested through the main runner:

```bash
python -m occupancy_ratio_benchmark.run \
  --profile dualdice-paper \
  --include-dualdice-gridwalk \
  --gridwalk-alphas 0.0 0.5 0.8 \
  --external-repo-path /tmp/google-research
```
