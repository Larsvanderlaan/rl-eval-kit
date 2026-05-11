# Causal OPE Benchmark User Guide

## Mental Model

The package has two layers:

- Problem generation returns `BenchmarkProblem(dataset, truth)`.
- Runner execution scores named estimators and writes a complete output bundle.

Estimators should only receive `LongitudinalDataset` or adapter outputs.
`TruthBundle` is for scoring and diagnostics only.

## Family Discovery

```python
from causal_ope_benchmark import list_families, describe_family, list_target_policies

for family in list_families():
    print(family.name, family.summary)

print(describe_family("clinic_dtr"))
print(list_target_policies("streamretain"))
```

## Generate One Problem

```python
from causal_ope_benchmark import make_benchmark_problem

problem = make_benchmark_problem(
    "clinic_dtr",
    sample_size=500,
    gamma=0.95,
    seed=12,
    target_policy="safety_constrained",
)

dataset = problem.dataset
print(dataset.states.shape, dataset.actions.shape, dataset.rewards.mean())
```

## Use Adapters

```python
from causal_ope_benchmark import (
    to_fqe_dataset,
    to_occupancy_ratio_dataset,
    to_scope_rl_logged_dataset,
)

fqe = to_fqe_dataset(problem.dataset, target_policy_expectation_mode="exact_discrete")
ratio = to_occupancy_ratio_dataset(problem.dataset)
scope = to_scope_rl_logged_dataset(problem.dataset)
```

## Run A Suite

```python
from causal_ope_benchmark import CausalOPEBenchmarkConfig, run_suite, load_results

config = CausalOPEBenchmarkConfig.for_profile("smoke", output_root="outputs/causal_ope_benchmark")
result = run_suite(
    config,
    sample_sizes=(120,),
    estimators=("direct_method", "ipw", "snipw", "linear_fqe", "oracle_diagnostic"),
)

bundle = load_results(result.output_dir)
print(bundle.results_path)
print(bundle.output_schema["result_schema_version"])
```

For automation or release checks, validate completed bundles before consuming
them:

```python
from causal_ope_benchmark import validate_output_bundle

validate_output_bundle(result.output_dir)
```

## Choose Difficulty

Difficulty is a first-class DGP choice, separate from run size. The public
labels are `easy`, `medium`, `hard`, and `realistic`:

```python
from causal_ope_benchmark import (
    describe_difficulty,
    list_difficulties,
    make_benchmark_problem,
)

print([spec.name for spec in list_difficulties()])
print(describe_difficulty("realistic").summary)

problem = make_benchmark_problem(
    "clinic_dtr",
    difficulty="hard",
    sample_size=700,
    target_policy="safety_constrained",
)
```

Primary difficulty cells are still stationary, time-invariant MDPs with
identified estimands. Hard cells use weak-but-nonzero overlap, smaller sample
sizes, nonlinear mechanisms, delayed effects, action constraints, dose/cost
effects, and observed missingness/censoring. Latent confounding, structural
positivity gaps, and nonstationarity are sensitivity rows only.

Difficulty is calibrated by method separation, not by making every estimator
fail. In healthy cells a suitable modern method should return a finite,
reasonably accurate estimate. Naïve StreamLift extrapolation, simple direct
regression, and underspecified FQE are expected to struggle as difficulty
increases; arm-stratified StreamLift g-computation and discounted occupancy
ratios are the sanity checks that the benchmark remains learnable.

Run the systematic stress harness when you want to calibrate whether standard
methods succeed or struggle:

```python
from causal_ope_benchmark import DifficultyStressStudyConfig, run_difficulty_study

config = DifficultyStressStudyConfig.for_scale("ci")
result = run_difficulty_study(
    config,
    difficulties=("easy", "medium"),
    families=("streamretain", "clinic_dtr"),
    methods=("direct_method", "neural_fqe_auto", "discounted_occupancy_neural_auto"),
)
print(result.readout_path)
```

CLI equivalent:

```bash
python -m causal_ope_benchmark.run --list-difficulties
python -m causal_ope_benchmark.run stress-test --scale ci --difficulty easy medium --families streamretain
```

Use `scale="ci"` for quick checks, `scale="audit"` for local calibration, and
`scale="exhaustive"` for overnight or cluster runs.

## Run Neural Calibration

Calibration answers a different question than the main leaderboard: are the
reasonable cells estimable by well-tuned neural FQE and neural discounted
occupancy ratios?

```python
from causal_ope_benchmark import CalibrationStudyConfig, run_calibration

config = CalibrationStudyConfig.for_preset(
    "core-lite",
    output_root="outputs/causal_ope_benchmark",
)
result = run_calibration(config)
print(result.readout_path)
```

The `proxy` tuning track uses only package tuning criteria. The `oracle` track
selects among already-fitted candidates with sealed truth and is diagnostic
only. Use the oracle gap to decide whether a hard cell reflects proxy-tuning
failure, model-class limitations, or genuine overlap/difficulty.

From the CLI:

```bash
python -m causal_ope_benchmark.run calibrate --preset smoke --dry-run
python -m causal_ope_benchmark.run calibrate --preset core-lite
```

Calibration outputs are:

- `calibration_results.csv`
- `calibration_summary.csv`
- `calibration_candidates.csv`
- `calibration_manifest.json`
- `calibration_readout.md`

## Canonical OPE Profiles

Use `profile="paper"` for the main benchmark suite in papers about FQE,
IPW/occupancy-ratio, and doubly robust OPE under observed-data assumptions. It
uses `streamretain` and `clinic_dtr` only, with three scenario cells:
`clean_randomized_good_overlap`, `observed_moderate_overlap`, and
`observed_weak_overlap`.

Those cells are deliberately free of latent confounding, structural positivity
gaps, informative missingness, informative censoring, and nonstationarity.
`StreamLift` is a separate short-panel forecasting track, and opt-in
sensitivity scenarios should be reported separately from the canonical
leaderboard.

For StreamLift, start with `streamlift_stratified_gcomp` rather than generic
FQE. It treats the simulator as a stationary MDP, fits public one-step
transition/reward/terminal models separately inside logged action arms, and
rolls them forward under the public `campaign_mode` and `campaign_length`.
This is the baseline meant to test whether the short panel contains enough
one-transition information for long-horizon causal forecasting.

Finite forecast horizons are controlled by `forecast_horizons`. To also score
discounted infinite-horizon effects, set
`streamlift_include_infinite_horizon=True` on `CausalOPEBenchmarkConfig` or use
`--streamlift-include-infinite-horizon` in the CLI. Infinite-horizon truth is
approximated with a long discounted rollout, controlled by
`streamlift_infinite_horizon_max_steps`.

Use `sensitivity_scenarios_for_profile(...)` when you intentionally want cells
that stress missingness, censoring, nonstationarity, or latent confounding.
Those scenarios are diagnostic and should not be averaged into canonical OPE
leaderboards.

## Gym And SCOPE-RL Compatibility

```python
from causal_ope_benchmark import make_gym_env, FixedPolicyWrapper

env = make_gym_env("streamretain", target_policy="moderate", seed=0)
obs, info = env.reset(seed=0)

policy = FixedPolicyWrapper("streamretain", "moderate")
action, pscore = policy.sample_action_and_output_pscore(obs.reshape(1, -1))
```

`StreamLift` is intentionally not a native Gym environment. It is a short-panel
causal forecasting problem. Its SCOPE-RL export is available for shape
compatibility and is marked `panel_only`.

## Output Contract

Every completed run writes:

- `results.csv`
- `summary.csv`
- `tuning_results.csv`
- `manifest.json`
- `diagnostics.json`
- `output_schema.json`
- `benchmark_readout.md`

Difficulty stress runs write the analogous calibration bundle under
`outputs/causal_ope_benchmark/difficulty/<scale>/`:

- `difficulty_results.csv`
- `difficulty_summary.csv`
- `difficulty_candidates.csv`
- `difficulty_manifest.json`
- `difficulty_readout.md`

Use `manifest.json` and `output_schema.json` for automation. Use
`benchmark_readout.md` for human review.

See `docs/release_checklist.md` for the package build, install, and PyPI
release checklist.

## Optional Dependencies

Optional dependencies are only imported when the corresponding feature is used.

- FQE estimators require the `fqe` package and, for neural FQE, Torch.
- Boosted FQE requires LightGBM.
- Native Gym wrappers work with Gymnasium when installed and provide a minimal
  fallback surface for lightweight tests.
- EpiCare requires the external EpiCare package plus Gym or Gymnasium.
