# Causal OPE Benchmark

Realistic, dependency-light benchmarks for offline policy evaluation, long-term
causal forecasting, and dynamic treatment-regime evaluation.

This package is built for industry and causal-inference-style OPE problems, not
classic Gym/Atari leaderboards. It gives users sealed-truth simulators,
public estimator-visible datasets, adapters for common OPE workflows, and
structured runner outputs that are suitable for papers, regression tests, and
product screens.

## What This Adds

- `StreamLift`: short-panel streaming experiments scored on long-horizon
  retention and revenue effects, with a public Markov customer state and an
  arm-stratified stationary g-computation baseline. Finite forecast horizons
  are default, and discounted infinite-horizon effects can be enabled when
  desired.
- `StreamRetain`: subscription lifecycle OPE with contact, budget, fatigue,
  retention, and revenue diagnostics.
- `ClinicDTR`: transparent cardiometabolic dynamic treatment-regime OPE with
  survival, RMST, biomarkers, dose, toxicity, and censoring.
- `EpiCare`: optional external medical RL benchmark adapter loaded lazily
  through `gym.make("EpiCare-v0")` when explicitly requested.

`ClinicDTR` and `EpiCare` are complementary. ClinicDTR is the transparent
causal-OPE diagnostic setting with controllable knobs. EpiCare is opt-in
external comparability for healthcare RL/OPE.

## Canonical Paper Suite

For papers comparing FQE, IPW/occupancy-ratio, and doubly robust OPE methods,
use the `paper` profile as the main no-hidden-confounding suite:

```bash
PYTHONPATH=packages/fqe:packages/causal-ope-benchmark python -m causal_ope_benchmark.run \
  --profile paper \
  --output-root outputs/causal_ope_benchmark
```

The profile intentionally uses only `streamretain` and `clinic_dtr`, where the
public longitudinal data, behavior propensities, target policy probabilities,
initial states/actions, and action masks support standard offline RL/OPE
estimators. It crosses three canonical observed-data scenario cells:

- `clean_randomized_good_overlap`
- `observed_moderate_overlap`
- `observed_weak_overlap`

All built-in canonical profile cells avoid latent/unmeasured confounding,
structural positivity gaps, informative missingness, informative censoring, and
nonstationarity. `StreamLift` remains available as a separate short-panel
forecasting track, and latent or missing/censoring stressors are sensitivity
settings rather than headline leaderboard cells.

## First 5 Minutes

Install the package and list what is available:

```bash
pip install causal-ope-benchmark
causal-ope-benchmark --list-families
causal-ope-benchmark --list-estimators
```

Generate one problem, run a tiny smoke suite, and load the output bundle:

```python
from causal_ope_benchmark import (
    make_benchmark_problem,
    run_suite,
    CausalOPEBenchmarkConfig,
    list_families,
    to_fqe_dataset,
)

print([family.name for family in list_families()])

problem = make_benchmark_problem(
    "streamretain",
    sample_size=250,
    gamma=0.95,
    seed=7,
    target_policy="moderate",
)

fqe_data = to_fqe_dataset(problem.dataset, target_policy_expectation_mode="exact_discrete")

config = CausalOPEBenchmarkConfig.for_profile("smoke", output_root="outputs/causal_ope_benchmark")
result = run_suite(config, sample_sizes=(120,), estimators=("direct_method", "ipw", "linear_fqe"))
print(result.readout_path)
```

For copy-paste workflows, see `examples/`:

- `streamlift_finite_infinite.py`
- `streamretain_ope.py`
- `clinic_dtr_survival.py`
- `calibration_smoke.py`
- `difficulty_stress.py`
- `scope_rl_export.py`
- `epicare_optional.py`

## CLI Quickstart

```bash
PYTHONPATH=packages/fqe:packages/causal-ope-benchmark python -m causal_ope_benchmark.run \
  --profile smoke \
  --output-root outputs/causal_ope_benchmark
```

Useful discovery commands:

```bash
python -m causal_ope_benchmark.run --list-families
python -m causal_ope_benchmark.run --list-estimators
python -m causal_ope_benchmark.run --list-difficulties
python -m causal_ope_benchmark.run --profile smoke --dry-run
python -m causal_ope_benchmark.run --profile smoke --families streamlift --streamlift-include-infinite-horizon
```

## Difficulty Profiles

The package also exposes first-class statistical difficulty profiles:
`easy`, `medium`, `hard`, and `realistic`. These are stationary,
time-invariant MDP settings with identifiable primary estimands. Hardness comes
from overlap, sample size, nonlinear transition/reward mechanisms, delayed
effects, action constraints, doses/costs, and observed missingness/censoring,
not from latent confounding or structural no-support.

The profiles are calibrated so that at least one modern method should remain
usable in primary cells, while simpler baselines can fail. For example,
StreamLift expects the specialized `streamlift_stratified_gcomp` dynamics
baseline to beat naïve short-term extrapolation, and the sequential families
expect discounted occupancy-ratio methods to handle policy shifts that simple
direct/FQE rows may miss.

```python
from causal_ope_benchmark import describe_difficulty, make_benchmark_problem

print(describe_difficulty("hard"))
problem = make_benchmark_problem("streamretain", difficulty="medium", target_policy="moderate")
```

Run a systematic stress calibration:

```bash
python -m causal_ope_benchmark.run stress-test \
  --scale ci \
  --difficulty easy medium hard realistic \
  --families streamretain clinic_dtr
```

Runtime scale is separate from difficulty:

- `ci`: one seed and tiny grids for smoke checks.
- `audit`: three seeds and one-factor stress sweeps for local calibration.
- `exhaustive`: ten seeds, sample-size ladders, and interaction cells for
  overnight/cluster runs.

Difficulty stress studies write `difficulty_results.csv`,
`difficulty_summary.csv`, `difficulty_candidates.csv`,
`difficulty_manifest.json`, and `difficulty_readout.md`.

## Neural Calibration Study

Use the calibration command to check whether the reasonable benchmark cells are
estimable by neural FQE and neural discounted occupancy ratios:

```bash
PYTHONPATH=packages/fqe:packages/occupancy-ratio:packages/causal-ope-benchmark \
  python -m causal_ope_benchmark.run calibrate \
  --preset core-lite \
  --output-root outputs/causal_ope_benchmark
```

Calibration writes two tracks. `proxy` rows use package AutoML/proxy criteria
only and represent deployable tuning behavior. `oracle` rows use sealed truth
after candidate fitting to estimate the best attainable neural-network row in
the candidate grid; they are diagnostic-only and never leaderboard eligible.

`StreamLift` appears in calibration as a diagnostic transition/OPE sanity
check. Its primary benchmark remains short-panel long-term causal forecasting,
not FQE or occupancy-ratio OPE.

For StreamLift forecasting runs, `streamlift_stratified_gcomp` fits separate
one-step dynamics models within logged control and treatment rows, then rolls
those public models forward under the public campaign schedule. It is the
recommended sanity baseline for the intended "stationary MDP from a short
panel" use case.

## Outputs

Each run writes a coherent output bundle:

- `results.csv`: row-level estimator scores, diagnostics, status, and
  leaderboard eligibility.
- `summary.csv`: family/estimator aggregates.
- `tuning_results.csv`: reserved tuning telemetry table.
- `manifest.json`: config, schema versions, optional dependency availability,
  platform, package version, and git metadata when available.
- `diagnostics.json`: run diagnostics and leakage checks.
- `output_schema.json`: machine-readable schema/version contract.
- `benchmark_readout.md`: human-readable run summary.

Load a completed bundle with:

```python
from causal_ope_benchmark import load_results

bundle = load_results("outputs/causal_ope_benchmark/smoke")
print(len(bundle.results), bundle.manifest["package_version"])
```

Calibration runs write a separate bundle under
`outputs/causal_ope_benchmark/calibration/<preset>/`:

- `calibration_results.csv`
- `calibration_summary.csv`
- `calibration_candidates.csv`
- `calibration_manifest.json`
- `calibration_readout.md`

## Optional Integrations

Core import only requires NumPy. Optional dependencies are lazy:

```bash
pip install causal-ope-benchmark[gym]
pip install causal-ope-benchmark[scope-rl]
pip install causal-ope-benchmark[fqe]
```

- `neural_fqe` and `boosted_fqe` use the external `fqe` package when available.
- `make_gym_env("streamretain", ...)` and `make_gym_env("clinic_dtr", ...)`
  expose Gymnasium-style native environments.
- `to_scope_rl_logged_dataset(dataset)` exports padded SCOPE-RL-style logged
  datasets.
- `family="epicare"` returns `status=missing_dependency` rows if EpiCare or Gym
  is unavailable.

## Leakage Policy

Estimator-visible data lives in `LongitudinalDataset`. Scorer-only quantities
live in `TruthBundle`. Public metadata, adapters, manifests, and tuning inputs
must not contain private scenario labels, sealed values, hidden parameters,
diagnostic ratio arrays, or target-policy Monte Carlo values.

The package includes tests for public schema alignment, deterministic seeds,
adapter row alignment, lazy optional dependencies, Gym/SCOPE-RL compatibility,
and public-output leakage boundaries.
