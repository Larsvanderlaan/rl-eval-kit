# RLEvalKit

RLEvalKit is a Python ecosystem for offline reinforcement-learning evaluation:
fitted Q evaluation, occupancy-ratio estimation, inverse-RL recovery, and
realistic causal OPE benchmark simulators.

The repository is organized as a package monorepo with paper and submission
material kept in a separate top-level `submissions/` area.

## Packages

| Package | Import | Purpose |
| --- | --- | --- |
| `packages/occupancy-ratio` | `occupancy_ratio` | Discounted occupancy-ratio estimators, diagnostics, tuning, and benchmark tools. |
| `packages/fqe` | `fqe` | Fitted Q evaluation, neural FQE, stationary weighting, and FQE benchmarks. |
| `packages/genpqr` | `genpqr` | Generalized policy-to-Q-to-reward tools for inverse reinforcement learning. |
| `packages/causal-ope-benchmark` | `causal_ope_benchmark` | Realistic causal inference and industry OPE benchmark simulators. |

Install the development packages from the repository root:

```bash
python -m pip install -e "packages/occupancy-ratio[neural,benchmark,dev]"
python -m pip install -e "packages/fqe[neural,benchmark,dev]"
python -m pip install -e "packages/genpqr[dev]"
python -m pip install -e "packages/causal-ope-benchmark[dev]"
```

## Repository Layout

- `packages/`: releasable Python packages and package-local tests/docs.
- `submissions/`: paper sources, submission bundles, provenance, and
  paper-specific reproduction code.
- `tools/`: repository maintenance scripts.
- `archive/generated/`, `outputs/`, `legacy/`, and `stale/`: local or
  historical material kept out of the release surface.

Generated results, local environments, model/data binaries, caches, and build
artifacts are ignored by default. Submission source files, configs, manifests,
and small canonical assets may be tracked when they are needed for reproducible
paper bundles.

## Development Checks

```bash
make test-packages
python -m ruff check packages/fqe packages/occupancy-ratio packages/genpqr packages/causal-ope-benchmark
```

Submission-specific smoke checks are available as separate Make targets:

```bash
make test-calibration
make smoke-fqe
make test-soft-fqi
make smoke-irl-conference
make smoke-irl-journal
make papers
```

## GitHub

The canonical GitHub repository is:

```text
https://github.com/Larsvanderlaan/rl-eval-kit
```
