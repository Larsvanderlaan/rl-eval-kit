# RL Evaluation Suite

This repository is organized around reusable RL evaluation packages, the
NeurIPS Bellman paper suite, inverse-reinforcement-learning papers, and
archived legacy research code.

## Layout

- `packages/`: installable packages for `fqe`, `occupancy_ratio`, and
  `bellman_trees`.
- `experiments/neurips_bellman/`: active experiment code for stationary-weighted
  FQE, value-space calibration, soft-FQI stationary weighting, and the Hopper
  benchmark support package.
- `papers/neurips_bellman/`: canonical paper sources, shared bibliography, and
  shared LaTeX style files from the curated NeurIPS Bellman bundle.
- `experiments/irl/`: active code for the debiased-IRL journal simulations and
  GenPQR conference reproducibility study.
- `papers/irl/`: canonical IRL journal and conference paper sources.
- `submission_bundles/neurips_bellman/`: anonymous reproducibility bundles and
  manifests used as the paper-facing source of truth.
- `provenance/`: migration snapshots, asset manifests, and source-bundle
  references.
- `legacy/`: old package namespaces, notebooks, and exploratory code retained
  for compatibility and audit history.
- `archive/generated/`: pre-migration generated outputs and caches. This
  directory is intentionally ignored by git.

The old `RLtools` name was intentionally retired in the documentation. The
working directory may still be named `RLtools` locally, but the organized
project is now the RL Evaluation Suite.

## Reproduction Shortcuts

```bash
make test-packages
make test-bellman-trees
make test-calibration
make smoke-fqe
make test-soft-fqi
make smoke-irl-conference
make smoke-irl-journal
make check-assets
```

Root compatibility packages keep existing commands such as
`python -m FQE_neurips.paper_outputs` and imports such as
`import FQE_calibration_neurips` working while the implementation lives under
`experiments/neurips_bellman`. The same compatibility idea applies to
`IRL_neurips` and the transitional `IRL` namespace.
