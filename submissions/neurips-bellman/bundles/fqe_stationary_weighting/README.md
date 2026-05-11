# Anonymous Reproducibility Bundle: Stationary-Weighted FQE

This bundle contains the code, selected final result summaries, and final figures/tables needed to reproduce the experiments for the submitted FQE paper. It intentionally omits manuscript files, author-identifying provenance, caches, notebooks, and exploratory raw runs.

## Setup

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Smoke Test

```bash
PYTHONPATH=code python -m FQE_neurips.controlled_discounted_benchmark.run_experiment   --stage smoke   --output-root code/FQE_neurips/results/smoke_reproduction
```

## Full Reproduction

The main controlled benchmark can be rerun with:

```bash
PYTHONPATH=code python -m FQE_neurips.controlled_discounted_benchmark.run_experiment   --stage final   --output-root code/FQE_neurips/results
PYTHONPATH=code python -m FQE_neurips.controlled_discounted_benchmark.run_experiment   --stage gamma_final   --output-root code/FQE_neurips/results
PYTHONPATH=code python -m FQE_neurips.controlled_discounted_benchmark.plot_results   --results-root code/FQE_neurips/results
```

Additional paper-output generation is available with:

```bash
PYTHONPATH=code python -m FQE_neurips.paper_outputs
```

## Included Final Artifacts

- `paper_artifacts/figures/`: figures referenced by the submitted FQE paper.
- `paper_artifacts/tables/`: compact summary tables used by the experiment section.
- `paper_artifacts/sections/`: experiment-section snippets used to document the final runs.
- `code/FQE_neurips/results/`: selected final CSV/MD summaries retained from the canonical experiment tree.

The full runs are stochastic but seeded. Small floating-point differences can occur across BLAS, hardware, or Python package versions.
