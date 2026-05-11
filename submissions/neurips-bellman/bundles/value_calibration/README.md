# Anonymous Reproducibility Bundle: Value-Space Bellman/FQE Calibration

This bundle contains the source code, configs, tests, audited final rescue results, final figures, and small benchmark support files needed to reproduce the value-space calibration experiments. It omits manuscript files, local provenance, caches, exploratory runs, and large external Hopper datasets.

## Setup

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r code/FQE_calibration_neurips/requirements.txt
```

## Smoke Test

```bash
PYTHONPATH=code:code/FQE_calibration_neurips MPLCONFIGDIR=.mplconfig   python -m pytest code/FQE_calibration_neurips/tests -q
```

## Full Reproduction

The audited final rescue workflow can be rerun with:

```bash
PYTHONPATH=code:code/FQE_calibration_neurips MPLCONFIGDIR=.mplconfig   python code/FQE_calibration_neurips/scripts/run_rescue_stage.py --stage final
```

A broader paper-suite reproduction is available with:

```bash
PYTHONPATH=code:code/FQE_calibration_neurips MPLCONFIGDIR=.mplconfig   bash code/FQE_calibration_neurips/scripts/run_paper_suite.sh
```

## Included Final Artifacts

- `code/FQE_calibration_neurips/results/rescue_final/`: audited final CSV/JSON/MD/TEX outputs and validation gate artifacts.
- `code/FQE_calibration_neurips/figures/rescue_final/`: audited final figures.
- `paper_artifacts/figures/`: figures referenced by the submitted paper.
- `code/hopper_fqe_benchmark/`: support package and small retained artifact subset needed by tests/appendix code; large external datasets are excluded.
- `code/FQE_neurips/`: minimal weighting/FQE utility subset required by the Hopper support package.

The full final workflow is seeded. Small floating-point differences can occur across hardware, BLAS, and package versions.
