# Anonymous NeurIPS Reproducibility Bundles

This directory contains three independent supplementary bundles for anonymous NeurIPS review. Each bundle is intended to stand alone: enter the bundle directory, create a Python environment, install the listed dependencies, and run the smoke or full reproduction commands from that README.

Bundles:

- `fqe_stationary_weighting/`: stationary/discouted-occupancy weighted FQE experiments and final artifacts.
- `soft_fqi_stationary_weighting/`: stationary-weighted soft-FQI simulation code and final artifacts.
- `value_calibration/`: value-space Bellman/FQE calibration experiments and audited final artifacts.

The bundles intentionally exclude manuscript sources, local provenance notes, caches, LaTeX build products, notebook checkpoints, archives, and unrelated paper material. `MANIFEST.json` inside each bundle lists every retained file and its role.
