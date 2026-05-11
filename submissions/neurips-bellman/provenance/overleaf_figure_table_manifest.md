# Figure And Table Manifest

This manifest maps the completed 100-rep paper outputs to the proposed NeurIPS main text and appendix. Paths are relative to `FQE_calibration_neurips/paper_draft/`.

Integration note: `experiments_section.tex` uses `\fqeCalibFigDir` for figures. If the snippet is moved into a different manuscript directory, define it before `\input{...}`, e.g.

```tex
\providecommand{\fqeCalibFigDir}{FQE_calibration_neurips/figures/paper}
```

## Main

| Role | Artifact | Source | Intended claim |
|---|---|---|---|
| Main Figure 1 | Undertraining / finite-iteration evidence | `../figures/paper/undertraining_iteration_sweep.pdf` | Calibration improves value MSE and Bellman/Brier diagnostics when the fitted value predictor has correctable finite-iteration/projection bias. |
| Main Table 1 | Evidence audit summary | Embedded in `experiments_section.tex`; values from `../results/paper/summary.csv` and `../results/paper/calibration_evidence_audit.csv` | Only rows with value-MSE and Bellman/Brier support count as main calibration evidence. |
| Main Figure 2a | Relative MSE under policy shift | `../figures/paper/relative_mse_vs_policy_shift.pdf` | Coverage stress is not a value-MSE win regime. |
| Main Figure 2b | Coverage-stratified diagnostics | `../figures/paper/coverage_stratified_error.pdf` | Calibration can improve local Bellman diagnostics without solving support mismatch. |

## Appendix

| Role | Artifact | Source | Intended claim |
|---|---|---|---|
| Appendix table | Well-specified gate | `../results/paper/tables/well_specified_debug_results.csv`, `../results/paper/tables/well_specified_debug_results.tex` | Validation passed; calibrated rows are neutral or MSE-only, not main positive evidence. |
| Appendix table | Undertraining full summary | `../results/paper/tables/undertraining_sweep_summary.csv`, `../results/paper/tables/undertraining_sweep_summary.tex` | Full finite-iteration results beyond representative main rows. |
| Appendix table/figure | Bellman-incomplete sweep | `../results/paper/tables/bellman_incomplete_sweep_summary.csv`, `../figures/paper/bellman_incomplete_capacity_sweep.pdf` | Supportive but weaker organic evidence; not the flagship result. |
| Appendix table | Model misspecification full summary | `../results/paper/tables/model_misspecification_sweep_summary.csv`, `../results/paper/tables/model_misspecification_sweep_summary.tex` | Affine/systematic misspecification is a main-positive source; nonmonotone rows require caution. |
| Appendix figure/table | Mechanism distortion | `../results/paper/tables/mechanism_distortion_sweep_summary.csv`, `../figures/paper/calibrator_comparison.pdf` | Mechanism-only calibrator sanity checks; not organic baseline evidence. |
| Appendix figure/table | Calibration protocol comparison | `../results/paper/tables/calibration_protocol_summary.csv`, `../figures/paper/calibration_protocol_comparison.pdf` | Cross-calibration is main; split/no-split are diagnostic. |
| Appendix figure/table | Split-fraction sweep | `../results/paper/tables/split_fraction_sweep_summary.csv`, `../figures/paper/split_fraction_comparison.pdf`, `../figures/paper/split_stability_diagnostics.pdf` | Split calibration has no stable main-text wins and is data-use diagnostic only. |
| Appendix figure/table | Baseline-family sweep | `../results/paper/tables/baseline_family_summary.csv`, `../figures/paper/baseline_family_comparison.pdf` | Calibration wraps multiple Bellman/FQE-style baselines, but many rows are calibration-only. |
| Appendix diagnostics | Runtime and failure rate | `../figures/paper/runtime_comparison.pdf`, `../figures/paper/failure_rate_comparison.pdf`, `../figures/paper/failure_rate_by_learner_quality.pdf` | Computational cost and instability diagnostics; final 100-rep run had no nonfinite method-group failures. |
| Appendix diagnostic | Calibration error versus value error | `../figures/paper/calibration_error_vs_value_error.pdf`, `../figures/paper/mse_vs_calibration_error_improvement.pdf` | Shows why MSE-only rows are separated from calibration-supported evidence. |
| Appendix diagnostic | Bias-variance decomposition | `../figures/paper/bias_variance_decomposition.pdf` | Diagnostic decomposition, not central claim. |

## Do Not Use For Main Success Claims

| Artifact / row class | Reason |
|---|---|
| `mse_only` rows in `../results/paper/calibration_evidence_audit.csv` | Scalar value cancellation without Bellman/Brier support. |
| Split-calibrated rows from `split_fraction_sweep` | Split-stability diagnostics found zero stable split-calibration groups. |
| No-split calibrated rows | In-sample calibration; practical diagnostic only. |
| Mechanism-only distorted predictors | Useful calibrator sanity checks, but not organic baseline failures. |
| Coverage-stress rows as positive value-MSE evidence | Calibration improves Bellman diagnostics but worsens scalar value MSE under coverage stress. |
| Saddle-style rows as optimizer-instability evidence | Final paper-mode run had no nonfinite failures; saddle rows are stabilized Bellman-family breadth only. |

## Traceability

Numeric claims in `experiments_section.tex` and `experiments_appendix.tex` are taken from:

- `../results/paper/summary.csv`
- `../results/paper/calibration_evidence_audit.csv`
- `../results/paper/paper_readiness_report.md`
- `../results/paper/paper_draft_readout.md`

The validation gate output is:

- `../results/paper/validation/well_specified_gate.json`
- `../results/paper/validation/well_specified_gate.csv`
