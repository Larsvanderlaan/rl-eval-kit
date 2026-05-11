# NeurIPS Calibration Suite for Bellman/FQE Estimators

This folder contains the experimental pipeline for studying calibration as a wrapper around discounted off-policy Bellman/FQE-style estimators. The main comparison is:

```text
baseline learner x calibration protocol x calibrator
```

Calibration is not treated as one estimator. It is applied to neural FQE, random-feature/linear FQE, regularized Bellman residual minimization, stabilized and iterative random-feature saddle/adversarial Bellman residual approximations, and ensemble FQE.

## Value-Space Calibration

The paper-mode calibration target is value-space Bellman calibration. First-stage learners estimate \(Q(s,a)\). Their induced value predictor is

\[
\hat V(s)=\sum_a \pi(a\mid s)\hat Q(s,a).
\]

Calibration then learns a scalar map \(g\) on raw value predictions by importance-weighted fitted value iteration:

\[
g_{k+1}=\arg\min_g \sum_i \rho_i\{g(\hat V(S_i))-[R_i+\gamma g_k(\hat V(S_i'))]\}^2,
\quad
\rho_i=\pi(A_i\mid S_i)/\mu(A_i\mid S_i).
\]

Action-ratio weights are clipped at `20.0` and normalized to mean one on each calibration fit. The calibrated value estimate is \(g(\hat V(s))\), not \(\sum_a\pi(a\mid s)g(\hat Q(s,a))\). Legacy Q-space targets are disabled by default; configs must use `calibration_targets: [value_bellman]` unless `enable_legacy_q_calibration: true` is set explicitly.

## Validation Gate

Paper-mode plots and tables are gated by the well-specified diagnostic. The gate checks that oracle values are finite and independent of training data, included estimators return finite values, value error/MSE behaves sensibly with sample size, calibration does not substantially degrade accurate well-specified baselines, and provenance fields show no test/oracle leakage.

If a method fails the gate, it is marked `main_evidence_eligible=false` and excluded from main-evidence tables/plots. It can still appear in failure-rate and instability diagnostics.

Gate outputs are written to:

```text
results/{debug,paper}/validation/well_specified_gate.json
results/{debug,paper}/validation/well_specified_gate.csv
results/{debug,paper}/validation/audit_notes.md
```

## Experiments

Implemented suites:

- `well_specified_debug`: simple well-specified setting for estimator sanity checks.
- `main_nonlinear`: nonlinear continuous-state, discrete-action MDP under moderate policy shift.
- `undertraining_sweep`: high-discount finite-iteration bias with one-, two-, four-, and well-tuned Bellman iteration variants.
- `bellman_incomplete_sweep`: organic Bellman-incomplete feature/environment pairs where restricted learners can converge to a biased projected fixed point.
- `model_misspecification_sweep`: restricted linear/random-feature/neural learners under model misspecification, separated from mechanism-only prediction distortions.
- `coverage_sweep`: increasing behavior-target policy shift and overlap stress.
- `sample_size_sweep`: finite-sample MSE and relative-MSE trends.
- `misspecification_sweep`: affine, monotone nonlinear, and nonmonotone regimes.
- `mechanism_distortion_sweep`: mechanism-only affine and monotone/saturation prediction distortions on a strong random-feature FQE base learner.
- `calibration_quality_sweep`: balanced well-tuned, realistically poorly calibrated, mechanism-only distorted, and diagnostic-only unstable learner variants.
- `split_fraction_sweep`: 50/50 through 90/10 train/calibration splits with matched comparators.
- `baseline_family_sweep`: compares baseline learner families.
- `calibration_protocol_sweep`: cross, split, and no-split protocol comparison.

The scientific design note is tracked at:

```text
results/simulation_study_design.md
```

## Data-Use Rules

The main uncalibrated comparator is the corresponding baseline learner trained on all available training data. Cross-calibration fits fold Q-models, computes out-of-fold \(\hat V(S)\), \(\hat V(S')\), rewards, and action-ratio weights, fits one pooled value-space FVI calibrator, then refits the final Q learner on all training data. Split-calibration uses only the training split for the Q learner and only the calibration split for the value-space FVI calibrator. No-split calibration is labeled in-sample.

Test transitions, diagnostic test transitions, and oracle Monte Carlo rollouts are independent and are not used for training, tuning, regularization, calibration, early stopping, or model selection. The diagnostic test batch is intentionally larger in paper configs so 50-bin Bellman calibration estimates have enough observations per bin.

## Outputs

Debug and paper outputs are separated:

```text
results/debug/      figures/debug/
results/paper/      figures/paper/
```

Raw rows include run mode, suite name, environment tier, oracle method, train/calibration/test provenance, baseline learner, calibration protocol, calibrator, calibration target, split fraction, data-use flags, value estimate, oracle value, errors, Bellman residual, calibration error, runtime, failure flag, diagnostic warning, and `main_evidence_eligible`.
Prediction diagnostics prioritize value-space quantities: `true_v_mse` / `true_value_function_mse`, the MSE against independently approximated true \(V^\pi(s)\), and `bellman_calibration_error`, a weighted debiased binned \(L^2\) Bellman calibration estimate using the same value-space outcome. The plug-in binned estimate, raw debiased estimate, bin count, diagnostic test size, action-ratio weight effective sample size, and max weight are saved as separate columns. `bellman_outcome_mse` / `brier_score` / `bellman_brier_score` remain in the raw outputs as diagnostics, but they are not part of the final promotion gate. `true_q_mse` remains a backward-compatible diagnostic for the raw first-stage Q learner and is not the main function metric.
Rows also include `learner_variant`, `learner_quality_regime`, `calibration_difficulty`, and `main_figure_role` so well-tuned and deliberately poorly calibrated variants are matched to their own all-data uncalibrated comparators.
Estimator diagnostics include actual neural Bellman iterations, Q prediction ranges, iterative saddle primal/critic norms, objective path summaries, gradient norms, condition proxies, and NaN/exploding flags when available.
Diagnostic bootstrap intervals over independent initial evaluation states are saved as `interval_lower_95`, `interval_upper_95`, `interval_coverage_95`, and `interval_length_95`. These intervals are descriptive diagnostics, not formal OPE inference intervals.

### Evidence Audit

Aggregation assigns every grouped calibrated method a `calibration_evidence_status`:

- `strong`: relative value MSE improves versus the matched all-data uncalibrated baseline, and plug-in Bellman calibration error also improves.
- `mse_only`: value MSE improves but Bellman calibration error does not. These rows are treated as scalar-cancellation warnings, not calibration evidence.
- `calibration_only`: Bellman calibration error improves without value-MSE improvement.
- `neutral`: no clear improvement and no failure.
- `failed`: failed, nonfinite, or ineligible method groups.

Main evidence plots filter to `strong` rows plus neutral uncalibrated controls. MSE-only rows remain in audit, runtime, and failure diagnostics but are not promoted as calibration-success evidence.

Aggregation writes:

- `combined_raw_results.csv`
- `coverage_stratified_error.csv`
- `coverage_stratified_error_raw.csv`
- `calibration_evidence_audit.csv`
- `scalar_cancellation_audit.csv`
- `split_stability_diagnostics.csv`
- `split_stability_diagnostics_raw.csv`
- `summary.csv`
- `eligible_summary.csv`
- `paper_draft_readout.md` after running `inspect_paper_draft.py`
- `tables/*.csv`
- `tables/*.tex`

Tables include main nonlinear, well-specified, coverage, sample-size, misspecification, calibration-quality, split-fraction, baseline-family, calibration-protocol, and all-eligible summaries.
They also include undertraining, Bellman-incomplete, and model-misspecification summaries when those suites are present.

Figures are written as both PNG and PDF with Monte Carlo standard-error bars when repeated seeds exist. Debug figures are explicitly labeled debug-only. The plotting suite includes coverage-stratified error, misspecification/distortion, learner-quality, and calibration-difficulty plots in addition to the main MSE, relative-MSE, protocol, calibrator, baseline, runtime, and failure-rate comparisons.
Additional audit figures include paired MSE/calibration-error improvement, undertraining by actual Bellman iteration count, and Bellman-incomplete capacity sweeps.

## Commands

Install dependencies if needed:

```bash
./.venv/bin/python -m pip install -r FQE_calibration_neurips/requirements.txt
```

Run tests:

```bash
./.venv/bin/python -m pytest FQE_calibration_neurips/tests -q
```

Run the full debug pipeline:

```bash
bash FQE_calibration_neurips/scripts/run_debug.sh
```

Run the paper suite:

```bash
bash FQE_calibration_neurips/scripts/run_paper_suite.sh
```

Run the focused NeurIPS calibration story:

```bash
./.venv/bin/python FQE_calibration_neurips/scripts/run_suite.py \
  --suite_config FQE_calibration_neurips/configs/focused_neurips_suite.yaml \
  --mode debug

./.venv/bin/python FQE_calibration_neurips/scripts/make_plots.py \
  --results_dir FQE_calibration_neurips/results/focused_debug \
  --figures_dir FQE_calibration_neurips/figures/focused_debug

./.venv/bin/python FQE_calibration_neurips/scripts/inspect_paper_draft.py \
  --results_dir FQE_calibration_neurips/results/focused_debug \
  --figures_dir FQE_calibration_neurips/figures/focused_debug
```

For a final focused run, use `--mode paper`; the focused config defaults to 50 replications and writes to
`FQE_calibration_neurips/results/focused_paper`.

Run the staged rescue-and-audit workflow:

```bash
./.venv/bin/python FQE_calibration_neurips/scripts/run_rescue_stage.py \
  --stage debug

./.venv/bin/python FQE_calibration_neurips/scripts/run_rescue_stage.py \
  --stage pilot

./.venv/bin/python FQE_calibration_neurips/scripts/run_rescue_stage.py \
  --stage confirm

./.venv/bin/python FQE_calibration_neurips/scripts/run_rescue_stage.py \
  --stage final
```

The rescue stages write to separate directories:
`results/rescue_debug`, `results/rescue_pilot`, `results/rescue_confirm`, and `results/rescue_final`.
Only `final` rows are eligible for main-text promotion; debug/pilot/confirm rows are audit-labeled as tuning-only.
Each stage writes `rescue_promotion_audit.csv`, `do_not_claim_manifest.csv`, `rescue_readout.md`, and
`rescue_audit_summary.json`.

### Final Rescue Evidence Snapshot

The current audited final rescue package is:

```text
FQE_calibration_neurips/results/rescue_final
FQE_calibration_neurips/figures/rescue_final
```

The final audit source of truth is `results/rescue_final/rescue_promotion_audit.csv`.
As of the latest run, the validation gate passes, validation controls are labeled
`validation_control`, mechanism rows are labeled `mechanism_only`, and main-text
claims should use only rows labeled `promote_main`.
The final run uses a seed block starting at `3000001` with
`replication_seed_stride: 1000000`, disjoint from the pilot and confirm seed
blocks.

The final promoted story is:

- affine/misspecified linear FQE: linear calibration is sufficient and improves
  true-value MSE plus Bellman calibration error;
- finite-iteration random-feature FQE: calibration improves true-value MSE and
  Bellman calibration error with positive raw rank correlation;
- temporal reward shift: an old-regime FQE value model trained on 2000 old
  behavior transitions can be recalibrated with 100 recent current-regime
  transitions, improving current-regime true-value MSE and Bellman calibration
  error; the same-small-current retrain comparator uses the same learner with a
  prespecified prediction clip and is reported but not promoted;
- mechanism affine-vs-monotone panel: linear wins/ties for affine distortion,
  while isotonic is clearly better for monotone saturation.

The optional quality/saddle/temporal-proxy tail is not promoted in this final
package. It should remain limitation or appendix work unless rerun and audited
separately. Brier/Bellman-outcome metrics are still saved for diagnostics, but
the final promotion gate and main figure focus on true-value MSE and Bellman
calibration error.

Run a 10-replication quick paper draft and inspect the candidate story:

```bash
./.venv/bin/python FQE_calibration_neurips/scripts/run_suite.py \
  --suite_config FQE_calibration_neurips/configs/paper_suite.yaml \
  --mode paper \
  --replications 10

./.venv/bin/python FQE_calibration_neurips/scripts/make_plots.py \
  --results_dir FQE_calibration_neurips/results/paper \
  --figures_dir FQE_calibration_neurips/figures/paper

./.venv/bin/python FQE_calibration_neurips/scripts/inspect_paper_draft.py \
  --results_dir FQE_calibration_neurips/results/paper \
  --figures_dir FQE_calibration_neurips/figures/paper
```

Run a suite directly:

```bash
./.venv/bin/python FQE_calibration_neurips/scripts/run_suite.py \
  --suite_config FQE_calibration_neurips/configs/paper_suite.yaml \
  --mode debug
```

Aggregate and plot explicitly:

```bash
./.venv/bin/python FQE_calibration_neurips/scripts/aggregate_results.py \
  --results_dir FQE_calibration_neurips/results/debug

./.venv/bin/python FQE_calibration_neurips/scripts/make_plots.py \
  --results_dir FQE_calibration_neurips/results/debug \
  --figures_dir FQE_calibration_neurips/figures/debug
```

## Compute Defaults

- Debug: 1 replication, small samples, CPU-runnable quickly; validates the pipeline only.
- Quick paper draft: use `--replications 10` or the defaults in `paper_suite.yaml`.
- Final production: rerun with `--replications 30` to `--replications 50` if compute is available.

All configs expose sample sizes, replications, learners, protocols, calibrators, targets, and split fractions.

## Known Limitations

The saddle-point/adversarial Bellman residual family includes a stabilized closed-form approximation and an iterative random-feature min-max approximation, not a full neural min-max solver. Unstable iterative saddle variants are marked diagnostic-only unless they pass the well-specified gate. Mechanism-only prediction distortions are included to identify which calibrators can fix affine or monotone errors; they should not be described as organic baseline learners. Core estimator changes are intentionally kept out of the results-pipeline layer unless a small fix is needed for the validation pipeline.
Some high-capacity or weakly regularized variants are intentionally allowed to fail and are retained as instability diagnostics. A paper run should inspect `calibration_evidence_audit.csv`, `scalar_cancellation_audit.csv`, and failure-rate plots before choosing main-text figures.
